"""Generate and replay a synthetic Optimal Edge Discovery planning response."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import uuid

import numpy as np

from .edfs_adapter import MANAGED_CONNECTIVITY, response_to_support
from .environment import build_environment


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "results" / "edfs_replay_v1"
NAMESPACE = uuid.UUID("f157c64f-8116-47a8-bad9-f7b9f2f56851")


def _zone_id(site_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, site_id))


def _target_area(area: dict) -> dict:
    return {
        "area": {
            "areaType": "CIRCLE",
            "center": {
                "latitude": 38.0 + float(area["y"]) / 1000.0,
                "longitude": 23.0 + float(area["x"]) / 1000.0,
            },
            "radius": 5000,
        }
    }


def _item(raw_status: str, zone_id: str, delay: float) -> dict:
    status = {
        "VIABLE": "VIABLE",
        "PARTIAL": "CONDITIONALLY_VIABLE",
        "UNVIABLE": "NOT_VIABLE",
        "UNKNOWN": "UNKNOWN",
    }[raw_status]
    item = {"edgeCloudZoneId": zone_id, "viability": status}
    if status == "CONDITIONALLY_VIABLE":
        item["conditionInfo"] = {
            "conditionType": "SERVICE_ACTIVATION_REQUIRED",
            "expectedActivationDelay": "PT1H",
            "description": "The contracted local-breakout service must be active.",
        }
    if np.isfinite(delay):
        item["estimatedMetrics"] = {
            "latencyEstimate": {
                "p50LatencyMs": round(float(delay) * 0.8, 2),
                "p95LatencyMs": round(float(delay), 2),
            }
        }
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=61_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    environment = build_environment(
        args.seed, topology="asymmetric", profile="moderate", demand_pattern="hotspots"
    )
    mno = 0
    profile_id = str(uuid.uuid5(NAMESPACE, "moderate-profile"))
    zone_ids = [_zone_id(site["id"]) for site in environment.sites]
    request = {
        "applicationProfileId": profile_id,
        "targetAreas": [_target_area(area) for area in environment.areas],
        "maxResults": 100,
        "includeNonViableCandidates": True,
    }
    area_results = []
    expected = []
    for area_index, area in enumerate(environment.areas):
        group = area_index * environment.metadata["n_mnos"] + mno
        expected.append(environment.resolved_support[group])
        area_results.append({
            "targetArea": _target_area(area),
            "edgeCloudZoneViabilities": [
                _item(str(environment.raw_status[group, site]), zone_ids[site],
                      float(environment.delay_ms[group, site]))
                for site in range(len(environment.sites))
            ],
        })
    response = {
        "applicationProfileId": profile_id,
        "validUntil": "2026-10-01T00:00:00Z",
        "edgeCloudZones": [
            {
                "edgeCloudZoneId": zone_id,
                "edgeCloudZoneName": site["id"],
                "edgeCloudProvider": f"Synthetic ECSP {site['ecsp']}",
                "edgeCloudRegion": "synthetic-study-region",
                "edgeCloudZoneStatus": "active",
            }
            for site, zone_id in zip(environment.sites, zone_ids)
        ],
        "areaResults": area_results,
    }
    condition_resolutions = {
        zone_id: bool(environment.commercial_package[f"MNO_{mno}"])
        for zone_id in zone_ids
    }
    connectivity = {zone_id: MANAGED_CONNECTIVITY for zone_id in zone_ids}
    replayed = response_to_support(
        response,
        [area["id"] for area in environment.areas],
        zone_ids,
        condition_resolutions,
        connectivity,
    )
    expected_array = np.asarray(expected)
    if not np.array_equal(replayed, expected_array):
        raise AssertionError("Response replay does not reproduce the simulator support map")

    # Explicit semantic negative control: delay alone must not turn a public
    # Internet route into managed-connectivity support.
    fast_public = {"edgeCloudZoneId": zone_ids[0], "viability": "VIABLE",
                   "estimatedMetrics": {"latencyEstimate": {"p95LatencyMs": 5}}}
    negative_response = {
        "areaResults": [{"targetArea": _target_area(environment.areas[0]),
                         "edgeCloudZoneViabilities": [fast_public]}]
    }
    negative = response_to_support(
        negative_response, [environment.areas[0]["id"]], zone_ids,
        connectivity_classes={zone_ids[0]: "PUBLIC_INTERNET"},
    )
    if negative[0, 0] != 0:
        raise AssertionError("Public-Internet negative control was incorrectly accepted")

    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output / "response.json").write_text(json.dumps(response, indent=2) + "\n")
    with (args.output / "resolved_support.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["area_id", "site_id", "resolved_support"])
        for area_index, area in enumerate(environment.areas):
            for site_index, site in enumerate(environment.sites):
                writer.writerow([area["id"], site["id"], int(replayed[area_index, site_index])])
    summary = {
        "status": "PASS",
        "areas": len(environment.areas),
        "sites": len(environment.sites),
        "tuples_replayed": int(replayed.size),
        "support_entries": int(replayed.sum()),
        "reproduces_truthful_resolved_map": True,
        "fast_public_internet_negative_control": "PASS",
        "note": "The planner consumes the resolved binary map, not API serialization.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
