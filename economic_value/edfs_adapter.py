"""Adapter from an EDFS-style planning response to binary planner support.

The economic study depends on the resolved support map, not on an API name or
serialization. This adapter documents one concrete integration path using the
current Optimal Edge Discovery pre-deployment response shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


MANAGED_CONNECTIVITY = "OPERATOR_MANAGED"


def resolve_viability(
    viability: dict[str, Any],
    condition_resolutions: Mapping[str, bool],
    connectivity_class: str = MANAGED_CONNECTIVITY,
) -> int:
    """Resolve one API viability item to F in {0, 1}.

    A low-delay public-Internet path is outside the managed-connectivity use
    case and therefore cannot create support, irrespective of the numeric
    latency reported by another system.
    """
    if connectivity_class != MANAGED_CONNECTIVITY:
        return 0
    status = viability.get("viability")
    if status == "VIABLE":
        return 1
    if status in {"NOT_VIABLE", "UNKNOWN"}:
        return 0
    if status != "CONDITIONALLY_VIABLE":
        raise ValueError(f"Unknown viability status: {status!r}")
    condition = viability.get("conditionInfo")
    if not isinstance(condition, dict) or "conditionType" not in condition:
        raise ValueError("CONDITIONALLY_VIABLE requires conditionInfo.conditionType")
    zone_id = str(viability["edgeCloudZoneId"])
    return int(bool(condition_resolutions.get(zone_id, False)))


def response_to_support(
    response: dict[str, Any],
    area_keys: list[str],
    site_ids: list[str],
    condition_resolutions: Mapping[str, bool] | None = None,
    connectivity_classes: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Convert one operator/profile response into an area-by-site support map.

    `area_keys` correspond positionally to `areaResults`. Missing candidates
    are unsupported because the replay requests non-viable candidates as well.
    """
    condition_resolutions = condition_resolutions or {}
    connectivity_classes = connectivity_classes or {}
    results = response.get("areaResults")
    if not isinstance(results, list) or len(results) != len(area_keys):
        raise ValueError("Response areaResults do not match requested target areas")
    lookup = {site_id: index for index, site_id in enumerate(site_ids)}
    support = np.zeros((len(area_keys), len(site_ids)), dtype=float)
    for area_index, area_result in enumerate(results):
        items = area_result.get("edgeCloudZoneViabilities", [])
        seen: set[str] = set()
        for item in items:
            zone_id = str(item.get("edgeCloudZoneId"))
            if zone_id not in lookup:
                raise ValueError(f"Unknown edge-cloud zone in response: {zone_id}")
            if zone_id in seen:
                raise ValueError(f"Duplicate edge-cloud zone result: {zone_id}")
            seen.add(zone_id)
            support[area_index, lookup[zone_id]] = resolve_viability(
                item,
                condition_resolutions,
                connectivity_classes.get(zone_id, MANAGED_CONNECTIVITY),
            )
    return support
