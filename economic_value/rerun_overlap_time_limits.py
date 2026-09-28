"""Rerun only time-limited controlled-overlap cases with a longer limit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

from .run_overlap_control import DEFAULT_CALIBRATION, DEFAULT_OUTPUT, _evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    checkpoints = args.output / "checkpoints"
    tasks = []
    for path in sorted(checkpoints.glob("*.json")):
        result = json.loads(path.read_text())
        pair = result["pair"]
        if pair["catalog_exact_status"] == "optimal" and pair["informed_exact_status"] == "optimal":
            continue
        tasks.append({
            "seed": int(pair["seed"]), "overlap": pair["overlap"],
            "support_pattern": pair["support_pattern"], "case_id": pair["case_id"],
            "exact_time_limit": args.time_limit, "checkpoint_path": str(path),
        })
    print(f"rerunning {len(tasks)} time-limited cases at {args.time_limit:g} seconds", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate,
                {key: value for key, value in task.items() if key != "checkpoint_path"},
                calibration,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            target = Path(task["checkpoint_path"])
            result = future.result()
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(target)
            print(f"completed {completed}/{len(tasks)} longer-limit cases", flush=True)


if __name__ == "__main__":
    main()
