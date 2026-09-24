"""Compare matched hard/soft five-minute warnings with paired patient CIs.

Both decision CSVs need the same val/test case-time rows, evaluability flags,
and patient IDs. A shared events CSV supplies onset annotations; optionally
pass the second model's events CSV to verify it is identical. patient_id is
optional and defaults to case_id, assuming one case per patient. If inputs
contain multiple runs or folds, their events require the same run/fold column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.monitoring import DEFAULT_BUDGETS
from hyponet.monitoring_compare import compare_monitoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-decisions", type=Path, required=True)
    parser.add_argument("--soft-decisions", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True,
                        help="Common val/test onset annotations, for example the hard model's events.csv.")
    parser.add_argument("--soft-events", type=Path,
                        help="Optional second events.csv; exact identity with --events is checked.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS,
                        metavar="FA_PER_HOUR")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hard = pd.read_csv(args.hard_decisions, dtype=str)
    soft = pd.read_csv(args.soft_decisions, dtype=str)
    events = pd.read_csv(args.events, dtype=str)
    soft_events = pd.read_csv(args.soft_events, dtype=str) if args.soft_events else None
    result = compare_monitoring(
        hard, soft, events, soft_events=soft_events,
        budgets=args.budgets, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "matched_monitoring.csv"
    result.to_csv(csv_path, index=False)
    method = {
        "threshold_fit": "separate model- and run-specific validation scores only",
        "test_thresholds": "fixed from validation; no test refitting",
        "matched_decisions": "same run/split/case/time/patient/evaluability flags",
        "matched_events": "shared event annotations; optional second file checked exactly",
        "bootstrap": "paired test patient-cluster percentile interval conditional on models and thresholds",
        "bootstrap_unit": "patient_id when supplied, otherwise case_id (one case per patient assumed)",
        "bootstrap_resamples": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "validation_budgets_per_hour": args.budgets,
        "differences": "soft minus hard",
        "confidence_interval": "2.5th and 97.5th percentiles of finite paired replicate differences",
        "lead_median": "median among timely detected events in each resample; undefined resamples excluded and counted",
    }
    (args.output_dir / "matched_monitoring_method.json").write_text(
        json.dumps(method, indent=2) + "\n", encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
