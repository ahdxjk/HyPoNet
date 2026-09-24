"""Evaluate chronological POH warnings from prepared decision and event CSVs.

The sampled training HDF5 files cannot be used directly: they do not retain
decision timestamps, event-onset times, or 20-second overlapping predictions.
Generate those rows from chronological recordings and trained model scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.monitoring import DEFAULT_BUDGETS, evaluate_monitoring


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit warning thresholds on validation and evaluate them on test recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "decisions.csv: split,case_id,time_s,score,ongoing,input_complete,future_complete\n"
            "events.csv: split,case_id,onset_s[,eligible]\n"
            "Use one file pair per model and evaluation run. time_s is the END of a\n"
            "complete preceding 60-s input, on a 20-s grid from recording start.\n"
            "Include invalid grid rows with exclusion flags; blank scores are allowed\n"
            "only on invalid rows. Events are distinct adjudicated episode onsets.\n"
            "The current HDF5 folds lack chronology and cannot produce these CSVs."
        ),
    )
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional CSV path for operating points.")
    parser.add_argument("--model", help="Optional model name added to output rows.")
    parser.add_argument("--run-id", help="Optional evaluation-run ID added to output rows.")
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=DEFAULT_BUDGETS,
        metavar="FA_PER_HOUR",
        help="Validation false-alarm budgets per hour (default: 0.25 0.5 1 2).",
    )
    args = parser.parse_args()

    decisions = pd.read_csv(args.decisions, dtype={"case_id": str})
    events = pd.read_csv(args.events, dtype={"case_id": str})
    results = evaluate_monitoring(decisions, events, budgets=args.budgets)
    if args.model:
        results.insert(0, "model", args.model)
    if args.run_id:
        results.insert(0, "run_id", args.run_id)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
