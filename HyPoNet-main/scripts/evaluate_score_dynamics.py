"""Evaluate matched hard/soft score stability and pre-onset response.

Each model CSV needs split, case_id, time_s, score, and a stable column for
test rows. The stable marks must be supplied from an external clinical rule.
Events need split, case_id, onset_s. patient_id is optional in both inputs;
when absent, case_id is used as the bootstrap cluster.
Optional ongoing, input_complete, and future_complete columns filter invalid
decision rows before score checks. Optional run/fold columns fit validation
transformations separately for each run. Events may include eligible marks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.score_dynamics import evaluate_score_dynamics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-scores", required=True, type=Path)
    parser.add_argument("--soft-scores", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--step-seconds", type=float, default=20.0)
    parser.add_argument("--rolling-predictions", type=int, default=10)
    parser.add_argument("--pre-onset-seconds", type=float, default=300.0)
    parser.add_argument("--min-lead-seconds", type=float, default=0.0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary, cases, events = evaluate_score_dynamics(
        pd.read_csv(args.hard_scores, dtype=str),
        pd.read_csv(args.soft_scores, dtype=str),
        pd.read_csv(args.events, dtype=str),
        step_seconds=args.step_seconds,
        window_predictions=args.rolling_predictions,
        pre_onset_seconds=args.pre_onset_seconds,
        min_lead_seconds=args.min_lead_seconds,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "score_dynamics_summary.csv", index=False)
    cases.to_csv(args.output_dir / "score_dynamics_cases.csv", index=False)
    events.to_csv(args.output_dir / "score_dynamics_events.csv", index=False)
    metadata = {
        "step_seconds": args.step_seconds,
        "rolling_predictions": args.rolling_predictions,
        "rolling_sd_ddof": 1,
        "stable_periods": "supplied by test score CSV; no detector inferred",
        "decision_filter": "when supplied: not ongoing, input_complete, future_complete",
        "events": "optional eligible flag; fewer than two pre-onset decisions are reported as excluded",
        "pre_onset_seconds": args.pre_onset_seconds,
        "min_lead_seconds": args.min_lead_seconds,
        "rise": "last minus first available pre-onset score in the selected window",
        "rise_endpoints": "implementation choice; manuscript does not specify exact endpoints",
        "percentile": "validation empirical CDF, right-inclusive",
        "standardized": "validation mean and population standard deviation",
        "validation_fit": "separate for each run/fold when supplied",
        "constant_score_spearman": 0.0,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_unit": "patient_id if supplied, otherwise case_id",
        "seed": args.seed,
        "thresholds_used": False,
    }
    (args.output_dir / "score_dynamics_method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
