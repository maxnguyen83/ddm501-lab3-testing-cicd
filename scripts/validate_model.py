"""
Validate a trained model before it is allowed to ship.

Reads the cross-validation metrics written by scripts/train_model.py and checks
them against fixed thresholds, then smoke-tests the saved artefact through the
same wrapper the API uses. Exits with code 1 if any check fails, so a CI step
running it goes red.

Usage (from the project root):
    python -m scripts.validate_model
    python -m scripts.validate_model --max-rmse 0.95 --max-mae 0.75
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import MAX_RATING, MIN_RATING
from app.model import MovieRatingModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thresholds (reasoning in docs/TESTING_STRATEGY.md):
# reference SVD reaches RMSE 0.935 / MAE 0.737; under-trained or mis-tuned
# variants measured 0.958-1.010 RMSE, so 0.95 / 0.75 separates them.
DEFAULT_MAX_RMSE = 0.95
DEFAULT_MAX_MAE = 0.75
SMOKE_PAIRS = [("196", "242"), ("186", "302"), ("22", "377"), ("new_user", "242")]


@dataclass
class Check:
    """Result of one validation check."""

    name: str
    passed: bool
    detail: str


def run_checks(
    metrics: Dict[str, Any], model_path: str, max_rmse: float, max_mae: float
) -> List[Check]:
    """Run every metric and artefact check and return the results."""
    rmse = float(metrics["cv_rmse_mean"])
    mae = float(metrics["cv_mae_mean"])
    baseline = float(metrics["baseline_only_rmse_mean"])
    checks = [
        Check("CV RMSE", rmse <= max_rmse, f"{rmse:.4f} (max {max_rmse})"),
        Check("CV MAE", mae <= max_mae, f"{mae:.4f} (max {max_mae})"),
        Check(
            "Beats BaselineOnly",
            rmse < baseline,
            f"{rmse:.4f} vs {baseline:.4f} (biases only)",
        ),
    ]

    try:
        model = MovieRatingModel(model_path=model_path)
        predictions = model.predict_batch(SMOKE_PAIRS)
        in_range = all(MIN_RATING <= p <= MAX_RATING for p in predictions)
        checks.append(Check("Artefact loads and predicts", in_range, f"{predictions}"))
    except Exception as e:  # report any load/predict failure as a failed check
        checks.append(Check("Artefact loads and predicts", False, repr(e)))
    return checks


def write_summary(checks: List[Check], metrics: Dict[str, Any]) -> str:
    """Render the results as a markdown table (also used for the job summary)."""
    lines = [
        "## Model validation",
        "",
        f"Model: {metrics.get('algorithm')} {metrics.get('params')}, "
        f"{metrics.get('cv_folds')}-fold CV, seed {metrics.get('random_state')}",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.detail} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, run checks, print a report and return the exit code."""
    parser = argparse.ArgumentParser(description="Validate a trained model before it ships.")
    parser.add_argument("--metrics", default=str(PROJECT_ROOT / "models" / "metrics.json"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "svd_model.pkl"))
    parser.add_argument("--max-rmse", type=float, default=DEFAULT_MAX_RMSE)
    parser.add_argument("--max-mae", type=float, default=DEFAULT_MAX_MAE)
    args = parser.parse_args(argv)

    metrics_file = Path(args.metrics)
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}. Run scripts/train_model.py first.")
        return 1
    metrics = json.loads(metrics_file.read_text())

    checks = run_checks(metrics, args.model, args.max_rmse, args.max_mae)
    report = write_summary(checks, metrics)
    print(report)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report)

    failed = [c.name for c in checks if not c.passed]
    if failed:
        print(f"Model validation FAILED: {', '.join(failed)}")
        return 1
    print("Model validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
