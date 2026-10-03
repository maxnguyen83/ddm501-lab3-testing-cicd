"""
Script to train and save the movie rating prediction model.

Usage:
    python scripts/train_model.py

Outputs:
    models/svd_model.pkl   trained model loaded by the API
    models/metrics.json    cross-validation metrics read by scripts/validate_model.py
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import surprise
from surprise import SVD, BaselineOnly, Dataset
from surprise.model_selection import KFold, cross_validate

# Fixed seed so CI runs train the same model and produce the same metrics
RANDOM_STATE = 42
CV_FOLDS = 5
SVD_PARAMS: Dict[str, Any] = {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}


def global_mean_baseline_rmse(data: Dataset, folds: KFold) -> List[float]:
    """RMSE of always predicting the training-fold mean (the naive baseline)."""
    scores = []
    for trainset, testset in folds.split(data):
        mean = trainset.global_mean
        errors = np.array([true_r - mean for (_, _, true_r) in testset])
        scores.append(float(np.sqrt(np.mean(errors**2))))
    return scores


def main() -> None:
    """Main function to train and save the model."""

    print("=" * 60)
    print("Movie Rating Prediction Model Training")
    print("=" * 60)

    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "svd_model.pkl"
    metrics_path = models_dir / "metrics.json"

    # Load data (prompt=False: download without asking, so CI runners do not hang on input())
    print("\n[1/4] Loading MovieLens 100K dataset...")
    data = Dataset.load_builtin("ml-100k", prompt=False)
    print("      Dataset loaded successfully!")

    # Define model
    print("\n[2/4] Performing cross-validation...")
    model = SVD(random_state=RANDOM_STATE, **SVD_PARAMS)
    folds = KFold(n_splits=CV_FOLDS, random_state=RANDOM_STATE)

    # Cross-validation (same seeded folds for the model and both baselines)
    cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=folds, verbose=True)
    bias_results = cross_validate(BaselineOnly(verbose=False), data, measures=["RMSE"], cv=folds)
    global_mean_rmse = global_mean_baseline_rmse(data, folds)
    print(f"\n      Mean RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"      Mean MAE:  {cv_results['test_mae'].mean():.4f}")
    print(f"      BaselineOnly RMSE: {bias_results['test_rmse'].mean():.4f}")
    print(f"      Global-mean RMSE:  {np.mean(global_mean_rmse):.4f}")

    # Train on full dataset
    print("\n[3/4] Training on full dataset...")
    trainset = data.build_full_trainset()
    model.fit(trainset)
    print("      Training completed!")

    # Save model
    print(f"\n[4/4] Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("      Model saved successfully!")

    metrics = {
        "algorithm": "SVD",
        "params": SVD_PARAMS,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "cv_rmse_mean": float(cv_results["test_rmse"].mean()),
        "cv_rmse_std": float(cv_results["test_rmse"].std()),
        "cv_mae_mean": float(cv_results["test_mae"].mean()),
        "cv_mae_std": float(cv_results["test_mae"].std()),
        "baseline_only_rmse_mean": float(bias_results["test_rmse"].mean()),
        "global_mean_rmse_mean": float(np.mean(global_mean_rmse)),
        "n_ratings": trainset.n_ratings,
        "n_users": trainset.n_users,
        "n_items": trainset.n_items,
        "surprise_version": surprise.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"      Metrics written to {metrics_path}")

    # Test prediction
    print("\n" + "=" * 60)
    print("Testing the model...")
    prediction = model.predict("196", "242")
    print(f"Sample prediction for user 196, movie 242: {prediction.est:.2f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
