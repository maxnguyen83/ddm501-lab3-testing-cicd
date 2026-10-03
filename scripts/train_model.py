"""
Script to train and save the movie rating prediction model.

Usage:
    python scripts/train_model.py
"""

import pickle
from pathlib import Path

from surprise import Dataset, SVD
from surprise.model_selection import cross_validate


def main():
    """Main function to train and save the model."""
    
    print("=" * 60)
    print("Movie Rating Prediction Model Training")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "svd_model.pkl"
    
    # Load data
    print("\n[1/4] Loading MovieLens 100K dataset...")
    data = Dataset.load_builtin('ml-100k')
    print("      Dataset loaded successfully!")
    
    # Define model
    print("\n[2/4] Performing cross-validation...")
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    
    # Cross-validation
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print(f"\n      Mean RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"      Mean MAE:  {cv_results['test_mae'].mean():.4f}")
    
    # Train on full dataset
    print("\n[3/4] Training on full dataset...")
    trainset = data.build_full_trainset()
    model.fit(trainset)
    print("      Training completed!")
    
    # Save model
    print(f"\n[4/4] Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("      Model saved successfully!")
    
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
