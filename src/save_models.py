import argparse
import json
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np

from .config import ExperimentConfig
from .dataset import prepare_dataset, get_feature_columns
from .models import EarlyDetectionModel, HybridDualModel

MODELS_DIR = Path('models')

def save_production_models(n_classes: int = 3):
    MODELS_DIR.mkdir(exist_ok=True)

    config = ExperimentConfig()
    df, feature_cols = prepare_dataset(config.data, config.train, n_classes=n_classes)

    labeled_mask = df['state'].notna()
    labeled_df = df[labeled_mask]

    X = labeled_df[feature_cols].values
    y_state = labeled_df['state'].values

    print(f"Training on {len(X)} labeled samples with {len(feature_cols)} features")
    print(f"Classes: {n_classes} (normal=0, hyper=1, severe=2)" if n_classes == 3 else "")

    print("\nTraining EarlyDetectionModel...")
    early_model = EarlyDetectionModel()
    early_model.fit(X, y_state, feature_names=feature_cols)

    early_path = MODELS_DIR / 'early_detection.joblib'
    joblib.dump(early_model, early_path)
    print(f"  Saved to {early_path}")

    print("\nTraining HybridDualModel...")
    hybrid_model = HybridDualModel()
    hybrid_model.fit(X, y_state, feature_names=feature_cols)

    hybrid_path = MODELS_DIR / 'hybrid_dual.joblib'
    joblib.dump(hybrid_model, hybrid_path)
    print(f"  Saved to {hybrid_path}")

    date_range = {
        'min': df['window_start'].min().isoformat(),
        'max': df['window_end'].max().isoformat()
    }

    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_classes': n_classes,
        'n_samples': len(X),
        'feature_columns': feature_cols,
        'date_range': date_range,
        'early_detection_threshold': early_model.params['threshold'],
        'hybrid_thresholds': {
            'hyper': hybrid_model.params['hyper_threshold'],
            'severe': hybrid_model.params['severe_threshold']
        }
    }

    metadata_path = MODELS_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")

    print("\nProduction models ready for inference")

def main():
    parser = argparse.ArgumentParser(description='Train and save production models')
    parser.add_argument('--n-classes', type=int, default=3, choices=[3, 4])
    args = parser.parse_args()

    save_production_models(n_classes=args.n_classes)

if __name__ == '__main__':
    main()
