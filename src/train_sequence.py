import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import mlflow

from .config import ExperimentConfig
from .dataset import prepare_dataset, temporal_train_test_split, create_sequences
from .sequence_models import SequenceModelWrapper
from .train import compute_metrics

def run_sequence_experiment(
    model_type: str,
    config: ExperimentConfig,
    model_params: Dict[str, Any] = None,
    test_start_date: str = '2025-08-01',
    seq_length: int = 6,
    experiment_name: str = None
):
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(experiment_name or config.name)

    print(f"Loading data...")
    df, feature_cols = prepare_dataset(config.data, config.train)

    labeled_count = df['state'].notna().sum()
    print(f"Total windows: {len(df)}, Labeled: {labeled_count}, Features: {len(feature_cols)}")

    if labeled_count == 0:
        print("ERROR: No labeled data found. Please create data/labels.csv")
        return None

    test_date = pd.Timestamp(test_start_date, tz='UTC')

    data = temporal_train_test_split(
        df, feature_cols, test_date,
        for_sequences=True,
        seq_length=seq_length
    )

    print(f"Train sequences: {len(data['X_train'])}, Test sequences: {len(data['X_test'])}")

    if len(data['X_train']) < 10:
        print("ERROR: Not enough training sequences. Need more labeled data or shorter sequence length.")
        return None

    default_params = {
        'model_type': model_type,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': min(32, len(data['X_train'])),
        'epochs': 100,
        'patience': 15
    }
    if model_params:
        default_params.update(model_params)

    with mlflow.start_run(run_name=f"{model_type}_seq{seq_length}"):
        mlflow.log_params({
            'model': model_type,
            'sequence_length': seq_length,
            'test_start_date': test_start_date,
            'n_features': len(feature_cols),
            'n_train_sequences': len(data['X_train']),
            'n_test_sequences': len(data['X_test']),
            **default_params
        })

        model = SequenceModelWrapper(**default_params)
        model.fit(
            data['X_train'],
            data['y_train_state'],
            data['y_train_trend']
        )

        state_pred, trend_pred = model.predict(data['X_test'])

        state_metrics = compute_metrics(data['y_test_state'], state_pred, prefix='state_')
        mlflow.log_metrics(state_metrics)

        print("\nState Classification Metrics:")
        for k, v in state_metrics.items():
            print(f"  {k}: {v:.4f}")

        trend_metrics = compute_metrics(data['y_test_trend'], trend_pred, prefix='trend_')
        if trend_metrics:
            mlflow.log_metrics(trend_metrics)
            print("\nTrend Classification Metrics:")
            for k, v in trend_metrics.items():
                print(f"  {k}: {v:.4f}")

        mlflow.log_metric('final_train_loss', model.training_history[-1]['loss'])
        mlflow.log_metric('epochs_trained', len(model.training_history))

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print(f"Device used: {model.device}")

        return {
            'run_id': run_id,
            'state_metrics': state_metrics,
            'trend_metrics': trend_metrics,
            'training_history': model.training_history
        }

def main():
    parser = argparse.ArgumentParser(description='Train sequence models for thyroid prediction')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--seq-length', type=int, default=6)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test-start-date', type=str, default='2025-08-01')
    parser.add_argument('--experiment-name', type=str, default=None)
    args = parser.parse_args()

    config = ExperimentConfig()

    model_params = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'epochs': args.epochs
    }

    result = run_sequence_experiment(
        model_type=args.model,
        config=config,
        model_params=model_params,
        test_start_date=args.test_start_date,
        seq_length=args.seq_length,
        experiment_name=args.experiment_name
    )

    print("\n" + "="*60)
    print("Training complete. View results with: mlflow ui")
    print("="*60)

if __name__ == '__main__':
    main()
