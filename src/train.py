import argparse
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)

from .config import ExperimentConfig, DataConfig, TrainConfig
from .dataset import prepare_dataset, temporal_train_test_split
from .models import get_model, SemiSupervisedWrapper, MODEL_REGISTRY

def ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    errors = np.abs(y_true - y_pred)
    max_error = 3
    return 1 - (errors.sum() / (len(y_true) * max_error))

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ''
) -> Dict[str, float]:
    mask = ~np.isnan(y_true) & (y_true >= 0)
    if mask.sum() == 0:
        return {}

    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    metrics = {
        f'{prefix}accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        f'{prefix}f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    if prefix == 'state_':
        metrics[f'{prefix}ordinal_accuracy'] = ordinal_accuracy(y_true, y_pred)

    if prefix == 'trend_':
        worsening_class = 2
        worsening_mask = y_true == worsening_class
        if worsening_mask.sum() > 0:
            worsening_recall = (y_pred[worsening_mask] == worsening_class).mean()
            metrics[f'{prefix}worsening_recall'] = worsening_recall

    return metrics

def run_experiment(
    model_name: str,
    config: ExperimentConfig,
    model_params: Dict[str, Any] = None,
    test_start_date: str = '2025-08-01',
    semi_supervised: bool = False,
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

    data = temporal_train_test_split(df, feature_cols, test_date)

    print(f"Train samples: {len(data['X_train'])}, Test samples: {len(data['X_test'])}")

    with mlflow.start_run(run_name=f"{model_name}{'_semi' if semi_supervised else ''}"):
        mlflow.log_params({
            'model': model_name,
            'semi_supervised': semi_supervised,
            'test_start_date': test_start_date,
            'n_features': len(feature_cols),
            'n_train': len(data['X_train']),
            'n_test': len(data['X_test']),
            **(model_params or {})
        })

        model = get_model(model_name, model_params)

        if semi_supervised:
            unlabeled_mask = df['state'].isna()
            X_unlabeled = df.loc[unlabeled_mask, feature_cols].values

            print(f"Semi-supervised: {len(X_unlabeled)} unlabeled samples")

            wrapper = SemiSupervisedWrapper(model, confidence_threshold=0.8, max_iterations=5)
            wrapper.fit(
                data['X_train'],
                data['y_train_state'],
                data['y_train_trend'],
                X_unlabeled
            )
            model = wrapper.base_model

            for stat in wrapper.iteration_stats:
                mlflow.log_metrics({
                    f"semi_iter_{stat['iteration']}_added": stat['samples_added'],
                    f"semi_iter_{stat['iteration']}_total": stat['total_training']
                })
        else:
            model.fit(data['X_train'], data['y_train_state'], data['y_train_trend'])

        state_pred, trend_pred = model.predict(data['X_test'])

        state_metrics = compute_metrics(data['y_test_state'], state_pred, prefix='state_')
        mlflow.log_metrics(state_metrics)

        print("\nState Classification Metrics:")
        for k, v in state_metrics.items():
            print(f"  {k}: {v:.4f}")

        if trend_pred is not None:
            trend_metrics = compute_metrics(data['y_test_trend'], trend_pred, prefix='trend_')
            mlflow.log_metrics(trend_metrics)

            print("\nTrend Classification Metrics:")
            for k, v in trend_metrics.items():
                print(f"  {k}: {v:.4f}")

        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance(feature_cols)
            top_features = dict(list(importance.items())[:20])
            mlflow.log_dict(top_features, 'feature_importance.json')

            print("\nTop 10 Features:")
            for feat, imp in list(importance.items())[:10]:
                print(f"  {feat}: {imp:.4f}")

        cm = confusion_matrix(
            data['y_test_state'].astype(int),
            state_pred.astype(int)
        )
        mlflow.log_dict({'confusion_matrix': cm.tolist()}, 'confusion_matrix.json')

        print(f"\nConfusion Matrix:\n{cm}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")

        return {
            'run_id': run_id,
            'state_metrics': state_metrics,
            'trend_metrics': trend_metrics if trend_pred is not None else {},
            'feature_importance': importance if hasattr(model, 'get_feature_importance') else {}
        }

def run_all_baselines(config: ExperimentConfig, test_start_date: str = '2025-08-01'):
    results = {}

    for model_name in MODEL_REGISTRY.keys():
        print(f"\n{'='*60}")
        print(f"Running {model_name}")
        print('='*60)

        result = run_experiment(
            model_name=model_name,
            config=config,
            test_start_date=test_start_date,
            semi_supervised=False
        )
        if result:
            results[model_name] = result

        print(f"\n{'='*60}")
        print(f"Running {model_name} with semi-supervised learning")
        print('='*60)

        result_semi = run_experiment(
            model_name=model_name,
            config=config,
            test_start_date=test_start_date,
            semi_supervised=True
        )
        if result_semi:
            results[f'{model_name}_semi'] = result_semi

    return results

def main():
    parser = argparse.ArgumentParser(description='Train thyroid state prediction models')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=list(MODEL_REGISTRY.keys()) + ['all'])
    parser.add_argument('--semi-supervised', action='store_true')
    parser.add_argument('--test-start-date', type=str, default='2025-08-01')
    parser.add_argument('--experiment-name', type=str, default=None)
    args = parser.parse_args()

    config = ExperimentConfig()

    if args.model == 'all':
        results = run_all_baselines(config, args.test_start_date)
    else:
        results = run_experiment(
            model_name=args.model,
            config=config,
            test_start_date=args.test_start_date,
            semi_supervised=args.semi_supervised,
            experiment_name=args.experiment_name
        )

    print("\n" + "="*60)
    print("Training complete. View results with: mlflow ui")
    print("="*60)

if __name__ == '__main__':
    main()
