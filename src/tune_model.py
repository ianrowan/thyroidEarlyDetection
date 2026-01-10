import argparse
from itertools import product
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import xgboost as xgb

from .config import ExperimentConfig
from .dataset import prepare_dataset, temporal_train_test_split
from .train import ordinal_accuracy, compute_metrics

def ordinal_scorer(y_true, y_pred):
    return ordinal_accuracy(y_true, y_pred)

def get_respiratory_features(feature_cols):
    return [c for c in feature_cols if 'respiratory' in c or 'resp_rate' in c]

def get_top_features(feature_cols, n=20):
    priority = ['respiratory', 'resting_heart_rate', 'sleep_efficiency', 'heart_rate_min', 'heart_rate_p5']
    top = []
    for p in priority:
        top.extend([c for c in feature_cols if p in c and c not in top])
    remaining = [c for c in feature_cols if c not in top]
    return (top + remaining)[:n]

def run_tuning(n_classes: int = 3, feature_set: str = 'all'):
    config = ExperimentConfig()
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment('thyroid_tuning')

    df, feature_cols = prepare_dataset(config.data, config.train, n_classes=n_classes)
    test_date = pd.Timestamp('2025-08-01', tz='UTC')
    data = temporal_train_test_split(df, feature_cols, test_date)

    if feature_set == 'respiratory':
        selected_cols = get_respiratory_features(feature_cols)
        col_indices = [feature_cols.index(c) for c in selected_cols]
    elif feature_set == 'top20':
        selected_cols = get_top_features(feature_cols, 20)
        col_indices = [feature_cols.index(c) for c in selected_cols]
    else:
        selected_cols = feature_cols
        col_indices = list(range(len(feature_cols)))

    X_train = data['X_train'][:, col_indices]
    X_test = data['X_test'][:, col_indices]
    y_train = data['y_train_state']
    y_test = data['y_test_state']

    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)

    print(f"Features: {len(selected_cols)} ({feature_set})")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'scale_pos_weight': [1, 2, 3],
    }

    best_score = 0
    best_params = None
    results = []

    total = np.prod([len(v) for v in param_grid.values()])
    print(f"Testing {total} parameter combinations...")

    for i, (depth, lr, n_est, spw) in enumerate(product(
        param_grid['max_depth'],
        param_grid['learning_rate'],
        param_grid['n_estimators'],
        param_grid['scale_pos_weight']
    )):
        params = {
            'max_depth': depth,
            'learning_rate': lr,
            'n_estimators': n_est,
            'scale_pos_weight': spw,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, prefix='')
        ord_acc = metrics.get('ordinal_accuracy', ordinal_accuracy(y_test, y_pred))

        results.append({**params, 'ordinal_accuracy': ord_acc, 'balanced_accuracy': metrics.get('balanced_accuracy', 0)})

        if ord_acc > best_score:
            best_score = ord_acc
            best_params = params
            print(f"[{i+1}/{total}] New best: {ord_acc:.4f} - depth={depth}, lr={lr}, n_est={n_est}, spw={spw}")

    print(f"\nBest ordinal accuracy: {best_score:.4f}")
    print(f"Best params: {best_params}")

    with mlflow.start_run(run_name=f"tuning_{feature_set}_{n_classes}class"):
        mlflow.log_params({
            'feature_set': feature_set,
            'n_features': len(selected_cols),
            'n_classes': n_classes,
            **best_params
        })
        mlflow.log_metric('best_ordinal_accuracy', best_score)

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test.astype(int), y_pred.astype(int))
        print(f"\nConfusion Matrix:\n{cm}")

        metrics = compute_metrics(y_test, y_pred, prefix='state_')
        mlflow.log_metrics(metrics)

        importances = dict(zip(selected_cols, model.feature_importances_))
        top_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])
        print(f"\nTop features: {list(top_imp.keys())}")

    return best_params, best_score, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-classes', type=int, default=3)
    parser.add_argument('--feature-set', type=str, default='all', choices=['all', 'respiratory', 'top20'])
    args = parser.parse_args()

    run_tuning(n_classes=args.n_classes, feature_set=args.feature_set)

if __name__ == '__main__':
    main()
