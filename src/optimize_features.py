import argparse
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from itertools import combinations
from typing import List, Tuple, Dict

from .config import ExperimentConfig
from .dataset import prepare_dataset, temporal_train_test_split
from .train import ordinal_accuracy, compute_metrics


def get_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    return {
        'respiratory': [c for c in feature_cols if 'respiratory' in c or 'resp_rate' in c],
        'rhr': [c for c in feature_cols if 'resting_heart_rate' in c or 'rhr_' in c],
        'hr': [c for c in feature_cols if 'heart_rate' in c and 'resting' not in c],
        'hrv': [c for c in feature_cols if 'hrv' in c],
        'sleep': [c for c in feature_cols if 'sleep' in c],
        'delta': [c for c in feature_cols if 'delta' in c or 'deviation' in c],
        'trend_features': [c for c in feature_cols if 'slope' in c or 'delta' in c],
    }


def evaluate_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: Dict = None
) -> Tuple[float, np.ndarray, np.ndarray]:

    if params is None:
        params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ord_acc = ordinal_accuracy(y_test, y_pred)

    return ord_acc, y_pred, model.feature_importances_


def run_feature_optimization(n_classes: int = 3):
    config = ExperimentConfig()
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment('thyroid_feature_optimization')

    df, feature_cols = prepare_dataset(config.data, config.train, n_classes=n_classes)
    test_date = pd.Timestamp('2025-08-01', tz='UTC')
    data = temporal_train_test_split(df, feature_cols, test_date)

    X_train_full = np.nan_to_num(data['X_train'], nan=0)
    X_test_full = np.nan_to_num(data['X_test'], nan=0)
    y_train = data['y_train_state']
    y_test = data['y_test_state']

    groups = get_feature_groups(feature_cols)

    print(f"Feature groups:")
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} features")

    results = []

    print("\n" + "="*60)
    print("Testing individual feature groups")
    print("="*60)

    for group_name, cols in groups.items():
        if not cols:
            continue

        indices = [feature_cols.index(c) for c in cols]
        X_train = X_train_full[:, indices]
        X_test = X_test_full[:, indices]

        ord_acc, y_pred, importances = evaluate_features(X_train, X_test, y_train, y_test)

        cm = np.zeros((3, 3), dtype=int)
        for true, pred in zip(y_test.astype(int), y_pred.astype(int)):
            cm[true, pred] += 1

        hyper_correct = cm[1, 1]
        hyper_total = cm[1, :].sum()
        severe_correct = cm[2, 2]
        severe_total = cm[2, :].sum()

        results.append({
            'name': group_name,
            'n_features': len(cols),
            'ordinal_accuracy': ord_acc,
            'hyper_detection': f"{hyper_correct}/{hyper_total}",
            'severe_detection': f"{severe_correct}/{severe_total}",
            'features': cols
        })

        print(f"\n{group_name} ({len(cols)} features): {ord_acc:.1%} ordinal, hyper {hyper_correct}/{hyper_total}, severe {severe_correct}/{severe_total}")

    print("\n" + "="*60)
    print("Testing feature group combinations")
    print("="*60)

    key_groups = ['respiratory', 'rhr', 'delta', 'sleep']

    for r in range(2, len(key_groups) + 1):
        for combo in combinations(key_groups, r):
            combined_cols = []
            for g in combo:
                combined_cols.extend(groups.get(g, []))

            if not combined_cols:
                continue

            indices = [feature_cols.index(c) for c in combined_cols]
            X_train = X_train_full[:, indices]
            X_test = X_test_full[:, indices]

            ord_acc, y_pred, importances = evaluate_features(X_train, X_test, y_train, y_test)

            cm = np.zeros((3, 3), dtype=int)
            for true, pred in zip(y_test.astype(int), y_pred.astype(int)):
                cm[true, pred] += 1

            hyper_correct = cm[1, 1]
            hyper_total = cm[1, :].sum()
            severe_correct = cm[2, 2]
            severe_total = cm[2, :].sum()

            combo_name = '+'.join(combo)
            results.append({
                'name': combo_name,
                'n_features': len(combined_cols),
                'ordinal_accuracy': ord_acc,
                'hyper_detection': f"{hyper_correct}/{hyper_total}",
                'severe_detection': f"{severe_correct}/{severe_total}",
                'features': combined_cols
            })

            print(f"\n{combo_name} ({len(combined_cols)} features): {ord_acc:.1%} ordinal, hyper {hyper_correct}/{hyper_total}, severe {severe_correct}/{severe_total}")

    print("\n" + "="*60)
    print("Testing curated feature sets")
    print("="*60)

    curated_sets = {
        'minimal_respiratory': [
            'respiratory_rate_mean', 'respiratory_rate_median', 'respiratory_rate_std',
            'resp_rate_delta'
        ],
        'minimal_vital_signs': [
            'respiratory_rate_mean', 'resting_heart_rate_mean',
            'resp_rate_delta', 'rhr_delta'
        ],
        'deviation_focused': [
            'rhr_deviation_14d', 'rhr_deviation_30d',
            'rhr_delta', 'resp_rate_delta',
            'respiratory_rate_mean', 'resting_heart_rate_mean'
        ],
        'high_signal': [
            'respiratory_rate_median', 'respiratory_rate_mean', 'respiratory_rate_std',
            'resting_heart_rate_median', 'resting_heart_rate_min',
            'resp_rate_delta', 'rhr_deviation_14d'
        ],
        'sleep_vital_combo': [
            'respiratory_rate_mean', 'resting_heart_rate_mean',
            'sleep_sleep_efficiency', 'sleep_total_minutes',
            'resp_rate_delta', 'rhr_delta'
        ],
    }

    for set_name, cols in curated_sets.items():
        valid_cols = [c for c in cols if c in feature_cols]
        if len(valid_cols) < len(cols):
            missing = set(cols) - set(valid_cols)
            print(f"\n{set_name}: Missing features: {missing}")

        if not valid_cols:
            continue

        indices = [feature_cols.index(c) for c in valid_cols]
        X_train = X_train_full[:, indices]
        X_test = X_test_full[:, indices]

        ord_acc, y_pred, importances = evaluate_features(X_train, X_test, y_train, y_test)

        cm = np.zeros((3, 3), dtype=int)
        for true, pred in zip(y_test.astype(int), y_pred.astype(int)):
            cm[true, pred] += 1

        hyper_correct = cm[1, 1]
        hyper_total = cm[1, :].sum()
        severe_correct = cm[2, 2]
        severe_total = cm[2, :].sum()

        results.append({
            'name': f'curated:{set_name}',
            'n_features': len(valid_cols),
            'ordinal_accuracy': ord_acc,
            'hyper_detection': f"{hyper_correct}/{hyper_total}",
            'severe_detection': f"{severe_correct}/{severe_total}",
            'features': valid_cols
        })

        print(f"\n{set_name} ({len(valid_cols)} features): {ord_acc:.1%} ordinal, hyper {hyper_correct}/{hyper_total}, severe {severe_correct}/{severe_total}")

        top_idx = np.argsort(importances)[-3:][::-1]
        print(f"  Top features: {[valid_cols[i] for i in top_idx]}")

    print("\n" + "="*60)
    print("Results Summary (sorted by ordinal accuracy)")
    print("="*60)

    results.sort(key=lambda x: x['ordinal_accuracy'], reverse=True)

    for i, r in enumerate(results[:15], 1):
        print(f"{i:2}. {r['name']:<30} {r['ordinal_accuracy']:.1%} ord, hyper {r['hyper_detection']}, severe {r['severe_detection']} ({r['n_features']} feat)")

    best = results[0]
    print(f"\nBest: {best['name']} with {best['ordinal_accuracy']:.1%} ordinal accuracy")
    print(f"Features: {best['features']}")

    with mlflow.start_run(run_name='feature_optimization_summary'):
        mlflow.log_params({
            'n_classes': n_classes,
            'n_combinations_tested': len(results),
            'best_feature_set': best['name'],
            'best_n_features': best['n_features']
        })
        mlflow.log_metric('best_ordinal_accuracy', best['ordinal_accuracy'])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-classes', type=int, default=3)
    args = parser.parse_args()

    run_feature_optimization(n_classes=args.n_classes)


if __name__ == '__main__':
    main()
