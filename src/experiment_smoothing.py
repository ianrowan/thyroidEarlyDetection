import argparse
import json
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .dataset import prepare_dataset
from .models import EarlyDetectionModel


def apply_moving_average(risk_scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return risk_scores
    smoothed = pd.Series(risk_scores).rolling(window=window, min_periods=1, center=False).mean()
    return smoothed.values


def apply_exponential_moving_average(risk_scores: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return risk_scores
    smoothed = pd.Series(risk_scores).ewm(span=span, adjust=False).mean()
    return smoothed.values


def apply_confirmation_window(risk_scores: np.ndarray, threshold: float, n_confirm: int) -> np.ndarray:
    alerts = (risk_scores >= threshold).astype(int)
    confirmed = np.zeros_like(alerts)
    for i in range(len(alerts)):
        if i < n_confirm - 1:
            confirmed[i] = 0
        else:
            if all(alerts[i - n_confirm + 1 : i + 1]):
                confirmed[i] = 1
    return confirmed


def evaluate_smoothing(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    threshold: float,
    test_start: str
):
    test_mask = df['window_start'] >= pd.Timestamp(test_start, tz='UTC')
    labeled_mask = df['state'].notna()
    combined_mask = test_mask & labeled_mask

    test_df = df[combined_mask].copy()
    test_risk = risk_scores[combined_mask]
    test_smoothed = smoothed_scores[combined_mask]
    test_labels = test_df['state'].values

    raw_alerts = (test_risk >= threshold).astype(int)
    smoothed_alerts = (test_smoothed >= threshold).astype(int)

    true_hyper = (test_labels >= 1).astype(int)

    raw_fp = ((raw_alerts == 1) & (true_hyper == 0)).sum()
    raw_fn = ((raw_alerts == 0) & (true_hyper == 1)).sum()
    raw_tp = ((raw_alerts == 1) & (true_hyper == 1)).sum()
    raw_tn = ((raw_alerts == 0) & (true_hyper == 0)).sum()

    smooth_fp = ((smoothed_alerts == 1) & (true_hyper == 0)).sum()
    smooth_fn = ((smoothed_alerts == 0) & (true_hyper == 1)).sum()
    smooth_tp = ((smoothed_alerts == 1) & (true_hyper == 1)).sum()
    smooth_tn = ((smoothed_alerts == 0) & (true_hyper == 0)).sum()

    raw_precision = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0
    raw_recall = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else 0

    smooth_precision = smooth_tp / (smooth_tp + smooth_fp) if (smooth_tp + smooth_fp) > 0 else 0
    smooth_recall = smooth_tp / (smooth_tp + smooth_fn) if (smooth_tp + smooth_fn) > 0 else 0

    return {
        'raw': {
            'fp': int(raw_fp), 'fn': int(raw_fn), 'tp': int(raw_tp), 'tn': int(raw_tn),
            'precision': raw_precision, 'recall': raw_recall,
            'total_alerts': int(raw_alerts.sum())
        },
        'smoothed': {
            'fp': int(smooth_fp), 'fn': int(smooth_fn), 'tp': int(smooth_tp), 'tn': int(smooth_tn),
            'precision': smooth_precision, 'recall': smooth_recall,
            'total_alerts': int(smoothed_alerts.sum())
        },
        'fp_reduction': int(raw_fp - smooth_fp),
        'fn_increase': int(smooth_fn - raw_fn),
        'test_windows': len(test_df),
        'hyper_windows': int(true_hyper.sum()),
        'normal_windows': int((true_hyper == 0).sum())
    }


def find_hyper_onset_dates(df: pd.DataFrame) -> list:
    labeled = df[df['state'].notna()].sort_values('window_start')
    onset_dates = []
    prev_state = 0

    for _, row in labeled.iterrows():
        current_state = row['state']
        if prev_state == 0 and current_state >= 1:
            onset_dates.append(row['window_start'])
        prev_state = current_state

    return onset_dates


def classify_false_positive(fp_date: pd.Timestamp, onset_dates: list, pre_onset_days: int = 30) -> str:
    for onset in onset_dates:
        days_before = (onset - fp_date).days
        if 0 <= days_before <= pre_onset_days:
            return 'pre_onset'
    return 'isolated'


def analyze_false_positives(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    threshold: float,
    test_start: str
):
    test_mask = df['window_start'] >= pd.Timestamp(test_start, tz='UTC')
    labeled_mask = df['state'].notna()
    combined_mask = test_mask & labeled_mask

    test_df = df[combined_mask].copy()
    test_risk = risk_scores[combined_mask]
    test_labels = test_df['state'].values

    onset_dates = find_hyper_onset_dates(df)

    alerts = (test_risk >= threshold).astype(int)
    true_hyper = (test_labels >= 1).astype(int)

    fp_mask = (alerts == 1) & (true_hyper == 0)
    fp_indices = np.where(fp_mask)[0]

    fp_details = []
    for idx in fp_indices:
        row = test_df.iloc[idx]
        risk = test_risk[idx]

        prev_context = []
        for back in range(1, 4):
            if idx - back >= 0:
                prev_risk = test_risk[idx - back]
                prev_label = test_labels[idx - back]
                if not np.isnan(prev_label):
                    prev_context.append({
                        'offset': -back,
                        'risk': float(prev_risk),
                        'label': int(prev_label)
                    })

        next_context = []
        for fwd in range(1, 4):
            if idx + fwd < len(test_risk):
                next_risk = test_risk[idx + fwd]
                next_label = test_labels[idx + fwd]
                if not np.isnan(next_label):
                    next_context.append({
                        'offset': fwd,
                        'risk': float(next_risk),
                        'label': int(next_label)
                    })

        fp_type = classify_false_positive(row['window_start'], onset_dates)

        fp_details.append({
            'window_start': row['window_start'].strftime('%Y-%m-%d'),
            'window_end': row['window_end'].strftime('%Y-%m-%d'),
            'risk': float(risk),
            'label': int(row['state']),
            'prev_context': prev_context,
            'next_context': next_context,
            'isolated': all(r['risk'] < threshold for r in prev_context + next_context) if prev_context or next_context else True,
            'fp_type': fp_type
        })

    return fp_details


def run_smoothing_experiment(n_classes: int = 3, test_start: str = '2025-08-01', threshold: float = 0.35):
    config = ExperimentConfig()
    df, feature_cols = prepare_dataset(config.data, config.train, n_classes=n_classes)

    labeled_mask = df['state'].notna()
    all_df = df.copy()

    train_mask = df['window_start'] < pd.Timestamp(test_start, tz='UTC')
    train_df = df[train_mask & labeled_mask]

    X_train = train_df[feature_cols].values
    y_train = train_df['state'].values

    print(f"Training EarlyDetectionModel on {len(X_train)} samples...")
    model = EarlyDetectionModel()
    model.fit(X_train, y_train, feature_names=feature_cols)

    X_all = all_df[feature_cols].values
    risk_scores = model.get_risk_score(X_all)

    all_df['risk_raw'] = risk_scores

    print("\n" + "="*70)
    print("FALSE POSITIVE ANALYSIS (Raw Model)")
    print("="*70)

    fp_details = analyze_false_positives(all_df, risk_scores, threshold, test_start)
    isolated_fps = [fp for fp in fp_details if fp['isolated']]

    pre_onset_fps = [fp for fp in fp_details if fp['fp_type'] == 'pre_onset']
    true_isolated_fps = [fp for fp in fp_details if fp['fp_type'] == 'isolated']

    print(f"\nTotal False Positives: {len(fp_details)}")
    print(f"  Pre-onset signals (within 30d of hyper start): {len(pre_onset_fps)}")
    print(f"  True isolated FPs (far from any episode): {len(true_isolated_fps)}")
    print(f"  Isolated from neighboring alerts: {len(isolated_fps)}")

    print("\nFalse Positive Details:")
    for fp in fp_details:
        markers = []
        if fp['isolated']:
            markers.append("NO-NEIGHBORS")
        markers.append(f"TYPE:{fp['fp_type'].upper()}")
        marker_str = f" [{', '.join(markers)}]"
        print(f"  {fp['window_start']} to {fp['window_end']}: risk={fp['risk']:.3f}{marker_str}")
        if fp['prev_context']:
            prev_str = ', '.join([f"t{c['offset']}={c['risk']:.2f}" for c in fp['prev_context']])
            print(f"    Previous: {prev_str}")
        if fp['next_context']:
            next_str = ', '.join([f"t+{c['offset']}={c['risk']:.2f}" for c in fp['next_context']])
            print(f"    Next: {next_str}")

    results = {
        'config': {
            'test_start': test_start,
            'threshold': threshold,
            'n_classes': n_classes,
            'train_samples': len(X_train)
        },
        'raw_fp_analysis': {
            'total_fps': len(fp_details),
            'isolated_fps': len(isolated_fps),
            'details': fp_details
        },
        'smoothing_experiments': []
    }

    print("\n" + "="*70)
    print("SMOOTHING EXPERIMENTS")
    print("="*70)

    sma_windows = [2, 3, 4, 5]

    print("\n--- Simple Moving Average (SMA) ---")
    for window in sma_windows:
        smoothed = apply_moving_average(risk_scores, window)
        all_df[f'risk_sma_{window}'] = smoothed

        eval_result = evaluate_smoothing(all_df, risk_scores, smoothed, threshold, test_start)

        print(f"\nSMA Window={window}:")
        print(f"  Raw:      FP={eval_result['raw']['fp']:2d}, FN={eval_result['raw']['fn']:2d}, "
              f"Precision={eval_result['raw']['precision']:.3f}, Recall={eval_result['raw']['recall']:.3f}")
        print(f"  Smoothed: FP={eval_result['smoothed']['fp']:2d}, FN={eval_result['smoothed']['fn']:2d}, "
              f"Precision={eval_result['smoothed']['precision']:.3f}, Recall={eval_result['smoothed']['recall']:.3f}")
        print(f"  FP Reduction: {eval_result['fp_reduction']}, FN Increase: {eval_result['fn_increase']}")

        results['smoothing_experiments'].append({
            'method': 'SMA',
            'params': {'window': window},
            **eval_result
        })

    ema_spans = [2, 3, 4, 5]

    print("\n--- Exponential Moving Average (EMA) ---")
    for span in ema_spans:
        smoothed = apply_exponential_moving_average(risk_scores, span)
        all_df[f'risk_ema_{span}'] = smoothed

        eval_result = evaluate_smoothing(all_df, risk_scores, smoothed, threshold, test_start)

        print(f"\nEMA Span={span}:")
        print(f"  Raw:      FP={eval_result['raw']['fp']:2d}, FN={eval_result['raw']['fn']:2d}, "
              f"Precision={eval_result['raw']['precision']:.3f}, Recall={eval_result['raw']['recall']:.3f}")
        print(f"  Smoothed: FP={eval_result['smoothed']['fp']:2d}, FN={eval_result['smoothed']['fn']:2d}, "
              f"Precision={eval_result['smoothed']['precision']:.3f}, Recall={eval_result['smoothed']['recall']:.3f}")
        print(f"  FP Reduction: {eval_result['fp_reduction']}, FN Increase: {eval_result['fn_increase']}")

        results['smoothing_experiments'].append({
            'method': 'EMA',
            'params': {'span': span},
            **eval_result
        })

    confirm_windows = [2, 3]

    print("\n--- Confirmation Window (require N consecutive alerts) ---")
    for n_confirm in confirm_windows:
        confirmed = apply_confirmation_window(risk_scores, threshold, n_confirm)

        test_mask = all_df['window_start'] >= pd.Timestamp(test_start, tz='UTC')
        labeled_mask = all_df['state'].notna()
        combined_mask = test_mask & labeled_mask
        test_labels = all_df[combined_mask]['state'].values
        test_confirmed = confirmed[combined_mask]
        test_risk = risk_scores[combined_mask]

        raw_alerts = (test_risk >= threshold).astype(int)
        true_hyper = (test_labels >= 1).astype(int)

        raw_fp = ((raw_alerts == 1) & (true_hyper == 0)).sum()
        confirm_fp = ((test_confirmed == 1) & (true_hyper == 0)).sum()
        confirm_fn = ((test_confirmed == 0) & (true_hyper == 1)).sum()
        confirm_tp = ((test_confirmed == 1) & (true_hyper == 1)).sum()

        confirm_precision = confirm_tp / (confirm_tp + confirm_fp) if (confirm_tp + confirm_fp) > 0 else 0
        confirm_recall = confirm_tp / (confirm_tp + confirm_fn) if (confirm_tp + confirm_fn) > 0 else 0

        print(f"\nConfirmation N={n_confirm}:")
        print(f"  Confirmed: FP={confirm_fp:2d}, FN={confirm_fn:2d}, "
              f"Precision={confirm_precision:.3f}, Recall={confirm_recall:.3f}")
        print(f"  FP Reduction: {raw_fp - confirm_fp}")

        results['smoothing_experiments'].append({
            'method': 'confirmation',
            'params': {'n_confirm': n_confirm},
            'raw': results['smoothing_experiments'][0]['raw'],
            'smoothed': {
                'fp': int(confirm_fp),
                'fn': int(confirm_fn),
                'tp': int(confirm_tp),
                'precision': confirm_precision,
                'recall': confirm_recall
            },
            'fp_reduction': int(raw_fp - confirm_fp)
        })

    thresholds = [0.30, 0.40, 0.45, 0.50]

    print("\n--- Alternative Thresholds (with SMA-3) ---")
    smoothed_sma3 = apply_moving_average(risk_scores, 3)

    for t in thresholds:
        eval_result = evaluate_smoothing(all_df, risk_scores, smoothed_sma3, t, test_start)

        print(f"\nThreshold={t:.2f} with SMA-3:")
        print(f"  Raw (t={t:.2f}):      FP={eval_result['raw']['fp']:2d}, FN={eval_result['raw']['fn']:2d}, "
              f"Precision={eval_result['raw']['precision']:.3f}, Recall={eval_result['raw']['recall']:.3f}")
        print(f"  Smoothed:            FP={eval_result['smoothed']['fp']:2d}, FN={eval_result['smoothed']['fn']:2d}, "
              f"Precision={eval_result['smoothed']['precision']:.3f}, Recall={eval_result['smoothed']['recall']:.3f}")

        results['smoothing_experiments'].append({
            'method': 'SMA+threshold',
            'params': {'window': 3, 'threshold': t},
            **eval_result
        })

    print("\n" + "="*70)
    print("SUMMARY: BEST APPROACHES")
    print("="*70)

    best_by_fp = min(
        [r for r in results['smoothing_experiments'] if r['smoothed']['recall'] >= 0.7],
        key=lambda r: r['smoothed']['fp'],
        default=None
    )

    best_balance = max(
        results['smoothing_experiments'],
        key=lambda r: r['smoothed']['precision'] + r['smoothed']['recall']
    )

    if best_by_fp:
        print(f"\nBest FP Reduction (recall >= 0.7):")
        print(f"  Method: {best_by_fp['method']}, Params: {best_by_fp['params']}")
        print(f"  FP: {best_by_fp['smoothed']['fp']}, Recall: {best_by_fp['smoothed']['recall']:.3f}")

    print(f"\nBest Balance (precision + recall):")
    print(f"  Method: {best_balance['method']}, Params: {best_balance['params']}")
    print(f"  FP: {best_balance['smoothed']['fp']}, Precision: {best_balance['smoothed']['precision']:.3f}, "
          f"Recall: {best_balance['smoothed']['recall']:.3f}")

    output_path = Path('data/smoothing_experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    csv_path = Path('data/smoothing_risk_scores.csv')
    export_cols = ['window_start', 'window_end', 'state', 'risk_raw']
    export_cols += [c for c in all_df.columns if c.startswith('risk_sma') or c.startswith('risk_ema')]
    all_df[export_cols].to_csv(csv_path, index=False)
    print(f"Risk scores saved to {csv_path}")

    return results


def analyze_smoothing_on_isolated_fps(
    labeled_df: pd.DataFrame,
    risk_scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    onset_dates: list
):
    results = {}

    alerts = (risk_scores >= threshold).astype(int)
    true_hyper = (y >= 1).astype(int)
    fp_mask = (alerts == 1) & (true_hyper == 0)

    isolated_fp_indices = []
    for idx in np.where(fp_mask)[0]:
        row = labeled_df.iloc[idx]
        fp_type = classify_false_positive(row['window_start'], onset_dates)
        if fp_type == 'isolated':
            isolated_fp_indices.append(idx)

    raw_isolated_fps = len(isolated_fp_indices)
    raw_tp = ((alerts == 1) & (true_hyper == 1)).sum()
    raw_fn = ((alerts == 0) & (true_hyper == 1)).sum()

    print(f"\nRaw Model (threshold={threshold}):")
    print(f"  Isolated FPs: {raw_isolated_fps}")
    print(f"  True Positives: {raw_tp}")
    print(f"  False Negatives: {raw_fn}")
    print(f"  Recall: {raw_tp/(raw_tp+raw_fn):.3f}")

    results['raw'] = {
        'isolated_fps': raw_isolated_fps,
        'tp': int(raw_tp),
        'fn': int(raw_fn),
        'recall': raw_tp/(raw_tp+raw_fn)
    }

    sma_windows = [2, 3, 4, 5]
    print("\n--- SMA Smoothing ---")
    for window in sma_windows:
        smoothed = apply_moving_average(risk_scores, window)
        smooth_alerts = (smoothed >= threshold).astype(int)

        smooth_isolated_fps = 0
        for idx in isolated_fp_indices:
            if smooth_alerts[idx] == 1:
                smooth_isolated_fps += 1

        smooth_tp = ((smooth_alerts == 1) & (true_hyper == 1)).sum()
        smooth_fn = ((smooth_alerts == 0) & (true_hyper == 1)).sum()
        recall = smooth_tp / (smooth_tp + smooth_fn)

        fp_removed = raw_isolated_fps - smooth_isolated_fps
        print(f"  SMA-{window}: Isolated FPs={smooth_isolated_fps} (-{fp_removed}), Recall={recall:.3f}")

        results[f'sma_{window}'] = {
            'isolated_fps': smooth_isolated_fps,
            'fp_removed': fp_removed,
            'tp': int(smooth_tp),
            'fn': int(smooth_fn),
            'recall': recall
        }

    ema_spans = [2, 3, 4, 5]
    print("\n--- EMA Smoothing ---")
    for span in ema_spans:
        smoothed = apply_exponential_moving_average(risk_scores, span)
        smooth_alerts = (smoothed >= threshold).astype(int)

        smooth_isolated_fps = 0
        for idx in isolated_fp_indices:
            if smooth_alerts[idx] == 1:
                smooth_isolated_fps += 1

        smooth_tp = ((smooth_alerts == 1) & (true_hyper == 1)).sum()
        smooth_fn = ((smooth_alerts == 0) & (true_hyper == 1)).sum()
        recall = smooth_tp / (smooth_tp + smooth_fn)

        fp_removed = raw_isolated_fps - smooth_isolated_fps
        print(f"  EMA-{span}: Isolated FPs={smooth_isolated_fps} (-{fp_removed}), Recall={recall:.3f}")

        results[f'ema_{span}'] = {
            'isolated_fps': smooth_isolated_fps,
            'fp_removed': fp_removed,
            'tp': int(smooth_tp),
            'fn': int(smooth_fn),
            'recall': recall
        }

    return results


def analyze_fp_clustering(labeled_df: pd.DataFrame, risk_scores: np.ndarray, threshold: float, onset_dates: list):
    alerts = (risk_scores >= threshold).astype(int)
    y = labeled_df['state'].values
    true_hyper = (y >= 1).astype(int)
    fp_mask = (alerts == 1) & (true_hyper == 0)

    isolated_fp_indices = []
    for idx in np.where(fp_mask)[0]:
        row = labeled_df.iloc[idx]
        fp_type = classify_false_positive(row['window_start'], onset_dates)
        if fp_type == 'isolated':
            isolated_fp_indices.append(idx)

    runs = []
    current_run = []

    for idx in isolated_fp_indices:
        if not current_run or idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
            current_run = [idx]

    if current_run:
        runs.append(current_run)

    print("\n--- Isolated FP Clustering Analysis ---")
    print(f"  Total isolated FPs: {len(isolated_fp_indices)}")
    print(f"  Number of FP runs: {len(runs)}")

    run_lengths = [len(r) for r in runs]
    if run_lengths:
        print(f"  Run length distribution:")
        for length in sorted(set(run_lengths)):
            count = run_lengths.count(length)
            print(f"    Length {length}: {count} runs")

        single_fp_runs = sum(1 for r in runs if len(r) == 1)
        print(f"  Single-window FPs (most filterable): {single_fp_runs}")

    return runs


def run_full_history_fp_analysis(n_classes: int = 3, threshold: float = 0.35):
    config = ExperimentConfig()
    df, feature_cols = prepare_dataset(config.data, config.train, n_classes=n_classes)

    labeled_mask = df['state'].notna()
    labeled_df = df[labeled_mask].copy().reset_index(drop=True)

    X = labeled_df[feature_cols].values
    y = labeled_df['state'].values

    print(f"Training on full labeled data: {len(X)} samples")
    model = EarlyDetectionModel()
    model.fit(X, y, feature_names=feature_cols)

    risk_scores = model.get_risk_score(X)
    onset_dates = find_hyper_onset_dates(df)

    print(f"\n{'='*70}")
    print("FULL HISTORY FALSE POSITIVE ANALYSIS")
    print(f"{'='*70}")

    alerts = (risk_scores >= threshold).astype(int)
    true_hyper = (y >= 1).astype(int)
    fp_mask = (alerts == 1) & (true_hyper == 0)
    fp_indices = np.where(fp_mask)[0]

    pre_onset = 0
    isolated = 0

    print(f"\nTotal FPs: {fp_mask.sum()}")
    print("\nDetails:")
    for idx in fp_indices:
        row = labeled_df.iloc[idx]
        risk = risk_scores[idx]
        fp_type = classify_false_positive(row['window_start'], onset_dates)

        if fp_type == 'pre_onset':
            pre_onset += 1
        else:
            isolated += 1

        print(f"  {row['window_start'].strftime('%Y-%m-%d')}: risk={risk:.3f} [{fp_type.upper()}]")

    print(f"\nSummary:")
    print(f"  Pre-onset signals: {pre_onset}")
    print(f"  True isolated FPs: {isolated}")

    runs = analyze_fp_clustering(labeled_df, risk_scores, threshold, onset_dates)

    smoothing_results = analyze_smoothing_on_isolated_fps(
        labeled_df, risk_scores, y, threshold, onset_dates
    )

    results_path = Path('data/full_history_fp_analysis.json')
    output = {
        'threshold': threshold,
        'total_fps': int(fp_mask.sum()),
        'pre_onset_fps': pre_onset,
        'isolated_fps': isolated,
        'fp_runs': [{'start_idx': r[0], 'length': len(r)} for r in runs],
        'smoothing_results': smoothing_results
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Experiment with risk score smoothing')
    parser.add_argument('--n-classes', type=int, default=3, choices=[3, 4])
    parser.add_argument('--test-start', type=str, default='2025-08-01')
    parser.add_argument('--threshold', type=float, default=0.35)
    parser.add_argument('--full-history', action='store_true', help='Analyze FPs across full history')
    args = parser.parse_args()

    if args.full_history:
        run_full_history_fp_analysis(n_classes=args.n_classes, threshold=args.threshold)
    else:
        run_smoothing_experiment(
            n_classes=args.n_classes,
            test_start=args.test_start,
            threshold=args.threshold
        )


if __name__ == '__main__':
    main()
