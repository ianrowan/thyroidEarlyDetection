import argparse
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
from lxml import etree
from tqdm import tqdm

from .parse_health_export import RELEVANT_TYPES, SLEEP_VALUE_MAP, parse_datetime
from .feature_extraction import load_parquet_files, extract_features

MODELS_DIR = Path('models')

STATE_NAMES = {0: 'Normal', 1: 'Hyper', 2: 'Severe'}
RISK_LABELS = [
    (0.6, 'HIGH', 'Take action'),
    (0.4, 'ELEVATED', 'Monitor closely'),
    (0.25, 'MODERATE', 'Watch trends'),
    (0.0, 'LOW', 'Normal baseline'),
]

def load_models():
    early_path = MODELS_DIR / 'early_detection.joblib'
    hybrid_path = MODELS_DIR / 'hybrid_dual.joblib'
    metadata_path = MODELS_DIR / 'metadata.json'

    if not early_path.exists() or not hybrid_path.exists():
        raise FileNotFoundError(
            "Models not found. Run: venv/bin/python -m src.save_models"
        )

    early_model = joblib.load(early_path)
    hybrid_model = joblib.load(hybrid_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    return early_model, hybrid_model, metadata


def load_ios_model():
    """Load the iOS-compatible model (raw XGBoost trained on 3 RHR features)."""
    ios_path = MODELS_DIR / 'early_detection_ios.joblib'
    metadata_path = MODELS_DIR / 'metadata.json'

    if not ios_path.exists():
        raise FileNotFoundError(
            "iOS model not found. Expected: models/early_detection_ios.joblib"
        )

    ios_model = joblib.load(ios_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    return ios_model, metadata

def parse_export_since(xml_path: Path, since_date: datetime = None):
    records = {name: [] for name in RELEVANT_TYPES.values()}

    context = etree.iterparse(str(xml_path), events=('end',), tag='Record')

    for event, elem in tqdm(context, desc='Parsing health data'):
        record_type = elem.get('type')

        if record_type in RELEVANT_TYPES:
            short_name = RELEVANT_TYPES[record_type]
            start_date = parse_datetime(elem.get('startDate'))

            if start_date is None:
                elem.clear()
                continue

            if since_date and start_date < since_date:
                elem.clear()
                continue

            end_date = parse_datetime(elem.get('endDate'))

            record = {
                'start_date': start_date,
                'end_date': end_date,
                'source': elem.get('sourceName', ''),
                'device': elem.get('device', ''),
            }

            if short_name == 'sleep':
                value = elem.get('value', '')
                record['value'] = SLEEP_VALUE_MAP.get(value, value)
                record['duration_minutes'] = (end_date - start_date).total_seconds() / 60 if end_date else 0
            else:
                try:
                    record['value'] = float(elem.get('value', 0))
                except ValueError:
                    record['value'] = None
                record['unit'] = elem.get('unit', '')

            records[short_name].append(record)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    del context

    data = {}
    for name, recs in records.items():
        if recs:
            df = pd.DataFrame(recs)
            df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'], utc=True)
            data[name] = df

    return data

def get_features_for_inference(data: dict, feature_cols: list) -> pd.DataFrame:
    features = extract_features(data, window_days=5)

    for col in feature_cols:
        if col not in features.columns:
            features[col] = np.nan

    return features


def compute_ios_features(rhr_df: pd.DataFrame, target_date: pd.Timestamp, lookback_days: int = 40) -> dict:
    """
    Compute features matching the iOS FeatureComputer.swift logic.

    iOS logic:
    1. Get 40 days of RHR samples ending at target_date
    2. Aggregate to daily averages
    3. currentWindow = last 5 daily averages
    4. priorWindow = 5 daily averages before currentWindow
    5. rhr_delta = mean(currentWindow) - mean(priorWindow)
    6. baseline14d = 14 daily averages before current window
    7. baseline30d = 30 daily averages before current window
    8. deviation = (currentWindowMean - baselineMean) / max(baselineStd, 0.01)
    """
    start_date = target_date - timedelta(days=lookback_days)

    mask = (rhr_df['start_date'] >= start_date) & (rhr_df['start_date'] <= target_date)
    samples = rhr_df[mask].copy()

    if len(samples) < 10:
        return None

    # Aggregate by day (iOS aggregateByDay)
    samples['date'] = samples['start_date'].dt.date
    daily_avg = samples.groupby('date')['value'].mean().sort_index()
    daily_values = daily_avg.values

    if len(daily_values) < 10:
        return None

    # currentWindow = last 5 daily averages
    current_window = daily_values[-5:]
    current_window_mean = np.mean(current_window)

    # priorWindow = 5 daily averages before current window
    prior_window = daily_values[-10:-5]
    prior_window_mean = np.mean(prior_window) if len(prior_window) >= 5 else np.nan

    rhr_delta = current_window_mean - prior_window_mean if not np.isnan(prior_window_mean) else 0.0

    # baseline14d = 14 daily values before current window
    baseline14d = daily_values[:-5][-14:]
    baseline14_mean = np.mean(baseline14d)
    baseline14_std = np.std(baseline14d, ddof=1) if len(baseline14d) > 1 else 0

    # baseline30d = 30 daily values before current window
    baseline30d = daily_values[:-5][-30:]
    baseline30_mean = np.mean(baseline30d)
    baseline30_std = np.std(baseline30d, ddof=1) if len(baseline30d) > 1 else 0

    # iOS uses epsilon = 0.01 to avoid division by zero
    epsilon = 0.01
    rhr_deviation_14d = (current_window_mean - baseline14_mean) / max(baseline14_std, epsilon)
    rhr_deviation_30d = (current_window_mean - baseline30_mean) / max(baseline30_std, epsilon)

    return {
        'window_start': pd.Timestamp(daily_avg.index[-5]),
        'window_end': pd.Timestamp(daily_avg.index[-1]),
        'rhr_deviation_14d': rhr_deviation_14d,
        'rhr_deviation_30d': rhr_deviation_30d,
        'rhr_delta': rhr_delta,
        'resting_heart_rate_mean': current_window_mean,
    }


def get_ios_features_for_dates(data: dict, target_dates: list) -> pd.DataFrame:
    """Compute iOS-style features for a list of target dates."""
    if 'resting_heart_rate' not in data:
        return pd.DataFrame()

    rhr_df = data['resting_heart_rate'].copy()
    rhr_df['start_date'] = pd.to_datetime(rhr_df['start_date'], utc=True)

    features_list = []
    for target_date in target_dates:
        if not isinstance(target_date, pd.Timestamp):
            target_date = pd.Timestamp(target_date, tz='UTC')
        features = compute_ios_features(rhr_df, target_date)
        if features:
            features_list.append(features)

    return pd.DataFrame(features_list)


def risk_bar(risk: float, width: int = 20) -> str:
    filled = int(risk * width)
    return '█' * filled + '░' * (width - filled)

def get_risk_label(risk: float) -> tuple:
    for threshold, label, action in RISK_LABELS:
        if risk >= threshold:
            return label, action
    return 'LOW', 'Normal baseline'

def arrow(val: float) -> str:
    if pd.isna(val):
        return ' '
    return '↑' if val > 0 else '↓' if val < 0 else '→'

def format_ios_dashboard(
    features: pd.DataFrame,
    ios_model,
    n_windows: int = 5
):
    """Format dashboard using iOS-compatible model and features."""
    recent = features.tail(n_windows).copy()

    if len(recent) == 0:
        return "No data windows available for prediction."

    # iOS model expects [rhr_deviation_14d, rhr_deviation_30d, rhr_delta]
    X = recent[['rhr_deviation_14d', 'rhr_deviation_30d', 'rhr_delta']].values
    X = np.nan_to_num(X, nan=0)

    risk_scores = ios_model.predict_proba(X)[:, 1]

    current = recent.iloc[-1]
    current_risk = risk_scores[-1]

    risk_label, risk_action = get_risk_label(current_risk)

    window_end = current['window_end']
    if isinstance(window_end, pd.Timestamp):
        window_end = window_end.strftime('%Y-%m-%d')
    else:
        window_end = str(window_end)

    lines = []
    lines.append('╭─────────────────────────────────────────────────────────────╮')
    lines.append('│             THYROID STATUS DASHBOARD (iOS Mode)             │')
    lines.append(f'│                     {window_end:^19}                      │')
    lines.append('╰─────────────────────────────────────────────────────────────╯')
    lines.append('')

    lines.append('EARLY WARNING (iOS-compatible)')
    lines.append(f'  Risk Score: {current_risk*100:5.1f}% {risk_bar(current_risk)}')
    lines.append(f'  Status: {risk_label} - {risk_action}')
    lines.append('')

    lines.append('KEY INDICATORS')
    rhr_14d = current.get('rhr_deviation_14d', np.nan)
    rhr_30d = current.get('rhr_deviation_30d', np.nan)
    rhr_delta = current.get('rhr_delta', np.nan)
    rhr_mean = current.get('resting_heart_rate_mean', np.nan)

    if not pd.isna(rhr_14d):
        lines.append(f'  RHR deviation (14d):  {rhr_14d:+5.2f} std  {arrow(rhr_14d)}')
    if not pd.isna(rhr_30d):
        lines.append(f'  RHR deviation (30d):  {rhr_30d:+5.2f} std  {arrow(rhr_30d)}')
    if not pd.isna(rhr_delta):
        lines.append(f'  RHR delta:            {rhr_delta:+5.2f} bpm  {arrow(rhr_delta)}')
    if not pd.isna(rhr_mean):
        lines.append(f'  Current RHR mean:     {rhr_mean:5.1f} bpm')
    lines.append('')

    lines.append(f'TRAJECTORY (last {len(recent)} days)')
    for i in range(len(recent) - 1, -1, -1):
        row = recent.iloc[i]
        start = row['window_start']
        end = row['window_end']

        if isinstance(start, pd.Timestamp):
            start_str = start.strftime('%b %d')
            end_str = end.strftime('%d')
        else:
            start_str = str(start)[:6]
            end_str = str(end)[:2]

        risk = risk_scores[i]
        label, _ = get_risk_label(risk)

        date_range = f'{start_str}-{end_str}'
        lines.append(f'  {date_range:12} {label:8} [{risk*100:4.0f}% risk] {risk_bar(risk)}')

    return '\n'.join(lines)


def format_dashboard(
    features: pd.DataFrame,
    early_model,
    hybrid_model,
    feature_cols: list,
    n_windows: int = 5
):
    recent = features.tail(n_windows).copy()

    if len(recent) == 0:
        return "No data windows available for prediction."

    X = recent[feature_cols].values

    risk_scores = early_model.get_risk_score(X)
    state_probs, _ = hybrid_model.predict_proba(X)
    state_preds, _ = hybrid_model.predict(X)

    current = recent.iloc[-1]
    current_risk = risk_scores[-1]
    current_state = int(state_preds[-1])
    current_conf = state_probs[-1].max() * 100

    risk_label, risk_action = get_risk_label(current_risk)

    today = datetime.now().strftime('%Y-%m-%d')
    window_end = current['window_end']
    if isinstance(window_end, pd.Timestamp):
        window_end = window_end.strftime('%Y-%m-%d')

    lines = []
    lines.append('╭─────────────────────────────────────────────────────────────╮')
    lines.append('│                  THYROID STATUS DASHBOARD                   │')
    lines.append(f'│                     {window_end:^19}                      │')
    lines.append('╰─────────────────────────────────────────────────────────────╯')
    lines.append('')

    lines.append('EARLY WARNING')
    lines.append(f'  Risk Score: {current_risk*100:5.1f}% {risk_bar(current_risk)}')
    lines.append(f'  Status: {risk_label} - {risk_action}')
    lines.append('')

    lines.append('CURRENT STATE')
    lines.append(f'  Severity: {STATE_NAMES[current_state]} ({current_conf:.0f}% confidence)')
    lines.append('')

    lines.append('KEY INDICATORS')
    rhr_14d = current.get('rhr_deviation_14d', np.nan)
    rhr_30d = current.get('rhr_deviation_30d', np.nan)
    rhr_delta = current.get('rhr_delta', np.nan)
    resp_rate = current.get('respiratory_rate_mean', np.nan)

    if not pd.isna(rhr_14d):
        lines.append(f'  RHR deviation (14d):  {rhr_14d:+5.1f} std  {arrow(rhr_14d)}')
    if not pd.isna(rhr_30d):
        lines.append(f'  RHR deviation (30d):  {rhr_30d:+5.1f} std  {arrow(rhr_30d)}')
    if not pd.isna(rhr_delta):
        lines.append(f'  RHR delta:            {rhr_delta:+5.1f} bpm  {arrow(rhr_delta)}')
    if not pd.isna(resp_rate):
        lines.append(f'  Respiratory rate:     {resp_rate:5.1f} /min')
    lines.append('')

    lines.append(f'TRAJECTORY (last {len(recent) * 5} days)')
    for i in range(len(recent) - 1, -1, -1):
        row = recent.iloc[i]
        start = row['window_start']
        end = row['window_end']

        if isinstance(start, pd.Timestamp):
            start_str = start.strftime('%b %d')
            end_str = end.strftime('%d')
        else:
            start_str = str(start)[:6]
            end_str = str(end)[:2]

        state = STATE_NAMES[int(state_preds[i])]
        risk = risk_scores[i]

        date_range = f'{start_str}-{end_str}'
        lines.append(f'  {date_range:12} {state:7} [{risk*100:4.0f}% risk] {risk_bar(risk)}')

    return '\n'.join(lines)

def run_inference(
    input_path: Path,
    since_date: str = None,
    n_windows: int = 5,
    use_cached: bool = True
):
    early_model, hybrid_model, metadata = load_models()
    feature_cols = metadata['feature_columns']

    cached_dir = Path('data/processed')
    if use_cached and cached_dir.exists() and any(cached_dir.glob('*.parquet')):
        print("Loading cached parquet files...")
        data = load_parquet_files(cached_dir)
        print(f"Loaded {sum(len(df) for df in data.values()):,} records from cache")
    else:
        parse_since = None
        if since_date:
            parse_since = pd.Timestamp(since_date, tz='UTC') - timedelta(days=30)
            print(f"Processing data from {parse_since.date()} (30 days before {since_date})")

        print(f"Loading {input_path}...")
        data = parse_export_since(input_path, parse_since)

        if not data:
            print("No relevant health data found in export.")
            return

        print(f"Found {sum(len(df) for df in data.values()):,} records")

    print("Extracting features...")
    features = get_features_for_inference(data, feature_cols)

    if since_date:
        display_since = pd.Timestamp(since_date, tz='UTC')
        features = features[features['window_start'] >= display_since]

    if len(features) == 0:
        print("No complete feature windows available.")
        return

    print(f"Generated {len(features)} windows")
    print()

    dashboard = format_dashboard(
        features,
        early_model,
        hybrid_model,
        feature_cols,
        n_windows=n_windows
    )
    print(dashboard)

def run_ios_inference(
    input_path: Path,
    target_date: str = None,
    n_windows: int = 5,
    use_cached: bool = True
):
    """Run inference using iOS-compatible feature computation and model."""
    ios_model, metadata = load_ios_model()

    cached_dir = Path('data/processed')
    if use_cached and cached_dir.exists() and any(cached_dir.glob('*.parquet')):
        print("Loading cached parquet files...")
        data = load_parquet_files(cached_dir)
        print(f"Loaded {sum(len(df) for df in data.values()):,} records from cache")
    else:
        print(f"Loading {input_path}...")
        data = parse_export_since(input_path, None)

        if not data:
            print("No relevant health data found in export.")
            return

        print(f"Found {sum(len(df) for df in data.values()):,} records")

    if 'resting_heart_rate' not in data:
        print("No resting heart rate data found.")
        return

    # Determine target dates for computing features
    rhr_df = data['resting_heart_rate']
    rhr_df['start_date'] = pd.to_datetime(rhr_df['start_date'], utc=True)

    if target_date:
        end_date = pd.Timestamp(target_date, tz='UTC')
    else:
        end_date = rhr_df['start_date'].max()

    # Generate target dates for the trajectory (going back by 5-day intervals)
    target_dates = [end_date - timedelta(days=i*5) for i in range(n_windows)]
    target_dates = sorted(target_dates)

    print("Computing iOS-compatible features...")
    features = get_ios_features_for_dates(data, target_dates)

    if len(features) == 0:
        print("No complete feature windows available (need 10+ days of RHR data).")
        return

    print(f"Generated {len(features)} windows")
    print()

    dashboard = format_ios_dashboard(features, ios_model, n_windows=n_windows)
    print(dashboard)


def main():
    parser = argparse.ArgumentParser(description='Thyroid state inference from Apple Health export')
    parser.add_argument('--input', type=Path, required=True, help='Path to Apple Health export.xml')
    parser.add_argument('--since', type=str, default=None, help='Only show predictions from this date (YYYY-MM-DD)')
    parser.add_argument('--windows', type=int, default=5, help='Number of windows to show in trajectory')
    parser.add_argument('--fresh', action='store_true', help='Re-parse XML instead of using cached parquet')
    parser.add_argument('--ios', action='store_true', help='Use iOS-compatible feature computation and model')
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.ios:
        run_ios_inference(args.input, args.since, args.windows, use_cached=not args.fresh)
    else:
        run_inference(args.input, args.since, args.windows, use_cached=not args.fresh)


if __name__ == '__main__':
    main()
