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
    n_windows: int = 5
):
    early_model, hybrid_model, metadata = load_models()
    feature_cols = metadata['feature_columns']

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

def main():
    parser = argparse.ArgumentParser(description='Thyroid state inference from Apple Health export')
    parser.add_argument('--input', type=Path, required=True, help='Path to Apple Health export.xml')
    parser.add_argument('--since', type=str, default=None, help='Only show predictions from this date (YYYY-MM-DD)')
    parser.add_argument('--windows', type=int, default=5, help='Number of windows to show in trajectory')
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    run_inference(args.input, args.since, args.windows)

if __name__ == '__main__':
    main()
