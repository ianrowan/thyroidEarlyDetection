import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

def load_parquet_files(data_dir: Path) -> dict:
    data = {}
    for f in data_dir.glob('*.parquet'):
        name = f.stem
        df = pd.read_parquet(f)
        df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
        data[name] = df
    return data

def compute_window_stats(series: pd.Series) -> dict:
    if len(series) == 0 or series.isna().all():
        return {
            'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'min': np.nan, 'max': np.nan, 'p5': np.nan, 'p95': np.nan,
            'iqr': np.nan, 'cv': np.nan, 'count': 0
        }

    mean = series.mean()
    std = series.std()

    return {
        'mean': mean,
        'median': series.median(),
        'std': std,
        'min': series.min(),
        'max': series.max(),
        'p5': series.quantile(0.05),
        'p95': series.quantile(0.95),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'cv': std / mean if mean != 0 else np.nan,
        'count': len(series)
    }

def compute_trend(series: pd.Series, dates: pd.Series) -> float:
    if len(series) < 2:
        return np.nan

    valid = ~(series.isna() | dates.isna())
    if valid.sum() < 2:
        return np.nan

    x = (dates[valid] - dates[valid].min()).dt.total_seconds().values
    y = series[valid].values

    if len(x) < 2:
        return np.nan

    slope, _ = np.polyfit(x, y, 1)
    return slope * 86400

def aggregate_sleep(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    mask = (df['start_date'] >= start) & (df['start_date'] < end)
    window_df = df[mask]

    if len(window_df) == 0:
        return {
            'total_sleep_minutes': np.nan,
            'deep_minutes': np.nan,
            'rem_minutes': np.nan,
            'core_minutes': np.nan,
            'awake_minutes': np.nan,
            'deep_ratio': np.nan,
            'rem_ratio': np.nan,
            'sleep_efficiency': np.nan,
            'sleep_sessions': 0,
        }

    grouped = window_df.groupby('value')['duration_minutes'].sum()

    deep = grouped.get('deep', 0)
    rem = grouped.get('rem', 0)
    core = grouped.get('core', 0)
    awake = grouped.get('awake', 0)
    asleep = grouped.get('asleep', 0)

    total_asleep = deep + rem + core + asleep
    total_in_bed = total_asleep + awake + grouped.get('in_bed', 0)

    dates = window_df['start_date'].dt.date.unique()
    sessions = len(dates)

    return {
        'total_sleep_minutes': total_asleep,
        'deep_minutes': deep,
        'rem_minutes': rem,
        'core_minutes': core,
        'awake_minutes': awake,
        'deep_ratio': deep / total_asleep if total_asleep > 0 else np.nan,
        'rem_ratio': rem / total_asleep if total_asleep > 0 else np.nan,
        'sleep_efficiency': total_asleep / total_in_bed if total_in_bed > 0 else np.nan,
        'sleep_sessions': sessions,
    }

def compute_baseline_deviation(value: float, baseline_values: pd.Series) -> float:
    if baseline_values.isna().all() or pd.isna(value):
        return np.nan
    baseline_mean = baseline_values.mean()
    baseline_std = baseline_values.std()
    if baseline_std == 0:
        return 0
    return (value - baseline_mean) / baseline_std

def extract_features(data: dict, window_days: int = 5) -> pd.DataFrame:
    all_dates = []
    for name, df in data.items():
        all_dates.extend(df['start_date'].tolist())

    if not all_dates:
        return pd.DataFrame()

    min_date = pd.Timestamp(min(all_dates)).normalize()
    max_date = pd.Timestamp(max(all_dates)).normalize()

    windows = []
    current = min_date
    while current < max_date:
        windows.append((current, current + timedelta(days=window_days)))
        current += timedelta(days=window_days)

    features_list = []

    rhr_values = []

    for start, end in windows:
        features = {
            'window_start': start,
            'window_end': end,
        }

        for signal in ['resting_heart_rate', 'heart_rate', 'hrv_sdnn', 'respiratory_rate']:
            if signal not in data:
                continue

            df = data[signal]
            mask = (df['start_date'] >= start) & (df['start_date'] < end)
            window_df = df[mask]

            stats = compute_window_stats(window_df['value'])
            for stat_name, stat_value in stats.items():
                features[f'{signal}_{stat_name}'] = stat_value

            trend = compute_trend(window_df['value'], window_df['start_date'])
            features[f'{signal}_trend'] = trend

        if 'sleep' in data:
            sleep_features = aggregate_sleep(data['sleep'], start, end)
            for name, value in sleep_features.items():
                features[f'sleep_{name}'] = value

        if 'steps' in data:
            df = data['steps']
            mask = (df['start_date'] >= start) & (df['start_date'] < end)
            daily_steps = df[mask].groupby(df[mask]['start_date'].dt.date)['value'].sum()
            features['steps_daily_mean'] = daily_steps.mean()
            features['steps_daily_std'] = daily_steps.std()

        sources = set()
        for df in data.values():
            mask = (df['start_date'] >= start) & (df['start_date'] < end)
            sources.update(df[mask]['source'].unique())

        features['source_apple_watch'] = any('Apple Watch' in s or 'Watch' in s for s in sources)
        features['source_whoop'] = any('WHOOP' in s.upper() for s in sources)

        if 'resting_heart_rate' in data and 'resting_heart_rate_mean' in features:
            rhr_values.append((start, features['resting_heart_rate_mean']))

        features_list.append(features)

    df = pd.DataFrame(features_list)

    if 'resting_heart_rate_mean' in df.columns:
        df['rhr_deviation_14d'] = np.nan
        df['rhr_deviation_30d'] = np.nan

        for i, row in df.iterrows():
            current_val = row['resting_heart_rate_mean']
            current_date = row['window_start']

            past_14d = df[
                (df['window_start'] >= current_date - timedelta(days=14)) &
                (df['window_start'] < current_date)
            ]['resting_heart_rate_mean']

            past_30d = df[
                (df['window_start'] >= current_date - timedelta(days=30)) &
                (df['window_start'] < current_date)
            ]['resting_heart_rate_mean']

            df.at[i, 'rhr_deviation_14d'] = compute_baseline_deviation(current_val, past_14d)
            df.at[i, 'rhr_deviation_30d'] = compute_baseline_deviation(current_val, past_30d)

    if 'resting_heart_rate_mean' in df.columns:
        df['rhr_delta'] = df['resting_heart_rate_mean'].diff()

    if 'respiratory_rate_mean' in df.columns:
        df['resp_rate_delta'] = df['respiratory_rate_mean'].diff()

    return df

def main():
    parser = argparse.ArgumentParser(description='Extract features from processed health data')
    parser.add_argument('--input', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, default=Path('data/features.parquet'))
    parser.add_argument('--window-days', type=int, default=5)
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    data = load_parquet_files(args.input)

    print(f"Loaded {len(data)} data types: {list(data.keys())}")

    print(f"Extracting features with {args.window_days}-day windows")
    features = extract_features(data, window_days=args.window_days)

    print(f"Generated {len(features)} windows with {len(features.columns)} features")
    print(f"Date range: {features['window_start'].min()} to {features['window_end'].max()}")

    features.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
