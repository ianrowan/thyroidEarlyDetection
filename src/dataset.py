from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from datetime import timedelta

from .config import DataConfig, TrainConfig

STATE_MAP = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
TREND_MAP = {'improving': 0, 'stable': 1, 'worsening': 2}

def load_features(config: DataConfig) -> pd.DataFrame:
    df = pd.read_parquet(config.features_path)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df['window_end'] = pd.to_datetime(df['window_end'])
    return df

def load_labels(config: DataConfig) -> Optional[pd.DataFrame]:
    if not config.labels_path.exists():
        return None

    df = pd.read_csv(config.labels_path)
    df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
    df['end_date'] = pd.to_datetime(df['end_date'], utc=True)
    return df

def assign_labels_to_windows(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features['state'] = np.nan
    features['state_confidence'] = 0.0

    for _, label_row in labels.iterrows():
        mask = (
            (features['window_start'] >= label_row['start_date']) &
            (features['window_end'] <= label_row['end_date'])
        )

        state_val = STATE_MAP.get(label_row['severity'].lower(), np.nan)
        confidence = label_row.get('confidence', 1.0)

        update_mask = mask & (features['state_confidence'] < confidence)
        features.loc[update_mask, 'state'] = state_val
        features.loc[update_mask, 'state_confidence'] = confidence

    features['trend'] = np.nan
    valid_states = features['state'].notna()

    for i in range(1, len(features)):
        if valid_states.iloc[i] and valid_states.iloc[i-1]:
            prev_state = features['state'].iloc[i-1]
            curr_state = features['state'].iloc[i]

            if curr_state < prev_state:
                features.loc[features.index[i], 'trend'] = TREND_MAP['improving']
            elif curr_state > prev_state:
                features.loc[features.index[i], 'trend'] = TREND_MAP['worsening']
            else:
                features.loc[features.index[i], 'trend'] = TREND_MAP['stable']

    return features

def get_feature_columns(df: pd.DataFrame, config: TrainConfig) -> List[str]:
    if config.feature_cols:
        return config.feature_cols

    exclude = config.exclude_patterns + ['state', 'state_confidence', 'trend']

    cols = []
    for col in df.columns:
        if any(pattern in col for pattern in exclude):
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            cols.append(col)

    return cols

def prepare_dataset(
    config: DataConfig,
    train_config: TrainConfig
) -> Tuple[pd.DataFrame, List[str]]:
    features = load_features(config)
    labels = load_labels(config)

    if labels is not None:
        features = assign_labels_to_windows(features, labels)
    else:
        features['state'] = np.nan
        features['trend'] = np.nan
        features['state_confidence'] = 0.0

    feature_cols = get_feature_columns(features, train_config)

    return features, feature_cols

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_seq = []
    y_state = []
    y_trend = []
    indices = []

    feature_matrix = df[feature_cols].values

    for i in range(seq_length, len(df)):
        if df['state'].iloc[i] is not np.nan and not pd.isna(df['state'].iloc[i]):
            seq = feature_matrix[i-seq_length:i]

            if not np.isnan(seq).all():
                X_seq.append(seq)
                y_state.append(df['state'].iloc[i])
                y_trend.append(df['trend'].iloc[i] if not pd.isna(df['trend'].iloc[i]) else -1)
                indices.append(i)

    return (
        np.array(X_seq),
        np.array(y_state),
        np.array(y_trend),
        np.array(indices)
    )

def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_start_date: pd.Timestamp,
    for_sequences: bool = False,
    seq_length: int = 6
) -> dict:
    labeled_mask = df['state'].notna()
    labeled_df = df[labeled_mask].copy()

    train_mask = labeled_df['window_start'] < test_start_date
    test_mask = labeled_df['window_start'] >= test_start_date

    train_df = labeled_df[train_mask]
    test_df = labeled_df[test_mask]

    if for_sequences:
        full_train_df = df[df['window_start'] < test_start_date]
        full_test_df = df[df['window_start'] >= test_start_date]

        X_train, y_train_state, y_train_trend, _ = create_sequences(
            full_train_df, feature_cols, seq_length
        )
        X_test, y_test_state, y_test_trend, _ = create_sequences(
            full_test_df, feature_cols, seq_length
        )

        return {
            'X_train': X_train, 'y_train_state': y_train_state, 'y_train_trend': y_train_trend,
            'X_test': X_test, 'y_test_state': y_test_state, 'y_test_trend': y_test_trend,
        }

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train_state = train_df['state'].values
    y_test_state = test_df['state'].values
    y_train_trend = train_df['trend'].values
    y_test_trend = test_df['trend'].values

    return {
        'X_train': X_train, 'y_train_state': y_train_state, 'y_train_trend': y_train_trend,
        'X_test': X_test, 'y_test_state': y_test_state, 'y_test_trend': y_test_trend,
        'train_dates': train_df['window_start'].values,
        'test_dates': test_df['window_start'].values,
    }
