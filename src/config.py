from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class DataConfig:
    features_path: Path = Path('data/features.parquet')
    labels_path: Path = Path('data/labels.csv')
    lab_results_path: Path = Path('data/results.csv')
    window_days: int = 5

    state_classes: List[str] = field(default_factory=lambda: ['normal', 'mild', 'moderate', 'severe'])
    trend_classes: List[str] = field(default_factory=lambda: ['improving', 'stable', 'worsening'])

@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5

    sequence_length: int = 6

    feature_cols: Optional[List[str]] = None

    exclude_patterns: List[str] = field(default_factory=lambda: [
        'window_start', 'window_end', 'source_', 'count'
    ])

@dataclass
class ExperimentConfig:
    name: str = 'thyroid_prediction'
    tracking_uri: str = 'mlruns'

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
