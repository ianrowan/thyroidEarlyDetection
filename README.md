# Early Detection of Hyperthyroid Episodes from Wearable Data

A machine learning system that detects hyperthyroid episodes 3-4 weeks before lab confirmation using Apple Watch and Whoop data. Analyzes resting heart rate deviation patterns to provide early warning, enabling proactive medication adjustment between blood tests.

If you are interested in the accompanied app the beta sign up is [here](https://tally.so/r/QKR9dA) with repo to follow

## Overview

Adjusting thyroid medication between blood tests is challenging because symptoms often lag behind physiological changes. This project demonstrates that wearable sensor data contains detectable signals of hyperthyroid onset weeks before labs confirm it. 

### Key Results

- **3-4 week early warning** before labeled hyperthyroid onset
- **97% recall** on confirmed hyperthyroid episodes (with SMA-4 smoothing)
- **Only 3 features needed**: RHR deviation from 14-day baseline, 30-day baseline, and delta
- Model flags transition windows that later prove to be episode onset

### How It Works

The model outputs a continuous risk score [0, 1] every 5 days. When resting heart rate begins deviating from personal baselines, the risk score rises - often weeks before symptoms or labs would indicate a problem.

```
Aug 03: Risk 0.53  <- Model alerts (labeled "normal")
Aug 08: Risk 0.73  <- Model alerts (labeled "normal")
Aug 13: Risk 0.57  <- Model alerts (labeled "normal")
Aug 30: Risk 0.44  <- Labeled hyper onset confirmed by labs
```

The "false positives" in early August were actually correct early detections.

## Installation

```bash
git clone https://github.com/ianrowan/thyroid-ml.git
cd thyroid-ml
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- Apple Health export (XML format)
- Historical labels for training (date ranges with severity)

## Usage

### Data Preparation

1. Export Apple Health data from iPhone: Health app > Profile > Export All Health Data
2. Extract the export to `data/apple_health_export/`

```bash
# Parse Apple Health export
python src/parse_health_export.py

# Extract 5-day window features
python src/feature_extraction.py

# Generate visualization for labeling
python src/visualize_for_labeling.py
```

### Training

Requires `data/labels.csv` with columns: `start_date`, `end_date`, `state`, `confidence`

```bash
# Train production models
python -m src.save_models

# Or train with experiment tracking
python -m src.train --model xgboost

# View experiment results
mlflow ui
```

### Inference

```bash
# Run inference on Apple Health export
python -m src.infer --input data/apple_health_export/export.xml

# Show predictions from a specific date
python -m src.infer --input export.xml --since 2025-12-01

# Show more history in trajectory
python -m src.infer --input export.xml --windows 10
```

## Architecture

### Data Pipeline

```
Apple Health XML (2.8GB)
    |
    v parse_health_export.py (streaming, ~2min)
Parquet files per signal type
    |
    v feature_extraction.py
5-day window features (63 features)
    |
    v + labels.csv
Training data (temporal split)
    |
    v train.py / save_models.py
Production models
```

### Feature Engineering

For each 5-day window, the system computes:

- **Central tendency**: mean, median
- **Variability**: std, IQR, coefficient of variation
- **Extremes**: min, max, 5th/95th percentiles
- **Trends**: linear regression slope, delta from prior window
- **RHR-specific**: deviation from 14-day and 30-day baselines
- **Sleep**: total time, efficiency, stage breakdown

### Model Architecture

**EarlyDetectionModel** (XGBoost binary classifier)
- Trained on 3 features: `rhr_deviation_14d`, `rhr_deviation_30d`, `rhr_delta`
- Outputs continuous probability [0, 1]
- Threshold 0.35 provides 3-week early warning (tunable for sensitivity)
- SMA-4 smoothing reduces isolated false positives by 24%

### Key Insight

The model detects when resting heart rate begins deviating from personal baselines. This deviation precedes other symptoms and lab changes by weeks, making it the primary signal for early detection.

## Limitations

- **Single subject**: Results are from one individual's data; generalization to others not validated
- **Device transitions**: May need recalibration when switching wearables (Apple Watch to Whoop, etc.)
- **Baseline dependency**: Requires stable "normal" periods to establish personal baselines
- **Not a diagnostic tool**: Intended to prompt earlier lab testing, not replace medical evaluation

## Repository Structure

```
thyroid-ml/
├── data/
│   ├── apple_health_export/    # Raw Apple Health XML (gitignored)
│   ├── processed/              # Parsed parquet files (gitignored)
│   ├── features.parquet        # 5-day window features
│   └── labels.csv              # Episode labels
├── src/
│   ├── parse_health_export.py  # Streaming XML parser
│   ├── feature_extraction.py   # Window feature aggregation
│   ├── dataset.py              # Label loading, temporal splits
│   ├── models.py               # RandomForest, XGBoost, Semi-supervised
│   ├── sequence_models.py      # LSTM/GRU (experimental)
│   ├── train.py                # Training with MLflow tracking
│   ├── save_models.py          # Production model export
│   └── infer.py                # CLI inference
├── models/                     # Saved model artifacts (gitignored)
├── mlruns/                     # MLflow experiments (gitignored)
├── research.md                 # Detailed research documentation
└── requirements.txt
```

## Research Documentation

See [research.md](research.md) for:
- Complete experiment results and hyperparameter tuning
- Signal analysis explaining detection limits
- Smoothing experiment results
- Model architecture rationale

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for research purposes only and is not intended for medical diagnosis or treatment decisions. Always consult with healthcare providers for thyroid management.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{thyroid_ml,
  title = {Early Detection of Hyperthyroid Episodes from Wearable Data},
  year = {2026},
  url = {https://github.com/ianrowan/thyroid-ml}
}
```

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.
