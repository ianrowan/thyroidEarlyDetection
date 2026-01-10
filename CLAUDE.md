# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I have hyperthroidism. It is often difficult to adjust dosage of medication etc between blood tests. I've tracked all apple watch/whoop data for years however and can cross reference periods of time where I was different degrees of hyper thyroid or trending towrds it. 

I would like to research and build/train ML models to find small patterns in a variety of data(RHR, HRV, Respitory rate, sleep etc) that indicate beginning current or end of hyperthyroid periods.

1. Initial Research Phase: 
- create a research plan
- understand data structure of available input data(assume apple health output data with >3 years of all data tracked by apple watch/whoop)
- There will be some subjective/ambiguous output data that we should make a decision on how to use/label training sets. I have both thyroid labs(T3, TSH, T4) from recent times and can label period from memory in the past
- All ML/statictical methods are on the table so we will need to research this
- Research of existing methods(if any) similar to this exist
- Results of this phase should include:
  - Understanding of data structure
  - A plan on how to create data set(including labeling efficiently)
  - proposed approachs for ML(can include multiple that we will benchmark in phase 2)
  - Fill in (Requests of user) section with request data/action
  - Research output to research.md
2. Implementation Phase
- should include data labelling activities 
- implementing proposed ML models/data pipelines from Phase 1
- Should implement robust experiment tracking infrastructure 
- implementation should be relatively portable across machines/environements
- use python and most effective packages for data/ML 
- Output:
  - data should be labeled/pipelines created to automate
  - ML model training should run effectively 
  - ready for multi-approach experimentation 
3. Experimentation Phase
- Run all ML approaches and record benchmarking data
- Include hyperparam tuning for all models
- run this process iteratively 
- Analyze results and create an md file writing up summary of results 
- Add to Request of User if new approaches need to be signed off on
- Output:
  - Successful runs of all Model/params
  - benchmark data in some file
  - ability to make decicision on the correct approach 
4. Productionization/infernce Phase
- Use chose approach from prior steps to build an infernece pipeline that can be integrated to some app(seperate project) to create the output predictions 
- Output:
  - local command line version of inference pipeline
  - approach or packages to use in other project 

Approach and ideas
These are just ideas and dont need to be incorporated if not effective
- Use timeseries approaches to use more then just the current data to predict states(ie trailing propobailities)
- Use trends in output to weight towards future output
- Above I'm basically saying to be baysian 
- Output current state and forcases

Hardware Access
- Local dev is on a 16GB macbook m3 pro
- Access locally to a titan RTX with 24Gb vram
- Open to using cloud services if necessary but only for vram or necessary speed up

## Always Rules 

- Respond concisely: only relevant info, code, plans
- Think relatively deeply: contrary to concise response think alot
- Do not comment code
- Flag strange patterns in data and add to request of user
- Programming language: python

## Workflow preference

- Plan -> confirm -> code -> test -> iterate
- commit changes prior to new tasks and refactors
- use version control smartly 

## Development Commands

```bash
# Parse Apple Health export (run once after new export)
venv/bin/python src/parse_health_export.py

# Extract features into 5-day windows
venv/bin/python src/feature_extraction.py

# Generate labeling visualization
venv/bin/python src/visualize_for_labeling.py

# Train models (requires data/labels.csv)
venv/bin/python -m src.train --model random_forest
venv/bin/python -m src.train --model xgboost
venv/bin/python -m src.train --model all  # run all baselines

# With semi-supervised learning
venv/bin/python -m src.train --model xgboost --semi-supervised

# Sequence models (LSTM/GRU)
venv/bin/python -m src.train_sequence --model lstm --seq-length 6
venv/bin/python -m src.train_sequence --model gru --seq-length 6

# View experiment results
venv/bin/mlflow ui
```

## Architecture

### Directory Structure
```
thyroid-ml/
├── data/
│   ├── apple_health_export/    # Raw Apple Health XML (2.8GB, gitignored)
│   │   └── export.xml
│   ├── processed/              # Parsed parquet files (gitignored)
│   │   ├── resting_heart_rate.parquet
│   │   ├── heart_rate.parquet
│   │   ├── hrv_sdnn.parquet
│   │   ├── respiratory_rate.parquet
│   │   ├── sleep.parquet
│   │   └── steps.parquet
│   ├── features.parquet        # 5-day window features (691 windows, 63 features)
│   ├── labels.csv              # User-provided episode labels (TODO)
│   ├── labels_template.csv     # Template for labeling format
│   ├── results.csv             # Lab results (TSH, T3, T4)
│   └── medication_history.csv  # Medication dosage changes
├── src/
│   ├── parse_health_export.py  # Streaming XML parser for Apple Health
│   ├── feature_extraction.py   # 5-day window feature aggregation
│   ├── visualize_for_labeling.py # Interactive Plotly visualization
│   ├── config.py               # Dataclass configs for experiments
│   ├── dataset.py              # Label loading, temporal splits, sequences
│   ├── models.py               # RandomForest, XGBoost, SemiSupervised
│   ├── sequence_models.py      # LSTM/GRU PyTorch models
│   ├── train.py                # Classical ML training with MLflow
│   └── train_sequence.py       # Sequence model training
├── outputs/
│   └── labeling_viz.html       # Interactive chart for labeling
├── mlruns/                     # MLflow experiment tracking (gitignored)
├── models/                     # Saved model artifacts (gitignored)
├── research.md                 # Research phase documentation
├── requirements.txt            # Python dependencies
└── venv/                       # Virtual environment (gitignored)
```

### Data Pipeline Flow
```
Apple Health XML (2.8GB)
    │
    ▼ parse_health_export.py (streaming, ~2min)
Parquet files per signal type
    │
    ▼ feature_extraction.py
5-day window features (features.parquet)
    │
    ├──▶ visualize_for_labeling.py → labeling_viz.html
    │
    ▼ + labels.csv
Training data (temporal split)
    │
    ├──▶ train.py (RF, XGBoost, Semi-supervised)
    └──▶ train_sequence.py (LSTM, GRU)
           │
           ▼
      MLflow metrics + artifacts
```

### Key Data Characteristics
- **Date range**: 2016-2026 (~9.5 years)
- **Primary signals**: RHR (2881), respiratory rate (62604), sleep (59741), HRV (19235)
- **Device transition**: Apple Watch → Whoop in July 2025
- **HRV gap**: Whoop not writing HRV to Apple Health (ends Nov 2025)
- **Lab results**: 5 from recent episode (Aug 2025 - Jan 2026), 1 from Oct 2022

### Feature Engineering (63 features per 5-day window)
- **Per signal**: mean, median, std, min, max, p5, p95, iqr, cv, count, trend
- **RHR specific**: deviation from 14-day baseline, deviation from 30-day baseline, delta
- **Sleep specific**: total minutes, deep/rem/core/awake, efficiency, session count
- **Derived**: respiratory rate delta, source flags (Apple Watch vs Whoop)

### Model Approaches
| Approach | Module | Description |
|----------|--------|-------------|
| A: Baseline | models.py | RandomForest, XGBoost on window features |
| B: Sequence | sequence_models.py | LSTM/GRU on sequences of windows |
| C: Semi-supervised | models.py | Self-training with unlabeled data |
| D: Anomaly | (planned) | Autoencoder on normal periods |

### Prediction Targets
- **State** (ordinal): normal=0, mild=1, moderate=2, severe=3
- **Trend**: improving=0, stable=1, worsening=2
- **Cadence**: Every 5 days

### Evaluation Strategy
- **Track 1**: Train on old data, test on recent 9-month episode
- **Track 2**: Train on all including recent, test on older episode
- **Metrics**: balanced accuracy, ordinal accuracy, F1, worsening recall

### Key Module APIs

**dataset.py**:
- `prepare_dataset(data_config, train_config)` → DataFrame with labels, feature_cols
- `temporal_train_test_split(df, cols, test_date, for_sequences=False)` → dict with X/y splits

**models.py**:
- `get_model(name, params)` → BaseModel (RandomForest or XGBoost)
- `model.fit(X, y_state, y_trend)` / `model.predict(X)` / `model.predict_proba(X)`
- `SemiSupervisedWrapper(base_model, threshold, max_iter)` → wraps any BaseModel

**sequence_models.py**:
- `SequenceModelWrapper(model_type='lstm'|'gru', hidden_size, ...)`
- Same fit/predict interface, handles preprocessing internally

### Current Phase Status
- [x] Phase 1: Research - Complete (research.md)
- [x] Phase 2: Implementation - Data pipeline and ML infrastructure complete
- [x] Phase 3: Experimentation - XGBoost best (71% ordinal accuracy)
- [ ] Phase 4: Productionization - Build inference pipeline

### Best Model: XGBoost
- Ordinal accuracy: 71% (predictions within 1 severity level)
- Top features: respiratory_rate_mean, respiratory_rate_median, respiratory_rate_p5
- View all results: `venv/bin/mlflow ui` or see research.md

## Requests of user

*Use this section to request dataset, etc from the user, use it like a checklist and checkoff once you receive data*

- [x] Apple Health export (full XML from iPhone Health app → Profile → Export All Health Data)
- [x] Labeling session: date ranges + severity (Normal/Mild/Moderate/Severe) for remembered hyperthyroid episodes
- [x] Lab results with dates (TSH, T3, T4 values for the 6 known lab draws)
- [x] Medication history with dates (optional, for validation/context)

### Data Flags
- HRV data ends Nov 2025 - Whoop may not be writing HRV to Apple Health (verify or export from Whoop directly)
