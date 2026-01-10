# Thyroid State Prediction from Wearable Data - Research Phase

## Problem Statement

Predict hyperthyroid state and trend from Apple Health wearable data to enable proactive medication adjustment between blood tests.

**Outputs:**
- **State**: Normal / Mild / Moderate / Severe (ordinal)
- **Trend**: Improving / Stable / Worsening

**Prediction Cadence**: Every 5 days

## Data Sources

### Input: Apple Health Export (XML)

Standard Apple Health export containing `<Record>` elements with:
- `type`: HealthKit identifier
- `value`: Measurement value
- `startDate`, `endDate`: Timestamps
- `sourceName`: Device source (Apple Watch pre-July 2025, Whoop post-July 2025)

**Relevant Record Types:**
| Type | Description |
|------|-------------|
| `HKQuantityTypeIdentifierHeartRate` | Raw heart rate |
| `HKQuantityTypeIdentifierRestingHeartRate` | Resting heart rate |
| `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` | HRV (SDNN method) |
| `HKQuantityTypeIdentifierRespiratoryRate` | Respiratory rate |
| `HKCategoryTypeIdentifierSleepAnalysis` | Sleep stages and duration |
| `HKQuantityTypeIdentifierStepCount` | Activity context |

### Labels: Hybrid Sources

1. **Lab-confirmed** (6 total): 5 recent labs over ~9 months, 1 from 2022 post-hyper
2. **Memory-based**: User-identified episodes with date ranges and severity
3. **Interpolated**: Derived between anchor points

## Feature Engineering

### 5-Day Window Aggregation

For each signal, compute per 5-day window:

**Central Tendency**
- Mean, median

**Variability**
- Standard deviation, IQR, coefficient of variation

**Trends**
- Linear regression slope over window
- Delta from prior window

**Extremes**
- Min, max, 5th/95th percentiles

### Signal-Specific Features

**Resting Heart Rate**
- Deviation from 14-day rolling baseline
- Deviation from 30-day rolling baseline

**Respiratory Rate**
- Normalized by activity level
- Trend direction

**Sleep**
- Total sleep time
- Wake episodes count
- Deep/REM sleep ratios
- Sleep efficiency

**HRV**
- Raw aggregations (may contain hidden patterns)
- Relationship to RHR (cross-signal ratio)

### Device Transition Handling

- Normalize signals by source-specific baseline
- Include `source` as categorical feature
- Validate consistency in any overlapping periods

## Labeling Strategy

### Class Definitions

| State | TSH Pattern | Physiological Indicators |
|-------|-------------|-------------------------|
| Normal | 0.5-4.0 mIU/L | Baseline sleep, RHR, resp rate |
| Mild | Suppressed, T3/T4 slightly elevated | Sleep disturbances, resp rate changes |
| Moderate | Suppressed, T3/T4 elevated | RHR trending up, resp rate up, sleep worse |
| Severe | Very suppressed, T3/T4 high | Clear RHR elevation, poor sleep, resp rate high |

### Trend Labels

Derived from state transitions:
- **Improving**: State at window N < state at window N-1
- **Stable**: State unchanged
- **Worsening**: State at window N > state at window N-1

### Labeling Protocol

1. Create structured labeling tool (CSV-based)
2. For each episode: start date, end date, peak severity, confidence (1-3)
3. Allow "uncertain" ranges with lower training weight
4. Cross-validate against medication history where available

## ML Approaches

### Approach A: Classical ML Baseline

- **Models**: Random Forest, XGBoost
- **Input**: 5-day window feature vectors
- **Output**: Separate models for state and trend, or multi-output
- **Pros**: Fast iteration, interpretable feature importance, works with limited data
- **Cons**: No sequential dependency modeling

### Approach B: Sequence Models

- **Models**: LSTM, GRU
- **Input**: Sequences of 5-day windows (4-6 window lookback)
- **Output**: Multi-head for state + trend
- **Pros**: Captures progression patterns, temporal dependencies
- **Cons**: Requires more data, less interpretable

### Approach C: Semi-Supervised Self-Training

- Train initial model on labeled windows
- Predict unlabeled windows, add high-confidence predictions to training
- Iterate until convergence
- **Pros**: Maximizes 3+ years of unlabeled data
- **Cons**: Error propagation risk

### Approach D: Anomaly Detection Hybrid

- Train autoencoder on confirmed "normal" periods
- Use reconstruction error as hyperthyroid indicator
- Combine with supervised classifier
- **Pros**: Minimal labels needed for normal class
- **Cons**: Requires clean normal period identification

### Recommended Progression

1. Approach A (baseline) → establish feature importance
2. Approach C (semi-supervised) → expand effective training set
3. Approach B (sequence) → if data permits temporal modeling

## Evaluation Strategy

### Dual Evaluation Tracks

**Track 1: Forward Temporal**
- Train: Older labeled periods (memory-based + 2022 lab)
- Test: Recent 9-month episode with 5 labs
- Question: Can older patterns predict new episodes?

**Track 2: Leave-Best-Out Reversed**
- Train: All data including recent clear episode
- Test: Older clearly-remembered episode
- Question: Does high-quality recent data improve generalization?

Run both tracks for each ML approach to understand data quality vs model complexity tradeoffs.

### Metrics

**State Classification**
- Balanced accuracy (handles class imbalance)
- Ordinal accuracy (penalize off-by-2 > off-by-1)
- Confusion matrix

**Trend Prediction**
- Accuracy, F1 per class
- False negative rate on "Worsening" (critical - don't miss deterioration)

**Clinical Utility**
- Days of early warning before lab confirmation
- Confidence calibration

### Baselines

- Naive: Most common class
- Lag: Previous window's state
- Single-feature threshold: Best individual predictor

## Literature Review Topics

1. Wearables for thyroid disorder detection
2. HRV/RHR for hormonal state prediction
3. Time-series classification with sparse labels
4. Semi-supervised learning for health applications

## Hardware Constraints

- Development: 16GB MacBook M3 Pro
- Training acceleration: Titan RTX 24GB VRAM (local)
- Cloud: Available if needed for large-scale experiments

## Next Steps (Implementation Phase)

1. Parse Apple Health XML export
2. Build feature extraction pipeline
3. Conduct labeling session with user
4. Implement experiment tracking (MLflow or similar)
5. Train and evaluate Approach A baseline
6. Iterate through approaches B, C, D
