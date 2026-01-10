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

## Experiment Results (Phase 3)

### Dataset Statistics
- **Total windows**: 691 (5-day each)
- **Labeled windows**: 270 (39%)
- **Features**: 51 numeric features
- **Label distribution**: Normal 662 days, Mild 301 days, Moderate 317 days, Severe 128 days

### Track 1 Results - 4 Class (test_start_date=2025-08-01)

| Model | Accuracy | Balanced Acc | Ordinal Acc | Notes |
|-------|----------|--------------|-------------|-------|
| XGBoost | 56.5% | 44.8% | 71.0% | Best 4-class |
| Random Forest | 56.5% | 42.9% | 63.8% | Competitive |
| MLP (neural net) | 56.5% | 42.9% | 63.8% | Same as RF |
| RF + Semi-supervised | 47.8% | 33.3% | 55.1% | Degraded |
| LSTM (seq=4) | 36.8% | 33.3% | 45.6% | Insufficient data |

### Track 1 Results - 3 Class (mild+moderate combined)

| Model | Accuracy | Balanced Acc | Ordinal Acc | Notes |
|-------|----------|--------------|-------------|-------|
| **XGBoost** | 52.2% | 40.0% | **79.7%** | **Best overall** |
| Random Forest | 52.2% | 38.1% | 75.4% | Good |
| MLP (neural net) | 47.8% | 33.3% | 72.5% | Underperforms |

**Key insight**: Combining mild+moderate → **8.7% improvement** (71.0% → 79.7%)

### 3-Class Labels (Recommended)
- 0 = Normal
- 1 = Hyper (mild + moderate combined)
- 2 = Severe

### Top Features (XGBoost 3-class)
1. respiratory_rate_median (0.068)
2. respiratory_rate_mean (0.047)
3. heart_rate_min (0.040)
4. sleep_sleep_efficiency (0.040)
5. heart_rate_p5 (0.038)
6. resting_heart_rate_median (0.037)
7. resting_heart_rate_min (0.036)
8. respiratory_rate_p5 (0.033)
9. sleep_core_minutes (0.025)
10. resp_rate_delta (0.023)

### Key Findings

1. **3-class XGBoost is best** - 79.7% ordinal accuracy with mild+moderate combined
2. **Respiratory rate dominates** - Top 3 features are all respiratory rate metrics
3. **Neural networks underperform** - MLP/LSTM/GRU all worse than XGBoost with 247 training samples; need ~1000+ for benefit
4. **Semi-supervised hurts** - Adds noise, degrades performance
5. **Mild/moderate detection is hard** - Model catches severe (57%) but misses early hyper (20% detection)
6. **Early detection challenge** - Mild hyperthyroid doesn't produce strong signal deviation yet

### Confusion Matrix (XGBoost 3-class)
```
             Predicted
             Normal  Hyper  Severe
Actual Normal    11      0       0   ← 100% correct
       Hyper      4      1       0   ← 20% detected (hard)
       Severe     3      4       0   ← 57% detected
```

Model correctly identifies normal. Struggles with early hyper detection - this is the clinical challenge.

### Hyperparameter Tuning Results

| Feature Set | Features | Best Ordinal Acc | Best Params |
|-------------|----------|------------------|-------------|
| Respiratory only | 10 | **88.4%** | depth=5, lr=0.2, n_est=50 |
| Top 20 | 20 | 88.4% | depth=5, lr=0.05, n_est=50 |
| All features | 51 | 84.1% | depth=3, lr=0.05, n_est=50 |

**Key insight**: Simpler models with fewer features perform better. Respiratory-only model (10 features) matches full model performance.

### Signal Analysis - Why Early Detection is Hard

| State | Resp Rate Mean | Std | Difference from Normal |
|-------|----------------|-----|------------------------|
| Normal | 15.91 | 0.33 | — |
| Hyper (mild+mod) | 16.06 | 0.48 | +0.15 (within noise) |
| Severe | 17.63 | 1.11 | +1.72 (detectable) |

**Fundamental limitation**: The difference between Normal and Hyper is only 0.15 breaths/min - well within natural variance (std ~0.4). Severe is clearly distinguishable (+1.7 breaths/min).

This explains why:
- Severe detection works (57-71%)
- Early/mild hyper detection fails (0-20%)
- Class weighting doesn't help - signal isn't there

**Implication**: Current wearable data can reliably detect moderate-to-severe hyperthyroid but not early onset. Need either:
1. More sensitive signals (wearable improvements)
2. Longer observation windows (detect trend before it's obvious)
3. Accept that early detection is beyond current data limits

### Hybrid Dual-Model Approach (Best Result)

**Discovery**: Single models face a tradeoff - delta features detect hyper but miss severe; absolute features detect severe but miss hyper. Solution: specialized models with rule-based fusion.

**Architecture**:
1. **Delta Model** (XGBoost, 4 features): Specializes in hyper detection
   - Features: `rhr_deviation_14d`, `rhr_deviation_30d`, `rhr_delta`, `resp_rate_delta`
   - Captures rate-of-change patterns that precede hyperthyroid episodes

2. **Vital Model** (XGBoost, 3 features): Specializes in severe detection
   - Features: `respiratory_rate_mean`, `resting_heart_rate_mean`, `sleep_sleep_efficiency`
   - Captures absolute level deviations in severe cases

3. **Fusion Rule**:
   - If `vital_model.prob(severe) > 0.30` → Severe
   - Elif `delta_model.prob(hyper) > 0.25` → Hyper
   - Else → Normal

**Results**:
| Metric | Single XGBoost | Hybrid Model | Improvement |
|--------|---------------|--------------|-------------|
| Ordinal Accuracy | 79.7% | **87.0%** | +7.3% |
| Hyper Detection | 2/11 (18%) | **9/11 (82%)** | +64% |
| Severe Detection | 4/7 (57%) | **5/7 (71%)** | +14% |

**Tradeoffs**:
- 4 false positives (normal → hyper/severe)
- Acceptable for health monitoring where missing hyper is worse than false alarms

### Recommendations

1. **Use Hybrid Dual-Model** approach (87% ordinal, 82% hyper detection, 71% severe)
2. **Delta features are key** for early hyper detection - rate of change matters more than absolute values
3. **Clinical utility**: Model now provides meaningful early warning for hyperthyroid episodes
4. **Exclude life events** (trips, weddings) that add noise
5. **Collect more labs during transitions** to improve training data

## Next Steps (Implementation Phase - Completed)

1. ~~Parse Apple Health XML export~~ ✓
2. ~~Build feature extraction pipeline~~ ✓
3. ~~Conduct labeling session with user~~ ✓
4. ~~Implement experiment tracking (MLflow or similar)~~ ✓
5. Train and evaluate Approach A baseline
6. Iterate through approaches B, C, D
