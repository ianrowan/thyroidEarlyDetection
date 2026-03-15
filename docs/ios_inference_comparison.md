# iOS vs Python Inference Comparison

**Date:** January 11, 2026

## Summary

Investigation into why the CLI inference script produced significantly different risk scores (84.8%) compared to the iOS app (~43%) for the same date (January 10, 2026).

## Observed Discrepancy

| Source | Risk Score | `rhr_deviation_14d` | `rhr_deviation_30d` | `rhr_delta` |
|--------|------------|---------------------|---------------------|-------------|
| iOS App | ~43% | ~-1.14 | ~-1.42 | ~-2.6 |
| CLI (original) | 84.8% | -4.14 | -1.69 | -3.4 |
| CLI (`--ios` mode) | 48.5% | -1.14 | -1.42 | -2.6 |

## Root Cause

The discrepancy was caused by **different feature computation methods** between the iOS app and the Python CLI.

### Python (Original) Method

```
1. Divide data into 5-day windows
2. Compute mean RHR for each window
3. For deviation_14d: compare current window mean to prior 2-3 window means
4. Standard deviation computed from 2-3 data points
```

### iOS Method

```
1. Aggregate RHR samples to daily averages
2. Current window = last 5 daily averages
3. For deviation_14d: compare current mean to prior 14 daily values
4. Standard deviation computed from 14 data points
```

### Key Difference: Baseline Data Points

| Method | Baseline 14d Data Points | Resulting Std Dev |
|--------|--------------------------|-------------------|
| Python (windows) | 2-3 window means | ~1.0 |
| iOS (daily) | 14 daily values | ~3.9 |

## Why iOS is More Accurate

### 1. More Stable Baseline Statistics

With only 2-3 data points, the Python baseline standard deviation is **unstable and artificially small**. A tiny std in the denominator causes the z-score to explode:

```
deviation = (current - baseline_mean) / baseline_std
```

**Example for Jan 10, 2026:**
- Python: `(45 - 49.1) / 0.99 = -4.14` ← inflated
- iOS: `(45.4 - 49.9) / 3.90 = -1.14` ← realistic

### 2. Daily Granularity Captures True Variability

RHR naturally fluctuates day-to-day due to:
- Sleep quality
- Stress levels
- Hydration
- Physical activity
- Circadian rhythm

The iOS method captures this real physiological variability in the baseline standard deviation. The Python method averages away this variability within 5-day windows, then compares window-to-window—missing the true spread of normal RHR values.

### 3. Model Training Alignment

The `early_detection_ios.joblib` model was trained on features computed with daily granularity (matching the iOS `FeatureComputer.swift` logic). When you feed it Python-style features with inflated z-scores, you're giving it inputs outside its training distribution—leading to overconfident/incorrect predictions.

### 4. Clinical Interpretability

- A deviation of **-1.14 std** means "RHR is about 1 standard deviation below your recent baseline"—clinically meaningful and interpretable.
- A deviation of **-4.14 std** would imply an extreme outlier (99.99th percentile event)—which doesn't match reality when RHR dropped from ~50 to ~45 bpm.

## Resolution

Added `--ios` flag to `src/infer.py` that:

1. Uses the `early_detection_ios.joblib` model (same XGBoost converted to CoreML)
2. Computes features using iOS-compatible logic:
   - Aggregates RHR to daily averages first
   - Uses 14/30 actual daily values for baseline statistics
   - Matches the `FeatureComputer.swift` implementation

### Usage

```bash
# iOS-compatible mode (matches iOS app)
venv/bin/python -m src.infer --input data/apple_health_export/export.xml --windows 5 --ios

# Original mode (uses full feature set and window-based computation)
venv/bin/python -m src.infer --input data/apple_health_export/export.xml --windows 5
```

## Feature Computation Comparison

### iOS (`FeatureComputer.swift`)

```swift
// Aggregate to daily averages
let dailyAverages = aggregateByDay(samples: sortedSamples)

// Current window = last 5 daily averages
let currentWindow = Array(dailyAverages.suffix(5))
let currentWindowMean = mean(currentWindow)

// Prior window for delta
let priorWindow = Array(dailyAverages.dropLast(5).suffix(5))
let rhr_delta = currentWindowMean - mean(priorWindow)

// Baselines exclude current window
let baseline14d = Array(dailyAverages.dropLast(5).suffix(14))
let baseline30d = Array(dailyAverages.dropLast(5).suffix(30))

// Z-score with epsilon to prevent division by zero
let epsilon = 0.01
let rhr_deviation_14d = (currentWindowMean - baseline14Mean) / max(baseline14Std, epsilon)
let rhr_deviation_30d = (currentWindowMean - baseline30Mean) / max(baseline30Std, epsilon)
```

### Python iOS-Compatible (`src/infer.py`)

```python
def compute_ios_features(rhr_df, target_date, lookback_days=40):
    # Aggregate by day
    samples['date'] = samples['start_date'].dt.date
    daily_avg = samples.groupby('date')['value'].mean().sort_index()
    daily_values = daily_avg.values

    # Current window = last 5 daily averages
    current_window = daily_values[-5:]
    current_window_mean = np.mean(current_window)

    # Prior window for delta
    prior_window = daily_values[-10:-5]
    rhr_delta = current_window_mean - np.mean(prior_window)

    # Baselines exclude current window (last 5 days)
    baseline14d = daily_values[:-5][-14:]
    baseline30d = daily_values[:-5][-30:]

    # Z-score with epsilon
    epsilon = 0.01
    rhr_deviation_14d = (current_window_mean - baseline14_mean) / max(baseline14_std, epsilon)
    rhr_deviation_30d = (current_window_mean - baseline30_mean) / max(baseline30_std, epsilon)
```

## Recommendations

1. **Use `--ios` mode** for inference that should match the iOS app
2. **Consider retraining** the main `early_detection.joblib` model using iOS-style feature computation for consistency
3. **Update `feature_extraction.py`** to use daily granularity for RHR deviation calculations if window-based features are needed for other purposes

## Files Modified

- `src/infer.py` - Added `--ios` flag, `load_ios_model()`, `compute_ios_features()`, `get_ios_features_for_dates()`, `format_ios_dashboard()`, and `run_ios_inference()` functions
