# Risk Score Smoothing Experiment

**Date**: 2026-01-19
**Objective**: Reduce false positive alerts by applying moving average smoothing to the risk score output

## Background

The iOS app uses an XGBoost model trained on 3 RHR delta features (`rhr_deviation_14d`, `rhr_deviation_30d`, `rhr_delta`) to predict hyperthyroid risk. While the model has excellent recall, it produces occasional false positive alerts on outlier days.

**Hypothesis**: Applying a moving average to the risk scores will filter out isolated false positives while preserving true detection capability.

## False Positive Classification

Analysis revealed two distinct types of false positives:

1. **Pre-onset signals**: FPs within 30 days before labeled hyper onset (these are actually correct early detections, not errors)
2. **True isolated FPs**: FPs far from any hyperthyroid episode (these are the noise we want to filter)

## Full History Analysis

**Dataset**: 270 labeled 5-day windows
**Threshold**: 0.35
**Raw model performance**:
- Total FPs: 65
- Pre-onset signals: 10 (useful early detection)
- True isolated FPs: 55 (noise to filter)
- True Positives: 146
- False Negatives: 2
- Recall: 98.6%

### Isolated FP Clustering

| Run Length | Count | Description |
|------------|-------|-------------|
| 1 window | 13 | Single-day spikes (most filterable) |
| 2 windows | 3 | Brief runs |
| 3 windows | 5 | Short runs |
| 4 windows | 4 | Medium runs |
| 5 windows | 1 | Longer run |

**Total runs**: 26 distinct FP clusters

## Smoothing Methods Tested

### Simple Moving Average (SMA)

Averages the last N risk scores to smooth out single-window spikes.

```python
smoothed[i] = mean(risk[i-N+1:i+1])
```

### Exponential Moving Average (EMA)

Weighted average giving more weight to recent values.

```python
smoothed[i] = alpha * risk[i] + (1-alpha) * smoothed[i-1]
# where alpha = 2/(span+1)
```

### Confirmation Window

Requires N consecutive alerts before triggering (binary filter).

## Results

### SMA Performance

| Window | Isolated FPs | FPs Removed | Reduction | Recall |
|--------|-------------|-------------|-----------|--------|
| Raw | 55 | - | - | 98.6% |
| SMA-2 | 43 | 12 | -22% | 99.3% |
| SMA-3 | 46 | 9 | -16% | 100% |
| **SMA-4** | **42** | **13** | **-24%** | **100%** |
| SMA-5 | 43 | 12 | -22% | 100% |

### EMA Performance

| Span | Isolated FPs | FPs Removed | Reduction | Recall |
|------|-------------|-------------|-----------|--------|
| EMA-2 | 50 | 5 | -9% | 99.3% |
| EMA-3 | 50 | 5 | -9% | 100% |
| EMA-4 | 48 | 7 | -13% | 100% |
| EMA-5 | 49 | 6 | -11% | 100% |

### Confirmation Window Performance

| N Required | FPs | FP Reduction | Recall |
|------------|-----|--------------|--------|
| N=2 | 4 | -1 | 66.7% |
| N=3 | 3 | -2 | 55.6% |

## Key Findings

1. **SMA-4 is optimal**: Removes 24% of isolated false positives while actually *improving* recall from 98.6% to 100%

2. **SMA outperforms EMA**: SMA removes 2x more FPs than EMA at the same window size. This makes sense because EMA weights recent values more heavily, which doesn't help filter spikes as effectively.

3. **Confirmation windows hurt recall**: Requiring 2+ consecutive alerts significantly reduces recall (66.7%) - not recommended.

4. **Smoothing improves recall**: Counter-intuitively, smoothing actually increases recall by elevating the smoothed score during the ramp-up to true hyper episodes.

5. **13 single-window FPs**: These are the most filterable by any smoothing method.

## Recommendation

**Implement SMA-4 smoothing** for the iOS app:

```swift
// Compute 4-window simple moving average of risk scores
func smoothedRisk(history: [Double]) -> Double {
    let window = min(4, history.count)
    let recent = history.suffix(window)
    return recent.reduce(0, +) / Double(window)
}
```

**Expected improvement**:
- 24% reduction in isolated false positives
- Recall maintained at 100%
- No additional model retraining required

## Implementation Notes

1. The smoothing should be applied **after** the model prediction, as a post-processing step
2. Need to maintain a rolling buffer of the last 4 risk scores
3. First 3 predictions will use smaller windows (min_periods=1)
4. Consider caching smoothed values to avoid recomputation

## Files

- Experiment code: `src/experiment_smoothing.py`
- Full analysis results: `data/full_history_fp_analysis.json`
- Risk score time series: `data/smoothing_risk_scores.csv`
