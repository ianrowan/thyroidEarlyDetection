# Inference Specification for iOS Integration

## Model Overview

**Type**: XGBoost binary classifier
**Model file**: `models/early_detection_ios.joblib`
**Purpose**: Early detection of hyperthyroid onset (3-4 weeks advance warning)
**Output**: Probability [0, 1] of hyper risk

## Required Input Features (3 total)

| Feature | Description | Computation |
|---------|-------------|-------------|
| `rhr_deviation_14d` | RHR deviation from 14-day baseline | `(current_rhr - mean_rhr_14d) / std_rhr_14d` |
| `rhr_deviation_30d` | RHR deviation from 30-day baseline | `(current_rhr - mean_rhr_30d) / std_rhr_30d` |
| `rhr_delta` | Change from prior 5-day window | `current_rhr_mean - prior_window_rhr_mean` |

**Window size**: 5 days
**RHR source**: `HKQuantityTypeIdentifierRestingHeartRate`

## Feature Computation

```
For each 5-day window:
1. Get all RHR readings in window
2. Compute window mean RHR

3. rhr_deviation_14d:
   - Get RHR readings from prior 14 days (before window start)
   - baseline_mean = mean of those readings
   - baseline_std = std of those readings
   - rhr_deviation_14d = (window_mean - baseline_mean) / baseline_std

4. rhr_deviation_30d:
   - Same as above but 30 days lookback

5. rhr_delta:
   - Get prior 5-day window mean RHR
   - rhr_delta = current_window_mean - prior_window_mean
```

## Model Input/Output

```python
import joblib

model = joblib.load('models/early_detection_ios.joblib')

features = [[rhr_deviation_14d, rhr_deviation_30d, rhr_delta]]

probability = model.predict_proba(features)[0, 1]
```

**Input**: Array shape (1, 3) with features in order: `[rhr_deviation_14d, rhr_deviation_30d, rhr_delta]`
**Output**: Probability (float, 0-1) of hyper risk

## Alert Logic

### Simple (Single Threshold)
```
if probability >= 0.35:
    alert("Elevated hyper risk")
```

### Recommended (Consecutive Windows)
```
if probability >= 0.35 for 2 consecutive windows:
    alert("Consider scheduling labs")
```

### Two-Tier (Optional)
```
if probability >= 0.50:
    alert(level: "red", "Schedule labs soon")
elif probability >= 0.35:
    alert(level: "yellow", "Monitor closely")
```

## Model Files

| File | Input Shape | Description |
|------|-------------|-------------|
| `early_detection_ios.joblib` | (n, 3) | Standalone XGBoost for iOS - direct 3-feature input |
| `early_detection.joblib` | (n, 51) | Full wrapper model used by CLI inference |

**Use `early_detection_ios.joblib` for iOS integration.**

## Model Export

The `early_detection_ios.joblib` model can be exported as:
- **CoreML**: Use `coremltools` to convert from XGBoost
- **ONNX**: Use `onnxmltools`
- **Raw weights**: XGBoost JSON format, implement inference natively

### CoreML Export Example
```python
import coremltools as ct
import joblib

model = joblib.load('models/early_detection_ios.joblib')
coreml_model = ct.converters.xgboost.convert(
    model,
    feature_names=['rhr_deviation_14d', 'rhr_deviation_30d', 'rhr_delta'],
    mode='classifier'
)
coreml_model.save('ThyroidEarlyDetection.mlmodel')
```

## Example Values

| State | rhr_dev_14d | rhr_dev_30d | rhr_delta | Probability |
|-------|-------------|-------------|-----------|-------------|
| Normal | -0.2 | -0.1 | -1.0 | 0.21 |
| Pre-onset | 0.8 | 0.5 | 2.0 | 0.63 |
| Hyper | 1.5 | 1.2 | 3.0 | 0.82 |

## Notes

- Model detects onset 3-4 weeks before symptoms become obvious
- Some false positives during "normal" periods are acceptable
- Consecutive window rule reduces noise significantly
- Does not replace blood tests - provides early warning to schedule them
