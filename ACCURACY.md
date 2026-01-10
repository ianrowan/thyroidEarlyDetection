# Model Accuracy Summary

## Recommended Models

### EarlyDetectionModel (Primary - Early Warning)

Best for: Detecting hyper onset 3-4 weeks before symptoms/labs

| Metric | Value |
|--------|-------|
| Features | 3 (rhr_deviation_14d, rhr_deviation_30d, rhr_delta) |
| Training | Binary classification (hyper vs normal) |
| Default Threshold | 0.35 |
| Hyper Detection | 18/18 (100%) |
| Early Detection | 5 windows flagged 3-4 weeks before labeled onset |
| Total Flagged | 20/23 (87%) |

Threshold tuning:
- 0.25: 4 weeks early, more alerts
- 0.35: 3 weeks early (recommended)
- 0.50: 1-2 weeks early, fewer alerts

### HybridDualModel (Secondary - Severity)

Best for: Distinguishing normal/hyper/severe when severity matters

| Metric | Value |
|--------|-------|
| Features | 7 (4 delta + 3 vital) |
| Ordinal Accuracy | 87.0% |
| Hyper Detection | 9/11 (82%) |
| Severe Detection | 5/7 (71%) |
| False Positives | 4 |

## Baseline Comparisons

| Model | Ordinal Acc | Hyper Det | Severe Det | Notes |
|-------|-------------|-----------|------------|-------|
| XGBoost 3-class | 79.7% | 2/11 (18%) | 4/7 (57%) | Previous best |
| Random Forest 3-class | 75.4% | - | - | |
| MLP 3-class | 72.5% | - | - | Insufficient data |
| LSTM | 45.6% | - | - | Insufficient data |
| Semi-supervised | 55.1% | - | - | Degrades performance |

## Test Set Details

- Test period: Aug 2025 - Dec 2025
- Total windows: 23 (5-day each)
- Normal: 5 windows (Aug 3-28, pre-onset)
- Hyper: 11 windows (Aug 30 - Oct 27)
- Severe: 7 windows (Oct 31 - Dec 7)

## Key Findings

1. **RHR deviation features** are the strongest predictors of hyper onset
2. **Early detection works**: Model flags transition 3-4 weeks before labeled onset
3. **Severe is easy**: Absolute vital signs clearly distinguish severe
4. **Neural networks underperform**: Need 1000+ samples, we have 247
