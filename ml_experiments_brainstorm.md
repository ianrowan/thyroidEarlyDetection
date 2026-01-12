# ML Experiments Brainstorm: Secondary Trends & HRV Noise Filtering

## Executive Summary

This document explores two complementary research directions to improve the thyroid early detection model:

1. **Secondary Trend Analysis**: Interpreting what falling/rising risk trajectories mean for actual risk
2. **HRV-Based Noise Filtering**: Using RMSSD and other HRV metrics to distinguish routine changes from true baseline shifts

Both approaches aim to reduce false positives while maintaining the current 100% hyper detection rate.

---

## Problem Context

### Current Model Performance

| Model | Hyper Detection | Early Warning | Key Limitation |
|-------|-----------------|---------------|----------------|
| EarlyDetectionModel | 100% (18/18) | 3-4 weeks | 5 "false positives" in Aug 2025 pre-onset |
| HybridDualModel | 82% | Less early | 4 false positives |

The "false positives" in the early detection model are actually early warnings—but the user correctly identifies that **not all elevated probabilities mean the same thing**. A probability going 0.3 → 0.6 is different from one going 0.8 → 0.6.

### The Core Questions

1. When risk probability is **falling**, is actual risk lower than the probability suggests?
2. Can HRV metrics (particularly RMSSD) help distinguish **life events/routine disruption** from **actual hyperthyroid onset**?

---

## Experiment 1: Secondary Trend Analysis

### Hypothesis

**H1**: The derivative of risk probability contains additional predictive information. Specifically:
- **Rising probability + high absolute value** → High confidence hyper
- **Falling probability + moderate absolute value** → "Residual noise" from prior window, lower actual risk
- **Stable probability** → Probability reflects true risk

**Intuition**: The 5-day window captures data that may include both hyperthyroid and normal periods. When exiting a hyper state, the window still contains hyper data even though the current trajectory is improving. The trend signal should discount the absolute risk.

### Mathematical Framework

Let:
- `P(t)` = probability of hyper at window t
- `ΔP(t) = P(t) - P(t-1)` = first derivative (trend)
- `Δ²P(t) = ΔP(t) - ΔP(t-1)` = second derivative (acceleration)

**Proposed Risk Adjustment**:

```
AdjustedRisk(t) = P(t) * TrendMultiplier(ΔP(t))

where TrendMultiplier:
  - ΔP > +0.1 (rising quickly):     1.2  (increase confidence)
  - ΔP in [-0.05, +0.1] (stable):   1.0  (no adjustment)
  - ΔP in [-0.15, -0.05] (falling): 0.7  (discount risk)
  - ΔP < -0.15 (falling quickly):   0.5  (significant discount)
```

### Bayesian Interpretation

Prior: `P(hyper|window_features)`
Update: `P(hyper|window_features, trend)` using trajectory as likelihood

**Key insight**: The current model gives `P(hyper|features)` but ignores `P(features|trajectory)`. A falling trajectory suggests the window is "contaminated" with old hyper data that is no longer reflective of current state.

### Experimental Design

**Data Required**:
- Existing probability outputs from EarlyDetectionModel across all test windows
- Ground truth labels with onset/offset dates

**Analysis Steps**:

1. **Compute Probability Trajectories**
   - For each window, compute P(t), ΔP(t), Δ²P(t)
   - Categorize windows: rising, stable, falling, recovering

2. **Examine "False Positive" Patterns**
   - The Aug 2025 pre-onset windows: were they rising or stable?
   - Hypothesis: these show rising trajectory (true early detection)
   - Counter-hypothesis: end-of-episode windows show falling trajectory

3. **Calibrate Trend Multipliers**
   - Use post-episode windows (if any labeled normal after hyper)
   - Find optimal multiplier values to minimize FP while maintaining TP

4. **Evaluate Combined Risk Score**
   - Compare: Threshold on P(t) vs Threshold on AdjustedRisk(t)
   - Metric: Precision/Recall tradeoff at same detection rate

### Expected Outcomes

| Scenario | Prediction |
|----------|------------|
| Pre-onset Aug 2025 | Rising or stable P → no adjustment, correctly flagged |
| Post-episode windows | Falling P → discounted, fewer false "persistent" alarms |
| Life event disruption | Spike then fall → quick recovery indicates not true hyper |

### Implementation Complexity

**Low** - Can be implemented as post-processing on existing model outputs:
1. Store last N window probabilities
2. Compute trend
3. Apply multiplier
4. Threshold on adjusted value

---

## Experiment 2: RMSSD HRV for Noise Filtering

### Background: SDNN vs RMSSD

**Currently Available (Apple Health)**:
- `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` - Standard deviation of NN intervals
- Measures overall HRV including long-term components
- More sensitive to circadian rhythm and long-term trends

**Available from Whoop (not in Apple Health)**:
- **RMSSD** - Root Mean Square of Successive Differences
- Measures beat-to-beat variation
- More sensitive to acute parasympathetic (vagal) activity
- Better indicator of acute stress/recovery

### Hypothesis

**H2a**: RMSSD drops acutely during life events (travel, weddings, alcohol) but recovers within 1-3 days. During hyperthyroid onset, RMSSD remains suppressed.

**H2b**: The RMSSD recovery pattern can distinguish:
- **Routine disruption**: RMSSD drops → recovers in 48-72h → RHR elevated is transient noise
- **Hyperthyroid onset**: RMSSD drops → stays suppressed → RHR elevated is true signal

### Physiological Basis

| Factor | RHR Change | RMSSD Change | Recovery Time |
|--------|------------|--------------|---------------|
| Travel/jet lag | ↑ 3-8 bpm | ↓ 15-30% | 2-4 days |
| Alcohol | ↑ 5-15 bpm | ↓ 20-50% | 24-48h |
| Wedding/event | ↑ 3-5 bpm | ↓ 10-20% | 1-2 days |
| Early hyperthyroid | ↑ 5-10 bpm | ↓ 10-20% | **Does not recover** |
| Moderate hyper | ↑ 10-20 bpm | ↓ 20-40% | **Does not recover** |

The key differentiator is **persistence**, not magnitude.

### Proposed Feature Engineering

**New Features from RMSSD**:

1. **rmssd_recovery_slope**: Linear trend of RMSSD over trailing 3 days
   - Positive = recovering from acute stress
   - Zero/negative = sustained suppression

2. **rmssd_deviation_7d**: RMSSD relative to 7-day baseline
   - Acute drops (< -1.5 std) suggest life event
   - Gradual decline suggests hyperthyroid

3. **rhr_rmssd_divergence**: Ratio of RHR elevation to RMSSD suppression
   - High ratio (RHR up but RMSSD stable) → possible life event
   - Proportional changes → physiological state change

4. **rmssd_recovery_24h**: Change in RMSSD from yesterday to today
   - Rapid recovery signals transient cause

### Noise Filter Logic

```
def should_discount_risk(rmssd_features, current_risk):
    """
    Returns True if elevated risk is likely noise from routine disruption
    """

    # Pattern 1: Acute drop with recovery trajectory
    if rmssd_features['rmssd_deviation_7d'] < -1.5:
        if rmssd_features['rmssd_recovery_slope'] > 0.05:
            return True  # Recovering from acute stressor

    # Pattern 2: RHR elevated but RMSSD stable
    if rmssd_features['rhr_elevated'] and rmssd_features['rmssd_stable']:
        return True  # Likely external factor

    # Pattern 3: Short-duration deviation
    if rmssd_features['days_suppressed'] < 3:
        return True  # Wait for confirmation

    return False  # Trust the model
```

### Data Requirements

**Critical**: Whoop RMSSD data must be exported separately (not in Apple Health).

**Whoop Export Options**:
1. Manual CSV export from Whoop app
2. Whoop API access (requires developer account)
3. Third-party aggregator (e.g., Terra API)

**Data Format Needed**:
```csv
date,rmssd_avg,rmssd_during_sleep,hrv_score
2025-07-15,45.2,52.1,68
2025-07-16,38.7,44.3,55
...
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Existing RHR Delta Model                                    │
│  Output: P(hyper) based on RHR deviations                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  RMSSD Noise Filter (NEW)                                   │
│  ─────────────────────────────────────────────────────────  │
│  Input: P(hyper), RMSSD features                            │
│  Logic: Check for recovery patterns                         │
│  Output: FilteredRisk = P(hyper) if not noise, else reduced │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Secondary Trend Adjustment (Experiment 1)                  │
│  ─────────────────────────────────────────────────────────  │
│  Input: FilteredRisk, historical FilteredRisk               │
│  Output: FinalRisk with trend multiplier                    │
└─────────────────────────────────────────────────────────────┘
```

### Experimental Validation

**Approach 1: Retrospective Life Event Analysis**
1. User provides list of known life events (travel, weddings)
2. Extract RMSSD patterns around those dates
3. Confirm RMSSD shows acute-drop-then-recovery pattern
4. Compare to Aug 2025 hyper onset (no recovery)

**Approach 2: Leave-Event-Out Testing**
1. Train model excluding known life event periods
2. Test: Does model produce false positives during events?
3. Add RMSSD filter: Do false positives decrease?

**Metrics**:
- **Filter Effectiveness**: % of non-hyper elevated windows correctly discounted
- **Safety**: 0% hyper onset windows incorrectly discounted

---

## Experiment 3: Combined Framework

### Unified Risk Pipeline

```python
def compute_final_risk(window_features, historical_probs, rmssd_features):
    # Step 1: Base probability from RHR model
    base_prob = rhr_delta_model.predict_proba(window_features)

    # Step 2: Apply noise filter (if RMSSD available)
    if rmssd_features is not None:
        if is_noise_pattern(rmssd_features):
            filtered_prob = base_prob * 0.5  # Heavy discount
        elif is_recovering_pattern(rmssd_features):
            filtered_prob = base_prob * 0.7  # Moderate discount
        else:
            filtered_prob = base_prob
    else:
        filtered_prob = base_prob

    # Step 3: Apply trend adjustment
    if len(historical_probs) >= 2:
        trend = filtered_prob - historical_probs[-1]
        trend_mult = get_trend_multiplier(trend)
        final_prob = filtered_prob * trend_mult
    else:
        final_prob = filtered_prob

    return final_prob, {
        'base': base_prob,
        'filtered': filtered_prob,
        'final': final_prob,
        'trend': trend if len(historical_probs) >= 2 else None,
        'noise_filtered': rmssd_features is not None and filtered_prob < base_prob
    }
```

### State Machine Interpretation

Rather than a single threshold, interpret risk through a state machine:

```
States:
  NORMAL: Low risk, stable
  WATCH: Elevated risk but uncertain (new elevation or falling)
  ALERT: Elevated risk, confirmed (rising or stable elevated)
  RECOVERY: Was elevated, now falling (likely resolving)

Transitions:
  NORMAL → WATCH: P > 0.25 first time
  WATCH → ALERT: P > 0.35 AND (rising OR 2+ windows elevated)
  WATCH → NORMAL: P < 0.20 for 2+ windows
  ALERT → RECOVERY: P falling for 2+ windows
  RECOVERY → NORMAL: P < 0.25 for 2+ windows
  RECOVERY → ALERT: P rises again
```

This captures the intuition that:
- A single elevated window could be noise
- Rising probability is more concerning than stable elevated
- Falling probability indicates resolution, not current risk

---

## Priority and Feasibility

| Experiment | Data Needed | Implementation | Expected Impact |
|------------|-------------|----------------|-----------------|
| **1: Secondary Trends** | None (existing) | Low | Medium - refines interpretation |
| **2: RMSSD Filter** | Whoop export | Medium | High - addresses life event FPs |
| **3: Combined** | Whoop export | Medium | Highest - synergistic |

### Recommended Sequence

1. **Immediate**: Implement Experiment 1 (trend analysis)
   - No new data required
   - Validates/invalidates hypothesis with existing Aug 2025 data

2. **Short-term**: Obtain Whoop RMSSD export
   - Request user to export from Whoop app
   - Parse and align with existing features

3. **Medium-term**: Implement Experiment 2 + 3
   - Build RMSSD feature pipeline
   - Validate noise filtering hypothesis
   - Integrate into combined framework

---

## Open Questions

1. **Aug 2025 Pre-Onset Windows**: Were these truly "false positives" or early detection?
   - If rising trajectory: early detection (good)
   - If stable trajectory: suggests model sensitivity might be too high

2. **Whoop RMSSD Data Gap**: Does Whoop write any HRV to Apple Health?
   - Current note says HRV ends Nov 2025
   - Need to verify if this is SDNN only or includes any RMSSD proxy

3. **Life Event Correlation**: Which specific events caused RHR spikes historically?
   - User mentioned trips, weddings—need specific date list
   - Will validate RMSSD recovery hypothesis

4. **Threshold Stability**: Do optimal trend multipliers generalize?
   - Only one major episode to train/validate
   - May need Bayesian approach with uncertainty

---

## User Data Requests

To proceed with these experiments:

- [ ] **Whoop RMSSD Export**: CSV export from Whoop app with daily RMSSD values
- [ ] **Life Event Calendar**: List of dates for trips, weddings, major events (2024-2025)
- [ ] **Confirmation**: Which Aug 2025 windows do you believe were true early warning vs noise?
