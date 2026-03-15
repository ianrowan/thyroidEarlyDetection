# iOS App Update: Risk Score Smoothing

## Summary

Add a 4-window simple moving average (SMA-4) to smooth the risk score output. This reduces false positive alerts by 24% while maintaining 100% detection recall.

## Current Behavior

The app currently displays the raw risk probability from the XGBoost model:

```
risk = model.predict_proba(features)[1]  // probability of hyper class
```

This causes occasional isolated false positive alerts on outlier days.

## New Behavior

Apply SMA-4 smoothing to the risk score before displaying or alerting:

```
smoothed_risk = average(last 4 risk scores)
```

## Implementation

### 1. Store Risk History

Maintain a rolling buffer of the last 4 raw risk scores. This can be stored in UserDefaults, Core Data, or in-memory if the app maintains state.

```swift
class RiskSmoother {
    private let windowSize = 4
    private var riskHistory: [Double] = []

    // Call this each time a new prediction is made
    func addRisk(_ rawRisk: Double) -> Double {
        riskHistory.append(rawRisk)

        // Keep only the last `windowSize` values
        if riskHistory.count > windowSize {
            riskHistory.removeFirst()
        }

        return smoothedRisk
    }

    var smoothedRisk: Double {
        guard !riskHistory.isEmpty else { return 0 }
        return riskHistory.reduce(0, +) / Double(riskHistory.count)
    }

    var rawRisk: Double {
        riskHistory.last ?? 0
    }
}
```

### 2. Persistence (Optional)

If risk history should persist across app launches:

```swift
extension RiskSmoother {
    private static let historyKey = "risk_history"

    func save() {
        UserDefaults.standard.set(riskHistory, forKey: Self.historyKey)
    }

    func load() {
        if let history = UserDefaults.standard.array(forKey: Self.historyKey) as? [Double] {
            riskHistory = history
        }
    }
}
```

### 3. Update Display Logic

Replace raw risk with smoothed risk in the UI:

```swift
// Before
let riskPercent = rawRisk * 100
riskLabel.text = String(format: "%.1f%%", riskPercent)

// After
let smoother = RiskSmoother()
smoother.load()  // if persisting
let smoothedRisk = smoother.addRisk(rawRisk)
smoother.save()  // if persisting

let riskPercent = smoothedRisk * 100
riskLabel.text = String(format: "%.1f%%", riskPercent)
```

### 4. Update Alert Threshold Logic

Apply the same smoothed value to alert decisions:

```swift
// Before
if rawRisk >= 0.35 {
    showAlert()
}

// After
if smoothedRisk >= 0.35 {
    showAlert()
}
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| First prediction (no history) | Use raw score (window size = 1) |
| 2nd prediction | Average of 2 scores |
| 3rd prediction | Average of 3 scores |
| 4+ predictions | Full SMA-4 (average of 4 scores) |
| App reinstall | History cleared, starts fresh |
| Gap in data (>30 days) | Consider clearing history |

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| False positive rate | ~20% of normal windows | ~15% of normal windows |
| Detection recall | 98.6% | 100% |
| Alert delay | Immediate | 0-3 windows (0-15 days max) |

## UI Considerations

### Option A: Show Only Smoothed Risk
Display only the smoothed value. Simpler UI, less confusion.

### Option B: Show Both Values
```
Current Risk: 45% (raw)
Trend Risk:   38% (smoothed) ← use this for alerts
```

### Option C: Show Smoothed with Indicator
```
Risk: 38% ↑
      (trending up from 35%)
```

## Testing

1. **Verify smoothing math**: Input [0.5, 0.3, 0.4, 0.6] → smoothed should be 0.45
2. **Verify cold start**: First prediction should equal raw score
3. **Verify persistence**: Kill app, relaunch, confirm history preserved
4. **Verify alert behavior**: Isolated spike (one high value among lows) should not trigger alert

## Test Scenarios

```swift
// Test 1: Isolated spike should be dampened
let smoother = RiskSmoother()
smoother.addRisk(0.20)  // smoothed: 0.20
smoother.addRisk(0.25)  // smoothed: 0.225
smoother.addRisk(0.22)  // smoothed: 0.223
smoother.addRisk(0.70)  // smoothed: 0.343 (spike dampened below 0.35 threshold)
// Raw would have triggered alert at 0.70, smoothed does not

// Test 2: Sustained elevation should still trigger
let smoother2 = RiskSmoother()
smoother2.addRisk(0.35)  // smoothed: 0.35
smoother2.addRisk(0.40)  // smoothed: 0.375
smoother2.addRisk(0.45)  // smoothed: 0.40
smoother2.addRisk(0.50)  // smoothed: 0.425 (alert triggers)
```

## Questions?

The ML experiment code is in `src/experiment_smoothing.py` and detailed analysis is in `docs/smoothing_experiment.md`.
