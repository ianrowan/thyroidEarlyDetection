# EarlyThyroid iOS App - Architecture & Implementation Plan

## Overview

Personal iOS app for early detection of hyperthyroid onset using Apple Watch resting heart rate data. Provides 3-4 week advance warning by running an XGBoost model (via CoreML) on computed RHR features.

**Target:** iOS 18+, personal use, single-screen dashboard

## Design Decisions

- **Architecture:** Simple single-view (no MVVM ceremony)
- **UI Style:** Dark/glassy iOS 18 aesthetic with glass effects
- **Primary View:** Risk-first with data below
- **Notifications:** Push alerts on risk elevation
- **Refresh:** Pull on app launch + background refresh every ~12 hours

---

## File Structure

```
EarlyThyroid/
├── EarlyThyroidApp.swift          # App entry, background task registration
├── ContentView.swift              # Main dashboard view
├── HealthKitManager.swift         # @Observable class for HK queries
├── FeatureComputer.swift          # Computes the 3 model features
├── ThyroidModel.swift             # CoreML wrapper
├── Models/
│   └── ThyroidClassifier.mlmodel  # Converted XGBoost model
└── Info.plist                     # HealthKit permissions, background modes
```

---

## Component Details

### 1. HealthKitManager.swift

`@Observable` class that:
- Requests authorization for `HKQuantityTypeIdentifierRestingHeartRate`
- Queries RHR samples for past 35 days
- Exposes state to UI: loading, error, or data ready
- Triggers feature computation and model inference

```swift
@Observable
class HealthKitManager {
    var rhrSamples: [RHRSample] = []
    var isLoading = false
    var error: String?
    var riskResult: RiskResult?

    func refresh() async { ... }
}

struct RHRSample {
    let date: Date
    let bpm: Double
}
```

### 2. FeatureComputer.swift

Pure functions matching INFERENCE_SPEC.md:

**Features computed:**
- `rhr_deviation_14d` = (window_mean - baseline_14d_mean) / baseline_14d_std
- `rhr_deviation_30d` = (window_mean - baseline_30d_mean) / baseline_30d_std
- `rhr_delta` = current_window_mean - prior_window_mean

**Window layout:**
```
Day 1                              Day 31    Day 35
|---------- 30d baseline ----------|
              |-- 14d baseline ----|
                        |--prior---|--current--|
                        Day 26     Day 31
```

**Edge cases:**
- Insufficient data (< 10 days): return nil, show "Need more data"
- Zero std_dev: use small epsilon

### 3. ThyroidModel.swift

CoreML wrapper:

```swift
class ThyroidModel {
    private let model: ThyroidClassifier
    func predict(features: ThyroidFeatures) -> Double
}
```

**Pre-build conversion** (Python script, run once):
```python
import coremltools as ct
import joblib

model = joblib.load("thyroid-ml/models/early_detection.joblib")
coreml_model = ct.converters.xgboost.convert(model)
coreml_model.save("ThyroidClassifier.mlmodel")
```

### 4. RiskResult Model

```swift
struct RiskResult {
    let probability: Double
    let level: RiskLevel
    let trend: Trend?
    let consecutiveElevatedDays: Int
}

enum RiskLevel {
    case normal    // < 0.35 (green)
    case elevated  // 0.35-0.50 (yellow)
    case high      // >= 0.50 (red)
}

enum Trend {
    case rising, stable, falling
}
```

### 5. ContentView.swift (UI)

Single scrolling dashboard:

1. **Risk Card** (hero)
   - Large circular indicator with probability %
   - Color: green/yellow/red
   - Label: "Looking good" / "Monitor closely" / "Schedule labs"
   - Glass background with risk-colored glow
   - Trend arrow (↑ → ↓)

2. **Context Bar**
   - "Based on last 35 days of resting heart rate"
   - Last updated timestamp

3. **RHR Trend Chart**
   - Swift Charts line graph, 30 days
   - Highlighted current 5-day window
   - Baseline reference line

4. **History Cards** (optional)
   - Last 3-4 assessments with dates

**States:** Loading (shimmer), No data (onboarding message), Error (retry)

### 6. Background Refresh & Notifications

**EarlyThyroidApp.swift:**
```swift
BGTaskScheduler.shared.register(
    forTaskWithIdentifier: "com.thyroid.refresh",
    using: nil
) { task in
    handleRefresh(task: task as! BGAppRefreshTask)
}
```

**Notification triggers:**
- Yellow (≥0.35): "RHR trend elevated - worth keeping an eye on"
- Red (≥0.50): "RHR pattern suggests scheduling thyroid labs"
- Only on transitions (avoid spam)

**Info.plist:**
```xml
<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>processing</string>
</array>
<key>NSHealthShareUsageDescription</key>
<string>Read resting heart rate to detect early thyroid changes</string>
```

---

## Implementation Order

### Phase 1: Project Setup
1. Create Xcode project (SwiftUI, iOS 18+)
2. Add Info.plist entries for HealthKit
3. Convert XGBoost model to CoreML, add to project

### Phase 2: Core Logic
4. Implement HealthKitManager (auth + query)
5. Implement FeatureComputer (3 features)
6. Implement ThyroidModel (CoreML wrapper)
7. Wire together: HK → Features → Model → RiskResult

### Phase 3: UI
8. Build ContentView with risk card
9. Add RHR trend chart (Swift Charts)
10. Implement loading/error/empty states
11. Style with dark theme + glass effects

### Phase 4: Notifications
12. Add background refresh task
13. Implement local notifications
14. Test notification triggers

### Phase 5: Polish
15. Test on device with real HealthKit data
16. Tune thresholds if needed
17. Add any missing edge case handling

---

## Verification

**Manual testing:**
1. Fresh install → requests HealthKit permission
2. With <10 days data → shows "Need more data"
3. With 35+ days data → shows risk assessment
4. Pull to refresh updates values
5. Background refresh fires notification on threshold cross

**Unit tests (optional for personal use):**
- FeatureComputer: known inputs → expected outputs
- RiskLevel thresholds: probability → correct level

---

## Dependencies

- iOS 18+ (for `.glassEffect()`)
- HealthKit framework
- CoreML framework
- Swift Charts

## Notes

- Never modify .pbxproj manually - use Xcode
- Add Logger() for async flows
- Test on real device (HealthKit doesn't work in simulator)
- Model conversion is a one-time pre-build step
