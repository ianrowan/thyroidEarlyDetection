# Testing Guide - ThyroidDetect iOS App

## Prerequisites

### HealthKit Data Requirements
- **Minimum**: 10 days of RHR data
- **Recommended**: 40+ days of RHR data
- **Source**: Apple Watch or iPhone Health app
- **Data Type**: Resting Heart Rate (HKQuantityTypeIdentifierRestingHeartRate)

## Testing on Real Device

### 1. Connect iPhone
```bash
# List connected devices
xcrun devicectl list devices

# Or use xcodebuildmcp
list_devices
```

### 2. Build and Install
```bash
# Build for device
cd ThyroidDetect
xcodebuild -scheme ThyroidDetect -destination 'platform=iOS,name=YOUR_DEVICE_NAME' build

# Or use xcodebuildmcp
build_device
```

### 3. First Launch Flow

**Expected behavior:**

1. **App Launch**
   - Shows loading spinner: "Analyzing your heart rate data..."
   - Requests HealthKit permission

2. **Grant Permission**
   - Tap "Allow" for Health data access
   - App queries last 40 days of RHR

3. **Scenarios:**

   **A. Sufficient Data (40+ days)**
   - Risk card displays with probability %
   - Color indicator: green/yellow/red
   - Status message: "Looking good" / "Monitor closely" / "Schedule labs"
   - RHR trend chart shows 30-day line graph

   **B. Insufficient Data (< 10 days)**
   - Error message: "Need more data. Currently have X days, need at least 10."
   - Shows empty state with instructions

   **C. No Permission**
   - Error message about HealthKit access
   - "Try Again" button to re-request permission

## Test Cases

### TC1: Normal Risk (< 0.35 probability)
**Setup:** User with stable RHR, no elevation
**Expected:**
- Green circular indicator
- Probability: < 35%
- Message: "Looking good"

### TC2: Elevated Risk (0.35-0.50)
**Setup:** User with mild RHR elevation
**Expected:**
- Yellow circular indicator
- Probability: 35-50%
- Message: "Monitor closely"

### TC3: High Risk (≥ 0.50)
**Setup:** User with significant RHR elevation
**Expected:**
- Red circular indicator
- Probability: ≥ 50%
- Message: "Schedule labs"

### TC4: Pull to Refresh
**Action:** Pull down on main view
**Expected:**
- Refresh animation
- Re-queries HealthKit data
- Updates risk assessment

### TC5: Insufficient Data
**Setup:** Device with < 10 days RHR data
**Expected:**
- Error state shown
- Message: "Need more data. Currently have X days, need at least 10."

## Debugging

### Enable Logging
The app uses OSLog for debugging. View logs in Console.app:

1. Open Console.app
2. Connect device
3. Filter by subsystem: `com.thyroid.detect`
4. Filter by category: `HealthKit`

**Key log messages:**
```
HealthKit authorization granted
Fetched N RHR samples
Risk computed: 0.XX
```

### Common Issues

**Issue: "HealthKit not available"**
- Cause: Running in simulator
- Fix: Use real device

**Issue: "Need more data"**
- Cause: < 10 days of RHR readings
- Fix: Wait for more Apple Watch data or import historical data

**Issue: Permission denied**
- Cause: User denied HealthKit access
- Fix: Settings → Privacy → Health → ThyroidDetect → Enable "Resting Heart Rate"

## Feature Verification

### FeatureComputer Validation

Manually verify feature computation matches spec:

```swift
// Sample test values
let samples = [/* 40 days of RHR data */]
let features = FeatureComputer.computeFeatures(from: samples)

print(features?.rhr_deviation_14d)  // Should be z-score relative to 14d baseline
print(features?.rhr_deviation_30d)  // Should be z-score relative to 30d baseline
print(features?.rhr_delta)          // Should be difference between windows
```

### Model Output Validation

Test model predictions:

```swift
let model = ThyroidModel()
let testFeatures = ThyroidFeatures(
    rhr_deviation_14d: 0.8,
    rhr_deviation_30d: 0.5,
    rhr_delta: 2.0
)
let probability = try model.predict(features: testFeatures)
// Expected: ~0.63 (elevated risk)
```

## Test Data

Historical test data: use your own Apple Health export placed at `data/apple_health_export/` in the repo root.

## Performance Testing

### Expected Response Times
- HealthKit query (40 days): < 2 seconds
- Feature computation: < 100ms
- Model inference: < 50ms
- Total refresh time: < 3 seconds

### Memory Usage
- Baseline: ~40MB
- Peak (during refresh): ~60MB
- CoreML model: ~40KB

## Simulator Limitations

**⚠️ The app will not work in iOS Simulator** because:
- HealthKit is not available in simulator
- No real RHR data to query
- Always shows "HealthKit not available" error

For UI-only testing without real data, you would need to mock HealthKitManager.
