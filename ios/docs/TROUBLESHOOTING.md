# Troubleshooting Guide - ThyroidDetect

## Build Issues

### Error: "HealthKit framework not found"

**Symptoms:**
```
error: cannot find 'HKHealthStore' in scope
```

**Cause:** HealthKit capability not enabled in Xcode

**Fix:**
1. Open project in Xcode
2. Select target "ThyroidDetect"
3. Go to "Signing & Capabilities"
4. Click "+ Capability"
5. Add "HealthKit"

### Error: "Code signing entitlements error"

**Symptoms:**
```
error: Entitlements file not found
```

**Cause:** Entitlements file not linked in Build Settings

**Fix:**
1. Select target → Build Settings
2. Search for "Code Signing Entitlements"
3. Set to: `ThyroidDetect/ThyroidDetect.entitlements`

### Error: "Info.plist copy command conflict"

**Symptoms:**
```
error: has copy command from 'Info.plist' to 'ThyroidDetect.app/Info.plist'
```

**Cause:** Modern Xcode projects don't use standalone Info.plist files

**Fix:**
1. Delete `ThyroidDetect/Info.plist` if it exists
2. Add Info.plist entries directly in target's Info tab
3. Clean build folder (Cmd+Shift+K)
4. Rebuild

### Warning: "Supported platforms for buildables is empty"

**Symptoms:**
```
[MT] IDERunDestination: Supported platforms for the buildables is empty
```

**Cause:** No simulator or device selected in scheme

**Fix:**
1. In Xcode, go to Product → Destination
2. Select a simulator (iPhone 17) or connected device
3. Try building again

## Runtime Issues

### "HealthKit not available"

**Symptoms:**
- App shows error: "HealthKit not available"
- Logs show: `HealthKitError.notAvailable`

**Cause:** Running in simulator (HealthKit only works on real devices)

**Fix:**
- Connect a physical iPhone or Apple Watch
- Build and run on the device
- Grant HealthKit permissions when prompted

**Note:** There is no workaround for simulator testing. HealthKit APIs are stubbed and return no data.

### "Need more data. Currently have X days"

**Symptoms:**
- App shows: "Need more data. Currently have X days, need at least 10."
- Risk assessment not shown

**Cause:** Insufficient RHR data in HealthKit

**Fix (Short-term):**
- Wait for more Apple Watch data to accumulate
- Wear Apple Watch consistently for 10+ days
- Check Health app has RHR readings

**Fix (Testing):**
- Import historical Health data from another device
- Use Apple Health XML export/import
- Use your own Apple Health export placed at `data/apple_health_export/` in the repo root

### Permission Denied / "Cannot access Health data"

**Symptoms:**
- App shows: "Cannot access Health data"
- HealthKit permission prompt never appeared

**Cause:** Permission denied or not requested

**Fix:**
1. Go to Settings → Privacy & Security → Health
2. Scroll to "ThyroidDetect"
3. Enable "Resting Heart Rate" read permission
4. Pull to refresh in app

**Note:** If app isn't listed in Health permissions, uninstall and reinstall.

### "Insufficient data for analysis. Need at least 35 days"

**Symptoms:**
- App has data but shows this error
- Less than 35 days of RHR available

**Cause:** Model requires 30-day baseline + 5-day current window + 5-day prior window (40 days total, but can work with 35 minimum)

**Fix:**
- Actually need 10+ days minimum (error message threshold)
- For best accuracy, need 35-40 days
- Wait for more data or import historical data

### Model Prediction Error

**Symptoms:**
```
error: Failed to load ThyroidEarlyDetection model
Fatal error: Failed to load ThyroidEarlyDetection model
```

**Cause:** CoreML model file missing or corrupted

**Fix:**
1. Check `ThyroidEarlyDetection.mlmodel` exists in project
2. Verify file is added to target (File Inspector → Target Membership)
3. Re-convert model:
   ```bash
   cd ~/thyroid-ml
   python convert_to_coreml.py
   cp ThyroidEarlyDetection.mlmodel ~/earlyThyroidApp/ThyroidDetect/ThyroidDetect/
   ```
4. Clean build folder and rebuild

## UI Issues

### Blank Screen / No Content

**Symptoms:**
- App launches but shows nothing
- No loading indicator, no error

**Cause:** HealthKitManager not initialized or task not running

**Fix:**
- Check ContentView has `.task { await healthManager.refresh() }`
- Verify HealthKitManager is @State property
- Check console for error logs

### Trend Chart Not Showing

**Symptoms:**
- Risk card shows but no trend chart below
- Only seeing risk percentage

**Cause:** Insufficient RHR samples or empty rhrSamples array

**Fix:**
- Check HealthKitManager.rhrSamples is not empty
- Verify HealthKit query returned samples
- Check console logs: "Fetched N RHR samples"

### Loading Indicator Stuck

**Symptoms:**
- Spinner never goes away
- "Analyzing your heart rate data..." shown forever

**Cause:** HealthKit query hanging or error not handled

**Fix:**
1. Check console for error messages
2. Force quit app
3. Relaunch and pull to refresh
4. Check HealthKit permission in Settings

### Colors Wrong / Theme Issues

**Symptoms:**
- Risk card showing wrong colors
- Light theme instead of dark

**Cause:** Color scheme not forced to dark

**Fix:**
- Verify ContentView has `.preferredColorScheme(.dark)`
- Check RiskLevel.color returns correct Color values
- Test with different risk levels

## Data Issues

### Risk Assessment Seems Wrong

**Symptoms:**
- Expected "normal" but seeing "elevated"
- Risk % doesn't match your perception

**Cause:** Model is detecting subtle RHR elevation you haven't noticed yet

**Explanation:**
- Model detects 3-4 weeks **before** obvious symptoms
- Elevated RHR can precede subjective symptoms
- This is by design (early warning system)

**Verification:**
1. Check raw RHR values in Health app
2. Compare current RHR to your 30-day baseline
3. Look for upward trend over past 2-3 weeks
4. Consider scheduling labs if consistently elevated

### RHR Data Gaps / Missing Days

**Symptoms:**
- Trend chart has gaps
- Some days missing from graph

**Cause:** Apple Watch not worn, or no sleep data for that day

**Impact:**
- Daily aggregation skips days with no samples
- Feature computation still works with partial data
- Model interpolates over gaps

**Fix:**
- Wear Apple Watch consistently
- Ensure automatic sleep tracking is enabled
- Check Health app for data completeness

## Performance Issues

### App Slow to Launch

**Symptoms:**
- Long delay before showing content
- Taking > 5 seconds to load

**Cause:** HealthKit query fetching 40 days of data

**Expected:** 2-3 seconds is normal

**If > 5 seconds:**
1. Check device storage (low space = slow Health queries)
2. Check for extremely large RHR sample count (> 10,000)
3. Restart device to clear Health daemon cache

### High Memory Usage

**Symptoms:**
- App using > 100MB memory
- Device feels sluggish

**Expected:** 40-60MB is normal

**If > 100MB:**
- Check for memory leak (unlikely with Swift value types)
- Restart app
- File bug report with memory graph

### Battery Drain

**Symptoms:**
- Significant battery usage from ThyroidDetect

**Cause:** Background refresh not implemented yet (shouldn't drain battery currently)

**Expected:** < 1% battery per day (only on-demand queries)

**If significant drain:**
- Force quit app when not in use
- Check for infinite refresh loop (bug)
- Verify no background activity in Settings → Battery

## Debugging Tips

### Enable Detailed Logging

View OSLog messages in Console.app:

1. Connect device via USB
2. Open Console.app (Mac)
3. Select your device
4. Filter by:
   - Subsystem: `com.thyroid.detect`
   - Category: `HealthKit`

Look for:
```
[HealthKit] HealthKit authorization granted
[HealthKit] Fetched 387 RHR samples
[HealthKit] Risk computed: 0.42
```

### Check HealthKit Data

Verify RHR data exists:

1. Open Health app on iPhone
2. Go to Browse → Heart → Resting Heart Rate
3. Check for consistent daily readings
4. Verify date range covers last 30+ days

### Test Feature Computation

Print feature values in HealthKitManager.swift:

```swift
if let features = features {
    print("rhr_deviation_14d: \(features.rhr_deviation_14d)")
    print("rhr_deviation_30d: \(features.rhr_deviation_30d)")
    print("rhr_delta: \(features.rhr_delta)")
}
```

Expected ranges:
- Deviations: -2.0 to +3.0 (z-scores)
- Delta: -5 to +5 bpm

### Test Model Directly

Create a simple test:

```swift
let model = ThyroidModel()
let testFeatures = ThyroidFeatures(
    rhr_deviation_14d: 0.0,
    rhr_deviation_30d: 0.0,
    rhr_delta: 0.0
)
let probability = try! model.predict(features: testFeatures)
print("Baseline risk: \(probability)")  // Should be low (~0.1-0.2)
```

## Getting Help

### Information to Include

When reporting issues, include:

1. **Device info:**
   - iPhone model
   - iOS version
   - Xcode version

2. **Data info:**
   - Days of RHR data available
   - Sample count
   - Date range

3. **Error details:**
   - Exact error message
   - Console logs
   - Steps to reproduce

4. **Screenshots:**
   - App state showing issue
   - Health app RHR data
   - Console logs

### Common False Alarms

**"Need more data" with 40+ days:**
- Check if data is actually continuous (gaps count against you)
- Verify RHR samples are in HealthKit (not just heart rate)

**Model always returns low risk:**
- Expected if RHR is stable
- Not a bug unless RHR is clearly elevated

**Model always returns high risk:**
- Could indicate actual RHR elevation
- Check Health app for upward trend
- Consider scheduling labs to verify

## Reset & Clean Slate

If all else fails:

1. **Clean build:**
   ```bash
   cd ThyroidDetect
   xcodebuild clean
   rm -rf ~/Library/Developer/Xcode/DerivedData/ThyroidDetect-*
   ```

2. **Reinstall app:**
   - Delete app from device
   - Clean build folder
   - Rebuild and install fresh

3. **Reset HealthKit permissions:**
   - Settings → General → Transfer or Reset iPhone → Reset Location & Privacy
   - ⚠️ This resets ALL app permissions

4. **Re-convert model:**
   ```bash
   cd ~/thyroid-ml
   python convert_to_coreml.py
   cp ThyroidEarlyDetection.mlmodel ~/earlyThyroidApp/ThyroidDetect/ThyroidDetect/
   ```

## Known Bugs (To Be Fixed)

1. **No trend detection** - Trend arrows never show (not implemented)
2. **No consecutive windows** - consecutiveElevatedDays always 0
3. **Simulator shows generic error** - Could be more specific about HealthKit unavailability
4. **No caching** - Re-queries HealthKit on every launch (wasteful)

These are on the roadmap for future releases.
