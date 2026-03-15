# ThyroidDetect iOS App

An iOS app that provides early warning of hyperthyroid episodes by analyzing Apple Watch resting heart rate data. Runs a lightweight XGBoost model (via CoreML) entirely on-device to detect patterns 3-4 weeks before symptoms become obvious.

## Prerequisites

- Xcode 26+ (or the latest available)
- iPhone with Apple Watch RHR data (10+ days minimum)
- Apple Developer account (free tier works for personal device testing)
- Python environment with the ML pipeline set up (see repo root README)

## Quick Start

### 1. Train Your Model

From the repo root, follow the full ML pipeline to train a model on your own data:

```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Parse your Apple Health export
python src/parse_health_export.py

# Extract features
python src/feature_extraction.py

# Create labels.csv with your hyperthyroid episode history
# (see data/labels_template.csv for format)

# Train and save production models
python -m src.save_models
```

This produces `models/early_detection_ios.joblib` -- the 3-feature XGBoost model optimized for iOS.

### 2. Convert to CoreML

```bash
pip install coremltools
python convert_to_coreml.py
```

This outputs `ThyroidEarlyDetection.mlmodel` in the repo root.

### 3. Load Your Model into the App

Copy the converted model into the iOS app:

```bash
cp ThyroidEarlyDetection.mlmodel ios/ThyroidDetect/ThyroidDetect/ThyroidEarlyDetection.mlmodel
```

The app ships with a default `.mlmodel` trained on the original author's data. **Replace it with your own** for predictions calibrated to your personal RHR baselines.

### 4. Build and Run

1. Open `ios/ThyroidDetect/ThyroidDetect.xcodeproj` in Xcode
2. Select your Apple Developer team in **Signing & Capabilities**
   - The bundle identifier (`com.thyroiddetect.app`) must be changed to something unique to your account
3. Verify HealthKit capability is enabled (should be pre-configured)
4. Select your iPhone as the build target (HealthKit requires a real device)
5. Build and run

On first launch the app will request HealthKit permission to read resting heart rate data.

## How the Model Works

The app uses only 3 features derived from your resting heart rate:

| Feature | Description |
|---------|-------------|
| `rhr_deviation_14d` | Z-score of current 5-day RHR mean vs. 14-day baseline |
| `rhr_deviation_30d` | Z-score of current 5-day RHR mean vs. 30-day baseline |
| `rhr_delta` | Change in mean RHR between current and prior 5-day windows |

When your RHR starts deviating from personal baselines, the risk score rises. The model catches this shift weeks before symptoms or labs would confirm it.

### Risk Levels

| Score | Level | Action |
|-------|-------|--------|
| < 0.35 | Normal (green) | No action needed |
| 0.35 - 0.50 | Elevated (yellow) | Monitor closely |
| >= 0.50 | High (red) | Consider scheduling labs |

The app applies SMA-4 smoothing (4-window moving average) to reduce isolated false positives.

## Retraining with New Data

As you collect more data and refine your labels, retrain and update the app model:

```bash
# From repo root
python -m src.save_models
python convert_to_coreml.py
cp ThyroidEarlyDetection.mlmodel ios/ThyroidDetect/ThyroidDetect/ThyroidEarlyDetection.mlmodel
```

Then rebuild in Xcode. No code changes needed -- the app loads whatever `.mlmodel` is bundled.

## Architecture

```
ContentView (SwiftUI)
    └── HealthKitManager (@Observable)
        ├── FeatureComputer (pure functions)
        ├── ThyroidModel (CoreML wrapper)
        ├── RiskSmoother (SMA-4)
        └── VacationManager (exclude date ranges)
```

- **HealthKitManager**: Queries 40 days of RHR from HealthKit, orchestrates feature computation and prediction
- **FeatureComputer**: Computes the 3 input features from raw RHR samples (matches `INFERENCE_SPEC.md` exactly)
- **ThyroidModel**: Thin CoreML wrapper, takes 3 features and returns probability [0, 1]
- **RiskSmoother**: Rolling 4-window moving average to smooth predictions
- **VacationManager**: Lets you exclude travel/event periods from calculations

All data stays on-device. No network calls, no analytics, no tracking.

## App Features

- Risk score with color-coded circular indicator
- 7-day risk trend chart
- RHR trend line (30-day window with baseline reference)
- Vacation mode to exclude date ranges from calculations
- Pull-to-refresh
- Date picker to check historical risk scores

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

**Common setup issues:**
- **"No such module" for HealthKit**: Ensure HealthKit capability is enabled in Signing & Capabilities
- **Build fails on simulator**: HealthKit requires a physical device
- **"Insufficient data"**: The app needs 10+ days of Apple Watch RHR data
- **Risk scores differ from CLI**: The iOS model uses 3 features vs. the full 51-feature CLI model; small differences are expected

## Further Documentation

- [INFERENCE_SPEC.md](INFERENCE_SPEC.md) -- Model input/output specification
- [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) -- Architecture decisions and trade-offs
- [docs/TESTING.md](docs/TESTING.md) -- Device testing guide
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) -- Common issues and fixes

## License

MIT License -- see LICENSE in the repo root.

## Disclaimer

This software is for research and personal monitoring purposes only. It is not a medical device and should not be used for diagnosis or treatment decisions. Always consult with healthcare providers for thyroid management.
