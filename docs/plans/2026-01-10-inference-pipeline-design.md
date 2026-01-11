# Inference Pipeline Design

## Overview

Phase 4 productionization: CLI tool for thyroid state prediction from Apple Health exports.

## Requirements

- Input: Apple Health XML export (full or incremental)
- Output: Human-readable dashboard
- Primary model: EarlyDetectionModel (early warning)
- Secondary model: HybridDualModel (severity classification)
- Model handling: Train once, save to `models/`, load for inference

## Architecture

```
Apple Health XML
       │
       ▼ (reuse parse_health_export logic)
Parquet files in temp/memory
       │
       ▼ (reuse feature_extraction logic)
Feature windows (most recent N windows)
       │
       ▼
Load models from models/*.joblib
       │
       ▼
Run EarlyDetectionModel → risk score (0-100%)
Run HybridDualModel → state (Normal/Hyper/Severe)
       │
       ▼
Format dashboard output
```

## Files

| File | Action | Purpose |
|------|--------|---------|
| `src/save_models.py` | New | Train & save production models |
| `src/infer.py` | New | Main inference CLI |
| `models/*.joblib` | Generated | Saved model artifacts |
| `models/metadata.json` | Generated | Feature cols, thresholds |

## Model Persistence

```
models/
├── early_detection.joblib
├── hybrid_dual.joblib
└── metadata.json
```

`metadata.json` contains:
- Feature column names (order matters)
- Training date
- Model thresholds
- Data date range

## CLI Interface

```bash
# One-time: train and save production models
venv/bin/python -m src.save_models

# Full export analysis
venv/bin/python -m src.infer --input data/apple_health_export/export.xml

# Incremental (only process data after date)
venv/bin/python -m src.infer --input export.xml --since 2025-12-01

# Show more history
venv/bin/python -m src.infer --input export.xml --windows 10
```

## Output Format

```
╭─────────────────────────────────────────────────────────────╮
│                  THYROID STATUS DASHBOARD                   │
│                     2025-12-08                              │
╰─────────────────────────────────────────────────────────────╯

EARLY WARNING
  Risk Score: 73% ██████████████░░░░░░
  Status: ELEVATED - Monitor closely

CURRENT STATE
  Severity: Hyper (67% confidence)

KEY INDICATORS
  RHR deviation (14d):  +1.8 std  ↑
  RHR deviation (30d):  +1.2 std  ↑
  RHR delta:            +3.2 bpm
  Respiratory rate:     16.2 /min

TRAJECTORY (last 25 days)
  Dec 03-07:  Hyper   [73% risk] ████████████████░░░░
  Nov 28-Dec02: Hyper [58% risk] ████████████░░░░░░░░
  Nov 23-27:  Normal  [34% risk] ███████░░░░░░░░░░░░░
  Nov 18-22:  Normal  [21% risk] ████░░░░░░░░░░░░░░░░
  Nov 13-17:  Normal  [15% risk] ███░░░░░░░░░░░░░░░░░
```

## Incremental Processing

`--since` flag behavior:
1. Parse from 30 days before specified date (for baseline calculations)
2. Extract features for all windows
3. Display only windows from specified date onward

No caching in initial implementation - add later if performance is an issue.

## Implementation Steps

1. Create `src/save_models.py` - train and save EarlyDetectionModel + HybridDualModel
2. Create `src/infer.py` - main inference script with:
   - Health export parsing (reuse existing logic)
   - Feature extraction (reuse existing logic)
   - Model loading
   - Dashboard formatting
3. Test end-to-end with existing export data
4. Update CLAUDE.md with new commands
