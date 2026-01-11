import coremltools as ct
import joblib
import numpy as np

model = joblib.load('models/early_detection_ios.joblib')

coreml_model = ct.converters.xgboost.convert(
    model,
    feature_names=['rhr_deviation_14d', 'rhr_deviation_30d', 'rhr_delta'],
    mode='classifier',
    force_32bit_float=True
)

coreml_model.author = 'Thyroid Detection System'
coreml_model.license = 'Personal Use'
coreml_model.short_description = 'Early detection of hyperthyroid onset using RHR patterns'
coreml_model.input_description['rhr_deviation_14d'] = 'RHR deviation from 14-day baseline (z-score)'
coreml_model.input_description['rhr_deviation_30d'] = 'RHR deviation from 30-day baseline (z-score)'
coreml_model.input_description['rhr_delta'] = 'Change in RHR from prior 5-day window (bpm)'

coreml_model.save('ThyroidEarlyDetection.mlmodel')
print("✓ Model converted successfully to ThyroidEarlyDetection.mlmodel")
