from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

class BaseModel:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.state_model = None
        self.trend_model = None
        self.preprocessor = None

    def _create_preprocessor(self):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    def fit(self, X: np.ndarray, y_state: np.ndarray, y_trend: np.ndarray = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

class RandomForestModel(BaseModel):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def fit(self, X: np.ndarray, y_state: np.ndarray, y_trend: np.ndarray = None):
        self.preprocessor = self._create_preprocessor()
        X_processed = self.preprocessor.fit_transform(X)

        self.state_model = RandomForestClassifier(**self.params)
        self.state_model.fit(X_processed, y_state)

        if y_trend is not None:
            valid_trend = y_trend >= 0
            if valid_trend.sum() > 10:
                self.trend_model = RandomForestClassifier(**self.params)
                self.trend_model.fit(X_processed[valid_trend], y_trend[valid_trend])

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_processed = self.preprocessor.transform(X)

        state_pred = self.state_model.predict(X_processed)
        trend_pred = None
        if self.trend_model:
            trend_pred = self.trend_model.predict(X_processed)

        return state_pred, trend_pred

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_processed = self.preprocessor.transform(X)

        state_proba = self.state_model.predict_proba(X_processed)
        trend_proba = None
        if self.trend_model:
            trend_proba = self.trend_model.predict_proba(X_processed)

        return state_proba, trend_proba

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        importances = self.state_model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

class XGBoostModel(BaseModel):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def fit(self, X: np.ndarray, y_state: np.ndarray, y_trend: np.ndarray = None):
        self.preprocessor = self._create_preprocessor()
        X_processed = self.preprocessor.fit_transform(X)

        self.state_model = xgb.XGBClassifier(**self.params)
        self.state_model.fit(X_processed, y_state)

        if y_trend is not None:
            valid_trend = y_trend >= 0
            n_classes = len(np.unique(y_trend[valid_trend]))
            if valid_trend.sum() > 10 and n_classes > 1:
                self.trend_model = xgb.XGBClassifier(**self.params)
                self.trend_model.fit(X_processed[valid_trend], y_trend[valid_trend])

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_processed = self.preprocessor.transform(X)

        state_pred = self.state_model.predict(X_processed)
        trend_pred = None
        if self.trend_model:
            trend_pred = self.trend_model.predict(X_processed)

        return state_pred, trend_pred

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_processed = self.preprocessor.transform(X)

        state_proba = self.state_model.predict_proba(X_processed)
        trend_proba = None
        if self.trend_model:
            trend_proba = self.trend_model.predict_proba(X_processed)

        return state_proba, trend_proba

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        importances = self.state_model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

class SemiSupervisedWrapper:
    def __init__(self, base_model: BaseModel, confidence_threshold: float = 0.8, max_iterations: int = 5):
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.iteration_stats = []

    def fit(
        self,
        X_labeled: np.ndarray,
        y_state_labeled: np.ndarray,
        y_trend_labeled: np.ndarray,
        X_unlabeled: np.ndarray
    ):
        X_train = X_labeled.copy()
        y_state_train = y_state_labeled.copy()
        y_trend_train = y_trend_labeled.copy()

        remaining_unlabeled = X_unlabeled.copy()

        for iteration in range(self.max_iterations):
            self.base_model.fit(X_train, y_state_train, y_trend_train)

            if len(remaining_unlabeled) == 0:
                break

            state_proba, _ = self.base_model.predict_proba(remaining_unlabeled)
            max_proba = state_proba.max(axis=1)
            confident_mask = max_proba >= self.confidence_threshold

            n_added = confident_mask.sum()

            if n_added == 0:
                break

            confident_X = remaining_unlabeled[confident_mask]
            confident_y_state = state_proba[confident_mask].argmax(axis=1)
            confident_y_trend = np.full(n_added, -1)

            X_train = np.vstack([X_train, confident_X])
            y_state_train = np.concatenate([y_state_train, confident_y_state])
            y_trend_train = np.concatenate([y_trend_train, confident_y_trend])

            remaining_unlabeled = remaining_unlabeled[~confident_mask]

            self.iteration_stats.append({
                'iteration': iteration + 1,
                'samples_added': n_added,
                'remaining_unlabeled': len(remaining_unlabeled),
                'total_training': len(X_train)
            })

        self.base_model.fit(X_train, y_state_train, y_trend_train)

        return self

    def predict(self, X: np.ndarray):
        return self.base_model.predict(X)

    def predict_proba(self, X: np.ndarray):
        return self.base_model.predict_proba(X)

MODEL_REGISTRY = {
    'random_forest': RandomForestModel,
    'xgboost': XGBoostModel,
}

def get_model(name: str, params: Dict[str, Any] = None) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](params)
