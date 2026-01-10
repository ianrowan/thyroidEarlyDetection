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

class MLPModel(BaseModel):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'hidden_sizes': [128, 64, 32],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 15,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self._device = None
        self._state_net = None
        self._trend_net = None

    def _get_device(self):
        import torch
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _build_network(self, input_size: int, output_size: int):
        import torch.nn as nn
        layers = []
        prev_size = input_size
        for hidden_size in self.params['hidden_sizes']:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.params['dropout']))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y_state: np.ndarray, y_trend: np.ndarray = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self.preprocessor = self._create_preprocessor()
        X_processed = self.preprocessor.fit_transform(X)

        self._device = self._get_device()
        n_state_classes = int(y_state.max()) + 1

        self._state_net = self._build_network(X_processed.shape[1], n_state_classes)
        self._state_net.to(self._device)

        X_tensor = torch.FloatTensor(X_processed).to(self._device)
        y_state_tensor = torch.LongTensor(y_state.astype(int)).to(self._device)

        dataset = TensorDataset(X_tensor, y_state_tensor)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(self._state_net.parameters(), lr=self.params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.params['epochs']):
            self._state_net.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self._state_net(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.params['patience']:
                    break

        if y_trend is not None:
            valid_trend = y_trend >= 0
            n_trend_classes = len(np.unique(y_trend[valid_trend]))
            if valid_trend.sum() > 10 and n_trend_classes > 1:
                self._trend_net = self._build_network(X_processed.shape[1], 3)
                self._trend_net.to(self._device)

                X_trend = torch.FloatTensor(X_processed[valid_trend]).to(self._device)
                y_trend_tensor = torch.LongTensor(y_trend[valid_trend].astype(int)).to(self._device)

                dataset = TensorDataset(X_trend, y_trend_tensor)
                loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
                optimizer = torch.optim.Adam(self._trend_net.parameters(), lr=self.params['learning_rate'])

                for epoch in range(self.params['epochs'] // 2):
                    self._trend_net.train()
                    for X_batch, y_batch in loader:
                        optimizer.zero_grad()
                        logits = self._trend_net(X_batch)
                        loss = criterion(logits, y_batch)
                        loss.backward()
                        optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        import torch
        X_processed = self.preprocessor.transform(X)
        X_tensor = torch.FloatTensor(X_processed).to(self._device)

        self._state_net.eval()
        with torch.no_grad():
            state_pred = self._state_net(X_tensor).argmax(dim=1).cpu().numpy()

        trend_pred = None
        if self._trend_net is not None:
            self._trend_net.eval()
            with torch.no_grad():
                trend_pred = self._trend_net(X_tensor).argmax(dim=1).cpu().numpy()

        return state_pred, trend_pred

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        import torch
        X_processed = self.preprocessor.transform(X)
        X_tensor = torch.FloatTensor(X_processed).to(self._device)

        self._state_net.eval()
        with torch.no_grad():
            state_proba = torch.softmax(self._state_net(X_tensor), dim=1).cpu().numpy()

        trend_proba = None
        if self._trend_net is not None:
            self._trend_net.eval()
            with torch.no_grad():
                trend_proba = torch.softmax(self._trend_net(X_tensor), dim=1).cpu().numpy()

        return state_proba, trend_proba

class HybridDualModel(BaseModel):
    DELTA_FEATURES = ['rhr_deviation_14d', 'rhr_deviation_30d', 'rhr_delta', 'resp_rate_delta']
    VITAL_FEATURES = ['respiratory_rate_mean', 'resting_heart_rate_mean', 'sleep_sleep_efficiency']

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'hyper_threshold': 0.25,
            'severe_threshold': 0.30,
            'xgb_params': {
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
        }
        if params:
            if 'xgb_params' in params:
                default_params['xgb_params'].update(params.pop('xgb_params'))
            default_params.update(params)
        super().__init__(default_params)
        self.delta_model = None
        self.vital_model = None
        self.delta_indices = None
        self.vital_indices = None
        self.feature_names = None

    def set_feature_names(self, feature_names: list):
        self.feature_names = feature_names
        self.delta_indices = [feature_names.index(f) for f in self.DELTA_FEATURES if f in feature_names]
        self.vital_indices = [feature_names.index(f) for f in self.VITAL_FEATURES if f in feature_names]

    def fit(self, X: np.ndarray, y_state: np.ndarray, y_trend: np.ndarray = None, feature_names: list = None):
        if feature_names is not None:
            self.set_feature_names(feature_names)

        if self.delta_indices is None or self.vital_indices is None:
            raise ValueError("Must call set_feature_names() or pass feature_names before fit()")

        X_delta = np.nan_to_num(X[:, self.delta_indices], nan=0)
        X_vital = np.nan_to_num(X[:, self.vital_indices], nan=0)

        self.delta_model = xgb.XGBClassifier(**self.params['xgb_params'])
        self.delta_model.fit(X_delta, y_state)

        self.vital_model = xgb.XGBClassifier(**self.params['xgb_params'])
        self.vital_model.fit(X_vital, y_state)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_delta = np.nan_to_num(X[:, self.delta_indices], nan=0)
        X_vital = np.nan_to_num(X[:, self.vital_indices], nan=0)

        delta_proba = self.delta_model.predict_proba(X_delta)
        vital_proba = self.vital_model.predict_proba(X_vital)

        predictions = np.zeros(len(X))

        for i in range(len(X)):
            severe_prob = vital_proba[i, 2] if vital_proba.shape[1] > 2 else 0
            hyper_prob = delta_proba[i, 1] if delta_proba.shape[1] > 1 else 0

            if severe_prob > self.params['severe_threshold']:
                predictions[i] = 2
            elif hyper_prob > self.params['hyper_threshold']:
                predictions[i] = 1
            else:
                predictions[i] = 0

        return predictions, None

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_delta = np.nan_to_num(X[:, self.delta_indices], nan=0)
        X_vital = np.nan_to_num(X[:, self.vital_indices], nan=0)

        delta_proba = self.delta_model.predict_proba(X_delta)
        vital_proba = self.vital_model.predict_proba(X_vital)

        combined_proba = np.zeros((len(X), 3))

        for i in range(len(X)):
            severe_prob = vital_proba[i, 2] if vital_proba.shape[1] > 2 else 0
            hyper_prob = delta_proba[i, 1] if delta_proba.shape[1] > 1 else 0
            normal_prob = 1 - hyper_prob - severe_prob

            combined_proba[i] = [max(0, normal_prob), hyper_prob, severe_prob]
            combined_proba[i] /= combined_proba[i].sum()

        return combined_proba, None

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        importance = {}

        delta_names = [self.feature_names[i] for i in self.delta_indices]
        for name, imp in zip(delta_names, self.delta_model.feature_importances_):
            importance[f"delta:{name}"] = imp

        vital_names = [self.feature_names[i] for i in self.vital_indices]
        for name, imp in zip(vital_names, self.vital_model.feature_importances_):
            importance[f"vital:{name}"] = imp

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


MODEL_REGISTRY = {
    'random_forest': RandomForestModel,
    'xgboost': XGBoostModel,
    'mlp': MLPModel,
    'hybrid': HybridDualModel,
}

def get_model(name: str, params: Dict[str, Any] = None) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](params)
