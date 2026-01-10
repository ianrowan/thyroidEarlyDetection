from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class ThyroidSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y_state: np.ndarray, y_trend: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y_state = torch.LongTensor(y_state.astype(int))
        self.y_trend = torch.LongTensor(y_trend.astype(int)) if y_trend is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y_trend is not None:
            return self.X[idx], self.y_state[idx], self.y_trend[idx]
        return self.X[idx], self.y_state[idx]

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_state_classes: int = 4,
        num_trend_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.state_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_state_classes)
        )

        self.trend_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_trend_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        state_logits = self.state_head(last_output)
        trend_logits = self.trend_head(last_output)

        return state_logits, trend_logits

class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_state_classes: int = 4,
        num_trend_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.state_head = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_state_classes)
        )

        self.trend_head = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_trend_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]

        state_logits = self.state_head(last_output)
        trend_logits = self.trend_head(last_output)

        return state_logits, trend_logits

class SequenceModelWrapper:
    def __init__(
        self,
        model_type: str = 'lstm',
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str = None
    ):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = None
        self.scaler = None
        self.imputer = None
        self.training_history = []

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])

        if fit:
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            X_flat = self.imputer.fit_transform(X_flat)
            X_flat = self.scaler.fit_transform(X_flat)
        else:
            X_flat = self.imputer.transform(X_flat)
            X_flat = self.scaler.transform(X_flat)

        return X_flat.reshape(original_shape)

    def fit(
        self,
        X: np.ndarray,
        y_state: np.ndarray,
        y_trend: np.ndarray = None,
        X_val: np.ndarray = None,
        y_state_val: np.ndarray = None,
        y_trend_val: np.ndarray = None
    ):
        X = self._preprocess(X, fit=True)
        input_size = X.shape[-1]

        if self.model_type == 'lstm':
            self.model = LSTMClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        else:
            self.model = GRUClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )

        self.model.to(self.device)

        valid_trend = y_trend is not None and (y_trend >= 0).any()
        dataset = ThyroidSequenceDataset(X, y_state, y_trend if valid_trend else None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        state_criterion = nn.CrossEntropyLoss()
        trend_criterion = nn.CrossEntropyLoss(ignore_index=-1) if valid_trend else None

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in loader:
                if valid_trend:
                    X_batch, y_state_batch, y_trend_batch = batch
                    y_trend_batch = y_trend_batch.to(self.device)
                else:
                    X_batch, y_state_batch = batch
                    y_trend_batch = None

                X_batch = X_batch.to(self.device)
                y_state_batch = y_state_batch.to(self.device)

                optimizer.zero_grad()

                state_logits, trend_logits = self.model(X_batch)

                loss = state_criterion(state_logits, y_state_batch)
                if valid_trend and y_trend_batch is not None:
                    trend_loss = trend_criterion(trend_logits, y_trend_batch)
                    loss = loss + 0.5 * trend_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_history.append({'epoch': epoch, 'loss': avg_loss})

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._preprocess(X, fit=False)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            state_logits, trend_logits = self.model(X_tensor)

            state_pred = state_logits.argmax(dim=1).cpu().numpy()
            trend_pred = trend_logits.argmax(dim=1).cpu().numpy()

        return state_pred, trend_pred

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._preprocess(X, fit=False)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            state_logits, trend_logits = self.model(X_tensor)

            state_proba = torch.softmax(state_logits, dim=1).cpu().numpy()
            trend_proba = torch.softmax(trend_logits, dim=1).cpu().numpy()

        return state_proba, trend_proba
