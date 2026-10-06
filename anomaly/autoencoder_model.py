"""
Per-user Autoencoder anomaly model (PyTorch).

Architecture: 9 → 16 → 8 → 16 → 9 (bottleneck autoencoder)
Anomaly signal: normalized reconstruction error in [0, 1].

Persists:
  - models/{user_id}_autoencoder.pt            — model state dict
  - models/{user_id}_autoencoder_meta.json     — max reconstruction error
  - models/{user_id}_ae_scaler.joblib          — StandardScaler
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly.features import UserBehaviorFeaturizer, _MIN_HISTORY
from models.expense import ExpenseRecord
from storage.expense_store import load_expenses

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_INPUT_DIM = 9


# ── PyTorch model ─────────────────────────────────────────────────────────────

class _ExpenseAutoencoder(nn.Module):
    """Bottleneck autoencoder: 9 → 16 → 8 → 16 → 9."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(_INPUT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, _INPUT_DIM),
            # No activation on final output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        z = self.encoder(x)
        return self.decoder(z)


# ── PerUserAutoencoder ────────────────────────────────────────────────────────

class PerUserAutoencoder:
    """Per-user autoencoder trained on behavioral feature vectors."""

    def __init__(self):
        self._featurizer = UserBehaviorFeaturizer()

    # ── Paths ─────────────────────────────────────────────────────────────

    @staticmethod
    def _model_path(user_id: str) -> Path:
        return _MODELS_DIR / f"{user_id}_autoencoder.pt"

    @staticmethod
    def _meta_path(user_id: str) -> Path:
        return _MODELS_DIR / f"{user_id}_autoencoder_meta.json"

    @staticmethod
    def _scaler_path(user_id: str) -> Path:
        return _MODELS_DIR / f"{user_id}_ae_scaler.joblib"

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        user_id: str,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> bool:
        """Train the autoencoder on all historical expenses for a user.

        Args:
            user_id: The user whose history to train on.
            epochs:  Number of training epochs.
            lr:      Learning rate for Adam optimizer.

        Returns:
            True if training succeeded, False if insufficient data.
        """
        history = load_expenses(user_id)

        if len(history) < _MIN_HISTORY:
            print(f"  ⚠ Autoencoder: {user_id} has {len(history)} records "
                  f"(need ≥{_MIN_HISTORY}). Skipping.")
            return False

        # Featurize
        X = np.array(
            [self._featurizer.featurize(exp, history) for exp in history],
            dtype=np.float32,
        )

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)

        # Build and train model
        model = _ExpenseAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            reconstructed = model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}/{epochs} — Loss: {loss.item():.4f}")

        # Compute max reconstruction error on training data
        model.eval()
        with torch.no_grad():
            recon = model(X_tensor)
            per_sample_mse = torch.mean((X_tensor - recon) ** 2, dim=1)
            max_error = float(per_sample_mse.max().item())

        # Ensure max_error > 0 for normalization
        if max_error < 1e-8:
            max_error = 1.0

        # Persist
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self._model_path(user_id))
        joblib.dump(scaler, self._scaler_path(user_id))

        meta = {"max_reconstruction_error": max_error}
        with open(self._meta_path(user_id), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return True

    # ── Inference ─────────────────────────────────────────────────────────

    def reconstruction_error(
        self,
        expense: ExpenseRecord,
        user_id: str,
    ) -> Optional[float]:
        """Compute normalized reconstruction error for a single expense.

        Args:
            expense: The new expense to evaluate.
            user_id: The user whose model to use.

        Returns:
            Float in [0, 1] (higher = more anomalous), or None if model
            not found or insufficient data.
        """
        model_path = self._model_path(user_id)
        meta_path = self._meta_path(user_id)
        scaler_path = self._scaler_path(user_id)

        if not model_path.exists() or not meta_path.exists() or not scaler_path.exists():
            return None

        try:
            # Load model
            model = _ExpenseAutoencoder()
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            # Load scaler and meta
            scaler = joblib.load(scaler_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            max_error = meta["max_reconstruction_error"]
        except Exception:
            return None

        # Featurize
        history = load_expenses(user_id)
        if len(history) < _MIN_HISTORY:
            return None

        features = self._featurizer.featurize(expense, history)
        scaled = scaler.transform(features.reshape(1, -1)).astype(np.float32)

        # Forward pass
        with torch.no_grad():
            x_tensor = torch.FloatTensor(scaled)
            recon = model(x_tensor)
            mse = float(torch.mean((x_tensor - recon) ** 2).item())

        # Normalize
        normalized = mse / max_error if max_error > 0 else 0.0
        return float(np.clip(normalized, 0.0, 1.0))
