"""
Per-user Isolation Forest anomaly model.

Trains a scikit-learn IsolationForest on user behavioral features and
provides normalized anomaly scores in [0, 1] (higher = more anomalous).

Persists:
  - models/{user_id}_isolation_forest.joblib   — trained model
  - models/{user_id}_if_score_range.npy        — [min, max] of decision_function
  - models/{user_id}_if_meta.json              — metadata (last fit record count)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly.features import UserBehaviorFeaturizer, _MIN_HISTORY
from models.expense import ExpenseRecord
from storage.expense_store import load_expenses

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"


class PerUserIsolationForest:
    """Per-user Isolation Forest trained on behavioral feature vectors."""

    def __init__(self):
        self._featurizer = UserBehaviorFeaturizer()

    # ── Paths ─────────────────────────────────────────────────────────────

    @staticmethod
    def _model_path(user_id: str) -> Path:
        return _MODELS_DIR / f"{user_id}_isolation_forest.joblib"

    @staticmethod
    def _range_path(user_id: str) -> Path:
        return _MODELS_DIR / f"{user_id}_if_score_range.npy"

    @staticmethod
    def _meta_path(user_id: str) -> Path:
        return _MODELS_DIR / f"{user_id}_if_meta.json"

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, user_id: str) -> bool:
        """Train Isolation Forest on all historical expenses for a user.

        Args:
            user_id: The user whose history to train on.

        Returns:
            True if training succeeded, False if insufficient data.
        """
        history = load_expenses(user_id)

        if len(history) < _MIN_HISTORY:
            print(f"  ⚠ IsolationForest: {user_id} has {len(history)} records "
                  f"(need ≥{_MIN_HISTORY}). Skipping.")
            return False

        # Featurize all records
        X = np.array(
            [self._featurizer.featurize(exp, history) for exp in history],
            dtype=np.float32,
        )

        # Train
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        # Compute decision_function range for normalization
        raw_scores = model.decision_function(X)
        score_range = np.array([raw_scores.min(), raw_scores.max()], dtype=np.float64)

        # Persist
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self._model_path(user_id))
        np.save(self._range_path(user_id), score_range)

        meta = {"last_fit_count": len(history)}
        with open(self._meta_path(user_id), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return True

    # ── Inference ─────────────────────────────────────────────────────────

    def score(self, expense: ExpenseRecord, user_id: str) -> Optional[float]:
        """Compute anomaly score for a single expense.

        Args:
            expense: The new expense to evaluate.
            user_id: The user whose model to use.

        Returns:
            Float in [0, 1] (higher = more anomalous), or None if model
            not found or insufficient history.
        """
        model_path = self._model_path(user_id)
        range_path = self._range_path(user_id)

        if not model_path.exists() or not range_path.exists():
            return None

        try:
            model = joblib.load(model_path)
            score_range = np.load(range_path)
        except Exception:
            return None

        history = load_expenses(user_id)
        if len(history) < _MIN_HISTORY:
            return None

        features = self._featurizer.featurize(expense, history)
        raw = model.decision_function(features.reshape(1, -1))[0]

        # Normalize: lower decision_function = more anomalous
        # Map to [0, 1] where 1 = most anomalous
        range_min, range_max = float(score_range[0]), float(score_range[1])
        if range_max - range_min < 1e-8:
            return 0.5

        normalized = 1.0 - (raw - range_min) / (range_max - range_min)
        return float(np.clip(normalized, 0.0, 1.0))

    # ── Lazy re-fit ───────────────────────────────────────────────────────

    def fit_if_needed(self, user_id: str) -> bool:
        """Re-fit only if model doesn't exist or history grew by 10+.

        Args:
            user_id: The user to check.

        Returns:
            True if a (re-)fit occurred, False otherwise.
        """
        meta_path = self._meta_path(user_id)
        model_path = self._model_path(user_id)

        if not model_path.exists():
            return self.fit(user_id)

        history = load_expenses(user_id)

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                last_count = meta.get("last_fit_count", 0)
                if len(history) - last_count >= 10:
                    return self.fit(user_id)
                return False
            except Exception:
                return self.fit(user_id)

        return self.fit(user_id)
