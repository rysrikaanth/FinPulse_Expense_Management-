"""
Anomaly Ensemble — weighted combination of Isolation Forest + Autoencoder.

Produces a unified AnomalyResult with:
  - Combined anomaly score (0.6 × IF + 0.4 × AE)
  - Signal attribution (top contributing features)
  - Human-readable explanation string

Also provides anomaly logging to data/{user_id}/anomaly_log.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly.features import UserBehaviorFeaturizer, FEATURE_NAMES, _MIN_HISTORY
from anomaly.isolation_forest_model import PerUserIsolationForest
from anomaly.autoencoder_model import PerUserAutoencoder
from models.expense import ExpenseRecord
from storage.expense_store import load_expenses

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

_ANOMALY_THRESHOLD = 0.7

# ── Day names for explanations ────────────────────────────────────────────────
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """Unified anomaly detection output."""

    anomaly_score: float
    is_anomaly: bool
    signals: list[dict] = field(default_factory=list)
    explanation: str = ""
    model_scores: dict = field(default_factory=dict)


# ── AnomalyEnsemble ──────────────────────────────────────────────────────────

class AnomalyEnsemble:
    """Weighted ensemble of Isolation Forest + Autoencoder."""

    def __init__(self):
        self._if_model = PerUserIsolationForest()
        self._ae_model = PerUserAutoencoder()
        self._featurizer = UserBehaviorFeaturizer()

    def score(self, expense: ExpenseRecord, user_id: str) -> AnomalyResult:
        """Score an expense for anomaly using the ensemble.

        Args:
            expense: The new expense to evaluate.
            user_id: The owner user.

        Returns:
            AnomalyResult with score, signals, and explanation.
        """
        history = load_expenses(user_id)

        # ── Cold start ───────────────────────────────────────────────────
        if len(history) < _MIN_HISTORY:
            return AnomalyResult(
                anomaly_score=0.0,
                is_anomaly=False,
                signals=[],
                explanation="Insufficient history for anomaly detection.",
                model_scores={"isolation_forest": None, "autoencoder": None},
            )

        # ── Step 1: Get individual model scores ─────────────────────────
        if_score = self._if_model.score(expense, user_id)
        ae_score = self._ae_model.reconstruction_error(expense, user_id)

        # ── Step 2: Ensemble combination ─────────────────────────────────
        if if_score is not None and ae_score is not None:
            final = 0.6 * if_score + 0.4 * ae_score
        elif if_score is not None:
            final = if_score
        elif ae_score is not None:
            final = ae_score
        else:
            return AnomalyResult(
                anomaly_score=0.0,
                is_anomaly=False,
                signals=[],
                explanation="Anomaly models not yet trained for this user.",
                model_scores={"isolation_forest": None, "autoencoder": None},
            )

        # ── Step 3: Threshold ────────────────────────────────────────────
        is_anomaly = final > _ANOMALY_THRESHOLD

        # ── Step 4: Signal attribution ───────────────────────────────────
        features = self._featurizer.featurize(expense, history)
        signals = self._compute_signals(features, history)

        # ── Step 5: Explanation ──────────────────────────────────────────
        explanation = self._build_explanation(
            features, signals, expense, history
        )

        return AnomalyResult(
            anomaly_score=round(final, 4),
            is_anomaly=is_anomaly,
            signals=signals,
            explanation=explanation,
            model_scores={
                "isolation_forest": round(if_score, 4) if if_score is not None else None,
                "autoencoder": round(ae_score, 4) if ae_score is not None else None,
            },
        )

    # ── Signal attribution ────────────────────────────────────────────────

    def _compute_signals(
        self,
        features: np.ndarray,
        history: list[ExpenseRecord],
    ) -> list[dict]:
        """Identify top-3 features contributing to the anomaly score.

        Measures each feature's deviation from its historical distribution.
        """
        # Build historical feature matrix
        hist_features = np.array(
            [self._featurizer.featurize(exp, history) for exp in history],
            dtype=np.float32,
        )

        if hist_features.shape[0] < 2:
            return []

        means = hist_features.mean(axis=0)
        stds = hist_features.std(axis=0)

        deviations = []
        for i in range(len(FEATURE_NAMES)):
            deviation = abs(float(features[i]) - float(means[i])) / (float(stds[i]) + 1e-6)
            direction = "high" if features[i] > means[i] else "low"
            deviations.append({
                "feature": FEATURE_NAMES[i],
                "value": round(float(features[i]), 4),
                "contribution": round(deviation, 4),
                "direction": direction,
            })

        # Sort by deviation descending, take top 3
        deviations.sort(key=lambda x: x["contribution"], reverse=True)
        return deviations[:3]

    # ── Explanation builder ───────────────────────────────────────────────

    def _build_explanation(
        self,
        features: np.ndarray,
        signals: list[dict],
        expense: ExpenseRecord,
        history: list[ExpenseRecord],
    ) -> str:
        """Build a human-readable explanation from the top signals."""
        if not signals:
            return "Insufficient history for anomaly detection."

        amount = expense.total_amount or 0.0
        vendor = expense.vendor or "Unknown"
        category = expense.category or "unknown"

        # Pre-compute category stats for explanation text
        cat_expenses = [
            e for e in history
            if (e.category or "").strip().lower() == category.strip().lower()
        ]
        cat_amounts = [e.total_amount for e in cat_expenses if e.total_amount is not None]
        cat_mean = float(np.mean(cat_amounts)) if cat_amounts else 0.0

        # Parse date for weekday name
        weekday_name = "this day"
        if expense.date:
            try:
                dt = datetime.fromisoformat(expense.date)
                weekday_name = _WEEKDAY_NAMES[dt.weekday()]
            except ValueError:
                pass

        hour_str = expense.time or "00:00"
        hour_val = hour_str.split(":")[0] if ":" in hour_str else "00"

        parts: list[str] = []
        for sig in signals[:2]:  # Top 2 only for brevity
            feat = sig["feature"]

            if feat == "amount_zscore" and sig["direction"] == "high":
                zscore = features[0]
                parts.append(
                    f"Amount ₹{amount:,.0f} is {abs(zscore):.1f}σ above your usual "
                    f"{category} spend (mean ₹{cat_mean:,.0f})."
                )
            elif feat == "amount_zscore" and sig["direction"] == "low":
                zscore = features[0]
                parts.append(
                    f"Amount ₹{amount:,.0f} is {abs(zscore):.1f}σ below your usual "
                    f"{category} spend (mean ₹{cat_mean:,.0f})."
                )
            elif feat == "is_new_vendor":
                parts.append(f"New vendor '{vendor}' — not seen in your history.")
            elif feat == "vendor_frequency":
                pct = features[2] * 100
                parts.append(
                    f"Rare vendor '{vendor}' — only {pct:.0f}% of your receipts."
                )
            elif feat == "day_of_week_score":
                parts.append(
                    f"Unusual day — you rarely make {category} purchases on "
                    f"{weekday_name}."
                )
            elif feat == "hour_of_day_score":
                parts.append(
                    f"Unusual time — {category} purchases rarely at {hour_val}:00."
                )
            elif feat == "category_budget_ratio":
                ratio = features[8]
                parts.append(
                    f"Amount is {ratio:.1f}× your average weekly {category} budget."
                )
            elif feat == "days_since_vendor":
                days = int(features[7] * 365)
                parts.append(
                    f"First purchase from '{vendor}' in {days} days."
                )
            elif feat == "amount_vs_global":
                zscore = features[6]
                direction = "above" if sig["direction"] == "high" else "below"
                parts.append(
                    f"Amount ₹{amount:,.0f} is {abs(zscore):.1f}σ {direction} "
                    f"your overall average spend."
                )
            elif feat == "amount_percentile":
                pctl = features[1] * 100
                parts.append(
                    f"This {category} amount is at the {pctl:.0f}th percentile "
                    f"of your spending."
                )

        explanation = " ".join(parts)
        return explanation[:200] if len(explanation) > 200 else explanation


# ── Anomaly logging ───────────────────────────────────────────────────────────

def _log_anomaly(
    user_id: str,
    expense: ExpenseRecord,
    result: AnomalyResult,
) -> None:
    """Append an anomaly event to the user's anomaly log.

    Args:
        user_id: The user whose log to write.
        expense: The anomalous expense.
        result:  The AnomalyResult for this expense.
    """
    user_dir = _DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    log_path = user_dir / "anomaly_log.jsonl"

    entry = {
        "receipt_id": expense.receipt_id,
        "vendor": expense.vendor,
        "date": expense.date,
        "amount": expense.total_amount,
        "category": expense.category,
        "anomaly_score": result.anomaly_score,
        "is_anomaly": result.is_anomaly,
        "explanation": result.explanation,
        "model_scores": result.model_scores,
        "signals": result.signals,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
