"""
User-centric behavioral feature engineering for anomaly detection.

Computes a 9-dimensional contextual feature vector for each expense,
evaluated against the user's own spending history. Features capture
amount deviation, vendor novelty, temporal regularity, and budget ratio.

Cold-start rule: returns np.zeros(9) if history has fewer than 10 records.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.stats import percentileofscore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.expense import ExpenseRecord

# ── Feature index mapping ─────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "amount_zscore",
    "amount_percentile",
    "vendor_frequency",
    "is_new_vendor",
    "day_of_week_score",
    "hour_of_day_score",
    "amount_vs_global",
    "days_since_vendor",
    "category_budget_ratio",
]

_MIN_HISTORY = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> Optional[datetime]:
    """Parse an ISO date string to a datetime object."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def _parse_hour(time_str: str | None) -> Optional[int]:
    """Extract the hour component from an HH:MM string."""
    if not time_str:
        return None
    try:
        return int(time_str.split(":")[0])
    except (ValueError, IndexError):
        return None


def _hour_bucket(hour: int) -> int:
    """Map an hour (0–23) to a 4-hour bucket index (0–5)."""
    return hour // 4


# ── UserBehaviorFeaturizer ────────────────────────────────────────────────────

class UserBehaviorFeaturizer:
    """Computes per-expense feature vectors in the context of user history.

    Each feature is designed to detect a different type of anomaly:
      - Amount features (0, 1, 6, 8) → unusual spend levels
      - Vendor features (2, 3, 7)    → novel or rare vendors
      - Temporal features (4, 5)     → unusual day/time patterns
    """

    def featurize(
        self,
        expense: ExpenseRecord,
        history: list[ExpenseRecord],
    ) -> np.ndarray:
        """Compute a 9-dim behavioral feature vector.

        Args:
            expense: The new expense to featurize.
            history: The user's prior expense history.

        Returns:
            np.ndarray of shape (9,). All zeros if cold-start.
        """
        if len(history) < _MIN_HISTORY:
            return np.zeros(9, dtype=np.float32)

        amount = expense.total_amount or 0.0
        vendor = (expense.vendor or "").strip().lower()
        category = (expense.category or "").strip().lower()

        # ── Filter history by category ───────────────────────────────────
        cat_expenses = [
            e for e in history
            if (e.category or "").strip().lower() == category
        ]
        cat_amounts = [e.total_amount for e in cat_expenses if e.total_amount is not None]

        # ── All-history amounts ──────────────────────────────────────────
        all_amounts = [e.total_amount for e in history if e.total_amount is not None]

        features = np.zeros(9, dtype=np.float32)

        # 0 — amount_zscore (per-category)
        if cat_amounts and len(cat_amounts) >= 2:
            mean_cat = float(np.mean(cat_amounts))
            std_cat = float(np.std(cat_amounts))
            features[0] = (amount - mean_cat) / std_cat if std_cat > 0 else 0.0
        else:
            features[0] = 0.0

        # 1 — amount_percentile (per-category)
        if cat_amounts:
            features[1] = percentileofscore(cat_amounts, amount, kind="rank") / 100.0
        else:
            features[1] = 0.5

        # 2 — vendor_frequency
        vendor_count = sum(
            1 for e in history
            if (e.vendor or "").strip().lower() == vendor
        )
        features[2] = vendor_count / len(history) if history else 0.0

        # 3 — is_new_vendor
        features[3] = 1.0 if vendor_count == 0 else 0.0

        # 4 — day_of_week_score
        exp_date = _parse_date(expense.date)
        if exp_date and cat_expenses:
            exp_weekday = exp_date.weekday()
            weekday_matches = sum(
                1 for e in cat_expenses
                if _parse_date(e.date) is not None
                and _parse_date(e.date).weekday() == exp_weekday
            )
            features[4] = weekday_matches / len(cat_expenses)
        else:
            features[4] = 0.0

        # 5 — hour_of_day_score
        exp_hour = _parse_hour(expense.time)
        if exp_hour is not None and cat_expenses:
            exp_bucket = _hour_bucket(exp_hour)
            bucket_matches = sum(
                1 for e in cat_expenses
                if _parse_hour(e.time) is not None
                and _hour_bucket(_parse_hour(e.time)) == exp_bucket
            )
            features[5] = bucket_matches / len(cat_expenses)
        else:
            features[5] = 0.0

        # 6 — amount_vs_global
        if all_amounts and len(all_amounts) >= 2:
            mean_all = float(np.mean(all_amounts))
            std_all = float(np.std(all_amounts))
            features[6] = (amount - mean_all) / std_all if std_all > 0 else 0.0
        else:
            features[6] = 0.0

        # 7 — days_since_vendor
        if vendor_count == 0:
            features[7] = 1.0  # never seen → max normalized
        else:
            vendor_dates = [
                _parse_date(e.date)
                for e in history
                if (e.vendor or "").strip().lower() == vendor
                and _parse_date(e.date) is not None
            ]
            if vendor_dates and exp_date:
                most_recent = max(vendor_dates)
                days_elapsed = abs((exp_date - most_recent).days)
                features[7] = min(days_elapsed, 365) / 365.0
            else:
                features[7] = 1.0

        # 8 — category_budget_ratio
        if cat_amounts and cat_expenses:
            cat_dates = [
                _parse_date(e.date)
                for e in cat_expenses
                if _parse_date(e.date) is not None
            ]
            if cat_dates and len(cat_dates) >= 2:
                date_range = (max(cat_dates) - min(cat_dates)).days
                if date_range >= 7:
                    weeks = date_range / 7.0
                    avg_weekly = sum(cat_amounts) / weeks
                    features[8] = amount / avg_weekly if avg_weekly > 0 else 0.5
                else:
                    features[8] = 0.5
            else:
                features[8] = 0.5
        else:
            features[8] = 0.5

        return features
