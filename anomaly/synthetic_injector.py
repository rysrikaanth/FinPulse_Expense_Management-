"""
Synthetic anomaly injector for evaluation purposes.

Generates four types of anomalous expenses from a user's real history:
  1. Amount spike  (5× category mean)
  2. Unknown vendor (random string)
  3. Unusual timing (weekend + late night)
  4. Same-day duplicate category spend
"""

from __future__ import annotations

import os
import random
import string
import sys
import uuid
from collections import Counter
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.expense import ExpenseRecord


def _generate_expense(
    user_id: str,
    vendor: str,
    date: str,
    time: str,
    amount: float,
    category: str,
    subcategory: str = "general",
) -> ExpenseRecord:
    """Create a minimal valid ExpenseRecord with a new UUID."""
    return ExpenseRecord(
        user_id=user_id,
        receipt_id=str(uuid.uuid4()),
        vendor=vendor,
        date=date,
        time=time,
        total_amount=amount,
        currency="INR",
        category=category,
        subcategory=subcategory,
    )


def inject_synthetic_anomalies(
    history: list[ExpenseRecord],
    n_anomalies: int = 15,
) -> list[tuple[ExpenseRecord, bool]]:
    """Generate synthetic anomalies injected into a copy of the user history.

    Args:
        history:     The user's real expense history.
        n_anomalies: Number of anomalies to inject.

    Returns:
        List of (expense, is_anomaly) tuples containing ALL records
        (normal + injected).
    """
    if not history:
        return []

    user_id = history[0].user_id
    result: list[tuple[ExpenseRecord, bool]] = [(e, False) for e in history]

    # Pre-compute category stats
    category_amounts: dict[str, list[float]] = {}
    category_dates: dict[str, list[str]] = {}
    for e in history:
        cat = (e.category or "other").strip().lower()
        if e.total_amount is not None:
            category_amounts.setdefault(cat, []).append(e.total_amount)
        if e.date:
            category_dates.setdefault(cat, []).append(e.date)

    categories = list(category_amounts.keys())
    if not categories:
        return result

    n_type1 = n_anomalies // 4
    n_type2 = n_anomalies // 4
    n_type3 = n_anomalies // 4
    n_type4 = n_anomalies - n_type1 - n_type2 - n_type3

    # ── TYPE 1: Amount spike (5× category mean) ─────────────────────────
    for _ in range(n_type1):
        source = random.choice(history)
        cat = (source.category or "other").strip().lower()
        cat_mean = float(np.mean(category_amounts.get(cat, [500.0])))
        anomaly = _generate_expense(
            user_id=user_id,
            vendor=source.vendor or "Unknown",
            date=source.date or "2025-03-15",
            time=source.time or "12:00",
            amount=round(cat_mean * 5, 2),
            category=cat,
        )
        result.append((anomaly, True))

    # ── TYPE 2: Unknown vendor ───────────────────────────────────────────
    for _ in range(n_type2):
        source = random.choice(history)
        cat = (source.category or "other").strip().lower()
        random_vendor = "VENDOR_" + "".join(
            random.choices(string.ascii_uppercase, k=6)
        )
        anomaly = _generate_expense(
            user_id=user_id,
            vendor=random_vendor,
            date=source.date or "2025-03-15",
            time=source.time or "12:00",
            amount=source.total_amount or 500.0,
            category=cat,
        )
        result.append((anomaly, True))

    # ── TYPE 3: Unusual timing (Saturday at 23:00) ───────────────────────
    # Find weekday-dominant categories (>70% on Mon-Fri)
    weekday_dominant_cats = []
    for cat, dates in category_dates.items():
        weekday_count = 0
        total = 0
        for d in dates:
            try:
                dt = datetime.fromisoformat(d)
                total += 1
                if dt.weekday() < 5:  # Mon-Fri
                    weekday_count += 1
            except ValueError:
                continue
        if total > 0 and weekday_count / total > 0.7:
            weekday_dominant_cats.append(cat)

    if not weekday_dominant_cats:
        weekday_dominant_cats = categories[:1]

    for _ in range(n_type3):
        cat = random.choice(weekday_dominant_cats)
        cat_mean = float(np.mean(category_amounts.get(cat, [500.0])))

        # Force a Saturday date
        dates_in_hist = category_dates.get(cat, [])
        if dates_in_hist:
            base_date = datetime.fromisoformat(random.choice(dates_in_hist))
            # Move to next Saturday
            days_until_sat = (5 - base_date.weekday()) % 7
            if days_until_sat == 0:
                days_until_sat = 7
            sat_date = base_date.replace(day=min(base_date.day + days_until_sat, 28))
        else:
            sat_date = datetime(2025, 3, 15)  # Saturday

        anomaly = _generate_expense(
            user_id=user_id,
            vendor=random.choice([e.vendor or "Unknown" for e in history]),
            date=sat_date.strftime("%Y-%m-%d"),
            time="23:00",
            amount=round(cat_mean, 2),  # normal amount — only timing anomaly
            category=cat,
        )
        result.append((anomaly, True))

    # ── TYPE 4: Same-day duplicate category ──────────────────────────────
    for _ in range(n_type4):
        source = random.choice(history)
        cat = (source.category or "other").strip().lower()
        anomaly = _generate_expense(
            user_id=user_id,
            vendor=source.vendor or "Unknown",
            date=source.date or "2025-03-15",
            time=source.time or "14:00",
            amount=source.total_amount or 500.0,
            category=cat,
        )
        result.append((anomaly, True))

    random.shuffle(result)
    return result
