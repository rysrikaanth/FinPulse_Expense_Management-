"""
Time-series aggregation for expense forecasting.

Supports two granularities:
  - daily  [ds, y]  — granular, but noisy (many zero days)
  - weekly [ds, y]  — aggregated to Monday-anchored weeks, much smoother

Weekly aggregation is the default for Prophet training because daily
expense data is "staccato" (sparse zeros) and weekly totals drastically
reduce noise while preserving the seasonality signal.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.expense_store import load_expenses


def build_category_timeseries(
    user_id: str,
    category: str,
    freq: str = "D",
) -> Optional[pd.DataFrame]:
    """Build a spend time-series for a single category.

    Args:
        user_id:  The user whose history to aggregate.
        category: Category to filter on (case-insensitive).
        freq:     Pandas frequency string — ``"D"`` (daily) or ``"W-MON"``
                  (weekly, Monday-anchored).  Weekly is recommended for
                  Prophet training on sparse data.

    Returns:
        DataFrame with columns [ds, y].  Daily: one row per calendar day,
        gaps filled with 0.  Weekly: one row per week (Monday), gaps filled
        with 0.  Returns None if no data exists for this category.
    """
    expenses = load_expenses(user_id)
    cat_lower = category.strip().lower()

    rows: list[dict] = []
    for exp in expenses:
        if (exp.category or "").strip().lower() != cat_lower:
            continue
        if exp.date and exp.total_amount is not None:
            rows.append({"ds": exp.date, "y": exp.total_amount})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])

    if freq == "D":
        # Aggregate same-day purchases, then fill daily gaps
        df = df.groupby("ds", as_index=False)["y"].sum()
        date_range = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="D")
        df = df.set_index("ds").reindex(date_range, fill_value=0.0).reset_index()
        df.columns = ["ds", "y"]
    else:
        # Weekly aggregation: resample to week-ending Monday (W-MON)
        df = df.set_index("ds").resample("W-MON")["y"].sum().reset_index()
        df.columns = ["ds", "y"]

    return df


def build_total_timeseries(
    user_id: str,
    freq: str = "D",
) -> Optional[pd.DataFrame]:
    """Build a spend time-series across all categories.

    Args:
        user_id: The user whose history to aggregate.
        freq:    Frequency — ``"D"`` or ``"W-MON"``.

    Returns:
        DataFrame with columns [ds, y].  Returns None if no data.
    """
    expenses = load_expenses(user_id)

    rows: list[dict] = []
    for exp in expenses:
        if exp.date and exp.total_amount is not None:
            rows.append({"ds": exp.date, "y": exp.total_amount})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])

    if freq == "D":
        df = df.groupby("ds", as_index=False)["y"].sum()
        date_range = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="D")
        df = df.set_index("ds").reindex(date_range, fill_value=0.0).reset_index()
        df.columns = ["ds", "y"]
    else:
        df = df.set_index("ds").resample("W-MON")["y"].sum().reset_index()
        df.columns = ["ds", "y"]

    return df


def get_user_categories(user_id: str) -> list[str]:
    """Return the distinct categories in a user's history.

    Args:
        user_id: The user to query.

    Returns:
        Sorted list of lowercase category strings.
    """
    expenses = load_expenses(user_id)
    cats = set()
    for exp in expenses:
        if exp.category:
            cats.add(exp.category.strip().lower())
    return sorted(cats)
