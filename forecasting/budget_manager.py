"""
Per-user budget management.

Stores category budgets in ``data/{user_id}/budgets.json`` and provides
budget-status checks combining actual spend with forecasted spend.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting.aggregator import build_category_timeseries
from forecasting.prophet_model import PerUserProphetForecaster
from storage.expense_store import load_expenses

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BudgetStatus:
    """Status of a user's budget for a single category."""
    category: str
    monthly_limit: float
    spent_so_far: float
    predicted_total: float
    will_breach: bool
    days_until_breach: Optional[int]
    percent_used: float
    alert_level: str   # "ok" | "warning" | "critical"


# ── Budget persistence ────────────────────────────────────────────────────────

def _budgets_path(user_id: str) -> Path:
    """Return the path to the user's budgets.json file."""
    user_dir = _DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "budgets.json"


def _load_budgets(user_id: str) -> dict[str, float]:
    """Load all budgets for a user."""
    path = _budgets_path(user_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_budgets(user_id: str, budgets: dict[str, float]) -> None:
    """Persist all budgets for a user."""
    path = _budgets_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(budgets, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def set_budget(user_id: str, category: str, monthly_limit: float) -> None:
    """Set or update a monthly budget for a category.

    Args:
        user_id:       Owner user.
        category:      Category to budget.
        monthly_limit: Monthly spending limit in the user's currency.
    """
    budgets = _load_budgets(user_id)
    budgets[category.strip().lower()] = monthly_limit
    _save_budgets(user_id, budgets)


def get_budget(user_id: str, category: str) -> Optional[float]:
    """Retrieve the monthly budget for a category.

    Args:
        user_id:  Owner user.
        category: Category to look up.

    Returns:
        The budget limit, or None if not set.
    """
    budgets = _load_budgets(user_id)
    return budgets.get(category.strip().lower())


def check_budget_status(user_id: str, category: str) -> Optional[BudgetStatus]:
    """Check the budget status for a category using actual + forecasted spend.

    Args:
        user_id:  Owner user.
        category: Category to check.

    Returns:
        BudgetStatus or None if no budget is set for this category.
    """
    limit = get_budget(user_id, category)
    if limit is None:
        return None

    cat_lower = category.strip().lower()

    # ── Actual spend this month ──────────────────────────────────────────
    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    expenses = load_expenses(user_id)
    spent = 0.0
    for exp in expenses:
        if (exp.category or "").strip().lower() != cat_lower:
            continue
        if exp.date and exp.total_amount is not None:
            try:
                exp_date = datetime.fromisoformat(exp.date)
                if exp_date >= month_start:
                    spent += exp.total_amount
            except ValueError:
                continue

    # ── Forecast remaining month ─────────────────────────────────────────
    days_remaining = (month_start.replace(
        month=month_start.month % 12 + 1 if month_start.month < 12 else 1,
        year=month_start.year + (1 if month_start.month == 12 else 0),
    ) - now).days
    days_remaining = max(days_remaining, 1)

    try:
        forecaster = PerUserProphetForecaster()
        forecast = forecaster.forecast(
            user_id, category,
            days=days_remaining,
            monthly_budget=limit,
        )
    except Exception:
        forecast = None

    if forecast is not None:
        predicted_total = spent + forecast.predicted_monthly_total
        days_until_breach = forecast.days_until_budget_breach
    else:
        # Naive fallback: project daily average for rest of month
        ts = build_category_timeseries(user_id, category)
        if ts is not None and len(ts) >= 7:
            daily_avg = float(ts["y"].tail(30).mean())
            predicted_total = spent + daily_avg * days_remaining
        else:
            predicted_total = spent
        days_until_breach = None

    # ── Alert level ──────────────────────────────────────────────────────
    percent_used = (predicted_total / limit * 100) if limit > 0 else 0.0
    will_breach = predicted_total > limit

    if percent_used >= 100:
        alert_level = "critical"
    elif percent_used >= 80:
        alert_level = "warning"
    else:
        alert_level = "ok"

    return BudgetStatus(
        category=cat_lower,
        monthly_limit=limit,
        spent_so_far=round(spent, 2),
        predicted_total=round(predicted_total, 2),
        will_breach=will_breach,
        days_until_breach=days_until_breach,
        percent_used=round(percent_used, 1),
        alert_level=alert_level,
    )
