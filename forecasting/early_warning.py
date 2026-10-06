"""
Early-warning system for budget breaches.

Runs after each ingested receipt and produces human-readable alerts
for categories approaching or exceeding their budgets.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting.aggregator import get_user_categories
from forecasting.budget_manager import (
    BudgetStatus,
    check_budget_status,
    get_budget,
    _load_budgets,
)


@dataclass
class BudgetAlert:
    """A single budget early-warning alert."""
    category: str
    alert_level: str                # "ok" | "warning" | "critical"
    message: str
    recommended_daily_limit: float  # remaining budget / remaining days


def run_early_warning(user_id: str) -> list[BudgetAlert]:
    """Run early-warning checks across all budgeted categories.

    Args:
        user_id: The user to check.

    Returns:
        List of BudgetAlert objects (only for warning / critical levels).
    """
    budgets = _load_budgets(user_id)
    if not budgets:
        return []

    alerts: list[BudgetAlert] = []

    for category, limit in budgets.items():
        status = check_budget_status(user_id, category)
        if status is None:
            continue

        if status.alert_level == "ok":
            continue

        # Compute recommended daily limit
        from datetime import datetime
        now = datetime.now()
        month_start = now.replace(day=1)
        next_month = month_start.replace(
            month=month_start.month % 12 + 1 if month_start.month < 12 else 1,
            year=month_start.year + (1 if month_start.month == 12 else 0),
        )
        days_remaining = max((next_month - now).days, 1)
        remaining_budget = max(limit - status.spent_so_far, 0)
        recommended = round(remaining_budget / days_remaining, 2)

        # Build message
        message = _build_message(status, recommended)

        alerts.append(BudgetAlert(
            category=category,
            alert_level=status.alert_level,
            message=message,
            recommended_daily_limit=recommended,
        ))

    # Sort: critical first, then warning
    alerts.sort(key=lambda a: 0 if a.alert_level == "critical" else 1)
    return alerts


def _build_message(status: BudgetStatus, recommended: float) -> str:
    """Build a human-readable alert message."""
    cat = status.category.title()
    limit = status.monthly_limit
    spent = status.spent_so_far
    predicted = status.predicted_total

    if status.alert_level == "critical":
        if status.days_until_breach is not None and status.days_until_breach > 0:
            msg = (
                f"At current pace, your {cat} spend will exceed "
                f"₹{limit:,.0f} budget in {status.days_until_breach} days. "
                f"Spent so far: ₹{spent:,.0f}. "
                f"Predicted total: ₹{predicted:,.0f}. "
                f"Recommended daily limit: ₹{recommended:,.0f}."
            )
        else:
            msg = (
                f"Your {cat} spend has already exceeded the "
                f"₹{limit:,.0f} budget! "
                f"Spent: ₹{spent:,.0f} ({status.percent_used:.0f}% used). "
                f"Reduce spending immediately."
            )
    else:  # warning
        msg = (
            f"Your {cat} spend is approaching the ₹{limit:,.0f} budget. "
            f"Spent so far: ₹{spent:,.0f} ({status.percent_used:.0f}% predicted). "
            f"Recommended daily limit: ₹{recommended:,.0f}."
        )

    return msg[:250]


def print_alerts(alerts: list[BudgetAlert]) -> None:
    """Pretty-print budget alerts to stdout."""
    if not alerts:
        return

    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │           BUDGET EARLY WARNINGS                │")
    print("  └─────────────────────────────────────────────────┘")

    for alert in alerts:
        icon = "🔴" if alert.alert_level == "critical" else "🟠"
        print(f"\n  {icon}  [{alert.alert_level.upper()}] {alert.category.title()}")
        print(f"     {alert.message}")
        print(f"     Daily limit: ₹{alert.recommended_daily_limit:,.0f}")

    print()
