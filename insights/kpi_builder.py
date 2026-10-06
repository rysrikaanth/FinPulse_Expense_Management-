"""
KPI builder — computes all user-level expense metrics for LLM grounding.

Every number that appears in an insight response MUST originate from
the ``UserKPIContext`` built here.  The LLM only *narrates*; it never
invents figures.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.expense_store import load_expenses

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class UserKPIContext:
    """All numeric KPIs needed to ground an LLM insight."""

    user_id: str
    generated_at: str
    period_start: str
    period_end: str
    total_spend_this_month: float
    spend_by_category: dict[str, float] = field(default_factory=dict)
    top_vendors: list[dict] = field(default_factory=list)
    anomalies_this_month: list[dict] = field(default_factory=list)
    budget_alerts: list[dict] = field(default_factory=list)
    forecast_summary: list[dict] = field(default_factory=list)
    month_over_month_change: dict[str, float] = field(default_factory=dict)
    unusual_transactions: list[dict] = field(default_factory=list)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_kpi_context(user_id: str) -> UserKPIContext:
    """Compute a full KPI snapshot for the current month.

    Integrates data from the expense store, anomaly log, budget manager,
    and forecasting system.  Gracefully degrades when any component has
    no data (returns empty lists / dicts).
    """
    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    period_start = month_start.strftime("%Y-%m-%d")
    period_end = now.strftime("%Y-%m-%d")

    # ── Load all expenses ────────────────────────────────────────────────
    all_expenses = load_expenses(user_id)

    # ── Current month filter (with fallback to latest active month) ──────
    current_month_exps = []
    for exp in all_expenses:
        if exp.date and exp.total_amount is not None:
            try:
                d = datetime.fromisoformat(exp.date)
                if d.year == now.year and d.month == now.month:
                    current_month_exps.append(exp)
            except ValueError:
                continue

    ref_year, ref_month = now.year, now.month
    if not current_month_exps and all_expenses:
        valid_dates = []
        for exp in all_expenses:
            if exp.date and exp.total_amount is not None:
                try:
                    valid_dates.append(datetime.fromisoformat(exp.date))
                except ValueError:
                    continue
        if valid_dates:
            latest = max(valid_dates)
            ref_year, ref_month = latest.year, latest.month
            month_start = datetime(ref_year, ref_month, 1)
            period_start = month_start.strftime("%Y-%m-%d")
            period_end = latest.strftime("%Y-%m-%d")
            for exp in all_expenses:
                if exp.date and exp.total_amount is not None:
                    try:
                        d = datetime.fromisoformat(exp.date)
                        if d.year == ref_year and d.month == ref_month:
                            current_month_exps.append(exp)
                    except ValueError:
                        continue

    # ── Last month filter (for MoM change) ───────────────────────────────
    if ref_month == 1:
        lm_year, lm_month = ref_year - 1, 12
    else:
        lm_year, lm_month = ref_year, ref_month - 1

    last_month_exps = []
    for exp in all_expenses:
        if exp.date and exp.total_amount is not None:
            try:
                d = datetime.fromisoformat(exp.date)
                if d.year == lm_year and d.month == lm_month:
                    last_month_exps.append(exp)
            except ValueError:
                continue

    # ── total_spend_this_month ───────────────────────────────────────────
    total_spend = sum(e.total_amount for e in current_month_exps if e.total_amount)

    # ── spend_by_category ────────────────────────────────────────────────
    cat_spend: dict[str, float] = defaultdict(float)
    for exp in current_month_exps:
        cat = (exp.category or "other").strip().lower()
        cat_spend[cat] += exp.total_amount or 0.0
    spend_by_category = dict(cat_spend)

    # ── top_vendors ──────────────────────────────────────────────────────
    vendor_total: dict[str, float] = defaultdict(float)
    vendor_count: dict[str, int] = defaultdict(int)
    for exp in current_month_exps:
        v = (exp.vendor or "Unknown").strip()
        vendor_total[v] += exp.total_amount or 0.0
        vendor_count[v] += 1

    top_vendors = sorted(
        [{"vendor": v, "total": round(vendor_total[v], 2), "count": vendor_count[v]}
         for v in vendor_total],
        key=lambda x: x["total"],
        reverse=True,
    )[:5]

    # ── anomalies_this_month & unusual_transactions ──────────────────────
    anomalies_this_month: list[dict] = []
    unusual_transactions: list[dict] = []

    anomaly_log_path = _DATA_DIR / user_id / "anomaly_log.jsonl"
    if anomaly_log_path.exists():
        all_anomalies: list[dict] = []
        with open(anomaly_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Filter for reference month or recent within 30 days
                    entry_date = entry.get("date", "")
                    if entry_date:
                        d = datetime.fromisoformat(entry_date)
                        if (d.year == ref_year and d.month == ref_month) or abs((now - d).days) <= 30:
                            all_anomalies.append(entry)
                except (json.JSONDecodeError, ValueError):
                    continue

        # If still empty, fall back to any recent entries in the log
        if not all_anomalies:
            with open(anomaly_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_anomalies.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Sort by score descending
        all_anomalies.sort(key=lambda x: x.get("anomaly_score", 0), reverse=True)

        # anomalies_this_month: only those with is_anomaly=True
        anomalies_this_month = [
            {
                "vendor": a.get("vendor", ""),
                "amount": a.get("amount", 0),
                "score": a.get("anomaly_score", 0),
                "explanation": a.get("explanation", ""),
            }
            for a in all_anomalies if a.get("is_anomaly", False)
        ]

        # unusual_transactions: top 3 by anomaly_score (regardless of threshold)
        unusual_transactions = [
            {
                "receipt_id": a.get("receipt_id", ""),
                "vendor": a.get("vendor", ""),
                "date": a.get("date", ""),
                "amount": a.get("amount", 0),
                "category": a.get("category", ""),
                "anomaly_score": a.get("anomaly_score", 0),
                "explanation": a.get("explanation", ""),
            }
            for a in all_anomalies[:3]
        ]

    # ── budget_alerts ────────────────────────────────────────────────────
    budget_alerts: list[dict] = []
    try:
        from forecasting.budget_manager import check_budget_status, get_budget
        for cat in spend_by_category:
            budget = get_budget(user_id, cat)
            if budget is None:
                continue
            status = check_budget_status(user_id, cat)
            if status is not None:
                budget_alerts.append({
                    "category": status.category,
                    "monthly_limit": status.monthly_limit,
                    "spent_so_far": status.spent_so_far,
                    "predicted_total": status.predicted_total,
                    "will_breach": status.will_breach,
                    "percent_used": status.percent_used,
                    "alert_level": status.alert_level,
                })
    except Exception:
        pass  # graceful degradation

    # ── forecast_summary ─────────────────────────────────────────────────
    forecast_summary: list[dict] = []
    try:
        from forecasting.prophet_model import PerUserProphetForecaster
        forecaster = PerUserProphetForecaster()
        for cat in spend_by_category:
            try:
                result = forecaster.forecast(user_id, cat, days=30)
                if result is not None:
                    forecast_summary.append({
                        "category": cat,
                        "predicted_total": result.predicted_monthly_total,
                        "trend": result.trend,
                        "will_breach": result.days_until_budget_breach is not None
                                       and result.days_until_budget_breach >= 0,
                        "days_until_breach": result.days_until_budget_breach,
                    })
            except Exception:
                continue
    except Exception:
        pass  # graceful degradation

    # ── month_over_month_change ──────────────────────────────────────────
    lm_cat_spend: dict[str, float] = defaultdict(float)
    for exp in last_month_exps:
        cat = (exp.category or "other").strip().lower()
        lm_cat_spend[cat] += exp.total_amount or 0.0

    all_cats = set(spend_by_category.keys()) | set(lm_cat_spend.keys())
    month_over_month: dict[str, float] = {}
    for cat in all_cats:
        this_m = spend_by_category.get(cat, 0.0)
        last_m = lm_cat_spend.get(cat, 0.0)

        if last_m == 0 and this_m > 0:
            month_over_month[cat] = 100.0
        elif last_m == 0 and this_m == 0:
            month_over_month[cat] = 0.0
        else:
            month_over_month[cat] = round(((this_m - last_m) / last_m) * 100, 1)

    return UserKPIContext(
        user_id=user_id,
        generated_at=now.isoformat(),
        period_start=period_start,
        period_end=period_end,
        total_spend_this_month=round(total_spend, 2),
        spend_by_category={k: round(v, 2) for k, v in spend_by_category.items()},
        top_vendors=top_vendors,
        anomalies_this_month=anomalies_this_month,
        budget_alerts=budget_alerts,
        forecast_summary=forecast_summary,
        month_over_month_change=month_over_month,
        unusual_transactions=unusual_transactions,
    )


# ── Prompt serialiser ─────────────────────────────────────────────────────────

def to_prompt_string(ctx: UserKPIContext) -> str:
    """Format UserKPIContext as a text block for LLM prompt injection.

    Returns:
        A dense, readable string (≤ 800 tokens) with all KPI data.
    """
    lines: list[str] = []

    lines.append(f"=== USER EXPENSE REPORT: {ctx.period_start} to {ctx.period_end} ===\n")
    lines.append(f"TOTAL SPEND THIS MONTH: ₹{ctx.total_spend_this_month:,.2f}\n")

    # ── Spend by category ────────────────────────────────────────────────
    if ctx.spend_by_category:
        lines.append("SPEND BY CATEGORY:")
        total = ctx.total_spend_this_month or 1.0
        for cat, amt in sorted(ctx.spend_by_category.items(), key=lambda x: x[1], reverse=True):
            pct = (amt / total * 100) if total > 0 else 0
            mom = ctx.month_over_month_change.get(cat, 0.0)
            arrow = f"↑{mom:+.1f}%" if mom >= 0 else f"↓{mom:.1f}%"
            lines.append(f"  {cat.title()}: ₹{amt:,.2f} ({pct:.1f}% of total) [{arrow} vs last month]")
        lines.append("")

    # ── Top vendors ──────────────────────────────────────────────────────
    if ctx.top_vendors:
        lines.append("TOP VENDORS:")
        for i, v in enumerate(ctx.top_vendors, 1):
            lines.append(f"  {i}. {v['vendor']} — ₹{v['total']:,.2f} across {v['count']} visits")
        lines.append("")

    # ── Anomalies ────────────────────────────────────────────────────────
    n_anom = len(ctx.anomalies_this_month)
    lines.append(f"ANOMALIES DETECTED THIS MONTH: {n_anom}")
    if ctx.anomalies_this_month:
        for a in ctx.anomalies_this_month[:5]:
            lines.append(
                f"  - {a['vendor']}: ₹{a['amount']:,.2f} (score {a['score']:.2f}) "
                f"— {a['explanation']}"
            )
    lines.append("")

    # ── Budget alerts ────────────────────────────────────────────────────
    if ctx.budget_alerts:
        lines.append("BUDGET ALERTS:")
        for b in ctx.budget_alerts:
            lines.append(
                f"  - {b['category'].title()}: spent ₹{b['spent_so_far']:,.2f} "
                f"of ₹{b['monthly_limit']:,.2f} limit "
                f"({b['percent_used']:.0f}%) [{b['alert_level'].upper()}]"
            )
        lines.append("")

    # ── Forecast ─────────────────────────────────────────────────────────
    if ctx.forecast_summary:
        lines.append("FORECASTS (next 30 days):")
        for f in ctx.forecast_summary:
            breach = ""
            if f.get("will_breach") and f.get("days_until_breach") is not None:
                breach = f" | ⚠ Will breach budget in {f['days_until_breach']} days"
            else:
                breach = " | ✓ Within budget"
            lines.append(
                f"  - {f['category'].title()}: predicted ₹{f['predicted_total']:,.2f} "
                f"| trend: {f['trend']}{breach}"
            )
        lines.append("")

    # ── Month-over-month ─────────────────────────────────────────────────
    if ctx.month_over_month_change:
        lines.append("MONTH-OVER-MONTH CHANGES:")
        for cat, pct in sorted(ctx.month_over_month_change.items()):
            sign = "+" if pct >= 0 else ""
            lines.append(f"  - {cat.title()}: {sign}{pct:.1f}%")
        lines.append("")

    return "\n".join(lines)
