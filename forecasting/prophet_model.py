"""
Per-user Prophet-based spending forecaster.

Key improvements over v1:
  - Weekly aggregation: trains on weekly totals instead of daily
    (eliminates zero-noise, dramatically improves accuracy)
  - Indian holidays: spending spikes during Diwali, Holi, etc. are
    modelled as regressors so Prophet doesn't mistake them for changepoints
  - Conservative changepoint_prior_scale=0.05 (avoids overfitting)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting.aggregator import build_category_timeseries

# Silence Prophet/cmdstanpy verbose output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

# Minimum weeks of data to train (≈ 2 weeks with weekly aggregation)
_MIN_WEEKS = 4


# ── Indian holidays DataFrame ─────────────────────────────────────────────────

def _get_indian_holidays() -> pd.DataFrame:
    """Return a Prophet-compatible holidays DataFrame for India (2023–2027).

    Covers major events that drive spending spikes: festival shopping
    (Diwali, Dussehra), travel (Durga Puja, Eid), and public holidays.
    """
    holidays_data = [
        # ── 2023 ──────────────────────────────────
        {"holiday": "Diwali",         "ds": "2023-11-12", "lower_window": -3, "upper_window": 2},
        {"holiday": "Dussehra",       "ds": "2023-10-24", "lower_window": -1, "upper_window": 1},
        {"holiday": "Holi",           "ds": "2023-03-08", "lower_window": -1, "upper_window": 1},
        {"holiday": "Eid_ul_Fitr",    "ds": "2023-04-22", "lower_window": -2, "upper_window": 1},
        {"holiday": "Durgapuja",      "ds": "2023-10-22", "lower_window": -2, "upper_window": 2},
        {"holiday": "Independence",   "ds": "2023-08-15", "lower_window":  0, "upper_window": 0},
        {"holiday": "Republic",       "ds": "2023-01-26", "lower_window":  0, "upper_window": 0},
        {"holiday": "Christmas",      "ds": "2023-12-25", "lower_window": -1, "upper_window": 1},
        {"holiday": "NewYear",        "ds": "2023-01-01", "lower_window": -1, "upper_window": 1},
        {"holiday": "BigBillionDays", "ds": "2023-10-08", "lower_window":  0, "upper_window": 4},
        # ── 2024 ──────────────────────────────────
        {"holiday": "Diwali",         "ds": "2024-11-01", "lower_window": -3, "upper_window": 2},
        {"holiday": "Dussehra",       "ds": "2024-10-12", "lower_window": -1, "upper_window": 1},
        {"holiday": "Holi",           "ds": "2024-03-25", "lower_window": -1, "upper_window": 1},
        {"holiday": "Eid_ul_Fitr",    "ds": "2024-04-10", "lower_window": -2, "upper_window": 1},
        {"holiday": "Durgapuja",      "ds": "2024-10-12", "lower_window": -2, "upper_window": 2},
        {"holiday": "Independence",   "ds": "2024-08-15", "lower_window":  0, "upper_window": 0},
        {"holiday": "Republic",       "ds": "2024-01-26", "lower_window":  0, "upper_window": 0},
        {"holiday": "Christmas",      "ds": "2024-12-25", "lower_window": -1, "upper_window": 1},
        {"holiday": "NewYear",        "ds": "2024-01-01", "lower_window": -1, "upper_window": 1},
        {"holiday": "BigBillionDays", "ds": "2024-10-07", "lower_window":  0, "upper_window": 4},
        # ── 2025 ──────────────────────────────────
        {"holiday": "Diwali",         "ds": "2025-10-20", "lower_window": -3, "upper_window": 2},
        {"holiday": "Dussehra",       "ds": "2025-10-02", "lower_window": -1, "upper_window": 1},
        {"holiday": "Holi",           "ds": "2025-03-14", "lower_window": -1, "upper_window": 1},
        {"holiday": "Eid_ul_Fitr",    "ds": "2025-03-31", "lower_window": -2, "upper_window": 1},
        {"holiday": "Durgapuja",      "ds": "2025-10-02", "lower_window": -2, "upper_window": 2},
        {"holiday": "Independence",   "ds": "2025-08-15", "lower_window":  0, "upper_window": 0},
        {"holiday": "Republic",       "ds": "2025-01-26", "lower_window":  0, "upper_window": 0},
        {"holiday": "Christmas",      "ds": "2025-12-25", "lower_window": -1, "upper_window": 1},
        {"holiday": "NewYear",        "ds": "2025-01-01", "lower_window": -1, "upper_window": 1},
        {"holiday": "BigBillionDays", "ds": "2025-10-06", "lower_window":  0, "upper_window": 4},
        # ── 2026 ──────────────────────────────────
        {"holiday": "Diwali",         "ds": "2026-11-08", "lower_window": -3, "upper_window": 2},
        {"holiday": "Holi",           "ds": "2026-03-03", "lower_window": -1, "upper_window": 1},
        {"holiday": "Independence",   "ds": "2026-08-15", "lower_window":  0, "upper_window": 0},
        {"holiday": "Republic",       "ds": "2026-01-26", "lower_window":  0, "upper_window": 0},
        {"holiday": "Christmas",      "ds": "2026-12-25", "lower_window": -1, "upper_window": 1},
        {"holiday": "NewYear",        "ds": "2026-01-01", "lower_window": -1, "upper_window": 1},
        # ── 2027 ──────────────────────────────────
        {"holiday": "Diwali",         "ds": "2027-10-29", "lower_window": -3, "upper_window": 2},
        {"holiday": "Holi",           "ds": "2027-03-22", "lower_window": -1, "upper_window": 1},
        {"holiday": "Independence",   "ds": "2027-08-15", "lower_window":  0, "upper_window": 0},
        {"holiday": "Republic",       "ds": "2027-01-26", "lower_window":  0, "upper_window": 0},
        {"holiday": "Christmas",      "ds": "2027-12-25", "lower_window": -1, "upper_window": 1},
        {"holiday": "NewYear",        "ds": "2027-01-01", "lower_window": -1, "upper_window": 1},
    ]

    df = pd.DataFrame(holidays_data)
    df["ds"] = pd.to_datetime(df["ds"])
    df["lower_window"] = df["lower_window"].astype(int)
    df["upper_window"] = df["upper_window"].astype(int)
    return df


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """Output of a Prophet weekly-aggregated forecast."""
    category: str
    forecast_df: pd.DataFrame          # columns: ds, yhat, yhat_lower, yhat_upper (weekly)
    trend: str                         # "increasing" | "stable" | "decreasing"
    predicted_monthly_total: float     # sum of yhat over next ~4 weeks (scale to days=30)
    days_until_budget_breach: Optional[int]
    granularity: str                   # "weekly" (always for training; daily for breach calc)


# ── PerUserProphetForecaster ──────────────────────────────────────────────────

class PerUserProphetForecaster:
    """Manages per-(user, category) Prophet models.

    Training is always on **weekly-aggregated** data to smooth out the
    zero-inflation in daily expense series.  Forecasts are produced in
    weekly bins then scaled back to a monthly total for display.
    """

    @staticmethod
    def _model_path(user_id: str, category: str) -> Path:
        cat_safe = category.strip().lower().replace(" ", "_")
        return _MODELS_DIR / f"{user_id}_{cat_safe}_prophet.json"

    def train(self, user_id: str, category: str) -> bool:
        """Fit a Prophet model on the user's weekly category timeseries.

        Args:
            user_id:  Owner user.
            category: Expense category.

        Returns:
            True if training succeeded, False if insufficient data.
        """
        from prophet import Prophet
        from prophet.serialize import model_to_json

        # Use weekly aggregation — the key improvement
        df = build_category_timeseries(user_id, category, freq="W-MON")
        if df is None or len(df) < _MIN_WEEKS:
            return False

        indian_holidays = _get_indian_holidays()

        model = Prophet(
            holidays=indian_holidays,
            weekly_seasonality=False,      # weekly data → no sub-weekly seasonality
            daily_seasonality=False,
            yearly_seasonality=True,       # captures seasonal spending (festivals, summer)
            changepoint_prior_scale=0.05,  # conservative: avoids fitting to outlier weeks
            holidays_prior_scale=10.0,     # allow holidays to have meaningful effect
            seasonality_mode="additive",         # safe for expense data
        )
        model.fit(df)

        # Persist
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = self._model_path(user_id, category)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(model_to_json(model))

        return True

    def forecast(
        self,
        user_id: str,
        category: str,
        days: int = 30,
        monthly_budget: Optional[float] = None,
    ) -> Optional[ForecastResult]:
        """Generate a spending forecast for the next ~N days (in weekly bins).

        Args:
            user_id:        Owner user.
            category:       Expense category.
            days:           Forecast horizon in days (rounded up to nearest week).
            monthly_budget: Optional limit for breach estimation.

        Returns:
            ForecastResult or None if insufficient data or model not found.
        """
        from prophet.serialize import model_from_json

        model_path = self._model_path(user_id, category)
        if not model_path.exists():
            if not self.train(user_id, category):
                return None

        try:
            with open(model_path, "r", encoding="utf-8") as f:
                model = model_from_json(f.read())
        except Exception:
            return None

        # Forecast in weeks
        n_weeks = max(1, int(np.ceil(days / 7)))
        future = model.make_future_dataframe(periods=n_weeks, freq="W")
        forecast_full = model.predict(future)

        # Extract only future weeks
        historical_weekly = build_category_timeseries(user_id, category, freq="W-MON")
        if historical_weekly is None:
            return None

        last_date = historical_weekly["ds"].max()
        mask = forecast_full["ds"] > last_date
        future_df = forecast_full.loc[
            mask, ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        # Clamp negatives to 0
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            future_df[col] = future_df[col].clip(lower=0)

        # Scale weekly totals to a monthly total
        # Each week has 7 days; take first 30 days worth of weeks
        predicted_total = float(future_df["yhat"].head(n_weeks).sum())

        trend = self._detect_trend(future_df)

        # Days until breach (uses daily data for precision)
        days_until_breach = None
        if monthly_budget is not None and monthly_budget > 0:
            daily_df = build_category_timeseries(user_id, category, freq="D")
            if daily_df is not None:
                days_until_breach = self._days_until_breach(
                    daily_df, future_df, monthly_budget
                )

        return ForecastResult(
            category=category,
            forecast_df=future_df.reset_index(drop=True),
            trend=trend,
            predicted_monthly_total=round(predicted_total, 2),
            days_until_budget_breach=days_until_breach,
            granularity="weekly",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_trend(future_df: pd.DataFrame) -> str:
        """Detect trend from the forecast — compares first vs last two weeks."""
        if len(future_df) < 2:
            return "stable"

        mid = max(1, len(future_df) // 2)
        first_half = future_df.iloc[:mid]["yhat"].mean()
        second_half = future_df.iloc[mid:]["yhat"].mean()

        if first_half == 0:
            return "stable"

        change_pct = (second_half - first_half) / first_half
        if change_pct > 0.10:
            return "increasing"
        elif change_pct < -0.10:
            return "decreasing"
        return "stable"

    @staticmethod
    def _days_until_breach(
        daily_df: pd.DataFrame,
        weekly_forecast_df: pd.DataFrame,
        monthly_budget: float,
    ) -> Optional[int]:
        """Estimate days until cumulative monthly spend hits budget.

        Combines already-spent amount this month with daily pro-rated
        weekly forecast values.
        """
        now = pd.Timestamp.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Amount already spent this calendar month
        this_month = daily_df[daily_df["ds"] >= month_start]
        spent_so_far = float(this_month["y"].sum())

        if spent_so_far >= monthly_budget:
            return 0

        # Pro-rate weekly to daily: divide each week's yhat by 7
        cumulative = spent_so_far
        day_count = 0
        for _, row in weekly_forecast_df.iterrows():
            daily_rate = row["yhat"] / 7.0
            for _ in range(7):
                day_count += 1
                cumulative += daily_rate
                if cumulative >= monthly_budget:
                    return day_count

        return None
