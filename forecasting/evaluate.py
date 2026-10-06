"""
Walk-forward evaluation of the forecasting module (v2).

Improvements over v1:
  - 365 days of synthetic history (was 90) — enough for yearly seasonality
  - Weekly evaluation: both Prophet and baseline operate on weekly totals
  - Indian holiday effects injected into synthetic data (Diwali spike, etc.)
  - Metric comparison: Prophet vs naive weekly average baseline

Usage::

    python forecasting/evaluate.py
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.expense import ExpenseRecord
from storage.expense_store import save_expense, _DATA_DIR
from forecasting.aggregator import build_category_timeseries

_USER_ID = "forecast_eval_user"

# ── Category configs ──────────────────────────────────────────────────────────

_CATEGORY_CONFIG = {
    "food": {
        "daily_mean": 350,
        "daily_std": 120,
        "frequency": 0.78,
        "vendors": ["Swiggy", "Zomato", "Kaiser", "Dominos", "Haldirams", "McDonald's"],
        "weekend_spend_multiplier": 1.35,
    },
    "transport": {
        "daily_mean": 200,
        "daily_std": 90,
        "frequency": 0.55,
        "vendors": ["Ola", "Uber", "Metro Card", "Rapido", "IRCTC"],
        "weekend_spend_multiplier": 0.40,
    },
    "shopping": {
        "daily_mean": 900,
        "daily_std": 600,
        "frequency": 0.12,
        "vendors": ["Amazon", "Flipkart", "BigBazaar", "DMart", "Myntra"],
        "weekend_spend_multiplier": 1.60,
    },
    "utilities": {
        "daily_mean": 600,
        "daily_std": 250,
        "frequency": 0.07,
        "vendors": ["Airtel", "BSES", "Jio", "Tata Power", "Mahanagar Gas"],
        "weekend_spend_multiplier": 1.0,
    },
    "entertainment": {
        "daily_mean": 450,
        "daily_std": 250,
        "frequency": 0.18,
        "vendors": ["Netflix", "PVR", "BookMyShow", "Hotstar", "Spotify"],
        "weekend_spend_multiplier": 2.20,
    },
}

# Key Indian holidays that create spending spikes (for synthetic injection)
_HOLIDAY_SPIKES = {
    # date_str -> (multiplier, categories_affected)
    "2025-10-20": (3.5, ["shopping", "food"]),   # Diwali 2025
    "2025-10-19": (2.0, ["shopping", "food"]),   # Diwali eve
    "2025-10-18": (1.5, ["shopping"]),            # Diwali shopping
    "2025-10-02": (2.0, ["food", "shopping"]),    # Dussehra 2025
    "2025-03-14": (1.8, ["food", "entertainment"]), # Holi 2025
    "2024-11-01": (3.5, ["shopping", "food"]),    # Diwali 2024
    "2024-10-31": (2.0, ["shopping", "food"]),
    "2024-10-12": (2.0, ["food", "shopping"]),    # Dussehra 2024
    "2024-03-25": (1.8, ["food", "entertainment"]), # Holi 2024
    "2024-10-07": (2.5, ["shopping"]),             # Big Billion Days 2024
    "2024-10-08": (2.2, ["shopping"]),
    "2025-08-15": (1.3, ["food", "entertainment"]), # Independence Day
    "2025-01-26": (1.3, ["food", "entertainment"]), # Republic Day
    "2025-12-25": (2.0, ["shopping", "food", "entertainment"]), # Christmas
}


def _generate_synthetic_history(
    n_days: int = 365,
    start_date: datetime = datetime(2024, 4, 1),
) -> list[ExpenseRecord]:
    """Generate n_days of realistic synthetic expense data with Indian patterns."""
    records: list[ExpenseRecord] = []

    for day_offset in range(n_days):
        current_date = start_date + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        is_weekend = current_date.weekday() >= 5

        # Check for holiday spike
        holiday_spikes = _HOLIDAY_SPIKES.get(date_str, (1.0, []))
        holiday_multiplier = holiday_spikes[0] if isinstance(holiday_spikes, tuple) else 1.0
        holiday_cats = holiday_spikes[1] if isinstance(holiday_spikes, tuple) else []

        for cat, cfg in _CATEGORY_CONFIG.items():
            freq = cfg["frequency"]
            multiplier = cfg.get("weekend_spend_multiplier", 1.0)

            # Apply weekend frequency adjustment
            if is_weekend:
                if cat == "transport":
                    freq *= 0.30
                elif cat in ("entertainment", "shopping"):
                    freq *= 1.80

            # Boost during holidays
            if cat in holiday_cats:
                freq = min(freq * holiday_multiplier, 0.98)

            if random.random() > freq:
                continue

            amount = max(10.0, random.gauss(cfg["daily_mean"], cfg["daily_std"]))

            # Weekend spend multiplier (food is pricier when eating out)
            if is_weekend:
                amount *= multiplier

            # Holiday monetary spike
            if cat in holiday_cats:
                amount *= holiday_multiplier

            record = ExpenseRecord(
                user_id=_USER_ID,
                receipt_id=str(uuid.uuid4()),
                vendor=random.choice(cfg["vendors"]),
                date=date_str,
                time=f"{random.randint(8, 22):02d}:{random.randint(0, 59):02d}",
                total_amount=round(amount, 2),
                currency="INR",
                category=cat,
                subcategory="general",
            )
            records.append(record)

    return records


def _precision_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE — returns None if insufficient non-zero data."""
    if len(actual) == 0 or np.all(actual == 0) and np.all(predicted == 0):
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    nonzero = actual > 0
    mape = float(np.mean(np.abs(
        (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
    ))) * 100 if nonzero.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 68)
    print("  FORECASTING MODULE -- WALK-FORWARD EVALUATION (v2, 365-day)")
    print("=" * 68)

    # ── 1. Clean previous eval data ──────────────────────────────────────
    user_dir = _DATA_DIR / _USER_ID
    if user_dir.exists():
        shutil.rmtree(user_dir)

    models_dir = Path(__file__).resolve().parent.parent / "models"
    for f in models_dir.glob(f"{_USER_ID}_*"):
        f.unlink(missing_ok=True)

    # ── 2. Generate 365-day history ──────────────────────────────────────
    print("\n  1. Generating 365-day synthetic history with Indian patterns ...")
    start_date = datetime(2024, 4, 1)
    records = _generate_synthetic_history(365, start_date)
    for r in records:
        save_expense(r)
    print(f"     Saved {len(records)} records.")
    print(f"     Date range: {start_date.strftime('%Y-%m-%d')} → "
          f"{(start_date + timedelta(days=364)).strftime('%Y-%m-%d')}")

    # ── 3. Walk-forward split: 80% train, 20% test ──────────────────────
    split_day_offset = int(365 * 0.80)   # day 292 (≈ week 41)
    split_date = start_date + timedelta(days=split_day_offset)
    test_weeks = max(1, (365 - split_day_offset) // 7)

    print(f"\n  2. Walk-forward split:")
    print(f"     Train: days 1–{split_day_offset} (weeks 1–{split_day_offset // 7})")
    print(f"     Test : days {split_day_offset + 1}–365 ({test_weeks} weeks)")
    print(f"     Split date: {split_date.strftime('%Y-%m-%d')}")

    # ── 4. Evaluate per category ─────────────────────────────────────────
    from prophet import Prophet
    from forecasting.prophet_model import _get_indian_holidays

    categories = list(_CATEGORY_CONFIG.keys())
    results: list[dict] = []

    print("\n  3. Training & evaluating per category ...\n")

    for cat in categories:
        # Get weekly timeseries
        weekly_ts = build_category_timeseries(_USER_ID, cat, freq="W-MON")
        if weekly_ts is None or len(weekly_ts) < 8:
            print(f"     Skipping {cat}: insufficient weekly data")
            continue

        # Split
        train_df = weekly_ts[weekly_ts["ds"] < pd.Timestamp(split_date)].copy()
        test_df  = weekly_ts[weekly_ts["ds"] >= pd.Timestamp(split_date)].copy()

        if len(train_df) < 4 or len(test_df) < 2:
            print(f"     Skipping {cat}: insufficient split data")
            continue

        actual = test_df["y"].values
        n_test = len(actual)

        # ── Prophet with holidays (v2) ───────────────────────────────────
        indian_holidays = _get_indian_holidays()
        prophet_model = Prophet(
            holidays=indian_holidays,
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=False,  # need 2+ full years for year seasonality
            changepoint_prior_scale=0.05,
            holidays_prior_scale=2.0,   # conservative: prevent holiday bleed
            seasonality_mode="additive",
        )
        prophet_model.fit(train_df)
        # Use W-MON to match aggregator frequency exactly
        n_test = len(test_df)  # use actual test length, not computed
        future = prophet_model.make_future_dataframe(periods=n_test, freq="W-MON")
        pred_full = prophet_model.predict(future)

        # Positional alignment: pred_full contains train + test rows
        # Cap at 5x training mean as sanity check
        train_mean = float(train_df["y"].mean())
        prophet_preds_full = pred_full["yhat"].clip(lower=0, upper=train_mean * 5).values
        prophet_preds = prophet_preds_full[-n_test:]

        # Match length (date alignment may produce slight mismatch)
        min_len = min(len(actual), len(prophet_preds))
        actual_aligned = actual[:min_len]
        prophet_aligned = prophet_preds[:min_len]

        # ── Naive baseline: last-8-week rolling average ──────────────────
        naive_val = float(train_df.tail(8)["y"].mean())
        naive_preds = np.full(min_len, naive_val)

        prophet_metrics = _precision_metrics(actual_aligned, prophet_aligned)
        naive_metrics   = _precision_metrics(actual_aligned, naive_preds)

        # Skip categories with NaN metrics (e.g. all-zero test windows)
        if any(np.isnan(v) for v in prophet_metrics.values()):
            print(f"     Skipping {cat}: all-zero test window (too sparse)")
            continue

        winner_mae  = "✓" if prophet_metrics["MAE"]  <= naive_metrics["MAE"]  else " "
        winner_rmse = "✓" if prophet_metrics["RMSE"] <= naive_metrics["RMSE"] else " "
        winner_mape = "✓" if (not np.isnan(prophet_metrics["MAPE"]) and not np.isnan(naive_metrics["MAPE"]) and prophet_metrics["MAPE"] <= naive_metrics["MAPE"]) else " "

        print(f"     {cat:<14s}  Prophet MAE={prophet_metrics['MAE']:6.1f}{winner_mae}  "
              f"Naive MAE={naive_metrics['MAE']:6.1f}  "
              f"MAPE={prophet_metrics['MAPE']:5.1f}%{winner_mape}")

        results.append({
            "category":      cat,
            "n_train_weeks": len(train_df),
            "n_test_weeks":  min_len,
            "prophet_mae":   prophet_metrics["MAE"],
            "prophet_rmse":  prophet_metrics["RMSE"],
            "prophet_mape":  prophet_metrics["MAPE"],
            "naive_mae":     naive_metrics["MAE"],
            "naive_rmse":    naive_metrics["RMSE"],
            "naive_mape":    naive_metrics["MAPE"],
        })

    if not results:
        print("\n  No results — check data generation.")
        return

    # ── 5. Summary table ─────────────────────────────────────────────────
    print("\n  ─" * 35)
    print(f"\n  {'Category':<14s}  {'P-MAE':>7s}  {'N-MAE':>7s}  "
          f"{'P-RMSE':>7s}  {'N-RMSE':>7s}  {'P-MAPE':>7s}  {'N-MAPE':>7s}")
    print("  " + "-" * 72)

    for r in results:
        pm = "✓" if r["prophet_mae"]  <= r["naive_mae"]  else " "
        pr = "✓" if r["prophet_rmse"] <= r["naive_rmse"] else " "
        print(
            f"  {r['category']:<14s}"
            f"  {r['prophet_mae']:>6.1f}{pm}"
            f"  {r['naive_mae']:>7.1f}"
            f"  {r['prophet_rmse']:>6.1f}{pr}"
            f"  {r['naive_rmse']:>7.1f}"
            f"  {r['prophet_mape']:>6.1f}%"
            f"  {r['naive_mape']:>6.1f}%"
        )

    # ── 6. Aggregate improvement ──────────────────────────────────────────
    valid = [r for r in results if not np.isnan(r["prophet_mae"])]
    if not valid:
        print("\n  No valid results (all categories too sparse in test window).")
        return

    avg_p_mae  = np.nanmean([r["prophet_mae"]  for r in valid])
    avg_n_mae  = np.nanmean([r["naive_mae"]    for r in valid])
    avg_p_rmse = np.nanmean([r["prophet_rmse"] for r in valid])
    avg_n_rmse = np.nanmean([r["naive_rmse"]   for r in valid])
    avg_p_mape = np.nanmean([r["prophet_mape"] for r in valid])
    avg_n_mape = np.nanmean([r["naive_mape"]   for r in valid])

    mae_improvement  = (1 - avg_p_mae  / avg_n_mae)  * 100 if avg_n_mae  > 0 else float("nan")
    rmse_improvement = (1 - avg_p_rmse / avg_n_rmse) * 100 if avg_n_rmse > 0 else float("nan")
    mape_improvement = (1 - avg_p_mape / avg_n_mape) * 100 if avg_n_mape > 0 else float("nan")

    wins = sum(1 for r in valid if r["prophet_mae"] <= r["naive_mae"])

    print(f"\n  Avg  MAE  — Prophet: {avg_p_mae:6.1f}  |  Naive: {avg_n_mae:6.1f}"
          f"  |  Improvement: {mae_improvement:+.1f}%")
    print(f"  Avg  RMSE — Prophet: {avg_p_rmse:6.1f}  |  Naive: {avg_n_rmse:6.1f}"
          f"  |  Improvement: {rmse_improvement:+.1f}%")
    print(f"  Avg  MAPE — Prophet: {avg_p_mape:6.1f}%  |  Naive: {avg_n_mape:6.1f}%"
          f"  |  Improvement: {mape_improvement:+.1f}%")
    print(f"\n  Prophet wins on MAE in {wins}/{len(results)} categories.")
    print()


if __name__ == "__main__":
    main()
