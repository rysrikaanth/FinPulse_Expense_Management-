"""
End-to-end evaluation of the anomaly detection system.

1. Generates a synthetic user with 50 realistic normal expenses
2. Fits both models (Isolation Forest + Autoencoder)
3. Injects 15 synthetic anomalies
4. Scores all 65 records using AnomalyEnsemble
5. Reports: AUC-ROC, Precision@K, Detection Rate, F1

Usage::

    python anomaly/evaluate.py
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import roc_auc_score, f1_score

from anomaly.ensemble import AnomalyEnsemble
from anomaly.isolation_forest_model import PerUserIsolationForest
from anomaly.autoencoder_model import PerUserAutoencoder
from anomaly.synthetic_injector import inject_synthetic_anomalies
from models.expense import ExpenseRecord
from storage.expense_store import save_expense, load_expenses, _DATA_DIR


# ── Synthetic corpus generation ───────────────────────────────────────────────

_USER_ID = "eval_user_001"

_CATEGORY_CONFIG = {
    "food": {
        "weight": 0.30,
        "vendors": ["Swiggy", "Zomato", "Kaiser", "Dominos", "Haldirams"],
        "amount_range": (100, 800),
        "hours": [(12, 14), (19, 21)],
        "subcategory": "restaurant",
    },
    "transport": {
        "weight": 0.20,
        "vendors": ["Ola", "Uber", "Metro Card", "Rapido"],
        "amount_range": (50, 500),
        "hours": [(8, 10), (17, 19)],
        "subcategory": "cab",
    },
    "shopping": {
        "weight": 0.20,
        "vendors": ["Amazon", "Flipkart", "BigBazaar", "DMart"],
        "amount_range": (200, 3000),
        "hours": [(10, 20)],
        "subcategory": "electronics",
    },
    "utilities": {
        "weight": 0.15,
        "vendors": ["Airtel", "BSES", "Jio", "Tata Power"],
        "amount_range": (300, 2000),
        "hours": [(10, 18)],
        "subcategory": "mobile_recharge",
    },
    "entertainment": {
        "weight": 0.15,
        "vendors": ["Netflix", "PVR", "BookMyShow", "Hotstar"],
        "amount_range": (100, 1500),
        "hours": [(18, 22)],
        "subcategory": "movies",
    },
}


def _random_date(start_offset: int = 0, span_days: int = 90) -> str:
    """Generate a random weekday-biased date in the past 90 days."""
    from datetime import datetime, timedelta

    base = datetime(2025, 1, 1) + timedelta(days=start_offset)
    day_offset = random.randint(0, span_days)
    dt = base + timedelta(days=day_offset)

    # 80% chance weekday
    if random.random() < 0.80 and dt.weekday() >= 5:
        dt -= timedelta(days=dt.weekday() - 4)  # move to Friday

    return dt.strftime("%Y-%m-%d")


def _random_time(hour_ranges: list[tuple[int, int]]) -> str:
    """Pick a random HH:MM from the given hour ranges."""
    low, high = random.choice(hour_ranges)
    hour = random.randint(low, high - 1)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"


def _generate_normal_expenses(n: int = 50) -> list[ExpenseRecord]:
    """Generate n realistic normal expenses spread across categories."""
    expenses: list[ExpenseRecord] = []

    # Build weighted category list
    category_pool: list[str] = []
    for cat, cfg in _CATEGORY_CONFIG.items():
        count = max(1, int(n * cfg["weight"]))
        category_pool.extend([cat] * count)

    random.shuffle(category_pool)
    category_pool = category_pool[:n]

    for cat in category_pool:
        cfg = _CATEGORY_CONFIG[cat]
        vendor = random.choice(cfg["vendors"])
        amount = round(random.uniform(*cfg["amount_range"]), 2)
        date = _random_date()
        time = _random_time(cfg["hours"])

        exp = ExpenseRecord(
            user_id=_USER_ID,
            receipt_id=str(uuid.uuid4()),
            vendor=vendor,
            date=date,
            time=time,
            total_amount=amount,
            currency="INR",
            category=cat,
            subcategory=cfg["subcategory"],
        )
        expenses.append(exp)

    return expenses


# ── Precision@K ───────────────────────────────────────────────────────────────

def _precision_at_k(scores: list[float], labels: list[bool], k: int) -> float:
    """Compute Precision@K: fraction of top-K scored items that are true anomalies."""
    indexed = sorted(
        zip(scores, labels), key=lambda x: x[0], reverse=True
    )
    top_k = indexed[:k]
    if not top_k:
        return 0.0
    return sum(1 for _, is_anom in top_k if is_anom) / k


# ── Main evaluation ──────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     ANOMALY DETECTION — EVALUATION REPORT              ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── 1. Clean previous eval data ──────────────────────────────────────
    user_dir = _DATA_DIR / _USER_ID
    if user_dir.exists():
        shutil.rmtree(user_dir)

    models_dir = Path(__file__).resolve().parent.parent / "models"
    for f in models_dir.glob(f"{_USER_ID}_*"):
        f.unlink(missing_ok=True)

    # ── 2. Generate and save 50 normal expenses ─────────────────────────
    print("  1. Generating 50 normal expenses …")
    normal_expenses = _generate_normal_expenses(50)
    for exp in normal_expenses:
        save_expense(exp)
    print(f"     Saved {len(normal_expenses)} records to storage.")

    # ── 3. Fit models ────────────────────────────────────────────────────
    print("\n  2. Training Isolation Forest …")
    if_model = PerUserIsolationForest()
    if_ok = if_model.fit(_USER_ID)
    print(f"     IsolationForest trained: {if_ok}")

    print("\n  3. Training Autoencoder …")
    ae_model = PerUserAutoencoder()
    ae_ok = ae_model.train(_USER_ID)
    print(f"     Autoencoder trained: {ae_ok}")

    # ── 4. Inject anomalies ──────────────────────────────────────────────
    print("\n  4. Injecting 15 synthetic anomalies …")
    mixed = inject_synthetic_anomalies(normal_expenses, n_anomalies=15)
    print(f"     Total records: {len(mixed)} "
          f"({sum(1 for _, a in mixed if not a)} normal + "
          f"{sum(1 for _, a in mixed if a)} anomalies)")

    # ── 5. Score all records ─────────────────────────────────────────────
    print("\n  5. Scoring all records with ensemble …")
    ensemble = AnomalyEnsemble()

    scores: list[float] = []
    labels: list[bool] = []
    all_results = []

    for exp, is_anomaly in mixed:
        result = ensemble.score(exp, _USER_ID)
        scores.append(result.anomaly_score)
        labels.append(is_anomaly)
        all_results.append((exp, is_anomaly, result))

    # ── 6. Compute metrics ───────────────────────────────────────────────
    print("\n  6. Computing metrics …\n")

    y_true = [1 if l else 0 for l in labels]
    y_scores = scores

    # AUC-ROC
    auc = roc_auc_score(y_true, y_scores)

    # Precision@K
    p_at_5 = _precision_at_k(scores, labels, 5)
    p_at_10 = _precision_at_k(scores, labels, 10)

    # Detection rate + F1 at threshold 0.7
    y_pred = [1 if s > 0.7 else 0 for s in y_scores]
    true_positives = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    total_positives = sum(y_true)
    detection_rate = true_positives / total_positives if total_positives > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    # ── 7. Print results ─────────────────────────────────────────────────
    print("  ╔══════════════════════════════════════╗")
    print("  ║   ANOMALY DETECTION EVALUATION       ║")
    print("  ╠══════════════════════════════════════╣")
    print(f"  ║  AUC-ROC         : {auc:.4f}           ║")
    print(f"  ║  Precision@5     : {p_at_5:.4f}           ║")
    print(f"  ║  Precision@10    : {p_at_10:.4f}           ║")
    print(f"  ║  Detection Rate  : {detection_rate:.4f}           ║")
    print(f"  ║  F1 @ 0.7        : {f1:.4f}           ║")
    print("  ╚══════════════════════════════════════╝")

    # ── 8. Top 5 detected anomalies ──────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │              TOP 5 DETECTED ANOMALIES                  │")
    print("  └─────────────────────────────────────────────────────────┘\n")

    ranked = sorted(all_results, key=lambda x: x[2].anomaly_score, reverse=True)[:5]
    for i, (exp, is_anom, result) in enumerate(ranked, 1):
        truth = "ANOMALY" if is_anom else "NORMAL"
        detected = "🔴" if result.is_anomaly else "🟢"
        print(f"  {i}. {detected}  [{truth}]  score={result.anomaly_score:.3f}")
        print(f"     Vendor: {exp.vendor}  |  ₹{exp.total_amount:,.0f}  |  {exp.category}")
        print(f"     Explanation: {result.explanation}")
        print(f"     Models: IF={result.model_scores.get('isolation_forest', '—')}  "
              f"AE={result.model_scores.get('autoencoder', '—')}")
        print()

    # ── 9. Score distribution ────────────────────────────────────────────
    anom_scores = [s for s, l in zip(scores, labels) if l]
    norm_scores = [s for s, l in zip(scores, labels) if not l]
    print(f"  Score distribution:")
    print(f"    Normal  — mean: {np.mean(norm_scores):.3f}, "
          f"std: {np.std(norm_scores):.3f}, max: {np.max(norm_scores):.3f}")
    print(f"    Anomaly — mean: {np.mean(anom_scores):.3f}, "
          f"std: {np.std(anom_scores):.3f}, min: {np.min(anom_scores):.3f}")
    print()


if __name__ == "__main__":
    main()
