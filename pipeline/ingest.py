"""
Main ingestion entry point.

Usage::

    python pipeline/ingest.py --image receipt.jpg --user_id user_001

Pipeline:  Mindee OCR → normalize → **dedup check** → store → **anomaly check** → **budget alerts** → summary.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ocr_extractor import MindeeFullParser
from pipeline.normalizer import normalize_mindee_json
from storage.expense_store import save_expense
from models.expense import ExpenseRecord
from deduplication.resolver import resolve_duplicates
from deduplication.detector import detector as dedup_detector
from anomaly.ensemble import AnomalyEnsemble, AnomalyResult, _log_anomaly
from anomaly.isolation_forest_model import PerUserIsolationForest
from anomaly.autoencoder_model import PerUserAutoencoder
from forecasting.early_warning import run_early_warning, print_alerts


def process_receipt(image_path: str, user_id: str) -> ExpenseRecord | None:
    """End-to-end receipt ingestion: OCR → normalize → dedup → store → anomaly → return.

    Args:
        image_path: Path to the receipt image file.
        user_id:    Owner user ID for scoping the record.

    Returns:
        The persisted ExpenseRecord, or None if rejected as duplicate.
    """
    # 1 ── OCR via Mindee V2
    parser = MindeeFullParser()
    raw_json = parser.get_full_json(image_path)

    if "error" in raw_json:
        raise RuntimeError(f"Mindee API error: {raw_json['error']}")

    # 2 ── Normalize
    source_file = os.path.basename(image_path)
    record = normalize_mindee_json(raw_json, user_id, source_file)

    # 3 ── Duplicate check (BEFORE saving)
    resolution = resolve_duplicates(record, user_id)

    if resolution.action == "reject":
        _print_rejected(record, resolution.reason)
        return None

    if resolution.action == "flag":
        record.flagged = True
        record.flagged_reason = resolution.reason

    # 4 ── Persist
    store_path = save_expense(record)

    # 5 ── Update Faiss index with the new record
    dedup_detector.add_to_index(user_id, record)

    # 6 ── Anomaly detection (AFTER saving so record is in history)
    #      Lazily re-train models when history grows
    if_model = PerUserIsolationForest()
    if_model.fit_if_needed(user_id)

    ensemble = AnomalyEnsemble()
    anomaly_result = ensemble.score(record, user_id)

    if anomaly_result.is_anomaly:
        _log_anomaly(user_id, record, anomaly_result)

    # 7 ── Budget early warnings
    budget_alerts = run_early_warning(user_id)

    # 8 ── Summary
    _print_summary(record, store_path, anomaly_result)
    if budget_alerts:
        print_alerts(budget_alerts)

    return record


def _print_rejected(record: ExpenseRecord, reason: str) -> None:
    """Print a rejection notice for duplicate receipts."""
    print("\n" + "═" * 52)
    print("  🚫  DUPLICATE REJECTED")
    print("═" * 52)
    print(f"  Vendor       : {record.vendor or '—'}")
    print(f"  Date         : {record.date or '—'}")
    print(f"  Total        : {record.currency or '—'} {record.total_amount or '—'}")
    print(f"  Reason       : {reason}")
    print("═" * 52 + "\n")


def _print_summary(
    record: ExpenseRecord,
    store_path,
    anomaly_result: AnomalyResult | None = None,
) -> None:
    """Pretty-print a quick receipt summary to stdout."""
    if record.flagged:
        review_tag = "🔶 FLAGGED — possible duplicate"
    elif record.needs_review:
        review_tag = "⚠️  NEEDS REVIEW"
    else:
        review_tag = "✅ OK"
    items_count = len(record.line_items)

    print("\n" + "═" * 52)
    print("  📝  EXPENSE INGESTED")
    print("═" * 52)
    print(f"  Vendor       : {record.vendor or '—'}")
    print(f"  Date         : {record.date or '—'}")
    print(f"  Time         : {record.time or '—'}")
    print(f"  Total        : {record.currency or '—'} {record.total_amount or '—'}")
    print(f"  Category     : {record.category or '—'} / {record.subcategory or '—'}")
    print(f"  Line items   : {items_count}")
    print(f"  Receipt #    : {record.receipt_number or '—'}")
    print(f"  Confidence   : {record.overall_confidence:.0%}")
    print(f"  Review       : {review_tag}")
    if record.flagged:
        print(f"  Flag reason  : {record.flagged_reason}")

    # ── Anomaly status ───────────────────────────────────────────────
    if anomaly_result is not None:
        score = anomaly_result.anomaly_score
        if anomaly_result.is_anomaly:
            print(f"  Anomaly      : 🔴 ANOMALY DETECTED — {anomaly_result.explanation}")
        elif score > 0.4:
            print(f"  Anomaly      : 🟡 UNUSUAL — score: {score:.2f}")
        elif score == 0.0 and "Insufficient" in anomaly_result.explanation:
            print(f"  Anomaly      : ⚪ Insufficient history (<10 records)")
        else:
            print(f"  Anomaly      : 🟢 Normal — score: {score:.2f}")

    print(f"  User         : {record.user_id}")
    print(f"  Receipt ID   : {record.receipt_id}")
    print(f"  Stored at    : {store_path}")
    print("═" * 52 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest a receipt image into the expense intelligence system."
    )
    ap.add_argument("--image", required=True, help="Path to receipt image.")
    ap.add_argument("--user_id", required=True, help="User ID for this expense.")
    args = ap.parse_args()

    try:
        process_receipt(args.image, args.user_id)
    except Exception as e:
        print(f"\n❌  Ingestion failed: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
