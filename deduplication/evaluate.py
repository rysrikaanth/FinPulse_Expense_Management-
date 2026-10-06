"""
Evaluation script for the duplicate detection system.

Generates synthetic duplicate pairs (exact + fuzzy variants) and measures
Precision@K (K=1, 3, 5) for Faiss-based fuzzy matching.

Usage::

    python deduplication/evaluate.py
"""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from deduplication.detector import DuplicateDetector, _EMBED_DIM
from deduplication.fingerprinter import generate_exact_key, generate_fuzzy_embedding
from models.expense import ExpenseRecord, LineItem


# ── Synthetic test data ───────────────────────────────────────────────────────

def _make_expense(
    vendor: str,
    date: str,
    amount: float,
    items: list[str],
    user_id: str = "eval_user",
) -> ExpenseRecord:
    """Helper to build an ExpenseRecord quickly."""
    return ExpenseRecord(
        user_id=user_id,
        receipt_id=str(uuid.uuid4()),
        vendor=vendor,
        date=date,
        total_amount=amount,
        currency="INR",
        line_items=[LineItem(description=d) for d in items],
    )


def _generate_corpus_and_queries():
    """Generate a corpus of 20 base expenses and 20 query expenses.

    Query types:
      - 5 exact duplicates  (identical vendor/date/amount/items)
      - 5 near-duplicates   (slight vendor variation, ±1 day, same items)
      - 5 near-duplicates   (same vendor, items reordered, ±1 day)
      - 5 non-duplicates    (completely different)

    Returns:
        corpus:  list[ExpenseRecord]   — 20 base records
        queries: list[tuple[ExpenseRecord, str | None]]
                 — (query, expected_match_receipt_id or None)
    """
    base_data = [
        ("Kaiser", "2025-03-22", 757.0, ["Soya Tandoori Chaap Tikka", "Special Achari Soya Chaap"]),
        ("Starbucks", "2025-03-20", 550.0, ["Latte", "Blueberry Muffin"]),
        ("Swiggy", "2025-03-18", 450.0, ["Butter Chicken", "Naan", "Raita"]),
        ("BigBasket", "2025-03-15", 850.0, ["Vegetables", "Fruits", "Milk", "Bread"]),
        ("Ola", "2025-03-19", 180.0, ["Cab ride Koramangala to MG Road"]),
        ("Apollo Pharmacy", "2025-03-14", 450.0, ["Paracetamol", "Vitamin D"]),
        ("Amazon", "2025-03-10", 3000.0, ["Bluetooth Headphones"]),
        ("BookMyShow", "2025-03-21", 750.0, ["Movie Ticket PVR IMAX", "Popcorn"]),
        ("IRCTC", "2025-03-16", 1500.0, ["Rajdhani Express 2AC Berth"]),
        ("Airtel", "2025-03-01", 599.0, ["Prepaid Recharge 84 Days"]),
        ("BSES", "2025-03-05", 2800.0, ["Electricity Bill Delhi"]),
        ("Zomato", "2025-03-22", 380.0, ["Chicken Biryani", "Raita"]),
        ("Uber", "2025-03-17", 350.0, ["Airport Drop Terminal 3"]),
        ("DMart", "2025-03-12", 1200.0, ["Rice", "Wheat Flour", "Sugar", "Dal"]),
        ("Netflix", "2025-03-01", 649.0, ["Monthly Premium Subscription"]),
        ("Myntra", "2025-03-08", 2500.0, ["Ethnic Wear Saree"]),
        ("Udemy", "2025-03-09", 450.0, ["Web Development Course Python"]),
        ("LIC", "2025-03-02", 15000.0, ["Life Insurance Annual Premium"]),
        ("Cafe Coffee Day", "2025-03-20", 280.0, ["Cappuccino", "Sandwich"]),
        ("Flipkart", "2025-03-11", 15000.0, ["Smartphone OnePlus"]),
    ]

    corpus = [_make_expense(*d) for d in base_data]
    queries: list[tuple[ExpenseRecord, str | None]] = []

    # ── 5 exact duplicates ───────────────────────────────────────────────
    for i in range(5):
        d = base_data[i]
        exact = _make_expense(*d)
        queries.append((exact, corpus[i].receipt_id))

    # ── 5 near-dupes: vendor variation + ±1 day ─────────────────────────
    near_variants = [
        ("KAISER", "2025-03-23", 757.0, ["Soya Tandoori Chaap Tikka", "Special Achari Soya Chaap"]),
        ("Starbuck's Coffee", "2025-03-21", 550.0, ["Latte", "Blueberry Muffin"]),
        ("Swiggy Delivery", "2025-03-19", 450.0, ["Butter Chicken", "Naan", "Raita"]),
        ("Big Basket", "2025-03-16", 850.0, ["Vegetables", "Fruits", "Milk", "Bread"]),
        ("OLA Cabs", "2025-03-20", 180.0, ["Cab ride Koramangala to MG Road"]),
    ]
    for i, d in enumerate(near_variants):
        near = _make_expense(*d)
        queries.append((near, corpus[i].receipt_id))

    # ── 5 near-dupes: items reordered + ±1 day ──────────────────────────
    reordered = [
        ("Apollo Pharmacy", "2025-03-15", 450.0, ["Vitamin D", "Paracetamol"]),
        ("Amazon", "2025-03-11", 3000.0, ["Bluetooth Headphones"]),
        ("BookMyShow", "2025-03-22", 750.0, ["Popcorn", "Movie Ticket PVR IMAX"]),
        ("IRCTC", "2025-03-17", 1500.0, ["Rajdhani Express 2AC Berth"]),
        ("Airtel", "2025-03-02", 599.0, ["Prepaid Recharge 84 Days"]),
    ]
    for i, d in enumerate(reordered):
        reordered_exp = _make_expense(*d)
        queries.append((reordered_exp, corpus[5 + i].receipt_id))

    # ── 5 non-duplicates ─────────────────────────────────────────────────
    non_dupes = [
        ("Taj Hotel", "2025-04-01", 8500.0, ["Room Service Premium"]),
        ("Shell Petrol", "2025-04-02", 2500.0, ["Full Tank Diesel"]),
        ("Monginis", "2025-04-03", 400.0, ["Birthday Cake Pastry"]),
        ("WeWork", "2025-04-04", 8000.0, ["Coworking Monthly Desk"]),
        ("Clove Dental", "2025-04-05", 1800.0, ["Scaling Cleaning Checkup"]),
    ]
    for d in non_dupes:
        queries.append((_make_expense(*d), None))

    return corpus, queries


# ── Evaluation ────────────────────────────────────────────────────────────────

def _precision_at_k(
    expected_id: str | None,
    result_ids: list[str],
    k: int,
) -> float:
    """Binary Precision@K: 1.0 if expected_id in top-K results, else 0.0."""
    if expected_id is None:
        # For non-duplicates, success = no results returned
        return 1.0 if len(result_ids[:k]) == 0 else 0.0
    return 1.0 if expected_id in result_ids[:k] else 0.0


def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║      DUPLICATE DETECTION — EVALUATION REPORT           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    corpus, queries = _generate_corpus_and_queries()
    user_id = "eval_user"

    # Build detector with corpus
    det = DuplicateDetector()
    # Manually populate cache
    import faiss as _faiss

    index = _faiss.IndexFlatL2(_EMBED_DIM)
    id_list = []
    key_set = {}

    vectors = []
    for exp in corpus:
        vec = generate_fuzzy_embedding(exp)
        vectors.append(vec)
        id_list.append(exp.receipt_id)
        key_set[generate_exact_key(exp)] = exp.receipt_id

    matrix = np.array(vectors, dtype=np.float32)
    _faiss.normalize_L2(matrix)
    index.add(matrix)

    det._cache[user_id] = {
        "index": index,
        "id_list": id_list,
        "key_set": key_set,
        "expenses": corpus,
    }

    print(f"  Corpus size       : {len(corpus)} expenses")
    print(f"  Query count       : {len(queries)}")
    print(f"    - Exact dupes   : 5")
    print(f"    - Near dupes    : 10  (5 vendor variation + 5 reordered items)")
    print(f"    - Non-dupes     : 5")

    # ── Run queries ──────────────────────────────────────────────────────
    p_at_1, p_at_3, p_at_5 = [], [], []
    results_table: list[tuple] = []

    for query_exp, expected_id in queries:
        matches = det.find_duplicates(query_exp, user_id, threshold=0.5)
        result_ids = [m.matched_expense_id for m in matches]

        p_at_1.append(_precision_at_k(expected_id, result_ids, 1))
        p_at_3.append(_precision_at_k(expected_id, result_ids, 3))
        p_at_5.append(_precision_at_k(expected_id, result_ids, 5))

        top_match = matches[0] if matches else None
        results_table.append((
            query_exp.vendor,
            "exact" if expected_id and any(m.match_type == "exact" for m in matches) else
            "near" if expected_id else "none",
            expected_id is not None,
            top_match.match_type if top_match else "—",
            f"{top_match.similarity_score:.2f}" if top_match else "—",
            expected_id in result_ids if expected_id else len(result_ids) == 0,
        ))

    # ── Results table ────────────────────────────────────────────────────
    print("\n  ┌────────────────────────────────────────────────────────────────────┐")
    print("  │                      QUERY RESULTS                               │")
    print("  └────────────────────────────────────────────────────────────────────┘\n")

    print(f"  {'Vendor':<22s} {'Type':<6s} {'IsDup':<6s} {'Found':<6s} {'Score':<7s} {'Hit':<5s}")
    print("  " + "─" * 58)
    for vendor, qtype, is_dup, found_type, score, hit in results_table:
        hit_str = "✅" if hit else "❌"
        print(f"  {vendor:<22s} {qtype:<6s} {str(is_dup):<6s} {found_type:<6s} {score:<7s} {hit_str}")

    # ── Precision@K ──────────────────────────────────────────────────────
    print("\n  ┌────────────────────────────────────────────────────────────────────┐")
    print("  │                    PRECISION @ K                                  │")
    print("  └────────────────────────────────────────────────────────────────────┘\n")

    avg_p1 = np.mean(p_at_1)
    avg_p3 = np.mean(p_at_3)
    avg_p5 = np.mean(p_at_5)

    print(f"  Precision@1  : {avg_p1:.2%}  ({sum(p_at_1):.0f}/{len(p_at_1)} correct)")
    print(f"  Precision@3  : {avg_p3:.2%}  ({sum(p_at_3):.0f}/{len(p_at_3)} correct)")
    print(f"  Precision@5  : {avg_p5:.2%}  ({sum(p_at_5):.0f}/{len(p_at_5)} correct)")

    # ── Breakdown by query type ──────────────────────────────────────────
    print("\n  By query type:")
    for label, start, end in [("Exact dupes", 0, 5), ("Near dupes", 5, 15), ("Non-dupes", 15, 20)]:
        slice_p1 = np.mean(p_at_1[start:end])
        print(f"    {label:<14s} P@1 = {slice_p1:.2%}")

    print()


if __name__ == "__main__":
    main()
