"""
Duplicate detection engine using exact-key matching + Faiss vector search.

Per-user Faiss indices are cached in memory and persisted to disk at
``data/{user_id}/faiss.index`` + ``data/{user_id}/index_map.json``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deduplication.fingerprinter import generate_exact_key, generate_fuzzy_embedding
from models.expense import ExpenseRecord
from storage.expense_store import load_expenses

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_EMBED_DIM = 384


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DuplicateMatch:
    """A single duplicate-match result."""
    matched_expense_id: str
    similarity_score: float
    match_type: str          # "exact" | "fuzzy"
    reason: str


# ── DuplicateDetector ─────────────────────────────────────────────────────────

class DuplicateDetector:
    """Detects duplicate receipts within a single user's expense history.

    Combines two strategies:
      1. **Exact key**: SHA-256 hash match — catches identical re-uploads.
      2. **Fuzzy Faiss**: L2 distance on SBERT embeddings + date/amount guards.
    """

    def __init__(self):
        # Per-user caches: user_id -> (faiss_index, id_list, key_set, expenses)
        self._cache: dict[str, dict] = {}

    # ── Index management ──────────────────────────────────────────────────

    def build_index(self, user_id: str) -> None:
        """Build (or rebuild) the Faiss flat L2 index for a user.

        Loads all existing expenses, computes embeddings + exact keys,
        and persists the index to disk.

        Args:
            user_id: The user whose index to build.
        """
        expenses = load_expenses(user_id)

        index = faiss.IndexFlatL2(_EMBED_DIM)
        id_list: list[str] = []
        key_set: dict[str, str] = {}  # exact_key -> receipt_id

        if expenses:
            vectors = []
            for exp in expenses:
                vec = generate_fuzzy_embedding(exp)
                vectors.append(vec)
                id_list.append(exp.receipt_id)
                key_set[generate_exact_key(exp)] = exp.receipt_id

            matrix = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(matrix)
            index.add(matrix)

        self._cache[user_id] = {
            "index": index,
            "id_list": id_list,
            "key_set": key_set,
            "expenses": expenses,
        }
        self._persist_index(user_id)

    def _ensure_index(self, user_id: str) -> dict:
        """Lazily build the index if not already cached."""
        if user_id not in self._cache:
            # Try loading from disk first
            if not self._load_index(user_id):
                self.build_index(user_id)
        return self._cache[user_id]

    def _persist_index(self, user_id: str) -> None:
        """Save the Faiss index and ID map to disk."""
        user_dir = _DATA_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        cache = self._cache.get(user_id)
        if cache is None:
            return

        faiss.write_index(cache["index"], str(user_dir / "faiss.index"))

        map_data = {
            "id_list": cache["id_list"],
            "key_set": cache["key_set"],
        }
        with open(user_dir / "index_map.json", "w", encoding="utf-8") as f:
            json.dump(map_data, f)

    def _load_index(self, user_id: str) -> bool:
        """Try to load index from disk. Returns True if successful."""
        user_dir = _DATA_DIR / user_id
        idx_path = user_dir / "faiss.index"
        map_path = user_dir / "index_map.json"

        if not idx_path.exists() or not map_path.exists():
            return False

        try:
            index = faiss.read_index(str(idx_path))
            with open(map_path, "r", encoding="utf-8") as f:
                map_data = json.load(f)

            expenses = load_expenses(user_id)

            self._cache[user_id] = {
                "index": index,
                "id_list": map_data["id_list"],
                "key_set": map_data["key_set"],
                "expenses": expenses,
            }
            return True
        except Exception:
            return False

    def add_to_index(self, user_id: str, expense: ExpenseRecord) -> None:
        """Add a new expense to the user's index (after saving).

        Args:
            user_id: The user.
            expense: The newly saved expense to add.
        """
        cache = self._ensure_index(user_id)

        vec = generate_fuzzy_embedding(expense)
        vec = vec.reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        cache["index"].add(vec)

        cache["id_list"].append(expense.receipt_id)
        cache["key_set"][generate_exact_key(expense)] = expense.receipt_id
        cache["expenses"].append(expense)

        self._persist_index(user_id)

    # ── Duplicate search ──────────────────────────────────────────────────

    def find_duplicates(
        self,
        expense: ExpenseRecord,
        user_id: str,
        threshold: float = 0.12,
    ) -> list[DuplicateMatch]:
        """Search for duplicates of the given expense in the user's history.

        Args:
            expense:   The new (unsaved) ExpenseRecord to check.
            user_id:   Owner user.
            threshold: L2 distance threshold for fuzzy matches (lower = stricter).

        Returns:
            List of DuplicateMatch objects, sorted by similarity descending.
        """
        cache = self._ensure_index(user_id)
        matches: list[DuplicateMatch] = []

        # ── 1. Exact-key check ───────────────────────────────────────────
        new_key = generate_exact_key(expense)
        if new_key in cache["key_set"]:
            matches.append(DuplicateMatch(
                matched_expense_id=cache["key_set"][new_key],
                similarity_score=1.0,
                match_type="exact",
                reason="Identical vendor + date + amount combination",
            ))

        # ── 2. Fuzzy Faiss check ─────────────────────────────────────────
        if cache["index"].ntotal == 0:
            return matches

        query = generate_fuzzy_embedding(expense).reshape(1, -1).copy()
        faiss.normalize_L2(query)

        k = min(5, cache["index"].ntotal)
        distances, indices = cache["index"].search(query, k)

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(cache["id_list"]):
                continue

            candidate_id = cache["id_list"][idx]

            # Skip if already matched as exact
            if any(m.matched_expense_id == candidate_id for m in matches):
                continue

            if dist > threshold:
                continue

            # Find the candidate expense for date/amount guards
            candidate = self._find_expense(cache["expenses"], candidate_id)
            if candidate is None:
                continue

            # Date guard: within 3 days
            if not self._dates_within(expense.date, candidate.date, days=3):
                continue

            # Amount guard: within 5%
            if not self._amounts_within(expense.total_amount, candidate.total_amount, pct=0.05):
                continue

            # Convert L2 distance to similarity (1 - dist/2 for normalized vectors)
            similarity = max(0.0, 1.0 - dist / 2.0)

            matches.append(DuplicateMatch(
                matched_expense_id=candidate_id,
                similarity_score=round(similarity, 4),
                match_type="fuzzy",
                reason=f"Similar receipt (L2={dist:.4f}), date within 3d, amount within 5%",
            ))

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_expense(
        expenses: list[ExpenseRecord], receipt_id: str
    ) -> Optional[ExpenseRecord]:
        for e in expenses:
            if e.receipt_id == receipt_id:
                return e
        return None

    @staticmethod
    def _dates_within(d1: str | None, d2: str | None, days: int) -> bool:
        """Check if two ISO date strings are within N days of each other."""
        if not d1 or not d2:
            return True  # missing date → don't block on this guard
        try:
            dt1 = datetime.fromisoformat(d1)
            dt2 = datetime.fromisoformat(d2)
            return abs(dt1 - dt2) <= timedelta(days=days)
        except ValueError:
            return True

    @staticmethod
    def _amounts_within(a1: float | None, a2: float | None, pct: float) -> bool:
        """Check if two amounts are within pct% of each other."""
        if a1 is None or a2 is None:
            return True
        if a1 == 0 and a2 == 0:
            return True
        ref = max(abs(a1), abs(a2))
        return abs(a1 - a2) / ref <= pct


# Module-level singleton for reuse across the pipeline
detector = DuplicateDetector()
