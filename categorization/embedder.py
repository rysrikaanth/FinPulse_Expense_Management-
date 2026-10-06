"""
Sentence-BERT embedding layer for the categorization pipeline.

Uses the lightweight ``all-MiniLM-L6-v2`` model (384-dim, ~80 MB) to
produce dense vectors from vendor names and line-item descriptions.
Embeddings are cached in-memory to avoid redundant forward passes.
"""

from __future__ import annotations

import sys
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if TYPE_CHECKING:
    from models.expense import ExpenseRecord

# ── Lazy model loading ───────────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ── Public API ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def embed_text(text: str) -> tuple:
    """Embed a single text string using SBERT.

    Results are cached via ``lru_cache`` (stored as tuple for hashability).

    Args:
        text: The text to embed.

    Returns:
        Tuple of floats (384-dim embedding). Convert to ndarray via
        ``np.array(result)`` if needed.
    """
    model = _get_model()
    vec = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return tuple(vec.tolist())


def embed_text_array(text: str) -> np.ndarray:
    """Convenience wrapper — returns a numpy array instead of a tuple.

    Args:
        text: The text to embed.

    Returns:
        np.ndarray of shape (384,).
    """
    return np.array(embed_text(text), dtype=np.float32)


def embed_expense(expense: "ExpenseRecord") -> np.ndarray:
    """Build a composite embedding for an expense record.

    Combines the vendor name and all line-item descriptions into a single
    string, then encodes it.  If both are missing, returns a zero vector.

    Args:
        expense: A validated ExpenseRecord.

    Returns:
        np.ndarray of shape (384,).
    """
    parts: list[str] = []

    if expense.vendor:
        parts.append(expense.vendor)

    for item in expense.line_items:
        if item.description:
            parts.append(item.description)

    combined = " ".join(parts).strip()

    if not combined:
        return np.zeros(384, dtype=np.float32)

    return embed_text_array(combined)


def get_embedding_dim() -> int:
    """Return the dimensionality of the embedding model (384 for MiniLM)."""
    return 384
