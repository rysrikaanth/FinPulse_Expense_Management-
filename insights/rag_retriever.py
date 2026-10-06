"""
RAG retriever — stores and retrieves past monthly expense summaries.

Uses SBERT embeddings + cosine similarity to find the most relevant
historical expense snapshots when generating a new insight.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from insights.kpi_builder import UserKPIContext, to_prompt_string

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# Max tokens worth of historical context to inject
_MAX_HISTORICAL_CHARS = 4800  # ≈ 1200 tokens


@dataclass
class InsightSummary:
    """A stored snapshot used for RAG retrieval."""

    period_start: str
    period_end: str
    summary_text: str
    embedding: list[float]
    created_at: str


class RAGRetriever:
    """Per-user vector store of past monthly expense summaries.

    Stores summaries as JSONL with SBERT embeddings.  Retrieves the
    top-k most similar historical periods via cosine similarity.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.store_path = _DATA_DIR / user_id / "insight_store.jsonl"
        self._summaries: list[InsightSummary] = []
        self._embedder = None  # lazy load
        self._load()

    def _get_embedder(self):
        """Lazy-load SBERT model to avoid import cost when not needed."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedder = None
        return self._embedder

    def _load(self) -> None:
        """Load existing summaries from JSONL on disk."""
        if not self.store_path.exists():
            self._summaries = []
            return

        summaries: list[InsightSummary] = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    summaries.append(InsightSummary(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        self._summaries = summaries

    def _save(self) -> None:
        """Persist all summaries to JSONL."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for summary in self._summaries:
                f.write(json.dumps(asdict(summary)) + "\n")

    def add_summary(self, context: UserKPIContext) -> None:
        """Store a new monthly snapshot for future RAG retrieval.

        Args:
            context: The current KPI snapshot to embed and persist.
        """
        summary_text = to_prompt_string(context)
        embedder = self._get_embedder()
        embedding = embedder.encode(summary_text).tolist() if embedder is not None else []

        summary = InsightSummary(
            period_start=context.period_start,
            period_end=context.period_end,
            summary_text=summary_text,
            embedding=embedding,
            created_at=datetime.now().isoformat(),
        )
        self._summaries.append(summary)
        self._save()

    def retrieve_similar_periods(
        self,
        query: str,
        k: int = 3,
    ) -> list[InsightSummary]:
        """Find the k most similar historical periods to a query.

        Args:
            query: Natural language query (e.g. template name + date).
            k:     Number of results to return.

        Returns:
            Up to k InsightSummary objects sorted by cosine similarity.
        """
        if not self._summaries:
            return []

        embedder = self._get_embedder()
        if embedder is None:
            return self._summaries[-k:]

        q_emb = np.array(embedder.encode(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []

        scored: list[tuple[float, InsightSummary]] = []
        for summary in self._summaries:
            s_emb = np.array(summary.embedding, dtype=np.float32)
            s_norm = np.linalg.norm(s_emb)
            if s_norm == 0:
                continue
            cos_sim = float(np.dot(q_emb, s_emb) / (q_norm * s_norm))
            scored.append((cos_sim, summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def format_retrieved_context(
        self,
        summaries: list[InsightSummary],
    ) -> str:
        """Format retrieved summaries for LLM prompt injection.

        Concatenates up to ~1200 tokens of historical context,
        truncating older periods if the combined text is too long.

        Args:
            summaries: Results from retrieve_similar_periods().

        Returns:
            Formatted string ready for prompt prepending.
        """
        if not summaries:
            return ""

        parts: list[str] = ["=== HISTORICAL CONTEXT ===\n"]
        total_chars = len(parts[0])

        for summary in summaries:
            block = (
                f"Period {summary.period_start} to {summary.period_end}:\n"
                f"{summary.summary_text}\n---\n"
            )
            if total_chars + len(block) > _MAX_HISTORICAL_CHARS:
                # Truncate: add as much as fits
                remaining = _MAX_HISTORICAL_CHARS - total_chars
                if remaining > 100:
                    parts.append(block[:remaining] + "\n[truncated]")
                break
            parts.append(block)
            total_chars += len(block)

        return "\n".join(parts)
