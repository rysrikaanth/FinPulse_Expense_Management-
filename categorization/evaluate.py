"""
Model evaluation with 5-fold stratified cross-validation.

Usage::

    python categorization/evaluate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from categorization.categories import MAIN_CATEGORIES
from categorization.classifier import (
    ExpenseCategorizer,
    _build_feature_vector,
    _generate_seed_data,
)
from categorization.embedder import embed_text_array


def _build_dataset() -> tuple[np.ndarray, list[str], list[float]]:
    """Encode all seed examples into feature vectors."""
    seed = _generate_seed_data()
    texts   = [r["text"]     for r in seed]
    labels  = [r["category"] for r in seed]
    amounts = [r["amount"]   for r in seed]

    print(f"  Total seed examples  : {len(seed)}")
    print(f"  Classes              : {len(set(labels))}")
    print(f"  Examples per class   : {len(seed) // len(set(labels))}")

    print("\n  Encoding SBERT embeddings … ", end="", flush=True)
    X = np.array(
        [_build_feature_vector(embed_text_array(t), a, t.split()[0])
         for t, a in zip(texts, amounts)],
        dtype=np.float32,
    )
    print(f"done.  Shape: {X.shape}")
    return X, labels, amounts


def _make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=20,
        min_child_samples=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )


def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        EXPENSE CATEGORIZER — EVALUATION REPORT v2      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    X, labels, _ = _build_dataset()

    le = LabelEncoder()
    le.fit(MAIN_CATEGORIES)
    y = le.transform(labels)

    # ── 1. 5-Fold Stratified Cross-Validation ────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │       5-FOLD STRATIFIED CROSS-VALIDATION            │")
    print("  └─────────────────────────────────────────────────────┘")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        _make_lgbm(), X, y, cv=skf, scoring="accuracy", n_jobs=-1
    )
    print(f"\n  Fold accuracies : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean accuracy   : {cv_scores.mean():.3f}  ±  {cv_scores.std():.3f}")

    cv_f1 = cross_val_score(
        _make_lgbm(), X, y, cv=skf, scoring="f1_macro", n_jobs=-1
    )
    print(f"  Mean macro-F1   : {cv_f1.mean():.3f}  ±  {cv_f1.std():.3f}")

    # ── 2. Held-out 20 % test for per-category report ────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │     PER-CATEGORY REPORT (80/20 stratified split)    │")
    print("  └─────────────────────────────────────────────────────┘\n")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = _make_lgbm()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_test_lbl = le.inverse_transform(y_test)
    y_pred_lbl = le.inverse_transform(y_pred)

    print(classification_report(
        y_test_lbl, y_pred_lbl,
        labels=MAIN_CATEGORIES,
        zero_division=0,
    ))

    # ── 3. Confusion matrix ──────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │               CONFUSION MATRIX                      │")
    print("  └─────────────────────────────────────────────────────┘\n")

    cm = confusion_matrix(y_test_lbl, y_pred_lbl, labels=MAIN_CATEGORIES)
    abbrevs = [c[:5] for c in MAIN_CATEGORIES]
    print("         " + "  ".join(f"{a:>5s}" for a in abbrevs))
    print("         " + "─" * (len(abbrevs) * 7))
    for i, cat in enumerate(MAIN_CATEGORIES):
        row = "  ".join(f"{cm[i][j]:5d}" for j in range(len(MAIN_CATEGORIES)))
        print(f"  {cat:<7s}│ {row}")

    # ── 4. Retrain on full data & save ───────────────────────────────────
    print("\n  Retraining on full dataset and saving … ", end="", flush=True)
    categorizer = ExpenseCategorizer()
    categorizer.train()
    model_path = categorizer.save_model()
    print(f"done.\n  Model saved → {model_path}")
    print(f"\n  Final feature dim    : {X.shape[1]}  (384 SBERT + 3 scalars)")
    print(f"  Total training rows  : {len(labels)}\n")


if __name__ == "__main__":
    main()
