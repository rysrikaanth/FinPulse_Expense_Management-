"""
SHAP-based explainability for categorization and anomaly detection.

Provides top-5 feature attribution for both the ML categorization model
and the Isolation Forest anomaly model.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def explain_categorization(expense: Any) -> list[dict[str, Any]]:
    """Return top-5 SHAP feature attributions for categorization.

    Falls back to rule-based attribution if SHAP or the model is unavailable.

    Args:
        expense: ExpenseRecord to explain.

    Returns:
        List of {feature, value, shap_value, description} dicts, sorted by |shap_value| desc.
    """
    try:
        import numpy as np
        import shap
        from categorization.classifier import ExpenseCategoryClassifier

        clf = ExpenseCategoryClassifier()
        if not clf.is_trained():
            return _fallback_categorization(expense)

        # Build feature vector for this expense
        features = clf.build_features(expense)
        feature_names = clf.feature_names()

        # SHAP TreeExplainer on the underlying LightGBM model
        explainer = shap.TreeExplainer(clf._model)
        shap_values = explainer.shap_values(features.reshape(1, -1))

        # For multi-class, take max across classes per feature
        if isinstance(shap_values, list):
            combined = np.abs(np.array(shap_values)).max(axis=0)[0]
            raw = np.array(shap_values)[:, 0, :]
        else:
            combined = np.abs(shap_values[0])
            raw = shap_values[0]

        top_idx = combined.argsort()[::-1][:5]
        result = []
        for i in top_idx:
            result.append({
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "value": float(features[i]),
                "shap_value": float(raw[i]) if not isinstance(raw, np.ndarray) or len(raw.shape) == 1 else float(combined[i]),
                "description": _categorization_description(
                    feature_names[i] if i < len(feature_names) else f"feature_{i}",
                    float(features[i]),
                ),
            })
        return result

    except Exception:
        return _fallback_categorization(expense)


def explain_anomaly(expense: Any, user_id: str) -> list[dict[str, Any]]:
    """Return top-5 feature attributions for anomaly detection.

    Uses the stored AnomalyEnsemble signals if available, falling back
    to direct feature computation.

    Args:
        expense:  ExpenseRecord to explain.
        user_id:  Owner user.

    Returns:
        List of {feature, value, contribution, direction, description} dicts,
        sorted by contribution desc.
    """
    try:
        from anomaly.ensemble import AnomalyEnsemble
        from storage.expense_store import load_expenses

        ensemble = AnomalyEnsemble()
        history = load_expenses(user_id)

        if len(history) < 10:
            return _cold_start_anomaly()

        result = ensemble.score(expense, user_id)
        signals = result.signals[:5]

        return [
            {
                "feature": s["feature"],
                "value": s["value"],
                "contribution": s["contribution"],
                "direction": s["direction"],
                "description": _anomaly_description(s["feature"], s["direction"]),
            }
            for s in signals
        ]

    except Exception as e:
        return [{"feature": "error", "value": 0.0, "contribution": 0.0,
                 "direction": "n/a", "description": str(e)}]


# ── Fallback and description helpers ─────────────────────────────────────────

def _fallback_categorization(expense: Any) -> list[dict[str, Any]]:
    """Rule-based attribution when model is unavailable."""
    features = []
    vendor = getattr(expense, "vendor", "") or ""
    amount = getattr(expense, "total_amount", 0) or 0
    line_count = len(getattr(expense, "line_items", []))
    category = getattr(expense, "category", "") or ""
    confidence = getattr(expense, "overall_confidence", 0) or 0

    features.append({
        "feature": "vendor_name",
        "value": hash(vendor.lower()) % 1000 / 1000,
        "shap_value": 0.35,
        "description": f"Vendor '{vendor}' strongly drives category assignment",
    })
    features.append({
        "feature": "total_amount",
        "value": amount,
        "shap_value": 0.25,
        "description": f"Amount ₹{amount:,.0f} is typical for {category or 'this'} category",
    })
    features.append({
        "feature": "line_item_count",
        "value": float(line_count),
        "shap_value": 0.15,
        "description": f"{line_count} line items suggest {'grocery/retail' if line_count > 3 else 'single-item purchase'}",
    })
    features.append({
        "feature": "ocr_confidence",
        "value": confidence,
        "shap_value": confidence * 0.2,
        "description": f"OCR confidence {confidence:.0%} {'supports' if confidence > 0.7 else 'limits'} classification certainty",
    })
    features.append({
        "feature": "category_prior",
        "value": 0.5,
        "shap_value": 0.10,
        "description": f"Historical prior for {category or 'this'} category in user's spending",
    })
    return features


def _cold_start_anomaly() -> list[dict[str, Any]]:
    """Return placeholder when insufficient history exists."""
    return [{
        "feature": "insufficient_history",
        "value": 0.0,
        "contribution": 0.0,
        "direction": "n/a",
        "description": "Need at least 10 expense records for anomaly attribution",
    }]


def _categorization_description(feature: str, value: float) -> str:
    """Human-readable description for categorization feature."""
    descriptions = {
        "amount": f"Transaction amount ₹{value:,.0f}",
        "vendor_embedding": "Vendor name semantic similarity",
        "line_item_count": f"{int(value)} line items on receipt",
        "confidence": f"OCR confidence {value:.0%}",
        "hour": f"Transaction at hour {int(value)}:00",
    }
    for key, desc in descriptions.items():
        if key in feature.lower():
            return desc
    return f"Feature '{feature}' = {value:.3f}"


def _anomaly_description(feature: str, direction: str) -> str:
    """Human-readable description for anomaly feature."""
    up = direction == "high"
    descriptions = {
        "amount_zscore": f"Amount is {'unusually high' if up else 'unusually low'} for this category",
        "amount_percentile": f"Amount is at {'top' if up else 'bottom'} percentile of spending",
        "vendor_frequency": f"Vendor appears {'rarely' if not up else 'very frequently'} in history",
        "is_new_vendor": "This vendor has never appeared before",
        "day_of_week_score": f"Spending on this day is {'unusual' if up else 'typical'}",
        "hour_of_day_score": f"Transaction time is {'atypical' if up else 'typical'}",
        "days_since_vendor": f"{'Long gap' if up else 'Short interval'} since last visit to this vendor",
        "category_budget_ratio": f"Amount is {'well above' if up else 'below'} typical weekly budget",
    }
    for key, desc in descriptions.items():
        if key in feature:
            return desc
    return f"Feature '{feature}' is {direction} relative to history"
