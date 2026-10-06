"""
Pydantic response schemas for the Expense Intelligence API.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


# ── Shared sub-models ─────────────────────────────────────────────────────────

class LineItemSchema(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None


class TaxEntrySchema(BaseModel):
    rate: Optional[float] = None
    amount: Optional[float] = None


# ── ExpenseRecord response ────────────────────────────────────────────────────

class ExpenseRecordOut(BaseModel):
    user_id: str
    receipt_id: str
    vendor: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    receipt_number: Optional[str] = None
    line_items: list[LineItemSchema] = []
    taxes: list[TaxEntrySchema] = []
    overall_confidence: float = 0.0
    needs_review: bool = False
    flagged: bool = False
    flagged_reason: Optional[str] = None
    source_file: Optional[str] = None
    created_at: str
    claim_status: str = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    rejection_note: Optional[str] = None


# ── Duplicate result ──────────────────────────────────────────────────────────

class DuplicateMatchOut(BaseModel):
    matched_expense_id: str
    similarity_score: float
    match_type: str
    reason: str


class DuplicateResultOut(BaseModel):
    is_duplicate: bool
    action: str           # accept | flag | reject
    reason: str
    duplicate_of: Optional[str] = None
    matches: list[DuplicateMatchOut] = []


# ── Anomaly result ────────────────────────────────────────────────────────────

class AnomalySignalOut(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class AnomalyResultOut(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    explanation: str
    signals: list[AnomalySignalOut] = []


# ── Budget alert ──────────────────────────────────────────────────────────────

class BudgetAlertOut(BaseModel):
    category: str
    alert_level: str
    message: str
    recommended_daily_limit: float


# ── Upload response ───────────────────────────────────────────────────────────

class UploadReceiptResponse(BaseModel):
    expense: ExpenseRecordOut
    duplicate_result: DuplicateResultOut
    anomaly_result: AnomalyResultOut
    early_warnings: list[BudgetAlertOut] = []
    rejected: bool = False
    rejection_reason: Optional[str] = None


# ── Expense list ──────────────────────────────────────────────────────────────

class ExpenseListResponse(BaseModel):
    user_id: str
    count: int
    expenses: list[ExpenseRecordOut]


# ── Dashboard KPI ─────────────────────────────────────────────────────────────

class DashboardResponse(BaseModel):
    user_id: str
    generated_at: str
    period_start: str
    period_end: str
    total_spend_this_month: float
    spend_by_category: dict[str, float]
    top_vendors: list[dict[str, Any]]
    anomalies_this_month: list[dict[str, Any]]
    budget_alerts: list[dict[str, Any]]
    forecast_summary: list[dict[str, Any]]
    month_over_month_change: dict[str, float]
    unusual_transactions: list[dict[str, Any]]


# ── Insight ───────────────────────────────────────────────────────────────────

class InsightResponse(BaseModel):
    insight_text: str
    template_used: str
    grounding_score: float
    grounding_verdict: str
    generated_at: str
    cached: bool


# ── Budget set ────────────────────────────────────────────────────────────────

class SetBudgetRequest(BaseModel):
    category: str
    monthly_limit: float


class SetBudgetResponse(BaseModel):
    user_id: str
    category: str
    monthly_limit: float
    message: str


# ── Anomaly log ───────────────────────────────────────────────────────────────

class AnomalyLogEntry(BaseModel):
    receipt_id: str
    vendor: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    anomaly_score: float
    is_anomaly: bool
    explanation: str
    logged_at: str


class AnomalyLogResponse(BaseModel):
    user_id: str
    count: int
    anomalies: list[AnomalyLogEntry]


# ── Forecast ──────────────────────────────────────────────────────────────────

class ForecastPointOut(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    user_id: str
    category: str
    trend: str
    predicted_monthly_total: float
    days_until_budget_breach: Optional[int] = None
    granularity: str
    forecast_points: list[ForecastPointOut]


# ── Review queue ──────────────────────────────────────────────────────────────

class ReviewQueueItem(BaseModel):
    expense: ExpenseRecordOut
    review_reason: str   # "low_confidence" | "flagged_duplicate"


class ReviewQueueResponse(BaseModel):
    user_id: str
    count: int
    items: list[ReviewQueueItem]


# ── Review action ─────────────────────────────────────────────────────────────

class ReviewActionResponse(BaseModel):
    receipt_id: str
    action: str          # "approved" | "rejected"
    message: str


# ── Explainability ────────────────────────────────────────────────────────────

class ExplanationResponse(BaseModel):
    receipt_id: Optional[str] = None
    explanation_type: str
    features: list[dict[str, Any]]
    summary: str
