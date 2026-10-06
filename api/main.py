"""
FastAPI application — Expense Intelligence API.

Endpoints:
  POST /receipts/upload
  GET  /users/{user_id}/expenses
  GET  /users/{user_id}/dashboard
  GET  /users/{user_id}/insights/{type}
  POST /users/{user_id}/budgets
  GET  /users/{user_id}/anomalies
  GET  /users/{user_id}/forecast/{category}
  GET  /users/{user_id}/review_queue
  POST /users/{user_id}/review/{expense_id}
  GET  /users/{user_id}/explain/anomaly/{expense_id}
  GET  /users/{user_id}/explain/category/{expense_id}

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    UploadReceiptResponse, ExpenseListResponse, DashboardResponse,
    InsightResponse, SetBudgetRequest, SetBudgetResponse,
    AnomalyLogResponse, AnomalyLogEntry, ForecastResponse, ForecastPointOut,
    ReviewQueueResponse, ReviewQueueItem, ReviewActionResponse,
    ExpenseRecordOut, DuplicateResultOut, DuplicateMatchOut,
    AnomalyResultOut, AnomalySignalOut, BudgetAlertOut, ExplanationResponse,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Expense Intelligence API",
    description="User-centric expense tracking, anomaly detection, and AI insights",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: convert internal record to schema ──────────────────────────────────

def _record_to_out(record) -> ExpenseRecordOut:
    return ExpenseRecordOut(
        user_id=record.user_id,
        receipt_id=record.receipt_id,
        vendor=record.vendor,
        date=record.date,
        time=record.time,
        total_amount=record.total_amount,
        currency=record.currency,
        category=record.category,
        subcategory=record.subcategory,
        receipt_number=record.receipt_number,
        line_items=[li.model_dump() for li in record.line_items],
        taxes=[t.model_dump() for t in record.taxes],
        overall_confidence=record.overall_confidence,
        needs_review=record.needs_review,
        flagged=record.flagged,
        flagged_reason=record.flagged_reason,
        source_file=record.source_file,
        created_at=record.created_at,
        claim_status=getattr(record, 'claim_status', 'pending'),
        reviewed_by=getattr(record, 'reviewed_by', None),
        reviewed_at=getattr(record, 'reviewed_at', None),
        rejection_note=getattr(record, 'rejection_note', None),
    )


# ── POST /receipts/upload ──────────────────────────────────────────────────────

@app.post("/receipts/upload", response_model=UploadReceiptResponse, tags=["Receipts"])
async def upload_receipt(
    file: UploadFile = File(..., description="Receipt image (JPG/PNG/PDF)"),
    user_id: str = Form(..., description="Owner user ID"),
):
    """Upload a receipt image and run the full ingestion pipeline.

    Pipeline: OCR → normalize → dedup → store → anomaly → budget alerts.
    Returns the full result including anomaly scores and budget warnings.
    """
    # Save upload to a temp file
    suffix = Path(file.filename or "receipt.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from ocr_extractor import MindeeFullParser
        from pipeline.normalizer import normalize_mindee_json
        from storage.expense_store import save_expense
        from deduplication.resolver import resolve_duplicates
        from deduplication.detector import detector as dedup_detector
        from anomaly.ensemble import AnomalyEnsemble, _log_anomaly
        from anomaly.isolation_forest_model import PerUserIsolationForest
        from forecasting.early_warning import run_early_warning

        # 1. OCR
        parser = MindeeFullParser()
        raw_json = parser.get_full_json(tmp_path)
        if "error" in raw_json:
            raise HTTPException(status_code=422, detail=f"OCR error: {raw_json['error']}")

        # 2. Normalize
        record = normalize_mindee_json(raw_json, user_id, file.filename or "upload")

        # 3. Dedup check
        from deduplication.detector import detector
        duplication_matches = detector.find_duplicates(record, user_id)
        from deduplication.resolver import resolve_duplicates
        resolution = resolve_duplicates(record, user_id)

        rejected = resolution.action == "reject"
        if rejected:
            dup_out = DuplicateResultOut(
                is_duplicate=True,
                action="reject",
                reason=resolution.reason,
                duplicate_of=resolution.duplicate_of,
                matches=[DuplicateMatchOut(**m.__dict__) for m in duplication_matches],
            )
            # Return without saving
            return UploadReceiptResponse(
                expense=_record_to_out(record),
                duplicate_result=dup_out,
                anomaly_result=AnomalyResultOut(
                    anomaly_score=0.0, is_anomaly=False, explanation="Rejected as duplicate"
                ),
                early_warnings=[],
                rejected=True,
                rejection_reason=resolution.reason,
            )

        if resolution.action == "flag":
            record.flagged = True
            record.flagged_reason = resolution.reason

        # 4. Save
        save_expense(record)
        dedup_detector.add_to_index(user_id, record)

        # 5. Anomaly
        if_model = PerUserIsolationForest()
        if_model.fit_if_needed(user_id)

        ensemble = AnomalyEnsemble()
        anomaly_result = ensemble.score(record, user_id)

        if anomaly_result.is_anomaly:
            _log_anomaly(user_id, record, anomaly_result)

        # 6. Budget alerts
        early_warning_alerts = run_early_warning(user_id)

        # Build response
        dup_out = DuplicateResultOut(
            is_duplicate=resolution.is_duplicate,
            action=resolution.action,
            reason=resolution.reason,
            duplicate_of=resolution.duplicate_of,
            matches=[DuplicateMatchOut(**m.__dict__) for m in duplication_matches],
        )

        anom_out = AnomalyResultOut(
            anomaly_score=anomaly_result.anomaly_score,
            is_anomaly=anomaly_result.is_anomaly,
            explanation=anomaly_result.explanation,
            signals=[AnomalySignalOut(**s) for s in anomaly_result.signals],
        )

        alerts_out = [
            BudgetAlertOut(
                category=a.category,
                alert_level=a.alert_level,
                message=a.message,
                recommended_daily_limit=a.recommended_daily_limit,
            )
            for a in early_warning_alerts
        ]

        return UploadReceiptResponse(
            expense=_record_to_out(record),
            duplicate_result=dup_out,
            anomaly_result=anom_out,
            early_warnings=alerts_out,
            rejected=False,
        )

    finally:
        os.unlink(tmp_path)


# ── GET /users/{user_id}/expenses ─────────────────────────────────────────────

@app.get("/users/{user_id}/expenses", response_model=ExpenseListResponse, tags=["Expenses"])
def list_expenses(
    user_id: str,
    category: Optional[str] = Query(None, description="Filter by category"),
    date_from: Optional[str] = Query(None, description="Filter from date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="Filter to date YYYY-MM-DD"),
    limit: int = Query(100, ge=1, le=1000),
):
    """List all expenses for a user with optional filters."""
    from storage.expense_store import load_expenses

    records = load_expenses(user_id)

    if category:
        records = [r for r in records if (r.category or "").lower() == category.lower()]

    if date_from:
        records = [
            r for r in records
            if r.date and r.date >= date_from
        ]

    if date_to:
        records = [
            r for r in records
            if r.date and r.date <= date_to
        ]

    # Sort newest first
    records.sort(key=lambda r: r.created_at, reverse=True)
    records = records[:limit]

    return ExpenseListResponse(
        user_id=user_id,
        count=len(records),
        expenses=[_record_to_out(r) for r in records],
    )


# ── GET /users/{user_id}/dashboard ───────────────────────────────────────────

@app.get("/users/{user_id}/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def get_dashboard(user_id: str):
    """Return full KPI context for the dashboard."""
    from insights.kpi_builder import build_kpi_context

    ctx = build_kpi_context(user_id)
    return DashboardResponse(
        user_id=ctx.user_id,
        generated_at=ctx.generated_at,
        period_start=ctx.period_start,
        period_end=ctx.period_end,
        total_spend_this_month=ctx.total_spend_this_month,
        spend_by_category=ctx.spend_by_category,
        top_vendors=ctx.top_vendors,
        anomalies_this_month=ctx.anomalies_this_month,
        budget_alerts=ctx.budget_alerts,
        forecast_summary=ctx.forecast_summary,
        month_over_month_change=ctx.month_over_month_change,
        unusual_transactions=ctx.unusual_transactions,
    )


# ── GET /users/{user_id}/insights/{type} ─────────────────────────────────────

@app.get("/users/{user_id}/insights/{insight_type}", response_model=InsightResponse, tags=["Insights"])
def get_insight(
    user_id: str,
    insight_type: str,
    vendor: Optional[str] = Query(None),
    anomaly_id: Optional[str] = Query(None),
):
    """Generate a grounded LLM insight. Cached for 24 hours."""
    valid_types = {"weekly_summary", "anomaly_explanation", "budget_advice",
                   "vendor_insight", "forecast_alert"}
    if insight_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"insight_type must be one of {valid_types}",
        )

    extra: dict = {}
    if insight_type == "vendor_insight":
        if not vendor:
            raise HTTPException(status_code=400, detail="vendor param required for vendor_insight")
        extra = {"vendor": vendor, "total": 0, "count": 0}
        # Populate from actual data
        from storage.expense_store import load_expenses
        now = datetime.now()
        for exp in load_expenses(user_id):
            if (exp.vendor or "").lower() == vendor.lower() and exp.date:
                try:
                    d = datetime.fromisoformat(exp.date)
                    if d.year == now.year and d.month == now.month:
                        extra["total"] = extra["total"] + (exp.total_amount or 0)
                        extra["count"] = extra["count"] + 1
                except ValueError:
                    pass

    elif insight_type == "anomaly_explanation":
        if not anomaly_id:
            raise HTTPException(status_code=400, detail="anomaly_id param required")
        log_path = _DATA_DIR / user_id / "anomaly_log.jsonl"
        extra = {}
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("receipt_id") == anomaly_id:
                            extra = {
                                "vendor": entry.get("vendor", ""),
                                "date": entry.get("date", ""),
                                "amount": entry.get("amount", 0),
                                "category": entry.get("category", ""),
                                "anomaly_score": entry.get("anomaly_score", 0),
                                "system_explanation": entry.get("explanation", ""),
                            }
                            break
                    except json.JSONDecodeError:
                        continue

    try:
        from insights.llm_client import GeminiInsightClient
        client = GeminiInsightClient()
        result = client.generate_insight(
            user_id=user_id,
            template_name=insight_type,
            extra_context=extra or None,
        )
        return InsightResponse(
            insight_text=result.insight_text,
            template_used=result.template_used,
            grounding_score=result.grounding_score,
            grounding_verdict=result.grounding_verdict,
            generated_at=result.generated_at,
            cached=result.cached,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {e}")


# ── POST /users/{user_id}/budgets ─────────────────────────────────────────────

@app.post("/users/{user_id}/budgets", response_model=SetBudgetResponse, tags=["Budgets"])
def set_budget(user_id: str, request: SetBudgetRequest):
    """Set or update a monthly budget for a category."""
    from forecasting.budget_manager import set_budget as _set_budget

    _set_budget(user_id, request.category, request.monthly_limit)
    return SetBudgetResponse(
        user_id=user_id,
        category=request.category.lower(),
        monthly_limit=request.monthly_limit,
        message=f"Budget set: ₹{request.monthly_limit:,.0f}/month for {request.category}",
    )


# ── GET /users/{user_id}/anomalies ────────────────────────────────────────────

@app.get("/users/{user_id}/anomalies", response_model=AnomalyLogResponse, tags=["Anomalies"])
def list_anomalies(
    user_id: str,
    limit: int = Query(50, ge=1, le=500),
):
    """Return the anomaly log for a user, sorted by score descending."""
    log_path = _DATA_DIR / user_id / "anomaly_log.jsonl"
    entries: list[AnomalyLogEntry] = []

    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(AnomalyLogEntry(
                        receipt_id=data.get("receipt_id", ""),
                        vendor=data.get("vendor"),
                        date=data.get("date"),
                        amount=data.get("amount"),
                        category=data.get("category"),
                        anomaly_score=data.get("anomaly_score", 0),
                        is_anomaly=data.get("is_anomaly", False),
                        explanation=data.get("explanation", ""),
                        logged_at=data.get("logged_at", ""),
                    ))
                except Exception:
                    continue

    entries.sort(key=lambda e: e.anomaly_score, reverse=True)
    entries = entries[:limit]

    return AnomalyLogResponse(user_id=user_id, count=len(entries), anomalies=entries)


# ── GET /users/{user_id}/forecast/{category} ──────────────────────────────────

@app.get("/users/{user_id}/forecast/{category}", response_model=ForecastResponse, tags=["Forecasts"])
def get_forecast(
    user_id: str,
    category: str,
    days: int = Query(30, ge=7, le=90),
):
    """Get a 30-day Prophet forecast for a specific category."""
    from forecasting.prophet_model import PerUserProphetForecaster
    from forecasting.budget_manager import get_budget

    forecaster = PerUserProphetForecaster()
    budget = get_budget(user_id, category)

    try:
        result = forecaster.forecast(user_id, category, days=days, monthly_budget=budget)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    if result is None:
        # Try training first
        ok = forecaster.train(user_id, category)
        if not ok:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient data to forecast '{category}' — need at least 4 weeks",
            )
        result = forecaster.forecast(user_id, category, days=days, monthly_budget=budget)
        if result is None:
            raise HTTPException(status_code=422, detail="Forecast unavailable after training")

    # Convert forecast_df to points
    df = result.forecast_df
    points = []
    for _, row in df.iterrows():
        points.append(ForecastPointOut(
            ds=str(row["ds"])[:10],
            yhat=max(0, float(row["yhat"])),
            yhat_lower=max(0, float(row.get("yhat_lower", row["yhat"]))),
            yhat_upper=max(0, float(row.get("yhat_upper", row["yhat"]))),
        ))

    return ForecastResponse(
        user_id=user_id,
        category=category,
        trend=result.trend,
        predicted_monthly_total=result.predicted_monthly_total,
        days_until_budget_breach=result.days_until_budget_breach,
        granularity=result.granularity,
        forecast_points=points,
    )


# ── GET /users/{user_id}/review_queue ─────────────────────────────────────────

@app.get("/users/{user_id}/review_queue", response_model=ReviewQueueResponse, tags=["Review"])
def get_review_queue(user_id: str):
    """Return all expenses flagged for human review."""
    from storage.expense_store import load_expenses

    records = load_expenses(user_id)
    items = []

    for r in records:
        if r.needs_review:
            items.append(ReviewQueueItem(
                expense=_record_to_out(r),
                review_reason="low_confidence",
            ))
        elif r.flagged:
            items.append(ReviewQueueItem(
                expense=_record_to_out(r),
                review_reason="flagged_duplicate",
            ))

    return ReviewQueueResponse(user_id=user_id, count=len(items), items=items)


# ── POST /users/{user_id}/review/{expense_id} ────────────────────────────────

@app.post("/users/{user_id}/review/{expense_id}", response_model=ReviewActionResponse, tags=["Review"])
def review_expense(
    user_id: str,
    expense_id: str,
    action: str = Query(..., description="'approve' or 'reject'"),
):
    """Human review: approve or reject a flagged/low-confidence expense.

    - approve: clears needs_review + flagged flags
    - reject: deletes the expense from the store
    """
    if action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    from storage.expense_store import load_expenses, save_expense
    import shutil

    records = load_expenses(user_id)
    target_idx = next((i for i, r in enumerate(records) if r.receipt_id == expense_id), None)

    if target_idx is None:
        raise HTTPException(status_code=404, detail=f"Expense {expense_id} not found")

    jsonl_path = _DATA_DIR / user_id / "expenses.jsonl"

    if action == "approve":
        records[target_idx].needs_review = False
        records[target_idx].flagged = False
        records[target_idx].flagged_reason = None
        # Rewrite the file
        tmp = jsonl_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
        shutil.move(str(tmp), str(jsonl_path))
        return ReviewActionResponse(
            receipt_id=expense_id,
            action="approved",
            message="Expense approved and flags cleared",
        )

    else:  # reject
        records_filtered = [r for r in records if r.receipt_id != expense_id]
        tmp = jsonl_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records_filtered:
                f.write(r.model_dump_json() + "\n")
        shutil.move(str(tmp), str(jsonl_path))
        return ReviewActionResponse(
            receipt_id=expense_id,
            action="rejected",
            message="Expense removed from store",
        )


# ── GET /users/{user_id}/explain/anomaly/{expense_id} ────────────────────────

@app.get("/users/{user_id}/explain/anomaly/{expense_id}",
         response_model=ExplanationResponse, tags=["Explainability"])
def explain_anomaly_endpoint(user_id: str, expense_id: str):
    """Return SHAP/signal-based anomaly explanation for an expense."""
    from storage.expense_store import get_expense_by_id
    from api.explainability import explain_anomaly

    expense = get_expense_by_id(user_id, expense_id)
    if expense is None:
        raise HTTPException(status_code=404, detail="Expense not found")

    features = explain_anomaly(expense, user_id)
    top = features[0] if features else {}
    return ExplanationResponse(
        receipt_id=expense_id,
        explanation_type="anomaly",
        features=features,
        summary=f"Top driver: {top.get('feature','')} — {top.get('description','')}",
    )


# ── GET /users/{user_id}/explain/category/{expense_id} ───────────────────────

@app.get("/users/{user_id}/explain/category/{expense_id}",
         response_model=ExplanationResponse, tags=["Explainability"])
def explain_category_endpoint(user_id: str, expense_id: str):
    """Return SHAP-based categorization explanation for an expense."""
    from storage.expense_store import get_expense_by_id
    from api.explainability import explain_categorization

    expense = get_expense_by_id(user_id, expense_id)
    if expense is None:
        raise HTTPException(status_code=404, detail="Expense not found")

    features = explain_categorization(expense)
    top = features[0] if features else {}
    return ExplanationResponse(
        receipt_id=expense_id,
        explanation_type="categorization",
        features=features,
        summary=f"Top driver: {top.get('feature','')} — {top.get('description','')}",
    )


# ── GET /admin/users ──────────────────────────────────────────────────────────

@app.get("/admin/users", tags=["Admin"])
def list_users():
    """Return all registered users (admin only — enforced in dashboard)."""
    from auth.users import get_all_users
    return {"users": get_all_users()}


# ── GET /admin/pending_claims ─────────────────────────────────────────────────

@app.get("/admin/pending_claims", tags=["Admin"])
def get_all_pending_claims(
    status: str = Query("pending", description="pending | accepted | rejected | all"),
    limit: int = Query(200, ge=1, le=1000),
):
    """Aggregate expense claims across all users for admin review."""
    from auth.users import get_regular_users
    from storage.expense_store import load_expenses

    all_records = []
    for user in get_regular_users():
        uid = user["user_id"]
        for r in load_expenses(uid):
            cs = getattr(r, "claim_status", "pending")
            if status == "all" or cs == status:
                d = r.model_dump()
                d["display_name"] = user["display_name"]
                all_records.append(d)

    # Sort by date desc
    all_records.sort(key=lambda x: x.get("date") or "", reverse=True)
    return {"count": len(all_records), "claims": all_records[:limit]}


# ── POST /users/{user_id}/claims/{receipt_id}/status ─────────────────────────

@app.post("/users/{user_id}/claims/{receipt_id}/status", tags=["Admin"])
def update_claim_status(
    user_id: str,
    receipt_id: str,
    action: str = Query(..., description="'accept' or 'reject'"),
    reviewer: str = Query("admin", description="Admin username performing the review"),
    note: Optional[str] = Query(None, description="Rejection note (optional)"),
):
    """Admin accepts or rejects a pending claim."""
    import shutil
    from storage.expense_store import load_expenses

    if action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'accept' or 'reject'")

    records = load_expenses(user_id)
    idx = next((i for i, r in enumerate(records) if r.receipt_id == receipt_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Claim not found")

    records[idx].claim_status = "accepted" if action == "accept" else "rejected"
    records[idx].reviewed_by = reviewer
    records[idx].reviewed_at = datetime.now(timezone.utc).isoformat()
    if action == "reject" and note:
        records[idx].rejection_note = note

    jsonl_path = _DATA_DIR / user_id / "expenses.jsonl"
    tmp = jsonl_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    shutil.move(str(tmp), str(jsonl_path))

    return {
        "receipt_id": receipt_id,
        "new_status": records[idx].claim_status,
        "reviewed_by": reviewer,
        "message": f"Claim {records[idx].claim_status} by {reviewer}",
    }


# ── GET /users/{user_id}/claims ───────────────────────────────────────────────

@app.get("/users/{user_id}/claims", tags=["Claims"])
def get_user_claims(
    user_id: str,
    status: str = Query("all", description="all | pending | accepted | rejected"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    """Return a user's own expense claims with their review status."""
    from storage.expense_store import load_expenses

    records = load_expenses(user_id)

    if status != "all":
        records = [r for r in records if getattr(r, 'claim_status', 'pending') == status]
    if date_from:
        records = [r for r in records if r.date and r.date >= date_from]
    if date_to:
        records = [r for r in records if r.date and r.date <= date_to]

    records.sort(key=lambda r: r.date or "", reverse=True)
    records = records[:limit]

    return {
        "user_id": user_id,
        "count": len(records),
        "claims": [_record_to_out(r).model_dump() for r in records],
    }


# ── Root health check ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "Expense Intelligence API", "version": "1.0.0"}
