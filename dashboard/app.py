"""
Expense Intelligence Dashboard — with Auth & Role-Based Views

Roles:
  admin — full dashboard: Overview, Anomalies, Forecasts, Review Queue,
           + cross-user Claim Management
  user  — My Claims view only: past expenses with status, new submission

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Expense Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Login card */
  .login-card {
    max-width: 400px; margin: 80px auto;
    background: linear-gradient(135deg,#1e1e2e,#2a2a3e);
    border: 1px solid #3d3d5c; border-radius: 16px; padding: 40px;
  }
  .login-title { color:#c5b4e3; font-size:1.6rem; font-weight:700;
                 text-align:center; margin-bottom:8px; }
  .login-sub   { color:#888; font-size:0.85rem; text-align:center;
                 margin-bottom:28px; }

  /* KPI cards */
  .kpi-card {
    background:linear-gradient(135deg,#1e1e2e,#2a2a3e);
    border:1px solid #3d3d5c; border-radius:12px;
    padding:20px; text-align:center; margin-bottom:10px;
  }
  .kpi-label { color:#a0a0b8; font-size:0.72rem; text-transform:uppercase;
               letter-spacing:1px; }
  .kpi-value { color:#fff; font-size:1.8rem; font-weight:700; margin:4px 0; }
  .kpi-sub   { color:#777; font-size:0.78rem; }

  /* Status badges */
  .badge-accepted { background:#00cc4420; color:#51cf66;
    border:1px solid #00cc4440; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-weight:600; }
  .badge-pending  { background:#ff8c0020; color:#ff9944;
    border:1px solid #ff8c0040; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-weight:600; }
  .badge-rejected { background:#ff4d4d20; color:#ff6b6b;
    border:1px solid #ff4d4d40; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-weight:600; }

  /* Section headers */
  .section-header {
    color:#c5b4e3; font-size:1.05rem; font-weight:600;
    padding:8px 0 4px; border-bottom:1px solid #2a2a3e; margin-bottom:12px;
  }

  /* Vendor row */
  .vendor-row {
    display:flex; justify-content:space-between;
    padding:8px 0; border-bottom:1px solid #2a2a3e; color:#d0d0e8;
  }
  .vendor-name  { font-weight:500; }
  .vendor-amount{ color:#c5b4e3; font-weight:600; }

  /* Claim row card */
  .claim-card {
    background:#1e1e2e; border:1px solid #3d3d5c;
    border-radius:10px; padding:14px 18px; margin-bottom:10px;
  }
  .claim-vendor { font-weight:600; color:#d0d0e8; font-size:1rem; }
  .claim-meta   { color:#888; font-size:0.82rem; margin-top:2px; }
  .claim-note   { color:#ff9944; font-size:0.8rem; margin-top:4px; }

  /* Insight box */
  .insight-box {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid #4a4a7a; border-left:4px solid #7c5cbf;
    border-radius:8px; padding:16px; color:#d0d0e8;
    font-size:0.95rem; line-height:1.6;
  }

  /* Admin claim table header */
  .tbl-header { background:#2a2a3e; border-radius:8px;
                padding:8px 12px; color:#a0a0b8;
                font-size:0.78rem; font-weight:600;
                text-transform:uppercase; letter-spacing:0.8px;
                margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, json_body=None, params: dict = None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_body, params=params, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def rupees(v): return f"₹{(v or 0):,.0f}"

def status_badge(s: str) -> str:
    cls = f"badge-{s}"
    icons = {"accepted": "✅", "pending": "⏳", "rejected": "❌"}
    return f'<span class="{cls}">{icons.get(s,"•")} {s.upper()}</span>'


# ── Session auth ──────────────────────────────────────────────────────────────

def get_session():
    return st.session_state.get("auth_user")


def do_login(username: str, password: str) -> bool:
    """Authenticate against auth module directly (no separate auth API needed)."""
    try:
        from auth.users import authenticate, bootstrap
        bootstrap()
        user = authenticate(username, password)
        if user:
            st.session_state["auth_user"] = user
            return True
        return False
    except Exception as e:
        st.error(f"Auth error: {e}")
        return False


def do_logout():
    st.session_state.pop("auth_user", None)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def show_login():
    st.markdown("""
    <div style="text-align:center;margin-top:60px">
      <div style="font-size:3rem">💸</div>
      <h1 style="color:#c5b4e3;font-weight:700;margin:8px 0">Expense Intelligence</h1>
      <p style="color:#666">Sign in to access the platform</p>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 1.2, 1])[1]

    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True,
                                              type="primary")

        if submitted:
            if do_login(username, password):
                st.success("Welcome!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("""
        <div style="margin-top:20px;padding:14px;background:#1e1e2e;border-radius:10px;
                    border:1px solid #3d3d5c;">
          <div style="color:#888;font-size:0.8rem;margin-bottom:8px">
            <b style="color:#c5b4e3">Demo credentials</b>
          </div>
          <div style="color:#aaa;font-size:0.78rem;line-height:1.8">
            🔑 <b>admin</b> / admin@123 &nbsp;(Admin)<br>
            👤 <b>rahul_m</b> / rahul@2024<br>
            👤 <b>priya_s</b> / priya@2024<br>
            👤 <b>arjun_k</b> / arjun@2024<br>
            👤 <b>sneha_r</b> / sneha@2024
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def admin_sidebar(user):
    with st.sidebar:
        st.markdown(f"## 💸 Expense Intelligence")
        st.markdown(f"<div style='color:#51cf66;font-size:0.8rem'>🛡️ Admin · {user['display_name']}</div>",
                    unsafe_allow_html=True)
        st.markdown("---")

        # User selector for admin analytics
        users_data = api_get("/admin/users") or {}
        all_users = [u for u in users_data.get("users", []) if u["role"] == "user"]
        user_ids   = [u["user_id"] for u in all_users]
        user_names = [u["display_name"] for u in all_users]

        sel_idx = st.selectbox("View user data for:", range(len(user_ids)),
                               format_func=lambda i: user_names[i])
        selected_uid = user_ids[sel_idx] if user_ids else "user_rahul"

        st.markdown("---")

        # Budget setter
        st.markdown("### 💰 Set Budget")
        budget_cat = st.selectbox("Category",
            ["food", "transport", "shopping", "entertainment",
             "utilities", "health", "education", "other"])
        budget_amt = st.number_input("Monthly limit (₹)", min_value=500,
                                      max_value=200000, value=15000, step=500)
        if st.button("Set Budget", use_container_width=True):
            r = api_post(f"/users/{selected_uid}/budgets",
                         json_body={"category": budget_cat, "monthly_limit": budget_amt})
            if r:
                st.success(r.get("message", "Budget saved"))

        st.markdown("---")

        page = st.radio("Navigate",
                        ["📊 Overview", "🔴 Anomalies",
                         "📈 Forecasts", "🔎 Review Queue",
                         "📋 Claim Management"],
                        label_visibility="collapsed")

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            do_logout()

    return page, selected_uid


# ══════════════════════════════════════════════════════════════════════════════
# USER SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def user_sidebar(user):
    with st.sidebar:
        st.markdown("## 💸 My Expenses")
        st.markdown(f"<div style='color:#c5b4e3;font-size:0.85rem'>👤 {user['display_name']}</div>",
                    unsafe_allow_html=True)
        st.markdown("---")

        # Upload receipt
        st.markdown("### 📸 Submit New Claim")
        uploaded_file = st.file_uploader("Drop receipt image",
                                         type=["jpg","jpeg","png","pdf"],
                                         label_visibility="collapsed")
        if uploaded_file and st.button("⚡ Submit", use_container_width=True):
            with st.spinner("Processing receipt..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/receipts/upload",
                        files={"file": (uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        uploaded_file.type)},
                        data={"user_id": user["user_id"]},
                        timeout=60,
                    )
                    res = r.json()
                    if res.get("rejected"):
                        st.error(f"🚫 Duplicate: {res.get('rejection_reason','')}")
                    else:
                        exp = res.get("expense", {})
                        st.success(
                            f"✅ Submitted: {exp.get('vendor','?')} — "
                            f"{rupees(exp.get('total_amount') or 0)}"
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        st.markdown("---")
        page = st.radio("View", ["📄 My Claims", "📊 My Summary"],
                        label_visibility="collapsed")

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            do_logout()

    return page


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PAGES
# ══════════════════════════════════════════════════════════════════════════════

def admin_overview(uid: str):
    data = api_get(f"/users/{uid}/dashboard")
    if not data:
        st.info("No data for this user.")
        return

    total   = data.get("total_spend_this_month", 0)
    by_cat  = data.get("spend_by_category", {})
    vendors = data.get("top_vendors", [])
    anoms   = data.get("anomalies_this_month", [])
    mom     = data.get("month_over_month_change", {})
    balerts = data.get("budget_alerts", [])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total This Month</div>'
                    f'<div class="kpi-value">{rupees(total)}</div></div>',
                    unsafe_allow_html=True)
    with k2:
        top_cat = max(by_cat, key=by_cat.get) if by_cat else "—"
        delta   = mom.get(top_cat, 0)
        arrow   = "▲" if delta > 0 else "▼"
        color   = "#ff6b6b" if delta > 0 else "#51cf66"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Top Category</div>'
                    f'<div class="kpi-value">{top_cat.title()}</div>'
                    f'<div style="color:{color};font-size:0.8rem">'
                    f'{arrow} {abs(delta):.1f}% MoM</div></div>',
                    unsafe_allow_html=True)
    with k3:
        color = "#ff6b6b" if anoms else "#51cf66"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Anomalies</div>'
                    f'<div class="kpi-value" style="color:{color}">{len(anoms)}</div></div>',
                    unsafe_allow_html=True)
    with k4:
        color = "#ff6b6b" if balerts else "#51cf66"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Budget Alerts</div>'
                    f'<div class="kpi-value" style="color:{color}">{len(balerts)}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_right = st.columns([3, 2])

    with col_chart:
        st.markdown('<div class="section-header">Spend by Category</div>',
                    unsafe_allow_html=True)
        if by_cat:
            cats = list(by_cat.keys()); vals = list(by_cat.values())
            colors = px.colors.qualitative.Pastel
            fig = go.Figure(go.Bar(
                x=cats, y=vals, marker=dict(color=colors[:len(cats)], line=dict(width=0)),
                text=[f"₹{v:,.0f}" for v in vals], textposition="outside",
                textfont=dict(color="#d0d0e8", size=11),
                hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
            ))
            annotations=[
                dict(x=c, y=by_cat[c]*1.05,
                     text=f"{'↑' if mom.get(c,0)>0 else '↓'}{abs(mom.get(c,0)):.0f}%",
                     font=dict(color="#ff9944" if mom.get(c,0)>0 else "#51cf66", size=10),
                     showarrow=False)
                for c in cats if mom.get(c,0)
            ]
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d0d0e8"),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
                margin=dict(t=20,b=10,l=10,r=10), height=300,
                annotations=annotations, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            fig2 = go.Figure(go.Pie(
                labels=cats, values=vals, hole=0.6,
                marker=dict(colors=colors[:len(cats)], line=dict(width=0)),
                textinfo="label+percent", textfont=dict(color="#d0d0e8", size=11),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d0d0e8"), margin=dict(t=0,b=0,l=0,r=0),
                height=220, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-header">Top Vendors</div>',
                    unsafe_allow_html=True)
        for v in vendors:
            st.markdown(
                f'<div class="vendor-row">'
                f'<span class="vendor-name">🏪 {v.get("vendor","?")}</span>'
                f'<span class="vendor-amount">{rupees(v.get("total",0))}'
                f'<span style="color:#777;font-size:0.8rem"> / {v.get("count",0)}×</span>'
                f'</span></div>', unsafe_allow_html=True)

    # AI insight
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 AI Weekly Summary</div>',
                unsafe_allow_html=True)
    if st.button("✨ Generate Insight", key="gen_insight"):
        with st.spinner("Generating insight…"):
            try:
                r = requests.get(f"{API_BASE}/users/{uid}/insights/weekly_summary",
                                 timeout=90)
                ins = r.json() if r.ok else {}
                if ins.get("insight_text"):
                    st.session_state[f"insight_{uid}"] = ins
                else:
                    detail = ins.get("detail", "Failed to generate insight.")
                    st.error(f"Insight error: {detail}")
            except Exception as e:
                st.error(f"Insight error: {e}")

    active_ins = st.session_state.get(f"insight_{uid}")
    if active_ins and active_ins.get("insight_text"):
        score   = active_ins.get("grounding_score", 0)
        verdict = active_ins.get("grounding_verdict", "")
        color   = "#51cf66" if verdict=="grounded" else \
                  "#ff9944" if verdict=="partial" else "#ff6b6b"
        st.markdown(f'<div class="insight-box">{active_ins["insight_text"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"<small>Grounding: "
                    f"<b style='color:{color}'>{score:.2f}</b> ({verdict})"
                    f"{'  · cached' if active_ins.get('cached') else ''}</small>",
                    unsafe_allow_html=True)


def admin_anomalies(uid: str):
    st.markdown("## 🔴 Anomalies")
    data = api_get(f"/users/{uid}/anomalies", {"limit": 50})
    if not data or data.get("count", 0) == 0:
        st.info("No anomalies yet.")
        return

    anoms = data.get("anomalies", [])
    scores = [a.get("anomaly_score", 0) for a in anoms]
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=20,
        marker=dict(color="#7c5cbf", line=dict(width=0)),
    ))
    fig.add_vline(x=0.6, line_dash="dash", line_color="#ff6b6b",
                  annotation_text="Threshold (0.6)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d0e8"), margin=dict(t=10,b=10), height=200,
        xaxis_title="Anomaly Score", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"anom_hist_{uid}")

    for idx, a in enumerate(anoms):
        score = a.get("anomaly_score", 0)
        icon  = "🔴" if score >= 0.7 else "🟠" if score >= 0.5 else "🟡"
        rid   = a.get("receipt_id", "") or f"anom_{idx}"
        with st.expander(f"{icon} {a.get('vendor','?')} · {a.get('date','?')} · "
                         f"₹{(a.get('amount') or 0):,.0f} · Score {score:.2f}"):
            st.markdown(f"**Category:** {a.get('category','?').title()}")
            st.markdown(f"**Explanation:** {a.get('explanation','?')}")
            ex  = api_get(f"/users/{uid}/explain/anomaly/{rid}")
            if ex and ex.get("features"):
                feats = ex["features"][:5]
                fig_s = go.Figure(go.Bar(
                    x=[f.get("contribution", 0) for f in feats],
                    y=[f["feature"] for f in feats],
                    orientation="h",
                    marker=dict(
                        color=["#ff6b6b" if f.get("contribution",0)>0
                               else "#51cf66" for f in feats],
                        line=dict(width=0)),
                    text=[f.get("description","")[:40] for f in feats],
                    textposition="outside",
                    textfont=dict(size=9, color="#a0a0b8"),
                ))
                fig_s.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#d0d0e8"), margin=dict(t=5,b=5,l=100,r=5),
                    height=180, showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"anom_feat_{uid}_{rid}_{idx}")

            if st.button("❌ Dismiss anomaly", key=f"dismiss_{rid}_{idx}"):
                api_post(f"/users/{uid}/review/{rid}", params={"action": "approve"})
                st.success("Cleared")
                st.rerun()


def admin_forecasts(uid: str):
    st.markdown("## 📈 30-Day Forecasts")
    data = api_get(f"/users/{uid}/dashboard")
    by_cat = (data or {}).get("spend_by_category", {})
    if not by_cat:
        st.info("No spending data.")
        return

    cats = list(by_cat.keys())
    cat  = st.selectbox("Select Category", cats)

    budget_alerts = (data or {}).get("budget_alerts", [])
    if budget_alerts:
        st.markdown('<div class="section-header">Budget Status</div>',
                    unsafe_allow_html=True)
        cols = st.columns(min(len(budget_alerts), 3))
        for i, b in enumerate(budget_alerts):
            with cols[i % 3]:
                pct   = min(b.get("percent_used", 0), 100)
                level = b.get("alert_level", "ok")
                color = "#ff4d4d" if level == "critical" else \
                        "#ff8c00" if level == "warning" else "#00cc44"
                st.markdown(f"**{b.get('category','').title()}**")
                st.progress(int(pct) / 100)
                st.markdown(
                    f"<small style='color:{color}'>{pct:.0f}% · {level.upper()}</small>",
                    unsafe_allow_html=True)

    with st.spinner(f"Loading forecast for {cat}…"):
        fd = api_get(f"/users/{uid}/forecast/{cat}")
    if not fd:
        st.warning("Forecast unavailable — more data needed.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Monthly", rupees(fd.get("predicted_monthly_total", 0)))
    m2.metric("Trend", fd.get("trend","stable").title())
    breach = fd.get("days_until_budget_breach")
    m3.metric("Days to Breach", f"{breach} days" if breach else "No breach")

    pts = fd.get("forecast_points", [])
    if pts:
        df = pd.DataFrame(pts)
        df["ds"] = pd.to_datetime(df["ds"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(df["ds"]) + list(df["ds"][::-1]),
            y=list(df["yhat_upper"]) + list(df["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(124,92,191,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence",
        ))
        fig.add_trace(go.Scatter(
            x=df["ds"], y=df["yhat"],
            mode="lines", line=dict(color="#c5b4e3", width=2.5),
            name="Forecast",
            hovertemplate="<b>%{x|%d %b}</b><br>₹%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d0e8"), height=360,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
            margin=dict(t=10,b=10,l=10,r=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def admin_review_queue(uid: str):
    st.markdown("## 🔎 Review Queue")
    rq = api_get(f"/users/{uid}/review_queue")
    if not rq or rq.get("count", 0) == 0:
        st.success("✅ Queue is empty.")
        return

    for item in rq.get("items", []):
        exp    = item.get("expense", {})
        reason = item.get("review_reason", "")
        rid    = exp.get("receipt_id", "")

        reason_label = "⚠️ Low Confidence" if reason == "low_confidence" else "🔶 Possible Duplicate"
        st.markdown(
            f'<div class="claim-card">'
            f'<div class="claim-vendor">{exp.get("vendor","?")} · '
            f'{rupees(exp.get("total_amount") or 0)}</div>'
            f'<div class="claim-meta">{exp.get("date","?")} · '
            f'{(exp.get("category") or "?").title()} · '
            f'{exp.get("overall_confidence",0):.0%} confidence · '
            f'{reason_label}</div>'
            f'</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve", key=f"appr_{rid}", use_container_width=True):
                api_post(f"/users/{uid}/review/{rid}", params={"action": "approve"})
                st.success("Approved")
                st.rerun()
        with c2:
            if st.button("❌ Reject", key=f"rej_{rid}", use_container_width=True):
                api_post(f"/users/{uid}/review/{rid}", params={"action": "reject"})
                st.warning("Rejected & removed")
                st.rerun()


def admin_claim_management():
    """Admin sees all users' claims and can accept/reject pending ones."""
    st.markdown("## 📋 Claim Management")

    status_filter = st.radio("Show claims with status:",
                             ["pending", "accepted", "rejected", "all"],
                             horizontal=True)

    data = api_get("/admin/pending_claims", {"status": status_filter, "limit": 300})
    if not data:
        return

    claims = data.get("claims", [])
    if not claims:
        st.success(f"No {status_filter} claims.")
        return

    st.markdown(f"**{len(claims)} claims** · status = `{status_filter}`")

    # Status counts bar
    all_data = api_get("/admin/pending_claims", {"status": "all", "limit": 1000}) or {}
    all_claims = all_data.get("claims", [])
    if all_claims:
        count_df = pd.DataFrame([{
            "status": c.get("claim_status","pending"),
            "amount": c.get("total_amount",0),
        } for c in all_claims])
        grp = count_df.groupby("status").agg(count=("amount","count"),
                                             total=("amount","sum")).reset_index()
        colors_map = {"accepted":"#51cf66","pending":"#ff9944","rejected":"#ff6b6b"}
        fig = go.Figure(go.Bar(
            x=grp["status"], y=grp["count"],
            marker_color=[colors_map.get(s,"#888") for s in grp["status"]],
            text=grp["count"], textposition="outside",
            textfont=dict(color="#d0d0e8"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d0e8"), height=200,
            margin=dict(t=10,b=10,l=10,r=10),
            xaxis_title="Status", yaxis_title="Count", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Tabular view with action buttons for pending
    for c in claims:
        cstatus  = c.get("claim_status","pending")
        rid      = c.get("receipt_id","")
        uid      = c.get("user_id","")
        dname    = c.get("display_name", uid)
        vendor   = c.get("vendor","?")
        date     = c.get("date","?")
        amount   = c.get("total_amount",0)
        category = (c.get("category") or "other").title()
        note     = c.get("rejection_note","")
        reviewed = c.get("reviewed_by","")

        with st.container():
            col_info, col_status, col_actions = st.columns([4, 1.5, 2])

            with col_info:
                st.markdown(
                    f'<div style="padding:8px 0">'
                    f'<span style="color:#c5b4e3;font-weight:600">{dname}</span>'
                    f'<span style="color:#666"> · {date}</span><br>'
                    f'<b style="color:#d0d0e8">{vendor}</b>'
                    f'<span style="color:#777"> · {category} · '
                    f'<b style="color:#c5b4e3">{rupees(amount)}</b></span>'
                    + (f'<br><span style="color:#ff9944;font-size:0.78rem">📝 {note}</span>'
                       if note else '')
                    + (f'<br><span style="color:#555;font-size:0.75rem">'
                       f'Reviewed by {reviewed}</span>' if reviewed else '')
                    + '</div>', unsafe_allow_html=True)

            with col_status:
                st.markdown(f"<div style='padding-top:16px'>{status_badge(cstatus)}</div>",
                            unsafe_allow_html=True)

            with col_actions:
                if cstatus == "pending":
                    ca, cr = st.columns(2)
                    with ca:
                        if st.button("✅", key=f"acc_{rid}",
                                     help="Accept claim", use_container_width=True):
                            api_post(
                                f"/users/{uid}/claims/{rid}/status",
                                params={"action": "accept", "reviewer": "admin"},
                            )
                            st.rerun()
                    with cr:
                        if st.button("❌", key=f"rej_{rid}",
                                     help="Reject claim", use_container_width=True):
                            api_post(
                                f"/users/{uid}/claims/{rid}/status",
                                params={"action": "reject",
                                        "reviewer": "admin",
                                        "note": "Rejected by admin"},
                            )
                            st.rerun()

            st.markdown("<hr style='border:0;border-top:1px solid #2a2a3e;margin:4px 0'>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# USER PAGES
# ══════════════════════════════════════════════════════════════════════════════

STATUS_COLORS = {
    "accepted": "#51cf66", "pending": "#ff9944", "rejected": "#ff6b6b"
}
STATUS_ICONS  = {"accepted": "✅", "pending": "⏳", "rejected": "❌"}


def user_claims(user: dict):
    uid = user["user_id"]
    st.markdown(f"## 📄 My Claims  "
                f"<span style='color:#777;font-size:0.9rem'>{user['display_name']}</span>",
                unsafe_allow_html=True)

    # Status filter tabs
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    status_choice = "all"
    if col_f1.button("All",      use_container_width=True, key="all"):  status_choice="all"
    if col_f2.button("⏳ Pending", use_container_width=True, key="pend"): status_choice="pending"
    if col_f3.button("✅ Accepted",use_container_width=True, key="acc"):  status_choice="accepted"
    if col_f4.button("❌ Rejected",use_container_width=True, key="rej"):  status_choice="rejected"

    chosen = st.session_state.get("claim_status_filter", "all")
    for btn_name, val in [("all","all"),("pend","pending"),("acc","accepted"),("rej","rejected")]:
        k = f"{btn_name}_pressed"
    # Use a selectbox for persistence
    status_choice = st.selectbox("Status filter:", ["all","pending","accepted","rejected"],
                                 label_visibility="collapsed")

    data = api_get(f"/users/{uid}/claims", {"status": status_choice})
    if not data:
        return

    claims = data.get("claims", [])
    total_amount = sum(c.get("total_amount") or 0 for c in claims)
    st.markdown(
        f"<small style='color:#777'>{len(claims)} claims · "
        f"Total: <b style='color:#c5b4e3'>{rupees(total_amount)}</b></small>",
        unsafe_allow_html=True)

    # Summary donut
    if claims:
        by_cat = {}
        for c in claims:
            cat = c.get("category","other")
            by_cat[cat] = by_cat.get(cat, 0) + (c.get("total_amount") or 0)

        col_donut, col_list = st.columns([1, 2])

        with col_donut:
            fig = go.Figure(go.Pie(
                labels=list(by_cat.keys()), values=list(by_cat.values()),
                hole=0.62,
                marker=dict(colors=px.colors.qualitative.Pastel, line=dict(width=0)),
                textinfo="percent", textfont=dict(color="#d0d0e8", size=10),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d0d0e8"),
                margin=dict(t=0,b=0,l=0,r=0), height=200, showlegend=True,
                legend=dict(font=dict(color="#d0d0e8", size=10)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_list:
            # Status counts
            from collections import Counter
            sc = Counter(c.get("claim_status","pending") for c in claims)
            for s, cnt in sorted(sc.items()):
                color = STATUS_COLORS.get(s, "#888")
                pct   = cnt / max(len(claims), 1) * 100
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                    f"{status_badge(s)}"
                    f"<span style='color:{color};font-weight:600'>{cnt}</span>"
                    f"<span style='color:#555;font-size:0.8rem'>({pct:.0f}%)</span>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Claim History</div>', unsafe_allow_html=True)

    for c in claims:
        cstatus = c.get("claim_status", "pending")
        color   = STATUS_COLORS.get(cstatus, "#888")
        icon    = STATUS_ICONS.get(cstatus, "•")
        vendor  = c.get("vendor", "?")
        date    = c.get("date", "?")
        amount  = c.get("total_amount", 0)
        cat     = (c.get("category") or "other").title()
        note    = c.get("rejection_note", "")
        rev_at  = (c.get("reviewed_at") or "")[:10]
        subcat  = c.get("subcategory","")

        st.markdown(
            f'<div class="claim-card" style="border-left:4px solid {color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
            f'  <div>'
            f'    <span class="claim-vendor">{vendor}</span>'
            f'    <span style="color:#777;font-size:0.82rem"> · {cat}'
            + (f' / {subcat}' if subcat else '') +
            f'    </span><br>'
            f'    <span class="claim-meta">{date}'
            + (f' · Reviewed {rev_at}' if rev_at else '') +
            f'    </span>'
            + (f'<div class="claim-note">📝 {note}</div>' if note else '') +
            f'  </div>'
            f'  <div style="text-align:right">'
            f'    <div style="color:#c5b4e3;font-size:1.1rem;font-weight:700">'
            f'{rupees(amount)}</div>'
            f'    {status_badge(cstatus)}'
            f'  </div>'
            f'</div></div>', unsafe_allow_html=True)


def user_summary(user: dict):
    uid = user["user_id"]
    st.markdown(f"## 📊 My Spend Summary")

    data = api_get(f"/users/{uid}/claims", {"status": "all"})
    claims = (data or {}).get("claims", [])
    if not claims:
        st.info("No expenses yet.")
        return

    total = sum(c.get("total_amount") or 0 for c in claims)
    accepted_total = sum(c.get("total_amount") or 0 for c in claims
                         if c.get("claim_status") == "accepted")
    pending_total = sum(c.get("total_amount") or 0 for c in claims
                        if c.get("claim_status") == "pending")
    rejected_total = sum(c.get("total_amount") or 0 for c in claims
                         if c.get("claim_status") == "rejected")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card">'
                    f'<div class="kpi-label">Total Submitted</div>'
                    f'<div class="kpi-value">{rupees(total)}</div>'
                    f'<div class="kpi-sub">{len(claims)} claims</div></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card">'
                    f'<div class="kpi-label">Accepted</div>'
                    f'<div class="kpi-value" style="color:#51cf66">'
                    f'{rupees(accepted_total)}</div></div>',
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card">'
                    f'<div class="kpi-label">Pending</div>'
                    f'<div class="kpi-value" style="color:#ff9944">'
                    f'{rupees(pending_total)}</div></div>',
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card">'
                    f'<div class="kpi-label">Rejected</div>'
                    f'<div class="kpi-value" style="color:#ff6b6b">'
                    f'{rupees(rejected_total)}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trend
    df = pd.DataFrame([{
        "month": (c.get("date","2026-01")[:7]),
        "amount": c.get("total_amount") or 0,
        "status": c.get("claim_status","pending"),
    } for c in claims])

    if not df.empty:
        monthly = df.groupby(["month","status"])["amount"].sum().reset_index()
        colors_map = {"accepted":"#51cf66","pending":"#ff9944","rejected":"#ff6b6b"}
        fig = go.Figure()
        for s in ["accepted","pending","rejected"]:
            sub = monthly[monthly["status"]==s]
            if not sub.empty:
                fig.add_trace(go.Bar(
                    x=sub["month"], y=sub["amount"],
                    name=s.title(), marker_color=colors_map[s],
                ))
        fig.update_layout(
            barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#d0d0e8"),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
            margin=dict(t=10,b=10,l=10,r=10), height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.markdown('<div class="section-header">Monthly Spend by Status</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Category breakdown
        cat_df = df.groupby("status")["amount"].sum().reset_index()
        by_cat_all = df.groupby("category" if "category" in df.columns else "status")
        cat_data_df = pd.DataFrame([{
            "category": c.get("category","?"),
            "amount": c.get("total_amount") or 0,
        } for c in claims])
        cat_grp = cat_data_df.groupby("category")["amount"].sum().reset_index()\
                              .sort_values("amount", ascending=False)

        fig2 = go.Figure(go.Bar(
            x=cat_grp["category"], y=cat_grp["amount"],
            marker_color=px.colors.qualitative.Pastel[:len(cat_grp)],
            text=[rupees(v) for v in cat_grp["amount"]],
            textposition="outside", textfont=dict(color="#d0d0e8"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d0d0e8"), height=280,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2a2a3e"),
            margin=dict(t=10,b=10), showlegend=False,
        )
        st.markdown('<div class="section-header">Spend by Category</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

auth_user = get_session()

if not auth_user:
    show_login()
    st.stop()

role = auth_user.get("role", "user")

if role == "admin":
    page, selected_uid = admin_sidebar(auth_user)

    if page == "📊 Overview":
        admin_overview(selected_uid)
    elif page == "🔴 Anomalies":
        admin_anomalies(selected_uid)
    elif page == "📈 Forecasts":
        admin_forecasts(selected_uid)
    elif page == "🔎 Review Queue":
        admin_review_queue(selected_uid)
    elif page == "📋 Claim Management":
        admin_claim_management()

else:
    page = user_sidebar(auth_user)

    if page == "📄 My Claims":
        user_claims(auth_user)
    elif page == "📊 My Summary":
        user_summary(auth_user)
