"""
Prompt templates for grounded LLM insight generation.

Each template expects ``{kpi_context}`` to be replaced with the output
of ``to_prompt_string()``.  Additional variables are documented per template.
"""

# ── System prompt (injected as system_instruction) ────────────────────────────

SYSTEM_PROMPT = """
You are a personal finance assistant analyzing expense data for a user in India.

STRICT RULES:
1. You ONLY reference numbers, amounts, vendors, and dates that appear in the
   provided KPI context. Never invent, estimate, or recall numbers from memory.
2. All currency amounts must use ₹ symbol and match the context exactly.
3. If a category or metric is not in the context, do not mention it.
4. Be concise, specific, and actionable. No generic financial advice.
5. Tone: friendly but professional. Plain language, no jargon.
"""

# ── Weekly summary ────────────────────────────────────────────────────────────

WEEKLY_SUMMARY_TEMPLATE = """
{kpi_context}

Using ONLY the data above, generate a concise weekly expense summary (3-5 sentences).
Cover: largest spending category, any anomalies flagged, any budget warnings,
and one specific actionable observation. Reference exact amounts from the data.
Do not add advice not supported by the numbers above.
"""

# ── Anomaly explanation ───────────────────────────────────────────────────────

ANOMALY_EXPLANATION_TEMPLATE = """
{kpi_context}

A transaction has been flagged as anomalous:
  Vendor   : {vendor}
  Date     : {date}
  Amount   : ₹{amount}
  Category : {category}
  Score    : {anomaly_score:.2f} / 1.0
  System reason: {system_explanation}

Using ONLY the data above, explain in 2-3 sentences why this transaction is unusual
for this user. Reference their typical spending pattern for this category from the context.
"""

# ── Budget advice ─────────────────────────────────────────────────────────────

BUDGET_ADVICE_TEMPLATE = """
{kpi_context}

The user has the following budget alerts (from the data above):
{alerts_summary}

Using ONLY the data above, suggest specific daily spending limits for each alerted
category to stay within budget by month end. Show your arithmetic.
Format as a short bullet list. Do not suggest saving strategies not grounded in the data.
"""

# ── Vendor insight ────────────────────────────────────────────────────────────

VENDOR_INSIGHT_TEMPLATE = """
{kpi_context}

Focus on this vendor: {vendor}
The user has spent ₹{total} across {count} visits this month.

Using ONLY the data above, describe this vendor's spending pattern in 2-3 sentences.
Mention frequency, average transaction size, and whether it is flagged as anomalous.
"""

# ── Forecast alert ────────────────────────────────────────────────────────────

FORECAST_ALERT_TEMPLATE = """
{kpi_context}

The following categories are forecast to breach their budgets:
{breach_summary}

Using ONLY the data above, write a 2-3 sentence proactive warning to the user.
State how many days remain, how much they can spend per day to stay within budget,
and which category needs the most urgent attention.
"""
