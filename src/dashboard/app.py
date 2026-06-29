from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import html  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from email.utils import parsedate_to_datetime  # noqa: E402

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.agent.strategic_agent import StrategicAgent  # noqa: E402
from src.dashboard.metrics import build_dashboard_metrics  # noqa: E402
from src.config import COMPANY_INDUSTRY, DEFAULT_INDUSTRY  # noqa: E402

# shadcn-inspired palette
GREEN = "#84cc16"
RED = "#ef4444"
GRAY = "#a1a1aa"

# Sentiment palette (high-contrast): Positive / Negative / Neutral
SENT_POS = "#22c55e"   # green
SENT_NEG = "#f97316"   # orange
SENT_NEU = "#3b82f6"   # blue

st.set_page_config(page_title="AI CEO – SAP", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1200px; }

      /* Headings */
      h1 { font-weight: 800; letter-spacing: -0.02em; }
      h2, h3 { font-weight: 700; letter-spacing: -0.01em; }

      /* Metric cards */
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05);
      }
      [data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #71717a; font-weight: 600; }
      [data-testid="stMetricValue"] {
        font-weight: 800; letter-spacing: -0.02em;
        font-size: 1.5rem; line-height: 1.25;
        white-space: normal; overflow-wrap: break-word;
      }

      /* Bordered containers -> cards */
      [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid #e4e4e7 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        background: #ffffff;
      }

      /* Badges */
      .badge {
        display: inline-block; padding: 3px 10px; border-radius: 9999px;
        font-size: 0.72rem; font-weight: 700; margin-right: 6px; line-height: 1.2;
        border: 1px solid transparent; vertical-align: middle;
      }
      .badge-dark { background: #18181b; color: #fafafa; }
      .badge-soft { background: #f4f4f5; color: #3f3f46; border-color: #e4e4e7; }
      .badge-green { background: #ecfccb; color: #3f6212; }
      .badge-red { background: #fee2e2; color: #991b1b; }

      /* Inline pills used inside cards */
      .pill { display:inline-block; padding:2px 9px; border-radius:9999px; font-size:0.72rem; font-weight:700; }
      .pill-high { background:#fee2e2; color:#991b1b; }
      .pill-med  { background:#fef9c3; color:#854d0e; }
      .pill-low  { background:#dcfce7; color:#166534; }

      .card-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 6px; }
      .card-meta { color:#71717a; font-size:0.8rem; margin: 2px 0; }
      .card-summary { color:#3f3f46; font-size:0.88rem; margin-top:8px; }

      /* Tabs */
      [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #e4e4e7; }
      [data-baseweb="tab"] { font-weight: 600; }

      /* Primary button */
      [data-testid="stBaseButton-primary"] { border-radius: 10px; font-weight: 700; }

      section[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #ececef; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent():
    return StrategicAgent()


def pill(level: str) -> str:
    cls = {"High": "pill-high", "Medium": "pill-med", "Low": "pill-low"}.get(level, "pill-med")
    return f'<span class="pill {cls}">{level}</span>'


def signal_card(item: dict, level_label: str, level_key: str, category_key: str = None):
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">{html.escape(item.get("title", "Untitled"))}</div>',
            unsafe_allow_html=True,
        )
        meta = (
            f'{level_label}: {pill(item.get(level_key, "Medium"))} '
            f'&nbsp;·&nbsp; Confidence: <b>{item.get("confidence_score", "N/A")}</b>'
        )
        if category_key and item.get(category_key):
            meta += f' &nbsp;·&nbsp; Category: <span class="badge badge-soft">{html.escape(str(item.get(category_key)))}</span>'
        st.markdown(f'<div class="card-meta">{meta}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-meta">Source: {html.escape(item.get("source", "—") or "—")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-summary">{html.escape(item.get("summary", ""))}</div>',
            unsafe_allow_html=True,
        )


def parse_dt(value: str):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = parsedate_to_datetime(value)
        except Exception:
            return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fmt_date(value: str) -> str:
    dt = parse_dt(value)
    return dt.strftime("%Y-%m-%d") if dt else "—"


def as_text(item) -> str:
    """Defensive: render a list item as readable text even if the model
    returned a dict where a string was expected."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "competitor" in item:
            text = str(item.get("competitor", "")).strip()
            if item.get("count") is not None:
                text += f" ({item.get('count')} signals)"
            titles = item.get("titles") or []
            if titles:
                text += ": " + "; ".join(str(t) for t in titles[:2])
            return text
        for key in ("title", "name", "text", "summary", "activity", "description"):
            if item.get(key):
                return str(item[key])
        return "; ".join(f"{k}: {v}" for k, v in item.items())
    return str(item)


def flatten_retrieved(retrieved: dict):
    """Deduplicate the multi-view retrieval into a single list of documents."""
    seen = set()
    items = []
    for view_items in retrieved.values():
        for it in view_items:
            key = (it.get("title", "").strip().lower() or it.get("url", "").strip().lower())
            if key and key not in seen:
                seen.add(key)
                items.append(it)
    return items


def recent_items(items, source_types, limit=8):
    filtered = [it for it in items if it.get("source_type") in source_types]
    filtered.sort(key=lambda it: parse_dt(it.get("published", "")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return filtered[:limit]


def news_row(it: dict):
    title = html.escape(it.get("title", "Untitled"))
    source = html.escape(it.get("source", "") or "")
    url = it.get("url", "")
    date_str = fmt_date(it.get("published", ""))
    link = f'<a href="{html.escape(url)}" target="_blank">{title}</a>' if url else title
    st.markdown(
        f'<div class="card-meta">{date_str} &nbsp;·&nbsp; {source}</div>'
        f'<div style="margin-bottom:8px;">{link}</div>',
        unsafe_allow_html=True,
    )


def competitor_headline(h: dict):
    """Render a competitor item as title + a 1-2 sentence summary, with a
    'Read more' link beside the summary for the full source article."""
    title = html.escape(h.get("title", "Untitled"))
    source = html.escape(h.get("source", "") or "")
    url = h.get("url", "")
    date_str = fmt_date(h.get("published", ""))
    summary = html.escape(h.get("summary", "") or "")
    meta_bits = [b for b in [date_str if date_str != "—" else "", source] if b]
    meta = " &nbsp;·&nbsp; ".join(meta_bits)
    read_more = f' <a href="{html.escape(url)}" target="_blank">Read more ↗</a>' if url else ""
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<div class="card-title" style="font-size:0.92rem; margin-bottom:2px;">{title}</div>'
        + (f'<div class="card-meta">{meta}</div>' if meta else "")
        + f'<div class="card-summary" style="margin-top:4px;">{summary}{read_more}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


PHASE_BADGE = {
    "Goal": "badge-dark", "Plan": "badge-soft", "Retrieve": "badge-green",
    "Analyze": "badge-green", "Decide": "badge-soft", "Re-plan": "badge-red",
    "Recommend": "badge-dark", "Validate": "badge-green", "Done": "badge-dark",
}


def phase_badge(phase: str, ok: bool = True) -> str:
    cls = "badge-red" if not ok else PHASE_BADGE.get(phase, "badge-soft")
    return f'<span class="badge {cls}">{html.escape(phase)}</span>'


def render_trace(trace: list):
    for t in trace:
        out = html.escape(t.get("output_summary", ""))
        st.markdown(
            '<div style="margin-bottom:8px; padding-left:8px; border-left:3px solid #e4e4e7;">'
            f'{phase_badge(t.get("phase", ""), t.get("ok", True))} '
            f'<span class="card-meta" style="font-weight:600;">{html.escape(t.get("action", ""))}</span>'
            f'<div class="card-summary" style="margin-top:2px;">{html.escape(t.get("detail", ""))}'
            + (f' <b>→ {out}</b>' if out else "")
            + '</div></div>',
            unsafe_allow_html=True,
        )


def render_plan(plan: list):
    for s in plan:
        st.markdown(
            '<div style="margin-bottom:8px;">'
            f'<b>{s.get("id")}. {html.escape(str(s.get("view", "")).title())}</b> '
            f'<span class="card-meta">— {html.escape(s.get("objective", ""))}</span>'
            f'<div class="card-summary">query: <code>{html.escape(s.get("query", ""))}</code></div>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_decisions(decisions: list):
    for i, d in enumerate(decisions, start=1):
        st.markdown(
            '<div style="margin-bottom:6px;">'
            f'<b>#{i}</b> {pill(d.get("level", "Medium"))} '
            f'<span class="badge badge-soft">{html.escape(d.get("type", ""))}</span> '
            f'<span class="card-meta">priority {d.get("priority_score")}</span>'
            f'<div class="card-summary">{html.escape(d.get("title", "")[:90])} '
            f'<i style="color:#71717a;">— {html.escape(d.get("rationale", ""))}</i></div>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_validation(v: dict):
    if v.get("validated"):
        head = '<span class="badge badge-green">VALIDATED</span>'
    else:
        head = '<span class="badge badge-red">NEEDS REVIEW</span>'
    st.markdown(
        f'{head} &nbsp; {v.get("passed", 0)} pass · {v.get("warned", 0)} warn · {v.get("failed", 0)} fail',
        unsafe_allow_html=True,
    )
    if v.get("dropped"):
        st.markdown(
            f'<div class="card-meta">Dropped (unsupported by evidence): '
            f'{html.escape(", ".join(v["dropped"]))}</div>',
            unsafe_allow_html=True,
        )
    status_cls = {"pass": "pill-low", "warn": "pill-med", "fail": "pill-high"}
    for it in v.get("items", []):
        cls = status_cls.get(it.get("status"), "pill-med")
        issues = "; ".join(it.get("issues", [])) or "all checks passed"
        st.markdown(
            f'<div style="margin-bottom:4px;"><span class="pill {cls}">{it.get("status")}</span> '
            f'{html.escape(it.get("title", "")[:70])} '
            f'<span class="card-meta">— grounded {it.get("grounded_evidence")}/{it.get("total_evidence")}; '
            f'{html.escape(issues)}</span></div>',
            unsafe_allow_html=True,
        )


def render_phase_stepper(state):
    """A compact horizontal summary band of the agent loop with one key
    result per phase - gives the whole run at a glance."""
    evidence = state.evidence or {}
    analysis = state.analysis or {}
    validation = state.validation or {}
    total_chunks = sum(len(v) for v in evidence.values())
    total_signals = (len(analysis.get("opportunities", []))
                     + len(analysis.get("risks", []))
                     + len(analysis.get("trends", []))
                     + len(analysis.get("competitor_signals", [])))
    val_label = "✓ passed" if validation.get("validated") else f"{validation.get('failed', 0)} failed"

    steps = [
        ("Goal", "received"),
        ("Plan", f"{len(state.plan or [])} steps"),
        ("Retrieve", f"{total_chunks} chunks"),
        ("Analyze", f"{total_signals} signals"),
        ("Decide", f"top {len(state.decisions or [])}"),
        ("Recommend", f"{len(state.recommendations or [])} recs"),
        ("Validate", val_label),
    ]
    chips = []
    for i, (name, val) in enumerate(steps):
        chips.append(
            '<div style="flex:1 1 88px; min-width:88px; border:1px solid #e4e4e7; border-radius:10px; '
            'padding:8px 6px; background:#fff; box-shadow:0 1px 2px rgba(16,24,40,.04); text-align:center;">'
            f'<div style="font-size:.62rem; color:#71717a; font-weight:700; text-transform:uppercase; '
            f'letter-spacing:.04em;">{html.escape(name)}</div>'
            f'<div style="font-weight:700; font-size:.82rem; margin-top:2px;">{html.escape(val)}</div></div>'
        )
        if i < len(steps) - 1:
            chips.append('<span style="color:#cbd5e1; font-weight:700;">→</span>')
    st.markdown(
        '<div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0 18px;">'
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


agent = get_agent()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 AI CEO")
    st.caption("Strategic Intelligence Agent")
    st.markdown(
        '<span class="badge badge-dark">Agent</span>'
        '<span class="badge badge-soft">Plan → Act → Validate</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    company = st.text_input("Company", value="SAP")
    goal = st.text_area(
        "Goal",
        value="Act as the CEO advisor for SAP: decide what to do next, and why.",
        height=90,
    )
    generate = st.button("Run Strategic Agent", type="primary", use_container_width=True)
    st.caption("The agent will plan, retrieve & analyze evidence, decide, recommend, and validate.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(f"# AI CEO: Strategic Intelligence Agent")
st.markdown(
    '<span class="badge badge-dark">SAP</span>',
   # '<span class="badge badge-green">Cloud &amp; AI</span>'
   # '<span class="badge badge-soft">Executive briefing</span>',
    unsafe_allow_html=True,
)
st.markdown(
    "Strategic intelligence, evidence retrieval, and CEO-style recommendations — "
    "synthesized from the collected document corpus."
)

# ----------------------------------------------------------------------------
# Run pipeline (persist in session_state so tab switching doesn't re-run it)
# ----------------------------------------------------------------------------
if generate:
    with st.spinner("Agent working: planning → retrieving evidence → analyzing → deciding → "
                    "recommending → validating. This can take a minute…"):
        st.session_state["company"] = company or "SAP"
        st.session_state["state"] = agent.run(goal=goal, company=company or "SAP")

state = st.session_state.get("state")
company_name = st.session_state.get("company", "SAP")

if not state:
    st.info("Use **Run Strategic Agent** in the sidebar to produce the dashboard.")
    st.stop()

response = state.response or {}
retrieved = state.evidence or {}
analysis = state.analysis or {}
corpus = state.corpus or {}
metrics = build_dashboard_metrics(retrieved, analysis)

if state.mode == "agent-fallback":
    st.warning("LLM produced no usable output for the briefing — showing the deterministic "
               "fallback. The agent loop (plan/retrieve/decide/validate) still ran in full.")

if not response:
    st.error("No response returned from the agent.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
st.markdown("### Company Overview")
industry = COMPANY_INDUSTRY.get(company_name, DEFAULT_INDUSTRY)
o1, o2, o3, o4, o5 = st.columns(5)
o1.metric("Company", company_name, border=True)
o2.metric("Industry", industry, border=True)
o3.metric("Documents collected", corpus.get("total_documents", 0), border=True)
o4.metric("Data sources", corpus.get("distinct_sources", 0), border=True)
o5.metric("Last updated", fmt_date(corpus.get("last_updated", "")), border=True)

p1, p2, p3 = st.columns(3)
p1.metric("Indexed chunks", corpus.get("total_chunks", 0), border=True)
p2.metric("Analyzed this run", analysis.get("documents_analyzed", 0), border=True)
p3.metric(
    "Dominant sentiment",
    analysis.get("sentiment_summary", {}).get("dominant_sentiment", "N/A"),
    border=True,
)

st.write("")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_agent, tab_market, tab_opp_risk, tab_sentiment, tab_sources = st.tabs(
    ["Overview", "🧠 Agent Reasoning", "Market Intelligence", "Opportunities & Risks", "Sentiment", "Sources"]
)

with tab_agent:
    mode_badge = ('<span class="badge badge-green">agent · llm</span>'
                  if state.mode == "agent"
                  else '<span class="badge badge-red">agent · fallback</span>')
    state_dict = state.to_dict()

    # --- Header + goal -------------------------------------------------
    st.markdown(
        '<div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">'
        '<h4 style="margin:0;">Agent reasoning</h4>' + mode_badge + "</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            '<div class="card-meta" style="font-weight:700; letter-spacing:.03em;">GOAL</div>'
            f'<div class="card-summary" style="margin-top:2px;">{html.escape(state.goal)}</div>',
            unsafe_allow_html=True,
        )

    # --- Phase stepper (run at a glance) ------------------------------
    render_phase_stepper(state)

    # --- Plan ----------------------------------------------------------
    st.markdown("##### 1 · Plan")
    st.caption("The agent's investigation plan — one search query it wrote per intelligence view.")
    with st.container(border=True):
        render_plan(state_dict.get("plan", []))

    # --- Decisions -----------------------------------------------------
    st.markdown("##### 2 · Prioritized decisions")
    suff = state_dict.get("decision_summary", {})
    st.caption(
        f"Autonomous ranking by impact × confidence × freshness. "
        f"{suff.get('reason', '')} — considered {suff.get('considered', 0)}, "
        f"selected {suff.get('selected', 0)}, avg confidence {suff.get('avg_confidence', 0)}."
    )
    decisions = state_dict.get("decisions", [])
    if decisions:
        ddf = pd.DataFrame([{
            "#": i + 1,
            "Type": d.get("type", ""),
            "Level": d.get("level", ""),
            "Priority": d.get("priority_score"),
            "Confidence": d.get("confidence"),
            "Signal": d.get("title", ""),
        } for i, d in enumerate(decisions)])
        st.dataframe(ddf, use_container_width=True, hide_index=True)
    else:
        st.info("No decisions recorded.")

    # --- Validation ----------------------------------------------------
    st.markdown("##### 3 · Validation (self-check before presenting)")
    st.caption("Every recommendation is checked for grounding (real evidence), completeness, and consistency.")
    v = state_dict.get("validation", {})
    verdict = ('<span class="badge badge-green">VALIDATED</span>'
               if v.get("validated") else '<span class="badge badge-red">NEEDS REVIEW</span>')
    st.markdown(
        f'{verdict} &nbsp; {v.get("passed", 0)} pass · {v.get("warned", 0)} warn · {v.get("failed", 0)} fail',
        unsafe_allow_html=True,
    )
    if v.get("dropped"):
        st.caption("Dropped (unsupported by evidence): " + ", ".join(v["dropped"]))
    items = v.get("items", [])
    if items:
        vdf = pd.DataFrame([{
            "Status": it.get("status", "").upper(),
            "Recommendation": it.get("title", ""),
            "Grounded": f"{it.get('grounded_evidence', 0)}/{it.get('total_evidence', 0)}",
            "Issues": "; ".join(it.get("issues", [])) or "—",
        } for it in items])
        st.dataframe(vdf, use_container_width=True, hide_index=True)

    # --- Full trace (tucked away) -------------------------------------
    with st.expander("Execution log — full reasoning trace", expanded=False):
        render_trace(state_dict.get("trace", []))

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### What happened?")
            st.write(response.get("what_happened", "No data"))
    with c2:
        with st.container(border=True):
            st.markdown("#### Why does it matter?")
            st.write(response.get("why_it_matters", "No data"))

    with st.container(border=True):
        st.markdown("#### CEO Briefing")
        st.write(response.get("ceo_briefing", "No data"))

    st.markdown("#### Strategic recommendations")
    recs = response.get("strategic_recommendations", [])
    if not recs:
        st.info("No strategic recommendations generated.")
    for rec in recs:
        with st.container(border=True):
            st.markdown(
                f'<div class="card-title">{rec.get("title", "Recommendation")} '
                f'&nbsp;{pill(rec.get("priority", "Medium"))}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card-meta">Expected impact: {rec.get("expected_impact", "N/A")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card-meta">Risk level: {pill(rec.get("risk_level", "Medium"))} '
                f'&nbsp;·&nbsp; Confidence: <b>{rec.get("confidence_score", "N/A")}</b></div>',
                unsafe_allow_html=True,
            )
            evidence = rec.get("supporting_evidence", [])
            if evidence:
                st.markdown('<div class="card-meta">Supporting evidence:</div>', unsafe_allow_html=True)
                for ev in evidence:
                    st.markdown(f"- {ev}")

with tab_opp_risk:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Opportunity signals")
        opps = analysis.get("opportunities", [])
        if not opps:
            st.info("No opportunity signals detected.")
        for item in opps:
            signal_card(item, "Impact", "impact_level")
    with col2:
        st.markdown("#### Risk signals")
        risks = analysis.get("risks", [])
        if not risks:
            st.info("No risk signals detected.")
        for item in risks:
            signal_card(item, "Severity", "severity_level", category_key="risk_category")

with tab_market:
    all_items = flatten_retrieved(retrieved)

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        with st.container(border=True):
            st.markdown("#### 📰 Recent news")
            news = recent_items(all_items, {"news", "market"}, limit=8)
            if news:
                for it in news:
                    news_row(it)
            else:
                st.info("No recent news found.")
    with mcol2:
        with st.container(border=True):
            st.markdown("#### 🏢 Company announcements")
            announcements = recent_items(all_items, {"company"}, limit=8)
            if announcements:
                for it in announcements:
                    news_row(it)
            else:
                st.info("No company announcements found.")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        with st.container(border=True):
            st.markdown("#### 🚀 Emerging technologies & trends")
            trends = response.get("trends_to_monitor", [])
            if trends:
                for item in trends:
                    st.markdown(f"- {as_text(item)}")
            else:
                st.info("No trends summarized.")
    with tcol2:
        with st.container(border=True):
            st.markdown("#### ⚔️ Competitor activity")
            activity = response.get("competitor_activity", [])
            if activity:
                for item in activity:
                    st.markdown(f"- {as_text(item)}")
            else:
                st.info("No competitor activity summarized.")

    st.markdown("#### Competitor signals")
    st.caption("Recent news where a competitor is the headline subject.")
    competitor_signals = analysis.get("competitor_signals", [])
    if not competitor_signals:
        st.info("No competitor-focused news found in the current corpus.")
    else:
        cols = st.columns(2)
        for idx, item in enumerate(competitor_signals):
            with cols[idx % 2]:
                with st.container(border=True):
                    n = item.get("count", 0)
                    st.markdown(
                        f'<div class="card-title">{html.escape(item.get("competitor", "Competitor"))} '
                        f'<span class="badge badge-soft">{n} article{"s" if n != 1 else ""}</span></div>',
                        unsafe_allow_html=True,
                    )
                    latest = fmt_date(item.get("latest_published", ""))
                    if latest != "—":
                        st.markdown(
                            f'<div class="card-meta">Latest: {latest}</div>',
                            unsafe_allow_html=True,
                        )
                    headlines = item.get("headlines")
                    if headlines:
                        for h in headlines:
                            competitor_headline(h)
                    else:
                        for title in item.get("titles", []):
                            st.markdown(f"- {html.escape(title)}")

with tab_sentiment:
    sentiment_summary = analysis.get("sentiment_summary", {})
    sentiment = sentiment_summary.get("counts", {})
    total_sentiment = sum(sentiment.values()) or 0

    st.caption(f"Sentiment computed across all {analysis.get('sentiment_documents', total_sentiment)} collected documents.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Positive", sentiment.get("Positive", 0), border=True)
    s2.metric("Negative", sentiment.get("Negative", 0), border=True)
    s3.metric("Neutral", sentiment.get("Neutral", 0), border=True)
    s4.metric("Dominant", sentiment_summary.get("dominant_sentiment", "N/A"), border=True)

    if total_sentiment:
        chart_col, share_col = st.columns([2, 1])
        with chart_col:
            with st.container(border=True):
                st.markdown("##### Sentiment distribution (analyzed documents)")
                sentiment_df = pd.DataFrame({
                    "sentiment": ["Positive", "Negative", "Neutral"],
                    "documents": [
                        sentiment.get("Positive", 0),
                        sentiment.get("Negative", 0),
                        sentiment.get("Neutral", 0),
                    ],
                })
                dist_chart = (
                    alt.Chart(sentiment_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("sentiment:N", sort=["Positive", "Negative", "Neutral"], title="sentiment"),
                        y=alt.Y("documents:Q", title="documents"),
                        color=alt.Color(
                            "sentiment:N",
                            scale=alt.Scale(
                                domain=["Positive", "Negative", "Neutral"],
                                range=[SENT_POS, SENT_NEG, SENT_NEU],
                            ),
                            legend=None,
                        ),
                    )
                )
                st.altair_chart(dist_chart, use_container_width=True)
        with share_col:
            with st.container(border=True):
                st.markdown("##### Share")
                for label in ["Positive", "Negative", "Neutral"]:
                    pct = (sentiment.get(label, 0) / total_sentiment) * 100
                    st.write(f"**{label}:** {pct:.0f}%")
                    st.progress(min(int(pct), 100))

        trend_col, type_col = st.columns(2)

        with trend_col:
            with st.container(border=True):
                st.markdown("##### Sentiment trend over time")
                trend = analysis.get("sentiment_trend", [])
                if trend:
                    trend_df = pd.DataFrame(trend).set_index("period")
                    # Streamlit maps the color list to series in alphabetical
                    # order (Negative, Neutral, Positive), so order both to match.
                    st.line_chart(
                        trend_df[["Negative", "Neutral", "Positive"]],
                        color=[SENT_NEG, SENT_NEU, SENT_POS],
                    )
                    st.caption("Monthly document counts by sentiment (dated documents only).")
                else:
                    st.info("Not enough dated documents to build a trend.")

        with type_col:
            sentiment_by_type = analysis.get("sentiment_by_source_type", {})
            with st.container(border=True):
                st.markdown("##### Sentiment by source type")
                if sentiment_by_type:
                    by_type_df = pd.DataFrame(sentiment_by_type).T.fillna(0)
                    for col in ["Positive", "Negative", "Neutral"]:
                        if col not in by_type_df.columns:
                            by_type_df[col] = 0
                    by_type_df = by_type_df[["Negative", "Neutral", "Positive"]].rename_axis("source type")
                    st.bar_chart(by_type_df, color=[SENT_NEG, SENT_NEU, SENT_POS])
                else:
                    st.info("No source-type sentiment available.")
    else:
        st.info("No sentiment data available for this run.")

with tab_sources:
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("##### Corpus by source type")
            corpus_type_counts = corpus.get("source_type_counts", {})
            if corpus_type_counts:
                type_df = pd.DataFrame({"chunks": corpus_type_counts}).rename_axis("source type")
                st.bar_chart(type_df, color=GREEN)
            else:
                st.info("No indexed documents yet — run the collection pipeline.")
    with col_b:
        with st.container(border=True):
            st.markdown("##### Top sources (corpus)")
            sorted_sources = sorted(
                corpus.get("source_counts", {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:8]
            if sorted_sources:
                source_df = pd.DataFrame(sorted_sources, columns=["source", "chunks"]).set_index("source")
                st.bar_chart(source_df, color=GREEN, horizontal=True)
            else:
                st.info("No source data available.")
