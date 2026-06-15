import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
import pandas as pd
import math

# ─────────────────── PAGE CONFIG ───────────────────
st.set_page_config(
    page_title="PM Calculator — EVM, EMV, Financial & Critical Path",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────── THEME / STYLES ───────────────────
COLORS = dict(
    bg="#0f172a", surface="#1e293b", surface2="#334155", border="#475569",
    accent="#3b82f6", green="#22c55e", red="#ef4444", yellow="#eab308",
    orange="#f97316", purple="#a78bfa", cyan="#06b6d4", text="#f1f5f9", muted="#94a3b8",
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f172a",
    font=dict(color="#94a3b8", family="Segoe UI, system-ui, sans-serif"),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    [data-testid="stHeader"] { background-color: #0f172a; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px; background: #1e293b; border-radius: 10px; padding: 4px; }
    [data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; font-weight: 600; }
    [data-testid="stTabs"] [aria-selected="true"] { background: #3b82f6; color: #fff; }
    .stMetric { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    div[data-testid="stExpander"] { background: #1e293b; border: 1px solid #334155; border-radius: 10px; }
    .metric-good [data-testid="stMetricValue"] { color: #22c55e !important; }
    .metric-bad [data-testid="stMetricValue"]  { color: #ef4444 !important; }
    .metric-warn [data-testid="stMetricValue"] { color: #eab308 !important; }
    h1, h2, h3, h4 { color: #f1f5f9 !important; }
    .stMarkdown p, .stMarkdown li { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


def pl(**kw):
    """Merge custom kwargs into the default plot layout."""
    layout = {**PLOT_LAYOUT}
    for k, v in kw.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


def dollar(v):
    if abs(v) >= 1_000_000:
        return f"${v/1e6:.1f}M"
    if abs(v) >= 1_000:
        return f"${v/1e3:.0f}k"
    return f"${v:,.0f}"


# ─────────────────── EVM CALCULATION ───────────────────
def calc_evm(bac, pv, ev, ac):
    cv = ev - ac
    sv = ev - pv
    cpi = ev / ac if ac else None
    spi = ev / pv if pv else None
    eac_cpi = bac / cpi if cpi else None
    eac_ac = ac + (bac - ev)
    eac_comb = ac + (bac - ev) / (cpi * spi) if cpi and spi and (cpi * spi) != 0 else None
    etc = eac_cpi - ac if eac_cpi is not None else None
    vac = bac - eac_cpi if eac_cpi is not None else None
    tcpi_bac = (bac - ev) / (bac - ac) if (bac - ac) != 0 else None
    tcpi_eac = (bac - ev) / (eac_cpi - ac) if eac_cpi and (eac_cpi - ac) != 0 else None
    pct_complete = (ev / bac) * 100 if bac else 0
    pct_spent = (ac / bac) * 100 if bac else 0
    return dict(
        cv=cv, sv=sv, cpi=cpi, spi=spi,
        eac_cpi=eac_cpi, eac_ac=eac_ac, eac_comb=eac_comb,
        etc=etc, vac=vac,
        tcpi_bac=tcpi_bac, tcpi_eac=tcpi_eac,
        pct_complete=pct_complete, pct_spent=pct_spent,
    )


# ─────────────────── EMV CALCULATION ───────────────────
def calc_emv(risks):
    items = []
    total_emv = 0
    total_threats = 0
    total_opportunities = 0
    for r in risks:
        emv = r["probability"] * r["impact"]
        total_emv += emv
        if r["impact"] < 0:
            total_threats += emv
        else:
            total_opportunities += emv
        items.append(dict(
            name=r["name"], type=r.get("type", "Threat"),
            probability=r["probability"], impact=r["impact"], emv=emv,
        ))
    return dict(
        items=items, total_emv=total_emv,
        total_threats=total_threats, total_opportunities=total_opportunities,
        risk_count=len(items),
        threat_count=sum(1 for i in items if i["impact"] < 0),
        opportunity_count=sum(1 for i in items if i["impact"] >= 0),
    )


# ─────────────────── FINANCIAL CALCULATIONS ───────────────────
def calc_financials(initial_investment, discount_rate, cash_flows, fixed_costs=0, variable_cost_per_unit=0, selling_price_per_unit=0):
    n = len(cash_flows)
    r = discount_rate / 100

    # Discounted cash flows
    dcf = [cf / (1 + r) ** (t + 1) if r != 0 else cf for t, cf in enumerate(cash_flows)]

    # NPV
    npv = -initial_investment + sum(dcf)

    # Cumulative cash flows (undiscounted)
    cum_cf = []
    running = -initial_investment
    for cf in cash_flows:
        running += cf
        cum_cf.append(running)

    # Cumulative discounted cash flows
    cum_dcf = []
    running_d = -initial_investment
    for d in dcf:
        running_d += d
        cum_dcf.append(running_d)

    # Payback Period (simple)
    payback = None
    for t in range(n):
        if cum_cf[t] >= 0:
            if t == 0:
                payback = initial_investment / cash_flows[0] if cash_flows[0] > 0 else 0
            else:
                prev = cum_cf[t - 1]
                payback = t + abs(prev) / cash_flows[t] if cash_flows[t] > 0 else t + 1
            break

    # Discounted Payback Period
    disc_payback = None
    for t in range(n):
        if cum_dcf[t] >= 0:
            if t == 0:
                disc_payback = initial_investment / dcf[0] if dcf[0] > 0 else 0
            else:
                prev = cum_dcf[t - 1]
                disc_payback = t + abs(prev) / dcf[t] if dcf[t] > 0 else t + 1
            break

    # IRR (bisection method)
    def npv_at(rate):
        return -initial_investment + sum(cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))

    irr = None
    lo, hi = -0.5, 5.0
    try:
        if npv_at(lo) * npv_at(hi) < 0:
            for _ in range(200):
                mid = (lo + hi) / 2
                if abs(npv_at(mid)) < 0.01:
                    irr = mid
                    break
                if npv_at(lo) * npv_at(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            if irr is None:
                irr = (lo + hi) / 2
    except (ZeroDivisionError, OverflowError):
        irr = None

    # ROI
    total_returns = sum(cash_flows)
    roi = ((total_returns - initial_investment) / initial_investment) * 100 if initial_investment > 0 else None

    # BCR (Benefit-Cost Ratio)
    pv_benefits = sum(d for d in dcf if d > 0)
    pv_costs = initial_investment + abs(sum(d for d in dcf if d < 0))
    bcr = pv_benefits / pv_costs if pv_costs > 0 else None

    # Profitability Index
    pv_future = sum(dcf)
    pi = pv_future / initial_investment if initial_investment > 0 else None

    # Break-Even Analysis
    bep_units = None
    bep_dollars = None
    contribution_margin = None
    if selling_price_per_unit > 0 and selling_price_per_unit > variable_cost_per_unit:
        contribution_margin = selling_price_per_unit - variable_cost_per_unit
        bep_units = fixed_costs / contribution_margin if contribution_margin > 0 else None
        cm_ratio = contribution_margin / selling_price_per_unit
        bep_dollars = fixed_costs / cm_ratio if cm_ratio > 0 else None

    # NPV sensitivity data
    npv_sensitivity = []
    for test_rate in range(0, 51, 2):
        tr = test_rate / 100
        test_npv = -initial_investment + sum(cf / (1 + tr) ** (t + 1) for t, cf in enumerate(cash_flows))
        npv_sensitivity.append(dict(rate=test_rate, npv=test_npv))

    return dict(
        npv=npv, irr=irr, roi=roi, bcr=bcr, pi=pi,
        payback=payback, disc_payback=disc_payback,
        cash_flows=cash_flows, dcf=dcf,
        cum_cf=cum_cf, cum_dcf=cum_dcf,
        total_returns=total_returns,
        initial_investment=initial_investment,
        discount_rate=discount_rate,
        bep_units=bep_units, bep_dollars=bep_dollars,
        contribution_margin=contribution_margin,
        fixed_costs=fixed_costs,
        variable_cost_per_unit=variable_cost_per_unit,
        selling_price_per_unit=selling_price_per_unit,
        npv_sensitivity=npv_sensitivity,
        pv_benefits=pv_benefits, pv_costs=pv_costs,
    )


# ─────────────────── CRITICAL PATH ───────────────────
def calc_critical_path(tasks):
    task_map = {t["id"]: t for t in tasks}
    in_degree = defaultdict(int)
    successors = defaultdict(list)
    for tid, t in task_map.items():
        in_degree.setdefault(tid, 0)
        for p in t.get("predecessors", []):
            successors[p].append(tid)
            in_degree[tid] += 1

    queue = deque([tid for tid in task_map if in_degree[tid] == 0])
    topo = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    if len(topo) != len(task_map):
        return None, "Cycle detected in task dependencies."

    es, ef = {}, {}
    for tid in topo:
        t = task_map[tid]
        preds = t.get("predecessors", [])
        es[tid] = max((ef[p] for p in preds), default=0)
        ef[tid] = es[tid] + t["duration"]

    proj_dur = max(ef.values()) if ef else 0
    lf, ls = {}, {}
    for tid in topo:
        if not successors[tid]:
            lf[tid] = proj_dur
    for tid in reversed(topo):
        if tid not in lf:
            lf[tid] = min(ls[s] for s in successors[tid])
        ls[tid] = lf[tid] - task_map[tid]["duration"]

    results = []
    cp = []
    for tid in topo:
        tf = ls[tid] - es[tid]
        crit = abs(tf) < 0.0001
        if crit:
            cp.append(tid)
        results.append(dict(
            id=tid, name=task_map[tid].get("name", tid),
            duration=task_map[tid]["duration"],
            es=es[tid], ef=ef[tid], ls=ls[tid], lf=lf[tid],
            total_float=round(tf, 2), critical=crit,
        ))
    return dict(project_duration=proj_dur, critical_path=cp, tasks=results), None


# ─────────────────── HEADER ───────────────────
st.markdown("<h1 style='text-align:center;'>📊 Project Management Calculator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;margin-top:-10px;'>Earned Value Management &bull; EMV Risk Analysis &bull; Financial Analysis &bull; S-Curve Tracker &bull; Critical Path</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────── TABS ───────────────────
tab_evm, tab_emv, tab_fin, tab_scurve, tab_cp, tab_ref = st.tabs(["📈 EVM Metrics", "🎯 EMV Calculator", "💰 Financial", "📉 S-Curve Tracker", "🔀 Critical Path", "📖 Formulas"])

# ═══════════════════════════════════════════════════════
#                     EVM TAB
# ═══════════════════════════════════════════════════════
with tab_evm:
    st.subheader("Input Parameters")
    c1, c2, c3, c4 = st.columns(4)
    bac = c1.number_input("BAC (Budget at Completion)", min_value=0.0, value=500000.0, step=1000.0, format="%.0f")
    pv = c2.number_input("PV (Planned Value)", min_value=0.0, value=200000.0, step=1000.0, format="%.0f")
    ev = c3.number_input("EV (Earned Value)", min_value=0.0, value=180000.0, step=1000.0, format="%.0f")
    ac = c4.number_input("AC (Actual Cost)", min_value=0.0, value=210000.0, step=1000.0, format="%.0f")

    if st.button("Calculate EVM", type="primary", use_container_width=True):
        st.session_state["evm"] = calc_evm(bac, pv, ev, ac)
        st.session_state["evm_inputs"] = dict(bac=bac, pv=pv, ev=ev, ac=ac)

    if "evm" in st.session_state:
        d = st.session_state["evm"]
        inp = st.session_state["evm_inputs"]
        bac, pv_v, ev_v, ac_v = inp["bac"], inp["pv"], inp["ev"], inp["ac"]

        def safe(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        # ── Metric cards ──
        st.markdown("### Performance Indices")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("CPI", safe(d["cpi"]), delta="Under budget" if d["cpi"] and d["cpi"] >= 1 else "Over budget",
                  delta_color="normal" if d["cpi"] and d["cpi"] >= 1 else "inverse")
        m2.metric("SPI", safe(d["spi"]), delta="Ahead" if d["spi"] and d["spi"] >= 1 else "Behind",
                  delta_color="normal" if d["spi"] and d["spi"] >= 1 else "inverse")
        m3.metric("TCPI (BAC)", safe(d["tcpi_bac"]))
        m4.metric("TCPI (EAC)", safe(d["tcpi_eac"]))
        m5.metric("% Complete", f'{d["pct_complete"]:.1f}%')
        m6.metric("% Spent", f'{d["pct_spent"]:.1f}%')

        st.markdown("### Variances")
        v1, v2, v3 = st.columns(3)
        v1.metric("Cost Variance (CV)", f'${d["cv"]:,.0f}', delta_color="normal" if d["cv"] >= 0 else "inverse",
                  delta="Favorable" if d["cv"] >= 0 else "Unfavorable")
        v2.metric("Schedule Variance (SV)", f'${d["sv"]:,.0f}', delta_color="normal" if d["sv"] >= 0 else "inverse",
                  delta="Favorable" if d["sv"] >= 0 else "Unfavorable")
        v3.metric("Variance at Completion", f'${d["vac"]:,.0f}' if d["vac"] is not None else "N/A",
                  delta_color="normal" if d["vac"] and d["vac"] >= 0 else "inverse",
                  delta="Under budget" if d["vac"] and d["vac"] >= 0 else "Over budget")

        st.markdown("### Forecasting")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("EAC (CPI)", f'${d["eac_cpi"]:,.0f}' if d["eac_cpi"] else "N/A")
        f2.metric("EAC (Atypical)", f'${d["eac_ac"]:,.0f}')
        f3.metric("EAC (CPI×SPI)", f'${d["eac_comb"]:,.0f}' if d["eac_comb"] else "N/A")
        f4.metric("ETC", f'${d["etc"]:,.0f}' if d["etc"] else "N/A")

        # ── CHARTS ──
        st.markdown("---")
        st.markdown("### 📊 Visual Analysis")

        # Row 1: Budget Comparison + Radar
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=["BAC", "PV", "EV", "AC"],
                y=[bac, pv_v, ev_v, ac_v],
                marker_color=[COLORS["accent"], COLORS["purple"], COLORS["green"],
                              COLORS["red"] if ac_v > ev_v else COLORS["yellow"]],
                text=[dollar(v) for v in [bac, pv_v, ev_v, ac_v]],
                textposition="outside", textfont=dict(color="#f1f5f9", size=11),
            ))
            fig.update_layout(**pl(title="Budget vs Actuals Comparison", yaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            cats = ["BAC", "PV", "EV", "AC", "EAC", "ETC"]
            fig.add_trace(go.Scatterpolar(r=[bac, pv_v, pv_v, pv_v, bac, bac - pv_v], theta=cats, fill="toself",
                                          name="Planned", line_color=COLORS["purple"], opacity=0.6))
            fig.add_trace(go.Scatterpolar(
                r=[bac, pv_v, ev_v, ac_v, d["eac_cpi"] or 0, d["etc"] or 0], theta=cats, fill="toself",
                name="Actual", line_color=COLORS["green"], opacity=0.6))
            fig.update_layout(**pl(title="Planned vs Actual Radar", polar=dict(
                bgcolor="#0f172a", radialaxis=dict(gridcolor="#1e293b", visible=True, showticklabels=False),
                angularaxis=dict(gridcolor="#334155", linecolor="#334155"))))
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: CPI/SPI bar + Cost Efficiency
        c1, c2 = st.columns(2)
        with c1:
            idx_labels = ["CPI", "SPI", "TCPI (BAC)", "TCPI (EAC)"]
            idx_vals = [d["cpi"] or 0, d["spi"] or 0, d["tcpi_bac"] or 0, d["tcpi_eac"] or 0]
            idx_colors = [
                COLORS["green"] if idx_vals[0] >= 1 else COLORS["red"],
                COLORS["green"] if idx_vals[1] >= 1 else COLORS["red"],
                COLORS["yellow"] if idx_vals[2] > 1 else COLORS["green"],
                COLORS["cyan"],
            ]
            fig = go.Figure(go.Bar(y=idx_labels, x=idx_vals, orientation="h",
                                   marker_color=idx_colors,                                   text=[f"{v:.4f}" for v in idx_vals], textposition="outside",
                                   textfont=dict(color="#f1f5f9", size=11)))
            fig.add_vline(x=1, line_dash="dash", line_color="#f1f5f9", line_width=1,
                          annotation_text="Target: 1.0", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="CPI / SPI / TCPI — Bar Gauge", xaxis=dict(title="Index Value")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            value_earned = min(ev_v, ac_v)
            waste = max(0, ac_v - ev_v)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["AC Breakdown"], y=[value_earned], name="Value Earned (EV)", marker_color=COLORS["green"]))
            fig.add_trace(go.Bar(x=["AC Breakdown"], y=[waste], name="Cost Overrun (AC-EV)", marker_color=COLORS["red"]))
            fig.update_layout(**pl(title="Cost Efficiency Breakdown", barmode="stack", yaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 3: EAC Comparison + Variance
        c1, c2 = st.columns(2)
        with c1:
            eac_labels = ["BAC\n(Original)", "EAC\n(CPI)", "EAC\n(Atypical)", "EAC\n(CPI×SPI)"]
            eac_vals = [bac, d["eac_cpi"] or 0, d["eac_ac"], d["eac_comb"] or 0]
            eac_colors = [COLORS["accent"]] + [COLORS["red"] if v > bac else COLORS["green"] for v in eac_vals[1:]]
            fig = go.Figure(go.Bar(x=eac_labels, y=eac_vals, marker_color=eac_colors,                                   text=[dollar(v) for v in eac_vals], textposition="outside",
                                   textfont=dict(color="#f1f5f9", size=11)))
            fig.add_hline(y=bac, line_dash="dash", line_color=COLORS["accent"], line_width=1,
                          annotation_text="BAC", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="EAC Forecast Comparison", yaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            var_labels = ["Cost Variance (CV)", "Schedule Variance (SV)", "VAC"]
            var_vals = [d["cv"], d["sv"], d["vac"] or 0]
            var_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in var_vals]
            fig = go.Figure(go.Bar(y=var_labels, x=var_vals, orientation="h", marker_color=var_colors,
                                                      text=[dollar(v) for v in var_vals], textposition="outside",
                                   textfont=dict(color="#f1f5f9", size=11)))
            fig.add_vline(x=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="Variance Analysis (CV, SV, VAC)", xaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 4: Progress donuts + Remaining
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure(go.Pie(
                values=[d["pct_complete"], 100 - d["pct_complete"]],
                labels=["Complete", "Remaining"], hole=0.7,
                marker_colors=[COLORS["green"], "#1e293b"],
                textinfo="none",
            ))
            fig.add_annotation(text=f'{d["pct_complete"]:.1f}%', x=0.5, y=0.55, font_size=28,
                               font_color="#f1f5f9", showarrow=False, font_family="Segoe UI")
            fig.add_annotation(text="Work Done", x=0.5, y=0.4, font_size=12,
                               font_color="#94a3b8", showarrow=False)
            fig.update_layout(**pl(title="Work Progress", showlegend=True,
                                   legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            spent_color = COLORS["red"] if d["pct_spent"] > d["pct_complete"] else COLORS["accent"]
            fig = go.Figure(go.Pie(
                values=[d["pct_spent"], 100 - d["pct_spent"]],
                labels=["Spent", "Remaining"], hole=0.7,
                marker_colors=[spent_color, "#1e293b"],
                textinfo="none",
            ))
            fig.add_annotation(text=f'{d["pct_spent"]:.1f}%', x=0.5, y=0.55, font_size=28,
                               font_color=spent_color, showarrow=False, font_family="Segoe UI")
            fig.add_annotation(text="Budget Used", x=0.5, y=0.4, font_size=12,
                               font_color="#94a3b8", showarrow=False)
            fig.update_layout(**pl(title="Budget Consumption", showlegend=True,
                                   legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")))
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            rem_work = bac - ev_v
            rem_budget = bac - ac_v
            fig = go.Figure(go.Bar(
                x=["Remaining\nWork", "Remaining\nBudget", "ETC\n(Forecast)"],
                y=[rem_work, rem_budget, d["etc"] or 0],
                marker_color=[COLORS["purple"],
                              COLORS["green"] if rem_budget >= rem_work else COLORS["red"],
                              COLORS["cyan"]],
                text=[dollar(v) for v in [rem_work, rem_budget, d["etc"] or 0]],
                textposition="outside", textfont=dict(color="#f1f5f9", size=11),
            ))
            fig.update_layout(**pl(title="Remaining Work vs Budget", yaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 5: Waterfall + Health Radar
        c1, c2 = st.columns(2)
        with c1:
            wf_labels = ["BAC", "PV", "EV", "AC", "CV", "SV", "EAC", "ETC", "VAC"]
            wf_vals = [bac, pv_v, ev_v, ac_v, d["cv"], d["sv"], d["eac_cpi"] or 0, d["etc"] or 0, d["vac"] or 0]
            wf_colors = [COLORS["accent"], COLORS["purple"], COLORS["green"], COLORS["yellow"],
                         COLORS["green"] if d["cv"] >= 0 else COLORS["red"],
                         COLORS["green"] if d["sv"] >= 0 else COLORS["red"],
                         COLORS["cyan"], COLORS["orange"],
                         COLORS["green"] if (d["vac"] or 0) >= 0 else COLORS["red"]]
            fig = go.Figure(go.Bar(x=wf_labels, y=wf_vals, marker_color=wf_colors,
                                   text=[dollar(v) for v in wf_vals], textposition="outside",
                                   textfont=dict(color="#f1f5f9", size=9)))
            fig.update_layout(**pl(title="All Values Waterfall"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cpi_v = d["cpi"] or 0
            spi_v = d["spi"] or 0
            cost_eff = min(ev_v / ac_v, 2) if ac_v > 0 else 0
            budget_health = max(0, min(2, 1 + (bac - (d["eac_cpi"] or bac)) / bac))
            comp_rate = d["pct_complete"] / 100
            tcpi_h = min(2, 1 / (d["tcpi_bac"])) if d["tcpi_bac"] and d["tcpi_bac"] > 0 else 0
            cats = ["Cost Perf (CPI)", "Schedule Perf (SPI)", "Cost Efficiency",
                    "Budget Health", "Completion", "TCPI Feasibility"]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=[min(cpi_v,2), min(spi_v,2), cost_eff, budget_health, comp_rate, tcpi_h],
                                          theta=cats, fill="toself", name="Project", line_color=COLORS["accent"], opacity=0.6))
            fig.add_trace(go.Scatterpolar(r=[1,1,1,1,1,1], theta=cats, fill="none", name="Ideal (1.0)",
                                          line=dict(color=COLORS["green"], dash="dash", width=1)))
            fig.update_layout(**pl(title="Project Health Radar", polar=dict(
                bgcolor="#0f172a", radialaxis=dict(range=[0,2], gridcolor="#1e293b", visible=True, showticklabels=False),
                angularaxis=dict(gridcolor="#334155", linecolor="#334155"))))
            st.plotly_chart(fig, use_container_width=True)

        # Row 6: TCPI + Budget Stack
        c1, c2 = st.columns(2)
        with c1:
            tcpi_labels = ["Current CPI", "TCPI (BAC)", "TCPI (EAC)"]
            tcpi_vals = [d["cpi"] or 0, d["tcpi_bac"] or 0, d["tcpi_eac"] or 0]
            fig = go.Figure(go.Bar(
                x=tcpi_labels, y=tcpi_vals,
                marker_color=[COLORS["accent"],
                              COLORS["red"] if (d["tcpi_bac"] or 0) > (d["cpi"] or 0) else COLORS["green"],
                              COLORS["cyan"]],
                text=[f"{v:.4f}" for v in tcpi_vals], textposition="outside",
                textfont=dict(color="#f1f5f9", size=11),
            ))
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS["green"], line_width=1,
                          annotation_text="Ideal: 1.0", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="TCPI Target Comparison"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            overrun = max(0, (d["eac_cpi"] or 0) - bac)
            etc_clean = max(0, d["etc"] or 0)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=["Budget"], x=[ac_v], name="Spent (AC)", orientation="h", marker_color=COLORS["yellow"]))
            fig.add_trace(go.Bar(y=["Budget"], x=[etc_clean], name="ETC", orientation="h", marker_color=COLORS["accent"]))
            fig.add_trace(go.Bar(y=["Budget"], x=[overrun], name="Overrun", orientation="h", marker_color=COLORS["red"]))
            fig.add_vline(x=bac, line_dash="dash", line_color=COLORS["green"], line_width=2,
                          annotation_text="BAC", annotation_font_color=COLORS["green"])
            fig.update_layout(**pl(title="Budget Breakdown — Spent / ETC / Overrun", barmode="stack",
                                   xaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.markdown("### 💡 Interpretation")
        interps = []
        if d["cpi"] is not None:
            if d["cpi"] >= 1:
                interps.append(f'✅ **CPI = {d["cpi"]:.4f}** — Project is **under budget**. For every $1 spent, ${d["cpi"]:.2f} of value is earned.')
            else:
                interps.append(f'❌ **CPI = {d["cpi"]:.4f}** — Project is **over budget**. For every $1 spent, only ${d["cpi"]:.2f} of value is earned.')
        if d["spi"] is not None:
            if d["spi"] >= 1:
                interps.append(f'✅ **SPI = {d["spi"]:.4f}** — Project is **ahead of schedule**.')
            else:
                interps.append(f'❌ **SPI = {d["spi"]:.4f}** — Project is **behind schedule**.')
        if d["tcpi_bac"] is not None:
            if d["tcpi_bac"] > 1:
                interps.append(f'⚠️ **TCPI (BAC) = {d["tcpi_bac"]:.4f}** — Must achieve CPI of {d["tcpi_bac"]:.4f} on remaining work to finish within budget.')
            else:
                interps.append(f'✅ **TCPI (BAC) = {d["tcpi_bac"]:.4f}** — Current pace is sufficient to finish within budget.')
        if d["vac"] is not None:
            if d["vac"] >= 0:
                interps.append(f'✅ **VAC = ${d["vac"]:,.0f}** — Expected to finish **under budget** by this amount.')
            else:
                interps.append(f'❌ **VAC = ${d["vac"]:,.0f}** — Expected to **overrun budget** by ${abs(d["vac"]):,.0f}.')
        for line in interps:
            st.markdown(line)


# ═══════════════════════════════════════════════════════
#                    EMV TAB
# ═══════════════════════════════════════════════════════
with tab_emv:
    st.subheader("Expected Monetary Value (EMV) Calculator")
    st.caption("Quantify project risk by calculating EMV for threats (negative impact) and opportunities (positive impact). EMV = Probability × Impact.")

    if st.button("Load Sample Risks"):
        st.session_state["emv_risks"] = pd.DataFrame([
            {"Name": "Server outage", "Type": "Threat", "Probability (%)": 30, "Impact ($)": -50000},
            {"Name": "Key resource leaves", "Type": "Threat", "Probability (%)": 20, "Impact ($)": -80000},
            {"Name": "Vendor delay", "Type": "Threat", "Probability (%)": 40, "Impact ($)": -30000},
            {"Name": "Scope creep", "Type": "Threat", "Probability (%)": 50, "Impact ($)": -40000},
            {"Name": "Early delivery bonus", "Type": "Opportunity", "Probability (%)": 25, "Impact ($)": 60000},
            {"Name": "Reuse existing module", "Type": "Opportunity", "Probability (%)": 60, "Impact ($)": 35000},
            {"Name": "Favorable exchange rate", "Type": "Opportunity", "Probability (%)": 15, "Impact ($)": 20000},
        ])

    default_emv = st.session_state.get("emv_risks", pd.DataFrame([
        {"Name": "", "Type": "Threat", "Probability (%)": 0, "Impact ($)": 0},
        {"Name": "", "Type": "Threat", "Probability (%)": 0, "Impact ($)": 0},
        {"Name": "", "Type": "Threat", "Probability (%)": 0, "Impact ($)": 0},
    ]))

    emv_df = st.data_editor(
        default_emv, num_rows="dynamic", use_container_width=True, key="emv_editor",
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Threat", "Opportunity"], required=True),
            "Probability (%)": st.column_config.NumberColumn("Probability (%)", min_value=0, max_value=100, step=1),
            "Impact ($)": st.column_config.NumberColumn("Impact ($)", step=1000),
        },
    )

    if st.button("Calculate EMV", type="primary", use_container_width=True):
        risks = []
        for _, row in emv_df.iterrows():
            name = str(row["Name"]).strip()
            if name and row["Probability (%)"] != 0 and row["Impact ($)"] != 0:
                prob = float(row["Probability (%)"]) / 100
                impact = float(row["Impact ($)"])
                if row["Type"] == "Threat" and impact > 0:
                    impact = -impact
                risks.append(dict(name=name, type=row["Type"], probability=prob, impact=impact))
        if risks:
            st.session_state["emv_result"] = calc_emv(risks)
        else:
            st.error("Add at least one risk with a name, probability, and impact.")

    if "emv_result" in st.session_state:
        r = st.session_state["emv_result"]
        items = r["items"]

        st.markdown("---")

        # Summary metrics
        st.markdown("### Summary")
        m1, m2, m3, m4 = st.columns(4)
        total_color = "normal" if r["total_emv"] >= 0 else "inverse"
        m1.metric("Total EMV", f'${r["total_emv"]:,.0f}',
                  delta="Net Positive" if r["total_emv"] >= 0 else "Net Negative",
                  delta_color=total_color)
        m2.metric("Total Threats EMV", f'${r["total_threats"]:,.0f}')
        m3.metric("Total Opportunities EMV", f'${r["total_opportunities"]:,.0f}')
        m4.metric("Risk Items", f'{r["risk_count"]} ({r["threat_count"]}T / {r["opportunity_count"]}O)')

        # Contingency Reserve recommendation
        contingency = abs(r["total_threats"])
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.04));
                    border: 1px solid rgba(59,130,246,0.3); border-radius: 10px; padding: 20px; margin: 16px 0;">
            <div style="color: #94a3b8; font-size: 0.85rem;">Recommended Contingency Reserve (based on threat EMV)</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #3b82f6; margin-top: 6px;">
                ${contingency:,.0f}
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
                Net project risk exposure (threats + opportunities): <strong style="color:#f1f5f9;">${r['total_emv']:,.0f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts
        st.markdown("### 📊 Visual Analysis")

        # Row 1: EMV by risk item + Threat vs Opportunity pie
        c1, c2 = st.columns(2)
        with c1:
            names = [i["name"] for i in items]
            emvs = [i["emv"] for i in items]
            colors = [COLORS["red"] if i["emv"] < 0 else COLORS["green"] for i in items]
            fig = go.Figure(go.Bar(
                y=names, x=emvs, orientation="h", marker_color=colors,
                text=[f'${v:,.0f}' for v in emvs], textposition="outside",
                textfont=dict(color="#f1f5f9", size=11),
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="EMV by Risk Item", xaxis=dict(title="EMV ($)"),
                                   height=max(300, len(items) * 45 + 80)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            threat_abs = abs(r["total_threats"])
            opp_abs = abs(r["total_opportunities"])
            fig = go.Figure(go.Pie(
                values=[threat_abs, opp_abs],
                labels=[f"Threats (${threat_abs:,.0f})", f"Opportunities (${opp_abs:,.0f})"],
                hole=0.6, marker_colors=[COLORS["red"], COLORS["green"]],
                textinfo="percent+label", textfont=dict(size=11),
            ))
            fig.add_annotation(text=f'${r["total_emv"]:,.0f}', x=0.5, y=0.55, font_size=22,
                               font_color=COLORS["green"] if r["total_emv"] >= 0 else COLORS["red"],
                               showarrow=False, font_family="Segoe UI")
            fig.add_annotation(text="Net EMV", x=0.5, y=0.4, font_size=12,
                               font_color="#94a3b8", showarrow=False)
            fig.update_layout(**pl(title="Threats vs Opportunities", showlegend=True,
                                   legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: Probability vs Impact scatter + Waterfall
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for item in items:
                color = COLORS["red"] if item["emv"] < 0 else COLORS["green"]
                fig.add_trace(go.Scatter(
                    x=[item["impact"]], y=[item["probability"] * 100],
                    mode="markers+text", text=[item["name"]], textposition="top center",
                    textfont=dict(size=10, color="#94a3b8"),
                    marker=dict(size=max(12, abs(item["emv"]) / max(abs(i["emv"]) for i in items) * 40),
                                color=color, opacity=0.7, line=dict(width=1, color="#f1f5f9")),
                    showlegend=False,
                    hovertemplate=f'<b>{item["name"]}</b><br>Probability: {item["probability"]*100:.0f}%<br>Impact: ${item["impact"]:,.0f}<br>EMV: ${item["emv"]:,.0f}<extra></extra>',
                ))
            fig.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1)
            fig.update_layout(**pl(title="Probability vs Impact (bubble size = |EMV|)",
                                   xaxis=dict(title="Impact ($)"), yaxis=dict(title="Probability (%)", range=[0, 105])))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sorted_items = sorted(items, key=lambda x: x["emv"])
            wf_names = [i["name"] for i in sorted_items] + ["Net EMV"]
            wf_vals = [i["emv"] for i in sorted_items]
            cumulative = 0
            wf_base = []
            wf_bar = []
            for v in wf_vals:
                if v < 0:
                    wf_base.append(cumulative + v)
                    wf_bar.append(abs(v))
                else:
                    wf_base.append(cumulative)
                    wf_bar.append(v)
                cumulative += v
            wf_base.append(0)
            wf_bar.append(abs(cumulative))
            wf_colors = [COLORS["red"] if v < 0 else COLORS["green"] for v in wf_vals]
            wf_colors.append(COLORS["accent"])
            fig = go.Figure(go.Bar(
                x=wf_names, y=wf_bar, base=wf_base, marker_color=wf_colors,
                text=[f'${v:,.0f}' for v in wf_vals] + [f'${cumulative:,.0f}'],
                textposition="outside", textfont=dict(color="#f1f5f9", size=10),
            ))
            fig.update_layout(**pl(title="EMV Waterfall", yaxis=dict(title="EMV ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 3: Risk ranking + Cumulative EMV
        c1, c2 = st.columns(2)
        with c1:
            sorted_abs = sorted(items, key=lambda x: abs(x["emv"]), reverse=True)
            fig = go.Figure(go.Bar(
                x=[i["name"] for i in sorted_abs],
                y=[abs(i["emv"]) for i in sorted_abs],
                marker_color=[COLORS["red"] if i["emv"] < 0 else COLORS["green"] for i in sorted_abs],
                text=[f'${abs(i["emv"]):,.0f}' for i in sorted_abs],
                textposition="outside", textfont=dict(color="#f1f5f9", size=10),
            ))
            fig.update_layout(**pl(title="Risk Ranking by |EMV|", yaxis=dict(title="|EMV| ($)"),
                                   xaxis=dict(tickangle=30)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sorted_emv = sorted(items, key=lambda x: x["emv"])
            cum = 0
            cum_vals = []
            cum_labels = []
            for i in sorted_emv:
                cum += i["emv"]
                cum_vals.append(cum)
                cum_labels.append(i["name"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cum_labels, y=cum_vals, mode="lines+markers",
                line=dict(color=COLORS["accent"], width=2),
                marker=dict(size=8, color=[COLORS["red"] if v < 0 else COLORS["green"] for v in cum_vals]),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="Cumulative EMV", yaxis=dict(title="Cumulative EMV ($)"),
                                   xaxis=dict(tickangle=30)))
            st.plotly_chart(fig, use_container_width=True)

        # Risk detail table
        st.markdown("### Risk Detail Table")
        detail_df = pd.DataFrame(items)
        detail_df["probability"] = detail_df["probability"].apply(lambda x: f"{x*100:.0f}%")
        detail_df["impact"] = detail_df["impact"].apply(lambda x: f"${x:,.0f}")
        detail_df["emv"] = detail_df["emv"].apply(lambda x: f"${x:,.0f}")
        detail_df.columns = ["Risk", "Type", "Probability", "Impact", "EMV"]
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        # Interpretation
        st.markdown("### 💡 Interpretation")
        interps = []
        if r["total_emv"] < 0:
            interps.append(f'❌ **Net EMV = ${r["total_emv"]:,.0f}** — Overall risk exposure is **negative**. Budget should include a contingency reserve of at least **${contingency:,.0f}**.')
        else:
            interps.append(f'✅ **Net EMV = ${r["total_emv"]:,.0f}** — Opportunities outweigh threats. Still recommend a contingency reserve of **${contingency:,.0f}** for threat coverage.')
        highest_risk = max(items, key=lambda x: abs(x["emv"]))
        interps.append(f'⚠️ **Highest impact risk:** "{highest_risk["name"]}" with EMV of **${highest_risk["emv"]:,.0f}** (Probability: {highest_risk["probability"]*100:.0f}%, Impact: ${highest_risk["impact"]:,.0f}).')
        if r["threat_count"] > 0:
            interps.append(f'🔴 **{r["threat_count"]} threats** contributing **${r["total_threats"]:,.0f}** in expected losses.')
        if r["opportunity_count"] > 0:
            interps.append(f'🟢 **{r["opportunity_count"]} opportunities** contributing **${r["total_opportunities"]:,.0f}** in expected gains.')
        for line in interps:
            st.markdown(line)


# ═══════════════════════════════════════════════════════
#                  FINANCIAL TAB
# ═══════════════════════════════════════════════════════
with tab_fin:
    st.subheader("Financial Analysis Calculator")
    st.caption("Calculate Payback Period, NPV, IRR, BCR, ROI, Profitability Index, and Break-Even Analysis from project cash flows.")

    fin_c1, fin_c2, fin_c3 = st.columns(3)
    fin_invest = fin_c1.number_input("Initial Investment ($)", min_value=0.0, value=500000.0, step=10000.0, format="%.0f", key="fin_invest")
    fin_rate = fin_c2.number_input("Discount Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="fin_rate")
    fin_periods = fin_c3.number_input("Number of Periods", min_value=1, max_value=30, value=7, step=1, key="fin_periods")

    st.markdown("#### Break-Even Parameters *(optional)*")
    be_c1, be_c2, be_c3 = st.columns(3)
    be_fixed = be_c1.number_input("Fixed Costs ($)", min_value=0.0, value=200000.0, step=1000.0, format="%.0f", key="be_fixed")
    be_var = be_c2.number_input("Variable Cost / Unit ($)", min_value=0.0, value=50.0, step=1.0, format="%.2f", key="be_var")
    be_price = be_c3.number_input("Selling Price / Unit ($)", min_value=0.0, value=120.0, step=1.0, format="%.2f", key="be_price")

    st.markdown("#### Cash Flows per Period")

    sample_cfs = [80000, 120000, 140000, 140000, 120000, 100000, 80000]
    use_sample_fin = st.checkbox("Load sample cash flows", value=True, key="fin_sample")

    if use_sample_fin:
        default_fin = {f"Period {i+1}": {"Cash Flow ($)": sample_cfs[i] if i < len(sample_cfs) else 0} for i in range(int(fin_periods))}
    else:
        default_fin = {f"Period {i+1}": {"Cash Flow ($)": 0} for i in range(int(fin_periods))}

    fin_df = pd.DataFrame(default_fin).T
    fin_df.index.name = "Period"
    edited_fin = st.data_editor(fin_df, use_container_width=True, num_rows="fixed", key="fin_cf_editor")

    if st.button("Calculate Financials", type="primary", use_container_width=True, key="calc_fin"):
        cfs = edited_fin["Cash Flow ($)"].values.astype(float).tolist()
        if sum(abs(c) for c in cfs) == 0:
            st.error("Enter at least one non-zero cash flow.")
        else:
            st.session_state["fin_result"] = calc_financials(
                fin_invest, fin_rate, cfs, be_fixed, be_var, be_price
            )

    if "fin_result" in st.session_state:
        d = st.session_state["fin_result"]

        st.markdown("---")

        # ── Key Metrics ──
        st.markdown("### Key Financial Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NPV", f'${d["npv"]:,.0f}',
                  delta="Profitable" if d["npv"] >= 0 else "Unprofitable",
                  delta_color="normal" if d["npv"] >= 0 else "inverse")
        irr_str = f'{d["irr"]*100:.2f}%' if d["irr"] is not None else "N/A"
        m2.metric("IRR", irr_str,
                  delta="Above hurdle" if d["irr"] and d["irr"] * 100 > d["discount_rate"] else "Below hurdle",
                  delta_color="normal" if d["irr"] and d["irr"] * 100 > d["discount_rate"] else "inverse")
        m3.metric("ROI", f'{d["roi"]:.1f}%' if d["roi"] is not None else "N/A",
                  delta="Positive" if d["roi"] and d["roi"] > 0 else "Negative",
                  delta_color="normal" if d["roi"] and d["roi"] > 0 else "inverse")
        pb_str = f'{d["payback"]:.1f} periods' if d["payback"] is not None else "Never"
        m4.metric("Payback Period", pb_str)

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("BCR", f'{d["bcr"]:.3f}' if d["bcr"] is not None else "N/A",
                  delta="Benefits > Costs" if d["bcr"] and d["bcr"] >= 1 else "Costs > Benefits",
                  delta_color="normal" if d["bcr"] and d["bcr"] >= 1 else "inverse")
        m6.metric("Profitability Index", f'{d["pi"]:.3f}' if d["pi"] is not None else "N/A",
                  delta="Accept" if d["pi"] and d["pi"] >= 1 else "Reject",
                  delta_color="normal" if d["pi"] and d["pi"] >= 1 else "inverse")
        dpb_str = f'{d["disc_payback"]:.1f} periods' if d["disc_payback"] is not None else "Never"
        m7.metric("Discounted Payback", dpb_str)
        m8.metric("Total Returns", f'${d["total_returns"]:,.0f}')

        # Break-Even metrics
        if d["bep_units"] is not None:
            st.markdown("### Break-Even Analysis")
            be1, be2, be3, be4 = st.columns(4)
            be1.metric("Break-Even (Units)", f'{d["bep_units"]:,.0f}')
            be2.metric("Break-Even (Revenue)", f'${d["bep_dollars"]:,.0f}')
            be3.metric("Contribution Margin / Unit", f'${d["contribution_margin"]:,.2f}')
            cm_ratio = d["contribution_margin"] / d["selling_price_per_unit"] * 100
            be4.metric("CM Ratio", f'{cm_ratio:.1f}%')

        # ── CHARTS ──
        st.markdown("---")
        st.markdown("### 📊 Visual Analysis")

        # Row 1: Cumulative Cash Flow + NPV Sensitivity
        c1, c2 = st.columns(2)
        with c1:
            periods = list(range(len(d["cash_flows"])))
            p_labels = [f'P{i+1}' for i in periods]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=["P0"] + p_labels, y=[-d["initial_investment"]] + d["cum_cf"],
                mode="lines+markers", name="Undiscounted",
                line=dict(color=COLORS["accent"], width=3), marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=["P0"] + p_labels, y=[-d["initial_investment"]] + d["cum_dcf"],
                mode="lines+markers", name="Discounted",
                line=dict(color=COLORS["purple"], width=3, dash="dash"), marker=dict(size=8),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS["green"], line_width=1,
                          annotation_text="Break-even", annotation_font_color="#94a3b8")
            if d["payback"] is not None:
                fig.add_vline(x=d["payback"], line_dash="dot", line_color=COLORS["yellow"], line_width=1,
                              annotation_text=f'Payback: {d["payback"]:.1f}', annotation_font_color=COLORS["yellow"])
            fig.update_layout(**pl(title="Cumulative Cash Flow (Payback Visualization)",
                                   yaxis=dict(title="Cumulative ($)"), xaxis=dict(title="Period")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sens = d["npv_sensitivity"]
            rates = [s["rate"] for s in sens]
            npvs = [s["npv"] for s in sens]
            colors_sens = [COLORS["green"] if v >= 0 else COLORS["red"] for v in npvs]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rates, y=npvs, mode="lines+markers",
                line=dict(color=COLORS["accent"], width=2), marker=dict(size=5, color=colors_sens),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            if d["irr"] is not None:
                fig.add_vline(x=d["irr"] * 100, line_dash="dot", line_color=COLORS["yellow"], line_width=2,
                              annotation_text=f'IRR: {d["irr"]*100:.1f}%', annotation_font_color=COLORS["yellow"])
            fig.add_vline(x=d["discount_rate"], line_dash="dot", line_color=COLORS["red"], line_width=1,
                          annotation_text=f'Discount: {d["discount_rate"]}%', annotation_font_color=COLORS["red"])
            fig.update_layout(**pl(title="NPV Sensitivity (shows IRR where NPV = 0)",
                                   xaxis=dict(title="Discount Rate (%)"), yaxis=dict(title="NPV ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: Period Cash Flows + Discounted vs Undiscounted
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            cf_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in d["cash_flows"]]
            fig.add_trace(go.Bar(
                x=p_labels, y=d["cash_flows"], marker_color=cf_colors,
                text=[f'${v:,.0f}' for v in d["cash_flows"]], textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="Cash Flow per Period", yaxis=dict(title="Cash Flow ($)"),
                                   xaxis=dict(title="Period")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=p_labels, y=d["cash_flows"], name="Undiscounted",
                                 marker_color=COLORS["accent"], opacity=0.7))
            fig.add_trace(go.Bar(x=p_labels, y=d["dcf"], name="Discounted (PV)",
                                 marker_color=COLORS["purple"], opacity=0.7))
            fig.update_layout(**pl(title="Undiscounted vs Discounted Cash Flows", barmode="group",
                                   yaxis=dict(title="Amount ($)"), xaxis=dict(title="Period")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 3: Financial Metrics Radar + Key Ratios Bar
        c1, c2 = st.columns(2)
        with c1:
            radar_cats = ["NPV Health", "IRR vs Hurdle", "ROI", "BCR", "PI", "Payback Speed"]
            npv_score = min(2, max(0, 1 + d["npv"] / d["initial_investment"])) if d["initial_investment"] > 0 else 0
            irr_score = min(2, (d["irr"] * 100 / d["discount_rate"])) if d["irr"] and d["discount_rate"] > 0 else 0
            roi_score = min(2, max(0, d["roi"] / 100 + 1)) if d["roi"] is not None else 0
            bcr_score = min(2, d["bcr"]) if d["bcr"] else 0
            pi_score = min(2, d["pi"]) if d["pi"] else 0
            n_periods = len(d["cash_flows"])
            pb_score = min(2, (n_periods - (d["payback"] or n_periods)) / n_periods * 2) if d["payback"] else 0
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[npv_score, irr_score, roi_score, bcr_score, pi_score, pb_score],
                theta=radar_cats, fill="toself", name="Project",
                line_color=COLORS["accent"], opacity=0.6,
            ))
            fig.add_trace(go.Scatterpolar(
                r=[1, 1, 1, 1, 1, 1], theta=radar_cats, fill="none", name="Baseline (1.0)",
                line=dict(color=COLORS["green"], dash="dash", width=1),
            ))
            fig.update_layout(**pl(title="Financial Health Radar", polar=dict(
                bgcolor="#0f172a", radialaxis=dict(range=[0, 2], gridcolor="#1e293b", visible=True, showticklabels=False),
                angularaxis=dict(gridcolor="#334155", linecolor="#334155"))))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            ratio_labels = ["BCR", "Profitability\nIndex", "ROI / 100"]
            ratio_vals = [d["bcr"] or 0, d["pi"] or 0, (d["roi"] or 0) / 100]
            ratio_colors = [
                COLORS["green"] if (d["bcr"] or 0) >= 1 else COLORS["red"],
                COLORS["green"] if (d["pi"] or 0) >= 1 else COLORS["red"],
                COLORS["green"] if (d["roi"] or 0) > 0 else COLORS["red"],
            ]
            fig = go.Figure(go.Bar(
                x=ratio_labels, y=ratio_vals, marker_color=ratio_colors,
                text=[f'{v:.3f}' for v in ratio_vals], textposition="outside",
                textfont=dict(color="#f1f5f9", size=12),
            ))
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS["yellow"], line_width=1,
                          annotation_text="Threshold: 1.0", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="Key Financial Ratios", yaxis=dict(title="Ratio")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 4: Investment Breakdown + ROI Waterfall
        c1, c2 = st.columns(2)
        with c1:
            net_gain = d["total_returns"] - d["initial_investment"]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=["Investment"], x=[d["initial_investment"]], name="Initial Investment",
                                 orientation="h", marker_color=COLORS["red"]))
            pv_total = sum(d["dcf"])
            fig.add_trace(go.Bar(y=["PV Returns"], x=[pv_total], name="PV of Returns",
                                 orientation="h", marker_color=COLORS["green"] if pv_total >= d["initial_investment"] else COLORS["yellow"]))
            fig.update_layout(**pl(title="Investment vs PV of Returns", xaxis=dict(title="Amount ($)"),
                                   barmode="group"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            wf_labels = ["Investment", "Total Returns", "Net Profit", "NPV"]
            wf_vals = [-d["initial_investment"], d["total_returns"], d["total_returns"] - d["initial_investment"], d["npv"]]
            wf_colors = [COLORS["red"], COLORS["green"],
                         COLORS["green"] if wf_vals[2] >= 0 else COLORS["red"],
                         COLORS["green"] if d["npv"] >= 0 else COLORS["red"]]
            fig = go.Figure(go.Bar(
                x=wf_labels, y=wf_vals, marker_color=wf_colors,
                text=[f'${v:,.0f}' for v in wf_vals], textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="Investment Waterfall", yaxis=dict(title="Amount ($)")))
            st.plotly_chart(fig, use_container_width=True)

        # Row 5: Break-Even Chart (if applicable)
        if d["bep_units"] is not None:
            st.markdown("### Break-Even Chart")
            max_units = int(d["bep_units"] * 2.5)
            units_range = list(range(0, max_units + 1, max(1, max_units // 50)))
            total_costs = [d["fixed_costs"] + d["variable_cost_per_unit"] * u for u in units_range]
            total_revenue = [d["selling_price_per_unit"] * u for u in units_range]
            fixed_line = [d["fixed_costs"]] * len(units_range)
            variable_costs = [d["variable_cost_per_unit"] * u for u in units_range]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=units_range, y=total_revenue, name="Total Revenue",
                                     line=dict(color=COLORS["green"], width=3), mode="lines"))
            fig.add_trace(go.Scatter(x=units_range, y=total_costs, name="Total Costs",
                                     line=dict(color=COLORS["red"], width=3), mode="lines"))
            fig.add_trace(go.Scatter(x=units_range, y=fixed_line, name="Fixed Costs",
                                     line=dict(color=COLORS["yellow"], width=2, dash="dash"), mode="lines"))
            fig.add_trace(go.Scatter(x=units_range, y=variable_costs, name="Variable Costs",
                                     line=dict(color=COLORS["orange"], width=2, dash="dot"), mode="lines"))
            fig.add_vline(x=d["bep_units"], line_dash="dot", line_color=COLORS["accent"], line_width=2,
                          annotation_text=f'BEP: {d["bep_units"]:,.0f} units', annotation_font_color=COLORS["accent"])
            fig.add_trace(go.Scatter(x=[d["bep_units"]], y=[d["bep_dollars"]],
                                     mode="markers", marker=dict(size=14, color=COLORS["accent"], symbol="star"),
                                     name=f'Break-Even Point', showlegend=True))
            fig.update_layout(**pl(title="Break-Even Analysis",
                                   xaxis=dict(title="Units Sold"), yaxis=dict(title="Amount ($)"),
                                   height=450, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")))
            st.plotly_chart(fig, use_container_width=True)

        # Summary Table
        st.markdown("### Period Detail Table")
        detail_data = {
            "Period": [f"P{i+1}" for i in range(len(d["cash_flows"]))],
            "Cash Flow": [f'${v:,.0f}' for v in d["cash_flows"]],
            "Discounted CF": [f'${v:,.0f}' for v in d["dcf"]],
            "Cumulative CF": [f'${v:,.0f}' for v in d["cum_cf"]],
            "Cumulative DCF": [f'${v:,.0f}' for v in d["cum_dcf"]],
        }
        st.dataframe(pd.DataFrame(detail_data).set_index("Period"), use_container_width=True)

        # Interpretation
        st.markdown("### 💡 Interpretation")
        interps = []
        if d["npv"] >= 0:
            interps.append(f'✅ **NPV = ${d["npv"]:,.0f}** — Project adds value. Accept the investment.')
        else:
            interps.append(f'❌ **NPV = ${d["npv"]:,.0f}** — Project destroys value at {d["discount_rate"]}% discount rate. Consider rejecting.')
        if d["irr"] is not None:
            if d["irr"] * 100 > d["discount_rate"]:
                interps.append(f'✅ **IRR = {d["irr"]*100:.2f}%** — Exceeds the hurdle rate of {d["discount_rate"]}%. Investment is attractive.')
            else:
                interps.append(f'❌ **IRR = {d["irr"]*100:.2f}%** — Below the hurdle rate of {d["discount_rate"]}%.')
        if d["roi"] is not None:
            interps.append(f'{"✅" if d["roi"]>0 else "❌"} **ROI = {d["roi"]:.1f}%** — For every $1 invested, ${1+d["roi"]/100:.2f} is returned.')
        if d["bcr"] is not None:
            interps.append(f'{"✅" if d["bcr"]>=1 else "❌"} **BCR = {d["bcr"]:.3f}** — {"Benefits exceed costs." if d["bcr"]>=1 else "Costs exceed benefits."}')
        if d["pi"] is not None:
            interps.append(f'{"✅" if d["pi"]>=1 else "❌"} **PI = {d["pi"]:.3f}** — {"Project creates ${:.2f} of value per $1 invested.".format(d["pi"]) if d["pi"]>=1 else "Project returns less than invested."}')
        if d["payback"] is not None:
            interps.append(f'⏱️ **Payback Period = {d["payback"]:.1f} periods** — Investment recovered in {d["payback"]:.1f} periods out of {len(d["cash_flows"])}.')
        else:
            interps.append(f'⚠️ **Payback Period = Never** — Investment is not recovered within the project timeline.')
        if d["bep_units"] is not None:
            interps.append(f'📊 **Break-Even = {d["bep_units"]:,.0f} units (${d["bep_dollars"]:,.0f} revenue)** — Must sell at least this many units at ${d["selling_price_per_unit"]:,.2f} each to cover costs.')
        for line in interps:
            st.markdown(line)


# ═══════════════════════════════════════════════════════
#                   S-CURVE TAB
# ═══════════════════════════════════════════════════════
with tab_scurve:
    st.subheader("Period-by-Period Data Entry")
    st.caption("Enter **cumulative** PV, EV, and AC values for each period. The tool generates S-Curve, CPI/SPI trends, variance trends, and more.")

    sc1, sc2 = st.columns(2)
    sc_bac = sc1.number_input("BAC (Budget at Completion)", min_value=0.0, value=100000.0, step=1000.0, key="sc_bac", format="%.0f")
    sc_label = sc2.text_input("Period Label", value="Month", key="sc_label")

    num_periods = st.slider("Number of Periods", min_value=2, max_value=24, value=12, key="sc_periods")

    # Sample data
    sample_pv = [2000,6000,14000,24000,38000,54000,68000,80000,88000,94000,98000,100000]
    sample_ev = [1500,5000,12000,20000,32000,46000,58000,72000,82000,90000,96000,100000]
    sample_ac = [2500,8000,16000,28000,42000,58000,74000,88000,98000,106000,112000,120000]

    use_sample = st.checkbox("Load sample data (12-month project)", value=True)

    st.markdown("#### Enter Cumulative Values")
    period_labels = [f"{sc_label} {i+1}" for i in range(num_periods)]

    # Build data entry using dataframe editor
    if use_sample:
        default_data = {
            period_labels[i]: {
                "PV": sample_pv[i] if i < len(sample_pv) else 0,
                "EV": sample_ev[i] if i < len(sample_ev) else 0,
                "AC": sample_ac[i] if i < len(sample_ac) else 0,
            } for i in range(num_periods)
        }
    else:
        default_data = {
            period_labels[i]: {"PV": 0, "EV": 0, "AC": 0}
            for i in range(num_periods)
        }

    df_input = pd.DataFrame(default_data).T
    df_input.index.name = "Period"
    edited_df = st.data_editor(df_input, use_container_width=True, num_rows="fixed")

    if st.button("Generate S-Curve Charts", type="primary", use_container_width=True, key="gen_scurve"):
        pv_arr = edited_df["PV"].values.astype(float)
        ev_arr = edited_df["EV"].values.astype(float)
        ac_arr = edited_df["AC"].values.astype(float)
        labels = list(edited_df.index)
        n_p = len(labels)

        # Derived metrics
        cpi_arr = [ev_arr[i] / ac_arr[i] if ac_arr[i] != 0 else 0 for i in range(n_p)]
        spi_arr = [ev_arr[i] / pv_arr[i] if pv_arr[i] != 0 else 0 for i in range(n_p)]
        cv_arr = [ev_arr[i] - ac_arr[i] for i in range(n_p)]
        sv_arr = [ev_arr[i] - pv_arr[i] for i in range(n_p)]
        eac_arr = [sc_bac / cpi_arr[i] if cpi_arr[i] != 0 else 0 for i in range(n_p)]
        pct_comp = [(ev_arr[i] / sc_bac) * 100 for i in range(n_p)]
        pct_spent = [(ac_arr[i] / sc_bac) * 100 for i in range(n_p)]
        inc_pv = [pv_arr[0]] + [pv_arr[i] - pv_arr[i-1] for i in range(1, n_p)]
        inc_ev = [ev_arr[0]] + [ev_arr[i] - ev_arr[i-1] for i in range(1, n_p)]
        inc_ac = [ac_arr[0]] + [ac_arr[i] - ac_arr[i-1] for i in range(1, n_p)]

        st.markdown("---")

        # ── 1. S-CURVE MAIN ──
        st.markdown("### S-Curve — PV vs EV vs AC Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=pv_arr, name="Planned Value (PV)", mode="lines+markers",
                                 line=dict(color="#94a3b8", width=3), marker=dict(size=8, color="#94a3b8")))
        fig.add_trace(go.Scatter(x=labels, y=ev_arr, name="Earned Value (EV)", mode="lines+markers",
                                 line=dict(color=COLORS["green"], width=3), marker=dict(size=8, color=COLORS["green"])))
        fig.add_trace(go.Scatter(x=labels, y=ac_arr, name="Actual Cost (AC)", mode="lines+markers",
                                 line=dict(color=COLORS["red"], width=3), marker=dict(size=8, color=COLORS["red"])))
        fig.add_trace(go.Scatter(x=labels, y=[sc_bac]*n_p, name="BAC (Budget)", mode="lines",
                                 line=dict(color=COLORS["accent"], width=2, dash="dash")))
        fig.update_layout(**pl(
            title="Cumulative S-Curve", height=450,
            yaxis=dict(title="Spending ($)", gridcolor="#1e293b"),
            xaxis=dict(title=sc_label, gridcolor="rgba(30,41,59,0.5)"),
            hovermode="x unified",
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ── 2 & 3. CPI / SPI Trend ──
        st.markdown("### Performance Trends Over Time")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=labels, y=cpi_arr, name="CPI", mode="lines+markers",
                                     line=dict(color=COLORS["accent"], width=2), marker=dict(size=6,
                                     color=[COLORS["green"] if v >= 1 else COLORS["red"] for v in cpi_arr])))
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS["green"], line_width=1,
                          annotation_text="Target: 1.0", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="CPI Trend", yaxis=dict(title="CPI"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=labels, y=spi_arr, name="SPI", mode="lines+markers",
                                     line=dict(color=COLORS["purple"], width=2), marker=dict(size=6,
                                     color=[COLORS["green"] if v >= 1 else COLORS["red"] for v in spi_arr])))
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS["green"], line_width=1,
                          annotation_text="Target: 1.0", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="SPI Trend", yaxis=dict(title="SPI"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        # ── 4 & 5. CV / SV Trend ──
        st.markdown("### Variance Trends Over Time")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=labels, y=cv_arr, name="CV",
                                   marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in cv_arr],
                                  ))
            fig.add_hline(y=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="Cost Variance (CV) Trend", yaxis=dict(title="CV ($)"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(go.Bar(x=labels, y=sv_arr, name="SV",
                                   marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in sv_arr],
                                  ))
            fig.add_hline(y=0, line_dash="dash", line_color="#f1f5f9", line_width=1)
            fig.update_layout(**pl(title="Schedule Variance (SV) Trend", yaxis=dict(title="SV ($)"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        # ── 6 & 7. Incremental + % trends ──
        st.markdown("### Period-over-Period Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=inc_pv, name="PV (incremental)", marker_color="#64748b"))
            fig.add_trace(go.Bar(x=labels, y=inc_ev, name="EV (incremental)", marker_color=COLORS["green"]))
            fig.add_trace(go.Bar(x=labels, y=inc_ac, name="AC (incremental)", marker_color=COLORS["red"]))
            fig.update_layout(**pl(title="Incremental Spend per Period", barmode="group",
                                   yaxis=dict(title="Amount ($)"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=labels, y=pct_comp, name="% Complete (EV/BAC)", mode="lines+markers",
                                     line=dict(color=COLORS["green"], width=2), marker=dict(size=6), fill="tozeroy",
                                     fillcolor="rgba(34,197,94,0.1)"))
            fig.add_trace(go.Scatter(x=labels, y=pct_spent, name="% Spent (AC/BAC)", mode="lines+markers",
                                     line=dict(color=COLORS["red"], width=2), marker=dict(size=6), fill="tozeroy",
                                     fillcolor="rgba(239,68,68,0.1)"))
            fig.add_hline(y=100, line_dash="dash", line_color=COLORS["accent"], line_width=1,
                          annotation_text="100%", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="Cumulative % Complete vs % Spent",
                                   yaxis=dict(title="%"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        # ── 8 & 9. EAC + CPI vs SPI ──
        st.markdown("### Forecasting Trends")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=labels, y=eac_arr, name="EAC (BAC/CPI)", mode="lines+markers",
                                     line=dict(color=COLORS["orange"], width=2), marker=dict(size=6),
                                     fill="tozeroy", fillcolor="rgba(249,115,22,0.1)"))
            fig.add_hline(y=sc_bac, line_dash="dash", line_color=COLORS["accent"], line_width=2,
                          annotation_text="BAC", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="EAC Forecast Trend (BAC/CPI)",
                                   yaxis=dict(title="EAC ($)"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=labels, y=cpi_arr, name="CPI", mode="lines+markers",
                                     line=dict(color=COLORS["accent"], width=2), marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=labels, y=spi_arr, name="SPI", mode="lines+markers",
                                     line=dict(color=COLORS["purple"], width=2), marker=dict(size=6)))
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS["green"], line_width=1,
                          annotation_text="Ideal: 1.0", annotation_font_color="#94a3b8")
            fig.update_layout(**pl(title="CPI vs SPI Combined",
                                   yaxis=dict(title="Index Value"), xaxis=dict(title=sc_label)))
            st.plotly_chart(fig, use_container_width=True)

        # ── Summary table ──
        st.markdown("### Period Summary Table")
        summary = pd.DataFrame({
            "Period": labels,
            "PV": [f"${v:,.0f}" for v in pv_arr],
            "EV": [f"${v:,.0f}" for v in ev_arr],
            "AC": [f"${v:,.0f}" for v in ac_arr],
            "CV": [f"${v:,.0f}" for v in cv_arr],
            "SV": [f"${v:,.0f}" for v in sv_arr],
            "CPI": [f"{v:.4f}" for v in cpi_arr],
            "SPI": [f"{v:.4f}" for v in spi_arr],
            "EAC": [f"${v:,.0f}" for v in eac_arr],
            "% Done": [f"{v:.1f}%" for v in pct_comp],
            "% Spent": [f"{v:.1f}%" for v in pct_spent],
        }).set_index("Period")
        st.dataframe(summary, use_container_width=True)


# ═══════════════════════════════════════════════════════
#                  CRITICAL PATH TAB
# ═══════════════════════════════════════════════════════
with tab_cp:
    st.subheader("Task Definitions")
    st.caption("Define tasks with durations and predecessor IDs (comma-separated). Leave predecessors empty for starting tasks.")

    if st.button("Load Sample Tasks"):
        st.session_state["cp_tasks"] = pd.DataFrame([
            {"ID": "A", "Name": "Requirements", "Duration": 5, "Predecessors": ""},
            {"ID": "B", "Name": "Design", "Duration": 8, "Predecessors": "A"},
            {"ID": "C", "Name": "Database Setup", "Duration": 4, "Predecessors": "A"},
            {"ID": "D", "Name": "Backend Dev", "Duration": 12, "Predecessors": "B"},
            {"ID": "E", "Name": "Frontend Dev", "Duration": 10, "Predecessors": "B,C"},
            {"ID": "F", "Name": "Integration", "Duration": 6, "Predecessors": "D,E"},
            {"ID": "G", "Name": "Testing", "Duration": 5, "Predecessors": "F"},
            {"ID": "H", "Name": "Deployment", "Duration": 3, "Predecessors": "G"},
        ])

    default_tasks = st.session_state.get("cp_tasks", pd.DataFrame([
        {"ID": "", "Name": "", "Duration": 0, "Predecessors": ""},
        {"ID": "", "Name": "", "Duration": 0, "Predecessors": ""},
        {"ID": "", "Name": "", "Duration": 0, "Predecessors": ""},
    ]))

    task_df = st.data_editor(default_tasks, num_rows="dynamic", use_container_width=True, key="task_editor")

    if st.button("Find Critical Path", type="primary", use_container_width=True):
        tasks = []
        for _, row in task_df.iterrows():
            tid = str(row["ID"]).strip()
            if tid and row["Duration"] > 0:
                preds = [p.strip() for p in str(row["Predecessors"]).split(",") if p.strip()]
                tasks.append(dict(id=tid, name=str(row["Name"]) or tid, duration=float(row["Duration"]),
                                  predecessors=preds))
        if tasks:
            result, err = calc_critical_path(tasks)
            if err:
                st.error(err)
            else:
                st.session_state["cp_result"] = result
        else:
            st.error("Add at least one task with a valid ID and duration.")

    if "cp_result" in st.session_state:
        r = st.session_state["cp_result"]
        tasks = r["tasks"]

        st.markdown("---")

        # Critical path banner
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
                    border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; padding: 20px; margin: 16px 0;">
            <div style="color: #94a3b8; font-size: 0.85rem;">Critical Path (longest sequence — zero float)</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #ef4444; margin-top: 6px; letter-spacing: 0.5px;">
                {' → '.join(r['critical_path'])}
            </div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 6px;">
                Project Duration: <strong style="color:#f1f5f9;">{r['project_duration']} units</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gantt Chart ──
        st.markdown("### Gantt Chart")
        fig = go.Figure()
        for t in reversed(tasks):
            label = f'{t["id"]} — {t["name"]}'
            color = COLORS["red"] if t["critical"] else COLORS["accent"]
            fig.add_trace(go.Bar(y=[label], x=[t["duration"]], base=[t["es"]], orientation="h",
                                 marker_color=color, name="Critical" if t["critical"] else "Non-Critical",
                                 showlegend=False, hovertemplate=f'<b>{label}</b><br>ES={t["es"]} → EF={t["ef"]}<br>Duration={t["duration"]}<extra></extra>'))
            if not t["critical"] and t["total_float"] > 0:
                fig.add_trace(go.Bar(y=[label], x=[t["total_float"]], base=[t["ef"]], orientation="h",
                                     marker_color="rgba(234,179,8,0.35)", name="Float",
                                     showlegend=False, hovertemplate=f'Float: {t["total_float"]}<extra></extra>'))
        # Legend entries
        fig.add_trace(go.Bar(y=[None], x=[0], marker_color=COLORS["red"], name="Critical", showlegend=True))
        fig.add_trace(go.Bar(y=[None], x=[0], marker_color=COLORS["accent"], name="Non-Critical", showlegend=True))
        fig.add_trace(go.Bar(y=[None], x=[0], marker_color="rgba(234,179,8,0.35)", name="Float", showlegend=True))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8", family="Segoe UI, system-ui, sans-serif"),
            margin=dict(l=50, r=30, t=40, b=50),
            height=max(300, len(tasks) * 40 + 80),
            barmode="overlay",
            xaxis=dict(title="Time (units)", gridcolor="#1e293b", zerolinecolor="#334155"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", zerolinecolor="#334155"),
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Analysis Charts ──
        st.markdown("### Task Analysis Charts")
        labels = [f'{t["id"]} — {t["name"]}' for t in tasks]

        # Duration + Float
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=labels, y=[t["duration"] for t in tasks],
                marker_color=[COLORS["red"] if t["critical"] else COLORS["accent"] for t in tasks],
            ))
            fig.update_layout(**pl(title="Task Duration Comparison", yaxis=dict(title="Duration"), xaxis=dict(tickangle=30)))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(go.Bar(
                x=labels, y=[t["total_float"] for t in tasks],
                marker_color=[COLORS["red"] if t["critical"] else COLORS["yellow"] for t in tasks],
            ))
            fig.update_layout(**pl(title="Total Float by Task", yaxis=dict(title="Float"), xaxis=dict(tickangle=30)))
            st.plotly_chart(fig, use_container_width=True)

        # Pie + ES/EF/LS/LF
        c1, c2 = st.columns(2)
        with c1:
            crit_count = sum(1 for t in tasks if t["critical"])
            non_count = len(tasks) - crit_count
            crit_dur = sum(t["duration"] for t in tasks if t["critical"])
            non_dur = sum(t["duration"] for t in tasks if not t["critical"])
            fig = go.Figure(go.Pie(
                values=[crit_dur, non_dur],
                labels=[f"Critical ({crit_count} tasks, {crit_dur} units)",
                        f"Non-Critical ({non_count} tasks, {non_dur} units)"],
                hole=0.5, marker_colors=[COLORS["red"], COLORS["accent"]],
            ))
            fig.update_layout(**pl(title="Critical vs Non-Critical Split",
                                   legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            ids = [t["id"] for t in tasks]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ids, y=[t["es"] for t in tasks], name="ES", marker_color=COLORS["accent"]))
            fig.add_trace(go.Bar(x=ids, y=[t["ef"] for t in tasks], name="EF", marker_color=COLORS["green"]))
            fig.add_trace(go.Bar(x=ids, y=[t["ls"] for t in tasks], name="LS", marker_color=COLORS["purple"]))
            fig.add_trace(go.Bar(x=ids, y=[t["lf"] for t in tasks], name="LF", marker_color=COLORS["orange"]))
            fig.update_layout(**pl(title="ES / EF / LS / LF Timeline", barmode="group", yaxis=dict(title="Time")))
            st.plotly_chart(fig, use_container_width=True)

        # Duration Distribution + Resource Density
        c1, c2 = st.columns(2)
        with c1:
            durs = [t["duration"] for t in tasks]
            fig = go.Figure(go.Histogram(x=durs, nbinsx=5, marker_color=COLORS["purple"],))
            fig.update_layout(**pl(title="Duration Distribution", xaxis=dict(title="Duration"), yaxis=dict(title="Task Count")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            proj_dur = int(r["project_duration"])
            density_crit = []
            density_non = []
            for t_point in range(proj_dur):
                c_active = sum(1 for t in tasks if t_point >= t["es"] and t_point < t["ef"] and t["critical"])
                n_active = sum(1 for t in tasks if t_point >= t["es"] and t_point < t["ef"] and not t["critical"])
                density_crit.append(c_active)
                density_non.append(n_active)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(proj_dur)), y=density_crit, name="Critical Tasks",
                                     fill="tozeroy", line_color=COLORS["red"], fillcolor="rgba(239,68,68,0.15)"))
            fig.add_trace(go.Scatter(x=list(range(proj_dur)), y=density_non, name="Non-Critical Tasks",
                                     fill="tozeroy", line_color=COLORS["accent"], fillcolor="rgba(59,130,246,0.15)"))
            total = [density_crit[i]+density_non[i] for i in range(proj_dur)]
            fig.add_trace(go.Scatter(x=list(range(proj_dur)), y=total, name="Total Active",
                                     line=dict(color="#f1f5f9", width=1, dash="dash")))
            fig.update_layout(**pl(title="Resource Density Over Time",
                                   xaxis=dict(title="Time (units)"), yaxis=dict(title="Active Tasks")))
            st.plotly_chart(fig, use_container_width=True)

        # Schedule table
        st.markdown("### Task Schedule Table")
        sched_df = pd.DataFrame(tasks)
        sched_df["Status"] = sched_df["critical"].map({True: "🔴 CRITICAL", False: "🟢 Has Float"})
        sched_df = sched_df[["id", "name", "duration", "es", "ef", "ls", "lf", "total_float", "Status"]]
        sched_df.columns = ["ID", "Name", "Duration", "ES", "EF", "LS", "LF", "Total Float", "Status"]
        st.dataframe(sched_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
#                   FORMULAS TAB
# ═══════════════════════════════════════════════════════
with tab_ref:
    st.subheader("EVM Formulas Reference")
    formulas_evm = [
        ("Cost Variance (CV)", "CV = EV - AC"),
        ("Schedule Variance (SV)", "SV = EV - PV"),
        ("Cost Performance Index (CPI)", "CPI = EV / AC"),
        ("Schedule Performance Index (SPI)", "SPI = EV / PV"),
        ("EAC (using CPI)", "EAC = BAC / CPI"),
        ("EAC (atypical variance)", "EAC = AC + (BAC - EV)"),
        ("EAC (CPI × SPI)", "EAC = AC + (BAC-EV) / (CPI×SPI)"),
        ("Estimate to Complete (ETC)", "ETC = EAC - AC"),
        ("Variance at Completion (VAC)", "VAC = BAC - EAC"),
        ("TCPI (to meet BAC)", "TCPI = (BAC-EV) / (BAC-AC)"),
        ("TCPI (to meet EAC)", "TCPI = (BAC-EV) / (EAC-AC)"),
        ("% Complete", "% = (EV / BAC) × 100"),
    ]
    cols = st.columns(3)
    for i, (name, formula) in enumerate(formulas_evm):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.code(formula, language=None)

    st.markdown("---")
    st.subheader("Expected Monetary Value (EMV)")
    formulas_emv = [
        ("EMV (single risk)", "EMV = Probability × Impact"),
        ("Total EMV", "Total EMV = Σ (P × I) for all risks"),
        ("Contingency Reserve", "Reserve = |Σ EMV of threats|"),
        ("Net Risk Exposure", "Net = Σ Threat EMV + Σ Opportunity EMV"),
    ]
    cols = st.columns(3)
    for i, (name, formula) in enumerate(formulas_emv):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.code(formula, language=None)

    st.markdown("---")
    st.subheader("Financial Calculations")
    formulas_fin = [
        ("Net Present Value (NPV)", "NPV = -I₀ + Σ CFₜ / (1+r)ᵗ"),
        ("Internal Rate of Return (IRR)", "Rate r where NPV = 0"),
        ("Payback Period", "Period where Cumulative CF ≥ 0"),
        ("Discounted Payback", "Period where Cum. DCF ≥ 0"),
        ("ROI", "ROI = (Returns - Cost) / Cost × 100"),
        ("Benefit-Cost Ratio (BCR)", "BCR = PV(Benefits) / PV(Costs)"),
        ("Profitability Index (PI)", "PI = PV(Cash Flows) / Initial Investment"),
        ("Break-Even (Units)", "BEP = Fixed Costs / (Price - Var Cost)"),
        ("Break-Even (Revenue)", "BEP$ = Fixed Costs / CM Ratio"),
    ]
    cols = st.columns(3)
    for i, (name, formula) in enumerate(formulas_fin):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.code(formula, language=None)

    st.markdown("---")
    st.subheader("Critical Path Method (CPM)")
    formulas_cpm = [
        ("Early Start (ES)", "ES = max(EF of predecessors)"),
        ("Early Finish (EF)", "EF = ES + Duration"),
        ("Late Finish (LF)", "LF = min(LS of successors)"),
        ("Late Start (LS)", "LS = LF - Duration"),
        ("Total Float", "Float = LS - ES (or LF - EF)"),
        ("Critical Path", "Tasks where Total Float = 0"),
    ]
    cols = st.columns(3)
    for i, (name, formula) in enumerate(formulas_cpm):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.code(formula, language=None)

    st.markdown("---")
    st.subheader("Interpretation Guide")
    st.markdown("""
    - **CPI > 1.0** — Under budget (good)
    - **CPI < 1.0** — Over budget (bad)
    - **SPI > 1.0** — Ahead of schedule (good)
    - **SPI < 1.0** — Behind schedule (bad)
    - **TCPI > 1.0** — Must improve efficiency to meet target
    - **TCPI < 1.0** — Can relax efficiency and still meet target
    - **Total Float = 0** — Task is on critical path; any delay extends the project
    - **NPV > 0** — Project adds value; accept
    - **NPV < 0** — Project destroys value; reject
    - **IRR > Discount Rate** — Investment is attractive
    - **BCR > 1.0** — Benefits exceed costs
    - **PI > 1.0** — Good investment (creates value per dollar)
    - **ROI > 0%** — Positive return on investment
    - **EMV < 0** — Threat (negative risk); expected loss
    - **EMV > 0** — Opportunity (positive risk); expected gain
    - **Total EMV < 0** — Net negative risk exposure; increase contingency reserve
    - **Contingency Reserve** — Should cover the sum of threat EMVs
    """)
