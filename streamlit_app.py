import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
import pandas as pd
import math

# ─────────────────── PAGE CONFIG ───────────────────
st.set_page_config(
    page_title="PM Calculator — EVM & Critical Path",
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
st.markdown("<p style='text-align:center;color:#94a3b8;margin-top:-10px;'>Earned Value Management &bull; S-Curve Tracker &bull; Critical Path Analysis</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────── TABS ───────────────────
tab_evm, tab_scurve, tab_cp, tab_ref = st.tabs(["📈 EVM Metrics", "📉 S-Curve Tracker", "🔀 Critical Path", "📖 Formulas"])

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
    """)
