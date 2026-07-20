"""Streamlit MVP for the agentic network evaluation dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


APP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("DASHBOARD_RESULTS_DIR", APP_DIR.parent / "results")
)

COLORS = {
    "background": "#07111f",
    "panel": "#0d1b2a",
    "text": "#e5edf7",
    "muted": "#8fa3b8",
    "cyan": "#22d3ee",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "grid": "rgba(148, 163, 184, 0.12)",
}


st.set_page_config(
    page_title="Agentic Network Dashboard",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #07111f; color: #e5edf7; }
    [data-testid="stSidebar"] { background: #091727; border-right: 1px solid #183047; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #0d1b2a, #0a1725);
        border: 1px solid #183047; border-radius: 14px; padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #8fa3b8; }
    [data-testid="stMetricValue"] { color: #f4f8fc; }
    div[data-testid="stDataFrame"] { border: 1px solid #183047; border-radius: 12px; }
    .dashboard-kicker { color: #22d3ee; letter-spacing: .14em; font-size: .75rem; font-weight: 700; }
    .dashboard-title { font-size: 2rem; font-weight: 750; margin: .15rem 0; }
    .dashboard-subtitle { color: #8fa3b8; margin-bottom: 1rem; }
    .section-title { font-size: 1.05rem; font-weight: 650; margin: .6rem 0 .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_results(results_dir: str) -> dict:
    base = Path(results_dir)
    files = {
        "network": "network_metrics.csv",
        "links": "link_history.csv",
        "flows": "flow_events.csv",
        "model": "model_metrics.csv",
        "decisions": "agent_decisions.csv",
    }
    data = {
        name: pd.read_csv(base / filename) if (base / filename).exists() else pd.DataFrame()
        for name, filename in files.items()
    }
    metadata_file = base / "run_metadata.json"
    data["metadata"] = (
        json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata_file.exists()
        else {}
    )
    return data


def style_figure(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=45, b=20),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        legend=dict(orientation="h", y=1.13, x=0),
        hoverlabel=dict(bgcolor="#12263a", font_color=COLORS["text"]),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def fmt_number(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.1f}{suffix}"


def topology_figure(
    link_history: pd.DataFrame,
    algorithm: str,
    step: int,
    congestion_threshold_pct: float,
) -> go.Figure:
    algorithm_links = link_history[link_history["algorithm"] == algorithm]
    current = algorithm_links[algorithm_links["time_step"] == step]
    graph = nx.Graph()
    graph.add_edges_from(
        algorithm_links[["source", "target"]].drop_duplicates().itertuples(index=False, name=None)
    )
    positions = nx.spring_layout(graph, seed=42, k=1.25)

    fig = go.Figure()
    for row in current.itertuples(index=False):
        x0, y0 = positions[row.source]
        x1, y1 = positions[row.target]
        if not bool(row.up):
            color, label = COLORS["red"], "DOWN"
        elif float(row.utilization_pct) >= congestion_threshold_pct:
            color, label = COLORS["amber"], "CONGESTED"
        else:
            color, label = COLORS["cyan"], "UP"
        width = 1.5 + min(6.0, float(row.utilization_pct) / 18.0)
        hover = (
            f"{row.source} ↔ {row.target}<br>Status: {label}"
            f"<br>Utilization: {row.utilization_pct:.1f}%"
            f"<br>Load: {row.utilized_mbps:.2f}/{row.capacity_mbps:.2f} Mbps"
            f"<br>Latency: {row.latency_ms:.2f} ms"
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=width, dash="dash" if not bool(row.up) else "solid"),
            hoverinfo="text", text=[hover, hover], showlegend=False,
        ))

    nodes = sorted(graph.nodes())
    fig.add_trace(go.Scatter(
        x=[positions[node][0] for node in nodes],
        y=[positions[node][1] for node in nodes],
        mode="markers+text",
        text=nodes,
        textposition="middle center",
        hovertext=[f"Node {node}" for node in nodes],
        hoverinfo="text",
        marker=dict(size=34, color="#102b43", line=dict(width=2, color=COLORS["cyan"])),
        textfont=dict(color="#f8fafc", size=12),
        showlegend=False,
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=f"Network state · step {step}")
    return style_figure(fig, height=510)


def show_legacy_results(results_dir: Path) -> None:
    st.info(
        "Los resultados encontrados pertenecen al formato anterior. Ejecuta `python Project/main.py` "
        "una vez con la nueva versión para habilitar los filtros, KPIs y la topología temporal."
    )
    cols = st.columns(2)
    for col, name, caption in zip(
        cols,
        ("results.png", "diagnostics.png"),
        ("Resultados actuales", "Diagnósticos actuales"),
    ):
        image = results_dir / name
        if image.exists():
            col.image(str(image), caption=caption, width="stretch")


results_path = st.sidebar.text_input("Results directory", str(DEFAULT_RESULTS_DIR))
if st.sidebar.button("Reload data", width="stretch"):
    st.cache_data.clear()

results_dir = Path(results_path)
data = load_results(str(results_dir))
network = data["network"]
links = data["links"]
flows = data["flows"]
model_metrics = data["model"]
agent_decisions = data["decisions"]
metadata = data["metadata"]

st.markdown('<div class="dashboard-kicker">ZERO-TOUCH NETWORK EVALUATION</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-title">Agentic Network Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">QoS performance, routing stability and AI model telemetry.</div>',
    unsafe_allow_html=True,
)

if network.empty or links.empty:
    show_legacy_results(results_dir)
    st.stop()

algorithms = sorted(network["algorithm"].dropna().unique().tolist())
algorithm = st.sidebar.selectbox("Algorithm", algorithms)
algorithm_network = network[network["algorithm"] == algorithm].sort_values("time_step")
steps = sorted(algorithm_network["time_step"].astype(int).unique().tolist())
if len(steps) == 1:
    selected_step = steps[0]
    st.sidebar.caption(f"Time step: {selected_step}")
else:
    selected_step = st.sidebar.slider("Time step", min(steps), max(steps), max(steps))

model_info = metadata.get("models", {}).get(
    algorithm, metadata.get("model", {})
)
sla_settings = metadata.get("sla", {})
visualization_settings = metadata.get("visualization", {})
st.sidebar.markdown("---")
st.sidebar.caption("ACTIVE MODEL")
st.sidebar.write(f"**{model_info.get('name', 'Not recorded')}**")
st.sidebar.caption(f"Provider: {model_info.get('provider', 'N/A')}")
st.sidebar.caption(f"Seed: {metadata.get('random_seed', 'N/A')}")
agent_info = metadata.get("agent", {})
if agent_info:
    st.sidebar.caption(
        f"Agent: {agent_info.get('version', 'N/A')} · "
        f"pressure threshold: {agent_info.get('pressure_threshold', 'N/A')}"
    )
st.sidebar.markdown("---")
st.sidebar.caption("TOPOLOGY DISPLAY")
congestion_threshold_pct = st.sidebar.slider(
    "Congestion threshold (%)",
    min_value=1,
    max_value=100,
    value=int(float(visualization_settings.get(
        "congestion_threshold",
        sla_settings.get("congestion_threshold", 0.8),
    )) * 100),
)

active = algorithm_network[algorithm_network["total_mbps"] > 0]
total_traffic = active["total_mbps"].sum()
overall_pdr = active["accepted_mbps"].sum() / total_traffic if total_traffic else 0.0
algorithm_links = links[links["algorithm"] == algorithm].copy()
algorithm_flows = flows[flows["algorithm"] == algorithm].copy()
selected_model_metrics = (
    model_metrics[model_metrics["algorithm"] == algorithm]
    if "algorithm" in model_metrics.columns
    else model_metrics
)
selected_agent_decisions = (
    agent_decisions[agent_decisions["algorithm"] == algorithm]
    if "algorithm" in agent_decisions.columns
    else agent_decisions
)
congested = algorithm_links[algorithm_links["utilization_pct"] >= congestion_threshold_pct]
latency_sla_violations = algorithm_flows[
    algorithm_flows["reason"] == "latency_requirement_not_met"
]
sla_steps = latency_sla_violations["time_step"].nunique()
congestion_steps = congested["time_step"].nunique()
step_duration = float(sla_settings.get("step_duration_seconds", 1.0))
bandwidth_sla = sla_steps * step_duration
congestion_duration = congestion_steps * step_duration

congestion_by_step = (
    congested.groupby("time_step").size().reindex(steps, fill_value=0)
)
sla_by_step = (
    latency_sla_violations.groupby("time_step").size().reindex(steps, fill_value=0)
)
successful_calls = (
    selected_model_metrics[selected_model_metrics["status"] == "ok"]
    if "status" in selected_model_metrics.columns
    else selected_model_metrics
)
avg_inference = successful_calls["inference_time_ms"].mean() if "inference_time_ms" in successful_calls else None
total_tokens = successful_calls["total_tokens"].sum(min_count=1) if "total_tokens" in successful_calls else None
llm_decision_count = (
    int(selected_agent_decisions["decision_source"].eq("llm_selected_tool").sum())
    if "decision_source" in selected_agent_decisions.columns
    else 0
)

kpi_cols = st.columns(6)
kpi_cols[0].metric("PDR proxy", f"{overall_pdr * 100:.1f}%")
kpi_cols[1].metric("Latency SLA duration", fmt_number(bandwidth_sla, " steps"))
kpi_cols[2].metric("Congestion duration", fmt_number(congestion_duration, " s"))
kpi_cols[3].metric("Avg. inference", fmt_number(avg_inference, " ms"))
kpi_cols[4].metric("Total tokens", fmt_number(total_tokens))
kpi_cols[5].metric("LLM decisions", fmt_number(llm_decision_count))

st.markdown('<div class="section-title">Network KPIs</div>', unsafe_allow_html=True)
left, right = st.columns(2)

traffic_fig = go.Figure()
traffic_fig.add_trace(go.Scatter(
    x=algorithm_network["time_step"], y=algorithm_network["total_mbps"],
    name="Demanded", line=dict(color=COLORS["muted"], width=2),
))
traffic_fig.add_trace(go.Scatter(
    x=algorithm_network["time_step"], y=algorithm_network["accepted_mbps"],
    name="Delivered", fill="tozeroy", line=dict(color=COLORS["green"], width=2),
))
traffic_fig.update_layout(title="Traffic delivery", yaxis_title="Mbps")
left.plotly_chart(style_figure(traffic_fig), width="stretch")

pdr_fig = make_subplots(specs=[[{"secondary_y": True}]])
pdr_fig.add_trace(go.Scatter(
    x=algorithm_network["time_step"], y=algorithm_network["pdr"] * 100,
    name="PDR proxy", line=dict(color=COLORS["cyan"], width=2),
), secondary_y=False)
pdr_fig.add_trace(go.Bar(
    x=steps, y=congestion_by_step.values,
    name="Congested links", marker_color=COLORS["amber"], opacity=.55,
), secondary_y=True)
pdr_fig.add_trace(go.Bar(
    x=steps, y=sla_by_step.values,
    name="Latency SLA violations", marker_color=COLORS["red"], opacity=.55,
), secondary_y=True)
pdr_fig.update_layout(title="PDR and SLA violations")
pdr_fig.update_yaxes(title_text="PDR (%)", range=[0, 105], secondary_y=False)
pdr_fig.update_yaxes(title_text="SLA events", secondary_y=True)
right.plotly_chart(style_figure(pdr_fig), width="stretch")

st.markdown('<div class="section-title">Dynamic topology</div>', unsafe_allow_html=True)
topology_col, event_col = st.columns([1.5, 1])
topology_col.plotly_chart(
    topology_figure(links, algorithm, selected_step, congestion_threshold_pct),
    width="stretch",
)

step_flows = flows[
    (flows["algorithm"] == algorithm) & (flows["time_step"] == selected_step)
].copy()
if not step_flows.empty:
    step_flows["sla_violation"] = step_flows["status"].eq("dropped")
event_col.caption(f"FLOW EVENTS · STEP {selected_step}")
if step_flows.empty:
    event_col.info("No active flow events at this step.")
else:
    display_columns = [column for column in [
        "source", "target", "demand_mbps", "path", "e2e_latency_ms",
        "max_latency_ms", "status", "reason", "sla_violation",
    ] if column in step_flows.columns]
    event_col.dataframe(
        step_flows[display_columns],
        hide_index=True,
        width="stretch",
        height=470,
    )

with st.expander("AI model metrics and routing stability", expanded=False):
    stability, ai = st.columns(2)
    stability_fig = go.Figure()
    stability_fig.add_trace(go.Scatter(
        x=algorithm_network["time_step"], y=algorithm_network["route_changes"],
        name="Route changes", line=dict(color=COLORS["amber"]),
    ))
    stability_fig.add_trace(go.Scatter(
        x=algorithm_network["time_step"], y=algorithm_network["decision_age_steps"],
        name="Decision age", line=dict(color=COLORS["cyan"]),
    ))
    stability_fig.update_layout(title="Routing stability")
    stability.plotly_chart(style_figure(stability_fig), width="stretch")

    if selected_model_metrics.empty:
        ai.info("No LLM inference calls were recorded for this run.")
    else:
        ai.dataframe(
            selected_model_metrics, hide_index=True, width="stretch", height=330
        )

    st.markdown("**V11 routing decisions**")
    if selected_agent_decisions.empty:
        st.info("No V11 decisions were recorded for this run.")
    else:
        decision_columns = [column for column in [
            "step", "snapshot_step", "situation", "decision_source", "strategy",
            "active_flows", "routed_flows", "unresolved_flows", "route_changes",
            "decision_age_steps", "max_primary_pressure", "failed_route_flows",
            "candidate_flows", "no_candidate_flows",
        ] if column in selected_agent_decisions.columns]
        st.dataframe(
            selected_agent_decisions[decision_columns].sort_values(
                "step", ascending=False
            ),
            hide_index=True,
            width="stretch",
            height=360,
        )

st.caption(
    "PDR proxy is calculated from delivered Mbps / demanded Mbps. "
    "Each flow is validated against its maximum end-to-end latency and shared-link capacity. "
    "Congestion is shown as an operational diagnostic."
)
