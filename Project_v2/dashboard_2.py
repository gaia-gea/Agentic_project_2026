"""Streamlit dashboard for the agentic network evaluation project.

Two views:
  - "Compare"  : overlay every selected agent model (+ heuristics) on the
    same charts so results can be judged side by side.
  - "Detail"   : single-series deep dive (topology, flow events, LLM
    telemetry, routing decisions).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


APP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = Path(
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
    "violet": "#a78bfa",
    "grid": "rgba(148, 163, 184, 0.12)",
}

# Fixed colors for the heuristic baselines (matches config.py's own scheme:
# blue / orange / red). Agent series get colors assigned dynamically, one
# per selected model, from AGENT_PALETTE below.
HEURISTIC_COLORS = {
    "heuristic_no_delay": "#3b82f6",
    "heuristic_low_delay": "#f59e0b",
    "heuristic_high_delay": "#ef4444",
}
HEURISTIC_LABELS = {
    "heuristic_no_delay": "Heuristic (no delay margin)",
    "heuristic_low_delay": "Heuristic (low delay margin)",
    "heuristic_high_delay": "Heuristic (high delay margin)",
}
AGENT_PALETTE = ["#22c55e", "#a78bfa", "#22d3ee", "#f472b6", "#facc15", "#fb923c", "#38bdf8", "#f87171"]

HEURISTIC_ORDER = ["heuristic_no_delay", "heuristic_low_delay", "heuristic_high_delay"]


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
    .series-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 10px; border-radius: 999px; font-size: .8rem;
        background: #0d1b2a; border: 1px solid #183047; margin-right: 6px; margin-bottom: 4px;
    }
    .series-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .winner-badge {
        background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.4);
        color: #4ade80; border-radius: 10px; padding: 10px 14px; font-size: .85rem;
    }
    .no-runs-warning {
        background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35);
        color: #fca5a5; border-radius: 10px; padding: 10px 14px; font-size: .85rem;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Run discovery & loading
# ---------------------------------------------------------------------------

REQUIRED_FILE = "network_metrics.csv"


def discover_runs(root: Path) -> list[Path]:
    """Find run folders under `root`.

    If `root` itself contains network_metrics.csv, it's treated as a single
    run (backward compatible with pointing straight at a run folder).
    Otherwise, every immediate child folder that contains the file is
    treated as a separate run.
    """
    if not root.exists():
        return []
    if (root / REQUIRED_FILE).exists():
        return [root]
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and (child / REQUIRED_FILE).exists()
    )


@st.cache_data(show_spinner=False)
def load_run(run_dir: str) -> dict:
    base = Path(run_dir)
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


def infer_provider_model_from_folder(run_dir: Path) -> dict:
    """
    Infer provider and model from the result folder name.

    Supported folder formats:
      1. timestamp__provider__model
         Example: 20260720_153000__openrouter__google-gemini-2.5-flash
      2. provider__model
         Example: anthropic__claude-haiku-4-5
      3. provider-model
         Example: anthropic-claude-haiku-4-5

    Since you manually renamed folders to provider-model, the dashboard now
    treats the first token as provider and everything after it as model.
    """
    name = run_dir.name.strip()

    known_providers = {
        "anthropic",
        "openrouter",
        "openai",
        "google",
        "gemini",
        "groq",
        "deepseek",
        "ollama",
        "local",
        "mistral",
        "together",
        "fireworks",
        "azure",
        "bedrock",
    }

    # Format: timestamp__provider__model or provider__model
    if "__" in name:
        parts = [part.strip() for part in name.split("__") if part.strip()]

        # timestamp__provider__model
        if len(parts) >= 3:
            return {
                "provider": parts[1] or "unknown",
                "model": "__".join(parts[2:]) or "unknown",
            }

        # provider__model
        if len(parts) == 2:
            return {
                "provider": parts[0] or "unknown",
                "model": parts[1] or "unknown",
            }

    # Format: provider-model, e.g. anthropic-claude-haiku-4-5
    lowered = name.lower()
    for provider in sorted(known_providers, key=len, reverse=True):
        prefix = provider + "-"
        if lowered.startswith(prefix):
            return {
                "provider": provider,
                "model": name[len(prefix):] or "unknown",
            }

    # Generic fallback: first token before '-' is provider, rest is model.
    if "-" in name:
        provider, model = name.split("-", 1)
        return {
            "provider": provider.strip() or "unknown",
            "model": model.strip() or "unknown",
        }

    # Last fallback: model is the folder name; provider unknown.
    return {
        "provider": "unknown",
        "model": name or "unknown",
    }


def extract_model_identity(run_dir: Path, metadata: dict) -> dict:
    """
    Robust identity extraction.

    Priority:
      1. Folder name, because folders were manually renamed as provider-model.
      2. run_metadata.json, only when the folder does not provide identity.
      3. nested metadata fields.

    Returns only provider and model. The dashboard no longer uses model_key.
    """
    folder_identity = infer_provider_model_from_folder(run_dir)
    provider = folder_identity.get("provider", "unknown")
    model = folder_identity.get("model", "unknown")

    # Metadata fallback only if folder name was not enough.
    if provider == "unknown":
        provider = metadata.get("provider") or "unknown"

    if model == "unknown":
        model = metadata.get("model") or metadata.get("model_name") or "unknown"

    for nested_key in ["llm", "model_config", "agent_model", "active_model", "agent"]:
        nested = metadata.get(nested_key)

        if isinstance(nested, dict):
            if provider == "unknown":
                provider = nested.get("provider") or provider

            if model == "unknown":
                model = (
                    nested.get("model")
                    or nested.get("model_name")
                    or nested.get("name")
                    or model
                )

    return {
        "provider": provider or "unknown",
        "model": model or "unknown",
    }


def run_model_label(run_dir: Path, metadata: dict) -> str:
    """Human-readable label identifying the LLM agent used in this run."""
    identity = extract_model_identity(run_dir, metadata)
    provider = identity["provider"]
    model = identity["model"]

    if provider != "unknown" and model != "unknown":
        return f"{provider} — {model}"

    if model != "unknown":
        return model

    if provider != "unknown":
        return provider

    return run_dir.name

# ---------------------------------------------------------------------------
# Multi-run dataset assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    selected_runs: list[Path],
    baseline_run: Path | None,
    per_run_heuristics: bool,
) -> dict:
    """
    Build combined dataframes tagged with a unique series id.

    Important:
    - network_metrics.csv, link_history.csv and flow_events.csv usually contain
      an algorithm column.
    - model_metrics.csv and agent_decisions.csv may NOT contain algorithm.
      They are agentic-only telemetry, so the dashboard assigns them to the
      selected agentic series manually.
    """
    frames = {
        "network": [],
        "links": [],
        "flows": [],
        "model": [],
        "decisions": [],
    }
    series_meta: dict[str, dict] = {}
    agent_color_idx = 0

    for run_dir in selected_runs:
        run_data = load_run(str(run_dir))
        metadata = run_data["metadata"]
        identity = extract_model_identity(run_dir, metadata)
        model_label = run_model_label(run_dir, metadata)
        agent_series_id = f"agentic::{run_dir.name}"

        if agent_series_id not in series_meta:
            series_meta[agent_series_id] = {
                "label": f"Agent — {model_label}",
                "color": AGENT_PALETTE[agent_color_idx % len(AGENT_PALETTE)],
                "kind": "agent",
                "run": run_dir.name,
                "run_path": str(run_dir),
                "provider": identity["provider"],
                "model": identity["model"],
            }
            agent_color_idx += 1

        include_heuristics = per_run_heuristics or (run_dir == baseline_run)

        for key in frames:
            df = run_data[key]

            if df.empty:
                continue

            df = df.copy()
            df["run"] = run_dir.name

            # Agent telemetry is usually agentic-only and may not include algorithm.
            if "algorithm" not in df.columns:
                if key in {"model", "decisions"}:
                    df["algorithm"] = "agentic"
                    df["series"] = agent_series_id
                    frames[key].append(df)
                continue

            is_agentic = df["algorithm"] == "agentic"
            df.loc[is_agentic, "series"] = agent_series_id

            if include_heuristics:
                heuristic_mask = df["algorithm"].isin(HEURISTIC_ORDER)

                if per_run_heuristics and len(selected_runs) > 1:
                    df.loc[heuristic_mask, "series"] = (
                        df.loc[heuristic_mask, "algorithm"] + "::" + run_dir.name
                    )
                else:
                    df.loc[heuristic_mask, "series"] = df.loc[heuristic_mask, "algorithm"]
            else:
                df = df[is_agentic]

            df = df.dropna(subset=["series"])

            if df.empty:
                continue

            frames[key].append(df)

            if key == "network":
                for algo in df.loc[df["algorithm"].isin(HEURISTIC_ORDER), "algorithm"].unique():
                    series_id = (
                        f"{algo}::{run_dir.name}"
                        if per_run_heuristics and len(selected_runs) > 1
                        else algo
                    )

                    if series_id not in series_meta:
                        base_label = HEURISTIC_LABELS.get(algo, algo)
                        label = (
                            f"{base_label} [{run_dir.name}]"
                            if per_run_heuristics and len(selected_runs) > 1
                            else base_label
                        )
                        series_meta[series_id] = {
                            "label": label,
                            "color": HEURISTIC_COLORS.get(algo, "#94a3b8"),
                            "kind": "heuristic",
                            "run": run_dir.name,
                            "run_path": str(run_dir),
                            "provider": "deterministic baseline",
                            "model": algo,
                        }

    combined = {}
    for key, parts in frames.items():
        combined[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    combined["series_meta"] = series_meta
    combined["run_metadata"] = {
        run_dir.name: load_run(str(run_dir))["metadata"] for run_dir in selected_runs
    }
    return combined


def series_label(series_meta: dict, series_id: str) -> str:
    return series_meta.get(series_id, {}).get("label", series_id)


def series_color(series_meta: dict, series_id: str) -> str:
    return series_meta.get(series_id, {}).get("color", "#94a3b8")


def ordered_series(series_meta: dict) -> list[str]:
    heuristics = sorted(sid for sid, m in series_meta.items() if m["kind"] == "heuristic")
    agents = sorted(sid for sid, m in series_meta.items() if m["kind"] == "agent")
    return heuristics + agents


def render_series_legend(series_meta: dict, series_ids: list[str]) -> None:
    chips = "".join(
        f'<span class="series-chip"><span class="series-dot" style="background:{series_color(series_meta, s)}"></span>'
        f'{series_label(series_meta, s)}</span>'
        for s in series_ids
    )
    st.markdown(chips, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalize to [0, 1]; if flat, returns 1.0 everywhere (no penalty)."""
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(1.0, index=series.index)
    scaled = (series - lo) / (hi - lo)
    return 1 - scaled if invert else scaled
def fmt_integer(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}{suffix}"


def parse_bool(value, default: bool = True) -> bool:
    """
    Robust boolean parser for values loaded from CSV.

    Important:
    bool("False") is True in Python.
    So we must explicitly parse strings.
    """
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "up", "active"}:
        return True

    if text in {"false", "0", "no", "n", "down", "inactive"}:
        return False

    return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def threshold_to_pct(value, default: float = 80.0) -> float:
    """
    Accept either 0.8 or 80 and convert it to 80%.
    """
    try:
        value = float(value)
    except Exception:
        return default

    if value <= 1.0:
        value *= 100.0

    return max(1.0, min(100.0, value))


def add_link_state_columns(df: pd.DataFrame, congestion_threshold_pct: float) -> pd.DataFrame:
    """
    Adds robust link state columns:
    - is_up
    - is_down
    - is_congested
    """
    if df.empty:
        return df

    df = df.copy()

    if "up" in df.columns:
        df["is_up"] = df["up"].apply(lambda value: parse_bool(value, default=True))
    else:
        df["is_up"] = True

    if "utilization_pct" in df.columns:
        utilization = pd.to_numeric(df["utilization_pct"], errors="coerce").fillna(0.0)
    else:
        utilization = 0.0

    df["is_down"] = ~df["is_up"]
    df["is_congested"] = df["is_up"] & (utilization >= congestion_threshold_pct)

    return df

def successful_llm_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Return only successful LLM calls if status exists."""
    if df.empty:
        return df

    if "status" not in df.columns:
        return df

    return df[df["status"].astype(str).str.lower().isin(["ok", "success", "successful"])]


def numeric_sum(df: pd.DataFrame, column: str) -> float | None:
    """Safe numeric sum."""
    if df.empty or column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.sum())


def numeric_mean(df: pd.DataFrame, column: str) -> float | None:
    """Safe numeric mean."""
    if df.empty or column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.mean())


def count_llm_decisions(decisions: pd.DataFrame, successful_calls: pd.DataFrame) -> dict:
    """
    Count LLM planning events in V5.

    In V5:
    - new_llm = a new valid LLM plan was received
    - active_plan = previous valid LLM plan reused
    - submitted = request sent to LLM
    - pending = waiting for LLM
    - not_needed = fast-path/trivial case
    """
    if decisions.empty or "decision_source" not in decisions.columns:
        successful_count = len(successful_calls) if not successful_calls.empty else 0

        return {
            "new_llm_plans": successful_count,
            "active_plan_reuse": 0,
            "submitted": 0,
            "pending": 0,
            "not_needed": 0,
            "total_llm_related_decisions": successful_count,
        }

    source = decisions["decision_source"].astype(str).str.lower()

    new_llm = int(source.isin(["new_llm", "llm_selected_tool", "llm_response"]).sum())
    active_plan = int(source.eq("active_plan").sum())
    submitted = int(source.eq("submitted").sum())
    pending = int(source.eq("pending").sum())
    not_needed = int(source.eq("not_needed").sum())

    return {
        "new_llm_plans": new_llm,
        "active_plan_reuse": active_plan,
        "submitted": submitted,
        "pending": pending,
        "not_needed": not_needed,
        "total_llm_related_decisions": new_llm + active_plan,
    }

# ---------------------------------------------------------------------------
# Comparison figures (all use the combined, series-tagged dataframes)
# ---------------------------------------------------------------------------

def build_series_summary(
    network: pd.DataFrame,
    links: pd.DataFrame,
    flows: pd.DataFrame,
    model_metrics: pd.DataFrame,
    series_ids: list[str],
    congestion_threshold_pct: float,
    step_duration: float,
) -> pd.DataFrame:
    rows = []
    for sid in series_ids:
        series_net = network[network["series"] == sid]
        series_links = links[links["series"] == sid] if not links.empty else links
        series_flows = flows[flows["series"] == sid] if not flows.empty else flows

        active = series_net[series_net["total_mbps"] > 0]
        total_traffic = active["total_mbps"].sum()
        accepted_traffic = active["accepted_mbps"].sum()
        pdr = safe_div(accepted_traffic, total_traffic) * 100

        congested = (
            series_links[series_links["utilization_pct"] >= congestion_threshold_pct]
            if not series_links.empty else series_links
        )
        congestion_steps = congested["time_step"].nunique() if not congested.empty else 0

        latency_violations = (
            series_flows[series_flows["reason"] == "latency_requirement_not_met"]
            if not series_flows.empty else series_flows
        )
        sla_steps = latency_violations["time_step"].nunique() if not latency_violations.empty else 0

        route_changes_avg = (
            series_net["route_changes"].mean() if "route_changes" in series_net.columns else np.nan
        )

        series_model_metrics = (
            model_metrics[model_metrics["series"] == sid]
            if not model_metrics.empty and "series" in model_metrics.columns
            else pd.DataFrame()
        )

        successful_calls = successful_llm_calls(series_model_metrics)
        avg_inference = numeric_mean(successful_calls, "inference_time_ms")
        total_tokens = numeric_sum(successful_calls, "total_tokens")

        rows.append({
            "series": sid,
            "pdr_pct": pdr,
            "accepted_mbps_total": accepted_traffic,
            "demanded_mbps_total": total_traffic,
            "congestion_steps": congestion_steps,
            "congestion_duration_s": congestion_steps * step_duration,
            "latency_sla_steps": sla_steps,
            "latency_sla_duration_s": sla_steps * step_duration,
            "route_changes_avg": route_changes_avg,
            "avg_inference_ms": avg_inference,
            "llm_calls": len(series_model_metrics) if not series_model_metrics.empty else 0,
            "successful_llm_calls": len(successful_calls) if not successful_calls.empty else 0,
            "total_tokens": total_tokens,
        })
    return pd.DataFrame(rows)


def comparison_pdr_figure(network: pd.DataFrame, series_meta: dict, series_ids: list[str]) -> go.Figure:
    fig = go.Figure()
    for sid in series_ids:
        series_net = network[network["series"] == sid].sort_values("time_step")
        if series_net.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series_net["time_step"],
            y=series_net["pdr"] * 100,
            name=series_label(series_meta, sid),
            mode="lines",
            line=dict(color=series_color(series_meta, sid), width=2.4),
        ))
    fig.update_layout(title="PDR over time")
    fig.update_yaxes(title_text="PDR (%)", range=[0, 105])
    fig.update_xaxes(title_text="Time step")
    fig = style_figure(fig, height=400)
    fig.update_layout(
        margin=dict(l=15, r=15, t=95, b=20),
        legend=dict(
            orientation="h",
            y=1.10,
            yanchor="bottom",
            x=0,
            xanchor="left",
        ),
    )
    return fig


def comparison_throughput_figure(summary: pd.DataFrame, series_meta: dict) -> go.Figure:
    labels = [series_label(series_meta, s) for s in summary["series"]]
    colors = [series_color(series_meta, s) for s in summary["series"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=summary["demanded_mbps_total"],
        name="Demanded", marker_color=COLORS["muted"], opacity=0.55,
    ))
    fig.add_trace(go.Bar(
        x=labels, y=summary["accepted_mbps_total"],
        name="Delivered", marker_color=colors,
    ))
    fig.update_layout(title="Total delivered vs. demanded traffic", barmode="group")
    fig.update_yaxes(title_text="Cumulative Mbps")
    return style_figure(fig, height=380)


def comparison_incidents_figure(summary: pd.DataFrame, series_meta: dict) -> go.Figure:
    labels = [series_label(series_meta, s) for s in summary["series"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=summary["latency_sla_duration_s"],
        name="Latency SLA violations (s)", marker_color=COLORS["red"],
    ))
    fig.update_layout(title="SLA delay violation", showlegend=False)
    fig.update_yaxes(title_text="Seconds")
    return style_figure(fig, height=380)


def comparison_agent_metric_figure(
    summary: pd.DataFrame,
    series_meta: dict,
    metric: str,
    title: str,
    yaxis_title: str,
    value_scale: float = 1.0,
    value_decimals: int = 0,
) -> go.Figure:
    """Compare one LLM telemetry metric across agent series only."""
    agents = summary[
        summary["series"].map(
            lambda sid: series_meta.get(sid, {}).get("kind") == "agent"
        )
    ].dropna(subset=[metric])

    fig = go.Figure()
    if agents.empty:
        fig.add_annotation(
            text="No successful LLM telemetry is available for the selected agents.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=15, color=COLORS["muted"]),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        labels = [series_label(series_meta, sid) for sid in agents["series"]]
        colors = [series_color(series_meta, sid) for sid in agents["series"]]
        display_values = agents[metric] * value_scale
        value_template = "%{text:,." + str(value_decimals) + "f}"
        hover_template = (
            "%{x}<br>%{y:,." + str(value_decimals) + "f} "
            + yaxis_title
            + "<extra></extra>"
        )
        fig.add_trace(go.Bar(
            x=labels,
            y=display_values,
            marker_color=colors,
            text=display_values,
            texttemplate=value_template,
            textposition="outside",
            hovertemplate=hover_template,
        ))
        fig.update_yaxes(title_text=yaxis_title, rangemode="tozero")

    fig.update_layout(title=title, showlegend=False)
    return style_figure(fig, height=380)


def radar_figure(summary: pd.DataFrame, series_meta: dict) -> go.Figure:
    metrics = [
        ("PDR", normalize(summary["pdr_pct"])),
        ("Delivered throughput", normalize(summary["accepted_mbps_total"])),
        ("Congestion resistance", normalize(summary["congestion_duration_s"], invert=True)),
        ("Latency SLA compliance", normalize(summary["latency_sla_duration_s"], invert=True)),
        ("Route stability", normalize(summary["route_changes_avg"].fillna(0), invert=True)),
    ]
    categories = [m[0] for m in metrics]

    fig = go.Figure()
    for i, sid in enumerate(summary["series"]):
        values = [round(float(m[1].iloc[i]), 3) for m in metrics]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=series_label(series_meta, sid),
            line=dict(color=series_color(series_meta, sid), width=2),
            fill="toself",
            opacity=0.35,
        ))
    fig.update_layout(
        title="Multi-metric comparison (normalized 0–1, farther = better)",
        polar=dict(
            bgcolor=COLORS["panel"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["grid"], color=COLORS["muted"]),
            angularaxis=dict(gridcolor=COLORS["grid"], color=COLORS["text"]),
        ),
        showlegend=True,
    )
    return style_figure(fig, height=460)


# ---------------------------------------------------------------------------
# Single-series detail figures
# ---------------------------------------------------------------------------

def topology_figure(
    link_history: pd.DataFrame,
    series_id: str,
    step: int,
    congestion_threshold_pct: float,
) -> go.Figure:
    series_links = link_history[link_history["series"] == series_id].copy()

    if series_links.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No link history available for this series.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["muted"]),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Network state")
        return style_figure(fig, height=510)

    series_links = add_link_state_columns(series_links, congestion_threshold_pct)

    current = series_links[series_links["time_step"].astype(int) == int(step)].copy()
    if not current.empty:
        # Draw healthy links first and failed links last so red dashed routes
        # remain visible if the input contains overlapping edge records.
        current["_draw_order"] = np.select(
            [current["is_down"], current["is_congested"]],
            [2, 1],
            default=0,
        )
        current = current.sort_values("_draw_order")

    graph = nx.Graph()
    graph.add_edges_from(
        series_links[["source", "target"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if graph.number_of_edges() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No topology edges available.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["muted"]),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"Network state · step {step}")
        return style_figure(fig, height=510)

    positions = nx.spring_layout(graph, seed=42, k=1.25)

    fig = go.Figure()
    legend_seen = set()

    if current.empty:
        fig.add_annotation(
            text=f"No link records found for step {step}.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["muted"]),
        )

    for row in current.itertuples(index=False):
        x0, y0 = positions[row.source]
        x1, y1 = positions[row.target]

        up = bool(getattr(row, "is_up", True))
        utilization_pct = safe_float(getattr(row, "utilization_pct", 0.0))
        utilized_mbps = safe_float(getattr(row, "utilized_mbps", 0.0))
        capacity_mbps = safe_float(getattr(row, "capacity_mbps", 0.0))
        latency_ms = safe_float(getattr(row, "latency_ms", 0.0))

        if not up:
            color = COLORS["red"]
            label = "DOWN"
            dash = "dash"
            width = 4.5
        elif utilization_pct >= congestion_threshold_pct:
            color = COLORS["amber"]
            label = "CONGESTED"
            dash = "solid"
            width = 2.5 + min(7.0, utilization_pct / 14.0)
        else:
            color = COLORS["cyan"]
            label = "UP"
            dash = "solid"
            width = 1.8 + min(5.0, utilization_pct / 22.0)

        hover = (
            f"{row.source} ↔ {row.target}"
            f"<br>Status: {label}"
            f"<br>Utilization: {utilization_pct:.1f}%"
            f"<br>Load: {utilized_mbps:.2f}/{capacity_mbps:.2f} Mbps"
            f"<br>Latency: {latency_ms:.2f} ms"
        )

        show_legend = label not in legend_seen
        legend_seen.add(label)

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            name=label,
            legendgroup=label,
            showlegend=show_legend,
            line=dict(
                color=color,
                width=width,
                dash=dash,
            ),
            hoverinfo="text",
            text=[hover, hover],
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
        marker=dict(
            size=34,
            color="#102b43",
            line=dict(width=2, color=COLORS["cyan"]),
        ),
        textfont=dict(color="#f8fafc", size=12),
        name="Node",
        showlegend=False,
    ))

    down_count = int((~current["is_up"]).sum()) if "is_up" in current.columns else 0
    congested_count = int(current["is_congested"].sum()) if "is_congested" in current.columns else 0

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=(
            f"Network state · step {step} "
            f"· down={down_count} · congested={congested_count}"
        )
    )

    return style_figure(fig, height=510)


def show_legacy_results(results_dir: Path) -> None:
    st.info(
        "The results found use the legacy format (no per-step CSVs). Run `python main.py` "
        "with the current version once to enable filters, KPIs and the temporal topology view."
    )
    cols = st.columns(2)
    for col, name, caption in zip(
        cols,
        ("results.png", "diagnostics.png"),
        ("Current results", "Current diagnostics"),
    ):
        image = results_dir / name
        if image.exists():
            col.image(str(image), caption=caption, width="stretch")


# ---------------------------------------------------------------------------
# Sidebar: pick which runs (agent models) to compare
# ---------------------------------------------------------------------------

results_root_path = st.sidebar.text_input("Results root directory", str(DEFAULT_RESULTS_ROOT))
if st.sidebar.button("Reload data", width="stretch"):
    st.cache_data.clear()

results_root = Path(results_root_path)
available_runs = discover_runs(results_root)

st.markdown('<div class="dashboard-kicker">ZERO-TOUCH NETWORK EVALUATION</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-title">Agentic Network Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">QoS performance, routing stability and AI model telemetry — '
    'compare different LLM agent models against each other and against the heuristic baselines.</div>',
    unsafe_allow_html=True,
)

if not available_runs:
    st.markdown(
        f'<div class="no-runs-warning">No run folders with <code>{REQUIRED_FILE}</code> were found under '
        f'<code>{results_root}</code>. Point the sidebar field at your <code>results/</code> directory '
        f'(the one containing one subfolder per simulation run), or at a single run folder.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


def run_display_name(run_dir: Path) -> str:
    metadata = load_run(str(run_dir))["metadata"]
    model_label = run_model_label(run_dir, metadata)
    return f"{model_label} — {run_dir.name}"


run_options = {run_display_name(r): r for r in available_runs}

st.sidebar.markdown("---")
st.sidebar.caption("RUNS TO COMPARE")
selected_run_names = st.sidebar.multiselect(
    "Agent runs (each = one LLM model + its heuristics)",
    list(run_options.keys()),
    default=list(run_options.keys()),
)
selected_runs = [run_options[name] for name in selected_run_names]

if not selected_runs:
    st.warning("Select at least one run in the sidebar.")
    st.stop()

baseline_options = {run_display_name(r): r for r in selected_runs}
st.sidebar.caption("HEURISTIC BASELINE")
per_run_heuristics = st.sidebar.checkbox(
    "Show heuristics separately for every run",
    value=False,
    help="Heuristics are deterministic given a fixed random seed, so by default only one "
         "run's heuristic curves are shown to avoid clutter. Enable this only if you changed "
         "the seed or topology between runs.",
)
if per_run_heuristics or len(selected_runs) == 1:
    baseline_run = None
else:
    baseline_name = st.sidebar.selectbox("Reference run for heuristic curves", list(baseline_options.keys()))
    baseline_run = baseline_options[baseline_name]
    st.sidebar.caption("Heuristics are assumed identical across runs (same fixed seed).")

dataset = assemble_dataset(selected_runs, baseline_run, per_run_heuristics)
network = dataset["network"]
links = dataset["links"]
flows = dataset["flows"]
model_metrics = dataset["model"]
agent_decisions = dataset["decisions"]
series_meta = dataset["series_meta"]
run_metadata_by_name = dataset["run_metadata"]

if network.empty or "series" not in network.columns:
    show_legacy_results(selected_runs[0])
    st.stop()

first_metadata = next(iter(run_metadata_by_name.values()), {})
sla_settings = first_metadata.get("sla", {})
visualization_settings = first_metadata.get("visualization", {})
step_duration = float(sla_settings.get("step_duration_seconds", 1.0))

all_series_ids = ordered_series(series_meta)

st.sidebar.markdown("---")
st.sidebar.caption("DISPLAY")
compare_series_ids = st.sidebar.multiselect(
    "Series shown in comparison charts",
    all_series_ids,
    default=all_series_ids,
    format_func=lambda s: series_label(series_meta, s),
)
default_congestion_threshold_pct = threshold_to_pct(
    visualization_settings.get(
        "congestion_threshold",
        sla_settings.get("congestion_threshold", 0.8),
    ),
    default=80.0,
)

congestion_threshold_pct = st.sidebar.slider(
    "Congestion warning threshold (%)",
    min_value=1,
    max_value=100,
    value=int(default_congestion_threshold_pct),
    help=(
        "This threshold is only for visualization and diagnostics. "
        "A link is highlighted as congested when utilization_pct is greater than or equal to this value. "
        "Set it to 100% if you only want to highlight links that are fully saturated."
    ),
)

# Filled from the Series detail tab. Keeping this slot here places the topology
# slider directly below the congestion threshold in the sidebar.
topology_step_sidebar_slot = st.sidebar.empty()

tab_compare, tab_detail = st.tabs(["📊 Compare agents", "🔍 Series detail"])

# ---------------------------------------------------------------------------
# TAB 1 — Compare
# ---------------------------------------------------------------------------
with tab_compare:
    if not compare_series_ids:
        st.warning("Select at least one series in the sidebar to compare.")
    else:
        render_series_legend(series_meta, compare_series_ids)

        summary = build_series_summary(
            network, links, flows, model_metrics,
            compare_series_ids, congestion_threshold_pct, step_duration,
        )

        best_row = summary.loc[summary["pdr_pct"].idxmax()]
        st.markdown(
            f'<div class="winner-badge">🏆 Best PDR: <b>{series_label(series_meta, best_row["series"])}</b> '
            f'at {best_row["pdr_pct"]:.1f}% effective delivery.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown('<div class="section-title">KPIs per series</div>', unsafe_allow_html=True)
        kpi_table = summary.copy()
        kpi_table["Series"] = kpi_table["series"].map(lambda s: series_label(series_meta, s))
        kpi_table = kpi_table[[
            "Series", "pdr_pct", "accepted_mbps_total", "congestion_duration_s",
            "latency_sla_duration_s", "route_changes_avg", "avg_inference_ms",
            "total_tokens",
        ]].rename(columns={
            "pdr_pct": "PDR (%)",
            "accepted_mbps_total": "Total Mbps delivered",
            "congestion_duration_s": "Congestion (s)",
            "latency_sla_duration_s": "SLA violations (s)",
            "route_changes_avg": "Route changes (avg/step)",
            "avg_inference_ms": "LLM inference (ms)",
            "total_tokens": "Total tokens",
        })
        st.dataframe(
            kpi_table.style.format({
                "PDR (%)": "{:.1f}",
                "Total Mbps delivered": "{:,.1f}",
                "Congestion (s)": "{:,.1f}",
                "SLA violations (s)": "{:,.1f}",
                "Route changes (avg/step)": "{:.2f}",
                "LLM inference (ms)": "{:,.1f}",
                "Total tokens": "{:,.0f}",
            }, na_rep="N/A"),
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            '<div class="section-title">Every selected agent, one chart</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(comparison_pdr_figure(network, series_meta, compare_series_ids), width="stretch")

        left, right = st.columns(2)
        left.plotly_chart(comparison_throughput_figure(summary, series_meta), width="stretch")
        right.plotly_chart(comparison_incidents_figure(summary, series_meta), width="stretch")

        tokens_col, response_col = st.columns(2)
        tokens_col.plotly_chart(
            comparison_agent_metric_figure(
                summary,
                series_meta,
                metric="total_tokens",
                title="Total token usage by agent",
                yaxis_title="Tokens",
            ),
            width="stretch",
        )
        response_col.plotly_chart(
            comparison_agent_metric_figure(
                summary,
                series_meta,
                metric="avg_inference_ms",
                title="Average inference time by agent",
                yaxis_title="Seconds",
                value_scale=0.001,
                value_decimals=2,
            ),
            width="stretch",
        )

        # st.plotly_chart(radar_figure(summary, series_meta), width="stretch")
        # st.caption(
        #     "The radar normalizes each metric between 0 and 1 across the selected series "
        #     "(1 = best performer on that metric). Congestion, SLA violations and route changes "
        #     "are inverted so 'farther from the center' always means better performance."
        # )

# ---------------------------------------------------------------------------
# TAB 2 — Detail
# ---------------------------------------------------------------------------
with tab_detail:
    detail_series_id = st.selectbox(
        "Series", all_series_ids, format_func=lambda s: series_label(series_meta, s)
    )

    detail_series_meta = series_meta.get(detail_series_id, {})
    detail_run_name = detail_series_meta.get("run")
    detail_run_path = detail_series_meta.get("run_path")
    detail_metadata = run_metadata_by_name.get(detail_run_name, {})

    identity = extract_model_identity(
        Path(detail_run_path or detail_run_name or ""),
        detail_metadata,
    )

    provider = detail_series_meta.get("provider") or identity["provider"]
    model_name = detail_series_meta.get("model") or identity["model"]

    if detail_series_meta.get("kind") == "heuristic":
        provider = "deterministic baseline"
        model_name = series_label(series_meta, detail_series_id)

    series_network = network[network["series"] == detail_series_id].sort_values("time_step")
    steps = sorted(series_network["time_step"].astype(int).unique().tolist())

    if not steps:
        st.info("No data for this series.")
        st.stop()

    series_links = (
        links[links["series"] == detail_series_id].copy()
        if not links.empty and "series" in links.columns
        else pd.DataFrame()
    )
    series_links = add_link_state_columns(series_links, congestion_threshold_pct)

    series_flows = (
        flows[flows["series"] == detail_series_id].copy()
        if not flows.empty and "series" in flows.columns
        else pd.DataFrame()
    )

    series_model_metrics = (
        model_metrics[model_metrics["series"] == detail_series_id].copy()
        if not model_metrics.empty and "series" in model_metrics.columns
        else pd.DataFrame()
    )

    series_agent_decisions = (
        agent_decisions[agent_decisions["series"] == detail_series_id].copy()
        if not agent_decisions.empty and "series" in agent_decisions.columns
        else pd.DataFrame()
    )

    if not series_links.empty and "time_step" in series_links.columns:
        topology_steps = sorted(
            series_links["time_step"].dropna().astype(int).unique().tolist()
        )
    else:
        topology_steps = steps

    if not topology_steps:
        st.info("No topology steps available for this series.")
        st.stop()

    default_topology_step = topology_steps[-1]
    topology_widget_suffix = detail_series_id.replace(":", "_").replace("/", "_")
    topology_widget_key = f"topology_step_v2_{topology_widget_suffix}"

    if (
        topology_widget_key not in st.session_state
        or st.session_state[topology_widget_key] not in topology_steps
    ):
        st.session_state[topology_widget_key] = default_topology_step

    with topology_step_sidebar_slot.container():
        st.markdown("---")
        st.caption("TOPOLOGY EXPLORATION")
        st.caption(
            "The topology is a snapshot of one time step. Move the slider "
            "to inspect healthy, congested and failed-link states."
        )
        selected_step = st.select_slider(
            "Topology time step",
            options=topology_steps,
            key=topology_widget_key,
            help=(
                "Move this slider to inspect the network topology, congested links, "
                "failed links and flow events at each simulation step."
            ),
        )

    info_cols = st.columns(4)

    info_cols[0].caption("SERIES")
    info_cols[0].write(f"**{series_label(series_meta, detail_series_id)}**")

    info_cols[1].caption("PROVIDER")
    info_cols[1].write(provider)

    info_cols[2].caption("MODEL")
    info_cols[2].write(model_name)

    info_cols[3].caption("RUN FOLDER")
    info_cols[3].write(detail_run_name or "N/A")

    active = series_network[series_network["total_mbps"] > 0]
    total_traffic = active["total_mbps"].sum()
    overall_pdr = safe_div(active["accepted_mbps"].sum(), total_traffic)

    congested = (
        series_links[series_links["is_congested"]]
        if not series_links.empty and "is_congested" in series_links.columns
        else pd.DataFrame()
    )

    latency_sla_violations = (
        series_flows[series_flows["reason"] == "latency_requirement_not_met"]
        if not series_flows.empty and "reason" in series_flows.columns
        else pd.DataFrame()
    )

    sla_steps = latency_sla_violations["time_step"].nunique() if not latency_sla_violations.empty else 0
    congestion_steps = congested["time_step"].nunique() if not congested.empty else 0
    bandwidth_sla = sla_steps * step_duration
    congestion_duration = congestion_steps * step_duration

    congestion_by_step = (
        congested.groupby("time_step").size().reindex(steps, fill_value=0)
        if not congested.empty and "time_step" in congested.columns
        else pd.Series(0, index=steps)
    )
    sla_by_step = (
        latency_sla_violations.groupby("time_step").size().reindex(steps, fill_value=0)
        if not latency_sla_violations.empty and "time_step" in latency_sla_violations.columns
        else pd.Series(0, index=steps)
    )

    successful_calls = successful_llm_calls(series_model_metrics)
    avg_inference = numeric_mean(successful_calls, "inference_time_ms")
    total_tokens = numeric_sum(successful_calls, "total_tokens")
    llm_call_count = len(series_model_metrics) if not series_model_metrics.empty else 0
    successful_llm_call_count = len(successful_calls) if not successful_calls.empty else 0
    decision_counts = count_llm_decisions(series_agent_decisions, successful_calls)
    route_changes_avg = numeric_mean(series_network, "route_changes")

    metric_row_1 = st.columns(4)
    metric_row_1[0].metric("PDR proxy", f"{overall_pdr * 100:.1f}%")
    metric_row_1[1].metric("Latency SLA", fmt_number(bandwidth_sla, " s"))
    metric_row_1[2].metric("Congestion", fmt_number(congestion_duration, " s"))
    metric_row_1[3].metric("Avg. route changes", fmt_number(route_changes_avg))

    metric_row_2 = st.columns(4)
    metric_row_2[0].metric("Avg. inference", fmt_number(avg_inference, " ms"))
    metric_row_2[1].metric("Total tokens", fmt_integer(total_tokens))
    metric_row_2[2].metric("LLM calls", f"{successful_llm_call_count}/{llm_call_count}")
    metric_row_2[3].metric(
        "Plans / reuse",
        f'{fmt_integer(decision_counts["new_llm_plans"])} / '
        f'{fmt_integer(decision_counts["active_plan_reuse"])}',
    )

    with st.expander("Telemetry source check", expanded=False):
        st.caption(
            "Provider/model are inferred from the run folder first. AI telemetry "
            "requires model_metrics.csv and agent_decisions.csv."
        )

        if detail_run_path:
            run_path = Path(detail_run_path)
            telemetry_status = pd.DataFrame([
                {
                    "file": "model_metrics.csv",
                    "exists": (run_path / "model_metrics.csv").exists(),
                    "rows_loaded": str(len(series_model_metrics)),
                    "used_for": "Avg inference, total tokens, LLM calls",
                },
                {
                    "file": "agent_decisions.csv",
                    "exists": (run_path / "agent_decisions.csv").exists(),
                    "rows_loaded": str(len(series_agent_decisions)),
                    "used_for": "New LLM plans, plan reuse, pending/submitted decisions",
                },
                {
                    "file": "run_metadata.json",
                    "exists": (run_path / "run_metadata.json").exists(),
                    "rows_loaded": "N/A",
                    "used_for": "Metadata fallback only",
                },
            ])
            # Streamlit uses PyArrow internally. Mixed integer/string columns can
            # trigger ArrowInvalid, so force the diagnostic table to strings.
            telemetry_status = telemetry_status.astype(str)
            st.dataframe(telemetry_status, hide_index=True, width="stretch")
        else:
            st.warning("No run path was stored for this series.")

        st.write("Current inferred identity:")
        st.json({
            "provider": provider,
            "model": model_name,
            "run_folder": detail_run_name,
        })

        if not series_model_metrics.empty:
            st.write("model_metrics.csv columns:")
            st.code(", ".join(series_model_metrics.columns.astype(str).tolist()))

        if not series_agent_decisions.empty:
            st.write("agent_decisions.csv columns:")
            st.code(", ".join(series_agent_decisions.columns.astype(str).tolist()))

    st.markdown('<div class="section-title">Network KPIs</div>', unsafe_allow_html=True)
    left, right = st.columns(2)

    traffic_fig = go.Figure()
    traffic_fig.add_trace(go.Scatter(
        x=series_network["time_step"], y=series_network["total_mbps"],
        name="Demanded", line=dict(color=COLORS["muted"], width=2),
    ))
    traffic_fig.add_trace(go.Scatter(
        x=series_network["time_step"], y=series_network["accepted_mbps"],
        name="Delivered", fill="tozeroy", line=dict(color=COLORS["green"], width=2),
    ))
    traffic_fig.update_layout(title="Traffic delivery", yaxis_title="Mbps")
    left.plotly_chart(style_figure(traffic_fig), width="stretch")

    pdr_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pdr_fig.add_trace(go.Scatter(
        x=series_network["time_step"], y=series_network["pdr"] * 100,
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

    current_links = (
        series_links[series_links["time_step"].astype(int) == int(selected_step)].copy()
        if not series_links.empty and "time_step" in series_links.columns
        else pd.DataFrame()
    )

    if series_links.empty:
        topology_col.warning(
            "No link_history.csv data available for this series. "
            "The topology view requires per-step link history."
        )
    elif current_links.empty:
        topology_col.warning(f"No link records available for step {selected_step}.")
    else:
        down_count = int(current_links["is_down"].sum()) if "is_down" in current_links.columns else 0
        congested_count = int(current_links["is_congested"].sum()) if "is_congested" in current_links.columns else 0
        total_links = int(len(current_links))

        topology_col.caption(
            f"STEP {selected_step} · links={total_links} · "
            f"down={down_count} · congested={congested_count} · "
            f"threshold={congestion_threshold_pct}%"
        )

        topology_col.plotly_chart(
            topology_figure(links, detail_series_id, selected_step, congestion_threshold_pct),
            width="stretch",
        )

    step_flows = (
        series_flows[series_flows["time_step"] == selected_step].copy()
        if not series_flows.empty and "time_step" in series_flows.columns
        else pd.DataFrame()
    )

    if not step_flows.empty and "status" in step_flows.columns:
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

        if "route_changes" in series_network.columns:
            stability_fig.add_trace(go.Scatter(
                x=series_network["time_step"], y=series_network["route_changes"],
                name="Route changes", line=dict(color=COLORS["amber"]),
            ))

        if "decision_age_steps" in series_network.columns:
            stability_fig.add_trace(go.Scatter(
                x=series_network["time_step"], y=series_network["decision_age_steps"],
                name="Decision age", line=dict(color=COLORS["cyan"]),
            ))

        stability_fig.update_layout(title="Routing stability")
        stability.plotly_chart(style_figure(stability_fig), width="stretch")

        if series_model_metrics.empty:
            ai.info(
                "No LLM inference calls were recorded for this series. "
                "Check whether model_metrics.csv exists in this run folder."
            )
        else:
            ai.dataframe(
                series_model_metrics, hide_index=True, width="stretch", height=330
            )

        st.markdown("**Routing decisions**")
        if series_agent_decisions.empty:
            st.info(
                "No agent decisions were recorded for this series. "
                "Check whether agent_decisions.csv exists in this run folder."
            )
        else:
            decision_columns = [column for column in [
                "step", "snapshot_step", "situation", "decision_source", "strategy",
                "active_flows", "routed_flows", "unresolved_flows", "route_changes",
                "decision_age_steps", "max_primary_pressure", "failed_route_flows",
                "candidate_flows", "no_candidate_flows",
            ] if column in series_agent_decisions.columns]
            st.dataframe(
                series_agent_decisions[decision_columns].sort_values(
                    "step", ascending=False
                ) if "step" in series_agent_decisions.columns else series_agent_decisions[decision_columns],
                hide_index=True,
                width="stretch",
                height=360,
            )

st.caption(
    "PDR proxy is calculated from delivered Mbps / demanded Mbps. "
    "Each flow is validated against its maximum end-to-end latency and shared-link capacity. "
    "Congestion is shown as an operational diagnostic. Heuristic series are shared across runs "
    "by default because the random seed is fixed — toggle 'Show heuristics separately for every "
    "run' in the sidebar if you changed the seed or topology between runs."
)
