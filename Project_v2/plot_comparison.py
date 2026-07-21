# plot_pdr_comparison.py
"""
Create a clean presentation-ready PDR comparison chart.

It scans result folders, reads network_metrics.csv, and plots:
- one agentic line per LLM model
- heuristic baselines once, taken from the first valid run

Usage:
    python plot_pdr_comparison.py --results-dir ../results

Or, if your results folder is inside Project_v2:
    python plot_pdr_comparison.py --results-dir results

Outputs:
    presentation_charts/pdr_comparison_presentation.png
    presentation_charts/pdr_comparison_presentation.pdf
    presentation_charts/pdr_comparison_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NETWORK_FILE = "network_metrics.csv"

HEURISTIC_ORDER = [
    "heuristic_no_delay",
    "heuristic_low_delay",
    "heuristic_high_delay",
]

HEURISTIC_LABELS = {
    "heuristic_no_delay": "Heuristic: no delay margin",
    "heuristic_low_delay": "Heuristic: low delay margin",
    "heuristic_high_delay": "Heuristic: high delay margin",
}

KNOWN_PROVIDERS = [
    "openrouter",
    "anthropic",
    "google",
    "gemini",
    "groq",
    "openai",
    "deepseek",
    "ollama",
    "local",
    "mistral",
]


def discover_run_dirs(results_root: Path) -> list[Path]:
    """
    Find result folders containing network_metrics.csv.
    If results_root itself contains network_metrics.csv, treat it as one run.
    """
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    if (results_root / NETWORK_FILE).exists():
        return [results_root]

    run_dirs = [
        child
        for child in results_root.iterdir()
        if child.is_dir() and (child / NETWORK_FILE).exists()
    ]

    return sorted(run_dirs, key=lambda path: path.name.lower())


def split_provider_model_from_name(folder_name: str) -> tuple[str, str]:
    """
    Infer provider and model from a folder name.

    Supported examples:
        anthropic-claude-haiku-4-5
        openrouter-google-gemini-2.5-flash
        groq-llama-3.1-8b-instant
        ollama-llama3.2-1b
        20260720_153000__openrouter__google-gemini-2.5-flash
        20260720_153000__anthropic-claude-haiku-4-5
    """
    name = folder_name.strip()

    # Old format with "__"
    if "__" in name:
        parts = [part.strip() for part in name.split("__") if part.strip()]

        # timestamp__provider__model
        if len(parts) >= 3:
            provider = parts[-2]
            model = parts[-1]
            return provider, model

        # timestamp__provider-model
        if len(parts) == 2:
            name = parts[-1]

    normalized = name.replace("_", "-").strip()

    for provider in sorted(KNOWN_PROVIDERS, key=len, reverse=True):
        prefix = provider + "-"

        if normalized.lower().startswith(prefix):
            model = normalized[len(prefix):]
            return provider, model or "unknown"

    # Fallback: first token is provider, rest is model
    if "-" in normalized:
        provider, model = normalized.split("-", 1)
        return provider, model

    return "unknown", normalized or "unknown"


def pretty_label_from_run(run_dir: Path) -> str:
    """Create a presentation-friendly agentic model label."""
    provider, model = split_provider_model_from_name(run_dir.name)

    provider_display = {
        "openrouter": "OpenRouter",
        "anthropic": "Anthropic",
        "google": "Google",
        "gemini": "Gemini",
        "groq": "Groq",
        "openai": "OpenAI",
        "deepseek": "DeepSeek",
        "ollama": "Ollama",
        "local": "Local",
        "mistral": "Mistral",
    }.get(provider.lower(), provider)

    if provider == "unknown":
        return model

    return f"{provider_display}: {model}"


def read_network_metrics(run_dir: Path) -> pd.DataFrame:
    """Read and normalize network_metrics.csv."""
    path = run_dir / NETWORK_FILE

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame()

    required = {"time_step"}

    if not required.issubset(df.columns):
        print(f"[WARN] Missing time_step in {path}")
        return pd.DataFrame()

    # If algorithm is missing, assume the whole CSV is agentic.
    if "algorithm" not in df.columns:
        df["algorithm"] = "agentic"

    # Compute pdr if missing.
    if "pdr" not in df.columns:
        if {"accepted_mbps", "total_mbps"}.issubset(df.columns):
            denominator = pd.to_numeric(df["total_mbps"], errors="coerce").replace(0, np.nan)
            numerator = pd.to_numeric(df["accepted_mbps"], errors="coerce")
            df["pdr"] = (numerator / denominator).fillna(0.0)
        else:
            print(f"[WARN] No pdr or accepted/total Mbps columns in {path}")
            return pd.DataFrame()

    df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce")
    df["pdr"] = pd.to_numeric(df["pdr"], errors="coerce")

    if "total_mbps" in df.columns:
        df["total_mbps"] = pd.to_numeric(df["total_mbps"], errors="coerce").fillna(0.0)
    else:
        df["total_mbps"] = 0.0

    if "accepted_mbps" in df.columns:
        df["accepted_mbps"] = pd.to_numeric(df["accepted_mbps"], errors="coerce").fillna(0.0)
    else:
        df["accepted_mbps"] = 0.0

    df = df.dropna(subset=["time_step", "pdr"]).copy()
    df["time_step"] = df["time_step"].astype(int)

    # Convert pdr to percentage.
    # If pdr is already 0-100, keep it. If it is 0-1, multiply by 100.
    max_pdr = df["pdr"].max()

    if pd.notna(max_pdr) and max_pdr <= 1.5:
        df["pdr_pct"] = df["pdr"] * 100.0
    else:
        df["pdr_pct"] = df["pdr"]

    df["pdr_pct"] = df["pdr_pct"].clip(lower=0, upper=100)

    return df


def add_smoothed_pdr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling average per series."""
    df = df.sort_values(["series_label", "time_step"]).copy()

    if window <= 1:
        df["pdr_plot"] = df["pdr_pct"]
        return df

    df["pdr_plot"] = (
        df.groupby("series_label")["pdr_pct"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )

    return df


def build_plot_dataframe(run_dirs: list[Path], include_heuristics: bool = True) -> pd.DataFrame:
    """
    Build one dataframe with:
    - one agentic series per run folder
    - heuristics from the first run only
    """
    all_frames = []

    baseline_heuristics_added = False

    for run_dir in run_dirs:
        df = read_network_metrics(run_dir)

        if df.empty:
            continue

        agent_label = pretty_label_from_run(run_dir)

        agent_df = df[df["algorithm"].astype(str).eq("agentic")].copy()

        if not agent_df.empty:
            agent_df["series_label"] = agent_label
            agent_df["series_type"] = "agentic"
            agent_df["run_folder"] = run_dir.name
            all_frames.append(agent_df)

        if include_heuristics and not baseline_heuristics_added:
            heuristic_df = df[df["algorithm"].astype(str).isin(HEURISTIC_ORDER)].copy()

            if not heuristic_df.empty:
                heuristic_df["series_label"] = heuristic_df["algorithm"].map(HEURISTIC_LABELS)
                heuristic_df["series_type"] = "heuristic"
                heuristic_df["run_folder"] = run_dir.name
                all_frames.append(heuristic_df)
                baseline_heuristics_added = True

    if not all_frames:
        return pd.DataFrame()

    plot_df = pd.concat(all_frames, ignore_index=True)

    # Average duplicate rows if any.
    plot_df = (
        plot_df
        .groupby(["series_label", "series_type", "run_folder", "time_step"], as_index=False)
        .agg({
            "pdr_pct": "mean",
            "total_mbps": "sum",
            "accepted_mbps": "sum",
        })
    )

    return plot_df


def summarize_series(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table for the presentation/report."""
    rows = []

    for label, group in plot_df.groupby("series_label"):
        active = group[group["total_mbps"] > 0].copy()

        if active.empty:
            active = group.copy()

        total = active["total_mbps"].sum()
        accepted = active["accepted_mbps"].sum()

        cumulative_pdr = (accepted / total * 100.0) if total > 0 else active["pdr_pct"].mean()

        rows.append({
            "series": label,
            "type": active["series_type"].iloc[0],
            "mean_pdr_pct": active["pdr_pct"].mean(),
            "final_pdr_pct": active.sort_values("time_step")["pdr_pct"].iloc[-1],
            "cumulative_pdr_pct": cumulative_pdr,
            "accepted_mbps_total": accepted,
            "demanded_mbps_total": total,
        })

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        ["type", "cumulative_pdr_pct"],
        ascending=[True, False],
    )


def plot_pdr_comparison(
    plot_df: pd.DataFrame,
    output_dir: Path,
    smooth_window: int = 5,
    title: str = "PDR comparison: LLM agents vs heuristic baselines",
) -> None:
    """Create a slide-ready PDR line chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = add_smoothed_pdr(plot_df, smooth_window)

    fig, ax = plt.subplots(figsize=(13.33, 7.5))  # 16:9 slide format

    agent_colors = [
        "#0f766e",
        "#7c3aed",
        "#2563eb",
        "#db2777",
        "#ca8a04",
        "#dc2626",
        "#0891b2",
        "#9333ea",
    ]

    heuristic_styles = {
        "Heuristic: no delay margin": {"color": "#64748b", "linestyle": "--"},
        "Heuristic: low delay margin": {"color": "#f59e0b", "linestyle": "--"},
        "Heuristic: high delay margin": {"color": "#ef4444", "linestyle": "--"},
    }

    agent_index = 0

    # Plot heuristics first, then agents on top.
    ordered_labels = (
        plot_df[["series_label", "series_type"]]
        .drop_duplicates()
        .sort_values(["series_type", "series_label"])
    )

    heuristic_labels = ordered_labels[ordered_labels["series_type"] == "heuristic"]["series_label"].tolist()
    agent_labels = ordered_labels[ordered_labels["series_type"] == "agentic"]["series_label"].tolist()

    final_order = heuristic_labels + agent_labels

    for label in final_order:
        group = plot_df[plot_df["series_label"] == label].sort_values("time_step")

        if group.empty:
            continue

        series_type = group["series_type"].iloc[0]

        if series_type == "heuristic":
            style = heuristic_styles.get(
                label,
                {"color": "#64748b", "linestyle": "--"},
            )

            ax.plot(
                group["time_step"],
                group["pdr_plot"],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                alpha=0.85,
            )

        else:
            color = agent_colors[agent_index % len(agent_colors)]
            agent_index += 1

            ax.plot(
                group["time_step"],
                group["pdr_plot"],
                label=label,
                color=color,
                linestyle="-",
                linewidth=3.2,
                alpha=0.95,
            )

    subtitle = (
        ""
    )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475569",
    )

    ax.set_xlabel("Simulation step", fontsize=13)
    ax.set_ylabel("PDR / delivered demand (%)", fontsize=13)

    ax.set_ylim(0, 102)
    ax.set_xlim(left=plot_df["time_step"].min())

    ax.grid(True, which="major", linewidth=0.8, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=11)

    legend_columns = 2 if len(final_order) <= 6 else 3

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=legend_columns,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    png_path = output_dir / "pdr_comparison_presentation.png"
    pdf_path = output_dir / "pdr_comparison_presentation.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Folder containing one subfolder per simulation run.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="presentation_charts",
        help="Output folder for generated charts.",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Rolling average window for PDR. Use 1 for raw data.",
    )

    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Only plot LLM agentic models, no heuristic baselines.",
    )

    args = parser.parse_args()

    results_root = Path(args.results_dir).resolve()
    output_dir = Path(args.out_dir).resolve()

    print(f"[INFO] Results directory: {results_root}")

    run_dirs = discover_run_dirs(results_root)

    if not run_dirs:
        print("[ERROR] No run folders found.")
        return

    print(f"[INFO] Found {len(run_dirs)} run folder(s):")

    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")

    plot_df = build_plot_dataframe(
        run_dirs,
        include_heuristics=not args.no_heuristics,
    )

    if plot_df.empty:
        print("[ERROR] No valid PDR data found.")
        return

    summary = summarize_series(plot_df)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "pdr_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] Saved: {summary_path}")
    print("\nSummary:")
    print(summary[["series", "type", "cumulative_pdr_pct", "mean_pdr_pct"]].to_string(index=False))

    plot_pdr_comparison(
        plot_df=plot_df,
        output_dir=output_dir,
        smooth_window=max(1, args.smooth),
    )


if __name__ == "__main__":
    main()