import os
import random
import json
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from agents_v5 import RoutingAgent
from config import *
from helpers import create_random_network, create_random_traffic_pattern
from simulation import NetworkGraph, TrafficModel, ControllerAPI, Simulator

if __name__ == "__main__":

    # Set random seeds for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        if verbosity_level >= 1:
            print(f"Random seed set to: {random_seed}")

    # Keep every experiment in a unique, model-labelled directory so later
    # simulations never overwrite earlier results.
    model_config = MODEL_CONFIGS[ACTIVE_AGENT_MODEL]
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_config["model"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}__{ACTIVE_AGENT_MODEL}__{model_name}"
    results_dir = os.path.join("results", run_id)
    suffix = 2
    while os.path.exists(results_dir):
        results_dir = os.path.join("results", f"{run_id}__{suffix}")
        suffix += 1
    os.makedirs(results_dir)

    def result_path(filename):
        return os.path.join(results_dir, filename)

    # Create network
    g = NetworkGraph()
    create_random_network(g, seed=random_seed, **network_args)

    # (Optional) visualize
    g.draw_to_file(result_path("network.png"), random_seed)
    g.save_to_text_file(result_path("network.txt"))

    # Create traffic model
    base_demands = create_random_traffic_pattern(
        g,
        seed=random_seed,
        base_demand=base_demand,
        start_t_max=num_steps,
        **traffic_pattern_args,
    )
    traffic = TrafficModel(base_demands, **traffic_args)

    # Setup controller + agent
    ctrl = ControllerAPI(g)
    llm_agent = RoutingAgent(ctrl, sla_resolver=traffic.get_max_latency_ms)

    with open(result_path("run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": os.path.basename(results_dir),
                "generated_at": datetime.now().isoformat(),
                "random_seed": random_seed,
                "num_steps": num_steps,
                "algorithms": algorithms,
                "agent": "agents_v5",
                "model_key": llm_agent.model_key,
                "provider": llm_agent.model_config["provider"],
                "model": llm_agent.model_config["model"],
            },
            handle,
            indent=2,
        )

    # Collect metrics
    all_history = {}
    all_history_per_demand = {}
    request_outcome_keys = [
        ("Accepted", "accepted_request_count", "#4c78a8"),
        ("Rejected: delay/SLA", "delay_dropped_request_count", "#f58518"),
        ("Rejected: no valid path", "no_valid_path_request_count", "#e45756"),
        ("Rejected: other", "other_dropped_request_count", "#72b7b2"),
    ]
    request_outcome_totals = {
        algo: {key: 0 for _, key, _ in request_outcome_keys}
        for algo in algorithms
    }

    with Simulator(g, traffic, ctrl, llm_agent, verbosity=verbosity_level) as simulator:
        for algo in algorithms:

            simulator.reset_agentic_state()
            ctrl.reset()
            g.reset_link_states()

            history = {k: [] for k in metric_names}
            history_per_demand = pd.DataFrame(
                columns=[(src, dst) for src in g.nodes for dst in g.nodes],
                index=range(num_steps),
            )
            for t in tqdm(range(num_steps)):
                _, metrics = simulator.step(algo)
                for k in metric_names:
                    history[k].append(metrics.get(k, 0.0))
                for _, key, _ in request_outcome_keys:
                    request_outcome_totals[algo][key] += metrics.get(key, 0)
                for item in metrics["accepted_demands"]:
                    history_per_demand.at[t, (item["src"], item["dst"])] = item["demand"]

            all_history[algo] = history
            all_history_per_demand[algo] = history_per_demand

    # Plot metrics
    fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 4), sharex=True)
    axes = axes.flatten()

    for ax, name in zip(axes, metric_names):
        for algo in algorithms:
            ax.plot(all_history[algo][name], label=algo, color=colors[algo])
        ax.set_title(name)
        ax.set_xlabel("time step")
        ax.set_ylabel(name)
        if name == "loss_rate":
            ax.set_ylim([0, 1])
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(result_path("results.png"), dpi=150)
    plt.close(fig)

    # Plot aggregate request outcomes as one stacked bar per algorithm.
    fig_bar, ax_bar = plt.subplots(figsize=(max(8, 1.4 * len(algorithms) + 3), 5))
    x = np.arange(len(algorithms))
    bottom = np.zeros(len(algorithms))

    for label, key, color in request_outcome_keys:
        values = np.array([request_outcome_totals[algo][key] for algo in algorithms])
        ax_bar.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values

    ax_bar.set_title("Request outcomes by algorithm")
    ax_bar.set_xlabel("algorithm")
    ax_bar.set_ylabel("number of active requests")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(algorithms, rotation=20, ha="right")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend()
    fig_bar.tight_layout()
    fig_bar.savefig(result_path("request_outcomes_stacked_bar.png"), dpi=150)
    plt.close(fig_bar)

    request_outcomes = pd.DataFrame.from_dict(request_outcome_totals, orient="index")
    request_outcomes.to_csv(result_path("request_outcomes_by_algorithm.csv"))

    for algo in algorithms:
        all_history_per_demand[algo].to_csv(
            result_path(f"all_history_{algo}.csv")
        )

    print(f"Simulation completed. Results were saved to {results_dir}.")
