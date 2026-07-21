"""Structured result export consumed by the Streamlit dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class DashboardResultsCollector:
    """Collect simulation state without coupling the dashboard to Simulator."""

    def __init__(
        self,
        graph,
        *,
        congestion_threshold: float,
        step_duration_seconds: float,
    ) -> None:
        self.graph = graph
        self.congestion_threshold = float(congestion_threshold)
        self.step_duration_seconds = float(step_duration_seconds)
        self.network_rows: list[dict[str, Any]] = []
        self.link_rows: list[dict[str, Any]] = []
        self.flow_rows: list[dict[str, Any]] = []

    def collect_step(self, algorithm: str, time_step: int, metrics: dict[str, Any]) -> None:
        accepted = [
            self._flow_row(algorithm, time_step, item, "accepted")
            for item in metrics.get("accepted_demands", [])
        ]
        dropped = [
            self._flow_row(algorithm, time_step, item, "dropped")
            for item in metrics.get("dropped_demands", [])
        ]
        flow_rows = accepted + dropped
        self.flow_rows.extend(flow_rows)

        self.network_rows.append({
            "algorithm": algorithm,
            "time_step": time_step,
            "total_mbps": metrics.get("total", 0.0),
            "accepted_mbps": metrics.get("accepted", 0.0),
            "dropped_mbps": metrics.get("dropped", 0.0),
            "pdr": metrics.get("acceptance_rate", 0.0),
            "loss_rate": metrics.get("loss_rate", 0.0),
            "active_flows": metrics.get(
                "active_request_count", metrics.get("active_flows", 0)
            ),
            "accepted_flows": metrics.get(
                "accepted_request_count", metrics.get("accepted_flows", 0)
            ),
            "latency_sla_drops": metrics.get("delay_dropped_request_count", 0),
            "no_valid_path_drops": metrics.get("no_valid_path_request_count", 0),
            "other_drops": metrics.get("other_dropped_request_count", 0),
            "route_changes": 0,
            "decision_age_steps": 0,
            "strategy": None,
            "situation": None,
            "max_primary_pressure": None,
            "congested_links": self._congested_link_count(),
        })

        seen: set[tuple[str, str]] = set()
        for (u, v), attrs in self.graph.links.items():
            source, target = sorted((u, v))
            key = (source, target)
            if key in seen:
                continue
            seen.add(key)
            capacity = float(attrs.get("capacity", 0.0))
            utilized = float(attrs.get("util", 0.0))
            utilization = utilized / capacity if capacity > 0 else 0.0
            self.link_rows.append({
                "algorithm": algorithm,
                "time_step": time_step,
                "source": source,
                "target": target,
                "capacity_mbps": capacity,
                "utilized_mbps": utilized,
                "utilization_pct": utilization * 100.0,
                "latency_ms": float(attrs.get("latency", 0.0)),
                "loss": float(attrs.get("loss", 0.0)),
                "up": bool(attrs.get("up", True)),
                "congested": utilization >= self.congestion_threshold,
            })

    def export(
        self,
        results_dir: str | Path,
        *,
        metadata: dict[str, Any],
        model_metrics: list[dict[str, Any]] | None = None,
        agent_decisions: list[dict[str, Any]] | None = None,
    ) -> None:
        output = Path(results_dir)
        output.mkdir(parents=True, exist_ok=True)

        network = pd.DataFrame(self.network_rows)
        decisions = pd.DataFrame(agent_decisions or [])
        if decisions.empty:
            decisions = pd.DataFrame(columns=[
                "algorithm", "step", "snapshot_step", "situation",
                "decision_source", "strategy", "active_flows",
                "routed_flows", "unresolved_flows", "route_changes",
                "decision_age_steps", "max_primary_pressure",
                "failed_route_flows", "candidate_flows",
                "no_candidate_flows",
            ])
        model_frame = pd.DataFrame(model_metrics or [])
        if model_frame.empty:
            model_frame = pd.DataFrame(columns=[
                "algorithm", "model_key", "provider", "model", "status",
                "inference_time_ms", "input_tokens", "output_tokens",
                "total_tokens", "step", "completed_step", "situation_type",
            ])
        network = self._attach_agent_decisions(network, decisions)

        network.to_csv(output / "network_metrics.csv", index=False)
        pd.DataFrame(self.link_rows).to_csv(output / "link_history.csv", index=False)
        pd.DataFrame(self.flow_rows).to_csv(output / "flow_events.csv", index=False)
        model_frame.to_csv(output / "model_metrics.csv", index=False)
        decisions.to_csv(output / "agent_decisions.csv", index=False)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sla": {
                "latency_requirement": "per-flow max_latency_ms",
                "capacity_requirement": "demand_mbps per shared link",
                "step_duration_seconds": self.step_duration_seconds,
            },
            "visualization": {
                "congestion_threshold": self.congestion_threshold,
            },
            **metadata,
        }
        with (output / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _attach_agent_decisions(
        network: pd.DataFrame,
        decisions: pd.DataFrame,
    ) -> pd.DataFrame:
        if network.empty or decisions.empty or "step" not in decisions.columns:
            return network

        fields = [
            "route_changes",
            "decision_age_steps",
            "strategy",
            "situation",
            "decision_source",
            "max_primary_pressure",
            "routed_flows",
            "unresolved_flows",
            "candidate_flows",
            "no_candidate_flows",
            "failed_route_flows",
        ]
        available = [field for field in fields if field in decisions.columns]
        keys = ["step"]
        if "algorithm" in decisions.columns:
            keys.insert(0, "algorithm")
        latest = decisions.drop_duplicates(keys, keep="last")

        for _, decision in latest.iterrows():
            step = int(decision["step"])
            algorithm = decision.get("algorithm", "agentic")
            mask = (
                network["algorithm"].eq(algorithm)
                & network["time_step"].eq(step)
            )
            for field in available:
                network.loc[mask, field] = decision[field]
        return network

    def _flow_row(
        self,
        algorithm: str,
        time_step: int,
        item: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        path = list(item.get("path", []))
        reason = item.get("reason", "accepted" if status == "accepted" else "unknown")
        latency = item.get("path_latency_ms")
        if latency is None:
            latency = self._path_latency_ms(path)
        max_latency = item.get("max_latency_ms")
        latency_violation = reason == "latency_requirement_not_met"
        return {
            "algorithm": algorithm,
            "time_step": time_step,
            "source": item.get("src"),
            "target": item.get("dst"),
            "demand_mbps": item.get("demand", 0.0),
            "path": " -> ".join(path),
            "hop_count": max(0, len(path) - 1),
            "e2e_latency_ms": latency,
            "max_latency_ms": max_latency,
            "status": status,
            "reason": reason,
            "latency_sla_violation": latency_violation,
            "sla_violation": status == "dropped",
        }

    def _path_latency_ms(self, path: list[str]) -> float | None:
        if len(path) < 2:
            return None
        total = 0.0
        for u, v in zip(path, path[1:]):
            link = self.graph.links.get((u, v))
            if link is None:
                return None
            total += float(link.get("latency", 0.0))
        return total

    def _congested_link_count(self) -> int:
        count = 0
        seen: set[tuple[str, str]] = set()
        for (u, v), attrs in self.graph.links.items():
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            capacity = float(attrs.get("capacity", 0.0))
            utilization = float(attrs.get("util", 0.0)) / capacity if capacity else 0.0
            if utilization >= self.congestion_threshold:
                count += 1
        return count
