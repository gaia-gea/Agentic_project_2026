import concurrent.futures
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Set, TypeAlias, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx

from config import algo_delays

Node = str
Link = Tuple[Node, Node]


@dataclass(frozen=True, slots=True)
class LinkId:
    u: Node
    v: Node


DemandSpec: TypeAlias = Dict[str, Any]
DemandSpecMap: TypeAlias = Dict[Tuple[Node, Node], DemandSpec]
DemandMap: TypeAlias = Dict[Tuple[Node, Node], float]


@dataclass(frozen=True, slots=True)
class FlowId:
    src: Node
    dst: Node


@dataclass(slots=True)
class LinkAttributes:
    capacity_mbps: float
    latency_ms: float
    loss: float
    fail_p: float
    up: bool
    weight: float
    util: float


@dataclass(slots=True)
class FlowEntry:
    src: Node
    dst: Node
    path: List[Node]
    demand_mbps: float | None = None
    installed_step: int | None = None
    version: int | None = None


@dataclass(slots=True)
class TopologySnapshot:
    version: int
    step: int
    nodes: Set[Node]
    links: Dict[LinkId, LinkAttributes]


@dataclass(slots=True)
class FlowTableSnapshot:
    version: int
    step: int
    flows: Dict[FlowId, FlowEntry]


@dataclass(slots=True)
class TrafficSnapshot:
    version: int
    step: int
    demands_mbps: Dict[FlowId, float]


@dataclass(slots=True)
class NetworkSnapshot:
    version: int
    step: int
    topology: TopologySnapshot
    flows: FlowTableSnapshot
    traffic: Optional[TrafficSnapshot] = None
    metadata: dict[str, Any] = field(default_factory=dict)


Severity: TypeAlias = Literal["info", "warning", "error"]


@dataclass(slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    location: Any | None = None


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass(slots=True)
class OperationResult:
    ok: bool
    message: str
    version_before: int | None = None
    version_after: int | None = None
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass(slots=True)
class FlowUpdate:
    src: Node
    dst: Node
    path: List[Node]
    demand_mbps: float | None = None


@dataclass(slots=True)
class LinkWeightUpdate:
    u: Node
    v: Node
    weight: float


@dataclass(slots=True)
class LinkStateUpdate:
    u: Node
    v: Node
    up: bool
    reason: str = "manual"


@dataclass(slots=True)
class RoutingPlan:
    flow_updates: List[FlowUpdate] = field(default_factory=list)
    weight_updates: List[LinkWeightUpdate] = field(default_factory=list)
    link_state_updates: List[LinkStateUpdate] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class CommitResult:
    ok: bool
    commit_id: str | None
    dry_run: bool
    message: str
    validation: ValidationResult
    version_before: int
    version_after: int | None
    applied_changes: int = 0


@dataclass(slots=True)
class CandidatePath:
    src: Node
    dst: Node
    path: List[Node]
    total_latency_ms: float
    bottleneck_capacity_mbps: float
    projected_max_util: float
    expected_loss: float
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class GoalReport:
    ok: bool
    score: float
    violations: List[ValidationIssue]
    metrics: Dict[str, float]


class NetworkGraph:
    def __init__(self):
        self.nodes: Set[Node] = set()
        # link dictionary: (u, v) -> {capacity, latency, loss, fail_p, up, weight, util}
        self.links: Dict[Link, Dict[str, Any]] = {}
        self.adjacency_list = defaultdict(list)

    def add_link(self, u: Node, v: Node, capacity: float, latency: float, loss: float, fail_p: float):
        self.nodes.update([u, v])
        self.links[(u, v)] = {
            "capacity": capacity,
            "latency": latency,  # per milliseconds
            "loss": loss,
            "fail_p": fail_p,
            "up": True,
            "weight": 1.0,
            "util": 0.0,
        }
        self.links[(v, u)] = dict(self.links[(u, v)])  # links are bidirectional
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)

    def set_link_attribute(self, link: Link, attr: Dict[str, Any]):
        self.links[link] = attr

    def get_neighbors(self, u: Node):
        return self.adjacency_list[u]

    def reset_link_states(self) -> None:
        """Reset util to 0.0 and up to True for every link."""
        for attrs in self.links.values():
            attrs["util"] = 0.0
            attrs["up"] = True
            attrs["weight"] = 1.0

    def draw_to_file(self, filename: str, seed: int):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)

        for (u, v), attrs in self.links.items():
            G.add_edge(u, v, **attrs)

        pos = nx.spring_layout(
            G,
            k=2,
            iterations=200,
            seed=seed
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=700,
            edge_color="#666666",
            node_color="#88b4e7",
        )

        edge_labels = {}
        for (u, v), a in G.edges.items():
            label_str = []
            for k, val in a.items():
                if k not in ["up", "weight", "util"]:
                    label_str.append(f"{k}={round(val, 2)}")
            edge_labels[(u, v)] = "\n".join(label_str)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_to_text_file(self, filename: str):
        ignored_keys = {"up", "weight", "util"}
        written_links = set()

        with open(filename, "w", encoding="utf-8") as f:
            f.write("Network Data\n")
            f.write("============\n\n")

            f.write("Nodes:\n")
            f.write("------\n")
            for node in self.nodes:
                f.write(f"{node} \t")

            f.write("\n------\n")
            f.write("\nLinks:\n")
            f.write("------\n")

            for (u, v), attrs in self.links.items():
                # Treat the link as bidirectional by sorting the endpoints
                link_key = tuple(sorted((u, v)))

                # Skip if this bidirectional link was already written
                if link_key in written_links:
                    continue

                written_links.add(link_key)

                f.write(f"{link_key[0]} -- {link_key[1]}\n")

                for key, value in attrs.items():
                    if key not in ignored_keys:
                        if isinstance(value, float):
                            value = round(value, 2)
                        f.write(f"  {key}: {value}\n")

                f.write("\n")


class TrafficModel:
    def __init__(self,
                 demands: DemandSpecMap,
                 seasonal_amplitude: float,
                 seasonal_frequency: float,
                 noise_power: float):
        self.base = demands
        self.max_latency_ms_by_pair = {
            pair: self._optional_float(spec.get("max_latency_ms"))
            for pair, spec in demands.items()
        }
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_frequency = seasonal_frequency
        self.noise_power = noise_power

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            out = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(out):
            return None

        return out

    def get_max_latency_ms(self, src: Node, dst: Node) -> float | None:
        return self.max_latency_ms_by_pair.get((src, dst))

    def step(self, t: int) -> Dict[Tuple[Node, Node], float]:
        """
        Stochastic Time-varying traffic:
        - 0 before each demand's start_time
        - after start_time: sinusoidal + random noise around base
        """
        out: Dict[Tuple[Node, Node], float] = {}

        for (src, dst), spec in self.base.items():
            base = spec["base"]
            start_time = spec["start_time"]

            if t < start_time:
                out[(src, dst)] = 0.0
                continue

            # Optional phase shift so sinusoid starts from demand start_time
            local_t = t - start_time

            seasonal = self.seasonal_amplitude * base * math.sin(
                2 * math.pi * local_t / self.seasonal_frequency
            )
            noise = random.uniform(-self.noise_power, self.noise_power) * base
            out[(src, dst)] = max(0.0, base + seasonal + noise)

        return out


class ControllerAPI:
    def __init__(self, graph: NetworkGraph):
        self.g = graph
        self.flow_table: Dict[FlowId, FlowEntry] = {}
        self.version = 0
        self.step_count = 0

    def reset(self) -> None:
        self.flow_table.clear()
        self.version = 0
        self.step_count = 0

    def get_topology_snapshot(self) -> TopologySnapshot:
        """
        Return a typed TopologySnapshot with node list and link attributes.
        Simple version uses self.version and self.step_count as placeholders.
        """
        links = {}
        for (u, v), data in self.g.links.items():
            # only store one direction in the snapshot
            if u < v:
                lid = LinkId(u=u, v=v)
                lobj = LinkAttributes(
                    capacity_mbps=data["capacity"],
                    latency_ms=data["latency"],
                    loss=data["loss"],
                    fail_p=data["fail_p"],
                    up=data["up"],
                    weight=data["weight"],
                    util=data["util"]
                )
                links[lid] = lobj

        snapshot = TopologySnapshot(
            version=self.version,
            step=self.step_count,
            nodes=set(self.g.nodes),
            links=links
        )
        return snapshot

    def get_traffic_snapshot(
            self,
            demands: Dict[Tuple[Node, Node], float] | Dict[FlowId, float] | TrafficSnapshot | None = None
    ) -> TrafficSnapshot:
        demands_mbps: Dict[FlowId, float] = {}

        if demands is None:
            for fid, entry in self.flow_table.items():
                if entry.demand_mbps is not None:
                    demands_mbps[FlowId(src=fid.src, dst=fid.dst)] = float(entry.demand_mbps)

        elif isinstance(demands, TrafficSnapshot):
            demands_mbps = dict(demands.demands_mbps)

        else:
            for key, demand in demands.items():
                if isinstance(key, FlowId):
                    fid = key
                else:
                    src, dst = key
                    fid = FlowId(src=src, dst=dst)

                demands_mbps[fid] = float(demand)

        return TrafficSnapshot(
            version=self.version,
            step=self.step_count,
            demands_mbps=demands_mbps
        )

    def get_flow_table_snapshot(self) -> FlowTableSnapshot:
        """
        Return installed flow paths as a typed FlowTableSnapshot.
        """
        flows_copy: Dict[FlowId, FlowEntry] = {}

        for fid, entry in self.flow_table.items():
            flows_copy[fid] = FlowEntry(
                src=entry.src,
                dst=entry.dst,
                path=list(entry.path),
                demand_mbps=entry.demand_mbps,
                installed_step=entry.installed_step,
                version=entry.version
            )

        return FlowTableSnapshot(
            version=self.version,
            step=self.step_count,
            flows=flows_copy
        )

    def set_flow(
            self,
            src: Node,
            dst: Node,
            path: List[Node],
            demand_mbps: float | None = None
    ):
        """
        Install or replace a flow using FlowId and FlowEntry.
        """
        fid = FlowId(src=src, dst=dst)

        self.flow_table[fid] = FlowEntry(
            src=src,
            dst=dst,
            path=list(path),
            demand_mbps=demand_mbps,
            installed_step=self.step_count,
            version=self.version
        )

    def set_link_weight(self, u: Node, v: Node, w: float):
        if (u, v) in self.g.links:
            self.g.links[(u, v)]["weight"] = w
            self.g.links[(v, u)]["weight"] = w

    def reset_link_state(self, u: Node, v: Node, up: bool = True):
        if (u, v) in self.g.links:
            self.g.links[(u, v)]["up"] = up
            self.g.links[(v, u)]["up"] = up

    def validate_path_logic(self, src: Node, dst: Node, path: List[Node]) -> bool:
        """
        Basic path validation:
          - path starts at src and ends at dst
          - each consecutive hop is an 'up' link
          - no repeated loops for simplicity
        """
        if not path or path[0] != src or path[-1] != dst:
            return False

        visited = set()
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if (u, v) not in self.g.links or not self.g.links[(u, v)]["up"]:
                return False
            if u in visited:
                return False
            visited.add(u)
        return True

    def validate_path_sla(self, path: List[Node], demand_mbps: float | None = None) -> bool:
        """
        Simple example:
         - Check if all the path links have enough capacity for the demand
         - Skip if demand is None
         Note: this function does not account for other traffic,
          so even if it returns True, success is not guaranteed.
        """
        if demand_mbps is None:
            return True
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if demand_mbps > self.g.links[(u, v)]["capacity"]:
                return False
        return True

    def step(self):
        """
        Increment controller's notion of time and version.
        """
        self.step_count += 1
        self.version += 1


class Simulator:
    def __init__(self,
                 graph: NetworkGraph,
                 traffic: TrafficModel,
                 controller: ControllerAPI,
                 routing_agent,
                 verbosity: int = 1,
                 ):
        self.g = graph
        self.traffic = traffic
        self.ctrl = controller
        self.agentic_manager = routing_agent
        self.verbosity = verbosity

        self.t = 0
        self.flow_metrics = {}

        # ── Async agentic-routing state ───────────────────────────────────
        self._route_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="agentic_router"
        )
        self._pending_future: Optional[concurrent.futures.Future] = None
        self._last_agentic_paths: Optional[Dict[Tuple[Node, Node], List[Node]]] = None
        self._agentic_submit_t: Optional[int] = None

        # ── Async heuristic-routing state (one pipeline per algo) ────────  ← NEW
        self._heuristic_executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self._heuristic_futures: Dict[str, Optional[concurrent.futures.Future]] = {}
        self._heuristic_paths: Dict[str, Optional[Dict[Tuple[Node, Node], List[Node]]]] = {}
        self._heuristic_submit_t: Dict[str, Optional[int]] = {}

    def close(self) -> None:
        """Cancel any in-flight LLM call and release the thread pool."""
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._route_executor.shutdown(wait=False)

        for algo, fut in self._heuristic_futures.items():
            if fut is not None:
                fut.cancel()

        self._heuristic_futures.clear()
        self._heuristic_paths.clear()
        self._heuristic_submit_t.clear()

        for executor in self._heuristic_executors.values():
            executor.shutdown(wait=False)
        self._heuristic_executors.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def reset_agentic_state(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._pending_future = None
        self._last_agentic_paths = None
        self._agentic_submit_t = None

        for algo, fut in self._heuristic_futures.items():
            if fut is not None:
                fut.cancel()
        self._heuristic_futures.clear()
        self._heuristic_paths.clear()
        self._heuristic_submit_t.clear()

        self.t = 0

    def induce_failures(self):
        for (u, v), attr in self.g.links.items():
            if u < v:  # only do one direction
                if random.random() < attr["fail_p"]:
                    self.g.links[(u, v)]["up"] = False
                    self.g.links[(v, u)]["up"] = False
                else:
                    self.g.links[(u, v)]["up"] = True
                    self.g.links[(v, u)]["up"] = True

    @staticmethod
    def _normalize_demands_to_flow_ids(
            demands: DemandMap | TrafficSnapshot | Dict[FlowId, float]
    ) -> Dict[FlowId, float]:
        """
        Accepts either:
          - DemandMap: Dict[Tuple[Node, Node], float]
          - Dict[FlowId, float]
          - TrafficSnapshot

        Returns:
          - Dict[FlowId, float]
        """
        if isinstance(demands, TrafficSnapshot):
            return dict(demands.demands_mbps)

        normalized: Dict[FlowId, float] = {}

        for key, demand in demands.items():
            if isinstance(key, FlowId):
                fid = key
            else:
                src, dst = key
                fid = FlowId(src=src, dst=dst)

            normalized[fid] = demand

        return normalized

    def route_flows_heuristic(
            self,
            demands: DemandMap | TrafficSnapshot | Dict[FlowId, float]
    ) -> FlowTableSnapshot:
        """
        Simple heuristic compatible with FlowId, FlowEntry, and FlowTableSnapshot.

        For each demand:
          - reuse the existing installed flow if present
          - otherwise compute the shortest path by link weight

        Returns:
          FlowTableSnapshot
        """
        demand_by_flow_id = self._normalize_demands_to_flow_ids(demands)

        suggested_flows: Dict[FlowId, FlowEntry] = {}

        for fid, demand_mbps in demand_by_flow_id.items():
            existing_entry = self.ctrl.flow_table.get(fid)

            if existing_entry is not None:
                path = list(existing_entry.path)
                installed_step = existing_entry.installed_step
                entry_version = existing_entry.version
            else:
                path = self._calculate_shortest_path(fid.src, fid.dst)
                installed_step = None
                entry_version = None

            suggested_flows[fid] = FlowEntry(
                src=fid.src,
                dst=fid.dst,
                path=path,
                demand_mbps=demand_mbps,
                installed_step=installed_step,
                version=entry_version
            )

        return FlowTableSnapshot(
            version=self.ctrl.version,
            step=self.ctrl.step_count,
            flows=suggested_flows,
        )

    def _calculate_shortest_path(self, src: Node, dst: Node) -> List[Node]:
        # Dijkstra-like approach by 'weight'
        dist = {n: float("inf") for n in self.g.nodes}
        prev = {n: None for n in self.g.nodes}
        dist[src] = 0.0
        q = set(self.g.nodes)

        while q:
            u = min(q, key=lambda x: dist[x])
            q.remove(u)
            if u == dst:
                break
            for v in self.g.get_neighbors(u):
                if not self.g.links[(u, v)]["up"]:
                    continue
                alt = dist[u] + self.g.links[(u, v)]["weight"]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

        if dist[dst] == float("inf"):
            return []
        path, curr = [], dst
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        return list(reversed(path))

    def compute_metrics(self,
                        demands: DemandMap,
                        paths: Dict[Tuple[Node, Node], List[Node]]
                        ) -> Dict[str, Any]:
        """
        Updates link utilization, tallies dropped/total traffic,
        and returns the list of accepted demands.

        A demand is accepted if:
        - a path exists,
        - all links on the path are up,
        - all links have enough remaining capacity,
        - total path latency is no higher than the demand's latency requirement.
        """

        # Reset utilization
        for k in self.g.links:
            self.g.links[k]["util"] = 0.0

        dropped = 0.0
        total = 0.0
        latency_dropped = 0.0
        active_request_count = 0
        accepted_request_count = 0
        delay_dropped_request_count = 0
        no_valid_path_request_count = 0
        other_dropped_request_count = 0

        accepted_demands = []
        dropped_demands = []

        for (src, dst), demand in demands.items():
            total += demand
            is_active_request = demand > 1e-12
            if is_active_request:
                active_request_count += 1

            path = paths.get((src, dst), [])
            max_latency_ms = self.traffic.get_max_latency_ms(src, dst)

            if not path:
                dropped += demand
                if is_active_request:
                    no_valid_path_request_count += 1
                dropped_demands.append({
                    "src": src,
                    "dst": dst,
                    "demand": demand,
                    "max_latency_ms": max_latency_ms,
                    "reason": "no_path"
                })
                continue

            feasible = True
            reason = None
            path_latency_ms = 0.0

            # Check link capacities, availability, and latency SLA.
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link = self.g.links[(u, v)]
                path_latency_ms += float(link.get("latency", 0.0))

                if not link["up"]:
                    feasible = False
                    reason = "link_down"
                    break

                if link["util"] + demand > link["capacity"]:
                    feasible = False
                    reason = "insufficient_capacity"
                    break

            if (
                    feasible
                    and max_latency_ms is not None
                    and path_latency_ms > max_latency_ms + 1e-9
            ):
                feasible = False
                reason = "latency_requirement_not_met"

            if not feasible:
                dropped += demand
                if reason == "latency_requirement_not_met":
                    latency_dropped += demand
                    if is_active_request:
                        delay_dropped_request_count += 1
                elif reason == "link_down":
                    if is_active_request:
                        no_valid_path_request_count += 1
                elif is_active_request:
                    other_dropped_request_count += 1

                dropped_demands.append({
                    "src": src,
                    "dst": dst,
                    "demand": demand,
                    "path": path,
                    "path_latency_ms": path_latency_ms,
                    "max_latency_ms": max_latency_ms,
                    "reason": reason
                })
                continue

            # Accept demand and update utilization
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.g.links[(u, v)]["util"] += demand

            accepted_demands.append({
                "src": src,
                "dst": dst,
                "demand": demand,
                "path": path,
                "path_latency_ms": path_latency_ms,
                "max_latency_ms": max_latency_ms,
            })
            if is_active_request:
                accepted_request_count += 1

        loss_rate = dropped / total if total > 0 else 0.0
        accepted = total - dropped
        acceptance_rate = accepted / total if total > 0 else 0.0

        metrics = {
            "dropped": dropped,
            "latency_dropped": latency_dropped,
            "accepted": accepted,
            "total": total,
            "loss_rate": loss_rate,
            "acceptance_rate": acceptance_rate,
            "active_request_count": active_request_count,
            "accepted_request_count": accepted_request_count,
            "delay_dropped_request_count": delay_dropped_request_count,
            "no_valid_path_request_count": no_valid_path_request_count,
            "other_dropped_request_count": other_dropped_request_count,
            "accepted_demands": accepted_demands,
            "dropped_demands": dropped_demands
        }

        return metrics

    # ── Non-blocking agentic and heuristic routing ──────────────────────────────────────

    def _get_agentic_paths_async(
            self,
            demands: Dict[Tuple[Node, Node], float],
    ) -> Dict[Tuple[Node, Node], List[Node]]:
        """
        Non-blocking wrapper around ``agentic_manager.route_flows``.

        Lifecycle per step
        ------------------
        1. If the in-flight future just finished, collect its result.
        2. If the pipeline is idle, submit a fresh request immediately.
        3. Return the most recently completed agentic routes, or fall back
           to the heuristic if no agentic result is available yet.
        """
        # 1. Collect a completed future
        if self._pending_future is not None and self._pending_future.done():
            try:
                self._last_agentic_paths = self._pending_future.result()
                if self.verbosity >= 2:
                    print(
                        f"[t={self.t}] ✓ Agentic routes received "
                        f"(submitted at t={self._agentic_submit_t})."
                    )
            except Exception as exc:
                if self.verbosity >= 2:
                    print(f"[t={self.t}] ✗ Agentic routing error: {exc}")
            finally:
                self._pending_future = None
                self._agentic_submit_t = None

        # 2. Submit a new request when the pipeline is idle
        if self._pending_future is None:
            frozen_demands = dict(demands)  # shallow copy — safe to pass to thread
            self._pending_future = self._route_executor.submit(
                self.agentic_manager.route_flows,
                frozen_demands,
            )
            self._agentic_submit_t = self.t
            if self.verbosity >= 2:
                print(f"[t={self.t}] → Agentic route request submitted.")
        else:
            if self.verbosity >= 2:
                print(
                    f"[t={self.t}] … Agentic routing in progress "
                    f"(submitted at t={self._agentic_submit_t})."
                )

        # 3. Return best available routes
        if self._last_agentic_paths is not None:
            return self._last_agentic_paths

        # No agentic result yet — heuristic warm-start
        if self.verbosity >= 1:
            print(f"[t={self.t}] ⚠ No agentic routes yet; using heuristic fallback.")
        flow_snapshot = self.route_flows_heuristic(demands)
        return {
            (fid.src, fid.dst): entry.path
            for fid, entry in flow_snapshot.flows.items()
            if entry.path
        }

    def _run_heuristic(
            self,
            algo: str,
            demands: Dict[Tuple[Node, Node], float],
            delay: float,
    ) -> Dict[Tuple[Node, Node], List[Node]]:
        """
        Worker executed in the thread pool.
        Simulates routing latency with an artificial delay, then
        runs the heuristic and returns the resulting path dict.
        """
        time.sleep(delay)  # ← artificial network/compute delay

        flow_snapshot = self.route_flows_heuristic(demands)
        return {
            (fid.src, fid.dst): entry.path
            for fid, entry in flow_snapshot.flows.items()
            if entry.path
        }

    def _get_heuristic_paths_async(
            self,
            algo: str,
            demands: Dict[Tuple[Node, Node], float],
            delay: float,
    ) -> Dict[Tuple[Node, Node], List[Node]]:
        """
        Non-blocking heuristic routing — mirrors _get_agentic_paths_async.

        Lifecycle per step
        ------------------
        1. If the in-flight future for this algo just finished, collect its result.
        2. If the pipeline is idle, submit a fresh request immediately.
        3. Return the most recently completed result
        """
        # Lazily initialise per-algo state on first encounter
        if algo not in self._heuristic_executors:
            self._heuristic_executors[algo] = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{algo}_router"
            )
            self._heuristic_futures[algo] = None
            self._heuristic_paths[algo] = None
            self._heuristic_submit_t[algo] = None

        # 1. Collect a completed future
        fut = self._heuristic_futures[algo]
        if fut is not None and fut.done():
            try:
                self._heuristic_paths[algo] = fut.result()
                if self.verbosity >= 2:
                    print(
                        f"[t={self.t}] ✓ [{algo}] routes received "
                        f"(submitted at t={self._heuristic_submit_t[algo]})."
                    )
            except Exception as exc:
                if self.verbosity >= 1:
                    print(f"[t={self.t}] ✗ [{algo}] routing error: {exc}")
            finally:
                self._heuristic_futures[algo] = None
                self._heuristic_submit_t[algo] = None

        # 2. Submit a new request when the pipeline is idle
        if self._heuristic_futures[algo] is None:
            frozen_demands = dict(demands)
            self._heuristic_futures[algo] = self._heuristic_executors[algo].submit(
                self._run_heuristic,
                algo,
                frozen_demands,
                delay,
            )
            self._heuristic_submit_t[algo] = self.t
            if self.verbosity >= 2:
                print(f"[t={self.t}] → [{algo}] route request submitted (delay={delay}s).")
        else:
            if self.verbosity >= 2:
                print(
                    f"[t={self.t}] … [{algo}] routing in progress "
                    f"(submitted at t={self._heuristic_submit_t[algo]})."
                )

        # 3. Return best available routes
        if self._heuristic_paths[algo] is not None:
            return self._heuristic_paths[algo]

        # No result yet — return only already-installed flows, no synchronous fallback
        if self.verbosity >= 2:
            print(f"[t={self.t}] ⚠ [{algo}] no routes yet; serving installed flows only.")
        return {
            (fid.src, fid.dst): entry.path
            for fid, entry in self.ctrl.flow_table.items()
            if (fid.src, fid.dst) in demands and entry.path
        }

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self, algo: str):
        """
        Orchestrates one simulation step:
        1) Induce random failures
        2) Get new demands from traffic model
        3) Route flows (heuristic or agentic)
        4) Apply the chosen routes in the controller
        5) Compute metrics
        6) Advance time, increment controller steps
        """

        # step 1
        self.induce_failures()

        # Step 2
        demands = self.traffic.step(self.t)

        # Step 3
        if algo in algo_delays:
            delay = algo_delays[algo]
            suggested_paths = self._get_heuristic_paths_async(
                algo, demands, delay
            )

        elif algo == "agentic":
            suggested_paths = self._get_agentic_paths_async(demands)
        else:
            raise NotImplementedError(f"Algorithm '{algo}' not implemented.")

        # Step 4
        for (src, dst), path in suggested_paths.items():
            if path:
                self.ctrl.set_flow(src, dst, path, demand_mbps=demands.get((src, dst)))

        # Step 5
        metrics = self.compute_metrics(demands, suggested_paths)

        # Bump time and controller
        time.sleep(0.25)
        self.t += 1
        self.ctrl.step()

        return demands, metrics
