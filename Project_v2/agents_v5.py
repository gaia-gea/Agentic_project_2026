from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal, Optional, TypedDict

import networkx as nx

from config import ACTIVE_AGENT_MODEL, MODEL_CONFIGS, verbosity_level
from llm_client import call_llm
from simulation import FlowId


Situation = Literal["trivial", "failure", "conflict"]
StrategyName = Literal[
    "maximize_acceptance",
    "minimize_delay",
    "protect_capacity",
    "stable_recovery",
]


class AgentState(TypedDict, total=False):
    demands: dict[tuple[str, str], float]
    topology: Any
    flow_table: Any
    situation_type: Situation
    situation_summary: dict
    pressure_summary: dict
    event_key: str
    strategy: dict
    routes: dict[tuple[str, str], list[str]]
    validation_results: dict[tuple[str, str], bool]
    strategy_source: str
    fresh_plan: bool
    routing_diagnostics: dict
    log: list[dict]
    start: float


STRATEGY_CONFIG: dict[str, str] = {
    "maximize_acceptance": (
        "Prioritize admitting more requests under scarce capacity."
    ),
    "minimize_delay": (
        "Prioritize tight latency SLAs and the lowest-delay candidates."
    ),
    "protect_capacity": (
        "Preserve link headroom and avoid high projected utilization."
    ),
    "stable_recovery": (
        "Preserve valid installed routes and minimize routing churn."
    ),
}

PLAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": list(STRATEGY_CONFIG)},
        "reason": {"type": "string"},
    },
    "required": ["strategy", "reason"],
    "additionalProperties": False,
}


class RoutingAgent:
    """Hybrid agent compatible with ``RoutingAgent(controller).route_flows``."""

    def __init__(
        self,
        controller,
        model_key: Optional[str] = None,
        sla_resolver=None,
    ):
        self.ctrl = controller
        self.model_key = model_key or ACTIVE_AGENT_MODEL
        self.model_config = MODEL_CONFIGS[self.model_key]
        self.sla_resolver = sla_resolver
        self.graph = self._build_graph()

        # One active supervisory plan. It always comes from a valid LLM
        # response and remains installed until another valid response replaces it.
        self.current_plan: Optional[dict] = None
        self.plan_update_count = 0
        self._last_submitted_event_key: Optional[str] = None

        # The LLM is the only slow operation owned by V5. The simulator's
        # routing worker must never wait for this future.
        self._llm_executor: Optional[ThreadPoolExecutor] = None
        self._llm_future: Optional[Future] = None
        self._llm_future_context: Optional[dict] = None

        self.decision_log: list[dict] = []
        self.model_metrics: list[dict] = []
        self.llm_call_metrics: list[dict] = []
        self.flow_decision_metrics: list[dict] = []
        self.invalid_action_count = 0
        self.last_inference_time_ms = 0.0

    def route_flows(self, demands):
        """Run one fast LangGraph cycle without waiting for an LLM response."""
        started = time.perf_counter()
        result = self.graph.invoke({
            "demands": self._normalize_demands(demands),
            "start": started,
        })
        self.last_inference_time_ms = (time.perf_counter() - started) * 1000
        self.decision_log.extend(result.get("log", []))
        return result.get("routes", {})

    def close(self, wait: bool = True):
        """Release the LLM worker. Safe to call more than once."""
        executor = self._llm_executor
        if executor is None:
            return
        if self._llm_future is not None and not self._llm_future.done():
            self._llm_future.cancel()
        executor.shutdown(wait=wait, cancel_futures=True)
        self._llm_executor = None

    def get_contribution_summary(self):
        """Return weighted routing-source totals across all active decisions."""
        rows = [row for row in self.flow_decision_metrics if row["active_flows"] > 0]
        total_active = sum(row["active_flows"] for row in rows)
        total_fast = sum(row["fast_path_flows"] for row in rows)
        total_complex = sum(row["complex_flows"] for row in rows)
        total_recovered = sum(row["llm_recovered_flows"] for row in rows)
        total_unresolved = sum(row["unresolved_complex_flows"] for row in rows)
        return {
            "decision_cycles": len(rows),
            "peak_active_flows": max((row["active_flows"] for row in rows), default=0),
            "flow_step_decisions": total_active,
            "fast_path_flow_steps": total_fast,
            "complex_flow_steps": total_complex,
            "llm_recovered_flow_steps": total_recovered,
            "unresolved_complex_flow_steps": total_unresolved,
            "fast_path_ratio": total_fast / total_active if total_active else 0.0,
            "complex_ratio": total_complex / total_active if total_active else 0.0,
            "llm_complex_recovery_ratio": (
                total_recovered / total_complex if total_complex else 0.0
            ),
            "llm_calls_completed": len(self.llm_call_metrics),
            "llm_plan_updates": self.plan_update_count,
        }

    # ------------------------------------------------------------------
    # LangGraph
    # ------------------------------------------------------------------

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("perceive", self._perceive)
        graph.add_node("classify", self._classify)
        graph.add_node("fast_path", self._fast_path)
        graph.add_node("plan_llm", self._plan_llm)
        graph.add_node("route_with_strategy", self._route_with_strategy)
        graph.add_node("validate", self._validate)
        graph.add_node("record", self._record)

        graph.add_edge(START, "perceive")
        graph.add_edge("perceive", "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: "fast" if state["situation_type"] == "trivial" else "llm",
            {"fast": "fast_path", "llm": "plan_llm"},
        )
        graph.add_edge("fast_path", "validate")
        graph.add_edge("plan_llm", "route_with_strategy")
        graph.add_edge("route_with_strategy", "validate")
        graph.add_edge("validate", "record")
        graph.add_edge("record", END)
        return graph.compile()

    def _perceive(self, state: AgentState):
        state["topology"] = self.ctrl.get_topology_snapshot()
        state["flow_table"] = self.ctrl.get_flow_table_snapshot()
        state["routes"] = {}
        state["validation_results"] = {}
        state["log"] = []
        self._collect_llm_result(state)
        return state

    def _classify(self, state: AgentState):
        """Classify from observable constraints; it does not choose paths."""
        active = self._active_demands(state)
        direct_available = 0
        direct_failed = 0
        direct_insufficient = 0

        for (src, dst), demand in active:
            link = self._link(state["topology"], src, dst)
            if link is not None and not link.up:
                direct_failed += 1
            elif link is None or (
                demand > link.capacity_mbps
                or not self._path_within_latency_sla(
                    (src, dst), [src, dst], state["topology"]
                )
            ):
                direct_insufficient += 1
            else:
                direct_available += 1

        pressure = self._calculate_pressure(state, active)
        state["pressure_summary"] = pressure
        pressure_threshold = float(
            self._get_config("V5_PRESSURE_THRESHOLD", 1.0)
        )
        failed_network_links = sum(
            not link.up for link in state["topology"].links.values()
        )

        if failed_network_links:
            situation: Situation = "failure"
        elif direct_insufficient or pressure["max_pressure"] > pressure_threshold:
            situation = "conflict"
        else:
            situation = "trivial"

        state["situation_type"] = situation
        state["situation_summary"] = {
            "active_flows": len(active),
            "direct_routes_available": direct_available,
            "direct_routes_failed": direct_failed,
            "direct_routes_insufficient": direct_insufficient,
            "failed_network_links": failed_network_links,
            "max_pressure": pressure["max_pressure"],
            "overloaded_links": pressure["overloaded_links"],
        }
        state["event_key"] = self._event_key(state)
        return state

    # ------------------------------------------------------------------
    # Trivial deterministic branch
    # ------------------------------------------------------------------

    def _fast_path(self, state: AgentState):
        """Apply the routine low-latency objective with one shared ledger."""
        routes = {}
        reservations = {}
        active = self._ordered_demands(
            self._active_demands(state), "smallest_first"
        )
        for flow, demand in active:
            direct = [flow[0], flow[1]]
            if self._exact_path_valid(
                flow, direct, demand, state["topology"]
            ) and self._has_capacity(
                state["topology"], direct, demand, reservations
            ):
                routes[flow] = direct
                self._reserve(direct, demand, reservations)
        for (src, dst), path in routes.items():
            state["log"].append({
                "step": self.ctrl.step_count,
                "flow": f"{src}->{dst}",
                "action": "agent_invoked_joint_routine_allocator",
                "status": "proposed",
                "path": path,
            })
        state["routes"] = routes
        state["strategy_source"] = "not_needed"
        state["routing_diagnostics"] = {
            "fast_path_flows": sum(
                path == [flow[0], flow[1]] for flow, path in routes.items()
            ),
            "complex_flows": 0,
            "llm_planned_flows": 0,
            "llm_recovered_flows": 0,
            "llm_pending_flows": 0,
            "unresolved_complex_flows": len(active) - len(routes),
            "candidate_flows": len(active),
            "no_candidate_flows": 0,
            "allocator": "joint_direct_routine",
            "exact_latency_sla": self.sla_resolver is not None,
        }
        return state

    # ------------------------------------------------------------------
    # LLM strategy branch
    # ------------------------------------------------------------------

    def _plan_llm(self, state: AgentState):
        situation = state["situation_type"]
        event_key = state["event_key"]
        if self.current_plan is not None:
            state["strategy"] = self.current_plan
            state["strategy_source"] = (
                "new_llm" if state.get("fresh_plan") else "active_plan"
            )

        # Never wait for the model. The active plan (if any) keeps controlling
        # current complex flows while a replacement is being generated.
        if self._llm_future is not None:
            if self.current_plan is None:
                state["strategy_source"] = "pending"
            state["log"].append({
                "step": self.ctrl.step_count,
                "action": "llm_strategy_pending",
                "status": "pending",
                "situation_type": situation,
            })
            return state

        context_changed = event_key != self._last_submitted_event_key
        if not context_changed:
            if self.current_plan is None:
                state["strategy_source"] = "same_event_waiting"
            return state

        payload = {
            "task": "Select the objective for a deterministic joint allocator.",
            "allowed_strategies": list(STRATEGY_CONFIG),
            "strategy_guide": STRATEGY_CONFIG,
            "network": self._network_summary(state),
            "case": self._compact_case(state),
            "required_output": {
                "strategy": "exact allowed name",
                "reason": "max 12 words",
            },
        }
        system_prompt = (
            "You supervise a deterministic network optimizer. Use the compact "
            "candidate and SLA evidence to choose its allocation objective. "
            "Choose based on the current evidence; do not alternate strategies "
            "merely for variety. "
            "Do not invent paths or IDs. Return only compact JSON."
        )
        submitted_step = self.ctrl.step_count
        self._last_submitted_event_key = event_key
        self._llm_future_context = {
            "situation_type": situation,
            "event_key": event_key,
            "submitted_step": submitted_step,
        }
        self._llm_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="v5_llm_strategy"
        )
        self._llm_future = self._llm_executor.submit(
            call_llm,
            system_prompt,
            payload,
            self.model_key,
            float(self._get_config("LLM_TIMEOUT_SECONDS", 15.0)),
            response_schema=PLAN_RESPONSE_SCHEMA,
        )
        if self.current_plan is None:
            state["strategy_source"] = "submitted"
        state["log"].append({
            "step": submitted_step,
            "action": "llm_strategy_submitted",
            "status": "pending",
            "situation_type": situation,
        })
        return state

    def _collect_llm_result(self, state: AgentState):
        """Collect a completed strategy request without ever blocking."""
        future = self._llm_future
        if future is None or not future.done():
            return

        context = self._llm_future_context or {}
        situation = context.get("situation_type")
        event_key = context.get("event_key")
        submitted_step = context.get("submitted_step")
        try:
            data, metrics = future.result()
        except Exception as exc:  # protects the routing cycle from worker errors
            data = None
            metrics = {
                "provider": self.model_config["provider"],
                "model": self.model_config["model"],
                "status": "error",
                "error": str(exc),
            }

        metrics["step"] = submitted_step
        metrics["completed_step"] = self.ctrl.step_count
        metrics["situation_type"] = situation
        metrics["event_key"] = event_key
        self.llm_call_metrics.append(metrics)

        plan = self._parse_plan(data)
        if plan is not None:
            plan["source_event_key"] = event_key
            plan["selected_step"] = self.ctrl.step_count
            self.current_plan = plan
            self.plan_update_count += 1
            state["fresh_plan"] = True
            status = "accepted"
            if verbosity_level >= 1:
                print(
                    f"[agent-v5][llm] new strategy={plan['name']} "
                    f"reason={plan['reason'] or 'not_provided'}"
                )
        else:
            status = "rejected"
            # A failed response did not solve this event. Permit a retry the
            # next time the same material event reaches the LLM node.
            self._last_submitted_event_key = None
            if verbosity_level >= 1:
                print("[agent-v5][llm] response rejected; no new plan installed")

        state["log"].append({
            "step": self.ctrl.step_count,
            "action": "llm_strategy_completed",
            "status": status,
            "situation_type": situation,
        })
        self._llm_future = None
        self._llm_future_context = None
        # The request is complete, so its one-use worker can be joined without
        # blocking. Releasing it here avoids idle threads at program shutdown.
        executor = self._llm_executor
        self._llm_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _route_with_strategy(self, state: AgentState):
        plan = state.get("strategy")
        active = self._active_demands(state)
        complex_flows = {
            flow for flow, demand in active
            if not self._exact_path_valid(
                flow, [flow[0], flow[1]], demand, state["topology"]
            )
        }

        # A pending/invalid LLM response never selects a substitute strategy.
        # Only independently routine direct paths remain operational meanwhile.
        if not plan:
            routes = {}
            reservations = {}
            for flow, demand in self._ordered_demands(active, "smallest_first"):
                direct = [flow[0], flow[1]]
                if flow not in complex_flows and self._has_capacity(
                    state["topology"], direct, demand, reservations
                ):
                    routes[flow] = direct
                    self._reserve(direct, demand, reservations)
            state["routes"] = routes
            state["routing_diagnostics"] = {
                "fast_path_flows": len(routes),
                "complex_flows": len(complex_flows),
                "llm_planned_flows": 0,
                "llm_recovered_flows": 0,
                "llm_pending_flows": (
                    len(complex_flows) if self._llm_future is not None else 0
                ),
                "unresolved_complex_flows": len(complex_flows),
                "allocator": "routine_direct_only_while_llm_pending",
                "exact_latency_sla": self.sla_resolver is not None,
            }
            return state

        routes, allocation = self._allocate_all_flows(state, plan["name"])
        for (src, dst), demand in active:
            path = routes.get((src, dst), [])
            state["log"].append({
                "step": self.ctrl.step_count,
                "flow": f"{src}->{dst}",
                "action": f"llm_selected_{plan['name']}",
                "status": "proposed" if path else "no_feasible_path",
                "path": path,
            })

        recovered_complex = sum(1 for flow in complex_flows if flow in routes)
        state["routes"] = routes
        state["routing_diagnostics"] = {
            "fast_path_flows": sum(
                1 for flow in routes
                if flow not in complex_flows and routes[flow] == [flow[0], flow[1]]
            ),
            "complex_flows": len(complex_flows),
            "llm_planned_flows": len(active),
            "llm_recovered_flows": recovered_complex,
            "llm_pending_flows": 0,
            "unresolved_complex_flows": len(complex_flows) - recovered_complex,
            "candidate_flows": allocation["candidate_flows"],
            "no_candidate_flows": allocation["no_candidate_flows"],
            "allocator": "joint_greedy",
            "exact_latency_sla": self.sla_resolver is not None,
        }
        return state

    # ------------------------------------------------------------------
    # Validation and metrics
    # ------------------------------------------------------------------

    def _validate(self, state: AgentState):
        valid_routes = {}
        results = {}
        reservations: dict[tuple[str, str], float] = {}
        demand_map = dict(self._active_demands(state))

        for flow, path in state.get("routes", {}).items():
            demand = demand_map.get(flow, 0.0)
            ok = (
                demand > 0
                and self._exact_path_valid(
                    flow, path, demand, state["topology"]
                )
                and self._has_capacity(state["topology"], path, demand, reservations)
            )
            results[flow] = ok
            if ok:
                valid_routes[flow] = path
                self._reserve(path, demand, reservations)
            else:
                self.invalid_action_count += 1

        state["routes"] = valid_routes
        state["validation_results"] = results
        return state

    def _record(self, state: AgentState):
        elapsed = (time.perf_counter() - state["start"]) * 1000
        active_count = len(self._active_demands(state))
        routed_count = len(state.get("routes", {}))
        strategy = state.get("strategy", {}).get("name")
        if state["situation_type"] == "trivial":
            strategy = "direct_fast_path"

        diagnostics = state.get("routing_diagnostics", {})
        flow_metrics = {
            "step": self.ctrl.step_count,
            "situation_type": state["situation_type"],
            "strategy": strategy,
            "strategy_source": state.get("strategy_source", "none"),
            "event_key": state.get("event_key", ""),
            "active_flows": active_count,
            "routed_flows": routed_count,
            "fast_path_flows": diagnostics.get("fast_path_flows", 0),
            "complex_flows": diagnostics.get("complex_flows", 0),
            "llm_planned_flows": diagnostics.get("llm_planned_flows", 0),
            "llm_recovered_flows": diagnostics.get("llm_recovered_flows", 0),
            "related_direct_flows": diagnostics.get("related_direct_flows", 0),
            "related_rerouted_flows": diagnostics.get("related_rerouted_flows", 0),
            "llm_pending_flows": diagnostics.get("llm_pending_flows", 0),
            "unresolved_complex_flows": diagnostics.get(
                "unresolved_complex_flows", 0
            ),
            "fast_path_ratio": (
                diagnostics.get("fast_path_flows", 0) / active_count
                if active_count else 0.0
            ),
            "llm_pending": self._llm_future is not None,
        }
        self.flow_decision_metrics.append(flow_metrics)

        self.model_metrics.append({
            "step": self.ctrl.step_count,
            "model_key": self.model_key,
            "model": self.model_config["model"],
            "provider": self.model_config["provider"],
            "time_ms": elapsed,
            "situation_type": state["situation_type"],
            "strategy": strategy,
            "active": active_count,
            "routed": routed_count,
            "rejected": active_count - routed_count,
            "invalid_total": self.invalid_action_count,
            "llm_pending": self._llm_future is not None,
        })
        if verbosity_level >= 1:
            print(
                f"[agent-v5] situation={state['situation_type']} "
                f"strategy={strategy} "
                f"source={state.get('strategy_source', 'none')} "
                f"active={active_count} routed={routed_count}"
            )
        return state

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _calculate_pressure(self, state: AgentState, active):
        """Estimate prospective shared-link load from one preferred path per flow."""
        loads = {}
        preferred_flows = 0
        for flow, demand in active:
            current = self._current_path(
                state["flow_table"], flow[0], flow[1]
            )
            direct = [flow[0], flow[1]]
            if current and self._exact_path_valid(
                flow, current, demand, state["topology"]
            ):
                path = current
            elif self._exact_path_valid(
                flow, direct, demand, state["topology"]
            ):
                path = direct
            else:
                candidates = self._candidate_paths(
                    state["topology"], flow[0], flow[1], demand, 1
                )
                path = candidates[0] if candidates else []
            if not path:
                continue
            preferred_flows += 1
            for u, v in zip(path, path[1:]):
                edge = self._edge_key(u, v)
                loads[edge] = loads.get(edge, 0.0) + demand

        links = []
        for edge, load in loads.items():
            link = self._link(state["topology"], edge[0], edge[1])
            if link is None:
                continue
            pressure = load / max(link.capacity_mbps, 1e-9)
            links.append({
                "edge": self._edge_label(edge[0], edge[1]),
                "pressure": round(pressure, 3),
                "load_mbps": round(load, 2),
                "capacity_mbps": round(link.capacity_mbps, 2),
            })
        links.sort(key=lambda item: (-item["pressure"], item["edge"]))
        max_pressure = links[0]["pressure"] if links else 0.0
        return {
            "max_pressure": max_pressure,
            "overloaded_links": sum(item["pressure"] > 1.0 for item in links),
            "preferred_path_flows": preferred_flows,
            "no_preferred_path_flows": len(active) - preferred_flows,
            "top_links": links[:3],
        }

    def _compact_case(self, state: AgentState):
        """Return bounded affected/candidate evidence for the LLM prompt."""
        direct_demands = []
        affected = []
        for flow, demand in self._active_demands(state):
            direct = [flow[0], flow[1]]
            if self._exact_path_valid(
                flow, direct, demand, state["topology"]
            ):
                direct_demands.append((flow, demand))
            else:
                affected.append((flow, demand))

        top_k = int(self._get_config("LLM_TOP_K_AFFECTED_FLOWS", 3))
        candidate_count = max(
            3, int(self._get_config("LLM_CANDIDATES_PER_FLOW", 3))
        )
        selected = sorted(affected, key=lambda item: (-item[1], item[0]))[
            :max(1, top_k)
        ]
        selected_candidates = {}
        affected_payload = []
        for flow, demand in selected:
            paths = self._candidate_paths(
                state["topology"], flow[0], flow[1], demand, candidate_count
            )
            selected_candidates[flow] = paths
            affected_payload.append(
                self._flow_candidate_summary(state["topology"], flow, demand, paths)
            )

        candidate_edges = {
            edge
            for paths in selected_candidates.values()
            for path in paths
            for edge in zip(path, path[1:])
        }
        related_limit = int(self._get_config("LLM_TOP_K_RELATED_FLOWS", 3))
        related = [item for item in direct_demands if item[0] in candidate_edges]
        related.sort(key=lambda item: (-item[1], item[0]))
        related_payload = []
        for flow, demand in related[:max(0, related_limit)]:
            paths = self._candidate_paths(
                state["topology"], flow[0], flow[1], demand, candidate_count
            )
            related_payload.append(
                self._flow_candidate_summary(state["topology"], flow, demand, paths)
            )

        return {
            "affected_total": len(affected),
            "affected_top": affected_payload,
            "related_direct_top": related_payload,
        }

    def _flow_candidate_summary(self, topology, flow, demand, paths):
        candidates = []
        for index, path in enumerate(paths):
            metrics = self._path_metrics(topology, path)
            candidates.append({
                "id": f"c{index}",
                "p": "-".join(path),
                "h": len(path) - 1,
                "lat": round(metrics["latency"], 2),
                "free": round(metrics["free_capacity"], 2),
            })
        return {
            "id": f"{flow[0]}>{flow[1]}",
            "d": round(demand, 2),
            "max_lat": self._sla_for_payload(flow),
            "c": candidates,
        }

    def _candidate_paths(self, topology, src, dst, demand, limit):
        """Generate a small, diverse portfolio and remove latency-SLA violations."""
        flow = (src, dst)
        limit = max(1, int(limit))
        modes = ("latency", "capacity", "utilization", "balanced")
        paths = []
        seen = set()
        graph = nx.Graph()
        graph.add_nodes_from(topology.nodes)
        for link_id, link in topology.links.items():
            if not link.up or demand > link.capacity_mbps:
                continue
            capacity = max(link.capacity_mbps, 1e-9)
            utilization = link.util / capacity
            free_ratio = max(capacity - link.util, 1e-9) / capacity
            graph.add_edge(
                link_id.u,
                link_id.v,
                latency=max(link.latency_ms, 1e-9),
                capacity=max(1.0 / free_ratio + 0.02 * link.latency_ms, 1e-9),
                utilization=max(
                    utilization * 10.0 + 0.05 * link.latency_ms, 1e-9
                ),
                balanced=max(
                    link.latency_ms + link.loss * 100.0 + utilization * 8.0,
                    1e-9,
                ),
            )

        for mode in modes:
            try:
                path = nx.shortest_path(graph, src, dst, weight=mode)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            candidate = tuple(path)
            if candidate in seen:
                continue
            if not self._exact_path_valid(flow, path, demand, topology):
                continue
            seen.add(candidate)
            paths.append(list(path))
            if len(paths) >= limit:
                break
        return paths

    def _allocate_all_flows(self, state, strategy_name):
        """Greedily select one SLA-valid candidate per flow with one ledger."""
        active = self._active_demands(state)
        candidate_count = max(
            3, int(self._get_config("LLM_CANDIDATES_PER_FLOW", 3))
        )
        portfolios = {
            flow: self._candidate_paths(
                state["topology"], flow[0], flow[1], demand, candidate_count
            )
            for flow, demand in active
        }
        for flow, demand in active:
            current = self._current_path(
                state["flow_table"], flow[0], flow[1]
            )
            if (
                current
                and current not in portfolios[flow]
                and self._exact_path_valid(
                    flow, current, demand, state["topology"]
                )
            ):
                portfolios[flow].append(current)
        ordered = sorted(
            active,
            key=lambda item: self._allocation_flow_key(
                item, portfolios, strategy_name, state
            ),
        )
        routes = {}
        reservations = {}
        for flow, demand in ordered:
            candidates = sorted(
                portfolios[flow],
                key=lambda path: self._allocation_candidate_key(
                    path, flow, demand, strategy_name, reservations, state
                ),
            )
            for path in candidates:
                if self._has_capacity(
                    state["topology"], path, demand, reservations
                ):
                    routes[flow] = path
                    self._reserve(path, demand, reservations)
                    break

        candidate_flows = sum(bool(paths) for paths in portfolios.values())
        return routes, {
            "candidate_flows": candidate_flows,
            "no_candidate_flows": len(active) - candidate_flows,
            "unallocated_flows": len(active) - len(routes),
        }

    def _allocation_flow_key(self, item, portfolios, strategy_name, state):
        flow, demand = item
        scarcity = len(portfolios[flow])
        if strategy_name == "minimize_delay":
            best_latency = min(
                (self._path_metrics(state["topology"], path)["latency"]
                 for path in portfolios[flow]),
                default=float("inf"),
            )
            sla_margin = self._resolve_flow_sla(flow) - best_latency
            return (sla_margin, scarcity, demand, flow)
        if strategy_name == "stable_recovery":
            has_current = self._current_path(
                state["flow_table"], flow[0], flow[1]
            ) is not None
            return (not has_current, scarcity, demand, flow)
        if strategy_name == "protect_capacity":
            return (scarcity, demand, flow)
        # Smaller demands and scarce alternatives generally admit more flows.
        return (scarcity, demand, flow)

    def _allocation_candidate_key(
        self, path, flow, demand, strategy_name, reservations, state
    ):
        metrics = self._path_metrics(state["topology"], path)
        projected = self._max_projected_utilization(
            state["topology"], path, demand, reservations
        )
        current = self._current_path(
            state["flow_table"], flow[0], flow[1]
        )
        changed = int(bool(current) and current != path)
        if strategy_name == "minimize_delay":
            return (metrics["latency"], projected, len(path), changed)
        if strategy_name == "stable_recovery":
            return (changed, projected, metrics["latency"], len(path))
        if strategy_name == "protect_capacity":
            capacity_cost = self._path_capacity_cost(
                state["topology"], path, demand
            )
            return (projected, capacity_cost, metrics["latency"], changed)
        capacity_cost = self._path_capacity_cost(
            state["topology"], path, demand
        )
        return (capacity_cost, len(path), projected, metrics["latency"], changed)

    def _path_capacity_cost(self, topology, path, demand):
        """Approximate how much shared capacity a path consumes."""
        cost = 0.0
        for u, v in zip(path, path[1:]):
            link = self._link(topology, u, v)
            if link is None:
                return float("inf")
            cost += demand / max(link.capacity_mbps, 1e-9)
        return cost

    def _path_metrics(self, topology, path):
        latency = 0.0
        free_capacity = float("inf")
        for u, v in zip(path, path[1:]):
            link = self._link(topology, u, v)
            if link is None:
                return {"latency": float("inf"), "free_capacity": 0.0}
            latency += link.latency_ms
            free_capacity = min(
                free_capacity, max(0.0, link.capacity_mbps - link.util)
            )
        if free_capacity == float("inf"):
            free_capacity = 0.0
        return {"latency": latency, "free_capacity": free_capacity}

    @staticmethod
    def _ordered_demands(active, flow_order):
        reverse = flow_order == "largest_first"
        return sorted(active, key=lambda item: (item[1], item[0]), reverse=reverse)

    @staticmethod
    def _reserve(path, demand, reservations):
        for u, v in zip(path, path[1:]):
            edge = RoutingAgent._edge_key(u, v)
            reservations[edge] = reservations.get(edge, 0.0) + demand

    def _has_capacity(self, topology, path, demand, reservations):
        for u, v in zip(path, path[1:]):
            link = self._link(topology, u, v)
            if link is None or not link.up:
                return False
            edge = self._edge_key(u, v)
            if reservations.get(edge, 0.0) + demand > link.capacity_mbps:
                return False
        return True

    def _max_projected_utilization(
        self, topology, path, demand, reservations
    ):
        projected = 0.0
        for u, v in zip(path, path[1:]):
            link = self._link(topology, u, v)
            if link is None:
                return float("inf")
            reserved = reservations.get(self._edge_key(u, v), 0.0)
            projected = max(
                projected,
                (reserved + demand) / max(link.capacity_mbps, 1e-9),
            )
        return projected

    def _exact_path_valid(self, flow, path, demand, topology):
        return bool(
            path
            and self._valid(flow[0], flow[1], path, demand)
            and self._has_capacity(topology, path, demand, {})
            and self._path_within_latency_sla(flow, path, topology)
        )

    def _path_within_latency_sla(self, flow, path, topology):
        limit = self._resolve_flow_sla(flow)
        latency = self._path_metrics(topology, path)["latency"]
        return latency <= limit + 1e-9

    def _resolve_flow_sla(self, flow):
        if self.sla_resolver is None:
            return float("inf")
        try:
            value = self.sla_resolver(flow[0], flow[1])
            return float(value) if value is not None else float("inf")
        except Exception:
            return float("inf")

    def _sla_for_payload(self, flow):
        value = self._resolve_flow_sla(flow)
        return None if value == float("inf") else round(value, 2)

    @staticmethod
    def _edge_key(u, v):
        return tuple(sorted((u, v), key=str))

    @staticmethod
    def _neighbors(topology):
        neighbors = {node: [] for node in topology.nodes}
        for link_id, link in topology.links.items():
            if link.up:
                neighbors[link_id.u].append((link_id.v, link))
                neighbors[link_id.v].append((link_id.u, link))
        return neighbors

    @staticmethod
    def _link(topology, u, v):
        for link_id, link in topology.links.items():
            if {link_id.u, link_id.v} == {u, v}:
                return link
        return None

    def _network_summary(self, state: AgentState):
        topology = state["topology"]
        active = self._active_demands(state)
        utilizations = []
        failed = 0
        for link in topology.links.values():
            if not link.up:
                failed += 1
            else:
                utilizations.append(link.util / max(link.capacity_mbps, 1e-9))

        return {
            **state["situation_summary"],
            "failed_links": failed,
            "total_links": len(topology.links),
            "total_demand_mbps": round(sum(d for _, d in active), 3),
            "max_demand_mbps": round(max((d for _, d in active), default=0.0), 3),
            "avg_link_utilization": round(
                sum(utilizations) / len(utilizations), 4
            ) if utilizations else 0.0,
            "max_link_utilization": round(max(utilizations), 4) if utilizations else 0.0,
            "pressure_top_links": state.get("pressure_summary", {}).get(
                "top_links", []
            ),
        }

    def _event_key(self, state: AgentState):
        """Build an event signature used as the non-LLM triage trigger."""
        active = self._active_demands(state)
        affected_demand = 0.0
        total_demand = sum(demand for _, demand in active)
        affected_count = 0
        for flow, demand in active:
            direct = [flow[0], flow[1]]
            if not self._exact_path_valid(
                flow, direct, demand, state["topology"]
            ):
                affected_demand += demand
                affected_count += 1

        failed_links = sorted(
            self._edge_label(link_id.u, link_id.v)
            for link_id, link in state["topology"].links.items()
            if not link.up
        )
        utilizations = [
            link.util / max(link.capacity_mbps, 1e-9)
            for link in state["topology"].links.values()
            if link.up
        ]
        max_utilization = max(utilizations, default=0.0)
        affected_ratio = affected_demand / total_demand if total_demand else 0.0
        pressure_summary = state.get("pressure_summary", {})
        pressure_edges = ",".join(
            item["edge"] for item in pressure_summary.get("top_links", [])[:2]
        ) or "none"

        return "|".join([
            state["situation_type"],
            f"active={self._count_bucket(len(active))}",
            f"affected={self._count_bucket(affected_count)}",
            f"demand={self._ratio_bucket(affected_ratio)}",
            f"failed={','.join(failed_links) if failed_links else 'none'}",
            f"util={self._utilization_bucket(max_utilization)}",
            f"pressure={self._pressure_bucket(pressure_summary.get('max_pressure', 0.0))}",
            f"pressure_edges={pressure_edges}",
        ])

    @staticmethod
    def _edge_label(u, v):
        a, b = sorted((str(u), str(v)))
        return f"{a}-{b}"

    @staticmethod
    def _count_bucket(value):
        if value == 0:
            return "0"
        if value <= 2:
            return "1-2"
        if value <= 5:
            return "3-5"
        if value <= 10:
            return "6-10"
        if value <= 20:
            return "11-20"
        return "21+"

    @staticmethod
    def _ratio_bucket(value):
        if value <= 0.10:
            return "0-10pct"
        if value <= 0.25:
            return "10-25pct"
        if value <= 0.50:
            return "25-50pct"
        return "50pct+"

    @staticmethod
    def _utilization_bucket(value):
        if value < 0.50:
            return "low"
        if value < 0.80:
            return "medium"
        if value < 1.0:
            return "high"
        return "saturated"

    @staticmethod
    def _pressure_bucket(value):
        if value <= 0.8:
            return "low"
        if value <= 1.0:
            return "near_capacity"
        if value <= 1.25:
            return "overloaded"
        if value <= 1.5:
            return "high"
        return "critical"

    @staticmethod
    def _parse_plan(data):
        if not isinstance(data, dict):
            return None
        name = data.get("strategy")
        if name not in STRATEGY_CONFIG:
            return None
        return {
            "name": name,
            "reason": str(data.get("reason", ""))[:200],
        }

    def _active_demands(self, state: AgentState):
        return [
            (flow, demand)
            for flow, demand in state["demands"].items()
            if demand > 0
        ]

    def _current_path(self, flow_table, src, dst):
        entry = flow_table.flows.get(FlowId(src=src, dst=dst))
        return list(entry.path) if entry and entry.path else None

    def _valid(self, src, dst, path, demand):
        return bool(
            path
            and self.ctrl.validate_path_logic(src, dst, path)
            and self.ctrl.validate_path_sla(path, demand)
        )

    @staticmethod
    def _normalize_demands(demands):
        normalized = {}
        for key, value in demands.items():
            if isinstance(key, FlowId):
                normalized[(key.src, key.dst)] = float(value)
            else:
                src, dst = key
                normalized[(src, dst)] = float(value)
        return normalized

    @staticmethod
    def _get_config(name, default):
        try:
            import config
            return getattr(config, name, default)
        except ImportError:
            return default
