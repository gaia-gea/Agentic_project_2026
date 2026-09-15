
# Agentic System for Zero-Touch Connectivity and Traffic Engineering

An agentic AI project that explores how Large Language Models (LLMs) can support autonomous traffic engineering in a dynamic network. Rather than asking an LLM to invent network paths, the system uses it as a **strategic supervisor**: it selects a routing objective while deterministic algorithms generate feasible paths, allocate shared capacity, and enforce SLA constraints.

In the evaluated simulation, the best agentic configuration delivered **44.4% more traffic than the best paired heuristic baseline**.

> This is a research and educational prototype. It is not intended for production network control without additional testing, security controls, and validation.

## Why an Agentic Approach?

Network routing involves strict, non-negotiable constraints such as link capacity, connectivity, link availability, and latency SLAs. Pure LLM routing would be risky: responses can be slow, stale, or structurally invalid. This project assigns each component the task it performs best:

| Component | Responsibility |
|---|---|
| **LLM agent** | Interprets a compact network event and selects a global routing objective. |
| **Deterministic tools** | Generate candidate paths, calculate latency, track capacity, and validate routes. |
| **Joint allocator** | Applies one shared capacity view to all active flows, avoiding conflicting decisions. |
| **Simulator/controller** | Models topology, traffic, failures, installed flows, and network outcomes. |

The LLM can select one of four objectives:

- `maximize_acceptance` — prioritize admitting more requests when capacity is scarce.
- `minimize_delay` — prioritize flows with strict latency requirements.
- `protect_capacity` — avoid highly loaded links and preserve headroom.
- `stable_recovery` — retain valid routes and reduce unnecessary rerouting after changes.

## Architecture

The final routing agent is implemented as a LangGraph workflow:

```text
START
  │
  ▼
Perceive ──► Classify ──► trivial ──► Fast Path ────────────┐
                         │                                  │
                         └► conflict/failure ► Plan with LLM ► Joint Allocator
                                                              │
                                                              ▼
                                                        Validate ──► Record ──► END
```

1. **Perceive** reads the current topology, flow table, link state, and any completed asynchronous LLM request.
2. **Classify** determines whether the current situation is trivial, a congestion or multi-path conflict, or a link failure. It also calculates prospective link pressure.
3. **Fast Path** safely assigns direct routes for routine cases without spending an LLM call.
4. **Plan with LLM** sends a small, aggregated event description and Top-*K* affected flows to the model. The required response is strict JSON with one strategy.
5. **Joint Allocator** generates deterministic candidate paths and allocates routes for all active flows using a shared bidirectional capacity ledger.
6. **Validate** rejects paths that violate connectivity, availability, capacity, or the exact latency SLA before they reach the controller.
7. **Record** exports network and agent telemetry for later comparison in the dashboard.

### Event-driven and asynchronous inference

LLM inference runs in a background worker, so the simulation loop is never blocked by a model request. The agent creates a compact `event_key` from the situation type, affected flow/demand bands, failures, utilization, and high-pressure links. It requests a new LLM plan only when that signature changes meaningfully; a valid active plan is reused while the network event remains similar.

This is intentionally not a hidden heuristic fallback. The deterministic fast path is an explicit LangGraph branch for routine direct routes. Complex flows are only processed with a valid LLM-selected objective.

## Results

Experiments used a 15-node stochastic network, 120 simulation steps, dynamic traffic, link failures, capacity constraints, and latency requirements between 4 and 8 ms.

| Model / baseline | Traffic acceptance ratio | Delivered traffic (aggregate Mbps) |
|---|---:|---:|
| GPT-5.4 Nano agentic | **38.56%** | **41,005.60** |
| Claude Sonnet 4.5 agentic | 38.50% | 40,941.42 |
| Claude Haiku 4.5 agentic | 38.50% | 40,937.10 |
| Gemini 2.5 Flash agentic | 37.16% | 39,514.98 |
| Best paired GPT heuristic | 26.71% | 28,403.43 |
| Local Llama 3.2 1B agentic | 19.79% | 21,045.21 |

The GPT-5.4 Nano agentic run achieved an **11.85 percentage-point** increase in accepted traffic over its best paired heuristic baseline, equivalent to a **44.4% relative improvement**. The remote models produced similar network results; deterministic path generation, shared-capacity allocation, and SLA validation remain essential contributors to the overall performance.

## Repository Structure

```text
.
├── Project_v2/
│   ├── agents_v5.py        # Final LangGraph routing agent
│   ├── llm_client.py       # Provider-specific structured LLM calls
│   ├── simulation.py       # Network, traffic, controller, and simulator
│   ├── helpers.py          # Random topology and traffic generation helpers
│   ├── config.py           # Simulation, agent, and model configuration
│   ├── main.py             # Experiment entry point
│   ├── results_export.py   # CSV/JSON telemetry exporter
│   ├── dashboard_2.py      # Streamlit comparison dashboard
│   └── plot_comparison.py  # Static result-comparison plotting utility
├── results/                # One timestamped folder per simulation run
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### Requirements

- Python 3.13+ (the project lockfile targets Python 3.13)
- - An LLM provider account/API key, or Ollama for a local model

Clone the repository and install dependencies:

```bash
git clone <your-repository-url>
cd Agentic_project
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, if you use `uv`:

```bash
uv sync
```

## Configure an LLM Provider

Create a `.env` file in the repository root or in `Project_v2/`. Never commit this file or place a real API key in `config.py`.

```dotenv
# Select only the variable needed by your active provider.
REQUESTY_API_KEY=your_requesty_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENROUTER_API_KEY=your_openrouter_key
```

Select the model in `Project_v2/config.py` by changing `ACTIVE_AGENT_MODEL`:

```python
ACTIVE_AGENT_MODEL = "requesty_gpt"
# ACTIVE_AGENT_MODEL = "requesty_claude"
# ACTIVE_AGENT_MODEL = "requesty_claude_haiku"
# ACTIVE_AGENT_MODEL = "requesty_gemini"
# ACTIVE_AGENT_MODEL = "local_llama"  # Requires Ollama running locally.
```

For local inference, install Ollama and download the configured model:

```bash
ollama pull llama3.2:1b
```

## Run a Simulation

From the repository root:

```bash
cd Project_v2
python main.py
```

The main configuration file controls the model, random seed, simulation duration, network parameters, traffic/SLA requirements, routing algorithms, Top-*K* prompt size, and pressure threshold.

Each execution writes its own timestamped and model-labelled directory under `results/`, for example:

```text
results/
└── 20260915_143000__requesty_gpt__openai-gpt-5.4-nano/
    ├── network.png
    ├── results.png
    ├── request_outcomes_by_algorithm.csv
    ├── network_metrics.csv
    ├── link_history.csv
    ├── flow_events.csv
    ├── model_metrics.csv
    ├── agent_decisions.csv
    └── run_metadata.json
```

This prevents new runs from overwriting earlier experiments.

## Launch the Dashboard

The Streamlit dashboard compares model runs, heuristic baselines, traffic acceptance, link history, flow outcomes, token usage, inference time, and strategy decisions.

```bash
cd Project_v2
streamlit run dashboard_2.py
```

By default, the dashboard reads the parent `results/` directory. To point it somewhere else, set `DASHBOARD_RESULTS_DIR` before launching Streamlit.

## Key Configuration Parameters

| Parameter | Location | Meaning |
|---|---|---|
| `ACTIVE_AGENT_MODEL` | `config.py` | Model/provider configuration used by the agent. |
| `num_steps` | `config.py` | Simulation horizon. |
| `random_seed` | `config.py` | Seed for reproducible topology and traffic generation. |
| `LLM_TOP_K_AFFECTED_FLOWS` | `config.py` | Number of high-impact flows summarized in an LLM event. |
| `V5_PRESSURE_THRESHOLD` | `config.py` | Prospective link-pressure threshold used for triage. |
| `algorithms` | `config.py` | Agentic and/or heuristic algorithms to execute. |
| `traffic_pattern_args` | `config.py` | Latency-SLA range for generated flow requests. |

## Limitations and Future Work

- The joint allocator is greedy and capacity-aware, not an exact global optimizer.
- Slower models may produce decisions based on an older network snapshot, despite asynchronous execution.
- The LLM selects a global objective; it does not directly manipulate individual links.
- A controlled ablation (fixed objective vs. LLM objective) and multiple random seeds would provide stronger evidence of the LLM's isolated contribution.
- Future work could add stale-response handling, adaptive Top-*K* selection, an exact allocator benchmark, and short-term reflection based on measurable action--outcome pairs.

## Credits

Developed as an academic project for the Institute of Networked Energy-Efficient Systems, Faculty of Electrical Engineering and Information Technology, Ruhr University Bochum.

Academic supervision: Prof. Tarik Taleb and Hamidreza Mazandarani.

