# Agentic System for Zero-Touch Connectivity and Traffic Engineering

This project investigates the use of **agentic AI** and **Foundation Models (FMs)** for autonomous network management.

The objective is to develop an agentic network management system that can:

- observe network conditions;
- reason about routing and traffic engineering decisions;
- apply forwarding rules through a controller interface;
- continuously validate network behavior against operational objectives.

The project focuses on **zero-touch connectivity** and **traffic engineering**, where the system autonomously configures or reconfigures routing behavior in response to:

- changing traffic demands;
- random network failures;
- link capacity constraints;
- QoS and SLA requirements.

The current prototype is implemented in a simulation-based environment and compares a baseline heuristic routing approach against an agentic routing approach.

---

## Project Overview

The simulator models a dynamic network environment with:

- a network topology containing link attributes such as capacity, latency, loss, and failure probability;
- a stochastic time-varying traffic model;
- a controller API exposing topology, telemetry, and flow-programming operations;
- a routing agent that computes paths using an agentic decision-making loop;
- a simulation engine that evaluates routing performance over time.

The system evaluates routing decisions under dynamic conditions such as traffic changes and link failures.

---

## Key Features

- **Random network topology generation**
- **Bidirectional links with capacity, latency, loss, and failure probability**
- **Time-varying stochastic traffic demands**
- **Controller API for flow installation and link updates**
- **Baseline heuristic shortest-path routing**
- **Agentic routing interface**
- **Asynchronous routing execution**
- **Random link failure simulation**
- **Traffic acceptance and loss-rate measurement**
- **Automatic result plotting**
- **Per-demand history export to CSV**

---

## System Architecture

The project is organized around the following core components:

| Component | Description |
|---|---|
| `NetworkGraph` | Represents the network topology, nodes, links, link states, and link attributes. |
| `TrafficModel` | Generates stochastic time-varying traffic demands. |
| `ControllerAPI` | Exposes topology snapshots, traffic snapshots, flow tables, and flow-programming methods. |
| `RoutingAgent` | Agentic routing module responsible for computing paths. |
| `Simulator` | Runs the simulation loop, induces failures, applies routing decisions, and computes metrics. |
| `config.py` | Stores simulation parameters, algorithm names, metric names, and delay settings. |
| `helpers.py` | Provides helper functions for network and traffic generation. |
| `agents.py` | Contains the routing agent implementation. (To be implemented) |
| `main.py` | Main entry point for running the experiment. |

---

## Repository Structure

```text
.
├── agents.py
├── config.py
├── helpers.py
├── main.py
├── simulation.py
├── results/
│   ├── network.png
│   ├── network.txt
│   ├── results.png
│   ├── all_history_agentic.csv
│   └── all_history_<algorithm>.csv
└── README.md
```

> Note: The `results/` directory is generated automatically when the simulation is executed.

---

## Requirements

Recommended Python version:

```text
Python 3.10+
```

Main Python packages:

```text
networkx
matplotlib
numpy
pandas
tqdm
```

Optional packages may be required depending on the implementation of `RoutingAgent` in `agents.py`.

---

## Running the Simulation

Run the main simulation script:

```bash
python main.py
```

The simulation will:

1. Initialize random seeds for reproducibility;
2. Create a random network topology;
3. Save the topology visualization to `results/network.png`;
4. Save topology details to `results/network.txt`;
5. Generate time-varying traffic demands;
6. Run selected routing algorithms;
7. Compare performance metrics over time;
8. Save plots and CSV outputs in the `results/` directory.

---

## Routing Algorithms

The simulator supports Heuristic and Agentic routing approaches. Both of them run asynchronously, allowing the simulator to model delayed decision-making.

### Heuristic Routing

The baseline heuristic computes shortest paths using link weights.

It reuses installed flows when possible and computes new paths when needed.

### Agentic Routing

The agentic approach delegates routing decisions to `RoutingAgent`.

The agent can observe the controller state and reason over:

- topology snapshots;
- traffic demands;
- installed flow tables;
- link states;
- capacity constraints;
- failure conditions.

---

## Network Model

The network is represented as a bidirectional graph.

Each link includes the following attributes:

| Attribute | Description |
|---|---|
| `capacity` | Link capacity in Mbps. |
| `latency` | Link latency in milliseconds. |
| `loss` | Link loss probability or loss score. |
| `fail_p` | Probability that the link fails at a simulation step. |
| `up` | Boolean link-state indicator. |
| `weight` | Routing weight used by shortest-path algorithms. |
| `util` | Current utilized capacity on the link. |

---

## Traffic Model

Traffic demands are stochastic and time-varying.

Each demand has:

- a source node;
- a destination node;
- a base demand;
- a start time.

After the configured start time, demand varies according to:

- a base demand;
- a seasonal sinusoidal component;
- random noise.

This allows the simulator to test routing behavior under dynamic load conditions.

---

## Controller API

The `ControllerAPI` exposes methods for interacting with the network state.

Important methods include:

| Method | Description |
|---|---|
| `get_topology_snapshot()` | Returns the current topology state. |
| `get_traffic_snapshot()` | Returns current demand information. |
| `get_flow_table_snapshot()` | Returns installed flow entries. |
| `set_flow()` | Installs or replaces a path for a source-destination pair. |
| `set_link_weight()` | Updates a link weight. |
| `reset_link_state()` | Changes the up/down state of a link. |
| `validate_path_logic()` | Checks path connectivity and loop freedom. |
| `validate_path_sla()` | Checks basic demand-capacity feasibility. |

---

## Reproducibility

To make experiments reproducible, set `random_seed` in `config.py`.

Example:

```python
random_seed = 42
```

When the seed is set, the following components become reproducible:

- random network generation;
- traffic pattern generation;
- random failure events;
- plotted topology layout.

---

## Credits and Ownership

- **Prof. Tarik Taleb** — Chair Professor  
- **Hamidreza Mazandarani** — Doctoral Researcher

Institute of Networked Energy-Efficient Systems

Faculty of Electrical Engineering & Information Technology

Ruhr University Bochum

---

## Disclaimer

This project is provided for research and educational purposes only.

The code, documentation, simulations, and related materials are provided **"as is"**, without warranty of any kind, express or implied, including but not limited to warranties of correctness, reliability, fitness for a particular purpose, or non-infringement.

The authors, contributors, project owners, and the affiliated institution are not responsible for any errors, omissions, damages, losses, or other consequences arising from the use, modification, execution, or distribution of this project.

Users are responsible for reviewing, testing, validating, and adapting the code before using it in any environment, especially production, operational, or safety-critical network systems.

