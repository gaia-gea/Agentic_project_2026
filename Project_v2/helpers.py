import random
from typing import Dict, Tuple, Any

Node = str
Link = Tuple[Node, Node]
DemandMap = Dict[Tuple[Node, Node], float]
DemandSpecMap = Dict[Tuple[Node, Node], Dict[str, Any]]


def create_random_network(g, seed,
                          num_nodes,
                          capacity_min, capacity_max,
                          latency_min, latency_max,
                          loss_min, loss_max,
                          fail_p_min, fail_p_max,
                          link_prob):
    """
    Creates a random connected (undirected) network of size {num_nodes}
    and adds random link attributes to the provided NetworkGraph {g}.
    """
    random.seed(seed)

    nodes = [chr(ord('A') + i) for i in range(num_nodes)]
    for n in nodes:
        g.nodes.add(n)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < link_prob:
                capacity = random.uniform(capacity_min, capacity_max) if capacity_max > capacity_min else capacity_min
                latency = random.uniform(latency_min, latency_max) if latency_max > latency_min else latency_min
                loss = random.uniform(loss_min, loss_max) if loss_max > loss_min else loss_min
                fail_p = random.uniform(fail_p_min, fail_p_max) if fail_p_max > fail_p_min else fail_p_min

                g.add_link(nodes[i], nodes[j], capacity, latency, loss, fail_p)

    return g


def create_random_traffic_pattern(
        g,
        base_demand: float,
        seed: int = 42,
        start_t_min: int = 0,
        start_t_max: int = 100,
        latency_requirement_min_ms: float = 5.0,
        latency_requirement_max_ms: float = 20.0,
) -> DemandSpecMap:
    """
    Creates traffic for each ordered node pair.
    Each pair gets:
    - random base demand factor
    - random start_time in [start_t_min, start_t_max]
    """
    random.seed(seed)

    nodes = sorted(list(g.nodes))
    demands: DemandSpecMap = {}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            factor = random.uniform(0.5, 1.5)
            base_val = base_demand * factor

            # one start time for both directions (or generate separately if you prefer)
            start_time = random.randint(start_t_min, start_t_max)
            if latency_requirement_max_ms > latency_requirement_min_ms:
                max_latency_ms = random.uniform(
                    latency_requirement_min_ms,
                    latency_requirement_max_ms,
                )
            else:
                max_latency_ms = latency_requirement_min_ms

            demands[(nodes[i], nodes[j])] = {
                "base": base_val,
                "start_time": start_time,
                "max_latency_ms": max_latency_ms,
            }
            demands[(nodes[j], nodes[i])] = {
                "base": base_val,
                "start_time": start_time,
                "max_latency_ms": max_latency_ms,
            }

    return demands
