import numpy as np
from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler
import time

def shortest_path_cqm(adj_matrix, start_idx, end_idx):
    n = len(adj_matrix)
    # Create binary variables for each feasible edge
    x = {}
    for u in range(n):
        for v in range(n):
            if u != v and not np.isinf(adj_matrix[u][v]):
                x[(u,v)] = Binary(f"x_{u}_{v}")

    cqm = ConstrainedQuadraticModel()

    # Objective: minimize sum of w_uv * x_uv
    objective = sum(adj_matrix[u][v]*x[(u,v)] for (u,v) in x)
    cqm.set_objective(objective)

    # Constraint: For start node, sum of outgoing edges = 1
    start_constraint = sum(x[(start_idx,v)] for v in range(n) if (start_idx,v) in x)
    cqm.add_constraint(start_constraint == 1, label="start_outgoing")

    # Constraint: For end node, sum of incoming edges = 1
    end_constraint = sum(x[(u,end_idx)] for u in range(n) if (u,end_idx) in x)
    cqm.add_constraint(end_constraint == 1, label="end_incoming")

    # Constraints for intermediate nodes: inflow = outflow
    for m in range(n):
        if m == start_idx or m == end_idx:
            continue
        in_flow = sum(x[(u,m)] for u in range(n) if (u,m) in x)
        out_flow = sum(x[(m,v)] for v in range(n) if (m,v) in x)
        cqm.add_constraint(in_flow - out_flow == 0, label=f"flow_balance_{m}")

    return cqm

def format_solution(solution_sample, adj_matrix, node_labels, start_idx, end_idx):
    """
    Given a solution sample (dictionary of variable assignments), this function:
    - Extracts the chosen edges (x_u_v variables).
    - Reconstructs the path from start to end.
    - Calculates the total cost.
    - Returns the path and cost in the desired format.
    """
    # Extract chosen edges
    chosen_edges = [var for var, val in solution_sample.items() if val == 1]

    # Parse chosen edges to get node pairs
    chosen_pairs = []
    for edge_var in chosen_edges:
        # Edge variable format: "x_u_v"
        _, u_str, v_str = edge_var.split('_')
        u, v = int(u_str), int(v_str)
        chosen_pairs.append((u, v))

    # Reconstruct the path
    path = [start_idx]
    current_node = start_idx
    total_cost = 0.0

    # Follow edges until we reach the end node
    while current_node != end_idx:
        for (u, v) in chosen_pairs:
            if u == current_node:
                path.append(v)
                total_cost += adj_matrix[u][v]
                current_node = v
                break

    path_labels = [node_labels[n] for n in path]
    return path_labels, total_cost

# Sample input
node_labels = ['A','B','C','D','E','F','G','H','I']
adj_matrix = [
    [0.0,         45.5953456, 54.1159678, 45.9217503, 72.9903195, 73.1256183, 73.3661445, 47.2301732, float('inf')],
    [45.5953456,  0.0,        float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
    [54.1159678,  52.2532469, 0.0,          52.3438548, 52.3040750, 52.4901996, 52.5940848, 52.1978229, 52.1211143],
    [45.9217503,  float('inf'), float('inf'), 0.0,       float('inf'), float('inf'), float('inf'), float('inf'), float('inf')],
    [72.9903195,  72.9696378, 72.9710793, 72.9709118, 0.0,          72.9707843, 72.9715048, 72.9706511, 72.9703181],
    [73.1256183,  73.1070049, 73.1085500, 73.1069196, 73.1066836, 0.0,          73.1080728, 73.1062890, 73.1053911],
    [73.3661445,  73.3478294, 73.3508599, 73.3492495, 73.3488926, 73.3495461, 0.0,          73.3478790, 73.3473067],
    [47.2301732,  float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 0.0,         float('inf')],
    [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 1e6]
]

start_node = 'A'
end_node = 'I'
start_idx = node_labels.index(start_node)
end_idx = node_labels.index(end_node)

cqm = shortest_path_cqm(adj_matrix, start_idx, end_idx)

sampler = LeapHybridCQMSampler()
start = time.time()
results = sampler.sample_cqm(cqm, label="Shortest Path CQM")
end = time.time()
print(f"Execution Time: ", end - start, " seconds")

feasible = results.filter(lambda d: d.is_feasible)
if feasible:
    best = feasible.first

    path_labels, total_cost = format_solution(best.sample, adj_matrix, node_labels, start_idx, end_idx)
    print(f"Shortest Path: ({path_labels}, {total_cost})")
else:
    print("No feasible solution found.")
