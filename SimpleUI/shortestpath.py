import numpy as np
import math
import time
from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler

# Backend logic for shortest path CQM
def shortest_path_cqm(adj_matrix, start_idx, end_idx):
    n = len(adj_matrix)
    x = {}
    for u in range(n):
        for v in range(n):
            if u != v and not np.isinf(adj_matrix[u][v]):
                x[(u,v)] = Binary(f"x_{u}_{v}")

    cqm = ConstrainedQuadraticModel()

    # Objective
    objective = sum(adj_matrix[u][v]*x[(u,v)] for (u,v) in x)
    cqm.set_objective(objective)

    # Constraints
    start_constraint = sum(x[(start_idx,v)] for v in range(n) if (start_idx,v) in x)
    cqm.add_constraint(start_constraint == 1, label="start_outgoing")

    end_constraint = sum(x[(u,end_idx)] for u in range(n) if (u,end_idx) in x)
    cqm.add_constraint(end_constraint == 1, label="end_incoming")

    # Flow constraints
    for m in range(n):
        if m == start_idx or m == end_idx:
            continue
        in_flow = sum(x[(u,m)] for u in range(n) if (u,m) in x)
        out_flow = sum(x[(m,v)] for v in range(n) if (m,v) in x)
        cqm.add_constraint(in_flow - out_flow == 0, label=f"flow_balance_{m}")

    return cqm

def format_solution(solution_sample, adj_matrix, node_labels, start_idx, end_idx):
    chosen_edges = [var for var, val in solution_sample.items() if val == 1]

    chosen_pairs = []
    for edge_var in chosen_edges:
        _, u_str, v_str = edge_var.split('_')
        u, v = int(u_str), int(v_str)
        chosen_pairs.append((u, v))

    path = [start_idx]
    current_node = start_idx
    total_cost = 0.0
    while current_node != end_idx:
        for (u, v) in chosen_pairs:
            if u == current_node:
                path.append(v)
                total_cost += adj_matrix[u][v]
                current_node = v
                break

    path_labels = [node_labels[n] for n in path]
    return path, path_labels, total_cost

def nash_bargaining_shortest_distance_path(adj_matrix, node_labels, start_node, end_node):
    n = len(adj_matrix)
    dist = np.copy(adj_matrix)
    next_node = [[j if adj_matrix[i][j] != float('inf') else None for j in range(n)] for i in range(n)]

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    def get_path(i, j):
        if next_node[i][j] is None:
            return []
        path = [node_labels[i]]
        while i != j:
            i = next_node[i][j]
            path.append(node_labels[i])
        return path

    s_idx = node_labels.index(start_node)
    e_idx = node_labels.index(end_node)
    path = get_path(s_idx, e_idx)
    cost = dist[s_idx][e_idx]
    return path, cost

def solve_cqm(adj_matrix, start_idx, end_idx, node_labels, positions):
    start_time = time.time()
    cqm = shortest_path_cqm(adj_matrix, start_idx, end_idx)
    sampler = LeapHybridCQMSampler()
    results = sampler.sample_cqm(cqm, label="Shortest Path CQM")
    end_time = time.time()
    cqm_time = end_time - start_time

    cqm_path_labels = []
    cqm_cost = float('inf')
    cqm_path_edges = []
    feasible = results.filter(lambda d: d.is_feasible)
    if feasible:
        best = feasible.first
        path_idx, path_labels, total_cost = format_solution(best.sample, adj_matrix, node_labels, start_idx, end_idx)
        cqm_path_labels = path_labels
        cqm_cost = total_cost
        # Construct edges for plotting
        for i in range(len(path_idx)-1):
            p1 = positions[path_idx[i]]
            p2 = positions[path_idx[i+1]]
            cqm_path_edges.append([p1, p2])

    return cqm_path_labels, cqm_cost, cqm_time, cqm_path_edges

def solve_nash(adj_matrix, node_labels, positions):
    start_time_nb = time.time()
    nb_path_labels, nb_cost = nash_bargaining_shortest_distance_path(np.array(adj_matrix), node_labels, 'S', 'END')
    end_time_nb = time.time()
    nb_time = end_time_nb - start_time_nb

    nb_path_edges = []
    if nb_cost != float('inf') and len(nb_path_labels) > 1:
        nb_path_idx = [node_labels.index(lbl) for lbl in nb_path_labels]
        for i in range(len(nb_path_idx)-1):
            p1 = positions[nb_path_idx[i]]
            p2 = positions[nb_path_idx[i+1]]
            nb_path_edges.append([p1, p2])

    return nb_path_labels, nb_cost, nb_time, nb_path_edges
