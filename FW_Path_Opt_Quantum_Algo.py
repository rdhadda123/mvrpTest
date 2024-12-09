import numpy as np
import time
from dwave.system import LeapHybridSampler


def nash_bargaining_shortest_distance_path(adj_matrix, node_labels, start_node, end_node):
    n = len(adj_matrix)

    # Floyd-Warshall algorithm to find all shortest paths
    dist = np.copy(adj_matrix)
    next_node = [[j if adj_matrix[i][j] != float('inf') else None for j in range(n)] for i in range(n)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    # Function to reconstruct path
    def get_path(i, j):
        if next_node[i][j] is None:
            return []
        path = [node_labels[i]]
        while i != j:
            i = next_node[i][j]
            path.append(node_labels[i])
        return path

    # Initialize Nash bargaining solution
    nash_solution = {}

    # Iterate over all pairs and apply Nash Bargaining concept
    for u in range(n):
        for v in range(n):
            if u != v:
                path = get_path(u, v)
                cost = dist[u][v]
                nash_solution[(node_labels[u], node_labels[v])] = (path, cost)

    return nash_solution[(start_node, end_node)]

# Function to convert the shortest path problem to a QUBO for D-Wave
def dwave_shortest_path(adj_matrix, node_labels, start_node, end_node):
    n = len(adj_matrix)
    start_index = node_labels.index(start_node)
    end_index = node_labels.index(end_node)

    # Function to convert the shortest path problem to a QUBO for D-Wave
    def get_var(i, j):
        return f"x_{i}_{j}"

    # Initialize QUBO dictionary
    Q = {}

    # Objective: Minimize the total path cost
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] != float('inf'):
                var = get_var(i, j)
                Q[(var, var)] = adj_matrix[i][j]

    # Constraints
    # Path starts at start_node (Enforce the first step)
    for j in range(n):
        if j != start_index:
            var = get_var(start_index, j)
            Q[(var, var)] = Q.get((var, var), 0) + 1000  # Strong penalty for non-continuity

    # Path ends at end_node (Enforce the last step)
    for i in range(n):
        if i != end_index:
            var = get_var(i, end_index)
            Q[(var, var)] = Q.get((var, var), 0) + 1000  # Strong penalty for non-continuity

    # Ensure node continuity (Each intermediate node should have exactly one incoming and one outgoing path)
    for k in range(n):
        if k != start_index and k != end_index:
            for i in range(n):
                for j in range(n):
                    if adj_matrix[i][k] != float('inf') and adj_matrix[k][j] != float('inf'):
                        var_in = get_var(i, k)
                        var_out = get_var(k, j)
                        # Introduce penalty to encourage both the incoming and outgoing variables
                        Q[(var_in, var_in)] = Q.get((var_in, var_in), 0) + 2  # Encourage incoming path
                        Q[(var_out, var_out)] = Q.get((var_out, var_out), 0) + 2  # Encourage outgoing path
                        Q[(var_in, var_out)] = Q.get((var_in, var_out), 0) - 4  # Strong penalty for invalid transition

    print("QUBO Matrix:", Q)

    # Solve QUBO using D-Wave
    sampler = LeapHybridSampler()
    response = sampler.sample_qubo(Q)

    # Debug: Check solver response
    print("Solver Response:", response)

    # Extract the best solution
    sample = response.first.sample

    # Decode solution to reconstruct the path
    path = []
    total_cost = 0
    current_node = start_index

    while current_node != end_index:
        found_next = False
        for j in range(n):
            var = get_var(current_node, j)
            if sample.get(var, 0) == 1:
                path.append(node_labels[current_node])
                total_cost += adj_matrix[current_node][j]
                current_node = j
                found_next = True
                break
        if not found_next:
            raise ValueError("No valid path found in D-Wave results.")

    path.append(node_labels[end_index])
    return path, total_cost


node_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

adj_matrix = [
    [0.00000000e+000, 4.55953456e+001, 5.41159678e+001, 4.59217503e+001, 7.29903195e+001, 7.31256183e+001,
     7.33661445e+001, 4.72301732e+001, float('inf')],
    [4.55953456e+001, 0.00000000e+000, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
     float('inf'), float('inf')],
    [5.41159678e+001, 5.22532469e+001, 0.00000000e+000, 5.23438548e+001, 5.23040750e+001, 5.24901996e+001,
     5.25940848e+001, 5.21978229e+001, 5.21211143e+001],
    [4.59217503e+001, float('inf'), float('inf'), 0.00000000e+000, float('inf'), float('inf'), float('inf'),
     float('inf'), float('inf')],
    [7.29903195e+001, 7.29696378e+001, 7.29710793e+001, 7.29709118e+001, 0.00000000e+000, 7.29707843e+001,
     7.29715048e+001, 7.29706511e+001, 7.29703181e+001],
    [7.31256183e+001, 7.31070049e+001, 7.31085500e+001, 7.31069196e+001, 7.31066836e+001, 0.00000000e+000,
     7.31080728e+001, 7.31062890e+001, 7.31053911e+001],
    [7.33661445e+001, 7.33478294e+001, 7.33508599e+001, 7.33492495e+001, 7.33488926e+001, 7.33495461e+001,
     0.00000000e+000, 7.33478790e+001, 7.33473067e+001],
    [4.72301732e+001, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
     0.00000000e+000, float('inf')],
    [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
     1.00000000e+006]
]

start_node = 'A'
end_node = 'I'
start = time.time()
shortest_path = nash_bargaining_shortest_distance_path(adj_matrix, node_labels, start_node, end_node)
end = time.time()
print(f"Classical FW algo took", end - start, "seconds")
print("Shortest Path:", shortest_path)

# Run D-Wave optimization
start_time = time.time()
shortest_path, path_cost = dwave_shortest_path(adj_matrix, node_labels, start_node, end_node)
end_time = time.time()

print("D-Wave Optimization Results:")
print(f"Shortest Path: {shortest_path}")
print(f"Path Cost: {path_cost}")
print(f"Execution Time: {end_time - start_time:.4f} seconds")