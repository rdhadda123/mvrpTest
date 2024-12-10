import numpy as np
import time

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
