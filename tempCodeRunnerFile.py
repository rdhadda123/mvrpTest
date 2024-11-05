import math
import random
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Real
import matplotlib.pyplot as plt

# Parameters
num_locations = 7
depot = (0, 0)

# Bounds
x_lower_bound = 10
x_upper_bound = 20
y_lower_bound = -10
y_upper_bound = 10

factor = 0.05
diagonal_length = math.sqrt((x_upper_bound - x_lower_bound)**2 + (y_upper_bound - y_lower_bound)**2)

# Randomly generate initial N locations (x, y) in 2D space
locations = [(random.randint(x_lower_bound, x_upper_bound), random.randint(y_lower_bound, y_upper_bound)) for _ in range(num_locations)]
print(f"Depot is located at: {depot}")
print(f"These are the current locations: {locations}")

# Using Hybrid CQM (Constrained Quadratic Model) Solver with cumulative approach
def solve_with_cumulative_cqm():
    locationsWithDepot = [depot]
    optimized_locations = []

    for j in range(num_locations):
        print(f"\nOptimizing location {j + 1}...")

        # Initialize the Constrained Quadratic Model
        cqm = ConstrainedQuadraticModel()

        # Variable for the current location (x, y coordinates)
        location_var = {
            'x': Real(f'x_{j}', lower_bound=x_lower_bound, upper_bound=x_upper_bound),
            'y': Real(f'y_{j}', lower_bound=y_lower_bound, upper_bound=y_upper_bound)
        }

        # Objective: Minimize the sum of Manhattan distances to all points in locationsWithDepot
        objective = 0
        for idx, ref_point in enumerate(locationsWithDepot):  # Use `idx` as an additional unique identifier
            x_i, y_i = location_var['x'], location_var['y']

            # Auxiliary variables for the absolute distances
            abs_x = Real(f'abs_x_{j}_{idx}', lower_bound=0)
            abs_y = Real(f'abs_y_{j}_{idx}', lower_bound=0)

            # Constraints to represent absolute value for Manhattan distance
            cqm.add_constraint(abs_x - (x_i - ref_point[0]) == factor*diagonal_length, label=f'abs_x_pos_{j}_{idx}')
            cqm.add_constraint(abs_x + (x_i - ref_point[0]) == factor*diagonal_length, label=f'abs_x_neg_{j}_{idx}')
            cqm.add_constraint(abs_y - (y_i - ref_point[1]) == factor*diagonal_length, label=f'abs_y_pos_{j}_{idx}')
            cqm.add_constraint(abs_y + (y_i - ref_point[1]) == factor*diagonal_length, label=f'abs_y_neg_{j}_{idx}')

            # Add to the objective
            objective += abs_x + abs_y

        # Set the objective in the CQM
        cqm.set_objective(objective)

        # Solve with the Leap Hybrid CQM sampler
        sampler = LeapHybridCQMSampler()
        result = sampler.sample_cqm(cqm)

        # Get the best solution
        best_solution = result.first.sample
        optimized_x = best_solution[f'x_{j}']
        optimized_y = best_solution[f'y_{j}']

        # Add the optimized location to both the cumulative list and the optimized results
        locationsWithDepot.append((optimized_x, optimized_y))
        optimized_locations.append((optimized_x, optimized_y))

        print(f"Location {j + 1}: Optimized Coordinates: (x = {optimized_x}, y = {optimized_y})")

    # Plotting the results
    plot_solution(locations, optimized_locations, depot)

def plot_solution(initial_locations, optimized_locations, depot):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the initial locations
    x_initial, y_initial = zip(*initial_locations)
    ax.scatter(x_initial, y_initial, color='blue', label='Initial Locations', marker='o')

    # Plot the optimized locations
    x_optimized, y_optimized = zip(*optimized_locations)
    ax.scatter(x_optimized, y_optimized, color='red', label='Optimized Locations', marker='x')

    # Plot the depot
    ax.scatter(*depot, color='green', label='Depot', marker='s')

    # Plot the rectangular boundary box
    box_x = [x_lower_bound, x_upper_bound, x_upper_bound, x_lower_bound, x_lower_bound]
    box_y = [y_lower_bound, y_lower_bound, y_upper_bound, y_upper_bound, y_lower_bound]
    ax.plot(box_x, box_y, color='black', linestyle='--', label='Rectangular Boundary Box')

    # Set axis limits
    ax.set_xlim(-1, 25)
    ax.set_ylim(-15, 15)

    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Optimization of Locations within a Rectangular Box')
    ax.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Run the cumulative CQM case
solve_with_cumulative_cqm()
