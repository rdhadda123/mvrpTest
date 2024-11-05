import math
import random
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Integer, Real
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
min_separation = factor * diagonal_length

# Randomly generate initial N locations (x, y) in 2D space
locations = [(random.randint(10, 20), random.randint(-10, 10)) for _ in range(num_locations)]
print(f"Depot is located at: {depot}")
print(f"These are the current locations: {locations}")

# Using Hybrid CQM (Constrained Quadratic Model) Solver
def solve_with_cqm():
    print("Solving with Hybrid CQM Solver...")

    # Initialize the Constrained Quadratic Model
    cqm = ConstrainedQuadraticModel()

    # Variables for each location (x, y coordinates)
    location_vars = {}
    for i in range(num_locations):
        location_vars[i] = {
            'x': Real(f'x_{i}', lower_bound = x_lower_bound, upper_bound = x_upper_bound),
            'y': Real(f'y_{i}', lower_bound = y_lower_bound, upper_bound = y_upper_bound)
        }

    # Objective: Minimize the sum of Manhattan distances from each location to the depot
    objective = 0
    for i in range(num_locations):
        x_i, y_i = location_vars[i]['x'], location_vars[i]['y']

        # Add auxiliary variables for the absolute value |x_i - depot[0]|, |y_i - depot[1]|
        abs_x = Real(f'abs_x_{i}', lower_bound=0)
        abs_y = Real(f'abs_y_{i}', lower_bound=0)

        # Add constraints to ensure abs_x represents the absolute value using two inequalities
        cqm.add_constraint(abs_x - (x_i - depot[0]) == 0, label=f'abs_x_pos_{i}')
        cqm.add_constraint(abs_x + (x_i - depot[0]) == 0, label=f'abs_x_neg_{i}')
        cqm.add_constraint(abs_y - (y_i - depot[1]) == 0, label=f'abs_y_pos_{i}')
        cqm.add_constraint(abs_y + (y_i - depot[1]) == 0, label=f'abs_y_neg_{i}')

        # Add the auxiliary variables to the objective (Manhattan distance)
        objective += abs_x + abs_y

    # Set the objective in the CQM
    cqm.set_objective(objective)

    # Solve with the Leap Hybrid CQM sampler
    sampler = LeapHybridCQMSampler()
    result = sampler.sample_cqm(cqm)

    # Get the best solution
    best_solution = result.first.sample

    # Initial optimized locations, all close to (10, 0)
    optimized_locations = [(best_solution[f'x_{i}'], best_solution[f'y_{i}']) for i in range(num_locations)]

    # Separate each point based on all previous points
    separated_locations = [optimized_locations[0]]  # Start with the first point at (10, 0)
    for i in range(1, num_locations):
        # Start the new point at (10, 0) and adjust as necessary
        new_x, new_y = optimized_locations[i]
        while True:
            # Check the minimum separation against all previous points
            too_close = False
            for (prev_x, prev_y) in separated_locations:
                if math.sqrt((new_x - prev_x)**2 + (new_y - prev_y)**2) < min_separation:
                    too_close = True
                    break

            # If all previous points satisfy the minimum separation, add the point
            if not too_close:
                separated_locations.append((new_x, new_y))
                break
            else:
                # Increment y to keep the separation, alternate between positive and negative offsets
                if i % 2 == 1:
                    new_y += min_separation
                else:
                    new_y -= min_separation

                # Clamp y within bounds
                new_y = max(y_lower_bound, min(y_upper_bound, new_y))

    print("\nSeparated Locations (after applying cumulative minimum separation):")
    for i, (x, y) in enumerate(separated_locations):
        print(f"Location {i + 1}: Coordinates: (x = {x}, y = {y})")

    # Plotting the results with separated locations
    plot_solution(locations, separated_locations, depot)

def plot_solution(initial_locations, optimized_locations, depot):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the initial locations
    x_initial, y_initial = zip(*initial_locations)
    ax.scatter(x_initial, y_initial, color='blue', label='Initial Locations', marker='o')

    # Plot the optimized locations
    x_optimized, y_optimized = zip(*optimized_locations)
    ax.scatter(x_optimized, y_optimized, color='red', label='Separated Locations', marker='x')

    # Plot the depot
    ax.scatter(*depot, color='green', label='Depot', marker='s')

    # Plot the rectangular boundary box
    box_x = [x_lower_bound, x_upper_bound, x_upper_bound, x_lower_bound, x_lower_bound]
    box_y = [y_lower_bound, y_lower_bound, y_upper_bound, y_upper_bound, y_lower_bound]
    ax.plot(box_x, box_y, color='black', linestyle='--', label='Rectangular Boundary Box')

    # Set axis limits
    ax.set_xlim(-5, 25)
    ax.set_ylim(-15, 15)

    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Separated Locations with Cumulative Minimum Distance Constraint')
    ax.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Run the CQM case with cumulative separation
solve_with_cqm()
