import numpy as np
import time
from scipy.optimize import minimize
from dwave.system import LeapHybridSampler
import dimod
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameter settings
NUM_UAV = 25
C = 3e8  # Speed of light, in m/s
f = 2.4e9  # Operating frequency, in Hz
wavelength = C / f
k = 2 * np.pi / wavelength  # Wave number

# Target direction for array factor calculation
target_angle = np.array([1, 0, 0])  # Example target in the x-direction

# Initial random positions (range assumed to be between 0 and 100 meters)
np.random.seed(0)
initial_positions = np.random.uniform(-100, 100, (NUM_UAV, 3)).flatten()


# Define the array factor calculation function
def array_factor(positions, target_dir):
    positions = positions.reshape((NUM_UAV, 3))
    phases = k * np.dot(positions, target_dir)
    af = np.abs(np.sum(np.exp(1j * phases)))
    return af

# SciPy optimization function and timing
def scipy_optimize():
    def objective(positions):
        return -array_factor(positions, target_angle)
    
    def constraint_z(positions):
        z_fixed = 50
        positions = positions.reshape((NUM_UAV, 3))
        return positions[:, 2] - z_fixed

    constraints = {'type': 'eq', 'fun': constraint_z}
    start_time = time.time()
    result = minimize(objective, initial_positions, method='SLSQP', constraints=constraints, options={'disp': True, 'maxiter': 1000})
    end_time = time.time()
    scipy_positions = result.x.reshape((NUM_UAV, 3))
    scipy_af = array_factor(result.x, target_angle)
    scipy_time = end_time - start_time
    return scipy_positions, scipy_af, scipy_time

def dwave_optimize():
    # D-Wave parameters
    num_uav = NUM_UAV
    fixed_altitude = 50
    grid_points = 25
    min_separation = 5.0  # Minimum separation for dispersion (set higher to encourage spreading)
    position_scale = 70.0
    one_hot_scale = 10.0
    separation_scale = 60.0
    dispersion_scale = 5.0  # Dispersion penalty scale

    x_points = np.linspace(0, 100, grid_points)
    y_points = np.linspace(0, 100, grid_points)
    all_positions = [(x, y, fixed_altitude) for x in x_points for y in y_points]
    phases = [k * np.dot(np.array(pos), target_angle) for pos in all_positions]

    Q = {}
    
    def get_var_name(uav_idx, pos_idx):
        return f"uav{uav_idx}_pos{pos_idx}"
    
    for i in range(num_uav):
        for p1 in range(len(all_positions)):
            var1 = get_var_name(i, p1)
            
            # Diagonal term: prioritize constructive interference (array factor)
            Q[(var1, var1)] = -2 * position_scale * np.cos(phases[p1])
            
            # One-hot constraint to ensure one position per UAV
            for p2 in range(p1 + 1, len(all_positions)):
                var2 = get_var_name(i, p2)
                Q[(var1, var2)] = one_hot_scale

            # Add separation constraint and dispersion penalty
            for j in range(i + 1, num_uav):
                for p2 in range(len(all_positions)):
                    var2 = get_var_name(j, p2)
                    dist = np.linalg.norm(np.array(all_positions[p1])[:2] - np.array(all_positions[p2])[:2])

                    # Separation constraint
                    if dist < min_separation:
                        Q[(var1, var2)] = Q.get((var1, var2), 0) + separation_scale * (min_separation - dist)
                    
                    # Dispersion penalty for too-close UAVs
                    if dist < min_separation * 2:  # Double the min separation threshold
                        Q[(var1, var2)] = Q.get((var1, var2), 0) + dispersion_scale * (min_separation * 2 - dist)

    sampler = LeapHybridSampler()
    start_time = time.time()
    response = sampler.sample_qubo(Q)
    end_time = time.time()
    dwave_time = end_time - start_time
    sample = response.first.sample

    dwave_positions = []
    for i in range(num_uav):
        assigned = False
        for p in range(len(all_positions)):
            var_name = f"uav{i}_pos{p}"
            if sample.get(var_name, 0) == 1:
                dwave_positions.append(all_positions[p])
                assigned = True
                break
        if not assigned:
            logger.warning(f"UAV {i} was not assigned a valid position.")
    
    dwave_positions = np.array(dwave_positions)
    dwave_af = array_factor(dwave_positions.flatten(), target_angle)
    return dwave_positions, dwave_af, dwave_time


# Run both optimizations and display results
scipy_positions, scipy_af, scipy_time = scipy_optimize()
dwave_positions, dwave_af, dwave_time = dwave_optimize()

print("SciPy Optimization Results:")
print(f"Optimized Array Factor: {scipy_af}")
print(f"Execution Time: {scipy_time:.4f} seconds")
print("Optimized Positions (meters):")
for i, pos in enumerate(scipy_positions):
    print(f"UAV {i+1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

print("\nD-Wave Optimization Results:")
print(f"Optimized Array Factor: {dwave_af}")
print(f"Execution Time: {dwave_time:.4f} seconds")
print("Optimized Positions (meters):")
for i, pos in enumerate(dwave_positions):
    print(f"UAV {i+1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

# Visualization of results
def visualize_positions(positions, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', s=100)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    plt.show()

visualize_positions(scipy_positions, "SciPy Optimized UAV Positions")
visualize_positions(dwave_positions, "D-Wave Optimized UAV Positions")
