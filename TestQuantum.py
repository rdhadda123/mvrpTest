import numpy as np
import time
from scipy.optimize import minimize

# Parameter settings
NUM_UAV = 10       # Number of UAVs
C = 3e8            # Speed of light, in m/s
f = 2.4e9          # Operating frequency, in Hz
wavelength = C / f
k = 2 * np.pi / wavelength  # Wave number

# Target direction (in Cartesian coordinates)
# For example, the target is in the positive x-direction
target_angle = np.array([1, 0, 0])  # Unit vector

# Initial random positions (range assumed to be between 0 and 100 meters)
np.random.seed(0)
initial_positions = np.random.uniform(0, 100, (NUM_UAV, 3)).flatten()

def array_factor(positions, target_dir):
    """
    Calculate the Array Factor
    positions: UAV position array, length NUM_UAV*3
    target_dir: target direction unit vector
    """
    positions = positions.reshape((NUM_UAV, 3))
    # Calculate the phase for each UAV relative to the reference point
    phases = k * np.dot(positions, target_dir)
    # Array factor is the coherent sum of all UAV signals
    af = np.abs(np.sum(np.exp(1j * phases)))
    return af

def objective(positions):
    # To maximize the array factor, minimize the negative of the array factor
    af = array_factor(positions, target_angle)
    return -af

# Constraint (e.g., UAVs' altitude is fixed to a certain value, can be adjusted as needed)
def constraint_z(positions):
    z_fixed = 50  # Fixed altitude at 50 meters
    positions = positions.reshape((NUM_UAV, 3))
    return positions[:, 2] - z_fixed

constraints = ({
    'type': 'eq',
    'fun': constraint_z
})

# Optimization
start = time.time()
result = minimize(objective, initial_positions, method='SLSQP', constraints=constraints,
                  options={'disp': True, 'maxiter': 1000})
end = time.time()
print(f"took", {end - start}, "seconds")
#result2 = Dwave(objective, initial_positions,condio)
# Display results
optimized_positions = result.x.reshape((NUM_UAV, 3))
print("Optimized UAV positions (meters):")
for i, pos in enumerate(optimized_positions):
    print(f"UAV {i+1}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

# Calculate the optimized array factor
optimized_af = array_factor(result.x, target_angle)
print(f"Optimized Array Factor: {optimized_af}")
