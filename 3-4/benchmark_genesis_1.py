import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import time

"""
BENCHMARK: EXPERIMENTO 1 - GÉNESIS
Headless execution for data validation and stability auditing.
"""

# --- CONFIGURACIÓN FÍSICA Y DE SIMULACIÓN ---
N = 256
T_sim_bench = 2000 # Extended run for stability check
LEARNING_RATE = 0.01
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Initialize Universe
np.random.seed(42) # Fixed seed for reproducibility
universe_state = np.random.rand(N, N, 4).astype(np.float32)

def free_energy_functional(state):
    R, G, B, A = state[:,:,0], state[:,:,1], state[:,:,2], state[:,:,3]
    grad_R = np.gradient(R)
    curvature_energy = np.sum(grad_R[0]**2 + grad_R[1]**2)
    interaction_energy = -np.sum(R * G) 
    entropy = -np.sum(A * np.log(np.abs(A) + 1e-6))
    return curvature_energy + interaction_energy + entropy, entropy

def shader_step(state):
    new_state = state.copy()
    r, g, b, a = state[:,:,0], state[:,:,1], state[:,:,2], state[:,:,3]
    
    diff_r = convolve2d(r, LAPLACIAN_KERNEL, mode='same', boundary='wrap')
    diff_g = convolve2d(g, LAPLACIAN_KERNEL, mode='same', boundary='wrap')
    
    dr = LEARNING_RATE * (diff_r + g * r - r**3)
    dg = LEARNING_RATE * (diff_g - r + 0.1*g)
    grad_x, grad_y = np.gradient(r)
    db = 0.05 * (np.abs(grad_x) + np.abs(grad_y))
    da = 0.01 * np.abs(dr)
    
    new_state[:,:,0] += dr
    new_state[:,:,1] += dg
    new_state[:,:,2] += db
    new_state[:,:,3] += da
    new_state = np.clip(new_state, 0.0, 1.0)
    
    # Calculate magnitude of change for stability check
    delta_magnitude = np.mean(np.abs(new_state - state))
    return new_state, delta_magnitude

print(f"Starting Benchmark: {T_sim_bench} epochs...")
results = []
start_time = time.time()

for i in range(T_sim_bench):
    universe_state, delta = shader_step(universe_state)
    
    if i % 10 == 0:
        energy, entropy = free_energy_functional(universe_state)
        results.append({
            "Epoch": i,
            "FreeEnergy": energy,
            "SystemEntropy": entropy,
            "Stability_dState": delta
        })
    
    if i % 100 == 0:
        print(f"Epoch {i}: Energy={results[-1]['FreeEnergy']:.2f}, dState={delta:.6f}")

total_time = time.time() - start_time
print(f"Benchmark Complete. Time: {total_time:.2f}s")

# Save Data
df = pd.DataFrame(results)
csv_path = "genesis_1_benchmark_data.csv"
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")

# Quick Audit
initial_energy = df.iloc[0]['FreeEnergy']
final_energy = df.iloc[-1]['FreeEnergy']
is_stable = df.iloc[-1]['Stability_dState'] < 1e-4

print("\n--- AUDIT VERDICT ---")
print(f"Energy Change: {initial_energy:.2f} -> {final_energy:.2f}")
print(f"Monotonic Decrease: {df['FreeEnergy'].is_monotonic_decreasing}") # Not strictly required but expected for gradient descent
print(f"System Stable (dState < 1e-4): {is_stable}")
