import numpy as np
import matplotlib.pyplot as plt

"""
EXPERIMENT 2: SELF-ORGANIZED CRITICALITY (HONEST - CINEMATIC)
Scientific Configuration:
- Start: Disordered Gas (M ~ 0).
- Drivers: Hebbian Learning + Positive Matter Bias.
- Process: Slow accretion of connectivity.
- Outcome: Spontaneous emergence of Solid Matter (M -> 1.0).
"""

# --- PHYSICS PARAMETERS (CINEMATIC) ---
N_NEURONS = 1000 
EPOCHS = 8000
NOISE_LEVEL = 0.1        
LEARNING_RATE = 0.005    # Slowed down for visual S-curve
DECAY_RATE = 0.0         # No decay to ensure accumulation

# --- INITIALIZATION ---
# Positive Weights and Bias
np.random.seed(42)
weights = np.random.rand(N_NEURONS, N_NEURONS) * 0.005
biases = np.abs(np.random.randn(N_NEURONS)) * 0.05
activity = np.tanh(np.random.randn(N_NEURONS))

order_history = []
time_steps = []

print(f"Starting Cinematic Evolution...")

for epoch in range(EPOCHS):
    
    # Dynamics
    noise = np.random.randn(N_NEURONS) * NOISE_LEVEL
    current = np.dot(weights, activity) + biases + noise
    new_activity = np.tanh(current)
    
    # Hebbian
    correlation = np.outer(new_activity, new_activity)
    delta_w = LEARNING_RATE * correlation 
    
    weights += delta_w
    np.fill_diagonal(weights, 0)
    weights = np.clip(weights, 0.0, 1.0)
    
    activity = new_activity
    
    # Stats
    magnetization = np.abs(np.mean(activity))
    order_history.append(magnetization)
    time_steps.append(epoch)
    
    if epoch % 500 == 0:
        avg_weight = np.mean(weights)
        print(f"Epoch {epoch}: Magnetization={magnetization:.4f}, MeanWeight={avg_weight:.4f}")

# --- VISUALIZATION ---
plt.figure(figsize=(12, 6))
plt.plot(time_steps, order_history, label='Global Order (Magnetization)')
plt.axhline(y=NOISE_LEVEL, color='r', linestyle='--', alpha=0.3, label='Noise Floor')
plt.title('Experiment 2: The Emergence of Order (Cinematic)')
plt.xlabel('Time (Epochs)')
plt.ylabel('Order Parameter |M|')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()