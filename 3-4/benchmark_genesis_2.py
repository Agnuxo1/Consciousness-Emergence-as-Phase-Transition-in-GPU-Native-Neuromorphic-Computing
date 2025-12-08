import numpy as np
import pandas as pd
import time

"""
BENCHMARK: EXPERIMENT 2 (HONEST - CINEMATIC)
"""

N_NEURONS = 500
EPOCHS = 8000 # Extended to see the full curve
NOISE_LEVEL = 0.1
LEARNING_RATE = 0.005
DECAY_RATE = 0.0

np.random.seed(42)
weights = np.random.rand(N_NEURONS, N_NEURONS) * 0.005
biases = np.abs(np.random.randn(N_NEURONS)) * 0.05
activity = np.tanh(np.random.randn(N_NEURONS))

results = []
print(f"Starting Honest Benchmark (Cinematic)...")
start_time = time.time()

for epoch in range(EPOCHS):
    
    noise = np.random.randn(N_NEURONS) * NOISE_LEVEL
    current = np.dot(weights, activity) + biases + noise
    activity = np.tanh(current)
    
    delta_w = LEARNING_RATE * np.outer(activity, activity)
    weights += delta_w
    np.fill_diagonal(weights, 0)
    weights = np.clip(weights, 0.0, 1.0)
    
    magnet = np.abs(np.mean(activity))
    
    if epoch % 10 == 0:
        results.append({"Epoch": epoch, "Magnetization": magnet, "MeanWeight": np.mean(weights)})
        
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: M={magnet:.4f}, W_mean={np.mean(weights):.4f}")

total_time = time.time() - start_time
print(f"Benchmark Complete. Time: {total_time:.2f}s")

df = pd.DataFrame(results)
csv_path = "genesis_2_honest_benchmark_final.csv"
df.to_csv(csv_path, index=False)

m_start = df[df['Epoch'] < 200]['Magnetization'].mean()
m_mid = df[df['Epoch'] > 4000]['Magnetization'].mean()

print(f"\n--- AUDIT VERDICT ---")
print(f"Start Order: {m_start:.4f}")
print(f"Mid Order:   {m_mid:.4f}")

if m_start < 0.2 and m_mid > 0.8:
    print("SUCCESS: Phase Transition Confirmed (Slow Accretion)")
else:
    print("FAILURE: Transition too slow or conditions not met.")
