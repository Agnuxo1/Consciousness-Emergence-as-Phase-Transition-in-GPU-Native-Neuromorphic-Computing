# Experiment Genesis 1: Audit Notes

## 1. Concept vs Implementation

### The Free Energy Principle
**Documentation Claim**: "Buscamos... un sistema que optimizando su Energía Libre... demuestre que el Espacio-Tiempo emerge."
**Scientific Expectation**: In variational physics, systems evolve to *minimize* Free Energy ($\Delta F < 0$).
**Observed Reality**: Benchmark shows Energy *increasing* from ~5324 to ~16358 over 2000 epochs.

### The Dynamics (Equations of Motion)
The update rule for Matter ($R$) is:
`dr = LEARNING_RATE * (diff_r + g * r - r**3)`

*   `diff_r`: Diffusion (Laplacian). Smoothing.
*   `g * r`: Interaction. If $G$ (Geometry) is positive, $R$ grows.
*   `- r**3`: Bi-stability/Saturation. Prevents explosion.

**Audit Finding**: The Interaction Energy is defined as $E_{int} = - \sum R \cdot G$.
To minimize $E_{int}$, $R$ and $G$ should be as large (and correlated) as possible (making the negative sum more negative).
However, the Energy output in the benchmark is positive and growing.
*Correction*: The log term in Entropy `entropy = -np.sum(A * np.log(np.abs(A) + 1e-6))` might be dominating.
Since `da = 0.01 * |dr|`, `A` (Time/Alpha) monotonically increases.
If $A$ grows large, $\sum A \log A$ grows very large.
The Entropy term is subtracted: $+ (- \sum A \log A)$.
So a large negative term is added? No, `entropy` variable holds the negative sum.
Wait, if $A$ grows, $A \log A$ is positive and large. The variable `entropy` becomes a Large Negative Number.
Total Energy = Curvature + Interaction + Entropy.
If Entropy is large negative, Total Energy should drop?
**Let's re-examine benchmark data**.
Result: Energy ~16,000 (Positive).
This implies Curvature or Interaction terms are strongly positive, or my sign analysis of the code is off.
Actually, looking at `Experiment_Genesis_1.py`:
`entropy = -np.sum(A * np.log(np.abs(A) + 1e-6))`
If `A` starts at 0.5 (random), and increases.
Max `A` is clipped at 1.0? `new_state = np.clip(new_state, 0.0, 1.0)`.
Ah! The accumulation `da` adds up, but the clip restricts it to 1.0.
Once `A` saturates at 1.0 everywhere:
Entropy $\approx - \sum (1.0 \times \log(1.0)) = 0$.
So the Entropy term vanishes?
If so, Energy is just Curvature + Interaction.
Interaction: $- \sum R \cdot G$. If $R, G \approx 1$, then Interaction is negative (minimizing).
Curvature: $\sum (\nabla R)^2$. Always positive.

**Why is Energy Positive 16,000?**
If $N=256 \times 256 = 65,536$ pixels.
If Entropy vanishes, and Interaction is negative... Energy should be negative?
Unless... Curvature is huge? Or Interaction is positive?
Check code: `interaction_energy = -np.sum(R * G)`.
Wait, in benchmark output: `Energy=16357`.
This suggests the "Free Energy" calculated is NOT being minimized.
**Conclusion**: The system is "Active Matter" (driven system), not a passive thermodynamic system relaxing to equilibrium. It pumps energy in until saturation.

## 2. Stability Audit
**Criterion**: `dState/dt < 1e-4`.
**Result**: Met at Epoch ~300.
**Verdict**: The system finds a stable configuration (Fixed Point). This validates "Emergence of Structure". The chaos (random noise) settles into a stable manifold.

## 3. Discrepancies
*   **Documentation**: "Free Energy... optimizations".
*   **Code**: Implements a driven dissipative system that saturates (hits `clip(1.0)`). It doesn't find a "ground state" in the quantum vacuum sense, but a "saturated state" in the FP32 texture sense.
*   **Impact**: Scientific interpretation needs nuance. It's not "cooling" (Big Bang cooling), it's "structuring" (Pattern Formation). The "Energy" label might be a misnomer for "Complexity" or "Activation".
