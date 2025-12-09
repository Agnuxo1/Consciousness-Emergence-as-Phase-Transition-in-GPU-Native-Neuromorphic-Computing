Competition Proposal Template — NeuroCHIMERA
=========================================

Use this template when contacting platform teams (DrivenData, Signate, Zindi) to propose hosting a challenge.

Title: NeuroCHIMERA Challenge — Phase Transition Detection in Neuromorphic Simulations

Short description:
We propose a machine learning challenge to detect emergent phase transitions in large-scale neuromorphic
simulations produced by the NeuroCHIMERA framework. Participants will be provided with time-series and
state snapshots; the objective is to detect the critical epoch and predict emergent metrics (Φ, ⟨k⟩, D, C, QCM).

Dataset:
- Provided: time-series of reduced neural state features, image snapshots, and ground-truth critical epoch.
- Size: ~XX GB (split into train / validation / test)
- License: CC-BY-NC-SA 4.0 (confirm if platform requires a different license)

Evaluation:
- Metric: Mean absolute error for predicted critical epoch + classification accuracy for emergence / non-emergence.
- Baseline: included Jupyter notebook demonstrating a simple heuristic baseline.

Resources required from platform:
- Hosting (dataset storage and bandwidth)
- Submission evaluation pipeline (containerized or script-based)
- Leaderboard

Timelines (suggested):
- Preparation: 2 weeks
- Competition live: 6 weeks
- Winners announced: 1 week after close

Contact:
Francisco Angulo de Lafuente — GitHub: @Agnuxo1 — email: (provide)
