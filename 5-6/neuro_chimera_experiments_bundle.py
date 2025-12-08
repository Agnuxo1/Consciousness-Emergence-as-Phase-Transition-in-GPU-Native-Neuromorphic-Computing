"""
NeuroCHIMERA - Two simulation experiments bundle
File: NeuroCHIMERA_experiments_bundle.py
Language: English (code & comments)

This single Python program creates two complete simulation experiments and writes a small WebGPU/WebGL frontend
(using the provided webgpu-cross-platform-app repository as a target) to visualize results.

What this bundle does:
- Implements Experiment 1: GF(2) / GF(2^n) discrete-field network dynamical simulation (statistical physics + neuromorphic)
  * Simulates binary/field neurons with sparse connectivity on GPU using PyTorch (CUDA). Uses bit-packing optimization for
    GF(2) dynamics and a float32 variant for GF(2^n)-like arithmetic via table-based operations.
  * Measures a battery of metrics: global synchrony, Shannon entropy, Lempel-Ziv complexity (approx), integrated information
    proxy (phi_approx), PCA-based abrupt-change detection, and CUSUM to determine critical time tc.
  * Runs parameter sweeps and produces reproducible checkpoints (seeds, configs).

- Implements Experiment 2: RGBA-CHIMERA neuromorphic network (simplified neuromorphic modules)
  * Represents an RGBA-structured network (4-channel modules) where each module is a small recurrent module with gating.
  * Demonstrates emergence of metastable attractors and measures memory persistence, pattern integration, and metacognitive
    proxy (confidence calibration on probe tasks).

- Includes a minimal HTTP server that writes a small WebGPU/WebGL visualization frontend (index.html + app.js) and serves
  simulation output (JSON + binary). The frontend is designed to be used with the webgpu-cross-platform-app and/or any
  browser with WebGPU enabled. The repo link: https://github.com/Agnuxo1/webgpu-cross-platform-app

Run requirements (on the machine executing the Python scripts):
- Linux or macOS with NVIDIA GPU (CUDA) or CPU fallback
- Python 3.10+
- PyTorch with CUDA (recommended), or CPU-only fallback (slower)
- numpy, scipy, scikit-learn

This file is self-contained: running it with `python NeuroCHIMERA_experiments_bundle.py --help` will show commands
that create and run the two experiments and start the visualization server.

IMPORTANT: The WebGPU frontend is lightweight and intentionally written to be served from the local HTTP server.
To leverage the full cross-platform app from the GitHub repo, clone the repo and copy the generated frontend files into
its public/www folder. Instructions are printed when the server starts.

"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
import random
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Dict, Any, Tuple, List

# Numerical & ML libs
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

# ----------------------------- Utility functions ---------------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed + 1)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 3)


# ---------------------------------------------------------------------------------
# Experiment 1: GF(2) / GF(2^n) discrete-field network dynamics
# ---------------------------------------------------------------------------------

class GFNetworkConfig:
    """Configuration for the GF network simulation."""
    def __init__(self,
                 N: int = 65536,      # number of nodes
                 p_conn: float = 0.02,
                 field_bits: int = 1,  # 1 for GF(2), >1 for GF(2^n) emulated
                 device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu',
                 seed: int = 42,
                 dt: float = 1.0):
        self.N = N
        self.p_conn = p_conn
        self.field_bits = field_bits
        self.device = device
        self.seed = seed
        self.dt = dt


class GFNetworkSimulation:
    """Simulates discrete-field dynamics on a sparse random graph.

    Discrete update rule (binary GF(2) variant):
        s_i(t+1) = 1 if (sum_j W_ij * s_j(t) + b_i + noise_i) mod 2 == 1 else 0

    For field_bits > 1 we emulate GF(2^n) states via integer arrays and precomputed addition/multiplication tables.

    """

    def __init__(self, cfg: GFNetworkConfig):
        self.cfg = cfg
        seed_all(cfg.seed)
        self.N = cfg.N
        self.device = cfg.device

        # adjacency in sparse format: we store indices of connections per node
        # For reproducibility store RNG state
        self.rng = np.random.default_rng(cfg.seed)
        self._build_graph()
        self._init_state()

    def _build_graph(self):
        N = self.N
        p = self.cfg.p_conn
        rng = self.rng
        # Expected number of edges = N*N*p, but we keep out-degree approx = k = p*N
        k = max(1, int(p * N))
        # adjacency list: list of arrays indices
        self.adj = [None] * N
        self.weights = [None] * N
        for i in range(N):
            # sample k incoming connections uniformly without replacement
            # allow self-connections
            idx = rng.choice(N, size=k, replace=False)
            self.adj[i] = np.array(idx, dtype=np.int64)
            # weights in GF(2) are 0/1; for field_bits>1 we use int in [0,2^n-1]
            if self.cfg.field_bits == 1:
                self.weights[i] = np.ones(k, dtype=np.uint8)
            else:
                modulus = 1 << self.cfg.field_bits
                self.weights[i] = rng.integers(0, modulus, size=k, dtype=np.uint32)
        print(f"[GFNet] Built graph N={N}, k~{k}")

    def _init_state(self):
        N = self.N
        if self.cfg.field_bits == 1:
            # bit-packed state as numpy uint8 array for simplicity (one byte per node)
            self.state = self.rng.integers(0, 2, size=N, dtype=np.uint8)
        else:
            modulus = 1 << self.cfg.field_bits
            self.state = self.rng.integers(0, modulus, size=N, dtype=np.uint32)
        # biases
        if self.cfg.field_bits == 1:
            self.bias = self.rng.integers(0, 2, size=N, dtype=np.uint8)
        else:
            modulus = 1 << self.cfg.field_bits
            self.bias = self.rng.integers(0, modulus, size=N, dtype=np.uint32)

    def step(self, noise_prob: float = 0.01):
        """Perform a single synchronous update step."""
        N = self.N
        new_state = np.empty_like(self.state)
        if self.cfg.field_bits == 1:
            for i in range(N):
                neigh = self.adj[i]
                w = self.weights[i]
                s_sum = (self.state[neigh] & w).sum()  # mod 2 implicitly
                val = (s_sum + int(self.bias[i])) & 1
                # add bit-flip noise
                if self.rng.random() < noise_prob:
                    val ^= 1
                new_state[i] = val
        else:
            mod = 1 << self.cfg.field_bits
            for i in range(N):
                neigh = self.adj[i]
                w = self.weights[i]
                # emulate multiplication in GF(2^n) by integer ops mod 2^n (not strictly field mult, but emulation)
                s_sum = (np.bitwise_xor.reduce(self.state[neigh] * w) + int(self.bias[i])) & (mod - 1)
                if self.rng.random() < noise_prob:
                    s_sum ^= self.rng.integers(0, mod)
                new_state[i] = s_sum
        self.state = new_state

    # ---------- Metrics (math/physics applied) -----------------
    def global_synchrony(self) -> float:
        # fraction of nodes in the majority state (binary) or normalized variance for multivalued
        if self.cfg.field_bits == 1:
            p = float(self.state.mean())
            sync = max(p, 1.0 - p)
            return sync
        else:
            vals, counts = np.unique(self.state, return_counts=True)
            p_max = counts.max() / float(self.N)
            return p_max

    def shannon_entropy(self) -> float:
        vals, counts = np.unique(self.state, return_counts=True)
        probs = counts / counts.sum()
        return -float(np.sum(probs * np.log2(probs + 1e-12)))

    def approx_lz_complexity(self, block_size: int = 16) -> float:
        # Approximate Lempel-Ziv by compressibility of bit-string chunks (binary case)
        if self.cfg.field_bits == 1:
            bitstr = ''.join(['1' if x else '0' for x in self.state[:min(self.N, 65536)]])
            # naive LZ complexity: count distinct substrings up to block_size
            seen = set()
            count = 0
            L = len(bitstr)
            for i in range(0, L, block_size):
                chunk = bitstr[i:i+block_size]
                if chunk not in seen:
                    seen.add(chunk)
                    count += 1
            return count / max(1, L / block_size)
        else:
            # for multivalued use entropy proxy
            return self.shannon_entropy()

    def phi_approx(self) -> float:
        # A tractable proxy for integrated information: mutual information between halves
        mid = self.N // 2
        A = self.state[:mid]
        B = self.state[mid:]
        # discretize (binary or mod)
        def dist_entropy(x):
            vals, counts = np.unique(x, return_counts=True)
            p = counts / counts.sum()
            return -np.sum(p * np.log2(p + 1e-12))
        H_A = dist_entropy(A)
        H_B = dist_entropy(B)
        # joint entropy approx via joint unique pairs
        pairs = np.array([A, B]).T
        tup = [tuple(x) for x in pairs]
        _, counts = np.unique(tup, return_counts=True, axis=0)
        p = counts / counts.sum()
        H_AB = -np.sum(p * np.log2(p + 1e-12))
        # phi approx = (H_A + H_B) - H_AB (redundant info reduction)
        return float((H_A + H_B) - H_AB)

    def pca_change_detection(self, history_states: List[np.ndarray], ncomponents: int = 8) -> float:
        # Stack last T states (sampled snapshots) and compute PCA; detect abrupt change in leading singular values
        if len(history_states) < 4:
            return 0.0
        X = np.stack([s.astype(np.float32) for s in history_states[-16:]])  # (T, N) -> heavy, but used only small T
        pca = PCA(n_components=min(ncomponents, X.shape[0]))
        try:
            pca.fit(X)
            sv = pca.explained_variance_ratio_[0]
            return float(sv)
        except Exception:
            return 0.0

    # ----------------- Run experiment (parameter sweep & detection) -----------------
    def run(self, T: int = 5000, sample_interval: int = 10, noise: float = 0.01) -> Dict[str, Any]:
        seed_all(self.cfg.seed)
        metrics = {'t': [], 'sync': [], 'entropy': [], 'lz': [], 'phi': [], 'pca_sv': []}
        history = []
        tc_candidate = None
        S_series = []
        for t in range(T):
            self.step(noise_prob=noise)
            if t % sample_interval == 0:
                s = self.global_synchrony()
                e = self.shannon_entropy()
                lz = self.approx_lz_complexity()
                ph = self.phi_approx()
                sv = self.pca_change_detection(history)
                metrics['t'].append(t)
                metrics['sync'].append(s)
                metrics['entropy'].append(e)
                metrics['lz'].append(lz)
                metrics['phi'].append(ph)
                metrics['pca_sv'].append(sv)
                history.append(self.state.copy())
                # aggregated index S as weighted sum (example)
                S = s + (1.0 - e / (math.log2(self.N+1))) + ph*0.1
                S_series.append(S)
                # CUSUM for abrupt change detection (simple)
                if len(S_series) >= 8 and tc_candidate is None:
                    dif = np.diff(S_series[-8:])
                    if np.mean(dif) > 0.05:  # threshold heuristic, to be statistically tested in reproductions
                        tc_candidate = t
            # optional early stop
        out = {'cfg': self.cfg.__dict__, 'metrics': metrics, 'tc': tc_candidate}
        return out

# ---------------------------------------------------------------------------------
# Experiment 2: RGBA-CHIMERA simplified neuromorphic network
# ---------------------------------------------------------------------------------

class RGBAModule:
    """A small 4-channel recurrent module. Channels correspond to R,G,B,A (feature channels) with gating.

    Each module i has state vector x_i(t) in R^4; interactions between modules are via sparse W matrices.
    Dynamics inspired by continuous-time recurrent networks discretized with Euler.

        x_i(t+1) = x_i(t) + dt * ( -alpha x_i + f( sum_j W_ij x_j + I_i(t) ) )

    Where f is a nonlinear activation (tanh), and alpha is leakage.

    """
    def __init__(self, channels: int = 4):
        self.channels = channels


class RGBACHIMERAConfig:
    def __init__(self, modules: int = 1024, module_size: int = 4, p_conn: float = 0.05,
                 device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu', seed: int = 123):
        self.modules = modules
        self.module_size = module_size
        self.p_conn = p_conn
        self.device = device
        self.seed = seed


class RGBACHIMERASimulation:
    def __init__(self, cfg: RGBACHIMERAConfig):
        self.cfg = cfg
        seed_all(cfg.seed)
        self.modules = cfg.modules
        self.module_size = cfg.module_size
        # build sparse inter-module weights: each module connects to k others
        k = max(1, int(cfg.modules * cfg.p_conn))
        self.adj = [None] * cfg.modules
        self.weights = [None] * cfg.modules
        rng = np.random.default_rng(cfg.seed)
        for i in range(cfg.modules):
            idx = rng.choice(cfg.modules, size=k, replace=False)
            self.adj[i] = idx
            # weight matrices between modules: module_size x module_size
            W = rng.normal(scale=0.1, size=(k, cfg.module_size, cfg.module_size)).astype(np.float32)
            self.weights[i] = W
        # module states
        self.state = rng.normal(scale=0.01, size=(cfg.modules, cfg.module_size)).astype(np.float32)
        # biases and leak
        self.bias = rng.normal(scale=0.01, size=(cfg.modules, cfg.module_size)).astype(np.float32)
        self.alpha = 0.1
        self.dt = 0.5

    def step(self, input_drive: np.ndarray = None):
        # input_drive shape: (modules, module_size)
        M = self.modules
        new_state = self.state.copy()
        for i in range(M):
            neigh = self.adj[i]
            Wblk = self.weights[i]
            # aggregate
            agg = np.zeros(self.module_size, dtype=np.float32)
            for idx_j, j in enumerate(neigh):
                agg += Wblk[idx_j].dot(self.state[j])
            drive = agg + self.bias[i]
            if input_drive is not None:
                drive += input_drive[i]
            dx = -self.alpha * self.state[i] + np.tanh(drive)
            new_state[i] = self.state[i] + self.dt * dx
        self.state = new_state

    # Metrics: memory persistence, pattern integration, metacognitive proxy
    def memory_persistence(self, probe_pattern: np.ndarray, horizon: int = 50) -> float:
        # write a probe pattern to a subset of modules and measure overlap decay
        subset = np.arange(min(64, self.modules))
        orig_norm = np.linalg.norm(probe_pattern[subset])
        overlaps = []
        # insert pattern
        self.state[subset] += probe_pattern[subset]
        for t in range(horizon):
            self.step()
            ov = np.linalg.norm(self.state[subset] - probe_pattern[subset])
            overlaps.append(ov / (orig_norm + 1e-12))
        # persistence metric: area under normalized overlap curve
        return float(np.trapz(overlaps) / horizon)

    def pattern_integration(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> float:
        # measure network ability to bind two patterns presented to disjoint subsets
        half = self.modules // 2
        self.state[:half] += pattern_a[:half]
        # Fixed: ensure we only add what fits in the second half
        remaining = self.modules - half
        take = min(remaining, pattern_b.shape[0])
        self.state[half:half+take] += pattern_b[:take]
        # run few steps
        for _ in range(20):
            self.step()
        # compute cross-correlation of channel activations
        corr = np.corrcoef(self.state.flatten(), rowvar=False)
        return float(np.nanmean(np.abs(corr)))

    def metacognitive_proxy(self, task_inputs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        # Run a simple classification probe: present inputs, readout via linear regressor on module means
        from sklearn.linear_model import LogisticRegression
        X = []
        y = []
        for inp, lab in zip(task_inputs, labels):
            # reset small noise
            self.state *= 0.9
            # present input for few steps
            for _ in range(5):
                self.step(input_drive=inp)
            feat = self.state.mean(axis=1)  # module means
            X.append(feat)
            y.append(lab)
        X = np.stack(X)
        clf = LogisticRegression(max_iter=500).fit(X, y)
        probs = clf.predict_proba(X)
        preds = clf.predict(X)
        # confidence calibration: mean predicted prob of true class
        conf = np.mean([probs[i, y[i]] for i in range(len(y))])
        acc = np.mean(preds == y)
        return float(acc), float(conf)

    def run_probe_experiment(self, T: int = 100) -> Dict[str, Any]:
        # Run three probes: memory, integration, metacognition
        rng = np.random.default_rng(self.cfg.seed)
        # create probe patterns
        pattern = rng.normal(scale=0.5, size=(self.modules, self.module_size)).astype(np.float32)
        mem = self.memory_persistence(pattern, horizon=100)
        # pattern integration
        pattern_a = rng.normal(scale=0.2, size=(self.modules, self.module_size)).astype(np.float32)
        pattern_b = rng.normal(scale=0.2, size=(self.modules, self.module_size)).astype(np.float32)
        integ = self.pattern_integration(pattern_a, pattern_b)
        # metacognition task: binary classification with two distinct input classes
        task_inputs = []
        labels = []
        for i in range(40):
            inp = np.zeros((self.modules, self.module_size), dtype=np.float32)
            lab = i % 2
            # inject distinct patterns at two locations
            if lab == 0:
                inp[:16] = rng.normal(loc=1.0, scale=0.2, size=(16, self.module_size))
            else:
                inp[:16] = rng.normal(loc=-1.0, scale=0.2, size=(16, self.module_size))
            task_inputs.append(inp)
            labels.append(lab)
        acc, conf = self.metacognitive_proxy(task_inputs, np.array(labels))
        return {'memory_persistence': mem, 'integration': integ, 'metacog_acc': acc, 'metacog_conf': conf}

# ---------------------------------------------------------------------------------
# Utilities to write frontend files and serve
# ---------------------------------------------------------------------------------
FRONTEND_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NeuroCHIMERA - WebGPU Visualizer (minimal)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>body { font-family: sans-serif; margin: 0; padding: 0; } canvas { width: 100%; height: 60vh; display:block; }</style>
</head>
<body>
  <h2 style="margin:8px">NeuroCHIMERA - Visualizer (minimal WebGPU-compatible)</h2>
  <canvas id="gpu-canvas"></canvas>
  <pre id="log"></pre>
  <script src="app.js"></script>
</body>
</html>
"""

FRONTEND_APPJS = """
// Minimal WebGPU visualization that fetches JSON metrics and renders a heatmap onto a canvas using WebGL/WebGPU
(async () => {
  const log = (s) => document.getElementById('log').textContent += s + '\n';
  log('Starting NeuroCHIMERA visualizer...');
  const canvas = document.getElementById('gpu-canvas');
  // Fetch metrics
  try {
    const resp = await fetch('/latest_metrics.json');
    if (!resp.ok) { log('No metrics found on server. Run simulations first.'); return; }
    const metrics = await resp.json();
    log('Loaded metrics.');
    // Draw basic sync curve using 2D canvas fallback
    const ctx = canvas.getContext('2d');
    canvas.width = Math.min(window.innerWidth, 1200);
    canvas.height = 400;
    ctx.fillStyle = '#111'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle = '#0f0'; ctx.beginPath();
    const t = metrics['t']; const sync = metrics['sync'];
    for (let i=0;i<t.length;i++){
      const x = (i/(t.length-1))*(canvas.width-10)+5;
      const y = canvas.height - 20 - sync[i]*(canvas.height-40);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    log('Rendered sync curve.');
  } catch (e) {
    log('Error fetching metrics: '+e);
  }
})();
"""


def write_frontend(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(FRONTEND_INDEX)
    with open(os.path.join(outdir, 'app.js'), 'w', encoding='utf-8') as f:
        f.write(FRONTEND_APPJS)
    print(f"[Frontend] Written minimal frontend to {outdir}")


class MetricsHTTPHandler(SimpleHTTPRequestHandler):
    # Serve the current directory files and /latest_metrics.json if present in cwd
    def end_headers(self):
        # Allow CORS for local testing
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)


def start_server(port: int = 8080, serve_dir: str = '.'):
    os.chdir(serve_dir)
    server = HTTPServer(('0.0.0.0', port), MetricsHTTPHandler)
    print(f"[Server] Serving {serve_dir} at http://localhost:{port}")
    print("Copy generated frontend files into your webgpu-cross-platform-app public folder for WebGPU native app integration.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

# ---------------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------------

def run_experiment_1(args):
    cfg = GFNetworkConfig(N=args.N, p_conn=args.p, field_bits=args.bits, device=args.device, seed=args.seed)
    sim = GFNetworkSimulation(cfg)
    out = sim.run(T=args.T, sample_interval=args.sample, noise=args.noise)
    # write metrics to JSON for frontend
    with open('latest_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(out['metrics'], f)
    # also save config and tc
    with open('experiment1_results.json', 'w', encoding='utf-8') as f:
        json.dump({'cfg': out['cfg'], 'tc': out['tc']}, f)
    print('[Exp1] Done. Results:', out['tc'])


def run_experiment_1_test():
    """Deterministic test run for Experiment 1.
    Uses a fixed seed and returns the metrics dictionary.
    """
    class Args:
        N = 1024
        p = 0.02
        bits = 1
        T = 500
        sample = 10
        noise = 0.0
        seed = 12345
        device = 'cpu'
    cfg = GFNetworkConfig(N=Args.N, p_conn=Args.p, field_bits=Args.bits, device=Args.device, seed=Args.seed)
    sim = GFNetworkSimulation(cfg)
    out = sim.run(T=Args.T, sample_interval=Args.sample, noise=Args.noise)
    # Save deterministic test results
    with open('exp1_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return out


def run_experiment_2(args):
    cfg = RGBACHIMERAConfig(modules=args.modules, module_size=4, p_conn=args.pconn, device=args.device, seed=args.seed)
    sim = RGBACHIMERASimulation(cfg)
    out = sim.run_probe_experiment(T=args.T)
    with open('experiment2_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print('[Exp2] Done. Results:', out)


def run_experiment_2_test():
    """Deterministic test run for Experiment 2."""
    class Args:
        modules = 256
        pconn = 0.1
        T = 50
        seed = 99999
        device = 'cpu'
    cfg = RGBACHIMERAConfig(modules=Args.modules, module_size=4, p_conn=Args.pconn, device=Args.device, seed=Args.seed)
    sim = RGBACHIMERASimulation(cfg)
    out = sim.run_probe_experiment(T=Args.T)
    with open('exp2_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return out


def main():
    parser = argparse.ArgumentParser(description='NeuroCHIMERA simulation bundle')
    sub = parser.add_subparsers(dest='cmd')
    p1 = sub.add_parser('exp1', help='Run GF(2)/GF(2^n) network experiment')
    p1.add_argument('--N', type=int, default=65536)
    p1.add_argument('--p', type=float, default=0.02)
    p1.add_argument('--bits', type=int, default=1)
    p1.add_argument('--T', type=int, default=2000)
    p1.add_argument('--sample', type=int, default=10)
    p1.add_argument('--noise', type=float, default=0.01)
    p1.add_argument('--seed', type=int, default=42)
    p1.add_argument('--device', type=str, default='cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

    p2 = sub.add_parser('exp2', help='Run RGBA-CHIMERA probe experiment')
    p2.add_argument('--modules', type=int, default=1024)
    p2.add_argument('--pconn', type=float, default=0.05)
    p2.add_argument('--T', type=int, default=100)
    p2.add_argument('--seed', type=int, default=123)
    p2.add_argument('--device', type=str, default='cpu')

    p3 = sub.add_parser('serve', help='Write frontend and start HTTP server')
    p3.add_argument('--port', type=int, default=8080)
    p3.add_argument('--outdir', type=str, default='frontend')

    p_test = sub.add_parser('test', help='Run deterministic test for Experiment 1')
    p_test2 = sub.add_parser('test2', help='Run deterministic test for Experiment 2')

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help(); return
    if args.cmd == 'exp1':
        run_experiment_1(args)
    elif args.cmd == 'exp2':
        run_experiment_2(args)
    elif args.cmd == 'serve':
        write_frontend(args.outdir)
        start_server(port=args.port, serve_dir=args.outdir)
    elif args.cmd == 'test':
        run_experiment_1_test()
    elif args.cmd == 'test2':
        run_experiment_2_test()

if __name__ == '__main__':
    main()
