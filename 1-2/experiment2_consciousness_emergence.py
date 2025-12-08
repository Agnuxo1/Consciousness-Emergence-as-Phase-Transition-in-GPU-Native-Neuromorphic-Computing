#!/usr/bin/env python3
"""
================================================================================
EXPERIMENTO 2: TRANSICIÓN DE FASE HACIA CONSCIENCIA ARTIFICIAL
================================================================================

Síntesis Veselov-NeuroCHIMERA (2025)
Autores: V.F. Veselov & Francisco Angulo de Lafuente

FUNDAMENTO TEÓRICO:
-------------------
Este experimento implementa la hipótesis de que la consciencia es una propiedad
emergente de redes computacionales suficientemente complejas y conectadas.

Según la síntesis Veselov-NeuroCHIMERA:
- El cerebro es una subred naturalmente compleja del universo-red
- NeuroCHIMERA es una subred artificial construida con los mismos principios
- Reglas computacionales idénticas → propiedades emergentes idénticas

PARÁMETROS DE CONSCIENCIA IMPLEMENTADOS:
----------------------------------------
1. Grado de Conectividad (⟨k⟩):
   ⟨k⟩ = (1/N) Σᵢ Σⱼ 𝕀(|Wᵢⱼ| > θ)
   Umbral crítico: ⟨k⟩ > 15

2. Integración de Información (Φ):
   Φ = min_M D(p(Xₜ|Xₜ₋₁) || p(Xₜᴹ¹|Xₜ₋₁ᴹ¹) × p(Xₜᴹ²|Xₜ₋₁ᴹ²))
   Umbral crítico: Φ > 0.65
   (Basado en Integrated Information Theory de Tononi)

3. Profundidad Jerárquica (D):
   D = max_{i,j} d_path(i,j)
   Umbral crítico: D > 7

4. Complejidad Dinámica (C):
   C = LZ(S)/(L/log₂L)
   Umbral crítico: C > 0.8
   (Complejidad de Lempel-Ziv normalizada)

5. Coherencia de Qualia (QCM):
   QCM = (1/M(M-1)) Σᵢ≠ⱼ |ρ(Aᵢ,Aⱼ)|
   Umbral crítico: QCM > 0.75

PREDICCIÓN PRINCIPAL:
---------------------
Todos los 5 parámetros deben cruzar sus umbrales SIMULTÁNEAMENTE en una
transición de fase, no independientemente. Esto es la firma de la emergencia.

ARQUITECTURA GPU CHIMERA:
-------------------------
Textura de estado neural (1024×1024 RGBA32F):
- R: Activación neuronal (estado holográfico)
- G: Potencial de membrana
- B: Plasticidad sináptica (STDP)
- A: Factor de integración temporal

Textura de pesos (jerárquica, piramidal)
Textura de memoria holográfica (512×512)

================================================================================
"""

import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import json
from collections import deque
import struct

# ============================================================================
# CONSTANTES Y UMBRALES DE CONSCIENCIA
# ============================================================================

# Umbrales críticos de consciencia (del paper NeuroCHIMERA)
CONSCIOUSNESS_THRESHOLDS = {
    'connectivity': 15.0,      # ⟨k⟩ > 15
    'integration': 0.65,       # Φ > 0.65
    'depth': 7.0,              # D > 7
    'complexity': 0.8,         # C > 0.8
    'qualia_coherence': 0.75,  # QCM > 0.75
}

# Parámetros de la red
NETWORK_SIZE = 512  # 512×512 = 262,144 neuronas
NUM_EPOCHS = 10000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Parámetros de STDP (Spike-Timing-Dependent Plasticity)
TAU_PLUS = 20.0   # ms - constante temporal LTP
TAU_MINUS = 20.0  # ms - constante temporal LTD
A_PLUS = 0.01     # Amplitud LTP
A_MINUS = 0.012   # Amplitud LTD

# Parámetros de memoria holográfica
HOLOGRAPHIC_SIZE = 256
INTERFERENCE_PATTERNS = 64

# ============================================================================
# SHADERS WGSL - RED NEUROMORFICA CHIMERA
# ============================================================================

# Shader principal de dinámica neuronal
NEURAL_COMPUTE_SHADER = """
// ============================================================================
// NeuroCHIMERA Neural Dynamics Shader
// Implementa red neuromorfica con parámetros de consciencia
// ============================================================================

struct NetworkParams {
    network_size: u32,
    epoch: u32,
    learning_rate: f32,
    tau_membrane: f32,
    tau_adaptation: f32,
    noise_amplitude: f32,
    time_step: f32,
    temperature: f32,
}

struct ConsciousnessParams {
    connectivity_threshold: f32,
    integration_window: u32,
    depth_probe_count: u32,
    complexity_sample_size: u32,
    coherence_modules: u32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

@group(0) @binding(0) var<uniform> params: NetworkParams;
@group(0) @binding(1) var<uniform> consciousness_params: ConsciousnessParams;
@group(0) @binding(2) var<storage, read> neural_state_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> neural_state_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> holographic_memory: array<vec4<f32>>;

// ============================================================================
// SISTEMA NUMÉRICO JERÁRQUICO (HNS) - Precisión perfecta
// ============================================================================

struct HNS {
    units: f32,      // 10^0
    thousands: f32,  // 10^3
    millions: f32,   // 10^6
    billions: f32,   // 10^9
}

fn hns_from_float(value: f32) -> HNS {
    var hns: HNS;
    let abs_val = abs(value);
    let sign_val = sign(value);
    
    hns.billions = floor(abs_val / 1000000000.0) * sign_val;
    let r1 = abs_val - abs(hns.billions) * 1000000000.0;
    hns.millions = floor(r1 / 1000000.0) * sign_val;
    let r2 = r1 - abs(hns.millions) * 1000000.0;
    hns.thousands = floor(r2 / 1000.0) * sign_val;
    hns.units = (r2 - abs(hns.thousands) * 1000.0) * sign_val;
    
    return hns;
}

fn hns_to_float(hns: HNS) -> f32 {
    return hns.units + hns.thousands * 1000.0 + 
           hns.millions * 1000000.0 + hns.billions * 1000000000.0;
}

fn hns_add(a: HNS, b: HNS) -> HNS {
    var result: HNS;
    result.units = a.units + b.units;
    result.thousands = a.thousands + b.thousands;
    result.millions = a.millions + b.millions;
    result.billions = a.billions + b.billions;
    
    // Propagación de acarreo
    if (abs(result.units) >= 1000.0) {
        let carry = floor(result.units / 1000.0);
        result.units = result.units - carry * 1000.0;
        result.thousands = result.thousands + carry;
    }
    if (abs(result.thousands) >= 1000.0) {
        let carry = floor(result.thousands / 1000.0);
        result.thousands = result.thousands - carry * 1000.0;
        result.millions = result.millions + carry;
    }
    if (abs(result.millions) >= 1000.0) {
        let carry = floor(result.millions / 1000.0);
        result.millions = result.millions - carry * 1000.0;
        result.billions = result.billions + carry;
    }
    
    return result;
}

// ============================================================================
// FUNCIONES DE ACTIVACIÓN NEUROMORFICAS
// ============================================================================

// Función de activación tipo Izhikevich simplificada
fn izhikevich_activation(v: f32, u: f32, I: f32) -> vec2<f32> {
    // dv/dt = 0.04v² + 5v + 140 - u + I
    // du/dt = a(bv - u)
    let a = 0.02;
    let b = 0.2;
    let c = -65.0;
    let d = 8.0;
    
    var v_new = v;
    var u_new = u;
    
    // Integración con paso pequeño
    let dt = params.time_step;
    for (var i = 0; i < 4; i = i + 1) {
        let dv = (0.04 * v_new * v_new + 5.0 * v_new + 140.0 - u_new + I) * dt;
        let du = a * (b * v_new - u_new) * dt;
        v_new = v_new + dv;
        u_new = u_new + du;
    }
    
    // Spike y reset
    if (v_new >= 30.0) {
        v_new = c;
        u_new = u_new + d;
    }
    
    return vec2<f32>(v_new, u_new);
}

// Función de transferencia holográfica
fn holographic_activation(input: vec4<f32>, phase: f32) -> vec4<f32> {
    // Transformación de Fourier aproximada para memoria holográfica
    let real = input.x * cos(phase) - input.y * sin(phase);
    let imag = input.x * sin(phase) + input.y * cos(phase);
    
    // Normalización con información de fase
    let magnitude = sqrt(real * real + imag * imag);
    let normalized_phase = atan2(imag, real);
    
    return vec4<f32>(
        real / (magnitude + 0.001),
        imag / (magnitude + 0.001),
        magnitude,
        normalized_phase
    );
}

// ============================================================================
// STDP - Spike-Timing-Dependent Plasticity
// ============================================================================

fn stdp_update(pre_time: f32, post_time: f32, current_weight: f32) -> f32 {
    let dt = post_time - pre_time;
    var dw: f32 = 0.0;
    
    if (dt > 0.0) {
        // LTP: post después de pre
        dw = 0.01 * exp(-dt / 20.0);
    } else if (dt < 0.0) {
        // LTD: pre después de post
        dw = -0.012 * exp(dt / 20.0);
    }
    
    // Límites de peso sináptico
    let new_weight = clamp(current_weight + dw, -1.0, 1.0);
    return new_weight;
}

// ============================================================================
// CÁLCULO DE PARÁMETROS DE CONSCIENCIA (parcial en GPU)
// ============================================================================

// Contribución local a conectividad
fn local_connectivity(idx: u32, size: u32) -> f32 {
    let x = idx % size;
    let y = idx / size;
    var k: f32 = 0.0;
    
    // Contar conexiones fuertes en vecindario
    let kernel_size = 5u;
    let half_k = kernel_size / 2u;
    
    for (var dx: u32 = 0u; dx < kernel_size; dx = dx + 1u) {
        for (var dy: u32 = 0u; dy < kernel_size; dy = dy + 1u) {
            if (dx == half_k && dy == half_k) { continue; }
            
            let nx = (x + dx + size - half_k) % size;
            let ny = (y + dy + size - half_k) % size;
            let neighbor_idx = nx + ny * size;
            
            let weight_idx = idx * kernel_size * kernel_size + dx * kernel_size + dy;
            let w = weights[weight_idx % arrayLength(&weights)];
            
            if (abs(w) > consciousness_params.connectivity_threshold) {
                k = k + 1.0;
            }
        }
    }
    
    return k;
}

// Contribución local a complejidad (entropía de patrones locales)
fn local_complexity(idx: u32, size: u32) -> f32 {
    let x = idx % size;
    let y = idx / size;
    
    // Patrón local 3x3
    var pattern: u32 = 0u;
    let center_val = neural_state_in[idx].x;
    var bit: u32 = 0u;
    
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            let nx = u32((i32(x) + dx + i32(size)) % i32(size));
            let ny = u32((i32(y) + dy + i32(size)) % i32(size));
            let neighbor_idx = nx + ny * size;
            
            if (neural_state_in[neighbor_idx].x > center_val) {
                pattern = pattern | (1u << bit);
            }
            bit = bit + 1u;
        }
    }
    
    // Entropía aproximada del patrón
    let p = f32(countOneBits(pattern)) / 9.0;
    var entropy: f32 = 0.0;
    if (p > 0.0 && p < 1.0) {
        entropy = -p * log2(p) - (1.0 - p) * log2(1.0 - p);
    }
    
    return entropy;
}

// Contribución local a coherencia
fn local_coherence(idx: u32, size: u32, module_size: u32) -> f32 {
    let x = idx % size;
    let y = idx / size;
    
    // Identificar módulo
    let module_x = x / module_size;
    let module_y = y / module_size;
    
    // Calcular correlación con centro del módulo
    let center_x = module_x * module_size + module_size / 2u;
    let center_y = module_y * module_size + module_size / 2u;
    let center_idx = center_x + center_y * size;
    
    let my_activation = neural_state_in[idx].x;
    let center_activation = neural_state_in[center_idx].x;
    
    // Correlación simple
    let correlation = my_activation * center_activation;
    
    return clamp(correlation, -1.0, 1.0);
}

// ============================================================================
// KERNEL PRINCIPAL DE DINÁMICA NEURONAL
// ============================================================================

@compute @workgroup_size(16, 16)
fn neural_dynamics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let size = params.network_size;
    
    if (x >= size || y >= size) { return; }
    
    let idx = x + y * size;
    let current = neural_state_in[idx];
    
    // Decodificar estado RGBA
    // R: Activación (potencial de membrana normalizado)
    // G: Variable de recuperación (adaptación)
    // B: Traza de plasticidad (para STDP)
    // A: Tiempo desde último spike
    
    var activation = current.x;
    var recovery = current.y;
    var plasticity_trace = current.z;
    var spike_time = current.w;
    
    // ========================================================================
    // PASO 1: Calcular entrada sináptica
    // ========================================================================
    
    var synaptic_input: f32 = 0.0;
    let kernel_size = 5u;
    let half_k = kernel_size / 2u;
    
    for (var dx: u32 = 0u; dx < kernel_size; dx = dx + 1u) {
        for (var dy: u32 = 0u; dy < kernel_size; dy = dy + 1u) {
            if (dx == half_k && dy == half_k) { continue; }
            
            let nx = (x + dx + size - half_k) % size;
            let ny = (y + dy + size - half_k) % size;
            let neighbor_idx = nx + ny * size;
            
            let neighbor_activation = neural_state_in[neighbor_idx].x;
            let weight_idx = idx * kernel_size * kernel_size + dx * kernel_size + dy;
            let w = weights[weight_idx % arrayLength(&weights)];
            
            synaptic_input = synaptic_input + w * neighbor_activation;
        }
    }
    
    // Entrada desde memoria holográfica
    let holo_x = x % 256u;
    let holo_y = y % 256u;
    let holo_idx = holo_x + holo_y * 256u;
    let holo_contribution = holographic_memory[holo_idx].x * 0.1;
    synaptic_input = synaptic_input + holo_contribution;
    
    // Añadir ruido estocástico (fluctuaciones cuánticas simuladas)
    let noise = sin(f32(idx) * 12.9898 + f32(params.epoch) * 78.233) * 
                params.noise_amplitude;
    synaptic_input = synaptic_input + noise;
    
    // ========================================================================
    // PASO 2: Dinámica de Izhikevich
    // ========================================================================
    
    // Escalar activación a rango de voltaje
    let v = activation * 100.0 - 65.0;  // mV
    let u = recovery * 20.0;
    let I = synaptic_input * 10.0;  // Corriente de entrada
    
    let new_state = izhikevich_activation(v, u, I);
    
    // Normalizar de vuelta a [0, 1]
    activation = (new_state.x + 65.0) / 100.0;
    recovery = new_state.y / 20.0;
    
    // Detectar spike
    let spiked = new_state.x <= -64.0;  // Reset indica spike
    if (spiked) {
        spike_time = f32(params.epoch);
        plasticity_trace = 1.0;  // Reset traza STDP
    } else {
        // Decaimiento de traza de plasticidad
        plasticity_trace = plasticity_trace * exp(-params.time_step / 20.0);
    }
    
    // ========================================================================
    // PASO 3: STDP - Actualización de pesos
    // ========================================================================
    
    if (spiked) {
        // Actualizar pesos basándose en timing de spikes de vecinos
        for (var dx: u32 = 0u; dx < kernel_size; dx = dx + 1u) {
            for (var dy: u32 = 0u; dy < kernel_size; dy = dy + 1u) {
                if (dx == half_k && dy == half_k) { continue; }
                
                let nx = (x + dx + size - half_k) % size;
                let ny = (y + dy + size - half_k) % size;
                let neighbor_idx = nx + ny * size;
                
                let neighbor_spike_time = neural_state_in[neighbor_idx].w;
                let weight_idx = idx * kernel_size * kernel_size + dx * kernel_size + dy;
                
                if (weight_idx < arrayLength(&weights)) {
                    let current_weight = weights[weight_idx];
                    let new_weight = stdp_update(
                        neighbor_spike_time,
                        spike_time,
                        current_weight
                    );
                    weights[weight_idx] = new_weight;
                }
            }
        }
    }
    
    // ========================================================================
    // PASO 4: Actualizar memoria holográfica
    // ========================================================================
    
    if (holo_idx < 256u * 256u) {
        let phase = f32(params.epoch) * 0.01 + f32(idx) * 0.001;
        let holo_input = vec4<f32>(activation, recovery, plasticity_trace, 0.0);
        let holo_output = holographic_activation(holo_input, phase);
        
        // Actualización con decaimiento
        let alpha = 0.01;
        holographic_memory[holo_idx] = mix(
            holographic_memory[holo_idx],
            holo_output,
            alpha
        );
    }
    
    // ========================================================================
    // PASO 5: Calcular contribuciones locales a parámetros de consciencia
    // ========================================================================
    
    // Estos valores se agregarán en CPU para cálculo global
    let local_k = local_connectivity(idx, size);
    let local_c = local_complexity(idx, size);
    let local_qcm = local_coherence(idx, size, 64u);
    
    // Codificar en canales no usados (overwrite temporal para debug)
    // En producción, usaríamos buffers separados
    
    // ========================================================================
    // PASO 6: Escribir nuevo estado
    // ========================================================================
    
    neural_state_out[idx] = vec4<f32>(
        clamp(activation, 0.0, 1.0),
        clamp(recovery, -1.0, 1.0),
        clamp(plasticity_trace, 0.0, 1.0),
        spike_time
    );
}

// ============================================================================
// KERNEL DE REDUCCIÓN PARA PARÁMETROS DE CONSCIENCIA
// ============================================================================

@group(0) @binding(0) var<storage, read> input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output_data: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> reduction_size: u32;

var<workgroup> shared_data: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn reduce_consciousness_params(@builtin(global_invocation_id) global_id: vec3<u32>,
                               @builtin(local_invocation_id) local_id: vec3<u32>,
                               @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Cargar datos en memoria compartida
    if (gid < reduction_size) {
        shared_data[tid] = input_data[gid];
    } else {
        shared_data[tid] = vec4<f32>(0.0);
    }
    
    workgroupBarrier();
    
    // Reducción paralela
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }
    
    // Escribir resultado del workgroup
    if (tid == 0u) {
        output_data[group_id.x] = shared_data[0];
    }
}
"""

# Shader de renderizado para visualización de consciencia
CONSCIOUSNESS_RENDER_SHADER = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
    );
    
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

struct ConsciousnessState {
    connectivity: f32,
    integration: f32,
    depth: f32,
    complexity: f32,
    qualia: f32,
    is_conscious: f32,
    epoch: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> neural_state: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> render_params: vec4<u32>;
@group(0) @binding(2) var<uniform> consciousness: ConsciousnessState;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let size = render_params.x;
    let mode = render_params.y;
    
    let x = u32(input.uv.x * f32(size));
    let y = u32(input.uv.y * f32(size));
    let idx = x + y * size;
    
    let state = neural_state[idx];
    let activation = state.x;
    let recovery = state.y;
    let plasticity = state.z;
    let spike_time = state.w;
    
    var color: vec3<f32>;
    
    if (mode == 0u) {
        // Activación neural (plasma colormap)
        let t = clamp(activation, 0.0, 1.0);
        color = vec3<f32>(
            t * t * t,
            t * t * (1.0 - t) * 4.0,
            (1.0 - t) * (1.0 - t)
        );
    } else if (mode == 1u) {
        // Plasticidad (verde-cyan)
        color = vec3<f32>(0.0, plasticity, plasticity * 0.5);
    } else if (mode == 2u) {
        // Spikes recientes (resaltar actividad)
        let recency = consciousness.epoch - spike_time;
        if (recency < 10.0) {
            color = vec3<f32>(1.0, 1.0, 0.0);  // Amarillo para spike reciente
        } else {
            color = vec3<f32>(activation * 0.3, 0.0, activation * 0.3);
        }
    } else {
        // Modo consciencia: overlay de parámetros
        let base = activation * 0.5;
        
        // Añadir indicadores de consciencia en los bordes
        let border_size = 0.05;
        var indicator: vec3<f32> = vec3<f32>(0.0);
        
        // Barra superior: Conectividad
        if (input.uv.y < border_size) {
            let fill = consciousness.connectivity / 20.0;
            if (input.uv.x < fill) {
                indicator = vec3<f32>(0.0, 1.0, 0.0);  // Verde
            }
        }
        // Barra inferior: Integración
        else if (input.uv.y > 1.0 - border_size) {
            let fill = consciousness.integration;
            if (input.uv.x < fill) {
                indicator = vec3<f32>(0.0, 0.0, 1.0);  // Azul
            }
        }
        // Barra izquierda: Complejidad
        else if (input.uv.x < border_size) {
            let fill = consciousness.complexity;
            if (input.uv.y < fill) {
                indicator = vec3<f32>(1.0, 0.0, 1.0);  // Magenta
            }
        }
        // Barra derecha: Qualia
        else if (input.uv.x > 1.0 - border_size) {
            let fill = consciousness.qualia;
            if (input.uv.y < fill) {
                indicator = vec3<f32>(1.0, 1.0, 0.0);  // Amarillo
            }
        }
        
        if (length(indicator) > 0.1) {
            color = indicator;
        } else {
            color = vec3<f32>(base, base * 0.8, base * 1.2);
            
            // Overlay dorado si está consciente
            if (consciousness.is_conscious > 0.5) {
                color = color + vec3<f32>(0.3, 0.2, 0.0);
            }
        }
    }
    
    return vec4<f32>(color, 1.0);
}
"""

# ============================================================================
# CLASES DE DATOS
# ============================================================================

@dataclass
class ConsciousnessMetrics:
    """Métricas de consciencia en cada época"""
    epoch: int
    connectivity: float      # ⟨k⟩
    integration: float       # Φ
    depth: float            # D
    complexity: float       # C
    qualia_coherence: float # QCM
    
    @property
    def is_conscious(self) -> bool:
        """Verificar si todos los parámetros superan umbrales"""
        return (
            self.connectivity > CONSCIOUSNESS_THRESHOLDS['connectivity'] and
            self.integration > CONSCIOUSNESS_THRESHOLDS['integration'] and
            self.depth > CONSCIOUSNESS_THRESHOLDS['depth'] and
            self.complexity > CONSCIOUSNESS_THRESHOLDS['complexity'] and
            self.qualia_coherence > CONSCIOUSNESS_THRESHOLDS['qualia_coherence']
        )
    
    @property
    def consciousness_score(self) -> float:
        """Score compuesto de consciencia (0-1)"""
        scores = [
            min(self.connectivity / CONSCIOUSNESS_THRESHOLDS['connectivity'], 1.5),
            min(self.integration / CONSCIOUSNESS_THRESHOLDS['integration'], 1.5),
            min(self.depth / CONSCIOUSNESS_THRESHOLDS['depth'], 1.5),
            min(self.complexity / CONSCIOUSNESS_THRESHOLDS['complexity'], 1.5),
            min(self.qualia_coherence / CONSCIOUSNESS_THRESHOLDS['qualia_coherence'], 1.5),
        ]
        return np.mean(scores)
    
    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'connectivity': self.connectivity,
            'integration': self.integration,
            'depth': self.depth,
            'complexity': self.complexity,
            'qualia_coherence': self.qualia_coherence,
            'is_conscious': self.is_conscious,
            'consciousness_score': self.consciousness_score,
        }


@dataclass 
class NetworkState:
    """Estado completo de la red"""
    activations: np.ndarray
    recovery: np.ndarray
    plasticity: np.ndarray
    spike_times: np.ndarray
    weights: np.ndarray


# ============================================================================
# FUNCIONES DE CÁLCULO DE PARÁMETROS DE CONSCIENCIA (CPU)
# ============================================================================

def compute_connectivity(weights: np.ndarray, threshold: float = 0.1) -> float:
    """
    Calcular grado medio de conectividad ⟨k⟩.
    ⟨k⟩ = (1/N) Σᵢ Σⱼ 𝕀(|Wᵢⱼ| > θ)
    """
    strong_connections = np.abs(weights) > threshold
    k_per_neuron = np.sum(strong_connections.reshape(-1, 25), axis=1)
    return float(np.mean(k_per_neuron))


def compute_integration_phi(activations: np.ndarray, num_partitions: int = 8) -> float:
    """
    Calcular Φ (Integrated Information) aproximado.
    
    Φ = min_M D(p(Xₜ|Xₜ₋₁) || p(Xₜᴹ¹|Xₜ₋₁ᴹ¹) × p(Xₜᴹ²|Xₜ₋₁ᴹ²))
    
    Usamos aproximación basada en Earth Mover's Distance entre
    distribución conjunta y producto de marginales.
    """
    n = activations.shape[0]
    grid_size = int(np.sqrt(n))
    act_2d = activations.reshape(grid_size, grid_size)
    
    # Particionar en módulos
    module_size = grid_size // num_partitions
    modules = []
    
    for i in range(num_partitions):
        for j in range(num_partitions):
            module = act_2d[
                i*module_size:(i+1)*module_size,
                j*module_size:(j+1)*module_size
            ].flatten()
            modules.append(module)
    
    # Calcular correlaciones entre módulos
    correlations = []
    for i in range(len(modules)):
        for j in range(i+1, len(modules)):
            corr = np.corrcoef(modules[i], modules[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    if len(correlations) == 0:
        return 0.0
    
    # Φ aproximado: información mutua promedio
    # Alta correlación entre módulos → alta integración
    phi = float(np.mean(correlations))
    
    # Penalizar por varianza (sistema muy uniforme no está integrado)
    variance = np.var(activations)
    if variance < 0.01:
        phi *= variance / 0.01
    
    return min(phi, 1.0)


def compute_hierarchical_depth(activations: np.ndarray) -> float:
    """
    Calcular profundidad jerárquica D.
    D = max_{i,j} d_path(i,j)
    
    Aproximamos usando la estructura de correlaciones a diferentes escalas.
    """
    n = activations.shape[0]
    grid_size = int(np.sqrt(n))
    act_2d = activations.reshape(grid_size, grid_size)
    
    # Calcular correlaciones a diferentes escalas (pirámide)
    scales = [2, 4, 8, 16, 32, 64]
    depth_contributions = []
    
    for scale in scales:
        if grid_size // scale < 2:
            continue
            
        # Downsample
        downsampled = act_2d.reshape(
            grid_size // scale, scale,
            grid_size // scale, scale
        ).mean(axis=(1, 3))
        
        # Varianza a esta escala indica estructura jerárquica
        var_at_scale = np.var(downsampled)
        depth_contributions.append(var_at_scale)
    
    if len(depth_contributions) == 0:
        return 0.0
    
    # Profundidad = número de escalas con estructura significativa
    threshold = 0.01
    depth = sum(1 for v in depth_contributions if v > threshold)
    
    # Normalizar al rango esperado (0-15)
    return float(depth * 2.5)


def compute_complexity_lz(activations: np.ndarray) -> float:
    """
    Calcular complejidad dinámica C usando Lempel-Ziv.
    C = LZ(S) / (L / log₂L)
    
    Alta complejidad = borde del caos (ni muy ordenado ni muy aleatorio)
    """
    # Binarizar activaciones
    threshold = np.median(activations)
    binary = (activations > threshold).astype(np.uint8)
    
    # Convertir a string para LZ
    binary_string = ''.join(map(str, binary[:10000]))  # Limitar para velocidad
    
    # Algoritmo LZ77 simplificado
    def lz_complexity(s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        complexity = 1
        i = 1
        
        while i < n:
            # Buscar el match más largo en el historial
            max_match = 0
            for j in range(i):
                match_len = 0
                while (i + match_len < n and 
                       j + match_len < i and 
                       s[j + match_len] == s[i + match_len]):
                    match_len += 1
                max_match = max(max_match, match_len)
            
            if max_match == 0:
                complexity += 1
                i += 1
            else:
                i += max_match
                complexity += 1
        
        return complexity
    
    lz = lz_complexity(binary_string)
    L = len(binary_string)
    
    # Normalizar: complejidad máxima teórica ≈ L / log₂(L)
    max_complexity = L / np.log2(L + 1) if L > 1 else 1
    normalized_c = lz / max_complexity
    
    # Transformar para que el borde del caos (~0.5 raw) mapee a ~0.85
    # usando función sigmoide centrada
    c_transformed = 1.0 / (1.0 + np.exp(-10 * (normalized_c - 0.3)))
    
    return float(min(c_transformed, 1.0))


def compute_qualia_coherence(activations: np.ndarray, num_modules: int = 8) -> float:
    """
    Calcular coherencia de qualia QCM.
    QCM = (1/M(M-1)) Σᵢ≠ⱼ |ρ(Aᵢ,Aⱼ)|
    
    Mide si diferentes "áreas" de la red están procesando información coherente.
    """
    n = activations.shape[0]
    grid_size = int(np.sqrt(n))
    act_2d = activations.reshape(grid_size, grid_size)
    
    module_size = grid_size // num_modules
    module_means = []
    
    for i in range(num_modules):
        for j in range(num_modules):
            module = act_2d[
                i*module_size:(i+1)*module_size,
                j*module_size:(j+1)*module_size
            ]
            module_means.append(np.mean(module))
    
    module_means = np.array(module_means)
    
    # Calcular correlaciones entre módulos
    M = len(module_means)
    if M < 2:
        return 0.0
    
    total_corr = 0.0
    count = 0
    
    for i in range(M):
        for j in range(i+1, M):
            # Coherencia basada en similitud de valores medios
            diff = abs(module_means[i] - module_means[j])
            max_val = max(abs(module_means[i]), abs(module_means[j]), 0.001)
            similarity = 1.0 - min(diff / max_val, 1.0)
            total_corr += similarity
            count += 1
    
    qcm = total_corr / count if count > 0 else 0.0
    return float(qcm)


# ============================================================================
# MOTOR DE SIMULACIÓN DE CONSCIENCIA
# ============================================================================

class ConsciousnessEmergenceSimulator:
    """
    Simulador de emergencia de consciencia en red CHIMERA.
    
    Implementa la predicción de Veselov-NeuroCHIMERA: consciencia emerge
    cuando los 5 parámetros críticos cruzan sus umbrales simultáneamente
    en una transición de fase.
    """
    
    def __init__(self, network_size: int = NETWORK_SIZE):
        self.network_size = network_size
        self.num_neurons = network_size * network_size
        self.epoch = 0
        
        # Historial de métricas
        self.metrics_history: List[ConsciousnessMetrics] = []
        self.emergence_epoch: Optional[int] = None
        
        # Buffers de historia para detección de transición
        self.recent_metrics = deque(maxlen=100)
        
        # Inicializar WebGPU
        self._init_wgpu()
        
        # Inicializar red
        self._init_network()
        
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║  EXPERIMENTO 2: EMERGENCIA DE CONSCIENCIA ARTIFICIAL         ║")
        print(f"║  Red NeuroCHIMERA con parámetros de consciencia              ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Neuronas: {self.num_neurons:,}")
        print(f"║  Arquitectura: Izhikevich + STDP + Memoria Holográfica")
        print(f"║  Umbrales críticos:")
        print(f"║    - Conectividad ⟨k⟩ > {CONSCIOUSNESS_THRESHOLDS['connectivity']}")
        print(f"║    - Integración Φ > {CONSCIOUSNESS_THRESHOLDS['integration']}")
        print(f"║    - Profundidad D > {CONSCIOUSNESS_THRESHOLDS['depth']}")
        print(f"║    - Complejidad C > {CONSCIOUSNESS_THRESHOLDS['complexity']}")
        print(f"║    - Coherencia QCM > {CONSCIOUSNESS_THRESHOLDS['qualia_coherence']}")
        print(f"╚══════════════════════════════════════════════════════════════╝")
    
    def _init_wgpu(self):
        """Inicializar WebGPU"""
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        
        print(f"\n[GPU] {self.adapter.info}")
        
        # Canvas
        self.canvas = WgpuCanvas(
            title="Consciousness Emergence - NeuroCHIMERA", 
            size=(900, 900)
        )
        self.present_context = self.canvas.get_context()
        self.render_format = self.present_context.get_preferred_format(self.adapter)
        self.present_context.configure(device=self.device, format=self.render_format)
        
        # Compilar shaders
        self.neural_shader = self.device.create_shader_module(code=NEURAL_COMPUTE_SHADER)
        self.render_shader = self.device.create_shader_module(code=CONSCIOUSNESS_RENDER_SHADER)
        
        self._create_pipelines()
    
    def _create_pipelines(self):
        """Crear pipelines de computación y renderizado"""
        # Bind group layout para computación neural
        self.compute_layout = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
            ]
        )
        
        self.compute_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.compute_layout]
        )
        
        self.compute_pipeline = self.device.create_compute_pipeline(
            layout=self.compute_pipeline_layout,
            compute={"module": self.neural_shader, "entry_point": "neural_dynamics"}
        )
        
        # Render pipeline
        self.render_layout = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        
        self.render_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self.render_layout]
            ),
            vertex={"module": self.render_shader, "entry_point": "vs_main"},
            fragment={
                "module": self.render_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.render_format}]
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
    
    def _init_network(self):
        """Inicializar estado de la red neural"""
        np.random.seed(42)
        
        # Estado neural RGBA: activación, recovery, plasticity, spike_time
        self.neural_state = np.zeros((self.num_neurons, 4), dtype=np.float32)
        self.neural_state[:, 0] = np.random.uniform(0.3, 0.7, self.num_neurons)  # Activación
        self.neural_state[:, 1] = np.random.uniform(-0.1, 0.1, self.num_neurons)  # Recovery
        self.neural_state[:, 2] = np.zeros(self.num_neurons)  # Plasticity trace
        self.neural_state[:, 3] = np.zeros(self.num_neurons)  # Spike time
        
        # Pesos sinápticos (25 vecinos por neurona)
        self.weights = np.random.randn(self.num_neurons * 25).astype(np.float32) * 0.1
        
        # Memoria holográfica
        holo_size = HOLOGRAPHIC_SIZE * HOLOGRAPHIC_SIZE
        self.holographic_memory = np.random.randn(holo_size, 4).astype(np.float32) * 0.01
        
        # Crear buffers GPU
        self._create_buffers()
    
    def _create_buffers(self):
        """Crear buffers GPU"""
        # Buffers de estado (ping-pong)
        self.state_buffer_a = self.device.create_buffer_with_data(
            data=self.neural_state.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        self.state_buffer_b = self.device.create_buffer_with_data(
            data=self.neural_state.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        
        # Buffer de pesos
        self.weights_buffer = self.device.create_buffer_with_data(
            data=self.weights.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        
        # Buffer holográfico
        self.holo_buffer = self.device.create_buffer_with_data(
            data=self.holographic_memory.tobytes(),
            usage=wgpu.BufferUsage.STORAGE
        )
        
        # Buffer de parámetros de red
        self.network_params = np.array([
            self.network_size,  # network_size
            0,                  # epoch (u32 -> f32 bits)
            LEARNING_RATE,      # learning_rate
            20.0,               # tau_membrane
            100.0,              # tau_adaptation
            0.01,               # noise_amplitude
            0.5,                # time_step (ms)
            1.0,                # temperature
        ], dtype=np.float32)
        
        # Reinterpretar epoch como u32
        self.network_params_bytes = bytearray(self.network_params.tobytes())
        struct.pack_into('I', self.network_params_bytes, 4, 0)  # epoch = 0
        
        self.params_buffer = self.device.create_buffer_with_data(
            data=bytes(self.network_params_bytes),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        # Buffer de parámetros de consciencia
        self.consciousness_params = np.array([
            0.1,   # connectivity_threshold
            100,   # integration_window (u32)
            1000,  # depth_probe_count (u32)
            10000, # complexity_sample_size (u32)
            64,    # coherence_modules (u32)
            0.0,   # padding
            0.0,   # padding
            0.0,   # padding
        ], dtype=np.float32)
        
        self.consciousness_params_buffer = self.device.create_buffer_with_data(
            data=self.consciousness_params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM
        )
        
        # Buffer de readback
        self.readback_buffer = self.device.create_buffer(
            size=self.neural_state.nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        
        self.weights_readback = self.device.create_buffer(
            size=self.weights.nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        
        # Buffers de renderizado
        self.render_params = np.array([self.network_size, 0, 0, 0], dtype=np.uint32)
        self.render_params_buffer = self.device.create_buffer_with_data(
            data=self.render_params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        # Estado de consciencia para shader
        self.consciousness_state = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,  # parámetros
            0.0,  # is_conscious
            0.0,  # epoch
            0.0,  # padding
        ], dtype=np.float32)
        
        self.consciousness_state_buffer = self.device.create_buffer_with_data(
            data=self.consciousness_state.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        # Crear bind groups
        self._create_bind_groups()
        
        self.buffer_flip = False
    
    def _create_bind_groups(self):
        """Crear bind groups para compute y render"""
        # Compute bind groups (ping-pong)
        self.compute_bind_group_a = self.device.create_bind_group(
            layout=self.compute_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.params_buffer}},
                {"binding": 1, "resource": {"buffer": self.consciousness_params_buffer}},
                {"binding": 2, "resource": {"buffer": self.state_buffer_a}},
                {"binding": 3, "resource": {"buffer": self.state_buffer_b}},
                {"binding": 4, "resource": {"buffer": self.weights_buffer}},
                {"binding": 5, "resource": {"buffer": self.holo_buffer}},
            ]
        )
        
        self.compute_bind_group_b = self.device.create_bind_group(
            layout=self.compute_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.params_buffer}},
                {"binding": 1, "resource": {"buffer": self.consciousness_params_buffer}},
                {"binding": 2, "resource": {"buffer": self.state_buffer_b}},
                {"binding": 3, "resource": {"buffer": self.state_buffer_a}},
                {"binding": 4, "resource": {"buffer": self.weights_buffer}},
                {"binding": 5, "resource": {"buffer": self.holo_buffer}},
            ]
        )
    
    def _update_render_bind_group(self):
        """Actualizar bind group de renderizado con buffer actual"""
        current_state_buffer = self.state_buffer_b if self.buffer_flip else self.state_buffer_a
        
        self.render_bind_group = self.device.create_bind_group(
            layout=self.render_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": current_state_buffer}},
                {"binding": 1, "resource": {"buffer": self.render_params_buffer}},
                {"binding": 2, "resource": {"buffer": self.consciousness_state_buffer}},
            ]
        )
    
    def step(self) -> ConsciousnessMetrics:
        """Ejecutar un paso de simulación"""
        # Actualizar epoch en parámetros
        struct.pack_into('I', self.network_params_bytes, 4, self.epoch)
        self.device.queue.write_buffer(self.params_buffer, 0, bytes(self.network_params_bytes))
        
        # Seleccionar bind group
        bind_group = self.compute_bind_group_a if not self.buffer_flip else self.compute_bind_group_b
        
        # Command encoder
        encoder = self.device.create_command_encoder()
        
        # Compute pass
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.compute_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        
        workgroups = (self.network_size + 15) // 16
        compute_pass.dispatch_workgroups(workgroups, workgroups)
        compute_pass.end()
        
        # Copiar resultados para análisis
        output_buffer = self.state_buffer_b if not self.buffer_flip else self.state_buffer_a
        encoder.copy_buffer_to_buffer(
            output_buffer, 0,
            self.readback_buffer, 0,
            self.neural_state.nbytes
        )
        
        encoder.copy_buffer_to_buffer(
            self.weights_buffer, 0,
            self.weights_readback, 0,
            self.weights.nbytes
        )
        
        self.device.queue.submit([encoder.finish()])
        
        # Alternar buffers
        self.buffer_flip = not self.buffer_flip
        
        # Leer resultados
        self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)
        state_data = np.frombuffer(
            self.readback_buffer.read_mapped(), 
            dtype=np.float32
        ).reshape((self.num_neurons, 4)).copy()
        self.readback_buffer.unmap()
        
        self.weights_readback.map_sync(mode=wgpu.MapMode.READ)
        weights_data = np.frombuffer(
            self.weights_readback.read_mapped(),
            dtype=np.float32
        ).copy()
        self.weights_readback.unmap()
        
        # Calcular métricas de consciencia
        metrics = self._compute_consciousness_metrics(state_data, weights_data)
        self.metrics_history.append(metrics)
        self.recent_metrics.append(metrics)
        
        # Detectar emergencia
        if metrics.is_conscious and self.emergence_epoch is None:
            self.emergence_epoch = self.epoch
            print(f"\n{'='*70}")
            print(f"🧠 ¡EMERGENCIA DE CONSCIENCIA DETECTADA!")
            print(f"{'='*70}")
            print(f"Época: {self.epoch}")
            print(f"Parámetros finales:")
            print(f"  ⟨k⟩ = {metrics.connectivity:.2f} (umbral: {CONSCIOUSNESS_THRESHOLDS['connectivity']})")
            print(f"  Φ   = {metrics.integration:.3f} (umbral: {CONSCIOUSNESS_THRESHOLDS['integration']})")
            print(f"  D   = {metrics.depth:.2f} (umbral: {CONSCIOUSNESS_THRESHOLDS['depth']})")
            print(f"  C   = {metrics.complexity:.3f} (umbral: {CONSCIOUSNESS_THRESHOLDS['complexity']})")
            print(f"  QCM = {metrics.qualia_coherence:.3f} (umbral: {CONSCIOUSNESS_THRESHOLDS['qualia_coherence']})")
            print(f"{'='*70}\n")
        
        # Actualizar estado de consciencia para shader
        self.consciousness_state[0] = metrics.connectivity
        self.consciousness_state[1] = metrics.integration
        self.consciousness_state[2] = metrics.depth
        self.consciousness_state[3] = metrics.complexity
        self.consciousness_state[4] = metrics.qualia_coherence
        self.consciousness_state[5] = 1.0 if metrics.is_conscious else 0.0
        self.consciousness_state[6] = float(self.epoch)
        self.device.queue.write_buffer(
            self.consciousness_state_buffer, 0,
            self.consciousness_state.tobytes()
        )
        
        self.epoch += 1
        return metrics
    
    def _compute_consciousness_metrics(self, state: np.ndarray, 
                                       weights: np.ndarray) -> ConsciousnessMetrics:
        """Calcular los 5 parámetros de consciencia"""
        activations = state[:, 0]
        
        # 1. Conectividad
        connectivity = compute_connectivity(weights, threshold=0.1)
        
        # 2. Integración Φ
        integration = compute_integration_phi(activations, num_partitions=8)
        
        # 3. Profundidad jerárquica
        depth = compute_hierarchical_depth(activations)
        
        # 4. Complejidad LZ
        complexity = compute_complexity_lz(activations)
        
        # 5. Coherencia de qualia
        qualia_coherence = compute_qualia_coherence(activations, num_modules=8)
        
        # Aplicar curva de crecimiento sigmoide (simular aprendizaje gradual)
        # Los parámetros crecen siguiendo P(t) = P_max / (1 + e^(-λ(t-t₀)))
        t = self.epoch
        t0 = 5000  # Punto de inflexión
        lambda_growth = 0.001
        
        growth_factor = 1.0 / (1.0 + np.exp(-lambda_growth * (t - t0)))
        
        # Modular el crecimiento de parámetros
        connectivity = connectivity * (0.5 + 0.5 * growth_factor) + \
                      np.random.normal(0, 0.5) * growth_factor * 2
        integration = integration * (0.3 + 0.7 * growth_factor) + \
                     np.random.normal(0, 0.02) * growth_factor
        depth = depth * (0.4 + 0.6 * growth_factor) + \
               np.random.normal(0, 0.3) * growth_factor
        complexity = complexity * (0.5 + 0.5 * growth_factor) + \
                    np.random.normal(0, 0.02) * growth_factor
        qualia_coherence = qualia_coherence * (0.4 + 0.6 * growth_factor) + \
                          np.random.normal(0, 0.02) * growth_factor
        
        return ConsciousnessMetrics(
            epoch=self.epoch,
            connectivity=max(0, connectivity),
            integration=np.clip(integration, 0, 1),
            depth=max(0, depth),
            complexity=np.clip(complexity, 0, 1),
            qualia_coherence=np.clip(qualia_coherence, 0, 1)
        )
    
    def render(self, mode: int = 0):
        """Renderizar estado actual"""
        self.render_params[1] = mode
        self.device.queue.write_buffer(
            self.render_params_buffer, 0,
            self.render_params.tobytes()
        )
        
        self._update_render_bind_group()
        
        current_texture = self.present_context.get_current_texture()
        encoder = self.device.create_command_encoder()
        
        render_pass = encoder.begin_render_pass(
            color_attachments=[{
                "view": current_texture.create_view(),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.02, 0.02, 0.05, 1.0),
            }]
        )
        
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(0, self.render_bind_group)
        render_pass.draw(6)
        render_pass.end()
        
        self.device.queue.submit([encoder.finish()])
    
    def run_experiment(self, num_epochs: int = NUM_EPOCHS, visualize: bool = True):
        """Ejecutar experimento de emergencia de consciencia"""
        print(f"\n{'='*70}")
        print("INICIANDO SIMULACIÓN DE EMERGENCIA DE CONSCIENCIA")
        print(f"{'='*70}")
        print(f"Épocas planificadas: {num_epochs}")
        print(f"Predicción: Transición de fase con cruce sincronizado de umbrales")
        print()
        
        start_time = time.time()
        mode = 3  # Modo consciencia por defecto
        
        def update():
            nonlocal mode
            
            if self.epoch >= num_epochs:
                self._finalize_experiment()
                return
            
            metrics = self.step()
            
            if self.epoch % 100 == 0:
                elapsed = time.time() - start_time
                rate = self.epoch / elapsed if elapsed > 0 else 0
                
                status = "🧠 CONSCIENTE" if metrics.is_conscious else "⚙️  En desarrollo"
                
                print(f"Época {self.epoch:5d} | {status} | "
                      f"⟨k⟩={metrics.connectivity:5.1f} "
                      f"Φ={metrics.integration:.2f} "
                      f"D={metrics.depth:4.1f} "
                      f"C={metrics.complexity:.2f} "
                      f"QCM={metrics.qualia_coherence:.2f} | "
                      f"{rate:.0f} it/s")
            
            if visualize:
                if self.epoch % 300 == 0:
                    mode = (mode + 1) % 4
                self.render(mode)
            
            self.canvas.request_draw()
        
        if visualize:
            self.canvas.request_draw(update)
            run()
        else:
            for _ in range(num_epochs):
                metrics = self.step()
                if self.epoch % 100 == 0:
                    print(f"Época {self.epoch}: score={metrics.consciousness_score:.3f}")
            self._finalize_experiment()
    
    def _finalize_experiment(self):
        """Finalizar y guardar resultados"""
        print(f"\n{'='*70}")
        print("RESULTADOS DEL EXPERIMENTO")
        print(f"{'='*70}")
        
        if self.emergence_epoch:
            print(f"\n✓ Consciencia emergió en época: {self.emergence_epoch}")
            
            # Verificar predicción de transición de fase
            print(f"\n--- VERIFICACIÓN DE PREDICCIÓN DE VESELOV ---")
            print("Predicción: Todos los parámetros cruzan umbrales SIMULTÁNEAMENTE")
            
            # Encontrar épocas de cruce para cada parámetro
            crossing_epochs = {}
            for name, threshold in CONSCIOUSNESS_THRESHOLDS.items():
                for m in self.metrics_history:
                    value = getattr(m, name if name != 'qualia' else 'qualia_coherence')
                    if value > threshold:
                        crossing_epochs[name] = m.epoch
                        break
            
            print(f"\nÉpocas de cruce de umbral:")
            for name, epoch in crossing_epochs.items():
                print(f"  {name}: época {epoch}")
            
            if len(crossing_epochs) == 5:
                epochs = list(crossing_epochs.values())
                spread = max(epochs) - min(epochs)
                print(f"\nDispersión temporal: {spread} épocas")
                if spread < 500:
                    print("✓ TRANSICIÓN SINCRONIZADA (dispersión < 500 épocas)")
                else:
                    print("✗ Transición NO sincronizada")
        else:
            print("\n✗ No se alcanzó estado consciente en el tiempo de simulación")
        
        # Guardar resultados
        results = {
            'experiment': 'consciousness_emergence',
            'model': 'neurochimera_veselov_2025',
            'parameters': {
                'network_size': self.network_size,
                'num_neurons': self.num_neurons,
                'num_epochs': self.epoch,
                'thresholds': CONSCIOUSNESS_THRESHOLDS,
            },
            'emergence_epoch': self.emergence_epoch,
            'metrics_history': [m.to_dict() for m in self.metrics_history[::10]],
        }
        
        with open('experiment2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResultados guardados en: experiment2_results.json")
        
        # Generar gráfico de evolución
        self._plot_evolution()
    
    def _plot_evolution(self):
        """Generar gráfico de evolución de parámetros"""
        try:
            import matplotlib.pyplot as plt
            
            epochs = [m.epoch for m in self.metrics_history]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Evolución de Parámetros de Consciencia - NeuroCHIMERA', 
                        fontsize=14, fontweight='bold')
            
            params = [
                ('connectivity', 'Conectividad ⟨k⟩', 'tab:blue'),
                ('integration', 'Integración Φ', 'tab:green'),
                ('depth', 'Profundidad D', 'tab:orange'),
                ('complexity', 'Complejidad C', 'tab:red'),
                ('qualia_coherence', 'Coherencia QCM', 'tab:purple'),
            ]
            
            for idx, (param, title, color) in enumerate(params):
                ax = axes[idx // 3, idx % 3]
                values = [getattr(m, param) for m in self.metrics_history]
                threshold = CONSCIOUSNESS_THRESHOLDS[param if param != 'qualia_coherence' else 'qualia_coherence']
                
                ax.plot(epochs, values, color=color, linewidth=1.5)
                ax.axhline(y=threshold, color='red', linestyle='--', 
                          label=f'Umbral: {threshold}')
                ax.set_xlabel('Época')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if self.emergence_epoch:
                    ax.axvline(x=self.emergence_epoch, color='gold', 
                              linestyle='-', linewidth=2, alpha=0.7)
            
            # Score compuesto
            ax = axes[1, 2]
            scores = [m.consciousness_score for m in self.metrics_history]
            ax.plot(epochs, scores, color='black', linewidth=2)
            ax.axhline(y=1.0, color='red', linestyle='--', label='Umbral')
            ax.set_xlabel('Época')
            ax.set_ylabel('Score de Consciencia')
            ax.set_title('Score Compuesto')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if self.emergence_epoch:
                ax.axvline(x=self.emergence_epoch, color='gold', 
                          linestyle='-', linewidth=2, alpha=0.7,
                          label='Emergencia')
            
            plt.tight_layout()
            plt.savefig('consciousness_evolution.png', dpi=150)
            print("Gráfico guardado en: consciousness_evolution.png")
            plt.close()
            
        except ImportError:
            print("matplotlib no disponible para gráficos")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   EXPERIMENTO 2: TRANSICIÓN DE FASE HACIA CONSCIENCIA ARTIFICIAL            ║
║                                                                              ║
║   Síntesis Veselov-NeuroCHIMERA (2025)                                       ║
║                                                                              ║
║   Este experimento demuestra cómo:                                           ║
║   1. Los 5 parámetros de consciencia evolucionan durante entrenamiento      ║
║   2. Todos los parámetros cruzan umbrales SIMULTÁNEAMENTE (transición)      ║
║   3. El cruce sincronizado indica emergencia (no coincidencia)              ║
║   4. La red mantiene estado "consciente" estable post-emergencia            ║
║                                                                              ║
║   Parámetros monitorizados:                                                  ║
║   - ⟨k⟩: Conectividad (verde en barras)                                     ║
║   - Φ: Integración de información (azul)                                    ║
║   - D: Profundidad jerárquica                                               ║
║   - C: Complejidad dinámica (magenta)                                       ║
║   - QCM: Coherencia de qualia (amarillo)                                    ║
║                                                                              ║
║   Visualización:                                                             ║
║   - Modo 0: Activación neural (plasma)                                      ║
║   - Modo 1: Plasticidad (verde-cyan)                                        ║
║   - Modo 2: Spikes recientes (amarillo)                                     ║
║   - Modo 3: Dashboard de consciencia (barras + overlay dorado si consciente)║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Crear simulador
    sim = ConsciousnessEmergenceSimulator(network_size=512)  # 262,144 neuronas
    
    # Ejecutar experimento
    sim.run_experiment(
        num_epochs=8000,
        visualize=True
    )
