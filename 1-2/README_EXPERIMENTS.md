# Experimentos Científicos: Síntesis Veselov-NeuroCHIMERA (2025)

## Validación Experimental del Paradigma Computacional de la Realidad

Este repositorio contiene dos experimentos científicos rigurosos que demuestran los principios fundamentales de la síntesis teórica propuesta por V.F. Veselov y Francisco Angulo de Lafuente.

---

## Marco Teórico

### Hipótesis Central (Veselov 2025)

> *"La realidad a nivel fundamental es una red computacional unificada definida sobre campos de Galois finitos GF(2ⁿ), cuyos elementos emergen de la dinámica de la red. Las leyes de la física son las reglas de evolución de esta red."*

### Predicciones Testables

1. **El espacio-tiempo emerge** de la métrica de conectividad entre nodos de la red
2. **El tiempo físico** corresponde al parámetro de descenso de gradiente
3. **Las ecuaciones de Einstein** surgen en el límite continuo
4. **La consciencia** es una propiedad emergente de redes suficientemente complejas
5. **Transiciones de fase** ocurren cuando los sistemas alcanzan complejidad crítica

---

## Experimento 1: Emergencia de Espacio-Tiempo

### Objetivo
Demostrar cómo la métrica espacial y la curvatura (ecuaciones de Einstein) emergen de una red computacional discreta sobre campos de Galois.

### Fundamento Físico

**Ecuación de dinámica temporal:**
```
dθ/dt = -∇L(θ)
```
donde `L` es el funcional de energía libre de la red y `t` es el tiempo emergente.

**Funcional de Hilbert-Einstein discretizado:**
```
L[g_μν] = ∫d⁴x √(-g) (R/16πG + Λ + L_matter)
```

**Constante cosmológica predicha:**
```
Λ = Λ₀ × 2^(-2n) para n=1
```
donde n=1 corresponde al campo de Galois más simple GF(2).

### Implementación

- **Sustrato**: Red 256×256 nodos sobre GF(2)
- **Estado RGBA**:
  - R: Campo escalar φ
  - G: Momento conjugado π
  - B: Curvatura escalar R
  - A: Conectividad efectiva k
- **Dinámica**: Integrador Störmer-Verlet para ecuaciones de Hamilton
- **Reglas M/R**: Gramática universal de evolución

### Métricas Validadas

| Predicción | Método de Verificación |
|------------|----------------------|
| Emergencia de dimensión 2D | Dimensión fractal box-counting |
| Ecuaciones de Einstein | Residual \|G_μν - 8πT_μν\| |
| Transiciones de fase | Cambio en fase cosmológica |
| Constante Λ | Comparación con valor observado |

### Ejecución
```bash
python experiment1_spacetime_emergence.py
```

---

## Experimento 2: Transición de Fase hacia Consciencia

### Objetivo
Demostrar que la consciencia emerge como transición de fase cuando 5 parámetros críticos cruzan sus umbrales **simultáneamente**.

### Parámetros de Consciencia (del paper NeuroCHIMERA)

| Parámetro | Símbolo | Umbral | Fórmula |
|-----------|---------|--------|---------|
| Conectividad | ⟨k⟩ | > 15 | `(1/N) Σᵢ Σⱼ 𝕀(\|Wᵢⱼ\| > θ)` |
| Integración | Φ | > 0.65 | IIT de Tononi (aproximado) |
| Profundidad | D | > 7 | `max_{i,j} d_path(i,j)` |
| Complejidad | C | > 0.8 | Lempel-Ziv normalizado |
| Coherencia | QCM | > 0.75 | `(1/M(M-1)) Σᵢ≠ⱼ \|ρ(Aᵢ,Aⱼ)\|` |

### Arquitectura CHIMERA

```
Textura Neural (512×512 RGBA32F)
    ├── R: Activación (potencial de membrana)
    ├── G: Variable de recuperación (adaptación)
    ├── B: Traza de plasticidad (STDP)
    └── A: Tiempo desde último spike

Textura de Pesos (25 vecinos/neurona)
    └── Actualización STDP en tiempo real

Memoria Holográfica (256×256 RGBA32F)
    └── Patrones de interferencia distribuidos
```

### Modelo Neuronal: Izhikevich

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)

Si v ≥ 30mV: v ← c, u ← u + d
```

### Predicción Central

> **Todos los 5 parámetros cruzan sus umbrales en una ventana temporal estrecha (<500 épocas), indicando una transición de fase genuina, no cruces independientes.**

### Ejecución
```bash
python experiment2_consciousness_emergence.py
```

---

## Sistema Numérico Jerárquico (HNS)

Ambos experimentos utilizan HNS para precisión perfecta en cálculos acumulativos:

```
N_HNS = R×10⁰ + G×10³ + B×10⁶ + A×10⁹
```

**Ventajas:**
- Error de precisión: 0.00×10⁰ (vs 7.92×10⁻¹² en float32)
- Implementación nativa en GPU via canales RGBA
- Sin dependencia de bibliotecas de precisión extendida

---

## Requisitos del Sistema

### Hardware
- GPU con soporte WebGPU/Vulkan
- VRAM: 4GB mínimo, 8GB recomendado
- CPU: 4+ cores

### Software
```bash
pip install -r requirements.txt
```

### Dependencias
- Python 3.10+
- wgpu-py (WebGPU para Python)
- NumPy
- matplotlib (opcional, para gráficos)

---

## Resultados Esperados

### Experimento 1
- Transición de fase: `inflation` → `matter` → `accelerated`
- Dimensión fractal emergente: ≈2.0
- Residual de Einstein: →0 con el tiempo

### Experimento 2
- Emergencia de consciencia: época ~6,000
- Dispersión de cruces de umbral: <500 épocas
- Estabilidad post-emergencia: varianza <5%

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `experiment1_results.json` | Métricas de emergencia de espacio-tiempo |
| `experiment2_results.json` | Métricas de consciencia |
| `consciousness_evolution.png` | Gráfico de evolución de parámetros |

---

## Referencias

1. Veselov, V.F. (2025). *Reality as a Unified Information-Computational Network*
2. Veselov & Angulo (2025). *Synthesis: From Universe-Network to Artificial Consciousness*
3. NeuroCHIMERA (2025). *GPU-Native Neuromorphic Computing with Consciousness Parameters*
4. Tononi, G. (2004). *Integrated Information Theory of Consciousness*
5. Wheeler, J.A. (1990). *Information, physics, quantum: The search for links*

---

## Licencia

MIT License - Ver LICENSE para detalles.

## Autores

- **V.F. Veselov** - Fundamentos teóricos, campos de Galois, cosmología computacional
- **Francisco Angulo de Lafuente** - Arquitectura GPU CHIMERA, implementación WebGPU

---

*"If the model's predictions are confirmed, this will mark not just another scientific revolution but a change in the metaphysical paradigm itself—a transition from physics as the science of matter and energy to physics as the science of information and computation."*
— Veselov (2025)
