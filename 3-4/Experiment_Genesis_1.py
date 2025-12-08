import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

"""
EXPERIMENTO 1: GÉNESIS EN GF(2^n) - LA EMERGENCIA DEL ESPACIO-TIEMPO
Basado en: Veselov (2025) "Reality as a Unified Information-Computational Network"
Arquitectura: NeuroCHIMERA RGBA Native (Simulada)
"""

# --- CONFIGURACIÓN FÍSICA Y DE SIMULACIÓN ---
N = 256  # Tamaño del Universo (Texture Size 256x256)
T_sim = 200 # Pasos de tiempo
LEARNING_RATE = 0.01 # Eta en el paper
COUPLING = 0.5 # Fuerza de interacción vecinal

# --- DEFINICIÓN DEL SUSTRATO (Simulando Textura WebGPU RGBA) ---
# Shape: (Height, Width, 4 Channels)
# Inicialización: Estado de alta simetría/ruido (Big Bang)
universe_state = np.random.rand(N, N, 4).astype(np.float32)

# Kernel Laplaciano para simular interacciones locales (interferencia de ondas)
# Esto imita el shader de "vecinos" en WebGPU
LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

def free_energy_functional(state):
    """
    Calcula la Energía Libre (Hamiltoniano) de la red según el Apéndice A.4.
    L[phi] = Energia Cinetica + Energia Potencial - Entropia
    """
    R, G, B, A = state[:,:,0], state[:,:,1], state[:,:,2], state[:,:,3]
    
    # 1. Energía de Curvatura (Término alpha*R^2 de Veselov)
    # Usamos gradientes locales como proxy de curvatura
    grad_R = np.gradient(R)
    curvature_energy = np.sum(grad_R[0]**2 + grad_R[1]**2)
    
    # 2. Energía de Interacción (Termino W_ij)
    interaction_energy = -np.sum(R * G) 
    
    # 3. Entropía (Termino T*S)
    # Evitamos log(0) con un epsilon
    entropy = -np.sum(A * np.log(np.abs(A) + 1e-6))
    
    return curvature_energy + interaction_energy + entropy

def shader_step(state):
    """
    Simula un paso de Compute Shader en la GPU.
    Aplica las reglas de evolución: d(theta)/dt = -nabla L
    """
    new_state = state.copy()
    
    # Separar canales (simulando lectura de textura)
    r = state[:,:,0] # Materia
    g = state[:,:,1] # Geometría
    b = state[:,:,2] # Información
    a = state[:,:,3] # Tiempo/Entropía
    
    # --- DINÁMICA DE LA RED (LAWS OF PHYSICS) ---
    
    # 1. Difusión (Interacción vecinal - M-tiling)
    # El término Laplaciano suaviza el espacio (gravedad emergente)
    diff_r = convolve2d(r, LAPLACIAN_KERNEL, mode='same', boundary='wrap')
    diff_g = convolve2d(g, LAPLACIAN_KERNEL, mode='same', boundary='wrap')
    
    # 2. Reglas de Actualización (Ecuaciones de campo discretizadas)
    # dR/dt: La materia tiende a agruparse donde la geometría (G) es intensa
    dr = LEARNING_RATE * (diff_r + g * r - r**3) # Término cúbico para bi-estabilidad (0 o 1)
    
    # dG/dt: La geometría responde a la materia (Einstein simplificado)
    dg = LEARNING_RATE * (diff_g - r + 0.1*g)
    
    # dB/dt: Flujo de información (Causalidad)
    # La información fluye ortogonalmente al gradiente de materia
    grad_x, grad_y = np.gradient(r)
    db = 0.05 * (np.abs(grad_x) + np.abs(grad_y))
    
    # dA/dt: Flecha del tiempo (siempre positiva, acumulando entropía local)
    da = 0.01 * np.abs(dr) # El tiempo avanza donde hay cambio
    
    # Actualizar estado
    new_state[:,:,0] += dr
    new_state[:,:,1] += dg
    new_state[:,:,2] += db
    new_state[:,:,3] += da
    
    # Clampear valores (simulando rango de textura normalizada o GF(2))
    new_state = np.clip(new_state, 0.0, 1.0)
    
    return new_state

# --- VISUALIZACIÓN ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Experimento 1: Emergencia de Estructura en Red RGBA (Veselov-Network)', fontsize=14)

im_structure = axes[0].imshow(universe_state[:,:,:3], interpolation='nearest')
axes[0].set_title("Canales RGB (Materia-Geometría-Info)")
axes[0].axis('off')

loss_line, = axes[1].plot([], [], 'r-', lw=2)
axes[1].set_xlim(0, T_sim)
axes[1].set_ylim(0, 1000) # Ajustar según escala
axes[1].set_title("Funcional de Energía Libre (L)")
axes[1].set_xlabel("Tiempo (Epochs)")
axes[1].set_ylabel("Energía Libre")

energy_history = []

def animate(i):
    global universe_state
    
    # Ejecutar múltiples pasos físicos por frame para velocidad
    for _ in range(5):
        universe_state = shader_step(universe_state)
    
    # Calcular métricas
    energy = free_energy_functional(universe_state)
    energy_history.append(np.abs(energy)) # Valor absoluto para visualización logarítmica
    
    # Actualizar gráficos
    im_structure.set_data(universe_state[:,:,:3]) # Mostrar RGB
    
    loss_line.set_data(range(len(energy_history)), energy_history)
    axes[1].set_ylim(min(energy_history)*0.9, max(energy_history)*1.1)
    
    return im_structure, loss_line

print("Iniciando simulación de Génesis...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=False)
plt.show()