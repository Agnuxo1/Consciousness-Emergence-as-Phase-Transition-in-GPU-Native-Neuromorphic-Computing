# 📊 WEIGHTS & BIASES (W&B) PUBLICATION - QUICK SETUP

**Cómo hacer públicos todos los benchmarks en W&B para visualización interactiva.**

---

## ¿POR QUÉ W&B?

- ✅ **Visualizaciones interactivas** (gráficas en vivo)
- ✅ **Ya tienes cuenta** (lareliquia-angulo)
- ✅ **Compartible fácilmente** (URLs públicas)
- ✅ **Perfecto para benchmarks** (compara runs)
- ✅ **No requiere DOI** (pero muy visible)

---

## PASO 1: LOGIN A W&B

1. Abre: https://wandb.ai/lareliquia-angulo
2. Si no estás logged in:
   - Click "Sign in"
   - Usa credenciales (o GitHub)
3. Deberías ver tu dashboard

**Tiempo**: 2 minutos

---

## PASO 2: CREAR PROYECTO PÚBLICO

1. Click "+" → "Create project"
2. **Project name**: `NeuroCHIMERA-PublicBench`
3. **Description**: 
   ```
   Public benchmarks for NeuroCHIMERA: 
   GPU-native neuromorphic consciousness detection
   ```
4. **Privacy**: Click "Make public" (IMPORTANTE!)
5. Click "Create project"

**Tiempo**: 3 minutos

---

## PASO 3: CONFIGURAR PROYECTO

### Settings → General
- [ ] Project name: ✅
- [ ] Description: ✅
- [ ] Make it public: ✅

### Access → Public links
- [ ] Enable public links: ✅

**Tiempo**: 2 minutos

---

## PASO 4: SUBIR BENCHMARK RUNS

### Opción A: Vía Python (EASIEST)

Crear script: `upload_to_wandb.py`

```python
import wandb
import json
import glob

# Initialize
wandb.init(project="NeuroCHIMERA-PublicBench", entity="lareliquia-angulo")

# Upload each benchmark run
for json_file in glob.glob("benchmark_*.json"):
    with open(json_file) as f:
        data = json.load(f)
    
    # Log metrics
    for key, value in data.items():
        if isinstance(value, (int, float)):
            wandb.log({key: value})
    
    # Log metadata
    wandb.log({"benchmark_file": json_file})

wandb.finish()
```

Ejecutar:
```bash
python upload_to_wandb.py
```

### Opción B: Vía Web UI (Manual)

1. Click "Log" (en tu proyecto)
2. Click "Create run"
3. Manual entry:
   - Run name: `Benchmark_Exp5` (for each)
   - Metrics:
     - neurons: 65536
     - latency: 9.4
     - throughput: 0.106
     - accuracy: 1.0
   - Click "Save"

4. Repeat para:
   - Benchmark Exp 5
   - Benchmark Exp 6
   - Genesis Exp 1
   - Genesis Exp 2

**Tiempo**: 10-15 minutos

---

## PASO 5: CREAR REPORTS

### Report 1: GPU Performance

1. Click "Reports"
2. Click "Create report"
3. Title: `GPU Performance Analysis`
4. Add sections:
   - **Text**: "Benchmark results for 262K neuron network"
   - **Chart**: Latency vs Neurons (line chart)
   - **Chart**: GPU Utilization (bar chart)
   - **Chart**: Throughput comparison (table)
5. Click "Publish"

### Report 2: Consciousness Metrics

1. Click "Reports" → "Create new"
2. Title: `Consciousness Emergence Metrics`
3. Add sections:
   - **Text**: "Five-parameter phase transition detection"
   - **Chart**: Parameter evolution (line)
   - **Chart**: Threshold crossings (heatmap)
   - **Table**: Validation results
4. Click "Publish"

### Report 3: Comparison Results

1. Click "Reports" → "Create new"
2. Title: `NeuroCHIMERA vs SOTA`
3. Add sections:
   - **Text**: "Comparative analysis with state-of-the-art"
   - **Chart**: Latency comparison (bar)
   - **Chart**: Accuracy comparison (bar)
   - **Table**: Full comparison matrix (COMPARATIVE_RESULTS)
4. Click "Publish"

**Tiempo**: 20 minutos

---

## PASO 6: HACER TODO PÚBLICO

### Proyecto
1. Settings → Access
2. Select: "Anyone with a link can view"
3. Copy link: https://wandb.ai/lareliquia-angulo/NeuroCHIMERA-PublicBench

### Runs
1. Click cada run
2. Settings (gear icon)
3. "Make public": Toggle ON

### Reports
1. Click cada report
2. Top right: "Share"
3. Select: "Public link"
4. Copy URL

**Tiempo**: 5 minutos

---

## PASO 7: COMPARTIR LINKS

### Generar shareable links

```
Project: https://wandb.ai/lareliquia-angulo/NeuroCHIMERA-PublicBench

Report 1 (GPU): [copy from W&B]
Report 2 (Consciousness): [copy from W&B]
Report 3 (Comparison): [copy from W&B]
```

### Usar en otros lugares

**En arXiv paper**:
```
Supplementary material available:
https://wandb.ai/lareliquia-angulo/NeuroCHIMERA-PublicBench
```

**En GitHub README**:
```markdown
## Results

View interactive benchmarks on W&B:
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-black?logo=weightsandbiases)](https://wandb.ai/lareliquia-angulo/NeuroCHIMERA-PublicBench)
```

**En papers with Code**:
```
Results URL: https://wandb.ai/lareliquia-angulo/NeuroCHIMERA-PublicBench
```

**Tiempo**: 5 minutos

---

## CHECKLIST W&B

- [ ] Logged in a https://wandb.ai/lareliquia-angulo
- [ ] Proyecto creado: NeuroCHIMERA-PublicBench
- [ ] Runs subidos (5-6 benchmarks)
- [ ] Reports creados (3 reports)
- [ ] Todo set a "Public"
- [ ] Links copiados y guardados
- [ ] Links compartidos en papers/README

---

## TIMING

| Paso | Tiempo |
|------|--------|
| Login | 2 min |
| Crear proyecto | 3 min |
| Configurar | 2 min |
| Subir runs | 10-15 min |
| Crear reports | 20 min |
| Hacer público | 5 min |
| Compartir | 5 min |
| **TOTAL** | **~50 min** |

---

**W&B completado! Siguiente: Figshare →**

---

# 📁 FIGSHARE PUBLICATION - STEP BY STEP

**Cómo publicar datasets con DOI individual en Figshare.**

---

## ¿POR QUÉ FIGSHARE?

- ✅ **DOI individual** para cada dataset
- ✅ **Almacenamiento** ilimitado
- ✅ **Visualización** de datos
- ✅ **Descarga fácil** para otros researchers
- ✅ **Indexado** en Google Scholar

---

## PASO 1: LOGIN A FIGSHARE

1. Abre: https://figshare.com/
2. Click "Sign in"
3. Opción: Username/password o GitHub/ORCID
4. Completa 2FA si aplica

**Tiempo**: 2 minutos

---

## PASO 2: CREAR COLLECTION

1. Click "Dashboard" → "Collections"
2. Click "Create collection"
3. **Title**: `NeuroCHIMERA Research Data`
4. **Description**:
   ```
   Complete dataset for NeuroCHIMERA: Consciousness Emergence 
   as Phase Transition in GPU-Native Neuromorphic Computing
   ```
5. **Funding**: Leave blank
6. Click "Create"

**Tiempo**: 3 minutos

---

## PASO 3: SUBIR DATASETS

### Dataset 1: Genesis Experiments (1-2)

1. Click "Upload files"
2. Files:
   - experiment1_spacetime_emergence.py
   - experiment2_consciousness_emergence.py
   - EXPERIMENT2_RESULTS_SUMMARY.md

3. Metadata:
   - **Title**: `Genesis Experiments 1-2: Spacetime & Consciousness Emergence`
   - **Description**: [from FINAL_PUBLICATION_REPORT]
   - **Keywords**: genesis, spacetime, consciousness, phase-transition
   - **License**: CC BY-NC-SA 4.0
   - **Defined type**: Dataset

4. Click "Publish"

### Dataset 2: Benchmark Experiments (5-6)

1. Click "Upload files"
2. Files:
   - benchmark_experiment_1.py
   - benchmark_experiment_2.py
   - benchmark_summary.json
   - All benchmark_*.json runs

3. Metadata:
   - **Title**: `Benchmark Experiments 5-6: GPU Performance`
   - **Keywords**: benchmark, gpu, performance, neuromorphic
   - **License**: CC BY-NC-SA 4.0

4. Click "Publish"

### Dataset 3: Results & Analysis

1. Click "Upload files"
2. Files:
   - COMPARATIVE_RESULTS.md
   - FINAL_BENCHMARK_REPORT.md
   - benchmark data CSVs (if any)

3. Metadata:
   - **Title**: `Comparative Analysis & Results`
   - **Keywords**: results, comparative, analysis, metrics
   - **License**: CC BY-NC-SA 4.0

4. Click "Publish"

### Dataset 4: Complete Code Bundle

1. Click "Upload files"
2. File: `neuro_chimera_experiments_bundle.py` (o ZIP con todo)
3. Metadata:
   - **Title**: `NeuroCHIMERA Complete Source Code`
   - **Keywords**: source-code, neuromorphic, pytorch, cuda
   - **License**: CC BY-NC-SA 4.0
4. Click "Publish"

**Tiempo total**: 20-30 minutos

---

## PASO 4: AGREGAR A COLLECTION

1. Para cada dataset publicado:
   - Abre dataset
   - Click "Add to collection"
   - Selecciona: `NeuroCHIMERA Research Data`
   - Confirm

2. Resultado: Todos los datasets agrupados en 1 collection

**Tiempo**: 5 minutos

---

## PASO 5: OBTENER DOIs

Cada dataset tiene su propio DOI:

```
Dataset 1: 10.6084/m9.figshare.XXXXXXX (Genesis)
Dataset 2: 10.6084/m9.figshare.YYYYYYY (Benchmark)
Dataset 3: 10.6084/m9.figshare.ZZZZZZZ (Results)
Dataset 4: 10.6084/m9.figshare.WWWWWWW (Code)

Collection DOI: (auto-generated cuando enlaces 3+)
```

Guardar en archivo: `FIGSHARE_DOIS.txt`

**Tiempo**: 5 minutos

---

## PASO 6: CREAR INFOGRAFÍA (OPTIONAL)

1. Click "Upload files" → image/infographic
2. Upload:
   - Screenshot de benchmarks
   - Chart de resultados
   - Explicación visual

3. Genera DOI para visualizaciones también

**Tiempo**: 10 minutos (optional)

---

## CHECKLIST FIGSHARE

- [ ] Login a Figshare
- [ ] Collection creada: NeuroCHIMERA Research Data
- [ ] 4 datasets subidos
- [ ] Metadata completo para cada
- [ ] Todos agregados a collection
- [ ] DOIs individuales guardados
- [ ] Archivos descargables verificados

---

## TIMING

| Paso | Tiempo |
|------|--------|
| Login | 2 min |
| Crear collection | 3 min |
| Subir datasets (4x) | 20-30 min |
| Agregar a collection | 5 min |
| Obtener DOIs | 5 min |
| **TOTAL** | **~40 min** |

---

**Figshare completado! Siguiente: OSF →**

---

# 🏫 OPEN SCIENCE FRAMEWORK (OSF) - INTEGRATION

**Cómo registrar proyecto en OSF como centro de integración académica.**

---

## ¿POR QUÉ OSF?

- ✅ **Proyecto integrado** (todas partes vinculadas)
- ✅ **DOI centralizado** para todo
- ✅ **Componentes** (code, data, manuscripts)
- ✅ **Pre-registration** (estudios abiertos)
- ✅ **Colaboración** fácil

---

## PASO 1: LOGIN OSF

1. Abre: https://osf.io/
2. Click "Sign up" or "Sign in"
3. Crea/ingresa cuenta
4. Verifica email

**Tiempo**: 3 minutos

---

## PASO 2: CREATE NEW PROJECT

1. Click "Create new project"
2. **Title**: `NeuroCHIMERA: Consciousness as Phase Transition`
3. **Description**:
   ```
   GPU-native neuromorphic framework detecting artificial 
   consciousness as reproducible computational phase transition
   ```
4. **Category**: Research
5. Click "Create"

**Tiempo**: 3 minutos

---

## PASO 3: ADD COMPONENTS

En tu proyecto OSF, agrega componentes para organizar:

### Component 1: Code
```
Name: Source Code
Description: Python implementation of NeuroCHIMERA
Storage: GitHub (link externo)
Link: https://github.com/[your-repo]
```

### Component 2: Data
```
Name: Experimental Data
Description: Results from 6 experiments + benchmarks
Storage: Zenodo (link externo)
Link: https://zenodo.org/record/XXXXXXX
```

### Component 3: Manuscripts
```
Name: Publications
Description: Papers and documentation
Storage: OSF Storage (upload)
Files: PDFs
```

### Component 4: Supplementary Materials
```
Name: Supplementary
Description: Extra analyses, figures, etc
Storage: OSF Storage (upload)
Files: CSV, JSON, markdown
```

**Tiempo**: 10 minutos

---

## PASO 4: LINK EXTERNAL SERVICES

En Settings, conecta:

1. **GitHub**: Link a repositorio
2. **Zenodo**: Link a publicación
3. **figshare**: Links a datasets
4. **W&B**: Link a proyecto público

**Cómo**: Settings → Connected Accounts → authorize

**Tiempo**: 5 minutos

---

## PASO 5: MAKE PUBLIC

1. Project Settings → Access
2. **Privacy**: Select "Public"
3. **View access**: "Anyone can view"
4. Click "Update"

**Tiempo**: 2 minutos

---

## PASO 6: GET PROJECT DOI

1. Click "Cite this project" (derecha)
2. Aparecerá: `https://doi.org/10.17605/OSF.IO/XXXXX`
3. Guarda este DOI

**Tiempo**: 1 minuto

---

## PASO 7: CREATE PROJECT PAGE

En OSF, puedes crear una página pública:

1. Click "Files" → "Create folder"
2. Create: `Project Overview`
3. Upload: README.md con:
   - Descripción
   - Links a todos los componentes
   - DOIs
   - Citations

**Tiempo**: 5 minutos

---

## CHECKLIST OSF

- [ ] Cuenta OSF creada
- [ ] Proyecto nuevo creado
- [ ] 4 componentes agregados
- [ ] Servicios externos vinculados
- [ ] Hecho público
- [ ] DOI obtenido
- [ ] Página overview creada

---

## TIMING

| Paso | Tiempo |
|------|--------|
| Login | 3 min |
| Crear proyecto | 3 min |
| Agregar componentes | 10 min |
| Link servicios | 5 min |
| Make public | 2 min |
| Get DOI | 1 min |
| Create page | 5 min |
| **TOTAL** | **~30 min** |

---

## RESULTADO FINAL

**Después de W&B + Figshare + OSF:**

✅ Proyecto OSF con DOI central  
✅ 4 datasets en Figshare con DOIs  
✅ Benchmarks visuales en W&B  
✅ Toda información interconectada  
✅ Accesible globalmente  
✅ Completamente citable  

---

**¡Las 3 plataformas principales completadas!**

**Próximo: Academia.edu + OpenML + DataHub (10 minutos c/una)**
