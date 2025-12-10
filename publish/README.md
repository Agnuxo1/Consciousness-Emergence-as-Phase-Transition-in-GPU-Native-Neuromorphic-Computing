# NeuroCHIMERA Publishing & Benchmarking System

Sistema completo para auditar, testear y publicar los 6 experimentos de NeuroCHIMERA en múltiples plataformas científicas.

## 🚀 Quick Start

### Verificar Estado del Sistema
```bash
python publish/quick_status.py
```

### Ejecutar Pipeline Completo
```bash
python publish/run_and_publish_benchmarks.py
```

### Paso a Paso
```bash
# 1. Auditar experimentos
python publish/audit_experiments.py

# 2. Ejecutar benchmarks
python publish/run_benchmarks.py

# 3. Crear dashboards W&B
export WANDB_API_KEY="b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae"
python publish/create_public_dashboards.py

# 4. Subir a plataformas
python publish/upload_all_platforms.py

# 5. Actualizar badges
python publish/update_readme_badges.py
```

## 📚 Documentación

| Documento | Descripción | Cuándo Usar |
|-----------|-------------|-------------|
| **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** | Resumen ejecutivo del sistema completo | Empezar aquí - Overview general |
| **[MASTER_CHECKLIST.md](MASTER_CHECKLIST.md)** | Lista completa de tareas (10 fases) | Guía paso a paso detallada |
| **[PLATFORM_GUIDE.md](PLATFORM_GUIDE.md)** | Detalles de cada plataforma | Consulta por plataforma |
| **[README_PUBLISHING.md](README_PUBLISHING.md)** | Instrucciones de publicación | How-to práctico |
| **[README_AUDIT.md](README_AUDIT.md)** | Guía de auditoría externa | Para revisores externos |

## 🔧 Scripts Disponibles

### Automatización Principal

| Script | Descripción | Uso |
|--------|-------------|-----|
| `quick_status.py` | Verificación rápida del sistema | Verificar estado |
| `run_and_publish_benchmarks.py` | Pipeline completo | Automatización total |
| `audit_experiments.py` | Auditoría de 6 experimentos | Verificar integridad |
| `run_benchmarks.py` | Ejecutar todos los benchmarks | Testing completo |
| `run_experiment.py` | Ejecutar experimento individual | Testing específico |

### Publicación

| Script | Descripción | Plataformas |
|--------|-------------|-------------|
| `create_public_dashboards.py` | Dashboards públicos W&B | W&B |
| `upload_all_platforms.py` | Upload multi-plataforma | Zenodo, OSF, W&B |
| `update_readme_badges.py` | Actualizar badges README | GitHub |

### Scripts Legacy (Individuales)

| Script | Descripción |
|--------|-------------|
| `upload_to_wandb.py` | Upload individual a W&B |
| `upload_to_zenodo.py` | Upload individual a Zenodo |
| `upload_to_figshare.py` | Upload individual a Figshare |
| `upload_to_openml.py` | Upload individual a OpenML |

## 🌐 Plataformas Soportadas

### ✓ Automatizadas (3)
1. **Weights & Biases** - Dashboards interactivos
2. **Zenodo** - DOI permanente (draft automático)
3. **Open Science Framework** - Proyecto completo

### 📤 Manuales con Scripts (6)
4. **Figshare** - Datasets con DOI
5. **OpenML** - ML benchmarks
6. **DataHub** - Data packages
7. **Academia.edu** - Networking académico
8. **DrivenData** - Challenges (opcional)
9. **Signate/Zindi** - Regionales (opcional)

### ⚡ Automáticas (1)
10. **OpenAIRE** - Harvest desde Zenodo (24-48h)

## 📊 Estado Actual

Ejecutar `python publish/quick_status.py` para ver:

```
✓ 7 scripts de automatización
✓ 5 documentos de guía
✓ 6 experimentos auditados
✓ 4/6 experimentos con resultados
✓ 9 plataformas listas
✓ CI/CD configurado
```

## 🎯 Workflow Recomendado

### Fase 1: Preparación (1-2 horas)
1. Resolver dependencias faltantes
2. Ejecutar auditoría completa
3. Revisar reportes de auditoría

### Fase 2: Benchmarking (30-60 min)
1. Ejecutar todos los benchmarks
2. Verificar resultados
3. Revisar métricas

### Fase 3: Publicación Automatizada (30 min)
1. Crear dashboards W&B públicos
2. Publicar draft en Zenodo
3. Crear proyecto OSF
4. Configurar GitHub secrets para CI/CD

### Fase 4: Publicación Manual (2-3 horas)
1. Subir a Figshare (web o FTP)
2. Publicar en OpenML (6 datasets)
3. Crear package en DataHub
4. Subir paper a Academia.edu
5. (Opcional) Proponer challenge en DrivenData

### Fase 5: Finalización (1 hora)
1. Actualizar README con DOIs
2. Añadir badges
3. Verificar todos los links
4. Commit y push

## 🔑 Credenciales

Todas las credenciales están configuradas en los scripts:

```bash
# W&B
WANDB_API_KEY="b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae"

# Zenodo
ZENODO_TOKEN="lDYsHSupjRQXYxMAMihKn5lQwamqnsBliy0kwXbdUBg4VmxxuePbXxCpq2iw"

# OSF
OSF_TOKEN="KSAPimE65LQJ648xovRICXTSKHSnQT2xRgunNM1QHf6tu3eI81x1Z7b0vHduNJFTFgVKhL"

# Figshare FTP
Username: 5292188
Password: $GNJmzWHcQL6XSS
```

⚠️ **Importante**: Estas credenciales están hardcoded para facilitar el uso. En producción, considerar usar variables de entorno o GitHub Secrets.

## 📋 Checklist Rápido

- [ ] Estado verificado: `python publish/quick_status.py`
- [ ] Auditoría ejecutada: `python publish/audit_experiments.py`
- [ ] Benchmarks ejecutados: `python publish/run_benchmarks.py`
- [ ] Dashboards W&B creados y públicos
- [ ] Zenodo draft publicado (DOI obtenido)
- [ ] OSF proyecto creado y completo
- [ ] Figshare dataset subido
- [ ] OpenML 6 datasets publicados
- [ ] DataHub package publicado
- [ ] Academia.edu paper subido
- [ ] README actualizado con badges
- [ ] GitHub secrets configurados
- [ ] OpenAIRE verificado (48h después)

## 🆘 Troubleshooting

### Problema: Experimentos fallan por dependencias
```bash
pip install wgpu  # Para experimentos 1-2
# Verificar neuro_chimera_experiments_bundle para 5-6
```

### Problema: Upload a W&B falla
```bash
# Verificar API key
echo $WANDB_API_KEY

# Re-login
wandb login b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae
```

### Problema: Zenodo draft no se crea
- Verificar token válido en https://zenodo.org/account/settings/applications/
- Revisar tamaño de archivos (límite 50GB por archivo)
- Verificar logs del script

### Problema: Encoding errors en Windows
Los scripts ya incluyen fix para UTF-8 en Windows. Si persiste:
```powershell
chcp 65001  # Cambiar codepage a UTF-8
```

## 📈 Métricas & Monitoreo

### GitHub Actions
- Workflow: `.github/workflows/benchmarks.yml`
- Frecuencia: Cada lunes 04:00 UTC
- Artifacts: Benchmark results automáticos

### W&B Dashboards
- Main: https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments
- Performance: https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments-performance

### Zenodo
- Uploads: https://zenodo.org/me/uploads
- DOI: (se obtiene después de publish)

## 🎓 Para Colaboradores Externos

Si eres un revisor o colaborador externo:

1. Lee **[README_AUDIT.md](README_AUDIT.md)** para instrucciones de auditoría
2. Clona el repositorio
3. Sigue instrucciones de reproducibilidad
4. Reporta issues en GitHub

## 📞 Soporte

- **Issues**: GitHub Issues en el repositorio principal
- **Documentación**: Ver archivos .md en esta carpeta
- **Contacto**: Francisco Angulo de Lafuente

## 📜 Licencia

- Código: GPL-3.0
- Datos: CC-BY-4.0
- Paper: CC-BY-4.0

## 🎉 Contribuciones

Contributions welcome! Por favor:
1. Fork el repositorio
2. Crea una branch para tu feature
3. Commit tus cambios
4. Push a la branch
5. Crea un Pull Request

---

**Última Actualización**: 2025-12-09
**Versión**: 1.0
**Estado**: ✅ Production Ready

**¡Todo listo para publicar!** 🚀
