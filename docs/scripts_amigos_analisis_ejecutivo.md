# Análisis Ejecutivo — scripts/scripts_amigos

**Proyecto:** MoE (Mixture of Experts) — Clasificación médica multi-dominio
**Fecha:** 19 de abril de 2026
**Análisis detallados:** [Complejidad](analisis_estadistico_1_complejidad.md) · [Deuda Técnica](analisis_estadistico_2_deuda_tecnica.md) · [Arquitectura](analisis_estadistico_3_arquitectura.md)

---

## 1. Resumen Ejecutivo

`scripts/scripts_amigos` contiene un sistema MoE (Mixture of Experts) para clasificación de imágenes médicas multi-dominio. Incluye un script monolítico de entrenamiento (`train_exp1v21.py`, 1502 LOC) y un paquete modular (`moe/`, 1356 LOC, 13 archivos) que implementa routing, expertos y preprocesamiento adaptativo. El paquete `moe/` representa la pieza central de la arquitectura pero vive fuera de `src/pipeline/`, indicando que aún es un prototipo no integrado formalmente.

### Health Score Global: 5.8 / 10

### Top 3 Hallazgos Críticos

- **Zero test coverage** — 0 tests para 2858 LOC; ratio código:tests inexistente
- **God File** — `train_exp1v21.py` concentra 10 clases y 52 funciones en un archivo con 96% de funciones sin docstrings ni type hints (TDI ratio = 0.49)
- **Paquete `moe/` desplazado** — la pieza central de la tesis vive en `scripts/` en vez de `src/pipeline/`, impidiendo integración formal con el pipeline

### Top 3 Oportunidades de Mejora

- **Quick wins de documentación** — agregar docstrings + type hints a 98 funciones reduce 58% de deuda técnica en ~10 horas
- **Descomponer `train_exp1v21.py`** en 5 módulos (data, models, training, eval, utils) — mejora mantenibilidad y reusabilidad
- **`moe/` ya tiene buena arquitectura** (TDI=0.18) — solo necesita docstrings y tests para estar production-ready

### ⚡ Acción Inmediata

Dedicar **10 horas** a quick wins (docstrings, type hints, limpieza de imports) para reducir la deuda técnica un 58% antes de cualquier refactoring mayor.

---

## 2. Estado Actual del Proyecto

### Composición

| Categoría | Tamaño | % Total |
|---|---|---|
| Datos (.npy embeddings) | 46.0 GB | 84.4% |
| Checkpoints (.zip) | 0.83 GB | 1.5% |
| Código fuente (.py) | 2.0 MB | ~0.01% |
| Notebooks (.ipynb) | 0.19 GB | 0.3% |
| Otros (caché, entorno) | ~68 GB | 12.5% |
| **Total** | **545 GB** | |

### Distribución de LOC por Categoría

| Categoría Funcional | LOC | % | Barra |
|---|---|---|---|
| Training/ML | 19,638 | 41.4% | ████████████████ |
| Data Processing | 16,091 | 33.9% | █████████████ |
| Evaluation/Deploy | 6,709 | 14.2% | █████ |
| Utilities | 3,295 | 7.0% | ██ |
| MoE System | 1,356 | 2.9% | █ |
| Tests | 311 | 0.7% | ▏ |

### Estadísticas Clave

| Métrica | Valor |
|---|---|
| LOC total (Python) | 47,400 |
| Archivos .py | 133 |
| Type hints coverage (media) | 57.3% |
| Docstring coverage (media) | 62.9% |
| Health score mediana | 90.0 / 100 |
| Ratio código:datos | 1:4,742 |
| Ratio código:tests | 1:152 |

**El proyecto en contexto:** Un proyecto ML de investigación maduro con buena separación por fases y bajo acoplamiento. El 75% del código se dedica a training + data processing, lo cual es esperable. La lógica crítica está correctamente en `.py` (no en notebooks). Sin embargo, el ratio código:tests de 1:152 y la concentración de deuda en `fase0/` y el script monolítico representan riesgos para evolucionar a producción.

---

## 3. Salud del Código

### Health Score por Componente

| Componente | Archivos | LOC | Health Avg | Riesgo |
|---|---|---|---|---|
| `fase5/` | 5 | 924 | 95.2 | 🟢 Muy bajo |
| `fase2/` (otros) | 20 | 4,126 | 93.2 | 🟢 Bajo |
| `moe/` | 13 | 1,356 | 90.0* | 🟢 Bajo |
| `fase1/` | 12 | 3,481 | 87.7 | 🟢 Bajo |
| `datasets/` | 5 | 2,288 | 82.3 | 🟡 Bajo |
| `fase6/` | 9 | 4,855 | 74.3 | 🟡 Medio |
| `fase2/` (DDP) | 5 | 3,976 | 63.4 | 🟠 Medio-Alto |
| `fase0/` | 9 | 5,418 | 57.0 | 🔴 Alto |
| `train_exp1v21.py` | 1 | 1,502 | 35.0 | 🔴 Crítico |

*\*Estimado basado en TDI ratio*

**Scatter LOC vs Health:** Correlación fuerte negativa (r = -0.712). Los archivos no crecen añadiendo funciones simples sino haciendo funciones más complejas. Los 5 archivos con health < 50 concentran 5,177 LOC (11% del código) y son los principales generadores de riesgo.

### Top 5 Hotspots

| Archivo | LOC | Max CC | Hints | Docs | Health |
|---|---|---|---|---|---|
| `fase0/pre_embeddings.py` | 1,714 | 45 | 0% | 29% | **20.8** |
| `train_exp1v21.py` | 1,502 | — | 4% | 4% | **35.0** |
| `fase6/paso10_verificacion.py` | 1,254 | 31 | 100% | 88% | **39.9** |
| `fase0/pre_modelo.py` | 939 | 35 | 0% | 0% | **41.2** |
| `fase0/fase0_pipeline.py` | 765 | 29 | 0% | 7% | **46.1** |

### Benchmarks

| Nivel | Health Score | CC Máxima | Type Hints | Docstrings |
|---|---|---|---|---|
| 🟢 Bueno | >80 | <15 | >80% | >80% |
| 🟡 Aceptable | 60–80 | 15–25 | 50–80% | 50–80% |
| 🔴 Problemático | <60 | >25 | <50% | <50% |

---

## 4. Deuda Técnica

### TDI por Componente

| Componente | TDI Ratio | Nivel | Horas estimadas |
|---|---|---|---|
| `train_exp1v21.py` | 0.49 | 🔴 Alto | 26 h |
| `fase0/` | ~0.40 | 🔴 Alto | 20 h |
| `fase2/` (DDP) | ~0.25 | 🟠 Medio | 12 h |
| `moe/` | 0.18 | 🟡 Bajo | 6 h |
| Resto del proyecto | ~0.10 | 🟢 Bajo | 2 h |

### Antipatrones Críticos

| Antipatrón | Severidad | LOC Afectadas | Impacto |
|---|---|---|---|
| God File (train_exp1v21.py) | 🔴 Crítico | 1,502 | Imposible de testear, mantener o reutilizar |
| Zero Test Coverage | 🔴 Crítico | 47,400 | No se detectan regresiones |
| Missing Type Hints (fase0) | 🟠 Alto | 4,436 | Sin soporte IDE, bugs silenciosos |
| Magic Numbers (~52) | 🟠 Alto | ~80 | Configuración enterrada en código |
| Runtime pip install | 🟡 Medio | 8 | Rompe reproducibilidad |
| Requirements inflado (~180 pkgs) | 🟡 Medio | N/A | Conflictos de dependencias, +2GB innecesarios |

### Estimación Total

| Categoría | Horas |
|---|---|
| Quick wins (docstrings, hints, limpieza) | 10 h |
| Refactoring medio (magic numbers, configs) | 3 h |
| Descomposición train_exp1v21.py | 16 h |
| Tests unitarios (moe/ + core) | 16 h |
| **Total** | **~45 h** |

**Costo de la deuda técnica en el tiempo:** La deuda actual añade un overhead estimado de 30-40% en cada nueva feature. Sin intervención, cada nuevo experto añadido incrementará la duplicación (ya existen versiones DDP y no-DDP de cada trainer). En 6 meses sin refactoring, la deuda se duplicaría haciendo cualquier cambio significativamente más costoso.

---

## 5. Dependencias y Arquitectura

### Top 10 Librerías Externas

| Librería | Imports | Archivos | Rol |
|---|---|---|---|
| `torch` | 176 | 72 (54%) | Framework ML principal |
| `pathlib` | 71 | 71 | Manejo de rutas |
| `logging` | 69 | 69 | Logging |
| `numpy` | 59 | 59 | Operaciones numéricas |
| `json` | 39 | 39 | Serialización |
| `pandas` | 24 | 24 | Manipulación de datos |
| `argparse` | 24 | — | CLI |
| `sklearn` | 22 | 20 | Métricas, routers |
| `timm` | 8 | 8 | Modelos pre-entrenados |
| `PIL` | 7 | 7 | Carga de imágenes |

**Peso muerto detectado:** `tensorflow` (2.20), `openai-whisper`, `ctranslate2`, `ray` — listados en requirements pero sin uso en código. Eliminación ahorra ~3GB y reduce conflictos.

### Fan-in / Fan-out — Módulos Críticos

| Módulo | Fan-in | Fan-out | Clasificación |
|---|---|---|---|
| `config.py` | 55 | Bajo | 🔴 Hub central — punto de fallo único |
| `datasets/*` | 18 | 10 | 🟠 Proveedor de datos |
| `fase1_config` | 12 | Bajo | 🟡 Config hub |
| `ddp_utils` | 7 | ~5 | 🟢 Utilidad estable |
| `fase2/` (total) | Medio | 38 | 🟠 Mayor fan-out del proyecto |

### ¿Está bien acoplado?

**Sí, razonablemente.** Las fases se comunican a través de `config.py` y `datasets/`, no directamente entre sí (~5 imports cross-fase). La arquitectura por fases (fase0→fase1→...→fase6) es lineal y clara. Dos problemas: (1) `config.py` con fan-in de 55 es un punto de fallo único, y (2) `moe/` debería estar en `src/pipeline/` para completar la integridad arquitectónica.

---

## 6. Plan de Acción

### Quick Wins — Semana 1 (≤10 h)

| Tarea | Esfuerzo | Impacto | TDI Reducción |
|---|---|---|---|
| Agregar 50 docstrings a train_exp1v21.py | 2.5 h | Alto | -550 pts |
| Agregar 50 type hints a train_exp1v21.py | 2.0 h | Alto | -550 pts |
| Agregar 48 docstrings a moe/ | 2.0 h | Medio | -500 pts |
| Eliminar imports duplicados + runtime pip install | 0.5 h | Bajo | -14 pts |
| Agregar `__all__` a `__init__.py` de moe/ | 0.25 h | Bajo | -3 pts |
| Extraer magic numbers a constantes | 3.0 h | Medio | -80 pts |

### Medium Term — Semanas 2-3 (≤20 h)

| Tarea | Esfuerzo | Impacto |
|---|---|---|
| Descomponer train_exp1v21.py → data/, models/, training/, eval/, utils/ | 16 h | 🔴 Crítico |
| Limpiar requirements.txt (separar core/dev/optional, eliminar peso muerto) | 2 h | 🟠 Alto |
| Mover moe/ de scripts/ a src/pipeline/ | 2 h | 🟠 Alto |

### Long Term — Semanas 4-7 (≤20 h)

| Tarea | Esfuerzo | Impacto |
|---|---|---|
| Tests unitarios para moe/ (routing + experts) | 8 h | 🔴 Crítico |
| Tests para funciones core de fase0/ y fase2/ | 8 h | 🔴 Crítico |
| Crear expert registry (agregar expertos declarativamente) | 3 h | 🟠 Medio |
| Unificar trainers DDP/no-DDP con flag | 4 h | 🟡 Bajo |

### Timeline

| Semana | Foco | Horas | Resultado |
|---|---|---|---|
| 1 | Quick wins: docs, hints, limpieza | 10 h | TDI -58% |
| 2-3 | Descomponer script + mover moe/ + limpiar deps | 20 h | Arquitectura sólida |
| 4-5 | Tests moe/ + routing | 8 h | Coverage >0% → ~30% |
| 6-7 | Tests core + registry + unificar DDP | 15 h | Sistema mantenible |
| **Total** | | **~53 h** | Proyecto production-ready |

---

## 7. Conclusiones y Recomendaciones

### Salud Global + Riesgos

El proyecto tiene una base arquitectónica sólida (fases bien separadas, bajo acoplamiento, mediana de health 90/100) pero con **puntos de dolor concentrados**: el 11% del código (5 archivos) genera el 80% del riesgo. La ausencia total de tests es el riesgo #1 para evolución y mantenimiento.

### Prioridades (Ranked)

| # | Prioridad | Por qué |
|---|---|---|
| 1 | **Quick wins de documentación** (10h) | ROI máximo: 58% reducción deuda con esfuerzo mínimo |
| 2 | **Descomponer train_exp1v21.py** (16h) | Desbloquea testabilidad y reutilización del código |
| 3 | **Tests unitarios** (16h) | Habilita CI/CD y detecta regresiones al agregar expertos |
| 4 | **Integrar moe/ + limpiar deps** (4h) | Completa la integridad arquitectónica |

### Siguientes Pasos — Primeros 30 Días

- **Día 1-3:** Ejecutar todos los quick wins (docstrings, type hints, limpieza)
- **Día 4-10:** Descomponer `train_exp1v21.py` en módulos
- **Día 11-14:** Mover `moe/` a `src/pipeline/`, limpiar `requirements.txt`
- **Día 15-25:** Escribir tests para `moe/` y funciones core
- **Día 26-30:** Crear expert registry, unificar DDP/no-DDP

### Métricas de Éxito

| Métrica | Actual | Target 30 días | Target 60 días |
|---|---|---|---|
| Health score mediana | 90 | 92 | 95 |
| Health score mínimo | 20.8 | 60 | 75 |
| TDI ratio promedio | 0.34 | 0.15 | 0.08 |
| Test coverage | 0% | 15% | 30% |
| Type hints coverage | 57% | 80% | 90% |
| Docstring coverage | 63% | 85% | 95% |
| Max LOC por archivo | 1,714 | 500 | 400 |
| Dependencias requirements.txt | ~180 | ~40 (core) | ~40 |
