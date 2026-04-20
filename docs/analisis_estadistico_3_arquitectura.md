# Análisis Estadístico 3 — Composición, Dependencias y Arquitectura

> Proyecto: MoE (Mixture of Experts) — Clasificación médica multi-dominio
> Fecha: 19 de abril de 2026
> Base: recopilación directa del repositorio

---

## 1. Composición del Proyecto

### 1.1 Tamaño total por categoría

| Categoría | Tamaño | % del total |
|---|---|---|
| Datos (.npy embeddings) | 46.0 GB | 84.4% |
| Checkpoints (.zip, modelos) | 0.83 GB | 1.5% |
| Notebooks (.ipynb) | 0.19 GB | 0.3% |
| Scripts auxiliares (scripts/) | 0.63 GB | 1.2% |
| Código fuente (src/) | 7.3 MB | ~0.01% |
| Tests | 20 KB | ~0.00% |
| **Otros (caché, entorno, etc.)** | **~68 GB** | **12.5%** |
| **Total proyecto** | **545 GB** | **100%** |

> **Nota:** El proyecto pesa 545 GB, de los cuales la inmensa mayoría (~497 GB) corresponde a datos crudos, cachés o entornos virtuales no rastreados. Los .npy solos son 46 GB.

### 1.2 Proporción por tipo de archivo

| Extensión | Cantidad | Tamaño total | LOC (estimado) |
|---|---|---|---|
| `.py` | 133 | 2.0 MB | 47,400 |
| `.ipynb` | 14 | 7.7 MB | ~3,000–5,000 (celdas código) |
| `.npy` | 131,412 | 46 GB | N/A (datos binarios) |
| `.zip` | 3 | 485 MB | N/A (comprimidos) |

### 1.3 Densidad de código

| Métrica | Valor |
|---|---|
| Total LOC (Python) | 47,400 |
| Tamaño total .py | 2,004 KB |
| **Densidad** | **23.7 LOC/KB** |
| Promedio LOC/archivo | 356 LOC |
| Mediana estimada | ~200 LOC |

### 1.4 Concentración de lógica

| Archivo | LOC | % del total |
|---|---|---|
| `pre_embeddings.py` (fase0) | 2,164 | 4.56% |
| `train_exp1v21.py` (scripts) | 1,502 | 3.17% |
| `paso10_verificacion.py` (fase6) | 1,448 | 3.05% |
| `pre_modelo.py` (fase0) | 1,233 | 2.60% |
| `train_expert1_ddp.py` (fase2) | 1,162 | 2.45% |
| **Top-5 acumulado** | **7,509** | **15.84%** |
| **Top-10 acumulado** | **12,673** | **26.74%** |

> La lógica no está hiper-concentrada: el archivo más grande contiene solo el 4.56% del código total. Distribución razonable.

---

## 2. Análisis de Dependencias

### 2.1 Dependencias externas (requirements.txt)

Total de librerías en `requirements.txt`: **~180 paquetes** (incluyendo transitivas).

**Librerías core del proyecto** (usadas directamente en código):

| Librería | Imports totales | Archivos que la usan | Rol |
|---|---|---|---|
| `torch` | 176 | 72 | Framework ML principal |
| `numpy` | 59 | 59 | Operaciones numéricas |
| `pathlib` | 71 | 71 | Manejo de rutas |
| `logging` | 69 | 69 | Logging |
| `json` | 39 | 39 | Serialización |
| `pandas` | 24 | 24 | Manipulación de datos |
| `sklearn` | 22 | 20 | Métricas, routers clásicos |
| `argparse` | 24 | — | CLI |
| `timm` | 8 | 8 | Modelos pre-entrenados |
| `torchvision` | 7 | 6 | Transforms, datasets |
| `PIL` | 7 | 7 | Carga de imágenes |
| `matplotlib` | 7 | 7 | Visualización |
| `monai` | — | — | Transforms médicos 3D |
| `gradio` | — | — | Web app |

### 2.2 Librería más usada

- **Por imports:** `torch` (176 apariciones)
- **Por archivos:** `torch` (72 de 133 archivos = **54%**)

### 2.3 Dependencias pesadas/riesgo

| Librería | Observación |
|---|---|
| `tensorflow` 2.20.0 | Listada en requirements pero **no se usa** en código .py del src/. Peso muerto (~2 GB). |
| `openai-whisper` | Irrelevante para el proyecto médico. Posible residuo. |
| `ctranslate2` | Idem, sin uso detectado en src/. |
| `ray` 2.54.0 | Listada, no se detecta uso en código fuente. |
| `monai` 1.5.2 | Activa y mantenida. Sin riesgo. |
| `timm` 1.0.25 | Activa y mantenida. Sin riesgo. |

> **Riesgo principal:** el requirements.txt incluye ~180 paquetes (muchos transitivos o no usados). Esto dificulta la reproducibilidad y aumenta la superficie de conflictos.

### 2.4 Grafo de dependencias (descripción textual)

```
                        ┌─────────────┐
                        │   config.py  │ ◄── fan-in: 55 archivos
                        └──────┬──────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
        ┌──────────┐    ┌──────────┐       ┌──────────┐
        │  fase0/  │    │  fase1/  │       │  fase2/  │
        │ 7,466 LOC│    │ 4,949 LOC│       │14,902 LOC│
        └────┬─────┘    └────┬─────┘       └────┬─────┘
             │               │                   │
             │               ▼                   ▼
             │        ┌────────────┐      ┌────────────┐
             │        │ datasets/  │◄─────│  models/   │
             │        │ 3,676 LOC  │      │  routers/  │
             │        └────────────┘      └────┬───────┘
             │                                 │
             ▼                                 ▼
        ┌──────────┐                    ┌──────────┐
        │  fase3/  │                    │  fase5/  │
        │ 3,452 LOC│                    │ 1,284 LOC│
        └──────────┘                    └────┬─────┘
                                             │
                                             ▼
                                      ┌──────────┐
                                      │  fase6/  │
                                      │ 6,709 LOC│
                                      │ webapp/  │
                                      │ dashboard│
                                      └──────────┘
                                             │
                                             ▼
                                      ┌──────────┐
                                      │  moe/    │
                                      │ 1,356 LOC│
                                      │(scripts/)│
                                      └──────────┘
```

---

## 3. Arquitectura del Proyecto

### 3.1 Acoplamiento — Imports cruzados

| Tipo | Cantidad |
|---|---|
| Imports internos únicos (from fase/config/datasets) | ~60 patrones distintos |
| Módulo más importado | `config.py` (55 archivos) |
| Segundo más importado | `datasets/*` (18 archivos) |
| Imports cross-fase directos | Bajo (~5 patrones fase→fase) |

> **Evaluación:** acoplamiento bajo-moderado. Las fases se comunican principalmente a través de `config.py` y `datasets/`, no entre sí directamente. Buen diseño.

### 3.2 Cohesión

| Fase | Responsabilidad | Cohesión |
|---|---|---|
| fase0 | Descarga, extracción, preprocesamiento | ✅ Alta |
| fase1 | Backbones, embeddings, transforms | ✅ Alta |
| fase2 | Training expertos + routers | ✅ Alta (pero grande) |
| fase3 | Modelos generativos (CAE, ResUNet) | ✅ Alta |
| fase5 | Fine-tune global MoE | ✅ Alta |
| fase6 | Evaluación, webapp, dashboard | ⚠️ Media (mezcla UI + eval) |

### 3.3 Modularización

| Ubicación | LOC | % del total |
|---|---|---|
| `src/pipeline/` (core) | ~43,000 | 90.7% |
| `scripts/` (standalone) | 3,230 | 6.8% |
| `moe/` (en scripts/) | 1,356 | 2.9% |
| `tests/` | 311 | 0.7% |

> **Problema:** el paquete `moe/` (la pieza central de la arquitectura MoE) vive dentro de `scripts/scripts_amigos/`, no en `src/pipeline/`. Esto sugiere que fue desarrollado como prototipo y no se integró formalmente.

### 3.4 Centralización

| Patrón | Presente |
|---|---|
| Hub central (`config.py`) | ✅ Sí — 55 archivos lo importan |
| Pipeline runner (`run_pipeline.py`) | ✅ Existe pero es órfano |
| Configs por fase | ✅ `fase1_config`, `fase2_config`, `fase5_config`, `fase6_config` |
| Registry de expertos | ❌ No hay registry centralizado |

> **Patrón:** arquitectura semi-centralizada. `config.py` actúa como hub de constantes, pero no hay un orquestador real que conecte las fases.

### 3.5 Fan-in / Fan-out por módulo

| Módulo | Fan-in (quién lo importa) | Fan-out (qué importa) | Clasificación |
|---|---|---|---|
| `config.py` | 55 | Bajo (stdlib) | **Hub central** |
| `datasets/*` | 18 | 10 | Proveedor de datos |
| `fase1_config` | 12 | Bajo | Config hub |
| `fase2_config` | 9 | Bajo | Config hub |
| `logging_utils` | 7 | Bajo | Utilidad |
| `ddp_utils` | 7 | ~5 | Utilidad DDP |
| `losses` | 6 | ~3 | Utilidad training |
| `moe_model` | 4 | ~8 | Integrador MoE |
| `fase0/` (total) | Bajo | 31 únicos | **Alto fan-out** |
| `fase1/` (total) | Medio | 36 únicos | **Alto fan-out** |
| `fase2/` (total) | Medio | 38 únicos | **Mayor fan-out** |
| `fase6/` (total) | Bajo | 18 únicos | Consumidor final |

---

## 4. Distribución de Responsabilidades

### 4.1 LOC por categoría funcional

| Categoría | Módulos | LOC | % |
|---|---|---|---|
| **Training/ML** | fase2, fase3, fase5 | 19,638 | 41.4% |
| **Data Processing** | fase0, fase1, datasets | 16,091 | 33.9% |
| **Evaluation/Deploy** | fase6 | 6,709 | 14.2% |
| **MoE System** | moe/ | 1,356 | 2.9% |
| **Utilities** | config, logging, scripts | 3,295 | 7.0% |
| **Tests** | tests/ | 311 | 0.7% |
| **Total** | | **47,400** | **100%** |

```
[Pie Chart — Composición LOC por categoría]

  Training/ML      ████████████████████░░░░░░░░░░░  41.4%
  Data Processing   █████████████████░░░░░░░░░░░░░░  33.9%
  Evaluation/Deploy ███████░░░░░░░░░░░░░░░░░░░░░░░░  14.2%
  Utilities         ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7.0%
  MoE System        █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.9%
  Tests             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.7%
```

```
[Bar Chart — Código vs Datos vs Checkpoints (GB)]

  Datos (.npy)     ████████████████████████████████████████ 46.0 GB
  Checkpoints      █                                        0.83 GB
  Código (.py)     ▏                                        0.002 GB
  Notebooks        ▏                                        0.008 GB
```

### 4.2 Equilibrio

- **Training (41%) + Data (34%) = 75%** del código. Esperable para un proyecto ML.
- **Tests (0.7%)** es preocupantemente bajo. Solo 311 LOC de tests para 47,400 LOC de código = **ratio 1:152**.
- **MoE (2.9%)** es sorprendentemente bajo dado que es la pieza central de la tesis.

### 4.3 Scripts críticos vs opcionales

**Críticos** (sin ellos el pipeline no funciona):

| Script | Rol | LOC |
|---|---|---|
| `config.py` | Constantes globales | ~200 |
| `fase0_pipeline.py` | Orquestador fase 0 | 992 |
| `fase1_pipeline.py` | Orquestador fase 1 | 1,127 |
| `train_expert*_ddp.py` (×5) | Training de cada experto | ~5,200 |
| `moe_model.py` | Modelo MoE central | ~250 |
| `datasets/*.py` | Cargadores de datos | 3,676 |
| `models/*.py` (fase2) | Arquitecturas de expertos | ~1,500 |

**Opcionales** (nice-to-have):

| Script | Rol |
|---|---|
| `paso11_webapp.py` | Demo web con Gradio |
| `paso12_dashboard.py` | Dashboard de métricas |
| `ablation_runner.py` | Ablation studies |
| `audit_dataset.py` | Auditoría de datos |
| `create_augmented_train.py` | Aumentación offline |
| `run_dryrun_rtx4090.py` | Dry-run hardware específico |

### 4.4 Scripts huérfanos (no importados por nadie)

Total: **26 scripts** huérfanos. Esto es **esperable** en un proyecto ML donde muchos scripts son entry-points ejecutados directamente (`python train_expert1_ddp.py`), no módulos importados.

Huérfanos que sí son entry-points legítimos:
- `fase0_pipeline.py`, `fase1_pipeline.py`, `fase2_pipeline.py` — orquestadores
- `train_expert*_ddp.py` — scripts de training
- `paso10_verificacion.py`, `paso11_webapp.py`, `paso12_dashboard.py` — evaluación/demo
- `run_pipeline.py` — runner global

Huérfanos potencialmente muertos:
- `expert1_config_rtx4090.py` — config hardware-específico
- `run_dryrun_rtx4090.py` — idem

---

## 5. Tipología de Archivos

### 5.1 Ratio código:datos

| Tipo | Tamaño |
|---|---|
| Código (.py + .ipynb) | 9.7 MB |
| Datos (.npy) | 46 GB |
| Checkpoints | 0.83 GB |
| **Ratio código:datos** | **1:4,742** |

> Por cada MB de código hay ~4.7 GB de datos. Típico de proyectos de deep learning con embeddings pre-calculados.

### 5.2 Scripts de alto impacto (muchas dependencias entrantes)

| Script | Fan-in | Impacto |
|---|---|---|
| `config.py` | 55 | 🔴 Crítico — un cambio afecta casi todo |
| `datasets/*` | 18 | 🟠 Alto — cambios en loaders afectan training |
| `fase1_config.py` | 12 | 🟡 Medio |
| `ddp_utils.py` | 7 | 🟡 Medio |
| `logging_utils.py` | 7 | 🟢 Bajo riesgo (utilidad estable) |

### 5.3 Notebook vs Script: dónde está la lógica crítica

| Ubicación | Lógica crítica | LOC |
|---|---|---|
| `src/pipeline/*.py` | ✅ **Toda la lógica crítica** | 43,000+ |
| `notebooks/*.ipynb` | EDA exploratorio solamente | ~2,000 |
| `scripts/*.ipynb` | Training notebooks (duplicados de .py) | ~3,000 |

> **Veredicto:** la lógica crítica está correctamente en `.py`, no en notebooks. Los notebooks son auxiliares para exploración. Buena práctica.

---

## 6. Evaluación de Escalabilidad y Mantenibilidad

### 6.1 Scorecard

| Dimensión | Puntuación | Justificación |
|---|---|---|
| Modularización | 7/10 | Fases bien separadas, pero moe/ fuera de src/ |
| Acoplamiento | 8/10 | Bajo acoplamiento cross-fase |
| Cohesión | 7/10 | Buena en general, fase6 mezcla responsabilidades |
| Testabilidad | 2/10 | 311 LOC de tests para 47K LOC de código |
| Reproducibilidad | 4/10 | Requirements inflado, no hay lockfile limpio |
| Documentación código | 5/10 | Configs documentadas, funciones poco documentadas |
| Escalabilidad | 6/10 | Agregar un experto requiere ~5 archivos nuevos |

### 6.2 Fortalezas

1. **Separación por fases** clara y lógica (fase0→fase1→...→fase6)
2. **Hub de configuración** centralizado (`config.py`) evita magic numbers dispersos
3. **Soporte DDP** desde el inicio — preparado para multi-GPU
4. **Lógica en .py, exploración en .ipynb** — buena separación de concerns
5. **Bajo acoplamiento** entre fases — cada una puede evolucionar independientemente

### 6.3 Debilidades

1. **Tests casi inexistentes** (ratio 1:152) — el mayor riesgo del proyecto
2. **`moe/` vive en scripts/** en vez de en `src/pipeline/` — arquitectura incompleta
3. **Requirements.txt inflado** con ~180 paquetes, muchos no usados (tensorflow, whisper, ray)
4. **No hay registry de expertos** — agregar uno nuevo requiere editar múltiples archivos manualmente
5. **Duplicación**: versiones DDP y no-DDP de cada trainer (`train_expert1.py` + `train_expert1_ddp.py`)
6. **`config.py` con fan-in de 55** — un solo punto de fallo para todo el proyecto

### 6.4 Recomendaciones arquitectónicas

| Prioridad | Acción | Impacto |
|---|---|---|
| 🔴 Alta | Mover `moe/` de `scripts/` a `src/pipeline/` | Integridad arquitectónica |
| 🔴 Alta | Agregar tests unitarios (target: ratio 1:10 mínimo) | Confiabilidad |
| 🟠 Media | Limpiar `requirements.txt` — separar en core vs dev vs optional | Reproducibilidad |
| 🟠 Media | Crear expert registry para agregar expertos declarativamente | Escalabilidad |
| 🟡 Baja | Unificar trainers DDP/no-DDP con flag | Reducir duplicación |
| 🟡 Baja | Separar fase6 en `evaluation/` y `deployment/` | Cohesión |

---

## 7. Resumen Ejecutivo

| Métrica | Valor |
|---|---|
| Archivos Python | 133 |
| LOC totales | 47,400 |
| Notebooks | 14 |
| Datos binarios | 46 GB (.npy) + 485 MB (.zip) |
| Checkpoints | 341 MB |
| Tamaño total proyecto | 545 GB |
| Dependencias externas | ~180 (requirements.txt) |
| Librerías core usadas | ~15 |
| Fan-in máximo | 55 (config.py) |
| Tests LOC | 311 (0.7%) |
| Ratio código:datos | 1:4,742 |
| Scripts huérfanos | 26 (mayoría son entry-points legítimos) |

**Veredicto general:** la arquitectura es funcional y razonablemente bien organizada para un proyecto de investigación. Las fases están bien separadas y el acoplamiento es bajo. Las debilidades principales son la falta de tests, el paquete `moe/` desplazado, y un `requirements.txt` que necesita limpieza. Para pasar de prototipo de investigación a sistema productivo, las prioridades son: tests, integración formal de `moe/`, y limpieza de dependencias.
