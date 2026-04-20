# Familias Funcionales — Scripts LUNA16 (Puntos 1-15)

> Generado: 2026-04-19 | Agrupación por función real, no por estructura de carpetas

---

## Tabla Resumen

| Familia | Scripts | Descripción breve |
|---------|---------|-------------------|
| A: Extracción y Preparación de Parches 3D | 4 scripts | Orquesta la extracción de parches 64³ desde CTs crudos, aplica zero-centering, augmentation offline y auditoría de calidad |
| B: Dataset y Transformaciones de Datos | 2 scripts | Provee la interfaz de carga de datos en runtime: Dataset PyTorch, augmentation inline, normalización HU y transformaciones 3D |
| C: Arquitectura del Modelo | 2 scripts | Define las dos variantes de DenseNet 3D: el backbone genérico de Fase 1 y el modelo específico de Expert 3 |
| D: Configuración e Hiperparámetros | 1 script | Centraliza todos los hiperparámetros de Expert 3 como fuente única de verdad |
| E: Infraestructura de Entrenamiento | 3 scripts | Construye DataLoaders, implementa FocalLoss y utilidades DDP para entrenamiento distribuido multi-GPU |
| F: Ejecución del Entrenamiento | 3 scripts | Scripts que ejecutan el loop de entrenamiento: versión DDP, versión single-GPU legacy, y launcher shell |

---

## Familia A: Extracción y Preparación de Parches 3D

**Scripts en esta familia (4):**
- `src/pipeline/fase0/fase0_pipeline.py` — Orquestador de Fase 0; invoca la extracción de parches y sub-pasos via CLI
- `src/pipeline/fase0/pre_embeddings.py` — Extrae parches 3D (64³) desde CTs LUNA16, incluye zero-centering y augmentation offline
- `src/pipeline/fase0/create_augmented_train.py` — Genera set aumentado offline copiando negativos y augmentando positivos (ratio 10:1 → ~2:1)
- `src/pipeline/fase0/audit_dataset.py` — Audita integridad de parches: shape, dtype, zero-centering, balance de clases, duplicados

**Qué hace esta familia:**
Transforma los CTs crudos de LUNA16 (archivos `.mhd`/`.raw`) en parches volumétricos de 64×64×64 listos para entrenamiento. El orquestador (`fase0_pipeline.py`) coordina todo el proceso: `pre_embeddings.py` realiza la extracción real de parches con zero-centering y cálculo de `global_mean.npy`, `create_augmented_train.py` opcionalmente balancea las clases mediante augmentation offline, y `audit_dataset.py` valida la integridad del dataset resultante antes de pasar a entrenamiento.

**Input:** CTs `.mhd` en `datasets/luna_lung_cancer/ct_volumes/`, `candidates_V2.csv`, `luna_splits.json`, máscaras `seg-lungs-LUNA16/`
**Output:** `patches/{train,val,test}/candidate_XXXXXX.npy` (float32 64³), `global_mean.npy`, manifests, `audit_report.json`, opcionalmente `patches/train_aug/`
**Orden de ejecución:** `fase0_pipeline.py` → `pre_embeddings.py` → `audit_dataset.py` → `create_augmented_train.py` (opcional)

---

## Familia B: Dataset y Transformaciones de Datos

**Scripts en esta familia (2):**
- `src/pipeline/datasets/luna.py` — Módulo de dataset/utilidades: `LUNA16Dataset`, `LUNA16PatchExtractor`, `LUNA16FROCEvaluator`, augmentation 3D inline
- `src/pipeline/fase1/transform_3d.py` — Normalización HU, resize trilineal, proyección multiplanar 3D→2D

**Qué hace esta familia:**
Proporciona la capa de abstracción entre los parches en disco y el modelo durante entrenamiento/inferencia. `luna.py` implementa el `Dataset` PyTorch que carga parches `.npy`, aplica augmentation 3D online y produce tensores `[B,1,64,64,64]`. `transform_3d.py` maneja la normalización de valores Hounsfield y transformaciones volumétricas que `luna.py` consume internamente. Juntos permiten que el DataLoader entregue datos listos al modelo.

**Input:** Parches `.npy`, `candidates_V2.csv`, `annotations.csv`
**Output:** Tensores PyTorch `[B,1,64,64,64]`, submissions FROC
**Orden de ejecución:** `transform_3d.py` es importado por `luna.py` (no se ejecutan secuencialmente sino como librería)

---

## Familia C: Arquitectura del Modelo

**Scripts en esta familia (2):**
- `src/pipeline/fase1/backbone_densenet3d.py` — DenseNet-121 3D (~11.14M params), block_config=[6,12,24,16], backbone genérico de Fase 1
- `src/pipeline/fase2/models/expert3_densenet3d.py` — DenseNet 3D custom (~6.7M params), blocks=[4,8,16,12], compression=0.5, clases `Expert3DenseNet3D` / `Expert3MC318`

**Qué hace esta familia:**
Define las arquitecturas neuronales DenseNet 3D del proyecto. `backbone_densenet3d.py` es la implementación más grande (~11M params) usada como backbone compartido en Fase 1 para generar embeddings. `expert3_densenet3d.py` es la versión compacta (~6.7M params) optimizada específicamente para Expert 3, con bloques más pequeños y mayor compresión. Ambas reciben tensores `[B,1,64,64,64]` y producen logits, pero se usan en contextos diferentes del pipeline.

**Input:** Tensor `[B,1,64,64,64]`
**Output:** Logits `[B, num_classes]` (backbone) o `[B,2]` (expert3)
**Orden de ejecución:** No aplica — son definiciones de arquitectura importadas por scripts de entrenamiento

---

## Familia D: Configuración e Hiperparámetros

**Scripts en esta familia (1):**
- `src/pipeline/fase2/expert3_config.py` — Fuente de verdad: LR=3e-4, WD=0.03, FocalLoss(γ=2, α=0.85), batch=8, accum=4, patience=20

**Qué hace esta familia:**
Centraliza todos los hiperparámetros y constantes de Expert 3 en un único módulo Python. Actúa como fuente de verdad que los scripts de entrenamiento, DataLoader y loss function consultan, garantizando consistencia entre ejecuciones y facilitando experimentación al tener un solo punto de cambio.

**Input:** Ninguno
**Output:** Constantes Python accesibles por importación
**Orden de ejecución:** No aplica — es un módulo de configuración importado por otros scripts

---

## Familia E: Infraestructura de Entrenamiento

**Scripts en esta familia (3):**
- `src/pipeline/fase2/dataloader_expert3.py` — `LUNA16ExpertDataset` y `build_dataloaders_expert3()`, escanea `.npy` en disco (~1s init)
- `src/pipeline/fase2/losses.py` — `FocalLoss(gamma, alpha)` binaria para Expert 3 (y Expert 4)
- `src/pipeline/fase2/ddp_utils.py` — Utilidades DDP reutilizables: `setup_ddp()`, `wrap_model_ddp()`, `get_ddp_dataloader()`

**Qué hace esta familia:**
Provee los componentes auxiliares que el loop de entrenamiento necesita pero que no son ni el modelo ni los datos en sí. `dataloader_expert3.py` construye los DataLoaders PyTorch específicos de Expert 3 usando la librería `luna.py`. `losses.py` implementa FocalLoss para manejar el desbalance de clases (mayoría negativos). `ddp_utils.py` abstrae la complejidad del entrenamiento distribuido multi-GPU. Los tres son dependencias directas de los scripts de entrenamiento.

**Input:** Parches `patches/{train,val,test}/`, `candidates_V2.csv`
**Output:** DataLoaders PyTorch, scalar loss, funciones helper DDP
**Orden de ejecución:** No aplica — son módulos importados simultáneamente por los scripts de entrenamiento

---

## Familia F: Ejecución del Entrenamiento

**Scripts en esta familia (3):**
- `src/pipeline/fase2/train_expert3_ddp.py` — Script principal: DenseNet3D from-scratch, AdamW + CosineAnnealingWarmRestarts, FocalLoss, early stopping, FP16, multi-GPU
- `src/pipeline/fase2/train_expert3.py` — Versión single-GPU (legacy, reemplazado por DDP)
- `run_expert.sh` — Launcher shell: `bash run_expert.sh 3` ejecuta torchrun auto-detectando GPUs

**Qué hace esta familia:**
Ejecuta el entrenamiento end-to-end de Expert 3. `run_expert.sh` es el punto de entrada que lanza `train_expert3_ddp.py` via `torchrun` con detección automática de GPUs. El script DDP orquesta el loop completo: carga modelo y datos usando las familias C/D/E, entrena con FP16 y early stopping, y guarda el mejor checkpoint. `train_expert3.py` es la versión legacy single-GPU mantenida como fallback.

**Input:** Parches `patches/{train,val,test}/`, `candidates_V2.csv`
**Output:** `checkpoints/expert_03_densenet3d/best.pt`, `expert3_ddp_training_log.json`
**Orden de ejecución:** `run_expert.sh` → `train_expert3_ddp.py` (o alternativamente `train_expert3.py` directo)

---

## Diagrama de Flujo entre Familias

```
CTs raw (.mhd) + candidates_V2.csv + luna_splits.json
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │  FAMILIA A: Extracción y            │
  │  Preparación de Parches 3D          │
  │  (4 scripts)                        │
  └──────────────────┬──────────────────┘
                     │
                     ▼
          patches/*.npy + global_mean.npy
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
  ┌───────────┐ ┌──────────┐ ┌──────────────┐
  │FAMILIA B  │ │FAMILIA C │ │FAMILIA D     │
  │Dataset +  │ │Arquitec- │ │Configuración │
  │Transform  │ │tura      │ │(1 script)    │
  │(2 scripts)│ │(2 scripts│ └──────┬───────┘
  └─────┬─────┘ └────┬─────┘        │
        │            │              │
        └────────────┼──────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  FAMILIA E: Infraestructura de      │
  │  Entrenamiento (3 scripts)          │
  │  DataLoaders + FocalLoss + DDP      │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  FAMILIA F: Ejecución del           │
  │  Entrenamiento (3 scripts)          │
  └──────────────────┬──────────────────┘
                     │
                     ▼
          best.pt + training_log.json
```

**Flujo resumido:**
```
Familia A (CTs raw) → patches .npy
    → Familias B+C+D (librerías/config, cargadas en memoria)
        → Familia E (ensambla DataLoaders + Loss + DDP)
            → Familia F (ejecuta entrenamiento) → best.pt
```

---

## Verificación de Cobertura

| # | Script | Familia |
|---|--------|---------|
| 1 | `src/pipeline/fase0/pre_embeddings.py` | A |
| 2 | `src/pipeline/fase0/fase0_pipeline.py` | A |
| 3 | `src/pipeline/fase0/create_augmented_train.py` | A |
| 4 | `src/pipeline/fase0/audit_dataset.py` | A |
| 5 | `src/pipeline/datasets/luna.py` | B |
| 6 | `src/pipeline/fase1/transform_3d.py` | B |
| 7 | `src/pipeline/fase1/backbone_densenet3d.py` | C |
| 8 | `src/pipeline/fase2/expert3_config.py` | D |
| 9 | `src/pipeline/fase2/models/expert3_densenet3d.py` | C |
| 10 | `src/pipeline/fase2/dataloader_expert3.py` | E |
| 11 | `src/pipeline/fase2/losses.py` | E |
| 12 | `src/pipeline/fase2/ddp_utils.py` | E |
| 13 | `src/pipeline/fase2/train_expert3_ddp.py` | F |
| 14 | `src/pipeline/fase2/train_expert3.py` | F |
| 15 | `run_expert.sh` | F |

**Total: 15 scripts → 6 familias → 0 scripts sin asignar ✓**
