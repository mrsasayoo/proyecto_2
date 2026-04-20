# Inventario Completo de Scripts — Expert 3 (LUNA16 / DenseNet3D)

> Generado: 2026-04-19 | Basado en análisis directo del código fuente

---

## 1. Tabla Maestra

| # | Ruta relativa | Tipo | Descripción | Input | Output | Dependencias | Estado |
|---|--------------|------|-------------|-------|--------|-------------|--------|
| 1 | `src/pipeline/fase0/pre_embeddings.py` | .py | Extracción de parches 3D (64³) desde CTs LUNA16. Función `run_luna_patches()`. Incluye zero-centering (paso 6b) y augmentation offline (paso 6c). | CTs `.mhd` en `datasets/luna_lung_cancer/ct_volumes/`, `candidates_V2.csv`, `luna_splits.json`, máscaras `seg-lungs-LUNA16/` | `datasets/luna_lung_cancer/patches/{train,val,test}/candidate_XXXXXX.npy` (64³ float32), `global_mean.npy`, manifests | SimpleITK, scipy, numpy, `luna_splits.json` | **Activo** — Fase 0 |
| 2 | `src/pipeline/fase0/fase0_pipeline.py` | .py | Orquestador de Fase 0. Paso 7 invoca `run_luna_patches()` y sub-pasos 6b/6c/6d. | CLI args (`--solo luna`), datasets raw | Delega a `pre_embeddings.py` | `pre_embeddings.py`, `descargar.py`, `extraer.py` | **Activo** — Orquestador |
| 3 | `src/pipeline/fase0/create_augmented_train.py` | .py | Crea set de entrenamiento aumentado (offline). Copia negativos, genera augmentations de positivos para reducir ratio de 10:1 a ~2:1. | `patches/train/` + `candidates_V2.csv` | `patches/train_aug/`, `train_aug_manifest.csv`, `train_aug_report.json` | numpy | **Activo** — Opcional |
| 4 | `src/pipeline/fase0/audit_dataset.py` | .py | Auditoría de integridad de parches LUNA16: shape, dtype, zero-centering, balance de clases, duplicados, `global_mean.npy`. | `patches/{train,val,test}/` | `audit_report.json` | numpy, pandas | **Activo** — QA |
| 5 | `src/pipeline/datasets/luna.py` | .py | Módulo de dataset/utilidades LUNA16: `LUNA16Dataset`, `LUNA16PatchExtractor`, `LUNA16FROCEvaluator`, `verify_hu_normalization()`. Augmentation 3D inline. | Parches .npy, `candidates_V2.csv`, `annotations.csv` | Tensores [B,1,64,64,64], submissions FROC | SimpleITK, scipy, pandas, `config.py`, `fase1.transform_3d` | **Activo** — Librería |
| 6 | `src/pipeline/fase1/transform_3d.py` | .py | Procesamiento de volúmenes CT 3D: normalización HU, resize trilineal, proyección multiplanar 3D→2D para ViT. | Volumen numpy [D,H,W] | Tensor normalizado | numpy, torch, `fase1_config.py` | **Activo** — Compartido (LUNA16 + Páncreas) |
| 7 | `src/pipeline/fase1/backbone_densenet3d.py` | .py | Implementación DenseNet-121 3D (~11.14M params). Backbone alternativo para Fase 1 (embeddings). block_config=[6,12,24,16]. | Tensor [B,1,64,64,64] | Logits [B, num_classes] | torch | **Activo** — Compartido (Fase 1 backbone) |
| 8 | `src/pipeline/fase2/expert3_config.py` | .py | Fuente de verdad de hiperparámetros Expert 3: LR=3e-4, WD=0.03, FocalLoss(γ=2,α=0.85), batch=8, accum=4, patience=20, etc. | — | Constantes Python | — | **Activo** — Config |
| 9 | `src/pipeline/fase2/models/expert3_densenet3d.py` | .py | Modelo DenseNet 3D custom (~6.7M params). growth_rate=32, blocks=[4,8,16,12], compression=0.5. Clases `Expert3DenseNet3D` / `Expert3MC318`. | Tensor [B,1,64,64,64] | Logits [B,2] | torch | **Activo** — Modelo |
| 10 | `src/pipeline/fase2/dataloader_expert3.py` | .py | Dataset y DataLoader para Expert 3. `LUNA16ExpertDataset` (escanea .npy en disco, ~1s init). `build_dataloaders_expert3()`. | `patches/{train,val,test}/`, `candidates_V2.csv` | DataLoaders PyTorch | `datasets.luna` (augmentation), `expert3_config` | **Activo** — DataLoader |
| 11 | `src/pipeline/fase2/losses.py` | .py | `FocalLoss(gamma, alpha)` binaria usada por Expert 3 (y Expert 4). | Logits + targets | Scalar loss | torch | **Activo** — Compartido |
| 12 | `src/pipeline/fase2/ddp_utils.py` | .py | Utilidades DDP reutilizables para los 5 expertos: `setup_ddp()`, `wrap_model_ddp()`, `get_ddp_dataloader()`, etc. | — | Funciones helper | torch.distributed | **Activo** — Compartido (todos los expertos) |
| 13 | `src/pipeline/fase2/train_expert3_ddp.py` | .py | **Script principal de entrenamiento DDP.** DenseNet3D from-scratch, AdamW + CosineAnnealingWarmRestarts, FocalLoss, early stopping, FP16. Multi-GPU con torchrun. | Parches en `patches/{train,val,test}/`, `candidates_V2.csv` | `checkpoints/expert_03_densenet3d/best.pt`, `expert3_ddp_training_log.json` | `expert3_densenet3d`, `dataloader_expert3`, `expert3_config`, `losses`, `ddp_utils` | **Activo** — Entrenamiento principal |
| 14 | `src/pipeline/fase2/train_expert3.py` | .py | Script de entrenamiento single-GPU (versión pre-DDP). Mismo pipeline que el DDP pero sin DistributedDataParallel. | Idem #13 | `expert3_best.pt`, `expert3_training_log.json` | `expert3_densenet3d`, `dataloader_expert3`, `expert3_config`, `losses` | **Legacy** — Reemplazado por DDP |
| 15 | `run_expert.sh` | .sh | Lanzador shell para todos los expertos. `bash run_expert.sh 3` lanza Expert 3 con torchrun auto-detectando GPUs. | Expert ID + opciones | Ejecuta torchrun | `train_expert3_ddp.py` | **Activo** — Launcher |
| 16 | `notebooks/evaluacion_expert3_densenet3d.ipynb` | .ipynb | Evaluación completa post-training: métricas, confusion matrix, curvas, análisis de errores. | `best.pt`, parches test, `training_log.json` | Gráficos, métricas, tablas | `expert3_densenet3d.py` | **Activo** — Evaluación |
| 17 | `scripts/evaluacion_expert3_densenet3d.ipynb` | .ipynb | Copia/mirror del notebook de evaluación (en `scripts/`). | Idem #16 | Idem #16 | Idem #16 | **Copia** de #16 |
| 18 | `notebooks/train_expert3_luna16.ipynb` | .ipynb | Notebook autosuficiente para entrenar Expert 3 (alternativa a CLI). Setup + extracción + entrenamiento completo. | Dataset raw o parches | Checkpoint, métricas | Autocontenido | **Activo** — Alternativa notebook |
| 19 | `scripts/train_expert3_luna16.ipynb` | .ipynb | Copia/mirror del notebook de entrenamiento. | Idem #18 | Idem #18 | Idem #18 | **Copia** de #18 |
| 20 | `checkpoints/expert_03_densenet3d/luna16-online-aug.ipynb` | .ipynb | Notebook Kaggle para entrenamiento con online augmentation. Diseñado para ejecutar en Kaggle con dataset pre-subido. | Parches .npy en Kaggle | Checkpoint | Autocontenido (Kaggle) | **Activo** — Entorno Kaggle |
| 21 | `scripts/find_fn_test.py` | .py | Identifica los 26 False Negatives del test set. Carga modelo, evalúa test, encuentra predicciones incorrectas. | `best.pt`, parches test, `candidates_V2.csv` | `false_negatives_test.txt` | `expert3_densenet3d` | **Activo** — Análisis post-hoc |
| 22 | `notebooks/evaluacion_expert3/false_negatives_visualization/visualize_fn.py` | .py | Visualización 3D interactiva de los False Negatives. Genera HTML con renders volumétricos. | Parches test FN | HTMLs 3D interactivos (`candidate_XXXXXX_3d.html`) | plotly/matplotlib | **Activo** — Visualización |
| 23 | `src/notebooks/luna_lung_cancer/000_eda.ipynb` | .ipynb | EDA del dataset LUNA16: distribuciones, estadísticas, visualizaciones de CTs y candidatos. | Dataset raw LUNA16 | Gráficos exploración | pandas, matplotlib | **Activo** — Exploración |
| 24 | `docs/exportaciones/export_training_data.sh` | .sh | Script de exportación/backup de datos de entrenamiento. Empaqueta parches LUNA16 en `luna16.7z` (~17.3 GB). | `patches/{train,val,test}/`, `candidates_V2.csv` | `luna16.7z` | 7z | **Activo** — DevOps |
| 25 | `scripts/scripts_amigos/moe/experts/archs/densenet3d_luna.py` | .py | Arquitectura DenseNet3D reconstruida para inferencia MoE. Clase `DenseNet3DLUNA` — reverse-engineered del checkpoint. | — | Modelo para inferencia | torch | **Activo** — Inferencia MoE |
| 26 | `scripts/scripts_amigos/moe/experts/loaders.py` | .py | Loader de expertos para MoE. `build_exp4_luna()` carga `LUNA-LIDCIDRI_best.pt.zip` con `DenseNet3DLUNA`. | Checkpoint `.pt.zip` | Modelo cargado | `densenet3d_luna.py` | **Activo** — Inferencia MoE |
| 27 | `datasets/luna_lung_cancer/luna_splits.json` | .json | Splits train/val/test por seriesuid. Fuente de verdad para partición reproducible. | — | Split definitions | — | **Activo** — Data |

---

## 2. Flujo de Ejecución (Secuencia Ordenada)

```
PASO 0: EDA y Exploración
─────────────────────────
  Script: src/notebooks/luna_lung_cancer/000_eda.ipynb
  Comando: jupyter notebook (manual)
  Input: Dataset raw LUNA16 (CTs .mhd + candidates_V2.csv + annotations.csv)
  Output: Comprensión del dataset, estadísticas, visualizaciones
  Tiempo: ~30 min (manual)

PASO 1: Descarga y Extracción de CTs
─────────────────────────────────────
  Script: src/pipeline/fase0/fase0_pipeline.py (pasos 1-2)
  Comando: python src/pipeline/fase0/fase0_pipeline.py --solo luna --paso 1 2
  Input: URLs de subsets LUNA16 (subset0-subset9)
  Output: datasets/luna_lung_cancer/ct_volumes/subset{0..9}/*.mhd + .raw
  Tiempo: ~2-4 horas (descarga ~120 GB)

PASO 2: Extracción de Parches 3D (64³)
───────────────────────────────────────
  Script: src/pipeline/fase0/pre_embeddings.py → run_luna_patches()
  Comando: python src/pipeline/fase0/fase0_pipeline.py --solo luna --paso 7
  Input: CTs .mhd, candidates_V2.csv, luna_splits.json, seg-lungs-LUNA16/
  Output: patches/{train,val,test}/candidate_XXXXXX.npy (64×64×64 float32)
         + global_mean.npy + manifests
  Tiempo: ~1-2 horas (CPU-bound, I/O intensivo)
  Nota: Incluye sub-pasos:
    6b: Zero-centering (resta global_mean del train)
    6c: Augmentation offline (si habilitada)
    6d: Auditoría automática

PASO 3: Auditoría de Parches (QA)
──────────────────────────────────
  Script: src/pipeline/fase0/audit_dataset.py
  Comando: python src/pipeline/fase0/audit_dataset.py
  Input: patches/{train,val,test}/
  Output: audit_report.json
  Tiempo: ~5 min

PASO 4: Augmentation Offline (Opcional)
───────────────────────────────────────
  Script: src/pipeline/fase0/create_augmented_train.py
  Comando: python src/pipeline/fase0/create_augmented_train.py
  Input: patches/train/, candidates_V2.csv
  Output: patches/train_aug/ (~18 GB), train_aug_manifest.csv
  Tiempo: ~30-60 min
  Nota: Reduce ratio neg:pos de 10:1 a ~2:1 con augmentations offline.
        El pipeline principal usa online augmentation, este paso es alternativo.

PASO 5: Entrenamiento Expert 3
──────────────────────────────
  Script: src/pipeline/fase2/train_expert3_ddp.py
  Comando: bash run_expert.sh 3
         (equivale a: torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py)
  Input: patches/{train,val,test}/, candidates_V2.csv
  Output: checkpoints/expert_03_densenet3d/best.pt
          checkpoints/expert_03_densenet3d/expert3_ddp_training_log.json
  Tiempo: ~6-18 horas (depende de GPUs y early stopping)
  Recursos: 2× Titan Xp (12 GB VRAM), FP16, batch_per_gpu=4, accum=4
  Alternativas:
    - train_expert3.py (single-GPU, legacy)
    - notebooks/train_expert3_luna16.ipynb (notebook)
    - checkpoints/expert_03_densenet3d/luna16-online-aug.ipynb (Kaggle)

PASO 6: Evaluación
──────────────────
  Script: notebooks/evaluacion_expert3_densenet3d.ipynb
  Comando: jupyter notebook (manual)
  Input: best.pt, patches/{val,test}/, training_log.json
  Output: Métricas (F1, AUC, confusion matrix), gráficos, curvas
  Tiempo: ~15 min

PASO 7: Análisis de Errores (Post-hoc)
───────────────────────────────────────
  Script: scripts/find_fn_test.py
  Comando: python scripts/find_fn_test.py
  Input: best.pt, patches/test/, candidates_V2.csv
  Output: notebooks/evaluacion_expert3/false_negatives_test.txt
  Tiempo: ~5 min

  Script: notebooks/evaluacion_expert3/false_negatives_visualization/visualize_fn.py
  Comando: python notebooks/evaluacion_expert3/false_negatives_visualization/visualize_fn.py
  Input: Parches FN del test set
  Output: HTMLs interactivos 3D
  Tiempo: ~10 min

PASO 8: Exportación y Backup
────────────────────────────
  Script: docs/exportaciones/export_training_data.sh
  Comando: bash docs/exportaciones/export_training_data.sh
  Input: patches/{train,val,test}/, candidates_V2.csv
  Output: luna16.7z (~17.3 GB)
  Tiempo: ~30 min
```

---

## 3. Gráfico de Dependencias

```
datasets/luna_lung_cancer/
├── ct_volumes/subset{0..9}/   ← PASO 1 (descarga)
├── candidates_V2/candidates_V2.csv
├── annotations.csv
├── luna_splits.json            ← Split definitions
├── seg-lungs-LUNA16/           ← Máscaras de segmentación
└── patches/                    ← PASO 2 (extracción)
    ├── train/candidate_*.npy
    ├── val/candidate_*.npy
    ├── test/candidate_*.npy
    ├── train_aug/              ← PASO 4 (opcional)
    └── global_mean.npy

                    ┌─────────────────────┐
                    │ CTs raw + CSV + JSON │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  pre_embeddings.py   │ (Paso 2: extracción parches)
                    │  run_luna_patches()  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  patches .npy 64³   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼───────┐ ┌─────▼─────┐ ┌────────▼────────┐
    │ audit_dataset.py│ │create_aug │ │                  │
    │    (QA)         │ │(opcional) │ │                  │
    └─────────────────┘ └───────────┘ │                  │
                                      ▼
                          ┌───────────────────────┐
                          │ train_expert3_ddp.py   │ (Paso 5)
                          │                        │
                          │ usa:                   │
                          │  ├ expert3_config.py    │
                          │  ├ expert3_densenet3d.py│
                          │  ├ dataloader_expert3.py│
                          │  │  └ datasets/luna.py  │
                          │  ├ losses.py (FocalLoss)│
                          │  └ ddp_utils.py         │
                          └───────────┬─────────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ best.pt + log.json     │
                          └───────────┬───────────┘
                                      │
              ┌───────────────────────┼───────────────────┐
              │                       │                   │
    ┌─────────▼──────────┐  ┌────────▼────────┐  ┌───────▼──────────┐
    │evaluacion_expert3  │  │find_fn_test.py  │  │loaders.py (MoE) │
    │_densenet3d.ipynb   │  │                 │  │build_exp4_luna() │
    └────────────────────┘  └────────┬────────┘  └──────────────────┘
                                     │
                            ┌────────▼────────┐
                            │visualize_fn.py  │
                            └─────────────────┘
```

### Dependencias por Script

| Script | Depende de |
|--------|-----------|
| `pre_embeddings.py` | SimpleITK, scipy, `luna_splits.json`, CTs raw |
| `fase0_pipeline.py` | `pre_embeddings.py`, `descargar.py`, `extraer.py` |
| `create_augmented_train.py` | Parches extraídos, `candidates_V2.csv` |
| `audit_dataset.py` | Parches extraídos |
| `datasets/luna.py` | SimpleITK, scipy, `config.py`, `fase1/transform_3d.py` |
| `transform_3d.py` | numpy, torch, `fase1_config.py` |
| `expert3_config.py` | Ninguna |
| `expert3_densenet3d.py` | torch |
| `dataloader_expert3.py` | `datasets/luna.py`, `expert3_config.py` |
| `losses.py` | torch |
| `ddp_utils.py` | torch.distributed |
| `train_expert3_ddp.py` | `expert3_densenet3d`, `dataloader_expert3`, `expert3_config`, `losses`, `ddp_utils` |
| `train_expert3.py` | `expert3_densenet3d`, `dataloader_expert3`, `expert3_config`, `losses` |
| `run_expert.sh` | `train_expert3_ddp.py`, torchrun |
| `find_fn_test.py` | `expert3_densenet3d`, parches test, checkpoint |
| `visualize_fn.py` | Parches FN |

---

## 4. Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| **Total de scripts LUNA16-específicos** | 16 |
| **Scripts de infraestructura compartida** | 5 (`transform_3d.py`, `backbone_densenet3d.py`, `losses.py`, `ddp_utils.py`, `run_expert.sh`) |
| **Scripts MoE/inferencia** | 2 (`densenet3d_luna.py`, `loaders.py`) |
| **Archivos de datos** | 1 (`luna_splits.json`) |
| **Notebooks** | 5 (EDA, train×2 copias, eval×2 copias, Kaggle) |
| **Total general** | 27 entradas |

### Tiempos Estimados (pipeline completo desde cero)

| Fase | Tiempo estimado |
|------|----------------|
| Descarga CTs | 2-4 h |
| Extracción parches | 1-2 h |
| Auditoría + QA | 5 min |
| Augmentation offline (opcional) | 30-60 min |
| **Entrenamiento** | **6-18 h** |
| Evaluación + análisis | 30 min |
| **Total** | **~10-25 horas** |

### Recursos Necesarios

- **GPU**: 2× NVIDIA Titan Xp (12 GB VRAM cada una) — mínimo 1 GPU con 12 GB
- **RAM**: ~32 GB recomendado (extracción de parches carga CTs completos)
- **Storage**: ~150 GB (CTs raw ~120 GB + parches ~17 GB + augmentados ~18 GB)
- **CPU**: Multi-core recomendado para extracción paralela de parches

### Estado de Reproducibilidad

**Sí, se puede reentrenar desde cero** siempre que:
1. Se tengan los CTs originales de LUNA16 (subsets 0-9)
2. `candidates_V2.csv` y `luna_splits.json` estén presentes
3. Se ejecute el pipeline en orden: Fase 0 → Fase 2 → Evaluación
4. Semillas fijadas (`seed=42` + rank offset para DDP)
5. El script `run_expert.sh 3` automatiza el paso de entrenamiento

### Notas Importantes

- **Expert 3 vs Expert 4**: En el código MoE (`loaders.py`), lo que se llama "exp4_luna" internamente corresponde al Expert 3 del pipeline de entrenamiento. Es un tema de numeración diferente entre el pipeline de training y el de inferencia MoE.
- **DenseNet3D variantes**: Existen dos implementaciones:
  - `fase2/models/expert3_densenet3d.py`: ~6.7M params, blocks=[4,8,16,12] — **usado en entrenamiento**
  - `fase1/backbone_densenet3d.py`: ~11.14M params, blocks=[6,12,24,16] — backbone Fase 1 (embeddings)
  - `scripts_amigos/moe/experts/archs/densenet3d_luna.py`: Reconstrucción para inferencia MoE
- **Online vs Offline augmentation**: El pipeline principal usa online augmentation (en `LUNA16ExpertDataset`/`LUNA16Dataset._augment_3d`). `create_augmented_train.py` es una alternativa offline.
