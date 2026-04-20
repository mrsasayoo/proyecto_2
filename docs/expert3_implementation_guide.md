# Expert 3 — Guía de Implementación Técnica

> Documentación generada por análisis directo del código fuente. Sin especulaciones.
> Archivos analizados: `expert3_config.py`, `train_expert3_ddp.py`, `dataloader_expert3.py`, `expert3_densenet3d.py`, `luna.py`, `losses.py`, `expert3_ddp_training_log.json`.

---

## Sección 1: Arquitectura del Modelo

**Nombre:** `Expert3DenseNet3D` (alias: `Expert3MC318`)
**Archivo:** `src/pipeline/fase2/models/expert3_densenet3d.py`
**Tipo:** DenseNet 3D implementado from scratch (Huang et al., 2017) con Conv3d.

### Configuración por defecto

| Parámetro | Valor |
|---|---|
| `in_channels` | 1 (CT monocanal) |
| `num_classes` | 2 |
| `growth_rate` | 32 |
| `block_layers` | [4, 8, 16, 12] (4 dense blocks) |
| `init_features` | 64 |
| `bn_size` | 4 |
| `compression` | 0.5 |
| `spatial_dropout_p` | 0.15 |
| `fc_dropout_p` | 0.4 |

**Entrada:** `[B, 1, 64, 64, 64]` — parche CT monocanal float32
**Salida:** `[B, 2]` — logits crudos (no-nódulo, nódulo)
**Parámetros totales:** ~6.7M

### Componentes

#### Stem
```
Conv3d(1 → 64, kernel=7×7×7, stride=2, padding=3, bias=False)
→ BatchNorm3d(64)
→ ReLU(inplace=True)
→ MaxPool3d(kernel=3, stride=2, padding=1)
→ SpatialDropout3d(p=0.15)
```
Transforma `[B,1,64,64,64]` → `[B,64,16,16,16]`.

#### Dense Blocks (4 bloques)
Cada `_DenseBlock` contiene N `_DenseLayer` con patrón bottleneck:
```
BN3d → ReLU → Conv3d(1×1×1, in → bn_size*growth_rate) → BN3d → ReLU → Conv3d(3×3×3, bn_size*growth_rate → growth_rate)
```
La salida se concatena con la entrada (conectividad densa). Bloques: [4, 8, 16, 12] capas respectivamente.

#### Transition Layers (3, entre bloques)
```
BN3d → ReLU → Conv3d(1×1×1) → AvgPool3d(2×2×2)
```
Compresión 0.5: reduce canales a la mitad y resolución espacial ×2.

#### Cabeza clasificadora
```
BatchNorm3d(n_features_final) → ReLU
→ AdaptiveAvgPool3d(1) → Flatten
→ Dropout(p=0.4) → Linear(n_features_final, 2)
```

### Inicialización de pesos
- **Conv3d:** Kaiming normal (He et al., 2015), `mode="fan_out"`, `nonlinearity="relu"`
- **BatchNorm3d:** `weight=1.0`, `bias=0.0`
- **Linear:** `normal(mean=0.0, std=0.01)`, `bias=0`

### SpatialDropout3d
Envuelve `nn.Dropout3d(p)` — desactiva canales completos del feature map (no activaciones individuales). Más efectivo para datos volumétricos con correlación espacial.

---

## Sección 2: Configuración de Entrenamiento

**Archivo:** `src/pipeline/fase2/expert3_config.py`

| Parámetro | Variable | Valor | Nota |
|---|---|---|---|
| Learning rate | `EXPERT3_LR` | `3e-4` | AdamW |
| Weight decay | `EXPERT3_WEIGHT_DECAY` | `0.03` | Más alto que default (0.01) |
| Focal gamma | `EXPERT3_FOCAL_GAMMA` | `2.0` | Lin et al. 2017 |
| Focal alpha | `EXPERT3_FOCAL_ALPHA` | `0.85` | Peso clase positiva (minoritaria) |
| Label smoothing | `EXPERT3_LABEL_SMOOTHING` | `0.05` | {0,1} → {0.025, 0.975} |
| Dropout FC | `EXPERT3_DROPOUT_FC` | `0.4` | Antes de capa final |
| Spatial Dropout 3D | `EXPERT3_SPATIAL_DROPOUT_3D` | `0.15` | Canales completos |
| Batch size por GPU | `EXPERT3_BATCH_SIZE` | `8` | Se divide entre GPUs |
| Accumulation steps | `EXPERT3_ACCUMULATION_STEPS` | `4` | Batch efectivo = 8×4 = 32 (single-GPU) |
| FP16 | `EXPERT3_FP16` | `True` | Mixed precision obligatorio |
| Max epochs | `EXPERT3_MAX_EPOCHS` | `100` | |
| Early stopping patience | `EXPERT3_EARLY_STOPPING_PATIENCE` | `20` | |
| Early stopping monitor | `EXPERT3_EARLY_STOPPING_MONITOR` | `"val_loss"` | |
| Seed | `_SEED` | `42` | En train script |
| Min delta (ES) | `_MIN_DELTA` | `0.001` | Mejora mínima para progreso |

**Batch efectivo con 2 GPUs DDP:** `(8 // 2) × 2 × 4 = 32`

---

## Sección 3: Data Pipeline

### Dataset
**Clase:** `LUNA16ExpertDataset` en `dataloader_expert3.py`
**Fuente:** Parches `.npy` pre-extraídos en `datasets/luna_lung_cancer/patches/{train,val,test}/`
**Labels:** CSV `candidates_V2.csv` (754K filas), indexado por `candidate_index` extraído del nombre de archivo (`candidate_NNNNNN.npy`).

**Tamaño del dataset (documentado en config):**
- Train: 14,728 muestras (1,258 positivas, 13,470 negativas)
- Ratio desbalance: ~10.7:1

### Preprocesamiento (offline, ya en disco)
- **Tamaño de parche:** 64×64×64 vóxeles
- **Clipping HU:** [-1000, 400] (aire puro a hueso excluido)
- **Normalización:** a [0, 1], luego zero-centering con media global `0.09921630471944809`
- **Rango en disco:** `[-0.099, 0.901]` aproximadamente

### Carga en __getitem__
```python
volume = np.load(patch_file)           # float32 (64,64,64)
# augmentation 3D si train
volume_t = torch.from_numpy(volume).unsqueeze(0)  # [1, 64, 64, 64]
```
**No hay normalización adicional en runtime** — los parches ya están normalizados en disco.

### Augmentation 3D (solo train)
Implementada en `LUNA16Dataset._augment_3d()` (`datasets/luna.py`), reutilizada vía herencia.

| # | Augmentation | Parámetros | Probabilidad |
|---|---|---|---|
| 1 | **Flip aleatorio** | 3 ejes independientes | P=0.5 por eje |
| 2 | **Rotación 3D** | ±15° en planos axial (1,2), coronal (0,2), sagital (0,1); `order=1`, `mode="nearest"` | Siempre (skip si \|angle\| < 0.5°) |
| 3 | **Escalado/Zoom** | Factor ∈ [0.80, 1.20]; crop/pad central si cambia tamaño | Siempre |
| 4 | **Traslación espacial** | ±4 vóxeles, zero-padding | Siempre |
| 5 | **Deformación elástica** | σ ∈ [1, 3] vóx, α ∈ [0, 5] mm; `map_coordinates` order=1 | P=0.5 |
| 6a | **Ruido gaussiano** | σ ∈ [0, 25 HU] / 1400 ≈ [0, 0.0179] normalizado | P=0.5 |
| 6b | **Brillo/contraste** | scale ∈ [0.9, 1.1], offset ∈ [-20, 20] HU / 1400 | Siempre |
| 6c | **Blur gaussiano** | σ ∈ [0.1, 0.5] vóxeles | P=0.5 |

**Clip final:** `np.clip(volume, -0.099, 0.901)` para mantener rango zero-centered.

### DataLoader
**En modo DDP (train_expert3_ddp.py):**
- Train: `DistributedSampler` (shuffle=True, drop_last=True)
- Val/Test: Sin `DistributedSampler` (todos los ranks ven todos los datos)
- `num_workers`: `max(1, os.cpu_count() // (2 * world_size))` (0 en dry-run)
- `pin_memory`: True si CUDA disponible
- `persistent_workers`: True si num_workers > 0

**En modo standalone (`build_dataloaders_expert3`):**
- `num_workers`: 4
- `pin_memory`: True
- `persistent_workers`: True

### Balanceo de clases
- **No hay oversampling ni undersampling** en el DataLoader
- El balanceo se maneja exclusivamente vía `FocalLoss(alpha=0.85)` y label smoothing
- No hay WeightedRandomSampler

---

## Sección 4: Loss Function y Métricas

### Loss Function
**Clase:** `FocalLoss` en `src/pipeline/fase2/losses.py`

```python
FocalLoss(gamma=2.0, alpha=0.85)
```

**Implementación:**
```python
bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
p_t = torch.exp(-bce)
alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
loss = alpha_t * (1 - p_t) ** gamma * bce
return loss.mean()
```

- Recibe `logits[:, 1]` (logit de clase positiva) y labels float
- Label smoothing aplicado ANTES de la loss: `labels * 0.95 + 0.025`
- En validación: label smoothing **NO** se aplica (labels crudos float)

### Métricas calculadas en validación

| Métrica | Cálculo | Detalles |
|---|---|---|
| `val_loss` | FocalLoss promedio | Sin label smoothing |
| `val_f1_macro` | `f1_score(labels, preds, average="macro", zero_division=0)` | Threshold = 0.5 sobre softmax[:,1] |
| `val_auc` | `roc_auc_score(labels, probs)` | Probs = softmax[:,1] |
| `confusion_matrix` | `confusion_matrix(labels, preds, labels=[0,1])` | [[TN, FP], [FN, TP]] |

**Protección contra NaN:** Verifica que ambas clases estén presentes antes de calcular AUC y F1. Si solo hay una clase → devuelve 0.0.

---

## Sección 5: Optimizador y Schedule

### Optimizador
```python
AdamW(lr=3e-4, weight_decay=0.03, betas=(0.9, 0.999))
```
Todos los parámetros del modelo (from-scratch, nada congelado).

### Learning Rate Schedule
```python
CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)
```
- **Primer ciclo:** 15 épocas (LR: 3e-4 → 1e-6)
- **Segundo ciclo:** 30 épocas (T_0 × T_mult)
- **Tercer ciclo:** 60 épocas
- `scheduler.step()` se llama después de cada época completa
- **No hay warm-up explícito**

### GradScaler (FP16)
```python
GradScaler(device=device.type, enabled=use_fp16)
```
Secuencia por paso de optimización: `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` → `optimizer.zero_grad()`

### Gradient Accumulation
- `EXPERT3_ACCUMULATION_STEPS = 4`
- `model.no_sync()` en pasos intermedios para evitar allreduce redundante
- Flush de gradientes residuales si el último bloque no completó accumulation

---

## Sección 6: Resultados Reales del Entrenamiento

**Fuente:** `checkpoints/expert_03_densenet3d/expert3_ddp_training_log.json`

### Resumen general
| Métrica | Valor |
|---|---|
| Épocas entrenadas | 23 |
| World size | 2 GPUs |
| Tiempo por época | ~206–216 s |
| Tiempo total estimado | ~4,850 s (~81 min) |
| Razón de parada | Early stopping activó en época 23 (mejor en época 23, patience=20 desde época 7) |

### Mejor epoch (época 23 — última registrada como `is_best`)

| Métrica | Valor |
|---|---|
| `val_loss` | **0.006632** |
| `val_f1_macro` | **0.9438** |
| `val_auc` | **0.9911** |
| Confusion Matrix | TN=1036, FP=14, FN=8, TP=97 |
| LR en ese punto | 2.505e-4 |

### Curva de entrenamiento (extracto)

| Epoch | train_loss | val_loss | val_f1_macro | val_auc | LR |
|---|---|---|---|---|---|
| 1 | 0.0315 | 0.0169 | 0.8474 | 0.9115 | 2.97e-4 |
| 3 | 0.0176 | 0.0106 | 0.8966 | 0.9712 | 2.71e-4 |
| 7 | 0.0129 | 0.0078 | 0.9063 | 0.9833 | 1.66e-4 |
| 15 | 0.0088 | 0.0083 | 0.9301 | 0.9765 | 3.00e-4 (restart) |
| 18 | 0.0110 | 0.0071 | 0.9354 | 0.9906 | 2.93e-4 |
| 23 | 0.0098 | 0.0066 | 0.9438 | 0.9911 | 2.51e-4 |

**Observación:** El LR sube a 3e-4 en época 15 (restart del cosine schedule con T_0=15). Tras el restart, el modelo encuentra un mejor mínimo (época 23).

### Resultados en Test Set
**No hay `evaluation_results.json`** en el directorio de checkpoints. El log JSON contiene solo épocas de entrenamiento/validación (23 entradas). El test set se evalúa al final del script, pero los resultados se agregan al mismo log — en este caso, **no hay entrada de evaluación test** en el log registrado (el log termina en época 23 sin entrada `"evaluation": "test"`).

---

## Sección 7: Lo que el Código HACE y NO HACE

### ✅ Lo que HACE

| Funcionalidad | Implementación |
|---|---|
| Carga datos LUNA16 | Scan de archivos `.npy` + lookup de labels en dict |
| Preprocessing offline | Clipping HU [-1000, 400], normalización [0,1], zero-centering |
| Augmentation online (7 tipos) | Flip, rotación, zoom, traslación, elastic deform, ruido/brillo/blur |
| Entrenamiento multi-GPU DDP | `DistributedDataParallel`, `DistributedSampler`, `model.no_sync()` |
| Gradient accumulation | 4 steps, batch efectivo = 32 |
| Mixed precision FP16 | `torch.amp.autocast` + `GradScaler` |
| Validación cada época | Todas las épocas |
| Early stopping | Por `val_loss`, patience=20, min_delta=0.001 |
| Early stopping broadcast | Rank 0 decide, broadcast vía `torch.distributed.broadcast` |
| Checkpoint saving | Solo best (por val_loss), incluye model + optimizer + scheduler + config |
| Evaluación en test set | Carga best checkpoint, ejecuta `validate()` sobre test_loader |
| Label smoothing | {0,1} → {0.025, 0.975} en train; no en val |
| VRAM logging | `torch.cuda.memory_allocated/reserved` por época |
| Gradient checkpointing | Opcional (`--gradient-checkpointing`), aplica `torch.utils.checkpoint` a dense blocks |
| Seed reproducibilidad | Seed=42 + rank offset; `cudnn.deterministic=True`, `benchmark=False` |
| Logging JSON | Training log con métricas por época |
| Dry-run | `--dry-run`: 2 batches train, 1 batch val, max_samples=64, max_epochs=2 |

### ❌ Lo que NO HACE

| Funcionalidad | Estado |
|---|---|
| Test Time Augmentation (TTA) | No implementado |
| Balanceo de clases por epoch (oversampling/undersampling) | No implementado — solo FocalLoss |
| Post-processing de predicciones | No implementado |
| Ensemble | No implementado |
| Warm-up de learning rate | No implementado (CosineAnnealingWarmRestarts arranca directo) |
| Guardado de checkpoint "latest" | No implementado — solo guarda best |
| FROC/CPM (métrica oficial LUNA16) | Clase `LUNA16FROCEvaluator` existe en `luna.py` pero NO se usa en el training script |
| Weighted sampling | No implementado |
| Transfer learning / LP-FT | No aplicable — entrenamiento from scratch |
| Logging a TensorBoard/W&B | No implementado — solo JSON + stdout |
| Learning rate warm-up | No implementado |
| Mixup / CutMix | No implementado |

---

## Sección 8: Reproducibilidad

### Seeds
```python
_SEED = 42
effective_seed = 42 + rank  # rank 0 → 42, rank 1 → 43
np.random.seed(effective_seed)
torch.manual_seed(effective_seed)
torch.cuda.manual_seed(effective_seed)
torch.cuda.manual_seed_all(effective_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Hardware (inferido del log)
- **GPUs:** 2 (world_size=2 en cada entrada del log)
- **Tipo GPU:** Titan Xp 12 GB (documentado en config/docstrings)
- **Batch per GPU:** `EXPERT3_BATCH_SIZE // world_size = 8 // 2 = 4`
- **Batch efectivo:** `4 × 2 × 4 = 32`

### Dataset
- **Ruta:** `datasets/luna_lung_cancer/patches/{train,val,test}/`
- **CSV:** `datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv` (fallback: `candidates.csv`)
- **Splits:** Pre-separados en disco (train/val/test)

### Dependencias principales (del código)
- PyTorch (torch, torch.nn, torch.cuda.amp, torch.distributed)
- NumPy
- pandas
- scikit-learn (f1_score, roc_auc_score, confusion_matrix)
- scipy (scipy.ndimage: rotate, zoom, gaussian_filter, map_coordinates)
- SimpleITK (en luna.py para extracción de parches)

### Cómo ejecutar

```bash
# Multi-GPU (2× Titan Xp):
torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py

# Single-GPU:
torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert3_ddp.py

# Dry-run (verificar pipeline):
torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py --dry-run

# Con gradient checkpointing (si OOM):
torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py --gradient-checkpointing

# Override batch per GPU:
torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py --batch-per-gpu 4

# Con script wrapper:
bash run_expert.sh 3
```

### Checkpoint guardado
- **Ruta:** `checkpoints/expert_03_densenet3d/best.pt`
- **Contenido:** `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `val_loss`, `val_f1_macro`, `val_auc`, `config` dict, `epoch`
- **Training log:** `checkpoints/expert_03_densenet3d/expert3_ddp_training_log.json`
