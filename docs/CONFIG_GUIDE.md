# Guia de Configuracion — expert1_config.py

Todos los hiperparametros del Expert 1 estan centralizados en `src/pipeline/fase2/expert1_config.py`.

## Parametros

### Entrenamiento

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `EXPERT1_EPOCHS` | 100 | Epocas maximas. Early stopping corta antes si no hay mejora. |
| `EXPERT1_LR` | 3e-4 | Learning rate para AdamW. Conservador para training desde cero con pos_weight alto. |
| `EXPERT1_WEIGHT_DECAY` | 1e-4 | L2 regularization. Menor que el 0.05 tipico de transfer learning. |
| `EXPERT1_DROPOUT_FC` | 0.4 | Dropout en el clasificador. Moderado-alto para compensar desbalance de clases. |

### Batch y Acumulacion

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `EXPERT1_BATCH_SIZE` | 48 | Batch total dividido entre GPUs. Con 2 GPUs = 24/GPU. |
| `EXPERT1_ACCUMULATION_STEPS` | 2 | Gradient accumulation. Batch efectivo = 48 x 2 = 96. |
| `EXPERT1_NUM_WORKERS` | 8 | Workers del DataLoader. Saturan la GPU en carga de datos. |
| `EXPERT1_FP16` | True | Mixed precision. Se desactiva automaticamente en CPU. |

### Modelo

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `EXPERT1_IMG_SIZE` | 256 | Resolucion de entrada (256x256 grayscale). |
| `EXPERT1_NUM_CLASSES` | 14 | Patologias del ChestXray14. |

### Early Stopping

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `EXPERT1_EARLY_STOPPING_PATIENCE` | 20 | Epocas sin mejora en val_macro_auc antes de detener. |
| `EXPERT1_EARLY_STOPPING_MONITOR` | "val_macro_auc" | Metrica monitoreada. |

## Ajuste por Hardware

### 2x Titan Xp (12 GB VRAM cada una) — Config actual

```python
EXPERT1_BATCH_SIZE = 48          # 24 por GPU
EXPERT1_ACCUMULATION_STEPS = 2   # Efectivo: 96
EXPERT1_NUM_WORKERS = 8
EXPERT1_FP16 = True
```

VRAM estimada: ~8 GB/GPU. Quedan ~4 GB libres.

### 1x RTX 4090 (24 GB VRAM)

```python
EXPERT1_BATCH_SIZE = 96          # Todo en 1 GPU
EXPERT1_ACCUMULATION_STEPS = 1   # Sin acumulacion
EXPERT1_NUM_WORKERS = 12         # Mas cores disponibles
EXPERT1_FP16 = True
```

Con 24 GB de VRAM se puede usar batch grande sin acumulacion. Considerar subir a batch=128 si hay margen.

### CPU only

```python
EXPERT1_BATCH_SIZE = 16          # Minimizar uso de RAM
EXPERT1_ACCUMULATION_STEPS = 4   # Efectivo: 64
EXPERT1_NUM_WORKERS = 4          # Menos overhead de fork
EXPERT1_FP16 = False             # CPU no soporta FP16 eficientemente
```

Lanzar con `torchrun --nproc_per_node=1` o directamente con `python`.

### 8x A100 (80 GB VRAM cada una)

```python
EXPERT1_BATCH_SIZE = 256         # 32 por GPU
EXPERT1_ACCUMULATION_STEPS = 1
EXPERT1_NUM_WORKERS = 8          # Por proceso
EXPERT1_FP16 = True
EXPERT1_LR = 5e-4               # Subir LR con batch grande (linear scaling)
```

## Trade-offs

### batch_size vs accum_steps
- **Batch grande** (batch_size alto, accum bajo): Menos pasos de optimizador por epoca, training mas rapido. Requiere mas VRAM.
- **Batch pequeno + accum alto**: Mismo batch efectivo con menos VRAM, pero mas pasadas forward/backward y mas lento.
- **Regla:** Maximizar batch_size dentro del VRAM disponible, luego ajustar accum para alcanzar el batch efectivo deseado (~96-128).

### num_workers
- **0 workers**: DataLoader carga datos en el proceso principal. Cuello de botella severo (~45 min/epoca).
- **4-8 workers**: Carga en paralelo. Saturacion de GPU. Optimo para 2 GPUs con imágenes .npy.
- **>8 workers**: Rendimiento marginal. Mayor uso de RAM por los procesos fork.
- **Regla:** `num_workers = min(cpu_count // world_size, 8)`.

### learning_rate
- **1e-3**: Agresivo para training desde cero. Puede divergir con pos_weight alto.
- **3e-4**: Conservador. Convergencia estable. Config actual.
- **1e-4**: Muy conservador. Convergencia lenta pero segura.
- **Linear scaling rule**: Si se duplica el batch efectivo, duplicar LR. Aplicar warmup si LR > 5e-4.

### FP16 vs FP32
- FP16 reduce VRAM ~40% y acelera compute en GPUs con Tensor Cores (RTX, A100).
- Titan Xp no tiene Tensor Cores pero FP16 aun reduce VRAM significativamente.
- FocalLoss con alpha clamped y gradient clipping hacen FP16 seguro en este pipeline.
