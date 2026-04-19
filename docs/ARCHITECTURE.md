# Arquitectura — Expert 1: Hybrid-Deep-Vision

## Diagrama General

```
Input: [B, 1, 256, 256] — Grayscale chest X-ray, float32

┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: Dense-Inception Backbone (5 bloques)                   │
│                                                                 │
│  Bloque 1: [B,1,256,256]   → [B,64,128,128]                   │
│  Bloque 2: [B,64,128,128]  → [B,128,64,64]                    │
│  Bloque 3: [B,128,64,64]   → [B,256,32,32]                    │
│  Bloque 4: [B,256,32,32]   → [B,512,16,16]                    │
│  Bloque 5: [B,512,16,16]   → [B,1024,8,8]                     │
│                                                                 │
│  Cada bloque:                                                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌───────────┐                  │
│  │Conv1×1│  │Conv3×3│  │Conv5×5│  │MaxPool→1×1│                  │
│  └──┬───┘  └──┬───┘  └──┬───┘  └────┬──────┘                  │
│     └─────────┴────────┴───────────┘                            │
│                    │ concat                                      │
│              ┌─────┴─────┐                                      │
│              │ Bottleneck │ Conv 1×1 (BN+ReLU)                  │
│              └─────┬─────┘                                      │
│              ┌─────┴─────┐                                      │
│              │ MaxPool 2×2│ stride=2 (downsampling)             │
│              └───────────┘                                      │
├─────────────────────────────────────────────────────────────────┤
│  FASE 2: Transition Bottleneck                                  │
│                                                                 │
│  Conv1×1: 1024 → 512 (BN+ReLU)                                │
│  Conv1×1:  512 → 256 (BN+ReLU)                                │
│  Conv1×1:  256 → 128 (BN+ReLU)                                │
│                                                                 │
│  Salida: [B, 128, 8, 8]                                       │
├─────────────────────────────────────────────────────────────────┤
│  FASE 3: ResNet Blocks (×3)                                     │
│                                                                 │
│  Cada bloque: x → Conv3×3 → BN → ReLU → Conv3×3 → BN → +x → ReLU │
│  Entrada/salida: [B, 128, 8, 8] (identidad preservada)        │
├─────────────────────────────────────────────────────────────────┤
│  FASE 4: Classification Head                                    │
│                                                                 │
│  GAP [B,128] ─┐                                                │
│                ├─ concat → [B, 256]                             │
│  GMP [B,128] ─┘                                                │
│                                                                 │
│  Linear(256, 128) → ReLU → Dropout(0.4) → Linear(128, 14)     │
│                                                                 │
│  Salida: [B, 14] — logits crudos (sin sigmoid)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Componentes Clave

### Inception Blocks (Fase 1)
Cada bloque tiene 4 ramas paralelas con distintos campos receptivos:
- **Branch 1:** Conv 1×1 — captura features puntuales
- **Branch 2:** Conv 3×3 — features locales
- **Branch 3:** Conv 5×5 — features de contexto medio
- **Branch 4:** MaxPool 3×3 → Conv 1×1 — features de contraste local

Las ramas se concatenan y pasan por un bottleneck Conv 1×1 que reduce canales, seguido de MaxPool 2×2 para downsampling espacial. Todas las convolutions usan BatchNorm + ReLU y `bias=False` (el bias del BN lo hace redundante).

### Transition Bottleneck (Fase 2)
Tres Conv 1×1 consecutivas (1024→512→256→128) que comprimen la representación de canales sin alterar la resolución espacial. Actúan como cuello de botella entre el backbone de alta capacidad y los bloques residuales.

### ResNet Blocks (Fase 3)
Tres bloques residuales básicos que refinan las features sin cambiar dimensiones. La conexión residual permite que el gradiente fluya directamente durante el backprop.

### Classification Head (Fase 4)
Combina Global Average Pooling (GAP) y Global Max Pooling (GMP) por concatenación. GAP captura la activación promedio (presencia difusa de patología), mientras que GMP captura la activación máxima (foco más prominente). La combinación da un vector de 256 dimensiones que pasa por un MLP de dos capas.

### Weight Initialization
- **Conv2d:** Kaiming Normal (fan_out, ReLU) — estándar para redes entrenadas desde cero con ReLU.
- **BatchNorm2d:** weight=1, bias=0.
- **Linear:** Normal(0, 0.01) con bias=0.

## Loss Function: FocalLoss

### Motivación
El dataset ChestXray14 tiene desbalance severo entre clases:
- Infiltration: 17.7% prevalencia
- Hernia: 0.2% prevalencia (~538× más negativos que positivos)
- ~53% son "No Finding" (vector todo-ceros)

BCE estándar con pos_weight sufrió de gradientes explosivos con la Hernia (pos_weight ~538). FocalLoss reduce el peso de ejemplos "fáciles" (bien clasificados), permitiendo que el modelo se concentre en los difíciles.

### Formulación

```
FL(pt) = -alpha_t * (1 - pt)^gamma * BCE(logit, target)

donde:
  pt = sigmoid(logit)  si target=1
  pt = 1 - sigmoid(logit)  si target=0

  alpha_t = alpha[class]  si target=1 (pos_weight)
  alpha_t = 1.0           si target=0

  gamma = 2.0 (standard)
```

### pos_weight Strategy
- Calculado como `n_neg / n_pos` por clase desde el training set.
- **Clamped a max=50** para evitar overflow. Hernia pasa de ~538 a 50.
- Se aplica solo a positivos (como `pos_weight` de `BCEWithLogitsLoss`).

## Data Pipeline

### Preprocesamiento Offline (Fase 0)
```
PNG original (1024×1024) → IMREAD_GRAYSCALE → validación (≥800px) →
resize INTER_LINEAR (256×256) → CLAHE(clip=2.0, tile=8×8) →
float32 / 255.0 → .npy
```

Se computan mean y std del training set y se guardan en `stats.json`.

### Augmentation Online (Fase 2, solo train)
8 augmentaciones estocásticas aplicadas con Albumentations:

| # | Transform | Params | p |
|---|-----------|--------|---|
| 9 | HorizontalFlip | — | 0.5 |
| 10 | Affine | translate=±6%, scale=0.85-1.10 | 0.5 |
| 11 | Rotate | ±10° | 0.5 |
| 12 | ElasticTransform | alpha=30, sigma=5 | 0.1 |
| 13 | RandomBrightnessContrast | ±15% | 0.5 |
| 14 | GaussianBlur | kernel=3-5 | 0.1 |
| 15 | GaussNoise | std=0.009-0.022 | 0.1 |
| 16 | CoarseDropout | 1-3 holes, 8-24px | 0.15 |

Seguidas de `A.Normalize(mean, std)` + `ToTensorV2()`.

### Normalización
Val/Test solo aplican `A.Normalize(mean, std) + ToTensorV2()` — sin augmentación estocástica.

## DDP (Distributed Data Parallel)

### Funcionamiento
- `torchrun` configura variables de entorno (RANK, WORLD_SIZE, LOCAL_RANK).
- `setup_ddp()` inicializa el process group con auto-detección de backend (NCCL/Gloo).
- El training DataLoader usa `DistributedSampler` para particionar datos entre GPUs.
- Val/Test DataLoaders **no** usan sampler distribuido (cada rank evalúa el dataset completo).

### Gradient Accumulation con DDP
- En pasos intermedios de accumulation, se usa `model.no_sync()` para evitar allreduce.
- Solo en el último paso del bloque de accumulation se sincronizan gradientes.
- Esto reduce la comunicación inter-GPU de `accum_steps` a 1 por bloque.

### NCCL Config
| Variable | Valor | Razón |
|----------|-------|-------|
| `NCCL_TIMEOUT` | 1800000 (30min) | Previene timeout en épocas largas |
| `NCCL_IB_DISABLE` | 1 | Sin InfiniBand en el servidor |
| `NCCL_P2P_DISABLE` | 0/1 | Depende del lanzamiento (shell vs directo) |
| `TORCH_NCCL_BLOCKING_WAIT` | 1 | Timeout bloqueante para debugging |
