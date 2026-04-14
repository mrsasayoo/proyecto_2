# Experto 1 — ConvNeXt-Small — Model Card

> **Nota sobre el directorio:** La carpeta se llama `expert_01_efficientnet_b3/` por compatibilidad con otros modulos del pipeline MoE que tienen esa ruta hardcodeada. El modelo actual es ConvNeXt-Small, no EfficientNet-B3.

## Resumen

| Campo | Valor |
|---|---|
| **Experto ID** | 1 (`EXPERT_IDS["isic"]`) |
| **Arquitectura** | ConvNeXt-Small (timm 1.0.25, pretrained ImageNet-1K) |
| **Backbone output** | 768 features (global average pool) |
| **Dataset** | ISIC 2019 Challenge |
| **Tarea** | Clasificacion multiclase — 8 clases |
| **Num. clases** | 8 |
| **Loss** | `CrossEntropyLoss` con `class_weights` + `label_smoothing=0.1` |
| **Metrica principal** | F1 Macro (objetivo > 0.72) |
| **Precision** | FP16 (mixed precision) + gradient clipping (`max_norm=1.0`) |
| **Total params** | ~50M |
| **Checkpoint** | `expert2_best.pt` (guardado por val_f1_macro maximo) |
| **Checkpoint dir** | `checkpoints/expert_01_efficientnet_b3/` |

---

## Arquitectura

### ConvNeXt-Small

ConvNeXt (Liu et al., CVPR 2022) es una CNN pura que adopta decisiones de diseno de Vision Transformers (kernel 7x7 depthwise, LayerNorm, GELU, fewer activation functions) sin usar atencion. El resultado es una red convolucional que iguala o supera a Swin Transformer en ImageNet con la misma cantidad de FLOPs.

La variante Small tiene 4 stages con bloques [3, 3, 27, 3] y dimension base 96, escalando a 192, 384, 768 en los stages posteriores. Cada bloque es: depthwise conv 7x7, LayerNorm, pointwise conv 1x1 (expansion 4x), GELU, pointwise conv 1x1 (proyeccion). Las conexiones residuales conectan entrada y salida de cada bloque.

### Head de clasificacion

```
Backbone (ConvNeXt-Small)
    |
    v
Global Average Pool → [B, 768]
    |
    v
LayerNorm(768)
    |
    v
Dropout(p=0.4)
    |
    v
Linear(768 → 256)
    |
    v
GELU()
    |
    v
Dropout(p=0.3)
    |
    v
Linear(256 → 8)
    |
    v
[B, 8] logits
```

La ultima capa lineal se inicializa con `trunc_normal(std=0.02)` y bias=0.

---

## Dataset

### ISIC 2019 Challenge

8 clases de lesiones dermatoscopicas. Split: train / val / test.

| Idx | Codigo | Nombre completo | Tipo |
|---|---|---|---|
| 0 | MEL | Melanoma | Maligno |
| 1 | NV | Nevus melanocitico | Benigno |
| 2 | BCC | Carcinoma basocelular | Maligno |
| 3 | AK | Queratosis actinica | Pre-maligno |
| 4 | BKL | Queratosis seborreica benigna | Benigno |
| 5 | DF | Dermatofibroma | Benigno |
| 6 | VASC | Lesion vascular | Benigno |
| 7 | SCC | Carcinoma escamoso | Maligno |

Clases minoritarias: AK (idx 3), DF (idx 5), VASC (idx 6), SCC (idx 7).

El desbalance se compensa con `WeightedRandomSampler` en train y class weights en la loss.

---

## Entrenamiento

3 fases, 40 epocas en total. Batch efectivo: 32 x accum_steps=3 = 96.

### Fase 1 — Head only (epocas 1-5)

Backbone congelado. Solo se entrena el head de clasificacion.

| Parametro | Valor |
|---|---|
| Optimizador | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=5, eta_min=3e-5) |

### Fase 2 — Fine-tuning diferencial (epocas 6-20)

Se descongelan los ultimos 2 stages del backbone. Learning rates diferenciados.

| Parametro | Valor |
|---|---|
| Head lr | 3e-4 |
| Backbone lr | 1e-5 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-7) |

### Fase 3 — Full fine-tuning (epocas 21-40)

Todos los parametros descongelados. Early stopping con patience=8 sobre val_f1_macro.

### Loss y regularizacion

| Parametro | Valor |
|---|---|
| Loss | CrossEntropyLoss + class weights + label_smoothing=0.1 |
| Precision | FP16 (mixed precision) |
| Gradient clipping | max_norm=1.0 |
| Sampler | WeightedRandomSampler (solo train) |

---

## Transforms

### Entrenamiento (11 pasos)

```
Resize(256)
RandomCrop(224)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.3)
RandomRotation(±30°)
ColorJitter(brightness, contrast, saturation, hue)
RandomGrayscale(p=0.05)
RandAugment(num_ops=2, magnitude=9)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
RandomErasing(p=0.2)
```

### Validacion (3 pasos)

```
Resize(224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_01_efficientnet_b3/
├── model_card.md          ← este archivo
└── expert2_best.pt        ← pesos del modelo (guardado por val_f1_macro maximo)
```

### Codigo para cargar el modelo

```python
import torch
import timm

# 1. Construir la arquitectura
model = timm.create_model("convnext_small", pretrained=False, num_classes=8)

# 2. Reemplazar el head con la arquitectura custom
import torch.nn as nn
model.head.fc = nn.Sequential(
    nn.LayerNorm(768),
    nn.Dropout(0.4),
    nn.Linear(768, 256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, 8),
)

# 3. Cargar pesos
ckpt_path = "checkpoints/expert_01_efficientnet_b3/expert2_best.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 4. Inferencia
model.eval()
with torch.no_grad():
    logits = model(image_tensor)             # [B, 8]
    probs = torch.softmax(logits, dim=1)     # [B, 8]
    predicted_class = probs.argmax(dim=1)    # [B]

# 5. Mapear a nombres de clase
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
diagnosis = CLASS_NAMES[predicted_class.item()]
```

---

## Objetivo de rendimiento

F1 Macro > 0.72 sobre el conjunto de validacion.
