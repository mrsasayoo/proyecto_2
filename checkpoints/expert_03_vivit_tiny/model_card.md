# Experto 3 — Faster R-CNN 3D (MC3-18) — Model Card

> **Nota:** El directorio sigue llamándose `expert_03_vivit_tiny/` por razones históricas.
> La arquitectura original era ViViT-Tiny pero fue reemplazada por 3D Faster R-CNN con
> backbone MC3-18 durante el desarrollo. El notebook de entrenamiento
> (`luna_training_kaggle.ipynb`) es la fuente de verdad.

## Resumen

| Campo                | Valor                                                                 |
|----------------------|-----------------------------------------------------------------------|
| **Experto ID**       | 3                                                                     |
| **Arquitectura**     | 3D Faster R-CNN (MC3-18 backbone + FPN 3D + RPN + ROI Align)        |
| **Dataset**          | LUNA16 / LIDC-IDRI (candidates_V2.csv + annotations.csv)            |
| **Modalidad**        | CT pulmonar 3D — parches centrados en candidatos a nódulo             |
| **Tarea**            | Clasificación binaria de parches (nódulo sí / nódulo no)              |
| **Num clases**       | 2                                                                     |
| **Tensor entrada**   | `[B, 1, 64, 64, 64]`                                                 |
| **Parámetros**       | ~13M entrenables (estimación del notebook)                           |
| **Loss**             | `FasterRCNN3DLoss` = L_rpn_cls + λ·L_rpn_reg + L_det_cls            |
| **Loss components**  | FocalLoss(γ=2, α=0.85) para cls, SmoothL1 para reg                  |
| **Métricas**         | AUC-ROC, F1 Macro, Accuracy, Confusion Matrix                       |
| **Preprocesado**     | On-the-fly desde .mhd: HU clip [-1000, 400] → norm [0,1] → parche 64³ (50mm físicos) |
| **Checkpoint dir**   | `checkpoints/expert_03_vivit_tiny/`                                   |
| **Pesos esperados**  | `best_luna_model.pth` (dict con `model_state_dict`, `optimizer_state_dict`, métricas) |
| **Entorno Kaggle**   | NVIDIA Tesla T4, PyTorch ≥2.8, Python 3.12                          |

## Arquitectura real: 3D Faster R-CNN

### Backbone: MC3-18 (Mixed Convolution 3D, 18 capas)

El backbone es `torchvision.models.video.mc3_18` sin pesos preentrenados (`weights=None`).
La primera convolución se reemplaza para aceptar 1 canal de entrada (CT grayscale) en lugar
de 3 canales RGB. Se aplica inicialización Kaiming.

Se añade `SpatialDropout3d(p=0.15)` después del stem para regularización espacial.

### Feature Maps del backbone

```
Input: [B, 1, 64, 64, 64]
  → stem:   [B,  64, 32, 32, 32]
  → layer1: [B,  64, 32, 32, 32]
  → layer2: [B, 128, 16, 16, 16] = C3
  → layer3: [B, 256,  8,  8,  8] = C4
  → layer4: [B, 512,  4,  4,  4] = C5
```

### FPN 3D (Feature Pyramid Network)

Toma C3, C4, C5 y produce P3, P4, P5 con 128 canales cada uno:

```
C3=[B,128,16³] → P3=[B,128,16³]
C4=[B,256, 8³] → P4=[B,128, 8³]
C5=[B,512, 4³] → P5=[B,128, 4³]
```

Laterales (Conv3d 1×1) + upsampling nearest neighbor + suavizado (Conv3d 3×3).

### RPN 3D (Region Proposal Network)

Un RPN Head compartido aplicado en P3, P4, P5. Genera por posición:
- `cls_logits`: score de objetividad (objectness)
- `bbox_deltas`: refinamiento de ancla en 6D (dz, dy, dx, dd, dh, dw)

Anclas por nivel (en vóxeles de input 64³):

| Nivel | Tamaño feature map | Tamaños de ancla | k (anclas/pos) |
|-------|-------------------|-------------------|----------------|
| P3    | 16³               | [8, 16, 24]       | 3              |
| P4    | 8³                | [16, 32, 48]      | 3              |
| P5    | 4³                | [24, 40, 56]      | 3              |

### ROI Align 3D

ROI Align 3D simplificado usando `grid_sample` trilineal en P4.
ROI centrado en el parche (coordenadas normalizadas [-0.5, 0.5]).
Tamaño de salida: 2×2×2 (`ROI_POOL_SIZE = 2`).

### Detection Head

```
ROI feat: [B, 128, 2, 2, 2]
  → Flatten: [B, 1024]
  → Linear(1024, 256) → ReLU → Dropout(0.4)
  → Linear(256, 2)
  → logits: [B, 2]
```

## Diagrama de arquitectura (ASCII)

```
                        ENTRADA: [B, 1, 64, 64, 64]
                        (parche CT centrado en candidato a nódulo)
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │     MC3-18 BACKBONE (1-canal, sin pretrain) │
              │  stem (Conv3d 1→64 + SpatialDropout3d 0.15) │
              │  layer1: [B, 64, 32³]                       │
              │  layer2: [B,128, 16³] = C3                  │
              │  layer3: [B,256,  8³] = C4                  │
              │  layer4: [B,512,  4³] = C5                  │
              │  (gradient checkpointing en layer2-4)       │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │              FPN 3D (128 ch)                │
              │  C3 → lat3+up(P4) → P3 [B,128,16³]         │
              │  C4 → lat4+up(P5) → P4 [B,128, 8³]         │
              │  C5 → lat5        → P5 [B,128, 4³]         │
              └─────────────────────────────────────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
              ┌─────────────────────────────────────────────┐
              │           RPN HEAD (compartido)             │
              │  Conv3d 3×3 → ReLU                         │
              │  cls: Conv3d → [B, k, D', H', W']          │
              │  reg: Conv3d → [B, k*6, D', H', W']        │
              │  k=3 anclas por posición                    │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │      ROI ALIGN 3D (en P4, trilineal)       │
              │  ROI centrado: [-0.5, 0.5]³                │
              │  output_size = 2³                           │
              │  → [B, 128, 2, 2, 2]                       │
              └─────────────────────────────────────────────┘
              ┌─────────────────────────────────────────────┐
              │           DETECTION HEAD                    │
              │  Flatten → [B, 1024]                       │
              │  Linear(1024, 256) → ReLU                  │
              │  Dropout(0.4)                              │
              │  Linear(256, 2) → [B, 2] logits            │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
                        SALIDA: [B, 2] logits
                        + rpn_outs (cls, reg) × 3 niveles
```

## Diagrama de integración MoE (ASCII)

```
    ┌──────────────────────────────────────────────────────────┐
    │                    PIPELINE MoE                          │
    │                                                          │
    │  Volumen CT completo                                     │
    │       │                                                  │
    │       ▼                                                  │
    │  Extractor de parches (on-the-fly desde .mhd)            │
    │  ├─ Lee candidates_V2.csv (⚠ NUNCA V1)                  │
    │  ├─ world_to_voxel(x, y, z) → (iz, iy, ix)             │
    │  ├─ HU clip [-1000, 400]                                 │
    │  ├─ Normalización [0, 1]                                 │
    │  └─ Parche 50mm centrado → zoom a 64³                   │
    │       │                                                  │
    │       ▼                                                  │
    │  Backbone compartido                                     │
    │  Parche → features multiescala                           │
    │       │                                                  │
    │       ▼                                                  │
    │  Router: g = softmax(W·z + b) ∈ ℝ^6                     │
    │       │                                                  │
    │       ├─── H(g) BAJA ──→ argmax(g) == 3 ──→ Experto 3   │
    │       │                                                  │
    │       └─── H(g) ALTA ──→ Experto 5 (CAE)                │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼ (si argmax == 3)
    ┌──────────────────────────────────────────────────────────┐
    │              EXPERTO 3: Faster R-CNN 3D                  │
    │                                                          │
    │  Recibe: parche [B, 1, 64, 64, 64]                      │
    │       │                                                  │
    │       ▼                                                  │
    │  Forward pass:                                           │
    │  MC3-18 → FPN 3D → RPN → ROI Align 3D → Det Head       │
    │       │                                                  │
    │       ▼                                                  │
    │  Logits [B, 2]                                           │
    │       │                                                  │
    │       ▼                                                  │
    │  softmax → p(nódulo), p(no-nódulo)                       │
    │  + Coordenada (x,y,z) del candidato original             │
    │  + Confianza del modelo                                  │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

## Función de pérdida: FasterRCNN3DLoss

La pérdida total combina tres componentes:

```
L_total = L_rpn_cls + λ_reg × L_rpn_reg + L_det_cls
```

| Componente    | Tipo                     | Descripción                                             |
|---------------|--------------------------|---------------------------------------------------------|
| `L_rpn_cls`   | FocalLoss(γ=2, α=0.85)  | Objectness en los 3 niveles FPN. Ancla central = positiva si gt=1. |
| `L_rpn_reg`   | SmoothL1 (β=0.1)        | Regresión de bbox, solo para anclas positivas.          |
| `L_det_cls`   | FocalLoss(γ=2, α=0.85)  | Clasificación final binaria. Con label smoothing(0.05). |

### Label Smoothing

Labels se suavizan: `{0, 1}` → `{0.025, 0.975}` (smoothing=0.05).

### Focal Loss con α=0.85

A diferencia del model card original que usaba α=0.25, el notebook usa **α=0.85** para la
clase positiva. Esto es coherente con el submuestreo de negativos a ratio ~10.7:1 en train:
con menos negativos relativos que el dataset raw (~490:1), se necesita un α más alto para
los positivos.

## Hiperparámetros de entrenamiento

| Parámetro                | Valor                                           |
|--------------------------|-------------------------------------------------|
| **Batch size**           | 4                                               |
| **Gradient accumulation**| 8 pasos → batch efectivo = 32                   |
| **Learning rate**        | 3e-4                                            |
| **Optimizer**            | AdamW (weight_decay=0.03, betas=(0.9, 0.999))  |
| **Scheduler**            | CosineAnnealingWarmRestarts (T_0=15, T_mult=2, eta_min=1e-6) |
| **Max epochs**           | 100                                             |
| **Early stopping**       | patience=20, min_delta=0.001                    |
| **Precisión**            | FP16 (torch.amp.GradScaler)                    |
| **Gradient checkpointing** | Sí — en layer2, layer3, layer4 del backbone  |
| **Gradient clipping**    | max_norm=1.0                                    |
| **Loss**                 | FasterRCNN3DLoss (ver tabla arriba)             |
| **Focal Loss γ**         | 2.0                                             |
| **Focal Loss α**         | 0.85                                            |
| **Label smoothing**      | 0.05                                            |
| **SpatialDropout3d**     | 0.15 (en stem)                                  |
| **Dropout FC**           | 0.4 (antes de FC final)                         |
| **FPN channels**         | 128                                             |
| **ROI pool size**        | 2 (2×2×2)                                       |
| **Num workers**          | 2 (Kaggle)                                      |
| **Seed**                 | 42                                              |
| **GPU**                  | NVIDIA Tesla T4 (Kaggle)                        |

## Datos y splits

### Fuentes

- **candidates_V2.csv**: Coordenadas de ~551K candidatos a nódulo con labels binarios (⚠ NUNCA usar V1)
- **annotations.csv**: ~1,186 nódulos reales con diámetros (para bounding boxes GT del RPN)
- **seg-lungs-LUNA16/**: Volúmenes CT en formato .mhd/.raw

### Splits

Split por `seriesuid` (por paciente, sin leakage). Si existe `luna_splits.json` se usa directamente;
si no, se genera un split 80/10/10 automáticamente.

### Submuestreo de negativos

| Split   | Submuestreo             | Ratio neg:pos | Notas                          |
|---------|-------------------------|---------------|--------------------------------|
| train   | `subsample_neg=True, ratio=10.7` | ~10.7:1 | WeightedRandomSampler adicional |
| val     | `subsample_neg=True, ratio=100`  | Variable | Ratio amplio para validación   |
| test    | `subsample_neg=False`            | Sin subsampleo | Distribución real              |

### WeightedRandomSampler

Para el train loader se usa `WeightedRandomSampler` con pesos inversamente proporcionales
a la frecuencia de clase. Esto asegura que cada batch contenga representación balanceada
de positivos y negativos, complementando el submuestreo de candidatos.

### Bounding boxes GT

Para candidatos positivos, se busca la anotación más cercana en `annotations.csv` (dentro de
30mm) y se convierte el diámetro a coordenadas de vóxel del parche. Para positivos sin
diámetro, se usa r=8 vox (~10mm). Para negativos: bbox = [-1, -1, -1, -1, -1, -1].

## Preprocesado on-the-fly

A diferencia del pipeline offline de 7 pasos documentado en `LUNA16_DATA_PIPELINE.md`
(que genera parches `.npy` en disco), **este notebook extrae parches on-the-fly desde los
volúmenes .mhd** sin caché en disco. El flujo es:

| Paso | Operación                                                    |
|------|--------------------------------------------------------------|
| 1    | Cargar .mhd con SimpleITK (cache LRU de 20 volúmenes)       |
| 2    | Conversión world→voxel con origin + spacing + direction       |
| 3    | Calcular tamaño de parche en mm: `PATCH_MM = 50mm`           |
| 4    | HU clip [-1000, 400] → normalización [0, 1]                  |
| 5    | Extraer parche centrado en candidato (tamaño variable por spacing) |
| 6    | Zero-padding si el candidato está cerca del borde del volumen |
| 7    | Zoom bilineal (`scipy.ndimage.zoom`) → 64×64×64              |

**Diferencias clave vs pipeline offline:**
- Sin resampling isotrópico previo (el zoom se hace por parche)
- Sin lung masking explícita (se confía en que los candidatos están dentro del pulmón)
- Sin zero-centering global (el parche queda en rango [0, 1])
- Cache LRU de 20 volúmenes (evita releer .mhd repetidamente)

## Augmentaciones online (solo train)

8 transformaciones aplicadas secuencialmente en `LUNAPatchDataset._augment_3d()`:

| #  | Augmentación                | Parámetros                        | Probabilidad |
|----|-----------------------------|-----------------------------------|--------------|
| 1  | Flip 3D                     | Independiente por eje             | P=0.5/eje    |
| 2  | Rotación 3D completa        | ±15° en 3 planos (axial, coronal, sagital) | Siempre (>0.5°) |
| 3  | Zoom 3D isótropo            | [0.85, 1.15], crop/pad central    | Siempre      |
| 4  | Deformación elástica 3D     | σ=2.0, α=8.0                     | P=0.4        |
| 5  | RandCoarseDropout 3D (CutOut)| 4 cubos de 8³                    | P=0.5        |
| 6  | Variación HU                | offset ±20 HU, escala ±5%        | Siempre      |
| 7  | Ruido gaussiano             | σ ~ U(0, 0.02)                   | P=0.3        |
| 8  | Traslación espacial         | ±4 vox, zero-pad (sin wrap)      | Siempre      |

**Descartadas (documentado en el notebook):**
- Transformación afín 3D: redundante con rotación + zoom
- Random crop 3D + resize: peligrosa para nódulos pequeños (<6mm)

El clip final `np.clip(0, 1)` se aplica después de todas las augmentaciones para garantizar
que los valores estén en rango.

## Test-Time Augmentation (TTA)

El notebook implementa TTA con las 8 combinaciones posibles de flip en los 3 ejes
espaciales (D, H, W). La probabilidad final es el promedio de las 8 predicciones.

## Métricas

### Métricas de validación durante entrenamiento

Se registran por época: `val_loss`, `val_f1_macro`, `val_auc`, `val_acc`, confusion matrix.

### Métricas en test

Pendiente de evaluación completa. El notebook define evaluación en test con:
- Classification report (precision, recall, F1 por clase)
- Confusion matrix (TN, FP, FN, TP)
- Sensitividad, Especificidad, PPV, NPV
- Curva ROC + AUC
- Curva Precision-Recall + Average Precision
- Análisis de threshold óptimo (max F1 Macro)
- Comparación sin TTA vs con TTA

**Nota:** El notebook fue ejecutado en Kaggle (timestamps muestran ejecución del training loop
de ~8 horas: 05:45 → 13:43 UTC del 2026-04-09). Las celdas de evaluación en test no
muestran outputs guardados en el notebook, por lo que las métricas finales exactas no están
disponibles en este archivo. Si se generaron, estarán en `/kaggle/working/training_log.json`.

### Techo de sensibilidad: 94.4%

No todos los nódulos del LIDC-IDRI tienen acuerdo unánime entre radiólogos. El 5.6%
restante corresponde a nódulos con anotación ambigua (acuerdo ≤2/4 radiólogos). Un modelo
perfecto en LUNA16 alcanza como máximo 94.4% de sensibilidad.

## Consideraciones especiales

### Gradient checkpointing

Habilitado en layer2, layer3, layer4 del backbone MC3-18. Reduce VRAM recalculando
activaciones intermedias durante el backward pass, a costa de ~30% más tiempo por época.

### FP16 (Mixed Precision Training)

Obligatorio para caber en la VRAM de Tesla T4 (16 GB). Se usa `torch.amp.GradScaler`.

### Cache LRU de volúmenes

El dataset mantiene un cache LRU (`OrderedDict`) de hasta 20 volúmenes .mhd en memoria.
Esto evita releer el mismo CT cuando múltiples candidatos pertenecen al mismo scan.

### candidates_V2.csv — SIEMPRE V2

`candidates_V2.csv` corrige 24 nódulos que estaban mal etiquetados o ausentes en V1.
Usar V1 produce una caída silenciosa en sensibilidad que no se detecta hasta evaluar
con FROC. **No hay ningún escenario donde V1 sea preferible.**

### Focal Loss α=0.85 (no 0.25)

El notebook usa α=0.85 para la clase positiva (nódulos). Esto difiere del valor clásico
α=0.25 de Lin et al. (2017), que fue calibrado para COCO (~1:80). Con el submuestreo
a ~10.7:1 y el `WeightedRandomSampler`, se necesita un α más alto para compensar el
desbalance residual y el label smoothing.

## Dependencias

| Librería          | Uso                                                  |
|-------------------|------------------------------------------------------|
| PyTorch ≥2.8      | Modelo, training loop, autocast, GradScaler          |
| torchvision       | `mc3_18` backbone preentrenado                       |
| SimpleITK         | Lectura de volúmenes .mhd/.raw                      |
| scipy             | `ndimage.zoom`, `ndimage.rotate`, `gaussian_filter`, `map_coordinates` |
| numpy             | Operaciones de array, augmentaciones                 |
| pandas            | Lectura de CSVs, manipulación de DataFrames          |
| scikit-learn      | `train_test_split`, `f1_score`, `roc_auc_score`, `confusion_matrix`, curvas ROC/PR |
| matplotlib        | Visualizaciones y curvas                             |

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_03_vivit_tiny/
├── model_card.md               ← este archivo
├── luna_training_kaggle.ipynb   ← notebook de entrenamiento (fuente de verdad)
├── LUNA16_DATA_PIPELINE.md      ← documentación del pipeline offline de 7 pasos
├── fase0_execution_plan.md      ← plan de ejecución de Fase 0
├── augmentations/               ← ejemplos visuales de augmentaciones
├── transforms/                  ← ejemplos visuales del pipeline de preprocesado
├── visualize_pipeline.py        ← script de visualización
└── best_luna_model.pth          ← pesos del modelo (generado por el notebook)
```

### Contenido del checkpoint (best_luna_model.pth)

```python
{
    'epoch': int,                    # Época del mejor modelo
    'model_state_dict': OrderedDict, # Pesos de Expert3FasterRCNN3D
    'optimizer_state_dict': dict,    # Estado del optimizer AdamW
    'val_loss': float,               # Val loss del mejor modelo
    'val_f1': float,                 # Val F1 Macro del mejor modelo
    'val_auc': float,                # Val AUC-ROC del mejor modelo
    'config': {
        'arch': 'FasterRCNN3D',
        'backbone': 'mc3_18',
        'fpn_ch': 128,
        'n_params': int,
        'seed': 42,
        'focal_gamma': 2.0,
        'focal_alpha': 0.85,
    },
}
```

### Cargar el modelo

```python
import torch
import torch.nn as nn
from torchvision.models.video import mc3_18

# Definir Expert3FasterRCNN3D (ver notebook completo para definición)
# ... (incluye FPN3D, RPN3DHead, roi_align_3d, Expert3FasterRCNN3D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Expert3FasterRCNN3D(
    spatial_dropout_p=0.15,
    fc_dropout_p=0.4,
    num_classes=2,
    lambda_reg=1.0,
    fpn_ch=128,
    roi_sz=2,
).to(device)

checkpoint_path = "checkpoints/expert_03_vivit_tiny/best_luna_model.pth"
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"Modelo cargado de época {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")

# Inferencia de un parche 3D
dummy_input = torch.randn(1, 1, 64, 64, 64, device=device)
with torch.no_grad():
    logits, rpn_outs = model(dummy_input)  # logits: [1, 2], rpn_outs: 3 niveles
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)  # 0 = no nódulo, 1 = nódulo
```
