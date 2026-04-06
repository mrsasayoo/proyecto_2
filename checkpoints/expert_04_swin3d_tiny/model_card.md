# Experto 4 — Swin3D-Tiny — Model Card

## Resumen

| Campo                | Valor                                                                       |
|----------------------|-----------------------------------------------------------------------------|
| **Experto ID**       | 4                                                                           |
| **Arquitectura**     | Swin3D-Tiny (Swin Transformer adaptado a 3D)                               |
| **Dataset**          | PANORAMA / Zenodo CT abdominal                                              |
| **Modalidad**        | CT abdominal 3D — volumen completo resized                                  |
| **Tarea**            | Clasificación binaria de volumen (PDAC+ / PDAC−)                            |
| **Num clases**       | 2                                                                           |
| **Tensor entrada**   | `[B, 1, 64, 64, 64]`                                                       |
| **Loss**             | `FocalLoss(alpha=0.75, gamma=2)` — alpha elevado por distribución de clases |
| **Métricas**         | AUC-ROC > 0.85 (bueno). Baseline nnU-Net ≈ 0.88                            |
| **Volumen datos**    | ~557 casos con CT, ~281 volúmenes × k-fold CV (k=5 obligatorio)            |
| **Preprocesado**     | HU clip [-100, 400] → z-score por volumen → resize 64³                     |
| **Checkpoint dir**   | `checkpoints/expert_04_swin3d_tiny/`                                        |
| **Pesos esperados**  | `weights_exp4.pt`                                                           |

## Teoría del modelo

### Swin Transformer: atención local con ventanas desplazadas

Swin Transformer (Liu et al., 2021) resuelve el problema fundamental del ViT estándar:
la complejidad cuadrática de la self-attention respecto al número de tokens. Mientras ViT
computa atención entre todos los pares de tokens (O(n²) con n=512 tokens para un volumen
64³/8³), Swin restringe la atención a ventanas locales y usa un mecanismo de desplazamiento
(shifting) para permitir comunicación entre ventanas.

**Adaptación a 3D para CT abdominal:**

En Swin3D, las ventanas son cubos de tamaño fijo (4×4×4 en este modelo). La atención se
computa solo dentro de cada ventana, reduciendo la complejidad de O(n²) a O(n·w³) donde
w=4. En capas alternas, las ventanas se desplazan en (2,2,2) para que tokens que estaban
en el borde de una ventana ahora compartan ventana con sus vecinos del otro lado, permitiendo
flujo de información global a lo largo del modelo.

### ¿Por qué Swin3D para PDAC pancreático?

El adenocarcinoma ductal pancreático (PDAC) es uno de los cánceres más letales, con
supervivencia a 5 años <10%. La detección temprana en CT es difícil porque:

1. **El páncreas es diminuto:** ocupa ~1% del volumen abdominal total. Un resize naïve
   de 512×512×Z a 64×64×64 diluye la señal pancreática en un mar de tejido irrelevante.

2. **El PDAC es isodenso:** muchos tumores pancreáticos tienen la misma densidad HU que
   el tejido pancreático normal, haciéndolos invisibles en una sola rebanada axial.

3. **Contexto multi-escala:** el diagnóstico requiere ver tanto la textura local del tumor
   como su relación con estructuras circundantes (conducto pancreático dilatado, atrofia
   parenquimatosa distal, invasión vascular).

Swin3D aborda estos desafíos con su arquitectura jerárquica de 4 stages: los primeros
stages capturan texturas locales a alta resolución (16×16×16), y los últimos stages, tras
patch merging, capturan relaciones anatómicas globales a baja resolución (2×2×2) con
dimensionalidad alta (384 canales).

### Focal Loss con alpha=0.75

El desbalance en PANORAMA es menos extremo que en LUNA16 (~490:1), pero sigue siendo
significativo. El alpha=0.75 (vs. 0.25 en LUNA16) refleja que la clase positiva (PDAC+)
necesita más peso relativo en este dataset. El gamma=2 se mantiene igual para down-ponderar
los negativos fáciles.

### z-score por volumen: compensación de bias multicéntrico

Los datos provienen de múltiples centros (Radboudumc, MSD, NIH), cada uno con diferentes
protocolos de adquisición CT (kVp, mAs, reconstrucción). Una normalización global (z-score
sobre todo el dataset) dejaría un bias sistemático por centro. La normalización por volumen
individual (media y std del volumen específico) elimina este bias, asegurando que el modelo
aprenda patrones de textura y morfología, no artefactos de protocolo.

## Diagrama de arquitectura (ASCII)

```
                        ENTRADA: [B, 1, 64, 64, 64]
                        (volumen CT abdominal completo, resized)
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               PATCH EMBEDDING 3D (Stage 0)                  │
    │  Conv3d(1, 48, kernel=4, stride=4)                          │
    │  [B, 1, 64, 64, 64] → [B, 48, 16, 16, 16]                 │
    │  Reshape → [B, 4096, 48]                                    │
    │  LayerNorm(48)                                              │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               STAGE 1: Swin3D Blocks ×2                     │
    │  Resolución: 16×16×16, Canales: C=48                        │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Bloque A: W-MSA (Window Multi-Head Self-Attention) │    │
    │  │    Ventana: 4×4×4 (64 tokens por ventana)           │    │
    │  │    Heads: 3, d_head=16                              │    │
    │  │    Atención LOCAL dentro de cada ventana             │    │
    │  │    + MLP(48→192→48) + LayerNorm + Residual          │    │
    │  ├─────────────────────────────────────────────────────┤    │
    │  │  Bloque B: SW-MSA (Shifted Window MSA)              │    │
    │  │    Ventana desplazada en (2,2,2)                     │    │
    │  │    Permite flujo de información entre ventanas       │    │
    │  │    + MLP(48→192→48) + LayerNorm + Residual          │    │
    │  └─────────────────────────────────────────────────────┘    │
    │  Salida: [B, 4096, 48]                                      │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               PATCH MERGING 3D (↓2×)                        │
    │  Agrupa 2×2×2 = 8 tokens vecinos → concatena → Linear      │
    │  [B, 4096, 48] → [B, 512, 96]                              │
    │  Resolución: 16³ → 8³, Canales: 48 → 96                    │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               STAGE 2: Swin3D Blocks ×2                     │
    │  Resolución: 8×8×8, Canales: C=96                           │
    │  W-MSA → SW-MSA (ventana 4×4×4, heads=6, d_head=16)        │
    │  Salida: [B, 512, 96]                                       │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               PATCH MERGING 3D (↓2×)                        │
    │  [B, 512, 96] → [B, 64, 192]                               │
    │  Resolución: 8³ → 4³, Canales: 96 → 192                    │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               STAGE 3: Swin3D Blocks ×6                     │
    │  Resolución: 4×4×4, Canales: C=192                          │
    │  3× (W-MSA → SW-MSA) (heads=12, d_head=16)                 │
    │  Salida: [B, 64, 192]                                       │
    │  (stage principal — mayor profundidad)                       │
    └─────────────────────────────────────────────────────────────┘
    │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               PATCH MERGING 3D (↓2×)                        │
    │  [B, 64, 192] → [B, 8, 384]                                │
    │  Resolución: 4³ → 2³, Canales: 192 → 384                   │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               STAGE 4: Swin3D Blocks ×2                     │
    │  Resolución: 2×2×2, Canales: C=384                          │
    │  W-MSA → SW-MSA (heads=24, d_head=16)                       │
    │  Salida: [B, 8, 384]                                        │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               CLASSIFICATION HEAD                           │
    │  LayerNorm(384)                                             │
    │  Reshape → [B, 384, 2, 2, 2]                                │
    │  AdaptiveAvgPool3d(1) → [B, 384, 1, 1, 1]                  │
    │  Flatten → [B, 384]                                         │
    │  Linear(384, 2) → [B, 2]                                   │
    └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        SALIDA: [B, 2] logits
                        FocalLoss(alpha=0.75, gamma=2)
```

## Diagrama de integración MoE (ASCII)

```
    ┌──────────────────────────────────────────────────────────┐
    │                    PIPELINE MoE                          │
    │                                                          │
    │  Volumen CT abdominal (DICOM / NIfTI)                    │
    │       │                                                  │
    │       ▼                                                  │
    │  Preprocesado Páncreas                                   │
    │  ├─ HU clip [-100, 400] (⚠ NO [-1000, 400])             │
    │  ├─ z-score POR VOLUMEN (no global)                      │
    │  │    μ = mean(vol), σ = std(vol)                        │
    │  │    vol_norm = (vol - μ) / (σ + ε)                     │
    │  └─ Resize a 64×64×64 (interpolación trilineal)          │
    │       │                                                  │
    │       ▼                                                  │
    │  Backbone compartido                                     │
    │  Volumen → CLS token z ∈ ℝ^d                            │
    │       │                                                  │
    │       ▼                                                  │
    │  Router: g = softmax(W·z + b) ∈ ℝ^6                     │
    │       │                                                  │
    │       ├─── H(g) BAJA ──→ argmax(g) == 4 ──→ Experto 4   │
    │       │                                                  │
    │       └─── H(g) ALTA ──→ Experto 5 (CAE)                │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼ (si argmax == 4)
    ┌──────────────────────────────────────────────────────────┐
    │              EXPERTO 4: Swin3D-Tiny                      │
    │                                                          │
    │  Recibe: volumen [B, 1, 64, 64, 64]                     │
    │       │                                                  │
    │       ▼                                                  │
    │  Swin3D-Tiny forward pass                                │
    │  (patch embed → 4 stages × Swin blocks → pool → Linear) │
    │       │                                                  │
    │       ▼                                                  │
    │  Logits [B, 2]                                           │
    │       │                                                  │
    │       ▼                                                  │
    │  softmax → p(PDAC+), p(PDAC−)                            │
    │       │                                                  │
    │       ▼                                                  │
    │  Diagnóstico: "PDAC sospechoso" / "Sin hallazgos PDAC"   │
    │  + Confianza del modelo                                  │
    │  + Recomendación: derivar a oncología si PDAC+           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

## Hiperparámetros de entrenamiento

| Parámetro                | Valor                                                 |
|--------------------------|-------------------------------------------------------|
| **Batch size**           | 1–2 (más restrictivo que LUNA16)                      |
| **Learning rate**        | 5e-5                                                  |
| **Optimizer**            | AdamW (weight_decay=0.05)                             |
| **Scheduler**            | CosineAnnealingWarmRestarts (T_0=10, T_mult=2)        |
| **Epochs**               | 100 (early stopping patience=15)                      |
| **Precisión**            | FP16 (torch.cuda.amp)                                 |
| **Gradient checkpointing** | Sí — **obligatorio** (más agresivo que Exp3)        |
| **VRAM estimada**        | ~14 GB (batch=1), ~20 GB (batch=2)                    |
| **GPU mínima**           | NVIDIA con ≥16 GB (RTX 4080, A5000, V100)             |
| **Loss**                 | FocalLoss(alpha=0.75, gamma=2)                        |
| **Validación**           | k-fold CV (k=5) — **obligatorio** por tamaño pequeño  |
| **Augmentation**         | Random flip 3D, rotación ±10°, intensity shift ±0.1   |
| **Num workers**          | 2 (volúmenes grandes, I/O limitante)                  |

## Consideraciones especiales

### VRAM: más restrictivo que LUNA16

Swin3D-Tiny con 4 stages jerárquicos y patch merging 3D consume más memoria que ViViT-Tiny
por dos razones:

1. **Feature maps intermedios:** Stage 1 tiene 4096 tokens × 48 canales, Stage 3 tiene
   64 tokens × 192 canales. La diversidad de resoluciones fuerza a mantener más tensores
   en memoria durante backpropagation.

2. **Shifted window attention:** El mecanismo de shift requiere operaciones de roll y
   masking que consumen VRAM adicional.

Batch size de 1 es lo seguro. Batch size de 2 solo si la GPU tiene ≥20 GB y gradient
checkpointing está activo en todos los stages.

### HU clip [-100, 400]: NO es un error

| Experto | HU clip       | Razón                                                   |
|---------|---------------|---------------------------------------------------------|
| Exp 3   | [-1000, 400]  | Pulmón: incluye aire (-1000 HU) y tejido blando (+400)  |
| Exp 4   | [-100, 400]   | Abdomen: excluye aire, focaliza en tejido blando/tumor  |

El rango [-100, 400] HU maximiza el contraste entre páncreas (~40 HU), tumor PDAC
(~30-50 HU), y estructuras vasculares con contraste (~150-300 HU). Incluir aire (-1000 HU)
comprimiría toda la señal útil en una fracción del rango dinámico.

### Resize naïve y el problema del 1%

El páncreas ocupa ~1% del volumen abdominal. Un resize de 512×512×200 a 64×64×64 reduce
la resolución en ~8× por eje, lo que puede hacer que un tumor de 2cm desaparezca en 1-2
vóxeles. Estrategias de mitigación:

- Data augmentation con crops parciales centrados en región pancreática
- Attention maps para verificar que el modelo mira la zona correcta
- Si AUC < 0.80, considerar ROI cropping como preprocesado

### k-fold CV obligatorio

Con solo ~281 volúmenes utilizables, un solo split train/val/test produce estimaciones
de rendimiento con alta varianza. k-fold con k=5 da 5 estimaciones independientes del
AUC-ROC, permitiendo reportar media ± std. Si la std del AUC entre folds es >0.05,
hay que sospechar overfitting o data leakage entre folds.

### Etiquetas en repo GitHub separado

Las etiquetas PDAC+/PDAC− no están en los archivos NIfTI/DICOM sino en un CSV de un
repositorio GitHub separado. **Fijar el hash del commit** del repo de etiquetas para
garantizar reproducibilidad. Un cambio inadvertido en las etiquetas (por ejemplo, por
correcciones de los autores) invalidaría silenciosamente los resultados.

```python
# Ejemplo de fijación de commit para etiquetas
LABEL_REPO_COMMIT = "a1b2c3d4e5f6..."  # fijar hash exacto
```

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_04_swin3d_tiny/
├── model_card.md          ← este archivo
└── weights_exp4.pt        ← pesos del modelo (pendiente de entrenamiento)
```

### Cargar el modelo

```python
import torch

# Asumiendo que Swin3DTiny está definido en el proyecto
from models.expert_04 import Swin3DTiny

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Swin3DTiny(
    in_channels=1,
    num_classes=2,
    embed_dim=48,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(4, 4, 4),
    patch_size=(4, 4, 4),
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
)

checkpoint_path = "checkpoints/expert_04_swin3d_tiny/weights_exp4.pt"
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Inferencia de un volumen 3D
dummy_input = torch.randn(1, 1, 64, 64, 64, device=device)
with torch.no_grad():
    logits = model(dummy_input)  # [1, 2]
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)  # 0 = PDAC−, 1 = PDAC+
```
