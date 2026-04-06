# Experto 3 — ViViT-Tiny — Model Card

## Resumen

| Campo                | Valor                                                                 |
|----------------------|-----------------------------------------------------------------------|
| **Experto ID**       | 3                                                                     |
| **Arquitectura**     | ViViT-Tiny (Video Vision Transformer adaptado a 3D médico)           |
| **Dataset**          | LUNA16 / LIDC-IDRI                                                    |
| **Modalidad**        | CT pulmonar 3D — parches centrados en candidatos a nódulo             |
| **Tarea**            | Clasificación binaria de parches (nódulo sí / nódulo no)              |
| **Num clases**       | 2                                                                     |
| **Tensor entrada**   | `[B, 1, 64, 64, 64]`                                                 |
| **Loss**             | `FocalLoss(gamma=2, alpha=0.25)` — **obligatoria**                   |
| **Métricas**         | CPM + curva FROC (`noduleCADEvaluationLUNA16.py`). Techo: 94.4%     |
| **Volumen datos**    | ~17K parches (train: 14,728 / val: 1,143 / test: 1,914)             |
| **Preprocesado**     | HU clip [-1000, 400] → normalización [0, 1] → parche 64³            |
| **Checkpoint dir**   | `checkpoints/expert_03_vivit_tiny/`                                   |
| **Pesos esperados**  | `weights_exp3.pt`                                                     |

## Teoría del modelo

### ViViT: de secuencias de vídeo a volúmenes médicos 3D

ViViT (Video Vision Transformer) fue propuesto originalmente por Arnab et al. (2021) para
clasificación de vídeo, extendiendo ViT de imágenes 2D a secuencias espacio-temporales. La
intuición clave es que un volumen CT 3D y un clip de vídeo comparten la misma estructura
tensorial: tres dimensiones espaciales (o dos espaciales + una temporal) que pueden
descomponerse en parches volumétricos (tubelets).

**¿Por qué ViViT para nódulos pulmonares?**

Los nódulos pulmonares son estructuras intrínsecamente 3D. Un nódulo de 6mm puede
ocupar 3-4 cortes axiales consecutivos, y su morfología (espiculado, lobulado, ground-glass)
solo es completamente visible cuando se analiza el volumen completo. Las CNNs 3D
(como ResNet3D) capturan patrones locales con kernels 3×3×3, pero los Transformers
permiten modelar dependencias de largo alcance entre regiones distantes del parche:
por ejemplo, la relación entre el borde espiculado de un nódulo y la retracción pleural
adyacente.

**Tubelet embedding vs. patch embedding frame-a-frame**

ViViT ofrece dos estrategias de tokenización. Este modelo usa **tubelet embedding**: cada
token corresponde a un cubo 8×8×8 del volumen original. Para un volumen de 64³, esto
produce (64/8)³ = 8³ = 512 tokens espaciales. La ventaja sobre tokenizar frame a frame es
que cada token ya contiene información 3D local, reduciendo la carga sobre la self-attention
para capturar coherencia entre cortes.

**Atención global y el CLS token**

Los 512 tokens + 1 CLS token pasan por 12 bloques de self-attention multi-head (3 heads,
d_model=192). El CLS token agrega información de todo el volumen y se usa como
representación para la clasificación final. Esto es análogo a cómo BERT usa su CLS token
para tareas de clasificación de secuencia completa.

### Focal Loss: por qué es no negociable

El dataset LUNA16 tiene un desbalance extremo de ~490:1 (candidatos negativos vs. nódulos
reales). Con BCELoss estándar, el gradiente está dominado por los negativos fáciles: el modelo
aprende a predecir siempre "no nódulo" y obtiene 99.8% de accuracy pero AUC ~0.50.

Focal Loss (Lin et al., 2017) resuelve esto con dos mecanismos:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

1. **Factor modulador `(1 - p_t)^gamma`**: con gamma=2, cuando el modelo predice
   correctamente un negativo fácil con p_t=0.95, el factor es (0.05)²=0.0025, reduciendo
   la contribución de ese ejemplo al loss en 400x. Los positivos difíciles (p_t bajo)
   mantienen su contribución completa.

2. **Factor de balanceo `alpha`**: alpha=0.25 para la clase positiva (nódulos) puede parecer
   contra-intuitivo, pero compensa la mayor cantidad de gradiente que los positivos reciben
   por el factor modulador. En la práctica, este alpha fue calibrado empíricamente para LUNA16.

## Diagrama de arquitectura (ASCII)

```
                        ENTRADA: [B, 1, 64, 64, 64]
                        (parche CT centrado en candidato a nódulo)
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │         TUBELET EMBEDDING 3D                │
              │  Conv3d(1, 192, kernel=8, stride=8)         │
              │  [B, 1, 64, 64, 64] → [B, 192, 8, 8, 8]   │
              │  Reshape → [B, 512, 192]                    │
              │  (512 tokens espaciales, dim=192)           │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │         PREPEND CLS TOKEN                   │
              │  [B, 512, 192] → [B, 513, 192]             │
              │  CLS = nn.Parameter(1, 1, 192)              │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │      POSITIONAL EMBEDDING 3D                │
              │  pos_embed = nn.Parameter(1, 513, 192)      │
              │  tokens = tokens + pos_embed                │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │       TRANSFORMER ENCODER ×12               │
              │                                             │
              │  ┌───────────────────────────────────────┐  │
              │  │  Layer Norm                           │  │
              │  │  Multi-Head Self-Attention            │  │
              │  │    heads=3, d_k=64, d_model=192      │  │
              │  │    Q, K, V: [B, 513, 192]            │  │
              │  │    Attn: [B, 3, 513, 513]            │  │
              │  │  + Residual connection               │  │
              │  ├───────────────────────────────────────┤  │
              │  │  Layer Norm                           │  │
              │  │  MLP (FFN):                          │  │
              │  │    Linear(192, 768) → GELU            │  │
              │  │    Dropout(0.1)                       │  │
              │  │    Linear(768, 192)                   │  │
              │  │    Dropout(0.1)                       │  │
              │  │  + Residual connection               │  │
              │  └───────────────────────────────────────┘  │
              │              × 12 bloques                   │
              │  (gradient checkpointing habilitado)        │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │           CLASSIFICATION HEAD               │
              │  Extraer CLS token: [B, 513, 192] → [:, 0] │
              │  → [B, 192]                                 │
              │  LayerNorm(192)                              │
              │  Linear(192, 2)                              │
              │  → [B, 2] (logits: nódulo / no-nódulo)      │
              └─────────────────────────────────────────────┘
                                      │
                                      ▼
                        SALIDA: [B, 2] logits
                        FocalLoss(gamma=2, alpha=0.25)
```

## Diagrama de integración MoE (ASCII)

```
    ┌──────────────────────────────────────────────────────────┐
    │                    PIPELINE MoE                          │
    │                                                          │
    │  Volumen CT completo                                     │
    │       │                                                  │
    │       ▼                                                  │
    │  LUNA16PatchExtractor                                    │
    │  ├─ Lee candidates_V2.csv (⚠ NUNCA V1)                  │
    │  ├─ world_to_voxel(x, y, z) → (i, j, k)                │
    │  ├─ HU clip [-1000, 400]                                 │
    │  ├─ Normalización [0, 1]                                 │
    │  └─ Extrae parche 64³ centrado en (i,j,k)               │
    │       │                                                  │
    │       ▼                                                  │
    │  Backbone compartido                                     │
    │  Parche → CLS token z ∈ ℝ^d                             │
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
    │              EXPERTO 3: ViViT-Tiny                       │
    │                                                          │
    │  Recibe: parche [B, 1, 64, 64, 64]                      │
    │       │                                                  │
    │       ▼                                                  │
    │  ViViT-Tiny forward pass                                 │
    │  (tubelet embed → 12× Transformer → CLS → Linear)       │
    │       │                                                  │
    │       ▼                                                  │
    │  Logits [B, 2]                                           │
    │       │                                                  │
    │       ▼                                                  │
    │  softmax → p(nódulo), p(no-nódulo)                       │
    │       │                                                  │
    │       ▼                                                  │
    │  Diagnóstico: "Nódulo detectado" / "Sin hallazgos"       │
    │  + Coordenada (x,y,z) del candidato original             │
    │  + Confianza del modelo                                  │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

## Hiperparámetros de entrenamiento

| Parámetro                | Valor                                           |
|--------------------------|-------------------------------------------------|
| **Batch size**           | 4                                               |
| **Learning rate**        | 1e-4                                            |
| **Optimizer**            | AdamW (weight_decay=0.01)                       |
| **Scheduler**            | CosineAnnealingLR (T_max=epochs)                |
| **Epochs**               | 50 (early stopping patience=10)                 |
| **Precisión**            | FP16 (torch.cuda.amp)                           |
| **Gradient checkpointing** | Sí — **obligatorio**                          |
| **VRAM estimada**        | ~12 GB                                          |
| **GPU mínima**           | NVIDIA con ≥12 GB (RTX 3060, A5000, etc.)       |
| **Loss**                 | FocalLoss(gamma=2, alpha=0.25)                  |
| **Augmentation**         | Random flip 3D, rotación ±15°, elastic deform   |
| **Num workers**          | 4                                               |

## Consideraciones especiales

### VRAM y gradient checkpointing

Con 512 tokens por volumen y 12 capas de Transformer, las activaciones intermedias consumen
memoria cuadráticamente respecto al número de tokens. Sin gradient checkpointing, un batch
de 4 volúmenes requiere >24 GB de VRAM. Con checkpointing activado, el forward pass no
almacena activaciones intermedias (se recalculan durante el backward pass), reduciendo el
consumo a ~12 GB a costa de ~30% más tiempo de entrenamiento.

```python
# Activar gradient checkpointing
from torch.utils.checkpoint import checkpoint

class ViViTBlock(nn.Module):
    def forward(self, x):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
```

### FP16 (Mixed Precision Training)

Obligatorio para mantener el consumo de VRAM dentro de los 12 GB. Se usa
`torch.cuda.amp.GradScaler` para evitar underflow en gradientes FP16.

### Spacing variable entre volúmenes LUNA16

Los volúmenes del LIDC-IDRI tienen spacing variable (e.g., 0.5mm×0.5mm×2.5mm vs.
0.7mm×0.7mm×1.0mm). El `__init__` del dataset verifica el spacing de cada volumen y la
conversión de coordenadas mundo a vóxel se realiza con `LUNA16PatchExtractor.world_to_voxel()`.
Si el spacing no se maneja correctamente, el parche 64³ puede no estar centrado en el nódulo
real.

### candidates_V2.csv — SIEMPRE V2

`candidates_V2.csv` corrige 24 nódulos que estaban mal etiquetados o ausentes en V1.
Usar V1 produce una caída silenciosa en sensibilidad que no se detecta hasta evaluar
con FROC. **No hay ningún escenario donde V1 sea preferible.**

### Focal Loss es no negociable

| Loss         | AUC-ROC | Sensibilidad @4FP/scan | Notas                       |
|--------------|---------|------------------------|-----------------------------|
| BCELoss      | ~0.50   | ~0%                    | Modelo trivial (todo = 0)   |
| FocalLoss    | >0.90   | >80%                   | Funcional                   |

Con BCELoss, el modelo converge a predecir siempre la clase mayoritaria (no-nódulo).
El accuracy es 99.8% pero la utilidad clínica es nula. Focal Loss fuerza al modelo a
aprender de los pocos positivos disponibles.

### Techo de sensibilidad: 94.4%

No todos los nódulos del LIDC-IDRI tienen acuerdo unánime entre radiólogos. El 5.6%
restante corresponde a nódulos con anotación ambigua (acuerdo ≤2/4 radiólogos). Un modelo
perfecto en LUNA16 alcanza como máximo 94.4% de sensibilidad.

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_03_vivit_tiny/
├── model_card.md          ← este archivo
└── weights_exp3.pt        ← pesos del modelo (pendiente de entrenamiento)
```

### Cargar el modelo

```python
import torch

# Asumiendo que ViViTTiny está definido en el proyecto
from models.expert_03 import ViViTTiny

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViViTTiny(
    in_channels=1,
    img_size=64,
    patch_size=8,
    num_classes=2,
    d_model=192,
    num_heads=3,
    num_layers=12,
    mlp_ratio=4,
)

checkpoint_path = "checkpoints/expert_03_vivit_tiny/weights_exp3.pt"
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Inferencia de un parche 3D
dummy_input = torch.randn(1, 1, 64, 64, 64, device=device)
with torch.no_grad():
    logits = model(dummy_input)  # [1, 2]
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)  # 0 = no nódulo, 1 = nódulo
```
