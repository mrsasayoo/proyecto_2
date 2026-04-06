# Swin-Tiny (swin_tiny_patch4_window7_224) — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Familia** | Swin Transformer (Shifted Window) |
| **Variante** | Tiny — `patch_size=4`, `window_size=7`, `img_size=224` |
| **d_model** | 768 (dimensión final tras 4 stages) |
| **Stages** | 4 (C: 96 → 192 → 384 → 768) |
| **Profundidad** | [2, 2, 6, 2] bloques por stage (12 total) |
| **Num. heads** | [3, 6, 12, 24] por stage |
| **MLP ratio** | 4× en todos los stages |
| **Parámetros** | ~28 M |
| **VRAM estimada** | ~4 GB |
| **Entrada** | `[B, 3, 224, 224]` — imágenes 2D RGB normalizadas (ImageNet stats) |
| **Salida** | CLS token (via global average pool) `z ∈ ℝ^768` |
| **Fase de uso** | Fase 1 — extracción de embeddings |
| **Checkpoint dir** | `checkpoints/backbone_03_swin_tiny/` |
| **Pesos esperados** | `backbone_swin_tiny.pt` (generado tras entrenamiento) |
| **Nombre timm** | `swin_tiny_patch4_window7_224` |
| **Pretrained** | `False` — se entrena desde cero (requisito del proyecto) |

## Teoría del modelo

Swin Transformer fue propuesto por Liu et al. (ICCV 2021, Best Paper) y aborda el principal cuello de botella computacional de ViT: la complejidad cuadrática de la auto-atención. En ViT, cada token atiende a todos los demás tokens — para una imagen de 224×224 con patches de 16×16 esto son 196 tokens y una matriz de atención de 196×196, que es manejable. Pero si se reduce el tamaño de patch a 4×4 (para capturar detalles más finos), la secuencia crece a 3136 tokens y la matriz de atención a 3136×3136 (~10M elementos por cabeza), que es prohibitivo.

**Window-based Self-Attention (W-MSA).** Swin divide el mapa de features en ventanas no solapadas de 7×7 tokens. La auto-atención se calcula solo dentro de cada ventana (49 tokens), reduciendo la complejidad de O(n²) global a O(n) respecto al tamaño de la imagen. Para una entrada de 56×56 tokens, esto significa 64 ventanas de 7×7, cada una con una matriz de atención de 49×49 — mucho más eficiente que la alternativa global de 3136×3136.

**Shifted Window (SW-MSA).** Si la atención fuera solo local (dentro de ventanas fijas), las ventanas no podrían comunicarse entre sí. Swin resuelve esto alternando entre dos configuraciones de ventana en bloques sucesivos:

```
Bloque par:  ventanas regulares (sin desplazamiento)
Bloque impar: ventanas desplazadas por (⌊W/2⌋, ⌊W/2⌋) = (3, 3) píxeles
```

Este desplazamiento hace que los tokens que estaban en el borde de ventanas adyacentes ahora compartan ventana, permitiendo la comunicación cross-window sin costo computacional adicional significativo.

**Patch Merging (downsampling jerárquico).** Entre stages, Swin reduce la resolución espacial 2× y duplica la dimensión de canales. Esto es análogo al maxpool + duplicación de canales en una CNN:

```
Stage 1: 56×56 tokens, C=96
   ↓ Patch Merge: concatenar 2×2 vecinos → Linear(4C→2C)
Stage 2: 28×28 tokens, C=192
   ↓ Patch Merge
Stage 3: 14×14 tokens, C=384
   ↓ Patch Merge
Stage 4: 7×7 tokens, C=768
```

Esta jerarquía produce feature maps multi-escala, haciendo a Swin compatible con Feature Pyramid Networks (FPN) y otros métodos que requieren representaciones piramidales.

**Relative Position Bias.** En lugar de los embeddings posicionales absolutos de ViT, Swin usa un sesgo de posición relativa aprendible que se suma a las puntuaciones de atención. Dado que la atención es local (dentro de ventanas de 7×7), los desplazamientos relativos posibles van de -6 a +6 en cada eje, resultando en una tabla de (2×7-1)² = 169 entradas por cabeza. Este sesgo relativo generaliza mejor a diferentes resoluciones que los embeddings absolutos.

**CLS Token via Global Average Pooling.** Swin no usa un CLS token explícito como ViT. En su lugar, tras el último stage, se aplica Global Average Pooling sobre los 7×7=49 tokens restantes para producir un vector de 768 dimensiones. En `timm`, configurar `num_classes=0` hace que el modelo devuelva este vector directamente.

## Diagrama de arquitectura (ASCII)

```
  Imagen de entrada
  [B, 3, 224, 224]
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│             Patch Embedding (Patch Partition + Linear)            │
│                                                                    │
│  Conv2d(3→96, kernel=4, stride=4)  →  [B, 96, 56, 56]            │
│  LayerNorm                                                         │
│  Reshape → [B, 3136, 96]    (56×56 = 3136 tokens, dim=96)        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│           STAGE 1 — [B, 3136, 96]  (56×56 tokens, C=96)          │
│                                                                    │
│  Swin Transformer Block ×2                                         │
│    ┌──────────────────────────────────────────────────────┐       │
│    │  Block 0: W-MSA (ventanas regulares 7×7)             │       │
│    │    - 3 heads, d_head=32                               │       │
│    │    - Atención local: 49 tokens por ventana            │       │
│    │    - 8×8 = 64 ventanas en paralelo                    │       │
│    │    - Relative position bias (tabla 13×13)             │       │
│    │  + MLP(96→384→96) + Residual                          │       │
│    ├──────────────────────────────────────────────────────┤       │
│    │  Block 1: SW-MSA (ventanas desplazadas por (3,3))    │       │
│    │    - Misma config, ventanas desplazadas               │       │
│    │    - Comunicación cross-window                        │       │
│    │  + MLP(96→384→96) + Residual                          │       │
│    └──────────────────────────────────────────────────────┘       │
│                                                                    │
│  Salida: [B, 3136, 96]                                            │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Patch Merging:  concat 2×2 → Linear(384→192)                    │
│  [B, 3136, 96] → [B, 784, 192]   (28×28 tokens, C=192)          │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│           STAGE 2 — [B, 784, 192]  (28×28 tokens, C=192)         │
│                                                                    │
│  Swin Transformer Block ×2                                         │
│    Block 0: W-MSA (6 heads, d_head=32)                             │
│    Block 1: SW-MSA (6 heads, desplazado)                           │
│                                                                    │
│  Salida: [B, 784, 192]                                            │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Patch Merging:  concat 2×2 → Linear(768→384)                    │
│  [B, 784, 192] → [B, 196, 384]  (14×14 tokens, C=384)           │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│           STAGE 3 — [B, 196, 384]  (14×14 tokens, C=384)         │
│                                                                    │
│  Swin Transformer Block ×6                                         │
│    Blocks 0,2,4: W-MSA (12 heads, d_head=32)                      │
│    Blocks 1,3,5: SW-MSA (12 heads, desplazado)                     │
│                                                                    │
│  Salida: [B, 196, 384]                                            │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Patch Merging:  concat 2×2 → Linear(1536→768)                   │
│  [B, 196, 384] → [B, 49, 768]  (7×7 tokens, C=768)              │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│           STAGE 4 — [B, 49, 768]  (7×7 tokens, C=768)            │
│                                                                    │
│  Swin Transformer Block ×2                                         │
│    Block 0: W-MSA (24 heads, d_head=32)                            │
│             (ventana = 7×7 = toda la feature map → atención global)│
│    Block 1: SW-MSA (24 heads)                                      │
│                                                                    │
│  Salida: [B, 49, 768]                                             │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│           Global Average Pooling                                   │
│                                                                    │
│  LayerNorm → mean(dim=1) → [B, 768]                               │
│                                                                    │
│  (No hay CLS token explícito — se promedian los 49 tokens)        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
   z ∈ ℝ^768
   Embedding global
```

## Diagrama en el sistema MoE (ASCII)

```
                         FASE 1 — Extracción de Embeddings
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                                                                         │
 │   Imagen / Volumen                                                      │
 │   ───────────────                                                       │
 │         │                                                               │
 │         ▼                                                               │
 │   ┌──────────────────────┐                                              │
 │   │  Preprocesador        │  Normalización ImageNet                     │
 │   │  Adaptativo (2D/3D)   │  mean=[0.485, 0.456, 0.406]                │
 │   │                        │  std =[0.229, 0.224, 0.225]                │
 │   │  - CLAHE (OA)          │  Resize a 224×224                          │
 │   │  - HU clip (CT)        │  Para 3D: proyección central slice → 2D    │
 │   │  - Gamma (ISIC)        │                                            │
 │   └──────────┬─────────────┘                                            │
 │              │                                                          │
 │              ▼                                                          │
 │   ┌──────────────────────────────────────────────────┐                  │
 │   │         Swin-Tiny (ESTE BACKBONE)                 │                 │
 │   │                                                    │                │
 │   │  [B, 3, 224, 224] ──→ GAP embedding z ∈ ℝ^768    │                 │
 │   │  (congelado, requires_grad=False, eval mode)       │                │
 │   │                                                    │                │
 │   │  timm.create_model("swin_tiny_patch4_window7_224") │                │
 │   └──────────┬─────────────────────────────────────────┘                │
 │              │                                                          │
 │              ▼                                                          │
 │   Embeddings guardados: Z_{train,val,test}.npy  +  backbone_meta.json   │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
          FASE 2 — Routing + Expertos

   z ∈ ℝ^768 ──→ Router (Linear/GMM/KNN/NaiveBayes)
                     │
                     ▼
              ┌──────┴──────────────────────────────────┐
              │    Selección del experto más adecuado     │
              │                                           │
              │  Exp 0: Chest (NIH ChestXray14)           │
              │  Exp 1: ISIC (dermatoscopía)              │
              │  Exp 2: OA (rodilla)                      │
              │  Exp 3: LUNA16 (CT pulmón 3D)             │
              │  Exp 4: Pancreas (CT abdomen 3D)          │
              │  Exp 5: OOD (fuera de distribución)       │
              └───────────────────────────────────────────┘
```

## Hiperparámetros clave

| Parámetro | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Descripción |
|---|---|---|---|---|---|
| `embed_dim` | 96 | 192 | 384 | 768 | Dimensión de los tokens |
| `num_heads` | 3 | 6 | 12 | 24 | Cabezas de atención |
| `depth` | 2 | 2 | 6 | 2 | Bloques Transformer por stage |
| `d_head` | 32 | 32 | 32 | 32 | Dimensión constante por cabeza |
| `tokens` | 56×56 | 28×28 | 14×14 | 7×7 | Resolución de tokens |

| Parámetro global | Valor | Descripción |
|---|---|---|
| `patch_size` | 4 | Tamaño del parche inicial en píxeles |
| `window_size` | 7 | Tokens por ventana en cada eje |
| `img_size` | 224 | Resolución de entrada |
| `mlp_ratio` | 4.0 | Expansión del MLP |
| `qkv_bias` | True | Bias en proyecciones Q, K, V |
| `drop_rate` | 0.0 | Dropout |
| `drop_path_rate` | 0.2 | Stochastic depth (linealmente creciente) |
| `num_classes` | 0 | Sin cabeza de clasificación (solo embedding) |
| `pretrained` | False | Pesos aleatorios (requisito del proyecto) |
| `d_model` (salida) | 768 | Dimensión del embedding final (GAP) |

## Rol en el proyecto

Swin-Tiny es el backbone más pesado y expresivo del sistema, reservado para el **ablation study final**:

1. **Embedding de alta dimensionalidad.** Con d_model=768, produce el embedding más rico del proyecto (4× ViT-Tiny, 2× CvT-13). Esto permite evaluar si los routers de Fase 2 se benefician de representaciones más detalladas o si la dimensionalidad extra introduce ruido y dificulta la separación entre dominios.

2. **Atención local eficiente.** La complejidad lineal O(n) de Swin (vs. O(n²) de ViT) permite usar patches más pequeños (4×4 en vez de 16×16), capturando detalles finos que son relevantes en imágenes médicas: micro-calcificaciones en radiografías, bordes de lesiones en dermatoscopía, y texturas tisulares en CT.

3. **Representación jerárquica.** Los 4 stages con resoluciones decrecientes (56→28→14→7) capturan información a múltiples escalas. Esta jerarquía es análoga a las Feature Pyramid Networks de las CNNs y permite representar tanto patrones locales (nódulos de 3mm) como contexto global (lateralidad de un pulmón).

4. **Ablation study final.** Swin-Tiny se ejecuta al final del proyecto para comparar con ViT-Tiny (atención global, patches gruesos), CvT-13 (convolución + atención) y DenseNet-121 (CNN pura). La pregunta experimental es: ¿la atención local con ventanas desplazadas supera a la atención global para routing de imágenes médicas?

5. **Costo computacional justificado.** Con ~4 GB de VRAM, solo se usa cuando los resultados de los backbones más ligeros sugieren que la capacidad del modelo es el factor limitante (no los datos ni el router).

## Checkpoint

**Estructura del directorio:**

```
checkpoints/backbone_03_swin_tiny/
├── model_card.md              ← este archivo
└── backbone_swin_tiny.pt      ← pesos del modelo (se genera al entrenar)
```

**Carga del modelo con código Python:**

```python
import torch
import timm

# 1. Crear modelo desde cero (sin pesos preentrenados)
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=0,  # sin cabeza → devuelve embedding tras GAP
)

# 2. Cargar pesos entrenados (si existen)
ckpt_path = "checkpoints/backbone_03_swin_tiny/backbone_swin_tiny.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 3. Congelar para extracción de embeddings (Fase 1)
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to("cuda")

# 4. Extraer embedding
img = torch.randn(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    z = model(img)  # z.shape = [1, 768]
```

**Uso a través del pipeline oficial (recomendado):**

```python
from backbone_loader import load_frozen_backbone

model, d_model = load_frozen_backbone("swin_tiny_patch4_window7_224", device="cuda")
# model: congelado, eval(), en device
# d_model: 768 (verificado empíricamente con dummy forward)
```

## Notas de entrenamiento

| Parámetro | Recomendación |
|---|---|
| **Batch size** | 32 (GPU con ≥8 GB VRAM) — reducir a 8 en CPU |
| **Learning rate** | 5e-5 a 2e-4 (AdamW) |
| **Optimizador** | AdamW con weight_decay=0.05 |
| **Scheduler** | Cosine annealing con warmup lineal (10-20 épocas) |
| **Criterio de parada** | Early stopping en val loss (paciencia 10 épocas) |
| **Épocas máximas** | 100-150 |
| **Normalización** | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **VRAM pico** | ~4 GB (batch_size=32, img 224×224) |
| **Augmentation** | RandAugment o AutoAugment recomendado por el mayor número de parámetros |

**Consideraciones:**

- Swin-Tiny tiene ~28 M de parámetros, significativamente más que ViT-Tiny (~5.7 M). Entrenar desde cero requiere más datos y más épocas para converger. Considerar un learning rate más bajo y warmup más largo.
- El batch size debería ser menor que para ViT-Tiny (32 vs. 64) debido al mayor consumo de VRAM. En CPU, reducir a 8 para ajustarse al caché.
- La `drop_path_rate=0.2` ya proporciona regularización sustancial. Evaluar si dropout adicional en el MLP es necesario según la curva de validación.
- En el Stage 4, la ventana de 7×7 cubre la totalidad del feature map de 7×7, convirtiendo la atención local en atención global de facto. Esto significa que la última capa sí captura dependencias globales.
- `torch.compile(mode="reduce-overhead")` se aplica automáticamente en `backbone_loader.py`. Para Swin, el speedup típico es del 15-25%.
- El pipeline monitorea VRAM cada 200 batches. Con Swin-Tiny y batch_size=32, el uso de VRAM debería estabilizarse en ~3.5-4 GB.
