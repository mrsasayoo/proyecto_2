# ViT-Tiny (vit_tiny_patch16_224) — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Familia** | Vision Transformer (ViT) |
| **Variante** | Tiny — `patch_size=16`, `img_size=224` |
| **d_model** | 192 |
| **Profundidad** | 12 bloques Transformer |
| **Num. heads** | 3 |
| **MLP ratio** | 4× |
| **Parámetros** | ~5.7 M |
| **VRAM estimada** | ~2 GB |
| **Entrada** | `[B, 3, 224, 224]` — imágenes 2D RGB normalizadas (ImageNet stats) |
| **Salida** | CLS token `z ∈ ℝ^192` |
| **Fase de uso** | Fase 1 — extracción de embeddings |
| **Checkpoint dir** | `checkpoints/backbone_01_vit_tiny/` |
| **Pesos esperados** | `backbone_vit_tiny.pt` (generado tras entrenamiento) |
| **Nombre timm** | `vit_tiny_patch16_224` |
| **Pretrained** | `False` — se entrena desde cero (requisito del proyecto) |

## Teoría del modelo

El Vision Transformer (ViT) fue propuesto por Dosovitskiy et al. (2020) y adapta la arquitectura Transformer, originalmente diseñada para procesamiento de lenguaje natural, al dominio de visión por computador. La idea central es tratar una imagen como una secuencia de tokens, análoga a una secuencia de palabras.

**Patch Embedding.** La imagen de entrada se divide en parches cuadrados no solapados de tamaño 16×16 píxeles. Cada parche se aplana a un vector de dimensión 16×16×3 = 768 y se proyecta linealmente a la dimensión del modelo (d=192) mediante una capa `Conv2d(3, 192, kernel_size=16, stride=16)`. Esto produce una secuencia de 14×14 = 196 tokens. Un token especial `[CLS]` se antepone a la secuencia, resultando en 197 tokens.

**Positional Embedding.** Se suman embeddings posicionales aprendibles a cada token para inyectar información espacial, ya que la auto-atención por sí sola es invariante al orden.

**Self-Attention (MHSA).** Cada bloque Transformer contiene un módulo Multi-Head Self-Attention con 3 cabezas. Cada cabeza opera en un subespacio de dimensión `d_head = 192 / 3 = 64`. La atención calcula:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_head) · V
```

donde Q, K, V son proyecciones lineales de los tokens de entrada. Esto permite que cada token atienda a todos los demás tokens, capturando dependencias globales. La complejidad es O(n²) respecto al número de tokens (197), lo cual es manejable para resolución 224×224 pero costoso para resoluciones mayores.

**MLP (Feed-Forward).** Tras la atención, cada token pasa por un MLP de dos capas con expansión 4× (192 → 768 → 192) y activación GELU. Esto agrega capacidad no lineal al modelo.

**CLS Token.** Después de 12 bloques Transformer, el token `[CLS]` en la posición 0 agrega información de toda la secuencia mediante atención. Este token se extrae como el embedding global de la imagen: un vector de dimensión 192.

**Sin cabeza de clasificación.** En este proyecto el modelo se instancia con `num_classes=0`, eliminando la capa lineal final de clasificación. El forward devuelve directamente el CLS token como embedding.

## Diagrama de arquitectura (ASCII)

```
  Imagen de entrada
  [B, 3, 224, 224]
        │
        ▼
┌──────────────────────────────────────────────────┐
│           Patch Embedding (Conv2d)                │
│  kernel=16×16, stride=16 → 14×14 = 196 patches   │
│  Proyección lineal: 768 → 192 por patch           │
│  + Prepend [CLS] token                            │
│  + Positional Embeddings (aprendibles, 197 pos.)  │
│                                                    │
│  Salida: [B, 197, 192]                            │
│          (196 patch tokens + 1 CLS token)          │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│         Transformer Block ×12 (idénticos)         │
│                                                    │
│  ┌────────────────────────────────────────────┐   │
│  │  LayerNorm                                  │   │
│  │  Multi-Head Self-Attention (3 heads)        │   │
│  │    Q, K, V ∈ ℝ^{197×64} por cabeza         │   │
│  │    Attn = softmax(Q·Kᵀ/√64)·V              │   │
│  │    Concat heads → Linear(192, 192)          │   │
│  │  + Residual connection                      │   │
│  ├────────────────────────────────────────────┤   │
│  │  LayerNorm                                  │   │
│  │  MLP: Linear(192→768) → GELU               │   │
│  │       Linear(768→192)                       │   │
│  │  + Residual connection                      │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  Salida: [B, 197, 192]                            │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│          Extracción del CLS token                 │
│                                                    │
│  output[:, 0, :]  →  [B, 192]                     │
│                                                    │
│  Este vector es el embedding z del backbone.       │
└──────────────────────────────────────────────────┘
        │
        ▼
   z ∈ ℝ^192
   CLS token
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
 │   │            ViT-Tiny (ESTE BACKBONE)               │                 │
 │   │                                                    │                │
 │   │  [B, 3, 224, 224] ──→ CLS token z ∈ ℝ^192        │                 │
 │   │  (congelado, requires_grad=False, eval mode)       │                │
 │   └──────────┬─────────────────────────────────────────┘                │
 │              │                                                          │
 │              ▼                                                          │
 │   Embeddings guardados: Z_{train,val,test}.npy  +  backbone_meta.json   │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
          FASE 2 — Routing + Expertos

   z ∈ ℝ^192 ──→ Router (Linear/GMM/KNN/NaiveBayes)
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

| Parámetro | Valor | Descripción |
|---|---|---|
| `img_size` | 224 | Resolución de entrada en píxeles |
| `patch_size` | 16 | Tamaño de cada parche en píxeles |
| `num_patches` | 196 | Patches por imagen (14×14) |
| `embed_dim` (d_model) | 192 | Dimensión del espacio de embedding |
| `depth` | 12 | Número de bloques Transformer |
| `num_heads` | 3 | Cabezas de atención por bloque |
| `d_head` | 64 | Dimensión por cabeza (192 / 3) |
| `mlp_ratio` | 4.0 | Factor de expansión del MLP (192 → 768) |
| `qkv_bias` | True | Bias en las proyecciones Q, K, V |
| `drop_rate` | 0.0 | Dropout en la salida del MLP |
| `attn_drop_rate` | 0.0 | Dropout en los pesos de atención |
| `drop_path_rate` | 0.0 | Stochastic depth (DropPath) |
| `num_classes` | 0 | Sin cabeza de clasificación (solo embedding) |
| `pretrained` | False | Pesos aleatorios — entrenamiento desde cero |

## Rol en el proyecto

ViT-Tiny es el **backbone por defecto** del sistema MoE y cumple varios roles:

1. **Iteración rápida.** Con solo ~5.7 M parámetros y ~2 GB de VRAM, es el backbone más ligero disponible. Permite validar el pipeline completo (Fase 0 → Fase 1 → Fase 2) con tiempos de ejecución mínimos antes de invertir en modelos más pesados.

2. **Embedding compacto.** El CLS token de 192 dimensiones produce embeddings que ocupan poco espacio en disco (~7.3 KB por cada 100 muestras en float32), facilitando la experimentación con routers y expertos en Fase 2 sin requerir almacenamiento excesivo.

3. **Baseline de atención global.** Al usar self-attention cuadrática sobre todos los patches (sin ventanas ni convoluciones), ViT-Tiny sirve como baseline puro de atención en el ablation study. La comparación con Swin-Tiny (atención local), CvT-13 (convolución + atención) y DenseNet-121 (CNN pura) permite aislar el efecto de cada mecanismo.

4. **Primera corrida de validación.** En `fase1_config.py` se define como el default en `BACKBONE_CONFIGS` y en el CLI (`--backbone vit_tiny_patch16_224`). Todo experimento comienza con ViT-Tiny para detectar errores de datos o configuración.

## Checkpoint

**Estructura del directorio:**

```
checkpoints/backbone_01_vit_tiny/
├── model_card.md              ← este archivo
└── backbone_vit_tiny.pt       ← pesos del modelo (se genera al entrenar)
```

**Carga del modelo con código Python:**

```python
import torch
import timm

# 1. Crear modelo desde cero (sin pesos preentrenados)
model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=False,
    num_classes=0,  # sin cabeza → devuelve CLS token directamente
)

# 2. Cargar pesos entrenados (si existen)
ckpt_path = "checkpoints/backbone_01_vit_tiny/backbone_vit_tiny.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 3. Congelar para extracción de embeddings (Fase 1)
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to("cuda")

# 4. Extraer CLS token
img = torch.randn(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    z = model(img)  # z.shape = [1, 192]
```

**Uso a través del pipeline oficial (recomendado):**

```python
# backbone_loader.py se encarga de todo: creación, congelamiento, verificación
from backbone_loader import load_frozen_backbone

model, d_model = load_frozen_backbone("vit_tiny_patch16_224", device="cuda")
# model: congelado, eval(), en device
# d_model: 192 (verificado empíricamente con dummy forward)
```

## Notas de entrenamiento

| Parámetro | Recomendación |
|---|---|
| **Batch size** | 64 (GPU con ≥8 GB VRAM) — reducir a 16 en CPU |
| **Learning rate** | 1e-4 a 3e-4 (AdamW) |
| **Optimizador** | AdamW con weight_decay=0.05 |
| **Scheduler** | Cosine annealing con warmup lineal (5-10 épocas) |
| **Criterio de parada** | Early stopping en val loss (paciencia 10 épocas) |
| **Épocas máximas** | 100 |
| **Normalización** | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **VRAM pico** | ~2 GB (batch_size=64, img 224×224) |
| **torch.compile** | Activado automáticamente si PyTorch ≥ 2.0 (mode=reduce-overhead) |

**Consideraciones:**

- El modelo se entrena **end-to-end desde cero** con pesos aleatorios (`pretrained=False`). No se utilizan pesos preentrenados de ImageNet ni ningún otro dataset externo, cumpliendo el requisito académico del proyecto.
- En modo CPU, el pipeline reduce automáticamente el batch size a 16 para ajustarse al caché L2/L3 (`OPT-2` en `fase1_pipeline.py`).
- `torch.compile(mode="reduce-overhead")` se aplica automáticamente tras congelar el modelo, mejorando el throughput de inferencia un 10-30%.
- La extracción usa `torch.inference_mode()` (no `torch.no_grad()`) para mayor eficiencia, ya que desactiva tanto autograd como el versionado de tensores.
- El pipeline monitorea VRAM cada 200 batches y emite un warning si el uso supera el 90%.
