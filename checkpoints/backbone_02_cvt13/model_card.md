# CvT-13 (Convolutional Vision Transformer) — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Familia** | CvT (Convolutional Vision Transformer) |
| **Variante** | CvT-13 — 3 etapas con convoluciones que generan tokens |
| **d_model** | 384 (dimensión de salida de la etapa 3) |
| **Etapas** | 3 (embed_dim: 64 → 192 → 384) |
| **Profundidad** | [1, 2, 10] bloques por etapa (13 total) |
| **Num. heads** | [1, 3, 6] por etapa |
| **MLP ratio** | 4× en todas las etapas |
| **Parámetros** | ~20 M |
| **VRAM estimada** | ~3 GB |
| **Entrada** | `[B, 3, 224, 224]` — imágenes 2D RGB normalizadas (ImageNet stats) |
| **Salida** | CLS token `z ∈ ℝ^384` |
| **Fase de uso** | Fase 1 — extracción de embeddings |
| **Checkpoint dir** | `checkpoints/backbone_02_cvt13/` |
| **Pesos esperados** | `backbone_cvt13.pt` (generado tras entrenamiento) |
| **Nombre timm** | `cvt_13` (interceptado → HuggingFace `CvtModel`) |
| **Pretrained** | `False` — se entrena desde cero (requisito del proyecto) |

## Teoría del modelo

CvT fue propuesto por Wu et al. (ICCV 2021) y resuelve una limitación fundamental de ViT: la falta de inductive bias espacial. Mientras ViT trata cada parche de imagen como un token independiente (perdiendo las relaciones de vecindad), CvT introduce convoluciones en la generación de tokens y en las proyecciones de atención, preservando la estructura espacial sin sacrificar la capacidad de modelar dependencias globales.

**Diseño multi-etapa con convoluciones.** CvT-13 organiza el procesamiento en 3 etapas jerárquicas, similar a cómo una CNN reduce la resolución progresivamente:

- **Etapa 1** (dim=64): Un Convolutional Token Embedding con kernel 7×7 y stride 4 convierte la imagen en tokens de baja resolución (56×56). Un bloque Transformer con 1 cabeza procesa estos tokens. Esta primera etapa captura patrones locales de baja frecuencia.

- **Etapa 2** (dim=192): Otro embedding convolucional (kernel 3×3, stride 2) reduce a 28×28 tokens y proyecta a 192 dimensiones. Dos bloques Transformer con 3 cabezas refinan las representaciones.

- **Etapa 3** (dim=384): Un tercer embedding convolucional (kernel 3×3, stride 2) comprime a 14×14 tokens con 384 dimensiones. Diez bloques Transformer con 6 cabezas producen las representaciones finales. Solo en esta etapa se introduce el CLS token.

**Convolutional Token Embedding.** En lugar del parche lineal de ViT (`Conv2d` con kernel=stride), CvT usa convoluciones con overlap (padding) entre etapas. Esto crea una reducción gradual de resolución que preserva información de frontera entre parches, generando mejores representaciones locales.

**Convolutional Projection en Q, K, V.** La innovación central de CvT reemplaza las proyecciones lineales estándar de Q, K y V por convoluciones depth-wise separables. En la auto-atención clásica de ViT:

```
Q = X · W_q    (proyección lineal)
```

CvT utiliza:

```
Q = DWConv(X) · W_q    (convolución depth-wise + proyección)
```

Esto inyecta inductive bias espacial directamente en el mecanismo de atención: cada token "sabe" quiénes son sus vecinos antes de calcular las puntuaciones de atención. El resultado es una convergencia más rápida y mejor rendimiento en tareas de visión, particularmente con datos limitados, que es relevante para imágenes médicas.

**CLS Token tardío.** A diferencia de ViT que prepone el CLS token desde el inicio, CvT solo lo introduce en la etapa 3. Las dos primeras etapas procesan tokens puramente espaciales sin el overhead del CLS token, mejorando la eficiencia computacional.

## Diagrama de arquitectura (ASCII)

```
  Imagen de entrada
  [B, 3, 224, 224]
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                ETAPA 1 (dim=64, depth=1)                      │
│                                                                │
│  Convolutional Token Embedding                                 │
│    Conv2d(3→64, k=7, stride=4, pad=2) + LayerNorm             │
│    Salida: [B, 64, 56, 56] → reshape → [B, 3136, 64]         │
│                                                                │
│  Transformer Block ×1                                          │
│    ┌─ Convolutional MHSA (1 head, d_head=64) ──────────┐      │
│    │  Q = DWConv3×3(X) · W_q                            │      │
│    │  K = DWConv3×3(X) · W_k                            │      │
│    │  V = DWConv3×3(X) · W_v                            │      │
│    │  Attn = softmax(Q·Kᵀ/√64)·V                       │      │
│    └────────────────────────────────────────────────────┘      │
│    MLP: Linear(64→256) → GELU → Linear(256→64)                │
│                                                                │
│  Salida: [B, 3136, 64]                                        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                ETAPA 2 (dim=192, depth=2)                     │
│                                                                │
│  Convolutional Token Embedding                                 │
│    Conv2d(64→192, k=3, stride=2, pad=1) + LayerNorm           │
│    Salida: [B, 192, 28, 28] → reshape → [B, 784, 192]        │
│                                                                │
│  Transformer Block ×2                                          │
│    Convolutional MHSA (3 heads, d_head=64)                     │
│    MLP: Linear(192→768) → GELU → Linear(768→192)              │
│                                                                │
│  Salida: [B, 784, 192]                                        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                ETAPA 3 (dim=384, depth=10)                     │
│                                                                │
│  Convolutional Token Embedding                                 │
│    Conv2d(192→384, k=3, stride=2, pad=1) + LayerNorm          │
│    Salida: [B, 384, 14, 14] → reshape → [B, 196, 384]        │
│    + Prepend [CLS] token → [B, 197, 384]                      │
│                                                                │
│  Transformer Block ×10                                         │
│    Convolutional MHSA (6 heads, d_head=64)                     │
│    MLP: Linear(384→1536) → GELU → Linear(1536→384)            │
│    drop_path_rate = 0.1 (stochastic depth)                     │
│                                                                │
│  Salida: [B, 197, 384]                                        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│              Extracción del CLS token                          │
│                                                                │
│  cls_token_value → squeeze(1) → [B, 384]                      │
│  (fallback: last_hidden_state[:, 0, :] si cls_token is None)  │
│                                                                │
│  Pasa por self.proj (nn.Identity por defecto)                  │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
   z ∈ ℝ^384
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
 │   │      CvT-13 (ESTE BACKBONE) — via HuggingFace    │                 │
 │   │                                                    │                │
 │   │  [B, 3, 224, 224] ──→ CLS token z ∈ ℝ^384        │                 │
 │   │  (congelado, requires_grad=False, eval mode)       │                │
 │   │                                                    │                │
 │   │  timm.create_model("cvt_13") interceptado →        │                │
 │   │  → backbone_cvt13.build_cvt13() → CvT13Wrapper     │                │
 │   └──────────┬─────────────────────────────────────────┘                │
 │              │                                                          │
 │              ▼                                                          │
 │   Embeddings guardados: Z_{train,val,test}.npy  +  backbone_meta.json   │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
          FASE 2 — Routing + Expertos

   z ∈ ℝ^384 ──→ Router (Linear/GMM/KNN/NaiveBayes)
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

| Parámetro | Etapa 1 | Etapa 2 | Etapa 3 | Descripción |
|---|---|---|---|---|
| `patch_size` | 7 | 3 | 3 | Kernel del embedding convolucional |
| `patch_stride` | 4 | 2 | 2 | Stride del embedding (reduce resolución) |
| `patch_padding` | 2 | 1 | 1 | Padding del embedding (overlap) |
| `embed_dim` | 64 | 192 | 384 | Dimensión de los tokens |
| `num_heads` | 1 | 3 | 6 | Cabezas de atención |
| `depth` | 1 | 2 | 10 | Bloques Transformer por etapa |
| `mlp_ratio` | 4.0 | 4.0 | 4.0 | Expansión del MLP |
| `drop_path_rate` | 0.0 | 0.0 | 0.1 | Stochastic depth (solo etapa 3) |
| `qkv_bias` | True | True | True | Bias en proyecciones Q, K, V |
| `cls_token` | False | False | True | CLS token solo en etapa 3 |

| Parámetro global | Valor | Descripción |
|---|---|---|
| `num_channels` | 3 | Canales de entrada (RGB) |
| `img_size` | 224 | Resolución de entrada |
| `pretrained` | False | Pesos aleatorios (requisito del proyecto) |
| `d_model` (salida) | 384 | Dimensión del CLS token final |

## Rol en el proyecto

CvT-13 ocupa una posición estratégica como **balance intermedio** en el ablation study:

1. **Inductive bias espacial + atención global.** CvT combina lo mejor de las CNNs (relaciones de vecindad, invarianza a traslación) con la capacidad de modelar dependencias globales de los Transformers. Para imágenes médicas, donde las patologías frecuentemente se manifiestan como patrones locales (nódulos, lesiones) dentro de un contexto global (anatomía), esta combinación tiene justificación clínica.

2. **Balance dimensionalidad-información.** Con d_model=384, produce embeddings que son el doble de ricos que ViT-Tiny (192) pero la mitad que Swin-Tiny (768). Esto permite evaluar si la dimensionalidad adicional aporta información relevante para el router o simplemente introduce ruido.

3. **Eficiencia computacional.** La reducción jerárquica de tokens (3136 → 784 → 196) reduce significativamente el costo de atención comparado con ViT, que mantiene 197 tokens constantes. Esto hace que CvT-13 sea más rápido que ViT-Tiny en la práctica a pesar de tener más parámetros.

4. **Referencia de arquitectura híbrida.** En el ablation study, CvT-13 representa la familia de modelos "convolution meets attention". Si supera a ViT-Tiny puro y a DenseNet-121 puro, confirma la hipótesis de que el inductive bias convolucional complementa la auto-atención para imágenes médicas.

## Checkpoint

**Estructura del directorio:**

```
checkpoints/backbone_02_cvt13/
├── model_card.md              ← este archivo
└── backbone_cvt13.pt          ← pesos del modelo (se genera al entrenar)
```

**Carga del modelo con código Python:**

```python
import torch
# El interceptor se activa al importar backbone_cvt13
import backbone_cvt13
from backbone_loader import load_frozen_backbone

# Opción 1: A través del pipeline (recomendado)
model, d_model = load_frozen_backbone("cvt_13", device="cuda")
# model: CvT13Wrapper congelado, eval(), en device
# d_model: 384 (verificado empíricamente)

# Opción 2: Construcción directa
from backbone_cvt13 import build_cvt13
model = build_cvt13(pretrained=False, device="cuda")
# Nota: pretrained se ignora siempre — pesos aleatorios (requisito del proyecto)

# Cargar pesos entrenados (si existen)
ckpt_path = "checkpoints/backbone_02_cvt13/backbone_cvt13.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# Extraer CLS token
img = torch.randn(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    z = model(img)  # z.shape = [1, 384]
```

**Mecanismo de interceptor (detalle técnico):**

CvT-13 no está disponible directamente en `timm`. El módulo `backbone_cvt13.py` parchea `timm.create_model` para interceptar el nombre `"cvt_13"` y construir el modelo via `transformers.CvtModel` (HuggingFace). El interceptor se activa automáticamente al importar el módulo (que `fase1_pipeline.py` importa explícitamente con `import backbone_cvt13`).

## Notas de entrenamiento

| Parámetro | Recomendación |
|---|---|
| **Batch size** | 32-64 (GPU con ≥8 GB VRAM) — reducir a 8-16 en CPU |
| **Learning rate** | 1e-4 a 5e-4 (AdamW) |
| **Optimizador** | AdamW con weight_decay=0.05 |
| **Scheduler** | Cosine annealing con warmup lineal (5-10 épocas) |
| **Criterio de parada** | Early stopping en val loss (paciencia 10 épocas) |
| **Épocas máximas** | 100 |
| **Normalización** | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **VRAM pico** | ~3 GB (batch_size=32, img 224×224) |
| **Dependencia extra** | `transformers` (HuggingFace) — para `CvtConfig`, `CvtModel` |

**Consideraciones:**

- CvT-13 se construye via HuggingFace `transformers`, no via `timm`. El wrapper `CvT13Wrapper` adapta la interfaz para que sea compatible con el pipeline. La configuración de la arquitectura se define localmente en `backbone_cvt13.py` — no se descarga nada de internet.
- El argumento `pretrained=True` en `build_cvt13()` se **ignora deliberadamente**. Se registra un warning en el log pero el modelo siempre se inicializa con pesos aleatorios.
- El CLS token se extrae de `outputs.cls_token_value`. Si es `None` (versiones antiguas de transformers), el fallback es `outputs.last_hidden_state[:, 0, :]`.
- Tras la construcción, se ejecuta un forward dummy para verificar que la salida tenga la dimensión esperada (384). Si no coincide, se lanza un `AssertionError`.
- El congelamiento se verifica rigurosamente: si queda algún parámetro con `requires_grad=True`, se lanza un `RuntimeError` para evitar embeddings no reproducibles.
