# DenseNet-121 Custom — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Familia** | DenseNet (Dense Convolutional Network) |
| **Variante** | Custom del proyecto — 121 capas, implementación pura `torch.nn` |
| **d_model** | 1024 (feature vector tras GAP + proyección lineal) |
| **Dense blocks** | 4 (config: [6, 12, 24, 16]) |
| **Growth rate** | 32 |
| **Compresión (θ)** | 0.5 (DenseNet-BC) |
| **Parámetros** | ~7.0 M |
| **VRAM estimada** | ~3 GB |
| **Entrada** | `[B, 3, 224, 224]` — imágenes 2D RGB normalizadas (ImageNet stats) |
| **Salida** | Feature vector `z ∈ ℝ^1024` (GAP de feature map + proyección lineal) |
| **Fase de uso** | Fase 1 — extracción de embeddings |
| **Checkpoint dir** | `checkpoints/backbone_04_densenet121/` |
| **Pesos esperados** | `backbone_densenet121.pt` (generado tras entrenamiento) |
| **Nombre timm** | `densenet121_custom` (interceptado → implementación propia) |
| **Pretrained** | `False` — se entrena desde cero (requisito del proyecto) |
| **Referencia** | Huang et al., "Densely Connected Convolutional Networks", CVPR 2017 |

## Teoría del modelo

DenseNet fue propuesta por Huang et al. (CVPR 2017) y parte de una observación sencilla pero poderosa: si cada capa tiene acceso directo a los feature maps de todas las capas anteriores, el flujo de gradiente mejora, se reduce el número de parámetros y se fomenta la reutilización de features. A diferencia de ResNet, que suma las representaciones (conexión residual), DenseNet las concatena.

**Dense Connections.** En un DenseBlock con L capas, la capa l-ésima recibe como entrada la concatenación de los feature maps de todas las capas previas (0, 1, ..., l-1):

```
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

donde `[...]` denota concatenación en la dimensión de canales y `H_l` es la función compuesta BN→ReLU→Conv. Esto significa que la capa l recibe `k_0 + k × (l-1)` canales de entrada, donde k_0 son los canales iniciales y k es el growth rate.

**Growth Rate (k=32).** Cada capa densa agrega exactamente k=32 feature maps nuevos. Este diseño permite que cada capa sea "estrecha" (pocos filtros propios) pero que el conocimiento acumulado crezca linealmente. Con growth rate bajo, el modelo es eficiente en parámetros porque cada capa contribuye poco volumen propio pero tiene acceso a un contexto rico.

**Bottleneck (BN-ReLU-Conv1×1-BN-ReLU-Conv3×3).** Antes de la convolución 3×3 (que es la que realmente extrae features espaciales), una convolución 1×1 reduce la dimensionalidad a `4k = 128` canales. Esto es el "cuello de botella" que limita el costo computacional del Conv 3×3 sin perder capacidad representativa.

**Transition Layers (θ=0.5).** Entre DenseBlocks, una Transition Layer reduce los canales a la mitad (compresión θ=0.5) via Conv 1×1, seguido de AvgPool 2×2 que reduce la resolución espacial. Esto controla el crecimiento de canales que, de otro modo, se acumularía sin control a lo largo de la red.

**Global Average Pooling + Proyección.** Tras el último DenseBlock, un BN final seguido de GAP colapsa el feature map espacial a un vector 1D. Luego una capa lineal proyecta al `embed_dim` objetivo (1024 por defecto), seguida de LayerNorm para consistencia con los otros backbones (ViT, Swin, CvT).

**No hay CLS token.** A diferencia de los Transformers del proyecto, DenseNet-121 no produce un CLS token por auto-atención. El embedding se obtiene por Global Average Pooling sobre el feature map final, que promedia toda la información espacial en un vector denso. Conceptualmente, el GAP actúa como un "resumen uniforme" de la imagen, mientras que el CLS token de un Transformer es un "resumen atentivo" ponderado por relevancia.

**Inicialización Kaiming/He.** Todos los pesos se inicializan con el método de Kaiming (He et al., 2015): `kaiming_normal_` para Conv2d (fan_out, ReLU), constantes para BatchNorm (weight=1, bias=0), y `kaiming_uniform_` para Linear. Esta inicialización está optimizada para redes profundas con activaciones ReLU.

## Diagrama de arquitectura (ASCII)

```
  Imagen de entrada
  [B, 3, 224, 224]
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│              Stem (reducción 4×)                                  │
│                                                                    │
│  Conv2d(3→64, k=7, stride=2, pad=3, bias=False)                  │
│  BatchNorm2d(64)                                                   │
│  ReLU(inplace=True)                                                │
│  MaxPool2d(k=3, stride=2, pad=1)                                  │
│                                                                    │
│  Salida: [B, 64, 56, 56]                                         │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│         DenseBlock 1 — 6 capas densas                             │
│                                                                    │
│  Capa densa (×6, cada una con bottleneck):                        │
│    BN → ReLU → Conv1×1(C_in→128) → BN → ReLU → Conv3×3(128→32)  │
│    Concatenar salida con todas las entradas previas               │
│                                                                    │
│  Canales: 64 → 64+6×32 = 256                                     │
│  Salida: [B, 256, 56, 56]                                        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Transition 1:  BN → ReLU → Conv1×1(256→128) → AvgPool 2×2      │
│  Salida: [B, 128, 28, 28]                                        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│         DenseBlock 2 — 12 capas densas                            │
│                                                                    │
│  Canales: 128 → 128+12×32 = 512                                  │
│  Salida: [B, 512, 28, 28]                                        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Transition 2:  BN → ReLU → Conv1×1(512→256) → AvgPool 2×2      │
│  Salida: [B, 256, 14, 14]                                        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│         DenseBlock 3 — 24 capas densas                            │
│                                                                    │
│  Canales: 256 → 256+24×32 = 1024                                 │
│  Salida: [B, 1024, 14, 14]                                       │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Transition 3:  BN → ReLU → Conv1×1(1024→512) → AvgPool 2×2     │
│  Salida: [B, 512, 7, 7]                                          │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│         DenseBlock 4 — 16 capas densas                            │
│                                                                    │
│  Canales: 512 → 512+16×32 = 1024                                 │
│  Salida: [B, 1024, 7, 7]                                         │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│   BN final → ReLU → Global Average Pooling → Proyección          │
│                                                                    │
│  BatchNorm2d(1024)                                                 │
│  ReLU(inplace=True)                                                │
│  AdaptiveAvgPool2d((1,1)) → view → [B, 1024]                     │
│  Linear(1024→1024) → LayerNorm(1024)                              │
│                                                                    │
│  Salida: [B, 1024]                                                │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
   z ∈ ℝ^1024
   Feature vector (embedding global)
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
 │   │    DenseNet-121 Custom (ESTE BACKBONE)            │                 │
 │   │                                                    │                │
 │   │  [B, 3, 224, 224] ──→ feature vector z ∈ ℝ^1024  │                 │
 │   │  (congelado, requires_grad=False, eval mode)       │                │
 │   │                                                    │                │
 │   │  Implementación propia (backbone_densenet.py)      │                │
 │   │  Sin timm, sin torchvision — torch.nn puro         │                │
 │   └──────────┬─────────────────────────────────────────┘                │
 │              │                                                          │
 │              ▼                                                          │
 │   Embeddings guardados: Z_{train,val,test}.npy  +  backbone_meta.json   │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
          FASE 2 — Routing + Expertos

   z ∈ ℝ^1024 ──→ Router (Linear/GMM/KNN/NaiveBayes)
                      │
                      ▼
              ┌───────┴─────────────────────────────────┐
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
| `in_channels` | 3 | Canales de entrada (RGB) |
| `embed_dim` (d_model) | 1024 | Dimensión del embedding de salida |
| `growth_rate` (k) | 32 | Feature maps nuevos por capa densa |
| `block_config` | (6, 12, 24, 16) | Capas densas por bloque (= DenseNet-121) |
| `init_features` | 64 | Canales de salida del stem (conv inicial) |
| `compression` (θ) | 0.5 | Factor de compresión en Transition Layers |
| `bn_size` | 4 | Factor bottleneck: canales intermedios = 4 × k = 128 |
| `img_size` | 224 | Resolución de entrada |
| `pretrained` | False | Pesos aleatorios — desde cero |

**Canales a lo largo de la red:**

| Punto en la red | Canales | Resolución |
|---|---|---|
| Entrada | 3 | 224×224 |
| Post-stem | 64 | 56×56 |
| Post-DenseBlock 1 | 256 | 56×56 |
| Post-Transition 1 | 128 | 28×28 |
| Post-DenseBlock 2 | 512 | 28×28 |
| Post-Transition 2 | 256 | 14×14 |
| Post-DenseBlock 3 | 1024 | 14×14 |
| Post-Transition 3 | 512 | 7×7 |
| Post-DenseBlock 4 | 1024 | 7×7 |
| Post-GAP | 1024 | 1×1 |
| Post-proyección | 1024 | — |

## Rol en el proyecto

DenseNet-121 es la **referencia CNN pura** del sistema y tiene un papel particular en el proyecto:

1. **Recomendación del profesor.** DenseNet-121 fue explícitamente recomendado para imágenes médicas 2D. Rajpurkar et al. (CheXNet, 2017) demostraron que DenseNet-121 alcanza AUC macro ~0.81 en NIH ChestXray14, estableciendo el benchmark de referencia para radiografía de tórax. Esta arquitectura sigue siendo competitiva para diagnóstico automatizado.

2. **Baseline sin atención.** Mientras ViT-Tiny, CvT-13 y Swin-Tiny utilizan mecanismos de auto-atención (global, convolucional y local respectivamente), DenseNet-121 es una CNN pura. Esto permite aislar en el ablation study el efecto de la auto-atención: si los Transformers no superan a DenseNet-121 en el routing, el mecanismo de atención no aporta valor adicional para esta tarea.

3. **Implementación propia.** A diferencia de los otros tres backbones que se instancian via `timm` o HuggingFace, DenseNet-121 es una implementación de cero en `torch.nn` (`backbone_densenet.py`). Esto demuestra comprensión profunda de la arquitectura y cumple el requisito académico de construir modelos desde los fundamentos.

4. **Embedding más rico.** Con d_model=1024, produce los embeddings de mayor dimensionalidad del proyecto. La pregunta experimental es si esta alta dimensionalidad, proveniente de una CNN (que captura texturas y bordes pero no dependencias globales), es suficiente para que los routers distingan entre los 5 dominios médicos.

5. **Eficiencia en parámetros.** Las dense connections reutilizan features agresivamente, logrando que DenseNet-121 tenga solo ~7M parámetros pese a tener 121 capas. Comparado con Swin-Tiny (~28M) o CvT-13 (~20M), es significativamente más eficiente en parámetros.

## Checkpoint

**Estructura del directorio:**

```
checkpoints/backbone_04_densenet121/
├── model_card.md              ← este archivo
└── backbone_densenet121.pt    ← pesos del modelo (se genera al entrenar)
```

**Carga del modelo con código Python:**

```python
import torch

# Opción 1: A través del pipeline (recomendado)
import backbone_densenet  # activa el interceptor en timm.create_model
from backbone_loader import load_frozen_backbone

model, d_model = load_frozen_backbone("densenet121_custom", device="cuda")
# model: DenseNet custom congelado, eval(), en device
# d_model: 1024 (verificado empíricamente)

# Opción 2: Construcción directa (sin timm)
from backbone_densenet import build_densenet

model = build_densenet(
    in_channels=3,
    embed_dim=1024,
    growth_rate=32,
    block_config=(6, 12, 24, 16),
)

# Cargar pesos entrenados (si existen)
ckpt_path = "checkpoints/backbone_04_densenet121/backbone_densenet121.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# Congelar para extracción de embeddings
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to("cuda")

# Extraer feature vector
img = torch.randn(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    z = model(img)  # z.shape = [1, 1024]
```

**Mecanismo de interceptor (detalle técnico):**

`backbone_densenet.py` registra un interceptor en `timm.create_model` que captura el nombre `"densenet121_custom"` y redirige la construcción a `build_densenet()`. El interceptor:
- Registra `densenet121_custom` en `BACKBONE_CONFIGS` de `fase1_config.py`
- Parchea `timm.create_model` (verificando que no haya ya un parche de DenseNet)
- Es idempotente: múltiples importaciones son seguras
- Se activa automáticamente al importar el módulo (`fase1_pipeline.py` importa `backbone_densenet`)

## Notas de entrenamiento

| Parámetro | Recomendación |
|---|---|
| **Batch size** | 32-64 (GPU con ≥8 GB VRAM) — reducir a 16 en CPU |
| **Learning rate** | 1e-3 a 3e-3 (SGD con momentum) o 1e-4 a 5e-4 (AdamW) |
| **Optimizador** | SGD(momentum=0.9, weight_decay=1e-4) o AdamW(weight_decay=0.01) |
| **Scheduler** | MultiStepLR con decaimiento 0.1× en épocas 30 y 60 |
| **Criterio de parada** | Early stopping en val loss (paciencia 15 épocas) |
| **Épocas máximas** | 100-120 |
| **Normalización** | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **VRAM pico** | ~3 GB (batch_size=32, img 224×224) |
| **Inicialización** | Kaiming/He (automática en `_initialize_weights()`) |

**Consideraciones:**

- Esta es una **implementación propia** sin dependencias de `torchvision.models` ni pesos preentrenados. Todo el código reside en `backbone_densenet.py` del proyecto.
- La inicialización Kaiming/He se aplica automáticamente al construir el modelo: `kaiming_normal_` para Conv2d, constantes para BatchNorm, `kaiming_uniform_` para Linear.
- La proyección lineal final (`Linear(C_final → 1024)`) seguida de `LayerNorm(1024)` normaliza la salida para consistencia con los backbones Transformer (ViT, CvT, Swin), que también producen embeddings normalizados por LayerNorm.
- DenseNet tiende a consumir más memoria de activaciones que ResNet debido a las concatenaciones. Si la VRAM es limitada, considerar gradient checkpointing durante el entrenamiento end-to-end.
- La variante DenseNet-BC (Bottleneck-Compression) usada aquí es más eficiente que la DenseNet básica: el bottleneck reduce parámetros en las Conv3×3 y la compresión θ=0.5 reduce canales entre bloques.
- Benchmark de referencia: DenseNet-121 con pesos preentrenados logra AUC macro ~0.81 en NIH ChestXray14 (CheXNet). Al entrenar desde cero, el rendimiento esperado será menor pero dependerá del tamaño del dataset de entrenamiento.
- El `embed_dim` es configurable: si se quiere un embedding más compacto (ej: 512 o 256), se puede pasar como argumento a `build_densenet(embed_dim=512)`.
