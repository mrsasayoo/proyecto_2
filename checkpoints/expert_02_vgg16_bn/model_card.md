# Experto 2 — EfficientNet-B0 — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Experto ID** | 2 (`EXPERT_IDS["oa"]`) |
| **Arquitectura** | EfficientNet-B0, preentrenado en ImageNet1K (transfer learning) |
| **Dataset** | Osteoarthritis Initiative (OAI) / Knee X-ray |
| **Modalidad** | Radiografia de rodilla 2D (PNG, 224x224 px) |
| **Tarea** | Clasificacion multiclase — 5 grados de severidad Kellgren-Lawrence (KL 0-4) |
| **Num. clases** | 5 (KL0=Normal, KL1=Dudoso, KL2=Leve, KL3=Moderado, KL4=Severo) |
| **Loss** | `CrossEntropyLoss` con `class_weights` inversos (1/count, normalizados) |
| **Metricas** | F1 Macro (principal, objetivo > 0.72) + Accuracy, F1 por clase |
| **Volumen** | 4,766 imagenes KL-graded (fuente KLGrade/KLGrade/{0,1,2,3,4}/) |
| **Checkpoint dir** | `checkpoints/expert_02_vgg16_bn/` (nombre historico preservado) |
| **Pesos** | `weights_exp2.pt` |

---

## Teoria del modelo

### EfficientNet: escalado compuesto de CNNs

EfficientNet (Tan & Le, ICML 2019) introduce el **compound scaling**, una metodologia que escala simultaneamente tres dimensiones de la red (ancho, profundidad y resolucion de entrada) usando coeficientes fijos derivados de una busqueda de arquitectura neural (NAS). En lugar de escalar solo la profundidad (como ResNet) o solo el ancho (como Wide ResNets), EfficientNet balancea las tres dimensiones con un coeficiente compuesto phi.

### Bloques MBConv (Mobile Inverted Bottleneck)

El bloque fundamental de EfficientNet es el **MBConv** (Mobile Inverted Bottleneck Convolution), heredado de MobileNetV2:

1. **Expansion:** Una convolucion 1x1 expande los canales por un factor de expansion (tipicamente 6x). Esto crea una representacion de alta dimension donde las transformaciones no-lineales preservan mejor la informacion.

2. **Depthwise convolution:** Un filtro espacial 3x3 o 5x5 que opera independientemente sobre cada canal expandido. Captura patrones espaciales con una fraccion de los parametros de una convolucion densa.

3. **Squeeze-and-Excitation (SE):** Un mecanismo de atencion por canal. Reduce el mapa de features a un vector global via GAP, pasa por dos capas FC (reduccion 4x y expansion), y genera pesos por canal con sigmoid. Los canales mas informativos reciben mayor peso. En radiografia de rodilla, esto permite al modelo ponderar canales que detectan estrechamiento articular, osteofitos y esclerosis subcondral.

4. **Projection:** Una convolucion 1x1 comprime de vuelta a la dimension original (bottleneck). No se aplica activacion para preservar la informacion en el espacio comprimido.

5. **Skip connection:** Conexion residual cuando las dimensiones de entrada y salida coinciden.

### Swish activation

EfficientNet usa **Swish** (tambien llamada SiLU): `f(x) = x * sigmoid(x)`. A diferencia de ReLU, Swish es suave, no-monotona y permite gradientes negativos. Esto mejora el entrenamiento en redes profundas al evitar la saturacion de gradientes.

### Coeficientes de B0 (baseline)

EfficientNet-B0 es el modelo base de la familia, con los coeficientes de escalado unitarios:
- **Width multiplier:** 1.0
- **Depth multiplier:** 1.0
- **Resolution:** 224 (compatible nativamente con el pipeline MoE)
- **Parametros:** ~4.01M — ordenes de magnitud menos que VGG16 (~131M)

### Por que EfficientNet-B0 para Osteoarthritis Knee

1. **Transfer learning desde ImageNet1K:** EfficientNet-B0 preentrenado en ImageNet proporciona features de bajo nivel (bordes, texturas) directamente transferibles a radiografias. Esto compensa el tamano pequeno del dataset (~4.7K imagenes) mucho mejor que entrenar desde cero.

2. **Squeeze-and-Excitation para texturas oseas:** Los signos radiograficos de osteoartritis (estrechamiento del espacio articular, osteofitos, esclerosis subcondral) son patrones que combinan textura y contraste local. SE permite al modelo ponderar adaptativamente los canales mas discriminativos para cada grado KL.

3. **5 clases KL sin remapeo:** A diferencia del modelo anterior (VGG16-BN, 3 clases), EfficientNet-B0 clasifica directamente los 5 grados KL (0-4). Esto preserva la granularidad clinica completa de la escala Kellgren-Lawrence y elimina la perdida de informacion del agrupamiento KL1+KL2 y KL3+KL4.

4. **Eficiencia parametrica:** ~4.01M parametros vs ~131M de VGG16-BN. Menor riesgo de overfitting, menor VRAM, entrenamiento mas rapido con mixed precision (FP16).

5. **LR diferencial:** El backbone preentrenado se ajusta con LR baja (5e-5) para no destruir las features de ImageNet, mientras que el head nuevo aprende con LR alta (5e-4). Esto maximiza la transferencia de conocimiento.

---

## Diagrama de arquitectura (ASCII)

```
INPUT: Radiografia de rodilla (sin CLAHE — RandomAutocontrast como sustituto)
[B, 3, 224, 224]
       |
       v
+-------------------------------------------+
|  STEM                                     |
|  Conv2d(3, 32, k=3, s=2, p=1)            |
|  BatchNorm2d(32) → Swish                 |
+-------------------------------------------+
       |
       v
[B, 32, 112, 112]
       |
       v
+-------------------------------------------+
|  STAGE 1 — MBConv1, k=3, 1 bloque        |
|  expand=1, channels=16                    |
+-------------------------------------------+
       |  [B, 16, 112, 112]
       v
+-------------------------------------------+
|  STAGE 2 — MBConv6, k=3, 2 bloques       |
|  expand=6, channels=24, stride=2          |
+-------------------------------------------+
       |  [B, 24, 56, 56]
       v
+-------------------------------------------+
|  STAGE 3 — MBConv6, k=5, 2 bloques       |
|  expand=6, channels=40, stride=2          |
+-------------------------------------------+
       |  [B, 40, 28, 28]
       v
+-------------------------------------------+
|  STAGE 4 — MBConv6, k=3, 3 bloques       |
|  expand=6, channels=80, stride=2          |
+-------------------------------------------+
       |  [B, 80, 14, 14]
       v
+-------------------------------------------+
|  STAGE 5 — MBConv6, k=5, 3 bloques       |
|  expand=6, channels=112                   |
+-------------------------------------------+
       |  [B, 112, 14, 14]
       v
+-------------------------------------------+
|  STAGE 6 — MBConv6, k=5, 4 bloques       |
|  expand=6, channels=192, stride=2         |
+-------------------------------------------+
       |  [B, 192, 7, 7]
       v
+-------------------------------------------+
|  STAGE 7 — MBConv6, k=3, 1 bloque        |
|  expand=6, channels=320                   |
+-------------------------------------------+
       |  [B, 320, 7, 7]
       v
+-------------------------------------------+
|  HEAD                                     |
|  Conv2d(320, 1280, k=1) → BN → Swish     |
|  AdaptiveAvgPool2d(1)                     |
|  → [B, 1280]                             |
|  Dropout(p=0.4)                           |
|  Linear(1280, 5)                          |
+-------------------------------------------+
       |
       v
[B, 5]  ← 5 logits (softmax para probabilidades)
       |
       v
softmax → P(KL_i)
argmax → grado KL predicho (0, 1, 2, 3, 4)
```

**Total: 16 bloques MBConv con Squeeze-and-Excitation.**

### Detalle de un bloque MBConv6 con SE

```
Input: [B, C_in, H, W]
       |
       v
  Conv1x1(C_in, C_in*6) → BN → Swish       ← Expansion 6x
       |
       v
  [B, C_in*6, H, W]
       |
       v
  DepthwiseConv(C_in*6, k=3|5, groups=C_in*6) → BN → Swish
       |
       v
  [B, C_in*6, H', W']                        ← H'=H/stride
       |
       +---> Global Average Pool → [B, C_in*6, 1, 1]
       |           |
       |     FC(C_in*6, C_in*6/4) → Swish    ← SE reduce
       |           |
       |     FC(C_in*6/4, C_in*6) → Sigmoid  ← SE expand
       |           |
       |     [B, C_in*6, 1, 1]               ← channel weights
       |           |
       x-----------*                          ← scale (element-wise multiply)
       |
       v
  Conv1x1(C_in*6, C_out) → BN               ← Projection (sin activacion)
       |
       v
  + Skip Connection (si C_in == C_out y stride == 1)
       |
       v
Output: [B, C_out, H', W']
```

### Conteo de parametros por seccion

```
BACKBONE (features — 16 bloques MBConv + stem + head conv):
  Stem:       Conv2d(3, 32)                  ~  0.9K params
  Stage 1:    MBConv1, 16ch, 1 bloque        ~  1.4K params
  Stage 2:    MBConv6, 24ch, 2 bloques       ~ 11.0K params
  Stage 3:    MBConv6, 40ch, 2 bloques       ~ 28.5K params
  Stage 4:    MBConv6, 80ch, 3 bloques       ~ 121K params
  Stage 5:    MBConv6, 112ch, 3 bloques      ~ 357K params
  Stage 6:    MBConv6, 192ch, 4 bloques      ~1.07M params
  Stage 7:    MBConv6, 320ch, 1 bloque       ~ 878K params
  Head conv:  Conv2d(320, 1280)              ~ 410K params
                                             -----------
  Subtotal backbone:                         ~4.01M params

CLASSIFIER HEAD:
  Linear(1280, 5)                            ~  6.4K params
                                             -----------
  Subtotal head:                             ~  6.4K params

TOTAL:                                       ~4.01M params
```

---

## Diagrama de integracion MoE (ASCII)

```
                     FASE 1 (inferencia del backbone, congelado)
                     ============================================

  Imagen medica       Backbone (e.g. ViT-Tiny)       CLS token
  [B, 3, 224, 224] → [Patch Embed → Transformer] → z ∈ R^{d_model}
                                                        |
                                                        v
                     FASE 2 (router, entrenado sobre embeddings)
                     ============================================

                          Router: softmax(W·z + b)
                          W ∈ R^{6 x d_model}, b ∈ R^6
                                    |
                                    v
                            g ∈ R^6 (probabilidades)
                                    |
                           argmax(g) == 2?
                          /                \
                        SI                  NO
                        |                    |
                        v                    v
              +-------------------+    (otro experto)
              | EXPERTO 2         |
              | EfficientNet-B0   |
              | (congelado)       |
              |                   |
              | Input: imagen     |
              | [1, 3, 224, 224]  |
              |                   |
              | Output:           |
              | 5 logits          |
              | → softmax → P(i) |
              +-------------------+
                        |
                        v
              Diagnostico multiclase:
              argmax(softmax(logits))
                        |
              +---------+---------+---------+---------+
              |         |         |         |         |
          KL 0      KL 1      KL 2      KL 3      KL 4
          Normal    Dudoso    Leve      Moderado  Severo

              Evaluacion:
              F1 Macro(y_true, y_pred)
```

**Flujo completo en inferencia:**
1. La imagen entra al backbone congelado, que produce el embedding `z`.
2. El router calcula `g = softmax(W*z + b)` y selecciona `argmax(g)`.
3. Si `argmax(g) == 2`, la imagen se envia al Experto 2 (EfficientNet-B0 congelado).
4. El experto recibe la imagen (sin CLAHE, normalizada con estadisticas de ImageNet) y produce 5 logits.
5. Se aplica softmax y argmax para obtener el grado KL predicho (0-4).
6. La evaluacion se mide con F1 Macro, que pondera por igual todas las clases independientemente de su frecuencia.

**Preprocesado obligatorio antes de la inferencia:**
```
Imagen original → resize(224x224) → ToTensor → normalize(ImageNet) → modelo
```

---

## Hiperparametros de entrenamiento

| Parametro | Valor |
|---|---|
| **Batch size** | 32 × accum_steps=2 = 64 efectivo |
| **Optimizador** | Adam |
| **Learning rate** | Diferencial: backbone=5e-5 / head=5e-4 |
| **Weight decay** | 1e-4 |
| **Scheduler** | CosineAnnealingLR(T_max=30, eta_min=1e-6) |
| **Epocas** | 30 |
| **Precision** | FP16 mixed precision (GradScaler) |
| **Input size** | 224 x 224 |
| **Normalizacion** | ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| **CLAHE** | NO — sustituido por RandomAutocontrast(p=0.3) en augmentacion de train |
| **Dropout** | 0.4 en el head (antes de Linear) |
| **Checkpoint** | Mejor val_f1_macro (objetivo > 0.72) |
| **VRAM estimada** | ~3-4 GB (batch=32, FP16) |

### Transforms

**Train:**
```
Resize(256, 256)
→ RandomCrop(224)
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(±15°)
→ ColorJitter(brightness=0.3, contrast=0.3)
→ RandomAutocontrast(p=0.3)
→ ToTensor
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Val / Test:**
```
Resize(224, 224)
→ ToTensor
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Nota sobre CLAHE vs RandomAutocontrast

El modelo anterior (VGG16-BN) requeria CLAHE como preprocesado obligatorio antes del resize. El nuevo modelo usa **RandomAutocontrast(p=0.3)** durante el entrenamiento como sustituto estocastico. RandomAutocontrast aplica ecualizacion de histograma con probabilidad 0.3, lo que:
- Actua como augmentacion de contraste, no como preprocesado fijo.
- No requiere parametros (clip_limit, tile_grid) que dependan de la resolucion original.
- En inferencia, **no se aplica** ningun ajuste de contraste — solo resize + normalize.

---

## Clases y metricas

### Escala Kellgren-Lawrence completa (5 clases)

| Grado KL | Clase del modelo | Descripcion clinica | Hallazgos radiograficos |
|---|---|---|---|
| **KL 0** | Cls 0 — Normal | Rodilla sin signos de osteoartritis | Sin estrechamiento articular, sin osteofitos, cartilago con grosor normal |
| **KL 1** | Cls 1 — Dudoso | Cambios dudosos, posible inicio de OA | Osteofitos diminutos de significancia dudosa, sin estrechamiento articular claro |
| **KL 2** | Cls 2 — Leve | Osteoartritis leve definida | Osteofitos definidos, posible estrechamiento del espacio articular |
| **KL 3** | Cls 3 — Moderado | Osteoartritis moderada | Multiples osteofitos, estrechamiento definido del espacio articular, esclerosis subcondral, posible deformidad osea |
| **KL 4** | Cls 4 — Severo | Osteoartritis severa, fase terminal | Osteofitos grandes, estrechamiento articular marcado o completo, esclerosis severa, deformidad osea evidente |

### Distribucion del dataset (KLGrade/KLGrade/)

| Grado KL | Imagenes | Proporcion |
|---|---|---|
| KL 0 (Normal) | 1,315 | 27.6% |
| KL 1 (Dudoso) | 1,266 | 26.6% |
| KL 2 (Leve) | 765 | 16.1% |
| KL 3 (Moderado) | 742 | 15.6% |
| KL 4 (Severo) | 678 | 14.2% |
| **Total** | **4,766** | 100% |

**Nota:** Las clases KL 2-4 estan subrepresentadas. Los class weights inversos (1/count, normalizados) en CrossEntropyLoss compensan este desbalance.

### Metricas objetivo

| Metrica | Descripcion | Notas |
|---|---|---|
| **F1 Macro** | Promedio del F1-score por clase, ponderando por igual a todas las clases. | Metrica principal para checkpoint selection. Objetivo > 0.72. Penaliza modelos que ignoran clases raras (KL3, KL4). |
| **Accuracy** | Proporcion de predicciones correctas. | Metrica complementaria. Puede ser enganosa con desbalance de clases. |
| **F1 por clase** | F1-score individual para cada grado KL. | Permite identificar que grados el modelo confunde mas (tipicamente KL1 vs KL0 y KL2 vs KL3). |

---

## Notas clinicas importantes

### RandomVerticalFlip prohibido

Las radiografias de rodilla tienen orientacion anatomica fija: la tibia siempre esta abajo y el femur arriba. Un flip vertical produce una imagen anatomicamente imposible que el modelo nunca veria en la practica clinica. `RandomHorizontalFlip` es valido porque la anatomia de la rodilla es simetrica izquierda-derecha (una rodilla izquierda flipped se parece a una derecha).

### Sin CLAHE en el nuevo modelo

El modelo anterior (VGG16-BN) usaba CLAHE (Contrast Limited Adaptive Histogram Equalization) como preprocesado obligatorio. El nuevo modelo EfficientNet-B0 **no usa CLAHE**. En su lugar, `RandomAutocontrast(p=0.3)` durante entrenamiento proporciona variabilidad de contraste como augmentacion estocastica. Esto simplifica el pipeline de inferencia y elimina la dependencia de parametros fijos de CLAHE (clip_limit, tile_grid).

### Dificultad de los grados intermedios (KL1 y KL2)

La escala Kellgren-Lawrence tiene alta variabilidad inter-observador en KL1 (cambios dudosos) y la frontera KL2/KL3. Con 5 clases, estas confusiones se manifiestan como:
- **KL0 vs KL1:** La diferencia entre "sin osteofitos" y "osteofitos dudosos" es subjetiva.
- **KL2 vs KL3:** La transicion de "posible estrechamiento" a "estrechamiento definido" depende del observador.

Estas confusiones de frontera son esperadas y consistentes con la variabilidad inter-observador humana (Kappa ~ 0.56-0.71 dependiendo del estudio).

### Sin Patient ID — limitacion del split

Este dataset no incluye metadatos de paciente. No se puede verificar que la misma rodilla no aparezca en train y test (e.g., imagen pre y post-tratamiento). Esta limitacion debe reportarse como una amenaza a la validez interna del modelo.

### Class weights inversos

La funcion de loss usa pesos inversamente proporcionales al conteo de cada clase:

```
weight_i = (1 / count_i) / sum(1 / count_j for all j)
```

Esto penaliza mas los errores en clases poco frecuentes (KL3, KL4) y evita que el modelo sesgue sus predicciones hacia las clases mayoritarias (KL0, KL1).

---

## Dataset

### Fuente

Osteoarthritis Initiative (OAI) / Knee X-ray dataset. Imagenes de radiografia de rodilla en vista anteroposterior (AP), organizadas por grado Kellgren-Lawrence.

### Estructura del directorio de datos

```
datasets/osteoarthritis/
├── KLGrade/
│   └── KLGrade/
│       ├── 0/    ← KL Grade 0 (Normal)      — 1,315 imagenes
│       ├── 1/    ← KL Grade 1 (Dudoso)      — 1,266 imagenes
│       ├── 2/    ← KL Grade 2 (Leve)        —   765 imagenes
│       ├── 3/    ← KL Grade 3 (Moderado)    —   742 imagenes
│       └── 4/    ← KL Grade 4 (Severo)      —   678 imagenes
│                                              -----------
│                                Total:         4,766 imagenes
├── oa_splits/
│   ├── train/    ← splits para entrenamiento
│   ├── val/      ← splits para validacion
│   └── test/     ← splits para evaluacion final
├── discarded/
└── withoutKLGrade/
```

**Nota sobre oa_splits/:** Los splits actuales contienen carpetas {0, 1, 2} (herencia del esquema de 3 clases del modelo anterior). Para el nuevo modelo de 5 clases, los splits se regeneran a partir de KLGrade/KLGrade/{0,1,2,3,4}/.

---

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_02_vgg16_bn/    ← nombre historico preservado
├── model_card.md                  ← este archivo
└── weights_exp2.pt                ← pesos entrenados del modelo
```

**Nota:** El nombre del directorio (`expert_02_vgg16_bn`) es un artefacto historico. El modelo actual es EfficientNet-B0, no VGG16-BN. El nombre se preserva para mantener compatibilidad con el sistema MoE que referencia al Experto 2 por esta ruta.

### Codigo para cargar el modelo

```python
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# 1. Construir la arquitectura con pesos preentrenados (para estructura)
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# 2. Modificar el clasificador para 5 clases KL
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4, inplace=True),
    torch.nn.Linear(1280, 5),
)

# 3. Cargar los pesos entrenados del Experto 2
ckpt_path = "checkpoints/expert_02_vgg16_bn/weights_exp2.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 4. Congelar para inferencia en el pipeline MoE
model.eval()
for param in model.parameters():
    param.requires_grad = False

# 5. Inferencia (sin CLAHE)
from torchvision import transforms
from PIL import Image

img = Image.open("knee_xray.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
img_tensor = transform(img).unsqueeze(0)          # [1, 3, 224, 224]

with torch.no_grad():
    logits = model(img_tensor)                     # [1, 5]
    probs = torch.softmax(logits, dim=1)           # [1, 5]
    predicted_class = probs.argmax(dim=1).item()   # 0, 1, 2, 3, o 4

KL_CLASS_NAMES = [
    "Normal (KL0)",
    "Dudoso (KL1)",
    "Leve (KL2)",
    "Moderado (KL3)",
    "Severo (KL4)",
]
diagnosis = KL_CLASS_NAMES[predicted_class]
```

### Configuracion de entrenamiento (referencia)

```python
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Modelo con transfer learning
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4, inplace=True),
    torch.nn.Linear(1280, 5),
)

# LR diferencial: backbone lento, head rapido
optimizer = torch.optim.Adam([
    {"params": model.features.parameters(), "lr": 5e-5},
    {"params": model.classifier.parameters(), "lr": 5e-4},
], weight_decay=1e-4)

# Scheduler coseno
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-6
)

# Loss con class weights inversos
class_counts = [1315, 1266, 765, 742, 678]  # KL 0-4
inv_freq = [1.0 / c for c in class_counts]
total = sum(inv_freq)
class_weights = torch.tensor([w / total for w in inv_freq])

criterion = torch.nn.CrossEntropyLoss(weight=class_weights.cuda())

# Mixed precision
scaler = torch.cuda.amp.GradScaler()
```

### Verificacion de integridad

```python
# Verificar que el state_dict tiene la estructura de EfficientNet-B0
assert "features.0.0.weight" in state_dict, "Falta el stem conv"
assert state_dict["features.0.0.weight"].shape == (32, 3, 3, 3), (
    f"Dimension inesperada en stem: {state_dict['features.0.0.weight'].shape}"
)

# Verificar clasificador modificado (5 clases, no 1000)
assert "classifier.1.weight" in state_dict, "Falta el head del clasificador"
assert state_dict["classifier.1.weight"].shape == (5, 1280), (
    f"Dimension inesperada en classifier: {state_dict['classifier.1.weight'].shape}"
)
```
