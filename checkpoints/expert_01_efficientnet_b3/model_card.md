# Experto 1 — EfficientNet-B3 — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Experto ID** | 1 (`EXPERT_IDS["isic"]`) |
| **Arquitectura** | EfficientNet-B3 (desde cero, sin pesos preentrenados) |
| **Dataset** | ISIC 2019 (HAM10000 + BCN_20000 + MSK) |
| **Modalidad** | Dermatoscopia 2D (JPG, 224x224 px) |
| **Tarea** | Clasificacion **multiclase** — 8 clases en entrenamiento + slot UNK en inferencia |
| **Num. clases** | 9 (8 clases de lesion + 1 slot UNK para imagenes desconocidas) |
| **Loss** | `CrossEntropyLoss` con `class_weights` (NO `BCEWithLogitsLoss`) |
| **Metricas** | BMCA (Balanced Multi-Class Accuracy, metrica oficial ISIC) + AUC-ROC por clase |
| **Volumen** | ~25K imagenes (train: 20,409 / val: 2,474 / test: 2,448) |
| **Checkpoint dir** | `checkpoints/expert_01_efficientnet_b3/` |
| **Pesos** | `weights_exp1.pt` |

---

## Teoria del modelo

### EfficientNet: escalado compuesto de CNNs

EfficientNet (Tan & Le, ICML 2019) introduce el **compound scaling**, una metodologia que escala simultaneamente tres dimensiones de la red (ancho, profundidad y resolucion de entrada) usando coeficientes fijos derivados de una busqueda de arquitectura neural (NAS). En lugar de escalar solo la profundidad (como ResNet) o solo el ancho (como Wide ResNets), EfficientNet balancea las tres dimensiones con un coeficiente compuesto phi.

### Bloques MBConv (Mobile Inverted Bottleneck)

El bloque fundamental de EfficientNet es el **MBConv** (Mobile Inverted Bottleneck Convolution), heredado de MobileNetV2:

1. **Expansion:** Una convolucion 1x1 expande los canales por un factor de expansion (tipicamente 6x). Esto crea una representacion de alta dimension donde las transformaciones no-lineales preservan mejor la informacion.

2. **Depthwise convolution:** Un filtro espacial 3x3 o 5x5 que opera independientemente sobre cada canal expandido. Captura patrones espaciales con una fraccion de los parametros de una convolucion densa.

3. **Squeeze-and-Excitation (SE):** Un mecanismo de atencion por canal. Reduce el mapa de features a un vector global via GAP, pasa por dos capas FC (reduccion 4x y expansion), y genera pesos por canal con sigmoid. Los canales mas informativos reciben mayor peso. Esto es especialmente util en dermatoscopia donde ciertos canales de color (e.g., azul-blanco en melanoma) son discriminativos.

4. **Projection:** Una convolucion 1x1 comprime de vuelta a la dimension original (bottleneck). No se aplica activacion para preservar la informacion en el espacio comprimido.

5. **Skip connection:** Conexion residual cuando las dimensiones de entrada y salida coinciden.

### Swish activation

EfficientNet usa **Swish** (tambien llamada SiLU): `f(x) = x * sigmoid(x)`. A diferencia de ReLU, Swish es suave, no-monotona y permite gradientes negativos. Esto mejora el entrenamiento en redes profundas al evitar la saturacion de gradientes.

### Coeficientes de B3

EfficientNet-B3 aplica compound scaling con:
- **Width multiplier:** 1.2 (canales ~20% mas anchos que B0)
- **Depth multiplier:** 1.4 (bloques ~40% mas profundos que B0)
- **Resolution:** 300 en la spec original (se usa 224 en este proyecto por compatibilidad con el backbone del pipeline MoE)

### Por que EfficientNet-B3 para ISIC 2019

1. **Squeeze-and-Excitation para color:** En dermatoscopia, las estructuras diagnosticas (redes de pigmento, velo azul-blanco, globulos) se distinguen tanto por textura como por color. SE modula la importancia de cada canal aprendido, lo que es analogico a como un dermatologo atiende a patrones cromaticos especificos.

2. **Escalado balanceado vs dataset mediano:** Con ~25K imagenes, una red demasiado grande (B7) sobreajustaria, y una demasiado pequena (B0) no capturaria la variabilidad de 3 fuentes con bias de dominio. B3 esta en el punto optimo para este volumen.

3. **Eficiencia:** ~12M parametros. Suficiente capacidad para 9 clases con augmentacion agresiva, sin requerir VRAM excesiva.

---

## Diagrama de arquitectura (ASCII)

```
INPUT: Imagen de dermatoscopia
[B, 3, 224, 224]
       |
       v
+-------------------------------------------+
|  STEM                                     |
|  Conv2d(3, 40, k=3, s=2, p=1)            |
|  BatchNorm2d(40) → Swish                 |
+-------------------------------------------+
       |
       v
[B, 40, 112, 112]
       |
       v
+-------------------------------------------+
|  STAGE 1 — MBConv1, k=3, 2 bloques       |
|  expand=1, channels=24                    |
+-------------------------------------------+
       |  [B, 24, 112, 112]
       v
+-------------------------------------------+
|  STAGE 2 — MBConv6, k=3, 3 bloques       |
|  expand=6, channels=32, stride=2          |
+-------------------------------------------+
       |  [B, 32, 56, 56]
       v
+-------------------------------------------+
|  STAGE 3 — MBConv6, k=5, 3 bloques       |
|  expand=6, channels=48, stride=2          |
+-------------------------------------------+
       |  [B, 48, 28, 28]
       v
+-------------------------------------------+
|  STAGE 4 — MBConv6, k=3, 5 bloques       |
|  expand=6, channels=96, stride=2          |
+-------------------------------------------+
       |  [B, 96, 14, 14]
       v
+-------------------------------------------+
|  STAGE 5 — MBConv6, k=5, 5 bloques       |
|  expand=6, channels=136                   |
+-------------------------------------------+
       |  [B, 136, 14, 14]
       v
+-------------------------------------------+
|  STAGE 6 — MBConv6, k=5, 6 bloques       |
|  expand=6, channels=232, stride=2         |
+-------------------------------------------+
       |  [B, 232, 7, 7]
       v
+-------------------------------------------+
|  STAGE 7 — MBConv6, k=3, 2 bloques       |
|  expand=6, channels=384                   |
+-------------------------------------------+
       |  [B, 384, 7, 7]
       v
+-------------------------------------------+
|  HEAD                                     |
|  Conv2d(384, 1536, k=1) → BN → Swish     |
|  Global Average Pooling                   |
|  → [B, 1536]                             |
|  BatchNorm1d(1536)                        |
|  Dropout(p=0.3)                           |
|  Linear(1536, 9)                          |
+-------------------------------------------+
       |
       v
[B, 9]  ← 9 logits (softmax para probabilidades)
       |
       v
softmax → P(clase_i)
argmax → clase predicha (0-7 lesion, 8 UNK)
```

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
                           argmax(g) == 1?
                          /                \
                        SI                  NO
                        |                    |
                        v                    v
              +-------------------+    (otro experto)
              | EXPERTO 1         |
              | EfficientNet-B3   |
              | (congelado)       |
              |                   |
              | Input: imagen     |
              | original          |
              | [1, 3, 224, 224]  |
              |                   |
              | Output:           |
              | 9 logits          |
              | → softmax → P(i) |
              +-------------------+
                        |
                        v
              Diagnostico multiclase:
              argmax(softmax(logits))
                        |
               +--------+--------+
               |                 |
          clase 0-7           clase 8
          (diagnostico)        (UNK)
               |                 |
               v                 v
          MEL / NV / BCC    H(g) alta →
          AK / BKL / DF     redireccion
          VASC / SCC        a Experto 5
                            (OOD handler)
```

**Flujo completo en inferencia:**
1. La imagen entra al backbone congelado, que produce el embedding `z`.
2. El router calcula `g = softmax(W*z + b)` y selecciona `argmax(g)`.
3. Si `argmax(g) == 1`, la imagen se envia al Experto 1 (EfficientNet-B3 congelado).
4. El experto recibe la imagen original y produce 9 logits.
5. Se aplica softmax para obtener probabilidades de cada clase.
6. Si la clase predicha es UNK (slot 8), la imagen tiene alta entropia y puede redirigirse al Experto 5 (OOD handler).

---

## Hiperparametros de entrenamiento

| Parametro | Valor |
|---|---|
| **Batch size** | 32-64 |
| **Learning rate** | 1e-3 a 3e-4 |
| **Optimizador** | AdamW (weight_decay=0.01) |
| **Scheduler** | CosineAnnealingLR con warmup |
| **Epocas** | 50-80 |
| **VRAM estimada** | ~4-6 GB (batch=32, FP32) |
| **Precision** | FP32 o AMP (FP16 mixed precision) |
| **Input size** | 224 x 224 |
| **Normalizacion** | ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| **Augmentacion (mayoria)** | HorizontalFlip, Rotation ±30, ColorJitter moderado |
| **Augmentacion (minoria)** | HFlip+VFlip, Rot 0/90/180/270, ColorJitter agresivo, RandomAffine, RandomErasing |
| **Sampler** | `WeightedRandomSampler` para compensar desbalance NV >> DF/VASC |

---

## Clases y metricas

### Clases de lesion (8 + UNK)

| Idx | Codigo | Nombre completo | Descripcion clinica | Tipo |
|---|---|---|---|---|
| 0 | MEL | Melanoma | Neoplasia maligna de melanocitos. La lesion mas peligrosa; requiere deteccion temprana. Patron ABCDE (asimetria, bordes, color, diametro, evolucion). | Maligno |
| 1 | NV | Nevus melanocitico | Lunar comun benigno. Clase mayoritaria (~50% del dataset). Patron simetrico, color uniforme. | Benigno |
| 2 | BCC | Carcinoma basocelular | Cancer de piel mas comun. Crecimiento lento, raramente metastatiza. Bordes perlados, telangiectasias. | Maligno |
| 3 | AK | Queratosis actinica | Lesion pre-maligna por dano solar. Superficie rugosa, base eritematosa. Puede evolucionar a SCC. | Pre-maligno |
| 4 | BKL | Queratosis seborreica benigna | Lesion benigna comun en adultos mayores. Apariencia "pegada", bordes definidos, patron cerebriforme. | Benigno |
| 5 | DF | Dermatofibroma | Nodulo fibroso benigno. Firme a la palpacion, signo del hoyuelo positivo. | Benigno |
| 6 | VASC | Lesion vascular | Hemangiomas, angioqueratomas. Patron lacunar rojo-azul en dermatoscopia. | Benigno |
| 7 | SCC | Carcinoma escamoso | Cancer de piel de celulas escamosas. Patron queratinizante, areas blancas sin estructura. | Maligno |
| 8 | UNK | Desconocido | Slot reservado para imagenes que no pertenecen a ninguna de las 8 clases. Solo activo en inferencia; en entrenamiento el ground truth nunca tiene UNK=1. | N/A |

### Distribucion de clases (aproximada)

| Clase | Proporcion | Nota |
|---|---|---|
| NV | ~50% | Clase dominante, arrastra BMCA si se ignora el desbalance |
| MEL | ~18% | Clase critica — los falsos negativos son clinicamente peligrosos |
| BCC | ~13% | Segunda clase maligna mas comun |
| BKL | ~10% | Patron variable, confundible con MEL |
| AK | ~4% | Clase minoritaria |
| SCC | ~2.5% | Clase minoritaria |
| DF | ~1.5% | Clase minoritaria |
| VASC | ~1% | Clase minoritaria |

### Metricas objetivo

| Metrica | Descripcion | Notas |
|---|---|---|
| **BMCA** | Balanced Multi-Class Accuracy (metrica oficial ISIC). Promedio del recall por clase, ponderando por igual a todas las clases. | Preferida sobre Accuracy global porque penaliza modelos que ignoran clases raras. |
| **AUC-ROC por clase** | Area bajo la curva ROC para cada clase (one-vs-rest) | Util para evaluar la discriminacion por clase, especialmente MEL y BCC |

---

## Notas clinicas importantes

### Bias de dominio por fuente

ISIC 2019 combina imagenes de 3 instituciones con caracteristicas visuales muy diferentes:

| Fuente | Origen | Caracteristicas | Sesgo |
|---|---|---|---|
| **HAM10000** | Viena | 600x450, recortadas, fondo uniforme | Distribucion de clases no representativa globalmente |
| **BCN_20000** | Barcelona | 1024x1024, campo circular con borde negro | Borde negro como artefacto — se aplica `apply_circular_crop()` |
| **MSK** | Nueva York | Variable, sufijo `_downsampled` | Imagenes redimensionadas, posible perdida de detalle |

**Contramedida implementada:** Augmentacion agresiva (`ColorJitter`, `RandomAffine`, `RandomErasing`) para reducir la dependencia del modelo en artefactos de adquisicion especificos de cada fuente.

### Split por lesion_id y riesgo de leakage

Multiples imagenes pueden corresponder a la misma lesion (diferentes angulos o seguimiento temporal). Si estas imagenes aparecen en train y test, el modelo "memoriza" lesiones en lugar de aprender patrones. La funcion `build_lesion_split()` en `datasets/isic.py` separa por `lesion_id` para evitar este leakage.

**Sin `metadata_csv`:** Si no se provee el archivo de metadatos, el split se realiza de forma aleatoria por imagen (no por lesion), lo que introduce riesgo de leakage entre fuentes. El codigo emite un warning explicito.

### Deduplicacion MD5

ISIC 2018 y 2019 comparten imagenes con IDs diferentes. La funcion `build_lesion_split()` calcula hashes MD5 de cada imagen y elimina duplicados antes del split. Los duplicados conocidos (`ISIC_0067980`, `ISIC_0069013`) se eliminan automaticamente.

### Slot UNK en inferencia

La clase UNK (indice 8) no tiene ground truth en entrenamiento (todas las filas tienen UNK=0 en el CSV). Su proposito es servir como indicador de incertidumbre durante inferencia: si el softmax del modelo asigna alta probabilidad a UNK, significa que la imagen no encaja bien en ninguna de las 8 clases conocidas. Estas imagenes pueden redirigirse al Experto 5 (OOD handler) del sistema MoE.

### Augmentacion diferenciada por clase

Las clases minoritarias (AK, DF, VASC, SCC) reciben augmentacion mas agresiva que las mayoritarias, siguiendo la estrategia de Gessert et al. 2020. Esto incluye RandomVerticalFlip (valido para dermatoscopia, donde no hay orientacion anatomica fija), rotaciones de 90/180/270 grados y RandomErasing (Cutout) para forzar al modelo a usar features distribuidas, no concentradas en un punto.

---

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_01_efficientnet_b3/
├── model_card.md          ← este archivo
└── weights_exp1.pt        ← pesos entrenados del modelo
```

### Codigo para cargar el modelo

```python
import torch
from torchvision.models import efficientnet_b3

# 1. Construir la arquitectura desde cero (sin pesos preentrenados)
model = efficientnet_b3(weights=None, num_classes=9)

# 2. Cargar los pesos entrenados del Experto 1
ckpt_path = "checkpoints/expert_01_efficientnet_b3/weights_exp1.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 3. Congelar para inferencia en el pipeline MoE
model.eval()
for param in model.parameters():
    param.requires_grad = False

# 4. Inferencia multiclase
with torch.no_grad():
    logits = model(image_tensor)             # [B, 9]
    probs = torch.softmax(logits, dim=1)     # [B, 9]
    predicted_class = probs.argmax(dim=1)    # [B]

# 5. Mapear a nombres de clase
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
diagnosis = CLASS_NAMES[predicted_class.item()]
```

### Verificacion de integridad

```python
# Verificar dimensiones del clasificador
assert "classifier.1.weight" in state_dict, "Falta el head del clasificador"
assert state_dict["classifier.1.weight"].shape == (9, 1536), (
    f"Dimension inesperada: {state_dict['classifier.1.weight'].shape}"
)
```
