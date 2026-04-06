# Experto 2 — VGG16-BN — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Experto ID** | 2 (`EXPERT_IDS["oa"]`) |
| **Arquitectura** | VGG16 con Batch Normalization (desde cero, sin pesos preentrenados) |
| **Dataset** | Osteoarthritis Knee X-ray |
| **Modalidad** | Radiografia de rodilla 2D (PNG, 224x224 px) |
| **Tarea** | Clasificacion **ordinal** — 3 grados de severidad (remapeados desde KL 0-4) |
| **Num. clases** | 3 (Normal KL0 → Cls0 / Leve KL1+2 → Cls1 / Severo KL3+4 → Cls2) |
| **Loss** | `CrossEntropyLoss` con `class_weights` (pragmatico) o `OrdinalLoss(n_classes=3)` |
| **Metricas** | QWK (Quadratic Weighted Kappa). **NUNCA** Accuracy ni F1 |
| **Volumen** | 4,766 imagenes KL-graded (train: 3,814 / val: 480 / test: 472) |
| **Checkpoint dir** | `checkpoints/expert_02_vgg16_bn/` |
| **Pesos** | `weights_exp2.pt` |

---

## Teoria del modelo

### VGG16: profundidad con filtros homogeneos

VGG16 (Simonyan & Zisserman, ICLR 2015) demostro que apilar muchas capas convolucionales con filtros pequenos (3x3) logra campos receptivos grandes sin los parametros de filtros grandes. Dos convoluciones 3x3 tienen el mismo campo receptivo que una 5x5 (5 pixeles), pero con menos parametros: `2 * (3^2 * C^2) = 18C^2` vs `25C^2`. Tres convoluciones 3x3 equivalen a una 7x7 con `27C^2` vs `49C^2` parametros.

### Batch Normalization (variante BN)

VGG16-BN anade **Batch Normalization** (Ioffe & Szegedy, 2015) despues de cada convolucion y antes de la activacion ReLU. BN normaliza cada mini-batch a media cero y varianza unitaria, luego aplica una transformacion afin aprendida (gamma, beta):

```
BN(x) = gamma * (x - mu_batch) / sqrt(sigma_batch^2 + epsilon) + beta
```

Beneficios para este dataset:
- **Estabiliza el entrenamiento** en un dataset pequeno (~4.7K imagenes) al reducir el internal covariate shift.
- **Actua como regularizador** por el ruido estocastico de las estadisticas del mini-batch, reduciendo overfitting.
- **Permite learning rates mas altas** sin divergencia.

### Estructura en bloques

VGG16-BN organiza sus 13 convoluciones en 5 bloques, con MaxPooling entre ellos para reducir la resolucion espacial progresivamente. El clasificador final reemplaza las 1000 clases de ImageNet por 3 clases ordinales.

### Por que VGG16-BN para Osteoarthritis Knee

1. **Dataset pequeno, modelo sencillo:** Con solo ~4.7K imagenes, una arquitectura mas simple tiene menor riesgo de overfitting que redes mas profundas (ResNet-152) o mas complejas (EfficientNet). VGG16 tiene ~138M parametros, pero el Dropout agresivo (0.5) en el clasificador y BN en cada capa lo regularizan.

2. **Patrones de textura osea:** Los signos radiograficos de osteoartritis (estrechamiento del espacio articular, osteofitos, esclerosis subcondral) son patrones de textura local. Los filtros 3x3 apilados de VGG capturan texturas a multiples escalas sin la complejidad de mecanismos de atencion.

3. **Tarea ordinal con 3 clases:** La salida es simple (3 logits). No se necesita la capacidad de representacion de arquitecturas mas grandes. El cuello de botella es el tamano del dataset, no la capacidad del modelo.

4. **Interpretabilidad:** Las feature maps de VGG son mas facilmente interpretables con Grad-CAM que las de arquitecturas con skip connections o atencion. Esto permite verificar que el modelo atiende al espacio articular y no a artefactos.

---

## Diagrama de arquitectura (ASCII)

```
INPUT: Radiografia de rodilla (CLAHE ya aplicado)
[B, 3, 224, 224]
       |
       v
+--------------------------------------------+
|  BLOQUE 1 — 2 conv                        |
|  Conv2d(3, 64, k=3, p=1) → BN → ReLU     |
|  Conv2d(64, 64, k=3, p=1) → BN → ReLU    |
|  MaxPool2d(k=2, s=2)                       |
+--------------------------------------------+
       |  [B, 64, 112, 112]
       v
+--------------------------------------------+
|  BLOQUE 2 — 2 conv                        |
|  Conv2d(64, 128, k=3, p=1) → BN → ReLU   |
|  Conv2d(128, 128, k=3, p=1) → BN → ReLU  |
|  MaxPool2d(k=2, s=2)                       |
+--------------------------------------------+
       |  [B, 128, 56, 56]
       v
+--------------------------------------------+
|  BLOQUE 3 — 3 conv                        |
|  Conv2d(128, 256, k=3, p=1) → BN → ReLU  |
|  Conv2d(256, 256, k=3, p=1) → BN → ReLU  |
|  Conv2d(256, 256, k=3, p=1) → BN → ReLU  |
|  MaxPool2d(k=2, s=2)                       |
+--------------------------------------------+
       |  [B, 256, 28, 28]
       v
+--------------------------------------------+
|  BLOQUE 4 — 3 conv                        |
|  Conv2d(256, 512, k=3, p=1) → BN → ReLU  |
|  Conv2d(512, 512, k=3, p=1) → BN → ReLU  |
|  Conv2d(512, 512, k=3, p=1) → BN → ReLU  |
|  MaxPool2d(k=2, s=2)                       |
+--------------------------------------------+
       |  [B, 512, 14, 14]
       v
+--------------------------------------------+
|  BLOQUE 5 — 3 conv                        |
|  Conv2d(512, 512, k=3, p=1) → BN → ReLU  |
|  Conv2d(512, 512, k=3, p=1) → BN → ReLU  |
|  Conv2d(512, 512, k=3, p=1) → BN → ReLU  |
|  MaxPool2d(k=2, s=2)                       |
+--------------------------------------------+
       |  [B, 512, 7, 7]
       v
+--------------------------------------------+
|  CLASIFICADOR (modificado)                 |
|  Flatten → [B, 25088]                     |
|  FC(25088, 4096) → BN → ReLU → Drop(0.5) |
|  FC(4096, 4096) → BN → ReLU → Drop(0.5)  |
|  FC(4096, 3)                               |
+--------------------------------------------+
       |
       v
[B, 3]  ← 3 logits ordinales
       |
       +-- Opcion A: softmax → argmax → clase {0, 1, 2}
       |
       +-- Opcion B (OrdinalLoss): sigmoid por umbral
           logit[0] → P(y > 0), logit[1] → P(y > 1)
           clase = sum(sigmoid(logits) > 0.5)
```

### Conteo de parametros por seccion

```
FEATURES (convoluciones):
  Bloque 1:   64 filtros 3x3           ~  37K params
  Bloque 2:  128 filtros 3x3           ~ 222K params
  Bloque 3:  256 filtros 3x3           ~ 885K params
  Bloque 4:  512 filtros 3x3           ~3.5M params
  Bloque 5:  512 filtros 3x3           ~7.1M params
                                       -----------
  Subtotal features:                   ~11.7M params

CLASSIFIER:
  FC1: 25088 x 4096                    ~102M params  ← domina
  FC2: 4096 x 4096                     ~ 16M params
  FC3: 4096 x 3                        ~ 12K params
                                       -----------
  Subtotal classifier:                 ~119M params

TOTAL:                                 ~131M params
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
              | VGG16-BN          |
              | (congelado)       |
              |                   |
              | Input: imagen     |
              | con CLAHE         |
              | [1, 3, 224, 224]  |
              |                   |
              | Output:           |
              | 3 logits          |
              | → softmax → P(i) |
              +-------------------+
                        |
                        v
              Diagnostico ordinal:
              argmax(softmax(logits))
                        |
              +---------+---------+
              |         |         |
          Cls 0     Cls 1     Cls 2
          Normal    Leve      Severo
          (KL 0)    (KL 1-2)  (KL 3-4)

              Evaluacion:
              QWK(y_true, y_pred)
```

**Flujo completo en inferencia:**
1. La imagen entra al backbone congelado, que produce el embedding `z`.
2. El router calcula `g = softmax(W*z + b)` y selecciona `argmax(g)`.
3. Si `argmax(g) == 2`, la imagen se envia al Experto 2 (VGG16-BN congelado).
4. El experto recibe la imagen original (con CLAHE ya aplicado antes del resize) y produce 3 logits.
5. Se aplica softmax y argmax para obtener el grado ordinal predicho.
6. La evaluacion se mide con QWK, que penaliza errores proporcionalmente a su distancia ordinal.

**Preprocesado obligatorio antes de la inferencia:**
```
Imagen original → apply_clahe() → resize(224x224) → normalize(ImageNet) → modelo
```

---

## Hiperparametros de entrenamiento

| Parametro | Valor |
|---|---|
| **Batch size** | 16-32 (dataset pequeno, batches mayores dan estadisticas BN mas estables) |
| **Learning rate** | 1e-3 a 1e-4 |
| **Optimizador** | SGD(momentum=0.9) o AdamW |
| **Scheduler** | StepLR o ReduceLROnPlateau (patience=5) |
| **Epocas** | 50-100 (monitorear QWK en validacion para early stopping) |
| **VRAM estimada** | ~6-8 GB (batch=32, FP32) |
| **Precision** | FP32 (BN requiere estadisticas estables; AMP con precaucion) |
| **Input size** | 224 x 224 |
| **Normalizacion** | ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| **Preprocesado** | CLAHE (clip_limit=2.0, tile_grid=8x8) ANTES del resize |
| **Augmentacion** | RandomHorizontalFlip(0.5), RandomRotation(10), ColorJitter(0.2, 0.15) |
| **Augmentacion prohibida** | RandomVerticalFlip — anatomia simétrica solo horizontalmente |
| **Dropout** | 0.5 en las dos capas FC del clasificador |

---

## Clases y metricas

### Remapeo Kellgren-Lawrence a 3 clases ordinales

| Grado KL | Clase del modelo | Descripcion clinica | Hallazgos radiograficos |
|---|---|---|---|
| **KL 0** | Cls 0 — Normal | Rodilla sin signos de osteoartritis | Sin estrechamiento articular, sin osteofitos, cartilago con grosor normal |
| **KL 1 + KL 2** | Cls 1 — Leve | Osteoartritis leve a moderada. KL1 tiene dudoso estrechamiento articular; KL2 tiene osteofitos definidos con posible estrechamiento | Osteofitos marginales pequenos, estrechamiento leve del espacio articular, esclerosis minima |
| **KL 3 + KL 4** | Cls 2 — Severo | Osteoartritis moderada-severa a terminal. KL3 tiene multiples osteofitos y estrechamiento definido; KL4 tiene contacto hueso-con-hueso | Estrechamiento articular marcado o completo, osteofitos grandes, esclerosis subcondral severa, deformidad osea |

### Matriz de costos ordinales

```
               Prediccion
               Cls0  Cls1  Cls2
Real  Cls0  [  0     1     4  ]
      Cls1  [  1     0     1  ]
      Cls2  [  4     1     0  ]
```

Un error Severo → Normal (costo 4) es clinicamente 4 veces peor que un error de clase adyacente (costo 1). QWK incorpora esta penalizacion cuadratica de forma natural.

### Metrica principal: QWK

| Metrica | Descripcion | Justificacion |
|---|---|---|
| **QWK** | Quadratic Weighted Kappa. Mide acuerdo inter-rater penalizando errores proporcionalmente al cuadrado de su distancia ordinal. | Es la metrica estandar para tareas de grading ordinal en imagenes medicas. Penaliza errores graves (Severo clasificado como Normal) mucho mas que errores de frontera (Leve clasificado como Normal). |
| **Accuracy** | NO usar | Trata todos los errores por igual. Clasificar Severo como Normal (clinicamente peligroso) pesa lo mismo que Leve como Normal. |
| **F1 Macro** | NO usar como metrica principal | Ignora la relacion ordinal entre clases. Util solo como metrica complementaria. |

### Interpretacion de QWK

| Valor QWK | Interpretacion |
|---|---|
| < 0.20 | Acuerdo pobre |
| 0.21 - 0.40 | Acuerdo justo |
| 0.41 - 0.60 | Acuerdo moderado |
| 0.61 - 0.80 | Acuerdo sustancial |
| 0.81 - 1.00 | Acuerdo casi perfecto |

**Objetivo:** QWK >= 0.60 (acuerdo sustancial). La variabilidad inter-observador humana en grading KL es QWK ~ 0.56-0.71, dependiendo del estudio.

---

## Notas clinicas importantes

### RandomVerticalFlip prohibido

Las radiografias de rodilla tienen orientacion anatomica fija: la tibia siempre esta abajo y el femur arriba. Un flip vertical produce una imagen anatomicamente imposible que el modelo nunca veria en la practica clinica. `RandomHorizontalFlip` es valido porque la anatomia de la rodilla es simetrica izquierda-derecha (una rodilla izquierda flipped se parece a una derecha).

### CLAHE siempre ANTES del resize

CLAHE (Contrast Limited Adaptive Histogram Equalization) realza el contraste local de la imagen. Aplicarlo a la resolucion original preserva el detalle fino del espacio articular, que es la region critica para el grading. Si se aplica despues del resize a 224x224, la informacion de textura osea ya se habria perdido por la interpolacion.

```
CORRECTO:  load(img) → apply_clahe(img) → resize(224) → normalize
INCORRECTO: load(img) → resize(224) → apply_clahe(img) → normalize
```

### Contaminacion KL1 en la clase Leve

La escala Kellgren-Lawrence tiene alta variabilidad inter-observador en KL1 (cambios dudosos). Al consolidar KL1 con KL2 en la clase "Leve", se introduce ruido en la frontera Cls0/Cls1. El metodo `evaluate_boundary_confusion()` monitorea esta frontera en cada epoca. Un `boundary_01_error_rate > 0.25` indica que el modelo no distingue Normal de Leve, lo cual es esperado y debe documentarse.

### Sin Patient ID — limitacion del split

Este dataset no incluye metadatos de paciente. Las imagenes estan organizadas en carpetas `oa_splits/{train,val,test}/` con split 80/10/10 predefinido. No se puede verificar que la misma rodilla no aparezca en train y test (e.g., imagen pre y post-tratamiento). Esta limitacion debe reportarse como una amenaza a la validez interna del modelo.

### Augmentacion offline detectada

El dataset ZIP puede contener imagenes con augmentacion ya aplicada (rotaciones, flips almacenados como archivos separados). El constructor `OAKneeDataset` detecta esto comparando el conteo de imagenes contra los valores esperados del dataset base (~8,260 imagenes originales). Si se detecta augmentacion offline, el augmentation en runtime se deshabilita para evitar doble augmentacion.

### Opciones de Loss para tarea ordinal

| Opcion | Loss | Salida del modelo | Ventaja |
|---|---|---|---|
| **A (pragmatica)** | `CrossEntropyLoss(weight=class_weights)` | [B, 3] logits → softmax | Simple, probada, compatible con pipeline estandar |
| **B (ordinal)** | `OrdinalLoss(n_classes=3)` | [B, 2] logits → sigmoid | Preserva relacion ordinal. `P(y>0)` y `P(y>1)` como clasificaciones binarias acumulativas |

La opcion B implementa Chen et al. (2019, PMC doi:10.1016/j.compmedimag.2019.05.007): reformula K clases ordinales como K-1 clasificaciones binarias `P(y > k)`. Para K=3: `P(Leve o Severo)` y `P(Severo)`. La clase predicha es `sum(sigmoid(logits) > 0.5)`.

---

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_02_vgg16_bn/
├── model_card.md          ← este archivo
└── weights_exp2.pt        ← pesos entrenados del modelo
```

### Codigo para cargar el modelo

```python
import torch
from torchvision.models import vgg16_bn

# 1. Construir la arquitectura desde cero (sin pesos preentrenados)
model = vgg16_bn(weights=None)

# 2. Modificar el clasificador para 3 clases ordinales
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 4096),
    torch.nn.BatchNorm1d(4096),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.BatchNorm1d(4096),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 3),
)

# 3. Cargar los pesos entrenados del Experto 2
ckpt_path = "checkpoints/expert_02_vgg16_bn/weights_exp2.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 4. Congelar para inferencia en el pipeline MoE
model.eval()
for param in model.parameters():
    param.requires_grad = False

# 5. Inferencia ordinal (con CLAHE obligatorio)
from preprocessing import apply_clahe
from torchvision import transforms
from PIL import Image

img = Image.open("knee_xray.png").convert("RGB")
img = apply_clahe(img)                           # CLAHE ANTES del resize
img = img.resize((224, 224), Image.BICUBIC)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
img_tensor = transform(img).unsqueeze(0)         # [1, 3, 224, 224]

with torch.no_grad():
    logits = model(img_tensor)                    # [1, 3]
    probs = torch.softmax(logits, dim=1)          # [1, 3]
    predicted_class = probs.argmax(dim=1).item()  # 0, 1, o 2

OA_CLASS_NAMES = ["Normal (KL0)", "Leve (KL1-2)", "Severo (KL3-4)"]
diagnosis = OA_CLASS_NAMES[predicted_class]
```

### Evaluacion con QWK

```python
import numpy as np
from datasets.osteoarthritis import OAKneeDataset

# Despues del loop de validacion
y_true = np.array(all_true_labels)    # e.g., [0, 1, 2, 1, 0, ...]
y_pred = np.array(all_pred_labels)    # e.g., [0, 1, 1, 1, 0, ...]

# Metrica principal
qwk = OAKneeDataset.compute_qwk(y_true, y_pred)
print(f"QWK: {qwk:.4f}")

# Analisis detallado de fronteras
metrics = OAKneeDataset.evaluate_boundary_confusion(y_true, y_pred)
OAKneeDataset.log_boundary_confusion(metrics, epoch=current_epoch)
```

### Verificacion de integridad

```python
# Verificar que el state_dict tiene la estructura esperada
assert "features.0.weight" in state_dict, "Falta la primera conv"
assert state_dict["features.0.weight"].shape == (64, 3, 3, 3), (
    f"Dimension inesperada en features.0: {state_dict['features.0.weight'].shape}"
)

# Verificar clasificador modificado (3 clases, no 1000)
last_key = [k for k in state_dict.keys() if "classifier" in k and "weight" in k][-1]
assert state_dict[last_key].shape[0] == 3, (
    f"El clasificador deberia tener 3 salidas, tiene {state_dict[last_key].shape[0]}"
)
```
