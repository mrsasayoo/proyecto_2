# Experto 0 — ConvNeXt-Tiny — Model Card

## Resumen

| Campo | Valor |
|---|---|
| **Experto ID** | 0 (`EXPERT_IDS["chest"]`) |
| **Arquitectura** | ConvNeXt-Tiny (desde cero, sin pesos preentrenados) |
| **Dataset** | NIH ChestXray14 |
| **Modalidad** | Radiografía de tórax 2D (PNG, 224x224 px) |
| **Tarea** | Clasificación **multi-label** — 14 patologías simultáneas |
| **Num. clases** | 14 (salida: 14 logits independientes, cada uno con sigmoid) |
| **Loss** | `BCEWithLogitsLoss` con `pos_weight` por clase. Alternativa: `FocalLossMultiLabel(gamma=2)` |
| **Metricas** | AUC-ROC por clase + F1 Macro + AUPRC. **NUNCA Accuracy** |
| **Volumen** | ~112K imagenes (train: 88,999 / val: 11,349 / test: 11,772) |
| **Checkpoint dir** | `checkpoints/expert_00_convnext_tiny/` |
| **Pesos** | `weights_exp0.pt` |

---

## Teoria del modelo

### ConvNeXt: modernizacion de la CNN clasica

ConvNeXt (Liu et al., CVPR 2022) es una red convolucional pura que incorpora las decisiones de diseno que hicieron exitosos a los Vision Transformers, pero las implementa con operaciones convolucionales clasicas. El resultado es una CNN que compite con Swin Transformer en ImageNet sin usar mecanismos de atencion.

### Bloques fundamentales

**Depthwise separable convolutions (7x7):** Cada canal se convoluciona de forma independiente con un filtro 7x7 espacial (depthwise), seguido de una convolucion 1x1 pointwise que mezcla canales. Esto reduce parametros drasticamente respecto a convoluciones densas 3x3 clasicas: un filtro depthwise 7x7 con C canales tiene 49C parametros vs 9C^2 de un conv denso 3x3.

**LayerNorm (no BatchNorm):** ConvNeXt reemplaza BatchNorm por LayerNorm, normalizando a traves de las dimensiones del canal en cada posicion espacial. Esto elimina la dependencia del tamano de batch durante el entrenamiento, lo cual es especialmente util con batches pequenos en GPU de VRAM limitada.

**GELU activation:** La funcion de activacion Gaussian Error Linear Unit reemplaza a ReLU. GELU aplica una compuerta probabilistica suave: `x * Phi(x)`, donde Phi es la CDF de la normal estandar. Produce gradientes mas suaves que ReLU y evita el problema de "dying neurons".

**Inverted bottleneck:** Siguiendo la estructura de los bloques MBConv de EfficientNet, cada bloque ConvNeXt expande la dimension de canales 4x con la capa pointwise, aplica la depthwise convolution en la dimension expandida, y luego comprime de vuelta.

### Por que ConvNeXt-Tiny para ChestXray14

1. **Multi-label con 14 salidas independientes:** Las radiografias de torax frecuentemente muestran multiples patologias simultaneas. ConvNeXt produce un mapa de features denso que se reduce por GAP, preservando informacion de todas las regiones del torax sin la perdida de detalle local que sufren los ViT con patch sizes grandes.

2. **Receptive field grande (7x7 depthwise):** Las patologias toracicas como Cardiomegaly o Effusion ocupan regiones extensas de la imagen. El filtro 7x7 captura contexto espacial amplio desde las primeras capas sin necesidad de apilar muchos bloques.

3. **Eficiencia computacional:** ConvNeXt-Tiny tiene ~28M parametros, factible para entrenamiento desde cero en ~112K imagenes sin overfitting severo (con regularizacion adecuada).

---

## Diagrama de arquitectura (ASCII)

```
INPUT: Radiografia de torax
[B, 3, 224, 224]
       |
       v
+-------------------------------+
|  STEM (Patchify)              |
|  Conv2d(3, 96, k=4, s=4)     |
|  LayerNorm(96)                |
+-------------------------------+
       |
       v
[B, 96, 56, 56]
       |
       v
+-------------------------------+
|  STAGE 1 — 3 bloques          |
|  DepthwiseConv(96, k=7, p=3)  |
|  LayerNorm → 1x1 Conv(96→384) |
|  GELU → 1x1 Conv(384→96)     |
+-------------------------------+
       |
       v
[B, 96, 56, 56]
       |
       v
+-------------------------------+
|  DOWNSAMPLE 1→2               |
|  LayerNorm → Conv(96→192,k=2) |
+-------------------------------+
       |
       v
[B, 192, 28, 28]
       |
       v
+-------------------------------+
|  STAGE 2 — 3 bloques          |
|  DepthwiseConv(192, k=7, p=3) |
|  LayerNorm → 1x1 Conv→GELU   |
|  → 1x1 Conv(768→192)         |
+-------------------------------+
       |
       v
[B, 192, 28, 28]
       |
       v
+-------------------------------+
|  DOWNSAMPLE 2→3               |
|  LayerNorm → Conv(192→384,k=2)|
+-------------------------------+
       |
       v
[B, 384, 14, 14]
       |
       v
+-------------------------------+
|  STAGE 3 — 9 bloques          |
|  DepthwiseConv(384, k=7, p=3) |  <-- stage mas profundo
|  LayerNorm → 1x1 Conv→GELU   |
|  → 1x1 Conv(1536→384)        |
+-------------------------------+
       |
       v
[B, 384, 14, 14]
       |
       v
+-------------------------------+
|  DOWNSAMPLE 3→4               |
|  LayerNorm → Conv(384→768,k=2)|
+-------------------------------+
       |
       v
[B, 768, 7, 7]
       |
       v
+-------------------------------+
|  STAGE 4 — 3 bloques          |
|  DepthwiseConv(768, k=7, p=3) |
|  LayerNorm → 1x1 Conv→GELU   |
|  → 1x1 Conv(3072→768)        |
+-------------------------------+
       |
       v
[B, 768, 7, 7]
       |
       v
+-------------------------------+
|  CLASSIFIER HEAD              |
|  Global Average Pool          |
|  → [B, 768]                  |
|  LayerNorm(768)               |
|  Linear(768, 14)              |
+-------------------------------+
       |
       v
[B, 14]  ← 14 logits independientes
       |
       v
sigmoid(logit_i) → P(patologia_i)
```

### Detalle de un bloque ConvNeXt

```
Input: [B, C, H, W]
       |
       v
  DepthwiseConv2d(C, C, k=7, pad=3, groups=C)   ← cada canal por separado
       |
       v
  Permute → LayerNorm(C) → Permute
       |
       v
  Linear(C, 4C)         ← expansion 4x (inverted bottleneck)
       |
       v
  GELU()
       |
       v
  Linear(4C, C)         ← compresion
       |
       v
  * Layer Scale (gamma)  ← escalar aprendido por canal, init ~1e-6
       |
       v
  + Residual Connection
       |
       v
Output: [B, C, H, W]
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
                           argmax(g) == 0?
                          /                \
                        SI                  NO
                        |                    |
                        v                    v
              +-------------------+    (otro experto)
              | EXPERTO 0         |
              | ConvNeXt-Tiny     |
              | (congelado)       |
              |                   |
              | Input: imagen     |
              | original          |
              | [1, 3, 224, 224]  |
              |                   |
              | Output:           |
              | 14 logits         |
              | → sigmoid → P(i) |
              +-------------------+
                        |
                        v
              Diagnostico multi-label:
              {Atelectasis: 0.82,
               Effusion: 0.71,
               Cardiomegaly: 0.15, ...}

              Umbral t (por clase) → etiquetas binarias
```

**Flujo completo en inferencia:**
1. La imagen entra al backbone congelado, que produce el embedding `z`.
2. El router calcula `g = softmax(W*z + b)` y selecciona `argmax(g)`.
3. Si `argmax(g) == 0`, la imagen se envia al Experto 0 (ConvNeXt-Tiny congelado).
4. El experto recibe la imagen original (no el embedding) y produce 14 logits.
5. Cada logit pasa por sigmoid para obtener la probabilidad de cada patologia.
6. Se aplica un umbral por clase para generar la etiqueta binaria final.

---

## Hiperparametros de entrenamiento

| Parametro | Valor |
|---|---|
| **Batch size** | 32-64 (depende de VRAM disponible) |
| **Learning rate** | 1e-4 a 5e-4 (cosine annealing) |
| **Optimizador** | AdamW (weight_decay=0.05) |
| **Scheduler** | CosineAnnealingLR o CosineAnnealingWarmRestarts |
| **Epocas** | 50-100 |
| **VRAM estimada** | ~6-8 GB (batch=32, FP32) |
| **Precision** | FP32 o AMP (FP16 mixed precision) |
| **Regularizacion** | DropPath (stochastic depth), Layer Scale |
| **Input size** | 224 x 224 |
| **Normalizacion** | ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| **Preprocesado** | CLAHE + TVF pipeline (via `transform_2d.py`), Resize 224x224 |

---

## Clases y metricas

### Patologias (14 clases multi-label)

| Idx | Patologia | Descripcion clinica | BBox | Prevalencia aprox. |
|---|---|---|---|---|
| 0 | Atelectasis | Colapso parcial o total de un lobulo pulmonar | Si | ~11% |
| 1 | Cardiomegaly | Agrandamiento del corazon (indice cardiotoracico >0.5) | Si | ~2.5% |
| 2 | Effusion | Acumulacion de liquido en el espacio pleural | Si | ~12% |
| 3 | Infiltration | Sustancia densa en el parenquima pulmonar (liquido, celulas) | Si | ~18% |
| 4 | Mass | Lesion opaca >3cm (sospecha de neoplasia) | Si | ~5% |
| 5 | Nodule | Lesion opaca <3cm (puede ser benigna o maligna) | Si | ~5.6% |
| 6 | Pneumonia | Infeccion pulmonar con consolidacion visible | Si | ~1.2% |
| 7 | Pneumothorax | Aire en el espacio pleural causando colapso pulmonar | Si | ~4.7% |
| 8 | Consolidation | Region pulmonar donde el aire alveolar es reemplazado por liquido | No | ~4.2% |
| 9 | Edema | Acumulacion de liquido en el tejido intersticial pulmonar | No | ~2.1% |
| 10 | Emphysema | Destruccion de paredes alveolares con hiperinsuflacion | No | ~2.2% |
| 11 | Fibrosis | Cicatrizacion del tejido pulmonar (patron reticular) | No | ~1.5% |
| 12 | Pleural_Thickening | Engrosamiento de la pleura (secuela de derrame o infeccion) | No | ~3% |
| 13 | Hernia | Hernia diafragmatica o hiatal visible en radiografia | No | ~0.2% |

### Metricas objetivo

| Metrica | Descripcion | Benchmark |
|---|---|---|
| **AUC-ROC por clase** | Area bajo la curva ROC, evaluada independientemente por patologia | DenseNet-121 (CheXNet) ~ 0.841 macro |
| **AUC-ROC macro** | Promedio de AUC-ROC de las 14 clases | Benchmark ~ 0.81 |
| **F1 Macro** | Promedio del F1-score por clase (requiere umbral optimizado) | Variable |
| **AUPRC** | Area bajo la curva Precision-Recall (robusta al desbalance) | Mas informativa que AUC-ROC para clases raras |

---

## Notas clinicas importantes

### Etiquetas NLP y ruido inherente
Las etiquetas de ChestXray14 fueron generadas automaticamente por procesamiento de lenguaje natural (NLP) sobre los reportes radiologicos. La precision estimada del sistema de etiquetado es >90%, pero introduce ruido sistematico. Esto significa que:
- El techo de rendimiento esta limitado por la calidad de las etiquetas.
- Un AUC > 0.85 debe investigarse por posible confounding (correlaciones espurias con artefactos de imagen, no con la patologia real).

### Clase "No Finding" y la trampa del Accuracy
Aproximadamente el 54% de las imagenes estan etiquetadas como "No Finding" (vector todo-ceros). Un modelo trivial que prediga siempre "No Finding" obtiene ~54% de Accuracy. Por esta razon, **nunca se usa Accuracy** como metrica para este experto. El codigo en `chest.py` emite un warning explicito al respecto (H6).

### Pos_weight para desbalance
El `pos_weight` de `BCEWithLogitsLoss` se calcula automaticamente como `n_neg / n_pos` por clase. Hernia (~0.2% prevalencia) tendra un peso ~500x mayor que Infiltration (~18%). Esto es critico para que la loss no ignore las clases raras.

### Vistas PA vs AP
Las radiografias anteroposteriores (AP, pacientes encamados) muestran el corazon aparentemente mas grande y mayor distorsion geometrica que las posteroanterior (PA, pacientes de pie). El filtro `--chest_view_filter PA` permite entrenar solo con vistas PA para estudios controlados.

### BBox para validacion
8 de las 14 patologias tienen anotaciones de bounding box (~1000 imagenes). Estas no se usan para entrenamiento, sino para validar heatmaps y verificar que el modelo atiende regiones clinicamente correctas.

### Split por Patient ID
El split se realiza exclusivamente con las listas oficiales `train_val_list.txt` y `test_list.txt`, que separan por Patient ID. Esto garantiza que imagenes de seguimiento (follow-up) del mismo paciente no aparezcan en train y test simultaneamente.

---

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_00_convnext_tiny/
├── model_card.md          ← este archivo
└── weights_exp0.pt        ← pesos entrenados del modelo
```

### Codigo para cargar el modelo

```python
import torch
from torchvision.models import convnext_tiny

# 1. Construir la arquitectura desde cero (sin pesos preentrenados)
model = convnext_tiny(weights=None, num_classes=14)

# 2. Cargar los pesos entrenados del Experto 0
ckpt_path = "checkpoints/expert_00_convnext_tiny/weights_exp0.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)

# 3. Congelar para inferencia en el pipeline MoE
model.eval()
for param in model.parameters():
    param.requires_grad = False

# 4. Inferencia multi-label
with torch.no_grad():
    logits = model(image_tensor)         # [B, 14]
    probs = torch.sigmoid(logits)        # [B, 14] — probabilidad por patologia
    preds = (probs > threshold).int()    # [B, 14] — etiquetas binarias
```

### Verificacion de integridad

```python
# Verificar que el state_dict tiene las keys esperadas
assert "classifier.2.weight" in state_dict, "Falta el head del clasificador"
assert state_dict["classifier.2.weight"].shape == (14, 768), (
    f"Dimension inesperada: {state_dict['classifier.2.weight'].shape}"
)
```
