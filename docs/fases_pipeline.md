# Fases del Pipeline — Sistema MoE para Imágenes Médicas

> Documento consolidado generado el 2026-04-12.
> Fuentes: 46+ archivos `.md` del proyecto + 7 scripts Python de pipeline.
> **Regla de verdad:** cuando la documentación contradice al código, el código prevalece.

---

## Visión General

El sistema es un **Mixture of Experts (MoE)** que clasifica imágenes médicas de 5 dominios clínicos. La arquitectura se compone de:

- **1 backbone compartido** (el ganador de un estudio de ablación entre 4 candidatos) que transforma cada imagen en un vector de embeddings.
- **1 router** (el ganador de un estudio de ablación entre 4 tipos) que decide a cuál experto enviar cada imagen.
- **5 expertos de dominio** (cada uno especializado en un dataset/patología).
- **1 experto CAE** (autoencoder convolucional para detección OOD, activado por entropía alta).
- **6 expertos en total** (IDs 0–5).

### Datasets

| ID | Dataset | Modalidad | Tarea | Volumen | Clases |
|---|---|---|---|---|---|
| 0 | NIH ChestXray14 | Radiografía tórax 2D | Multi-label | ~112K imgs | 14 patologías |
| 1 | ISIC 2019 | Dermatoscopía 2D | Multiclase | ~25K imgs | 8 + 1 UNK = 9 |
| 2 | OA-Knee (OAI) | Radiografía rodilla 2D | Multiclase (5 KL) | ~4.7K imgs | 5 grados KL (KL0–KL4) |
| 3 | LUNA16 / LIDC-IDRI | CT pulmonar 3D (parches 64³) | Binaria | ~17.8K parches | nódulo sí/no |
| 4 | PANORAMA (Zenodo) | CT abdominal 3D (vol. NIfTI) | Binaria | 557 CTs disponibles (de 1,864 labels) | PDAC+/PDAC− |

### Backbones candidatos (Fase 1)

| # | Nombre | d_model | Params | Biblioteca |
|---|---|---|---|---|
| 1 | ViT-Tiny (`vit_tiny_patch16_224`) | 192 | ~5.7M | timm |
| 2 | CvT-13 | 384 | ~20M | HuggingFace CvtModel |
| 3 | Swin-Tiny (`swin_tiny_patch4_window7_224`) | 768 | ~28M | timm |
| 4 | DenseNet-121 Custom | 1024 | ~7.0M | Implementación propia `torch.nn` |

### Routers candidatos (Fase 2)

| Tipo | Descripción |
|---|---|
| Linear + Softmax | Red lineal con 50 épocas, lr=1e-3, batch=512, L_aux α=0.01 |
| GMM | Gaussian Mixture Model con covarianza full, 200 iteraciones |
| Naive Bayes | MLE analítico (sin iteración) |
| kNN-FAISS | IndexFlatIP, k=5, Laplace ε=0.01 |

### Expertos de dominio

| ID | Dataset | Responsable | Arquitectura | Params | Loss | Entrada |
|---|---|---|---|---|---|---|
| 0 | NIH Chest | Martín | ConvNeXt-Tiny | ~28M | BCEWithLogitsLoss (pos_weight) | [B, 3, 224, 224] |
| 1 | ISIC | Isabella | ConvNeXt-Small | ~50M | FocalLossMultiClass(γ=2.0, class_weights, label_smoothing=0.1) | [B, 3, 224, 224] |
| 2 | OA Knee | Isabella | EfficientNet-B3 (pretrained) | ~10.7M | CrossEntropyLoss(weight=class_weights) | [B, 3, 224, 224] |
| 3 | LUNA16 | Nicolás | DenseNet 3D | ~7M | FocalLoss(α=0.85, γ=2) | [B, 1, 64, 64, 64] |
| 4 | Pancreas | Luz | ResNet 3D | ~15M | FocalLoss(α=0.75, γ=2) | [B, 1, 64, 64, 64] |
| 5 | CAE (OOD) | — | ConvAutoEncoder 2D | ~206M | MSE + 0.1·L1 | [B, 3, 224, 224] |

> ⚠️ Nota: Los directorios de checkpoints conservan nombres históricos (e.g. `expert_03_vivit_tiny/`) por compatibilidad, aunque las arquitecturas han cambiado.

---

## Fase 0 — Preparación de datos

### Propósito

Descargar, extraer, validar y dividir los 5 datasets médicos. Generar los archivos CSV de splits (train/val/test) y preparar datos 3D. Producir un reporte de estado final.

### Datasets que procesa

Los 5: NIH ChestXray14, ISIC 2019, OA-Knee, LUNA16, PANORAMA (páncreas).

### Pasos internos

1. **Descarga de datos** — Descarga automatizada de los 5 datasets desde sus fuentes originales (NIH: links directos, ISIC: API, OA: credenciales, LUNA16: Zenodo, Pancreas: Zenodo 13715870). Verificación de integridad con checksums MD5/SHA256 cuando están disponibles.
2. **Extracción de archivos** — Descompresión de `.tar.gz`, `.zip`, `.7z` según el formato de cada dataset. Manejo de archivos multi-parte (NIH: 12 `.tar.gz`, LUNA16: 10 subsets `.zip`).
3. **Post-procesado de NIH** — Generación de `nih_labels_14.csv` con las 14 patologías como columnas binarias a partir de `Data_Entry_2017.csv`. Eliminación de la etiqueta "No Finding" como clase explícita.
4. **Etiquetas páncreas** — Parseo de `datasets/panorama_labels/` para generar `pancreas_labels_binary.csv` (1,864 filas: 1,756 PDAC+, 108 PDAC−). Solo 557 CTs disponibles en disco (batch_1 de 1,864 totales).
5. **Generación de splits** — Splits estratificados train/val/test para cada dataset. Estratificación por clase en datasets 2D, por `seriesuid` en LUNA16 (para evitar leakage paciente→parche), por `patient_id` con `GroupKFold(k=5)` en páncreas. Genera los CSVs: `nih_splits.csv`, `isic_splits.csv`, `oa_splits.csv`, `luna16_splits.csv`, `pancreas_splits.csv`, `cae_splits.csv` (162,611 filas, todos los datasets combinados para CAE).
6. **ISIC Preprocesamiento** — Preprocesamiento offline del dataset ISIC 2019: auditoría → eliminación de vello (DullRazor) → resize aspect-ratio-preserving (lado corto=224px) → guardado en caché como `{isic_id}_pp_224.jpg`. Script: `pre_isic.py`.
7. **Preparación datos 3D** — Extracción de parches LUNA16 a `.npy` (64×64×64, centrados en coordenadas de `candidates_V2.csv`, conversión mundo→vóxel). Verificación de matching entre CTs y annotations.
8. **Instalación CvT** — Instalación del paquete HuggingFace Transformers para CvT-13 (no disponible en timm estándar).
9. **Reporte** — Genera `src/pipeline/fase0/fase0_report.md` con estadísticas de cada dataset, conteos por split, y estado de cada paso.

### Entradas requeridas

- Credenciales/URLs de descarga para cada dataset.
- `Data_Entry_2017.csv` (NIH) con las etiquetas originales.
- `candidates_V2.csv` (LUNA16) con coordenadas de candidatos a nódulos.
- Archivos de labels de PANORAMA en `datasets/panorama_labels/`.

### Salidas producidas

| Archivo | Descripción |
|---|---|
| `datasets/nih_chest_xrays/images/*.png` | ~112K radiografías de tórax |
| `datasets/isic_2019/images/*.jpg` | ~25K imágenes dermatoscópicas |
| `datasets/oa_knee/images/*.png` | ~4.7K radiografías de rodilla |
| `datasets/luna_lung_cancer/patches/*.npy` | ~17.8K parches CT 64³ |
| `datasets/zenodo_13715870/*.nii.gz` | 557 volúmenes CT abdominales |
| `datasets/nih_splits.csv` | Splits NIH (train: 88,999 / val: 11,349 / test: 11,772) |
| `datasets/isic_splits.csv` | Splits ISIC |
| `datasets/oa_splits.csv` | Splits OA Knee |
| `datasets/luna16_splits.csv` | Splits LUNA16 (train: 14,728 / val: 1,143 / test: 1,914) |
| `datasets/pancreas_splits.csv` | Splits páncreas (GroupKFold k=5) |
| `datasets/cae_splits.csv` | 162,611 filas (5 datasets combinados, train: 130,002 / val: 15,959) |
| `datasets/nih_labels_14.csv` | Labels multi-label 14 patologías |
| `datasets/pancreas_labels_binary.csv` | Labels binarias PDAC+/PDAC− (1,864 filas) |
| `src/pipeline/fase0/fase0_report.md` | Reporte de estado de fase 0 |

### Script de entrada

`src/pipeline/fase0/fase0_pipeline.py` (992 líneas)

### Criterio de completado

- Todos los datasets descargados y extraídos.
- Todos los CSV de splits generados y verificados.
- Parches LUNA16 extraídos a `.npy`.
- Cero casos de leakage entre splits (180 instancias LUNA16 originales movidas a `_LEAKED_DO_NOT_USE/`).
- Reporte generado.

> ⚠️ Nota: Páncreas solo tiene batch_1 (557 de 1,864 CTs). `_build_pairs()` omite silenciosamente los casos sin CT en disco con `log.warning`.

> 🔲 Pendiente: Fase 0 completada. Todos los splits están generados y los bugs fueron corregidos (OOM, leakage, paths).

---

## Fase 1 — Entrenamiento de backbones y extracción de embeddings

### Propósito

Entrenar los 4 backbones candidatos desde cero con una tarea proxy de clasificación de dominio (5 clases: qué dataset originó cada imagen), y luego extraer embeddings (vectores CLS/GAP) congelando los pesos entrenados.

### Datasets que procesa

Los 5 datasets simultáneamente (el backbone recibe imágenes de todos los dominios para la tarea proxy de clasificación de dominio).

### Pasos internos

1. **Paso 4.1 — Entrenamiento de backbones** — Se entrena cada uno de los 4 backbones end-to-end desde cero (`pretrained=False`, requisito del proyecto). La tarea proxy es clasificación de dominio: dado una imagen, predecir de cuál de los 5 datasets proviene (etiqueta: 0=NIH, 1=ISIC, 2=OA, 3=LUNA, 4=Pancreas). Esta tarea fuerza al backbone a aprender representaciones generales que distinguen modalidades médicas.

   | Hiperparámetro | Valor |
   |---|---|
   | Optimizador | AdamW |
   | Learning rate | 3×10⁻⁴ |
   | Weight decay | 0.01 |
   | Épocas | 20 |
   | Warmup | 2 épocas (linear warmup) |
   | Scheduler | CosineAnnealingLR (tras warmup) |
   | Cabeza de clasificación | Linear(d_model, 5) — temporal, descartada después |

   Los datos 3D (LUNA, Pancreas) se convierten a slices 2D centrales y se replican a 3 canales antes de entrar al backbone 2D.

2. **Paso 4.2 — Extracción de embeddings** — Se congela cada backbone entrenado y se pasa cada imagen por forward pass para obtener el token CLS (ViT, CvT) o el vector GAP (Swin, DenseNet). Se generan 3 archivos `.npy` por backbone (train, val, test). La cabeza de clasificación temporal se descarta.

   | Backbone | Dimensión embedding | Archivo salida (por split) |
   |---|---|---|
   | ViT-Tiny | 192 | `embeddings/vit_tiny/{train,val,test}.npy` |
   | CvT-13 | 384 | `embeddings/cvt13/{train,val,test}.npy` |
   | Swin-Tiny | 768 | `embeddings/swin_tiny/{train,val,test}.npy` |
   | DenseNet-121 | 1024 | `embeddings/densenet121/{train,val,test}.npy` |

### Entradas requeridas

- Los 5 datasets preparados por Fase 0 (imágenes y splits CSV).
- GPU con VRAM suficiente (~4 GB mínimo para el backbone más grande, Swin-Tiny).

### Salidas producidas

| Archivo | Descripción |
|---|---|
| `checkpoints/backbone_01_vit_tiny/backbone_vit_tiny.pt` | Pesos ViT-Tiny entrenados |
| `checkpoints/backbone_02_cvt13/backbone_cvt13.pt` | Pesos CvT-13 entrenados |
| `checkpoints/backbone_03_swin_tiny/backbone_swin_tiny.pt` | Pesos Swin-Tiny entrenados |
| `checkpoints/backbone_04_densenet121/backbone_densenet121.pt` | Pesos DenseNet-121 entrenados |
| `embeddings/{backbone}/{split}.npy` | 4 backbones × 3 splits = 12 archivos `.npy` |

### Script de entrada

`src/pipeline/fase1/fase1_pipeline.py` (1,127 líneas), que invoca internamente `fase1_train_pipeline.py` para el Paso 4.1.

### Criterio de completado

- Los 4 backbones entrenados y sus checkpoints guardados.
- Embeddings extraídos: 12 archivos `.npy` (4 backbones × 3 splits) con shapes consistentes.
- Verificación de que los embeddings no están colapsados (varianza > 0 en todas las dimensiones).

> ⚠️ Nota: Auditoría previa encontró que los embeddings de E1 (ISIC) estaban completamente colapsados — los 23,257 embeddings eran idénticos en los 4 backbones. Los embeddings previos (stale, 1,463 MB) fueron eliminados.

> ⚠️ Nota: `proyecto_moe.md` menciona 3 backbones; la implementación real tiene 4 (DenseNet-121 custom fue añadido).

> ⚠️ Nota: `proyecto_moe.md` dice 2 splits de embeddings; la implementación genera 3 (train/val/test).

> 🔲 Pendiente: Backbones NO entrenados aún. Scripts creados y verificados con dry-run (EXIT 0). Embeddings stale eliminados.

---

## Fase 2 — Estudio de ablación de routers y entrenamiento de expertos de dominio

### Propósito

Doble propósito: (a) evaluar los 4 tipos de router sobre los embeddings de cada backbone para seleccionar la combinación ganadora, y (b) entrenar los 5 expertos de dominio (IDs 0–4) desde cero.

### Datasets que procesa

- Router: embeddings generados en Fase 1 (de los 5 datasets).
- Expertos: cada uno usa su dataset específico.

### Pasos internos

**Parte A — Ablación de routers:**

1. **Evaluación de 4 routers × 4 backbones = 16 combinaciones** — Cada router se entrena/ajusta sobre los embeddings del split train y se evalúa en val. Métricas calculadas: `val_accuracy`, `load_balance` (ratio `max(f_i)/min(f_i)` donde f_i es la fracción de muestras asignadas al experto i).

2. **Criterio de selección del ganador:**
   - Filtro 1: `load_balance ≤ 1.30` (descarta combinaciones donde un experto recibe demasiada carga).
   - Filtro 2: Entre las combinaciones que pasan, seleccionar la de mayor `val_accuracy`.

   Configuración por tipo de router:

   | Router | Hiperparámetros clave |
   |---|---|
   | Linear + Softmax | 50 épocas, lr=1e-3, batch=512, L_aux con α=0.01 (penaliza desbalance) |
   | GMM | Covarianza full, 200 iteraciones EM, 5 componentes |
   | Naive Bayes | MLE analítico (fit directo, sin iteración) |
   | kNN-FAISS | IndexFlatIP (producto interno), k=5, Laplace ε=0.01 |

**Parte B — Entrenamiento de 5 expertos de dominio:**

3. **Experto 0 (NIH ChestXray14) — ConvNeXt-Tiny (LP-FT):**

   | Campo | Valor |
   |---|---|
   | Arquitectura | ConvNeXt-Tiny pretrained (`timm: convnext_tiny.in12k_ft_in1k`) + `domain_conv` residual |
   | Params | ~28M |
   | Tarea | Multi-label, 14 patologías simultáneas |
   | Salida | 14 logits → sigmoid independiente por clase |
   | Loss | BCEWithLogitsLoss con `pos_weight` por clase. Alternativa: FocalLossMultiLabel(γ=2) |
   | Métrica principal | AUC-ROC por clase + Macro AUC + F1 Macro + AUPRC. **Nunca accuracy** |
   | Estrategia | **LP-FT** (Linear Probing → Fine-Tuning) |
   | Fase LP | 5 épocas, AdamW lr=1e-3, backbone congelado (solo head + domain_conv) |
   | Fase FT | 30 épocas, AdamW lr=1e-4, todo descongelado, CosineAnnealingLR |
   | Weight decay | 0.05 |
   | Dropout | 0.3 |
   | Batch efectivo | 128 (32 × 4 accumulation), FP16 |
   | Early stopping | patience=10 en `val_macro_auc` (solo fase FT) |
   | Preprocesado offline | cv2 grayscale → CLAHE(2.0, 8×8) → multistage_resize(224) → cache RAM |
   | Augmentación online | Albumentations: HFlip(0.5) + BrightnessContrast(0.1) + RandomGamma + GaussNoise(0.1) |
   | Normalización | `model_mean`/`model_std` resueltos via `timm.data.resolve_data_config()` |
   | Evaluación final | TTA: promedio logits test original + test HorizontalFlip(p=1.0) |

4. **Experto 1 (ISIC 2019) — ConvNeXt-Small:**

   | Campo | Valor |
   |---|---|
   | Arquitectura | ConvNeXt-Small (desde cero, `weights=None`) |
   | Params | ~50M |
   | Tarea | Multiclase, 8 clases + 1 UNK = 9 neuronas de salida |
   | Salida | 9 logits → FocalLossMultiClass(gamma=2.0, weight=class_weights, label_smoothing=0.1) |
   | Clases | MEL, NV, BCC, AK, BKL, DF, VASC, SCC + UNK (solo inferencia) |
   | Métrica principal | BMCA (Balanced Multi-Class Accuracy = macro recall). **Nunca accuracy** (~53% siempre prediciendo NV) |
   | Optimizador | AdamW, lr=3e-4, wd=0.05 |
   | Cabeza | AdaptiveAvgPool → Dropout(0.3) → Linear(768, 9) |
   | Batch efectivo | 128 (32 × 4 accumulation) |
   | FP16 | Sí |
   | Max épocas | 50, patience=10 en `val_loss` |
   | Preprocesado offline | DullRazor (hair removal) → resize lado corto=224px → caché JPEG 95 (`pre_isic.py`) |
   | Augmentación online | RandomCrop(224) → flips → rotation 360° → ColorJitter → RandomGamma(p=0.5) → ShadesOfGray(p=0.5) → CoarseDropout(p=0.5) → ToTensor → Normalize(ImageNet) |
   | Augmentación batch | CutMix (p=0.3) + MixUp (p=0.2) en training loop (mutuamente excluyentes) |
   | Sampler | WeightedRandomSampler (reemplaza shuffle=True) |
   | Checkpoint | `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` (ruta histórica mantenida por compatibilidad) |

5. **Experto 2 (OA Knee) — EfficientNet-B3:**

   | Campo | Valor |
   |---|---|
   | Arquitectura | EfficientNet-B3 pretrained (ImageNet1K) |
   | Params | ~10.7M |
   | Tarea | Multiclase, 5 grados KL (KL0–KL4) |
   | Salida | 5 logits → CrossEntropyLoss(weight=class_weights, inverse-frequency) |
   | Clases | KL0 (Normal), KL1 (Dudoso), KL2 (Leve), KL3 (Moderado), KL4 (Severo) |
   | Métrica principal | F1 Macro. **Nunca accuracy** |
   | Optimizador | Adam diferencial (backbone lr=5e-5, head lr=5e-4), wd=1e-4 |
   | Cabeza FC | Dropout(0.4) → Linear(1536, 5) |
   | Batch efectivo | 64 (32 × 2 accumulation), FP16 |
   | Max épocas | 30, patience=10 en `val_f1_macro` |
   | Scheduler | CosineAnnealingLR(T_max=30, eta_min=1e-6) |
   | Train transforms | Resize(256)→RandomCrop(224)→HFlip(0.5)→Rotation(±15°)→ColorJitter(0.3,0.3)→RandomAutocontrast(0.3)→ToTensor→Normalize(ImageNet) |
   | Val transforms | Resize(224)→ToTensor→Normalize(ImageNet) |

6. **Experto 3 (LUNA16) — DenseNet 3D:**

   | Campo | Valor |
   |---|---|
   | Arquitectura | DenseNet 3D (desde cero, `weights=None`) |
   | Params | ~7M |
   | Entrada | [B, 1, 64, 64, 64] — parche CT centrado en candidato a nódulo |
   | Tarea | Binaria (nódulo sí/no), desbalance ~10.7:1 |
   | Loss | FocalLoss(α=0.85, γ=2.0) + label_smoothing=0.05 |
   | Métrica principal | AUC-ROC. **Nunca accuracy** (~99.8% prediciendo siempre negativo) |
   | Optimizador | AdamW, lr=3e-4, wd=0.03 |
   | Dropout FC | 0.4, Spatial Dropout 3D=0.15 |
   | Batch efectivo | 32 (4 × 8 accumulation) |
   | FP16 | Sí |
   | Max épocas | 100, patience=20 en `val_loss` |
   | Checkpoint | `checkpoints/expert_03_vivit_tiny/expert3_best.pt` (ruta histórica mantenida por compatibilidad) |
   | Preprocesado offline | Carga .mhd/.raw → HU (slope/intercept) → fix FOV pixels (-2000→0) → remuestreo isotrópico 1×1×1 mm³ → segmentación pulmonar → clip HU [-1000, +400] → normalización [0,1] → zero-centering → parches 3D 64³ |
   | Augmentación online (train) | Oversampling positivos (ratio ~1:10) → flips X/Y/Z (p=0.5) → rotaciones 3D (±15° o 90°/180°/270°) → escalado [0.8,1.2] → traslación ±3–5mm → deformación elástica 3D → ruido Gaussiano + ajuste brillo/contraste |

   > ℹ️ Nota: Se usa DenseNet 3D por su eficiencia y buen desempeño en datos volumétricos médicos. El directorio de checkpoint conserva el nombre `expert_03_vivit_tiny/` por compatibilidad histórica.

7. **Experto 4 (Pancreas) — ResNet 3D:**

   | Campo | Valor |
   |---|---|
   | Arquitectura | ResNet 3D (desde cero, `weights=None`) |
   | Params | ~15M |
   | Entrada | [B, 1, 64, 64, 64] — volumen CT abdominal |
   | Tarea | Binaria (PDAC+/PDAC−), desbalance ~16.3:1 |
   | Loss | FocalLoss(α=0.75, γ=2.0) |
   | Métrica principal | AUC-ROC (target > 0.85; baseline nnU-Net ≈ 0.88) |
   | Optimizador | AdamW, lr=5e-5, wd=0.05 |
   | Batch efectivo | 16 (2 × 8 accumulation) — el más restrictivo del proyecto |
   | FP16 | Sí |
   | Gradient checkpointing | Habilitado (obligatorio para VRAM < 20 GB) |
   | Max épocas | 100, patience=15 en `val_loss` |
   | Scheduler | CosineAnnealingWarmRestarts(T_0=10, T_mult=2) |
   | Validación | k-fold CV (k=5, GroupKFold por `patient_id`) |
   | Dataset | PANORAMA Challenge (Zenodo 13715870, 13742336, 11034011, 10999754) |
   | Preprocesado offline | Carga .nii.gz → verificación orientación + alineación máscara → remuestreo isotrópico 0.8×0.8×0.8 mm³ → clip HU [-150,+250] → segmentación páncreas (nnUNet 3d_lowres) → recorte bounding box + margen 20mm → normalización Z-score (percentil 0.5–99.5) → parches 3D (96³ o 192³ vóxeles) |
   | Augmentación online (train) | Oversampling parches PDAC (33% del batch) → flips X/Y/Z (p=0.5) → rotaciones 3D ±30° X/Y, ±180° Z → escalado [0.7,1.4] → deformación elástica 3D (σ=5–8) → ajuste Gamma [0.7,1.5] → ruido Gaussiano → blur axial (p=0.1) |

   > ⚠️ Nota: Rango HU [-150, 250] es diferente de LUNA16 [-1000, 400]. El páncreas es tejido blando (HU ~30–50); incluir aire (-1000 HU) diluiría la señal diagnóstica.

### Entradas requeridas

- Embeddings de Fase 1 (para ablación de routers).
- Los 5 datasets preparados en Fase 0 (para entrenamiento de expertos).

### Salidas producidas

| Archivo | Descripción |
|---|---|
| `logs/ablation_results.json` | Resultados de las 16 combinaciones router×backbone |
| `checkpoints/router_best.pt` | Pesos del router ganador |
| `checkpoints/expert_00_convnext_tiny/weights_exp0.pt` | Pesos Experto 0 (NIH Chest) |
| `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` | Pesos Experto 1 (ISIC) |
| `checkpoints/expert_02_vgg16_bn/expert_oa_best.pt` | Pesos Experto 2 (OA Knee) |
| `checkpoints/expert_03_vivit_tiny/expert3_best.pt` | Pesos Experto 3 (LUNA16, DenseNet 3D) |
| `checkpoints/expert_04_swin3d_tiny/expert4_best.pt` | Pesos Experto 4 (Pancreas) |

### Script de entrada

`src/pipeline/fase2/fase2_pipeline.py` (238 líneas)

### Criterio de completado

- Las 16 combinaciones router×backbone evaluadas.
- Router ganador seleccionado (load_balance ≤ 1.30, luego max val_accuracy).
- Los 5 expertos de dominio entrenados y checkpoints guardados.
- Métricas validadas por experto (AUC-ROC, F1, BMCA según corresponda).

> ⚠️ Nota: `arquitectura_flujo.md` referencia una "Fase 4 — Ablation Study" con script `fase4_ablation.py`, pero el estudio de ablación está implementado en `src/pipeline/fase2/fase2_pipeline.py`. La Fase 4 fue absorbida en la Fase 2 (ver sección Fase 4).

> 🔲 Pendiente: Expertos de dominio NO entrenados aún. Scripts creados y verificados con dry-run (EXIT 0). Bloqueados por finalización de Fase 1 (embeddings necesarios para router ablation).

---

## Fase 3 — Entrenamiento del CAE (Experto 5 / Detección OOD)

### Propósito

Entrenar el Convolutional AutoEncoder 2D (Experto 5, `expert_id=5`) para detección de muestras fuera de distribución (OOD) mediante error de reconstrucción. El CAE no clasifica patologías — se activa solo cuando la entropía del gating vector supera un umbral calibrado.

### Datasets que procesa

Los 5 datasets combinados simultáneamente: el CAE ve imágenes de todos los dominios para aprender qué "se ve normal" en el universo de imágenes médicas del sistema.

### Pasos internos

1. **Construcción del dataset multimodal** — Se usa `cae_splits.csv` (162,611 filas) como manifiesto. Datos 3D se convierten a slices 2D centrales: LUNA patches → slice axial central del `.npy`, volúmenes NIfTI páncreas → HU clip [-100, 400] → slice central. Todos los slices se replican a 3 canales y se redimensionan a 224×224 para uniformidad con imágenes 2D nativas.

2. **Entrenamiento del autoencoder** — Entrenamiento no supervisado (la "etiqueta" es la propia imagen de entrada). El modelo aprende a reconstruir imágenes médicas; las imágenes OOD producirán reconstrucciones pobres con alto error MSE.

   | Campo | Valor |
   |---|---|
   | Modelo | ConvAutoEncoder 2D |
   | Params | ~206M (mayoría en capas lineales del bottleneck: 2 × Linear(200704, 512) ≈ 205.6M) |
   | Encoder | Conv2d(3→32→64→128, stride=2) → Flatten → Linear(200704, 512) |
   | Latent dim | 512 (z ∈ ℝ^512) |
   | Decoder | Linear(512, 200704) → Unflatten → ConvTranspose2d(128→64→32→3) → Sigmoid |
   | Loss | MSE(recon, x) + 0.1·L1(recon, x) |
   | Optimizador | AdamW, lr=1e-3, wd=1e-5 |
   | Scheduler | ReduceLROnPlateau(factor=0.5, patience=5) sobre val_mse |
   | **FP32 obligatorio** | MSE en FP16 sufre inestabilidad numérica con errores pequeños (< 1e-4) |
   | Batch size | 32 |
   | Max épocas | 100, early stopping patience=15 en `val_mse` |

3. **Calibración del umbral de entropía** — Se calcula el percentil 95 de entropía del gating vector sobre el set de validación. Este umbral se persiste en `entropy_threshold.pkl` y se usa en inferencia para decidir cuándo activar el CAE.

### Entradas requeridas

- `datasets/cae_splits.csv` (162,611 filas con rutas a imágenes de los 5 datasets).
- Las imágenes/parches/volúmenes de los 5 datasets.

### Salidas producidas

| Archivo | Descripción |
|---|---|
| `checkpoints/expert_05_cae/cae_best.pt` | Pesos del CAE (mejor val_mse) |
| `entropy_threshold.pkl` | Umbral P95 para activación OOD |

### Script de entrada

`src/pipeline/fase3/train_cae.py` (498 líneas)

### Criterio de completado

- CAE entrenado y checkpoint guardado.
- val_mse converge (no diverge ni oscila).
- Umbral de entropía calibrado y persistido.
- Forward pass + `reconstruction_error()` producen shapes correctas: recon `[B, 3, 224, 224]`, latente `[B, 512]`, error `[B]`.

> ⚠️ Nota: El CAE es una extensión aprobada verbalmente por el profesor. No está definido en `proyecto_moe.md`. Introduce un tercer término `β·L_error` en la función de pérdida total del fine-tuning global (Fase 5) que no está contemplado en la guía original.

> ⚠️ Nota: El CAE NO participa en load balancing. La restricción `max(f_i)/min(f_i) ≤ 1.30` aplica solo a los expertos 0–4.

> 🔲 Pendiente: CAE NO entrenado aún. Script creado y verificado con dry-run (EXIT 0). Dry-run confirmó: recon shape (2, 3, 224, 224), latent shape (2, 512), ~206M params, dataset 130,002 train / 15,959 val.

---

## Fase 4 — Estudio de ablación (ABSORBIDA EN FASE 2)

### Propósito original

Según `arquitectura_flujo.md`, esta fase debía ejecutar el estudio de ablación de routers como script independiente `fase4_ablation.py`.

### Estado actual

**La Fase 4 no existe como directorio ni como script independiente.** El estudio de ablación fue implementado directamente dentro de `src/pipeline/fase2/fase2_pipeline.py` (Parte A de la Fase 2).

El orquestador `run_pipeline.py` ejecuta las fases: **0, 1, 2, 3, 5, 6** — no hay fase 4.

> ⚠️ Nota: `arquitectura_flujo.md` todavía referencia `fase4_ablation.py` como script de esta fase. Este archivo no existe. La funcionalidad equivalente está en `src/pipeline/fase2/fase2_pipeline.py`.

---

## Fase 5 — Fine-tuning global por etapas

### Propósito

Ajustar conjuntamente todo el sistema MoE (backbone + router + expertos) en 3 etapas progresivas de descongelamiento, para que los componentes aprendan a trabajar juntos como un sistema end-to-end sin destruir lo aprendido individualmente.

### Datasets que procesa

Los 5 datasets simultáneamente (el sistema completo procesa imágenes de todos los dominios).

### Pasos internos

1. **Stage 1 — Solo router (backbone + expertos congelados):**

   | Campo | Valor |
   |---|---|
   | Componentes desbloqueados | Solo el router |
   | Congelados | Backbone, todos los expertos (0–5) |
   | LR router | 1e-3 |
   | Épocas | 50 |
   | Propósito | El router aprende a asignar imágenes a expertos usando embeddings fijos |

2. **Stage 2 — Router + cabezas de expertos:**

   | Campo | Valor |
   |---|---|
   | Componentes desbloqueados | Router + cabezas FC de cada experto |
   | Congelados | Backbone, cuerpo de cada experto |
   | LR router | 1e-4 |
   | LR cabezas | 1e-4 |
   | Épocas | 30 |
   | Propósito | Las cabezas de clasificación se afinan para colaborar mejor con el router |

3. **Stage 3 — Todo desbloqueado:**

   | Campo | Valor |
   |---|---|
   | Componentes desbloqueados | Todo (backbone + router + expertos completos) |
   | LR backbone | 1e-6 |
   | LR router | 1e-4 |
   | LR expertos | 1e-6 |
   | Épocas | 7–10 |
   | Propósito | Ajuste fino end-to-end con LR muy bajas para no destruir representaciones aprendidas |

   > ⚠️ Nota: `proyecto_moe.md` especifica LR expertos = 1e-5 en Stage 3. La implementación usa 1e-6 para mayor estabilidad con expertos 3D heterogéneos (DenseNet 3D, ResNet 3D).

**Función de pérdida total:**

```
L_total = L_task + α·L_aux + β·L_error
```

| Término | Descripción |
|---|---|
| `L_task` | Pérdida de la tarea del experto seleccionado (BCE, CE, Focal, etc.) |
| `L_aux` | Penalización de load balancing: fuerza distribución uniforme entre expertos 0–4 |
| `L_error` | Término anti-CAE-perezoso: penaliza si el router delega demasiado al CAE |

**Configuración transversal:**

| Campo | Valor |
|---|---|
| FP16 | Sí, excepto CAE (FP32 obligatorio) |
| Gradient checkpointing | Habilitado para expertos 3D (DenseNet 3D, ResNet 3D) |
| `RESET_OPTIMIZER_BETWEEN_STAGES` | False — el optimizador mantiene su estado (momentum, variance) entre stages |

### Entradas requeridas

- Backbone entrenado ganador (de Fase 1 → ablación Fase 2).
- Router entrenado ganador (de Fase 2).
- Los 5 expertos de dominio entrenados (de Fase 2).
- CAE entrenado (de Fase 3).
- Los 5 datasets originales (imágenes).

### Salidas producidas

| Archivo | Descripción |
|---|---|
| `checkpoints/moe_finetuned_stage1.pt` | Checkpoint tras Stage 1 |
| `checkpoints/moe_finetuned_stage2.pt` | Checkpoint tras Stage 2 |
| `checkpoints/moe_finetuned_final.pt` | Checkpoint final (Stage 3) |
| `logs/finetune_metrics.json` | Métricas por stage (loss, accuracy, load_balance) |

### Script de entrada

`src/pipeline/fase5/fase5_finetune_global.py` (445 líneas)

### Criterio de completado

- Los 3 stages completados sin divergencia de loss.
- Load balance ≤ 1.30 mantenido durante todo el fine-tuning.
- Métricas de cada experto no degradan respecto al entrenamiento individual (Stage 2 vs individual, Stage 3 vs Stage 2).
- Checkpoint final guardado.

> 🔲 Pendiente: Fine-tuning NO ejecutado aún. El script tiene un `sys.exit(1)` que bloquea ejecución sin flag `--dry-run`. Bloqueado por Fases 1, 2 y 3.

---

## Fase 6 — Inferencia, evaluación, verificación y despliegue

### Propósito

Evaluar el sistema MoE completo con datos de test, ejecutar verificaciones automáticas de integridad, y desplegar la interfaz web + dashboard.

### Datasets que procesa

Los splits de test de los 5 datasets.

### Pasos internos

1. **Paso 9 — Inferencia y evaluación** (`fase9_test_real.py`):

   El motor de inferencia (`InferenceEngine`) procesa cada imagen de test con el siguiente flujo:

   ```
   imagen → backbone → embedding → router → gating vector g ∈ ℝ^5
                                              ↓
                                         entropía H(g)
                                              ↓
                                    H(g) > threshold? ──sí──→ Experto 5 (CAE) → OOD score
                                              │no
                                              ↓
                                    argmax(g) → Experto k → predicción de tarea
   ```

   - **Routing por muestra** (no por batch): cada imagen se procesa individualmente para determinar su experto.
   - **Umbral de entropía**: P95 calibrado en Fase 3 (persistido en `entropy_threshold.pkl`). Fallback si no existe: `log(5)/2 ≈ 0.8047`.
   - **Gating vector**: `g ∈ ℝ^5` (5 componentes, una por experto de dominio). El CAE no tiene slot en g.

   > ⚠️ Nota: `proyecto_moe.md` dice `g ∈ ℝ^5`; `arquitectura_documentacion.md` dice `g ∈ ℝ^6` (incluye slot CAE). La implementación usa `g ∈ ℝ^5` — el CAE se activa por entropía, no por argmax.

   Métricas evaluadas por experto:

   | Experto | Métricas |
   |---|---|
   | 0 (NIH Chest) | AUC-ROC por clase (14), F1 Macro, AUPRC |
   | 1 (ISIC) | BMCA, AUC-ROC por clase (9) |
   | 2 (OA Knee) | F1 Macro |
   | 3 (LUNA16) | AUC-ROC, F1, Sensitivity, Specificity |
   | 4 (Pancreas) | AUC-ROC, F1 |
   | 5 (CAE) | OOD AUROC |

   También se genera Grad-CAM para visualización de las regiones de interés del modelo.

2. **Paso 10 — Verificación automática** (`paso10_verificacion.py`):

   34 checks automáticos distribuidos en 9 categorías:

   | Categoría | Verificaciones |
   |---|---|
   | Checkpoints | Existencia de todos los `.pt`, carga sin error, shapes correctas |
   | Embeddings | Shapes, no-NaN, varianza > 0, dimensiones consistentes |
   | Splits | Sin leakage train↔val↔test, conteos correctos, balanceo |
   | Router | Load balance ≤ 1.30, accuracy > baseline, gating probs suman 1.0 |
   | Expertos | Forward pass produce shapes correctas, no-NaN en outputs |
   | CAE | Reconstruction error > 0, FP32 enforced, shapes correctas |
   | Métricas | F1 Macro > 0.65 (expertos 3D), AUC-ROC > 0.70 |
   | Pipeline | run_pipeline.py ejecuta sin error, exit codes correctos |
   | Integridad | CSV/JSON/NPY legibles, paths válidos, timestamps coherentes |

3. **Paso 11 — Página web** (interfaz de usuario):

   Aplicación web para subir imágenes médicas y obtener predicciones del sistema MoE en tiempo real. Muestra: predicción, experto seleccionado, confianza, y Grad-CAM superpuesto.

4. **Paso 12 — Dashboard:**

   Panel de visualización con métricas agregadas: distribución de routing entre expertos, performance por dominio, load balance, distribución de entropía, curvas ROC, matrices de confusión.

### Entradas requeridas

- Checkpoint final del MoE fine-tuned (Fase 5).
- Los splits de test de los 5 datasets.
- `entropy_threshold.pkl` (Fase 3).

### Salidas producidas

| Archivo | Descripción |
|---|---|
| `logs/evaluation_results.json` | Métricas completas de evaluación |
| `logs/verification_report.json` | Resultado de los 34 checks (pass/fail/skip) |
| `outputs/gradcam/*.png` | Visualizaciones Grad-CAM |
| `outputs/dashboard/` | Archivos del dashboard |
| Aplicación web | Interfaz desplegable para inferencia en tiempo real |

### Script de entrada

`src/pipeline/fase6/inference_engine.py` (220 líneas) para inferencia; `fase9_test_real.py` para evaluación; `paso10_verificacion.py` para verificación.

### Criterio de completado

- Todas las métricas por experto calculadas y dentro de umbrales mínimos (F1 Macro > 0.65, AUC-ROC > 0.70).
- 34/34 verificaciones pasadas (o justificación documentada para las que fallan).
- Dashboard funcional con visualizaciones.
- Web app desplegable.

> 🔲 Pendiente: Inferencia, evaluación, verificación y despliegue NO ejecutados aún. Scripts creados y verificados con dry-run (EXIT 0). Bloqueado por Fases 1–5.

---

## Flujo de datos entre fases

```
FASE 0: Datos crudos
  │
  ├── datasets/{nih,isic,oa,luna,pancreas}/
  ├── datasets/*_splits.csv
  └── datasets/cae_splits.csv (162,611 filas)
        │
        ▼
FASE 1: Backbones + Embeddings
  │
  ├── checkpoints/backbone_0{1,2,3,4}_*/backbone_*.pt   (4 checkpoints)
  └── embeddings/{backbone}/{split}.npy                  (12 archivos)
        │
        ▼
FASE 2: Ablación routers + Expertos de dominio
  │
  ├── checkpoints/router_best.pt
  ├── checkpoints/expert_0{0,1,2,3,4}_*/weights.pt       (5 checkpoints)
  └── logs/ablation_results.json
        │
        ▼
FASE 3: CAE (Experto 5 / OOD)
  │
  ├── checkpoints/expert_05_cae/cae_best.pt
  └── entropy_threshold.pkl
        │
        ▼
[FASE 4: No existe — absorbida en Fase 2]
        │
        ▼
FASE 5: Fine-tuning global (3 stages)
  │
  └── checkpoints/moe_finetuned_final.pt
        │
        ▼
FASE 6: Inferencia + Evaluación + Verificación + Despliegue
  │
  ├── logs/evaluation_results.json
  ├── logs/verification_report.json (34 checks)
  ├── outputs/gradcam/*.png
  ├── outputs/dashboard/
  └── Aplicación web
```

---

## Ejecución del Pipeline

### Orquestador

`src/pipeline/run_pipeline.py` (372 líneas)

### Fases ejecutadas (en orden)

0 → 1 → 2 → 3 → 5 → 6

> No hay fase 4 en el orquestador.

### Argumentos CLI

| Argumento | Descripción |
|---|---|
| `--only N` | Ejecutar solo la fase N |
| `--skip N [N ...]` | Saltar las fases indicadas |
| `--from-fase N` | Comenzar desde la fase N (ejecuta N y las siguientes) |
| `--skip-errors` | Continuar la ejecución aunque una fase falle |
| `--dry-run` | Ejecutar todas las fases en modo dry-run (sin entrenamiento real) |

### Timeouts

Cada fase tiene un timeout de **7,200 segundos** (2 horas).

### Reporte

Al finalizar, genera un archivo JSON en `logs/` con:
- Fases ejecutadas y sus exit codes.
- Tiempo de ejecución por fase.
- Errores capturados (si los hay).

### Ejemplo de uso

```bash
# Ejecutar todo el pipeline
PYENV_VERSION=3.12.3 python src/pipeline/run_pipeline.py

# Solo Fase 0
PYENV_VERSION=3.12.3 python src/pipeline/run_pipeline.py --only 0

# Desde Fase 2 en adelante
PYENV_VERSION=3.12.3 python src/pipeline/run_pipeline.py --from-fase 2

# Dry-run completo
PYENV_VERSION=3.12.3 python src/pipeline/run_pipeline.py --dry-run
```

---

## Contradicciones detectadas entre documentación y código

| # | Ubicación doc | Dice | Realidad (código) | Severidad |
|---|---|---|---|---|
| 1 | `arquitectura_flujo.md` | Fase 4 existe como `fase4_ablation.py` | Ablación absorbida en `src/pipeline/fase2/fase2_pipeline.py` | Media |
| 2 | `arquitectura_mermaid.md` | Experto 3 es "ViViT-Tiny" | Implementación real usa DenseNet 3D (cambio definitivo) | Alta |
| 3 | `proyecto_moe.md` | `g ∈ ℝ^5` | `arquitectura_documentacion.md` dice `g ∈ ℝ^6`. Implementación: `g ∈ ℝ^5` + CAE activado por entropía | Media |
| 4 | Directorio | `expert_03_vivit_tiny/` | Contiene DenseNet 3D, no ViViT | Baja (cosmético) |
| 5 | `proyecto_moe.md` | 3 backbones | Implementación tiene 4 (DenseNet-121 custom añadido) | Media |
| 6 | `proyecto_moe.md` | 2 splits de embeddings | Implementación genera 3 (train/val/test) | Baja |
| 7 | `proyecto_moe.md` | Stage 3 LR expertos = 1e-5 | Implementación usa 1e-6 para estabilidad con expertos 3D | Baja |
| 8 | `proyecto_moe.md` | No menciona CAE (Experto 5) | CAE implementado como extensión verbal aprobada | Media |

---

## Estado general del proyecto (2026-04-12)

| Fase | Estado |
|---|---|
| **Fase 0** | ✅ Completada — splits generados, bugs corregidos (leakage, OOM, paths) |
| **Fase 1** | 🔲 Scripts creados y dry-run verificado — backbones NO entrenados |
| **Fase 2** | 🔲 Scripts creados y dry-run verificado — expertos NO entrenados |
| **Fase 3** | 🔲 Script creado y dry-run verificado — CAE NO entrenado |
| **Fase 4** | ❌ No existe (absorbida en Fase 2) |
| **Fase 5** | 🔲 Script creado — bloqueado por sys.exit(1) sin --dry-run |
| **Fase 6** | 🔲 Scripts creados y dry-run verificado — bloqueado por Fases 1–5 |

---

*Documento generado automáticamente. Fuentes: 46+ archivos `.md`, 7 scripts Python del pipeline, cross-referencia código↔documentación.*
