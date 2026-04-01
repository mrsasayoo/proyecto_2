# Transformaciones de Datasets — Proyecto MoE Imágenes Médicas

## Introducción

Este documento detalla **exhaustivamente** todas las transformaciones aplicadas a cada dataset del proyecto Mixture of Experts (MoE) para imágenes médicas, cubriendo las dos fases principales del pipeline de preprocesamiento:

- **Fase 0** — Preparación de datos crudos: descarga, extracción, generación de splits, filtros de calidad, validación de integridad y, para datos 3D, extracción de parches `.npy`.
- **Fase 1** — Transformaciones para extracción de embeddings: cadena de preprocesamiento de imagen (2D o 3D), paso por backbone congelado (ViT/Swin/CvT/DenseNet) y extracción del CLS token como vector de embedding.

El proyecto maneja 5 dominios clínicos (Expertos 0–4) con dos familias de pipeline: **2D** (Chest, ISIC, OA) y **3D** (LUNA16, Páncreas). Cada dominio tiene transformaciones específicas además de las compartidas.

---

## 1. NIH ChestXray14 — Experto 0

### 1.1 Fase 0 — Preparación

**Archivo fuente:** `src/pipeline/fase0/pre_chestxray14.py`

| Paso | Descripción | Detalles |
|------|-------------|----------|
| 1. Symlinks | `crear_symlinks_all_images()` crea `all_images/` con symlinks relativos a los PNGs en las 12 subcarpetas `images_*/images/*.png` | Symlinks relativos para portabilidad entre servidores. Idempotente: si ya existen ≥ total de PNGs, se salta |
| 2. Splits oficiales | `verificar_split_txts()` verifica `train_val_list.txt` (≥80,000 líneas) y `test_list.txt` (≥20,000 líneas) | Si están truncados o ausentes, se extraen de `data.zip` automáticamente |
| 3. Auditoría CSV | `auditar_csv()` verifica `Data_Entry_2017.csv` | Columnas requeridas: `Image Index`, `Finding Labels`, `Patient ID`, `View Position`, `Follow-up #`. Calcula prevalencia por patología |
| 4. Split por Patient ID | En el Dataset (`chest.py`), se aplica el split usando `file_list` (archivos `.txt` oficiales de NIH) | Separación estricta por Patient ID, no aleatorio por imagen |
| 5. Verificación de leakage | `patient_ids_other` permite verificar 0 Patient IDs compartidos entre splits | Previene data leakage longitudinal (Follow-up #) |

### 1.2 Fase 1 — Transformaciones para Embedding

**Archivos fuente:** `src/pipeline/fase1/transform_2d.py`, `src/pipeline/fase1/dataset_builder.py`

En modo `embedding`, ChestXray14 recibe el transform 2D estándar construido por `build_2d_transform()`:

| Paso | Transformación | Parámetros exactos |
|------|---------------|-------------------|
| 1 | `transforms.Resize` | `(224, 224)` — `IMG_SIZE = 224` |
| 2 | `TotalVariationFilter` | `weight=10.0` (`TVF_WEIGHT`), `n_iter=30` (`TVF_N_ITER`). Usa `skimage.restoration.denoise_tv_chambolle`. El peso se escala internamente: `weight / 255.0 = 0.0392`. Soporta imágenes 2D (grises) y 3D (RGB con `channel_axis=-1`) |
| 3 | `GammaCorrection` | `gamma=1.0` (`DEFAULT_GAMMA`). Usa LUT precalculada de 256 entradas. Fórmula: `output = (input/255)^γ * 255`. Con γ=1.0 es identidad |
| 4 | `CLAHETransform` | `clip_limit=2.0` (`CLAHE_CLIP_LIMIT`), `tile_grid=(8, 8)` (`CLAHE_TILE_GRID`). Delega a `transform_domain.apply_clahe()`: convierte a escala de grises → `cv2.createCLAHE` → devuelve RGB (3 canales iguales) |
| 5 | `transforms.ToTensor` | Convierte PIL Image HWC uint8 → tensor CHW float32 [0, 1] |
| 6 | `transforms.Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (estadísticas ImageNet) |

**Carga de imagen:** `Image.open(img_path).convert("RGB")`. En caso de error, se sustituye por `np.zeros((224, 224, 3), dtype=np.uint8)`.

**Retorno en modo embedding:** `(img_tensor, expert_id=0, img_name)`

### 1.3 Consideraciones especiales

- **Filtro AP/PA (H4):** Parámetro opcional `filter_view` (valores: `"PA"`, `"AP"`, o `None`). Las imágenes AP (pacientes encamados) presentan corazón aparentemente más grande y distorsión geométrica — sesgo de dominio. Se emite warning cuando AP supera cierto porcentaje.
- **Pesos de clase (H6):** En modo `expert`, se calcula `pos_weight = n_neg / n_pos` por cada una de las 14 patologías para `BCEWithLogitsLoss`.
- **Ruido NLP (H3):** Las etiquetas fueron generadas por NLP con ~>90% precisión. AUC > 0.85 requiere revisión de confounding.
- **Modo expert vs embedding:**
  - `embedding` → devuelve `(img, expert_id, img_name)` — para routing
  - `expert` → devuelve `(img, label_vector_14, img_name)` — vector multi-label float32 [14]
- **BBox index (H5):** ~1,000 imágenes (~0.9%) tienen anotaciones de bounding box para 8 de 14 patologías.

---

## 2. ISIC 2019 — Experto 1

### 2.1 Fase 0 — Preparación

**Archivo fuente:** `src/pipeline/datasets/isic.py` (método `build_lesion_split`)

| Paso | Descripción | Detalles |
|------|-------------|----------|
| 1. Carga GT CSV | Lee `ISIC_2019_Training_GroundTruth.csv` | Verifica one-hot encoding (exactamente un 1.0 por fila) |
| 2. Eliminación de duplicados conocidos (H4) | Remueve IDs en `KNOWN_DUPLICATES = {"ISIC_0067980", "ISIC_0069013"}` | IDs hardcodeados como duplicados entre ISIC 2018/2019 |
| 3. Deduplicación MD5 (H4) | Si `img_dir` proporcionado, calcula hash MD5 de cada `.jpg` y elimina duplicados silenciosos | Puede tardar ~1-2 min para el dataset completo |
| 4. Split por lesion_id (H2) | Si `metadata_csv` proporcionado, merge con columna `lesion_id` y split por lesiones únicas | `frac_train=0.8`, `random_state=42`. Lesiones sin ID → ID único sintético `_unique_{image}` |
| 5. Verificación UNK (H5) | Confirma que columna `UNK=0` en train y val | UNK solo debe aparecer en test set oficial (slot 8 de softmax para OOD) |
| 6. Filtro _downsampled | En `dataset_builder.py`, se filtran imágenes con `_downsampled` en el nombre | Estas imágenes MSK no existen como archivos independientes en disco |
| 7. Auditoría de fuentes (H3) | `_source_audit()` clasifica por rango de ID numérico | HAM10000 (≤67977), BCN_20000 (>67977), MSK (_downsampled) |

### 2.2 Fase 1 — Transformaciones para Embedding

**Archivos fuente:** `src/pipeline/datasets/isic.py`, `src/pipeline/fase1/dataset_builder.py`

En modo `embedding`, ISIC usa su propio pipeline simplificado (definido en `build_isic_transforms()`):

| Paso | Transformación | Parámetros exactos |
|------|---------------|-------------------|
| 1 | `apply_circular_crop` (H3/Item-8) | Eliminación de bordes negros circulares de BCN_20000. Umbralización en 10 (no 0) para tolerar artefactos JPEG. Recorta al bounding rect del contenido visible. Idempotente: si el contenido ocupa >95% del frame, devuelve sin modificar |
| 2 | `transforms.Resize` | `(224, 224)` — `img_size=224` |
| 3 | `transforms.ToTensor` | HWC uint8 → CHW float32 [0, 1] |
| 4 | `transforms.Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

**Carga de imagen:** `Image.open(img_path).convert("RGB")` donde `img_path = img_dir / f"{img_name}.jpg"`.

**En modo `expert` (FASE 2),** se usan tres pipelines diferenciados:

**Pipeline `standard` (clases mayoritarias: MEL, NV, BCC, BKL):**

| Paso | Transformación | Parámetros |
|------|---------------|-----------|
| 1 | `Resize` | `(224, 224)` |
| 2 | `RandomHorizontalFlip` | `p=0.5` |
| 3 | `RandomRotation` | `degrees=30` |
| 4 | `ColorJitter` | `brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05` |
| 5 | `ToTensor` | — |
| 6 | `Normalize` | ImageNet mean/std |

**Pipeline `minority` (clases minoritarias: AK=3, DF=5, VASC=6, SCC=7, ref. Gessert et al. 2020):**

| Paso | Transformación | Parámetros |
|------|---------------|-----------|
| 1 | `Resize` | `(224, 224)` |
| 2 | `RandomHorizontalFlip` | `p=0.5` |
| 3 | `RandomVerticalFlip` | `p=0.5` |
| 4 | `RandomApply([RandomRotation((90, 90))])` | `p=0.33` |
| 5 | `RandomApply([RandomRotation((180, 180))])` | `p=0.33` |
| 6 | `ColorJitter` | `brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1` |
| 7 | `RandomAffine` | `degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)` |
| 8 | `ToTensor` | — |
| 9 | `Normalize` | ImageNet mean/std |
| 10 | `RandomErasing` | `p=0.2, scale=(0.02, 0.2)` (Cutout) |

### 2.3 Consideraciones especiales

- **Crop circular BCN (H3/Item-8):** Las imágenes de dermoscopio BCN_20000 tienen campo de visión circular con fondo negro. `apply_circular_crop()` recorta al bounding rect del contenido visible. Umbral de binarización = 10 (tolera artefactos JPEG). Idempotente para HAM/MSK.
- **3 fuentes con bias de dominio:** HAM10000 (Viena, 600×450, recortado), BCN_20000 (Barcelona, 1024×1024, dermoscopio), MSK (NYC, variable, `_downsampled`).
- **9 clases:** 8 en train (`MEL, NV, BCC, AK, BKL, DF, VASC, SCC`) + UNK solo en test (slot 8 softmax para OOD).
- **WeightedRandomSampler (H6):** Disponible via `get_weighted_sampler()` para compensar desbalance NV >> DF/VASC en FASE 2.
- **Class weights:** `total / (N_TRAIN_CLS * counts)` por clase + peso 1.0 para UNK.

---

## 3. OA Rodilla — Experto 2

### 3.1 Fase 0 — Preparación

**Archivo fuente:** `src/pipeline/datasets/osteoarthritis.py`

| Paso | Descripción | Detalles |
|------|-------------|----------|
| 1. Lectura por carpetas | Las clases se leen desde subdirectorios numéricos dentro de `root_dir/split/` | Carpetas `0/`, `1/`, `2/` contienen imágenes `.jpg` y `.png` |
| 2. Split por carpeta (H3) | Sin metadatos de paciente disponibles — split tomado tal cual de las carpetas del ZIP | `train/`, `val/`, `test/` predefinidos |
| 3. Verificación de clases | Se verifican 3 clases consolidadas (0=Normal, 1=Leve, 2=Severo) | Warning si se encuentran 5 clases (KL 0-4 sin consolidar) |
| 4. Detección de augmentation offline (H2) | Compara conteo de imágenes vs esperados (`train=5778, val=826, test=1656`) | Si ratio > 1.10x, se detecta augmentation offline. Verificación cruzada con hash MD5 de 20 muestras |
| 5. Verificación de canales (Item-4) | Confirma que las imágenes son grises guardadas como RGB | Muestra de 5 imágenes: verifica `R ≈ G ≈ B` (atol=2) |
| 6. Verificación de dimensiones (Item-8) | Verifica consistencia de dimensiones en muestra de 10 | Dimensiones variables se compensan con resize |

### 3.2 Fase 1 — Transformaciones para Embedding

**Archivos fuente:** `src/pipeline/datasets/osteoarthritis.py`, `src/pipeline/preprocessing.py`

En modo `embedding`, OA aplica un pipeline especial con CLAHE ANTES del resize:

| Paso | Transformación | Parámetros exactos |
|------|---------------|-------------------|
| 1 | `Image.open().convert("RGB")` | Carga como RGB |
| 2 | `apply_clahe()` (H4) | **ANTES del resize**, a resolución original. `clip_limit=2.0`, `tile_grid=(8, 8)`. Convierte a escala de grises → `cv2.createCLAHE` → RGB (3 canales iguales) |
| 3 | `img.resize()` | `(img_size, img_size)` = `(224, 224)`, interpolación `Image.BICUBIC` |
| 4 | `transforms.ToTensor` | HWC uint8 → CHW float32 [0, 1] |
| 5 | `transforms.Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (IMAGENET_MEAN/STD de `config.py`) |

**Nota:** OA **no** usa `build_2d_transform()` de Fase 1 directamente. Tiene su propio `base_transform` y `aug_transform` definidos en el constructor. El CLAHE se aplica manualmente en `__getitem__()` antes del resize.

**En modo `expert` (FASE 2), si no hay augmentation offline:**

| Paso | Transformación | Parámetros |
|------|---------------|-----------|
| 1 | `apply_clahe()` | Mismos parámetros que embedding |
| 2 | `img.resize()` | `(224, 224)`, BICUBIC |
| 3 | `RandomHorizontalFlip` | `p=0.5` |
| 4 | `RandomRotation` | `degrees=10` |
| 5 | `ColorJitter` | `brightness=0.2, contrast=0.15` |
| 6 | `ToTensor` | — |
| 7 | `Normalize` | ImageNet mean/std |

### 3.3 Consideraciones especiales

- **CLAHE antes del resize (H4):** Orden crítico — CLAHE a alta resolución preserva la densidad estadística del histograma local del espacio articular. Aplicar después del resize destruiría información diagnóstica.
- **3 clases ordinales:** 0=Normal, 1=Leve (KL 0-1), 2=Severo (KL 3-4). `OA_N_CLASSES = 3`.
- **Contaminación KL1 (H5):** La frontera Clase0↔Clase1 es la más difícil por alta ambigüedad inter-observador en grado KL1. Herramienta de monitoreo: `evaluate_boundary_confusion()`.
- **Métrica principal:** QWK (Quadratic Weighted Kappa), no Accuracy.
- **NUNCA usar `RandomVerticalFlip`** — las radiografías de rodilla tienen orientación anatómica fija.
- **Class weights:** `total / (OA_N_CLASSES * counts)` para `CrossEntropyLoss`.
- **Augmentation offline:** Si se detecta (ratio > 1.10x), se desactiva el augmentation online y se usa solo `base_transform`.

---

## 4. LUNA16 — Experto 3

### 4.1 Fase 0 — Preparación 3D

**Archivo fuente:** `src/pipeline/fase0/pre_embeddings.py`

| Paso | Descripción | Detalles |
|------|-------------|----------|
| 1. Descubrimiento de CTs | Busca subcarpetas `subset*/` dentro de `ct_volumes/` | Busca archivos `.mhd` recursivamente |
| 2. Validación de `.raw` | Filtra CTs con archivo `.raw` ausente o < 1 MB (`MIN_RAW_BYTES = 1,048,576`) | Segundo filtro: verifica tamaño real vs declarado en header `.mhd` (tolerancia 95%) |
| 3. Carga `candidates_V2.csv` | Lee candidatos y filtra por seriesuids disponibles | Verifica uso de V2 (no V1 obsoleto). V2 detecta 1,162/1,186 nódulos (+24 que V1) |
| 4. Muestreo de negativos | `apply_neg_sampling()`: ratio configurable N:1 respecto a positivos | Default `neg_ratio=10`. Semillas por split: train=42, val=43, test=44 |
| 5. Opción `--max_neg` | Límite absoluto de negativos totales | Semilla: `RANDOM_SEED + 1 = 43` |
| 6. Splits por seriesuid | Usa `luna_splits.json` generado previamente | `train_uids`, `val_uids`, `test_uids`. Fallback: split 80/10/10 ad-hoc con `RANDOM_SEED=42` |
| 7. Extracción de parches | `_worker()` — ProcessPoolExecutor | Parches `candidate_{idx:06d}.npy` en `patches/{train,val,test}/` |
| 8. Conversión world→vóxel | `world_to_voxel(coord_world, origin, spacing, direction)` | `coord_shifted = coord_world - origin`; `coord_voxel = solve(direction.reshape(3,3) * spacing, coord_shifted)`; `[::-1]` para obtener [iz, iy, ix] |
| 9. Extracción del parche | Centrado en vóxel convertido, tamaño `PATCH_SIZE=64` (64³) | `half = 32`. Padding con `HU_LUNG_CLIP[0] = -1000` si el parche excede los bordes del volumen |
| 10. Normalización HU | `clip[-1000, 400]` → escala lineal a `[0, 1]` | `(patch - (-1000)) / (400 - (-1000))` = `(patch + 1000) / 1400` |
| 11. Validación | `validate_patches()`: verifica shape `(64, 64, 64)` y media > 0.05 | Muestra de 20 parches por split (o 10 en validación final) |

### 4.2 Fase 1 — Transformaciones 3D

**Archivos fuente:** `src/pipeline/datasets/luna.py`, `src/pipeline/preprocessing.py`, `src/pipeline/fase1/transform_3d.py`

En modo `embedding`, LUNA16 aplica el pipeline 3D→2D:

| Paso | Transformación | Parámetros exactos |
|------|---------------|-------------------|
| 1 | `np.load(patch_file)` | Carga parche `.npy` float32 [64, 64, 64], ya normalizado a [0, 1] |
| 2 | `resize_volume_3d(volume)` | Resize trilineal `[D,H,W]` → `(64, 64, 64)` (`PATCH_3D_SIZE`). Usa `F.interpolate(mode="trilinear", align_corners=False)`. Para LUNA los parches ya son 64³, así que es identidad |
| 3 | `volume_to_vit_input(volume_t)` | Proyección 3D→2D: extrae 3 cortes centrales |
| 3a | Corte axial | `v[d//2, :, :]` — corte central eje Z |
| 3b | Corte coronal | `v[:, h//2, :]` — corte central eje Y |
| 3c | Corte sagital | `v[:, :, w//2]` — corte central eje X |
| 3d | Stack RGB | `torch.stack([axial, coronal, sagittal], dim=0)` → [3, 64, 64] |
| 3e | Resize bilineal | `F.interpolate(size=(224, 224), mode="bilinear", align_corners=False)` → [3, 224, 224] |
| 3f | Normalización ImageNet | `(rgb - mean) / std` con `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

**Retorno en modo embedding:** `(img_2d_tensor [3,224,224], expert_id=3, patch_stem)`

**En modo `expert` (FASE 2):**

| Paso | Transformación | Parámetros |
|------|---------------|-----------|
| 1 | `np.load(patch_file)` | Parche [64, 64, 64] |
| 2 | `torch.from_numpy(volume).float().unsqueeze(0)` | → tensor [1, 64, 64, 64] |

### 4.3 Consideraciones especiales

- **Justificación del rango HU [-1000, 400]:**
  - `-1000` HU = aire puro (fondo del parche, padding consistente)
  - `-900 a -500` HU = tejido pulmonar
  - `-100 a +100` HU = nódulos sólidos (objetivo de detección)
  - `+400` HU = límite. Hueso (>400 HU) excluido — dominaría la dinámica de la red
- **Verificación HU (`verify_hu_normalization()`):** Parche bien normalizado: min ≈ 0.0, max ≈ 1.0, mean entre 0.0 y 0.3.
- **Ratio de desbalance:** ~490:1 (neg/pos). FocalLoss(gamma=2, alpha=0.25) obligatoria. BCELoss convergería al mínimo trivial (accuracy >99.7%).
- **Candidates V2 vs V1:** V1 es obsoleto (ISBI 2016). V2 detecta 24 nódulos más. Techo teórico de sensitividad: `1120/1186 ≈ 0.9443` (66 nódulos sin candidato).
- **Spacing variable (Item-9):** Se verifican dimensiones de parches en disco — todos deben ser `(64, 64, 64)`.
- **Gradient checkpointing (H5):** Obligatorio con 12 GB VRAM. Condiciones simultáneas: FP16, gradient checkpointing, batch_size ≤ 4.
- **Métrica oficial:** CPM (Competition Performance Metric), NO AUC-ROC. Promedio de sensitividad en 7 puntos FP/scan: `{1/8, 1/4, 1/2, 1, 2, 4, 8}`.
- **Constante en parche:** Warning si `volume.max() == volume.min()` — posible error en conversión world→vóxel.

---

## 5. Páncreas PANORAMA — Experto 4

### 5.1 Fase 0 — Preparación 3D

**Archivos fuente:** `src/pipeline/fase0/pre_embeddings.py`, `src/pipeline/datasets/pancreas.py`

| Paso | Descripción | Detalles |
|------|-------------|----------|
| 1. Carga de etiquetas (H1) | `PanoramaLabelLoader.load_labels()` — Las etiquetas PDAC están en un repositorio GitHub separado (`panorama_labels`) | El ZIP de Zenodo NO incluye etiquetas. Verificación del hash del commit para reproducibilidad |
| 2. Cross-match (H1) | `PanoramaLabelLoader.cross_match()` — Cruza case_ids del repo con `.nii.gz` del ZIP | FIX: normaliza sufijo nnU-Net `_XXXX` (4 dígitos) antes del match. Ej: `100298_00001_0000.nii.gz` → `100298_00001` |
| 3. Lectura de etiquetas | Busca archivos `.json` y `.csv` en el repo clonado | Etiquetas binarias: 0=PDAC negativo, 1=PDAC positivo |
| 4. k-fold CV (H5/Item-8) | `build_kfold_splits()` — `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | Agrupa por patient_id (primeros dígitos antes de `_`). Previene leakage por paciente |
| 5. Validación de preprocesado | `validar_preprocesado_pancreas()` — muestra de ~10 volúmenes | Verifica shape `(64,64,64)`, rango `[0,1]`, sin NaN/Inf, std > 0.01 (no constante) |
| 6. Preprocesado isotrópico | `preprocess_pancreas_volume()` en `pre_embeddings.py` | HU clip `[-100, 400]` → `[0, 1]`, zoom isotrópico con `scipy.ndimage.zoom(order=1)`, resize final a `(64, 64, 64)` |

### 5.2 Fase 1 — Transformaciones 3D

**Archivos fuente:** `src/pipeline/datasets/pancreas.py`, `src/pipeline/preprocessing.py`, `src/pipeline/fase1/transform_3d.py`

En modo `embedding`, Páncreas aplica:

| Paso | Transformación | Parámetros exactos |
|------|---------------|-------------------|
| 1 | `sitk.ReadImage(nii_path)` → `GetArrayFromImage()` | Carga volumen `.nii.gz` como array float32 |
| 2a | **Con z-score** (`z_score_per_volume=True`, default): Clip HU | `np.clip(volume, -100, 400)` — `HU_ABDOMEN_CLIP = (-100, 400)` |
| 2b | Z-score por volumen | `volume = (volume - mean_v) / std_v`. Luego clip a `[-3, 3]` y normalización a `[0, 1]`: `(volume - (-3)) / 6.0`. Si `std_v ≤ 1e-6` → tensor de ceros |
| 2c | **Sin z-score** (`z_score_per_volume=False`): | `normalize_hu(volume, -100, 400)` — normalización lineal `(v - lo) / (hi - lo)` |
| 3 | `resize_volume_3d(volume, target=(64, 64, 64))` | Resize trilineal con `F.interpolate(mode="trilinear", align_corners=False)` |
| 4 | `volume_to_vit_input(volume_t)` | Proyección 3D→2D: misma cadena que LUNA16 (3 cortes centrales → stack RGB → resize bilineal a 224×224 → normalización ImageNet) |

**Retorno en modo embedding:** `(img_2d_tensor [3,224,224], expert_id=4, case_stem)`

**En modo `expert` (FASE 2):**

| Paso | Transformación | Parámetros |
|------|---------------|-----------|
| 1-2 | Misma normalización HU y z-score | — |
| 3 | ROI extraction | **Opción A:** `PancreasROIExtractor.extract_option_a()` — Resize completo a `(64,64,64)` con `F.interpolate(trilinear)`. Solo válida para routing. **Opción B:** `extract_option_b()` — Recorte `Z[120:220]` (`PANCREAS_Z_MIN=120`, `PANCREAS_Z_MAX=220`) + resize a `(64,64,64)`. Mejora ratio páncreas/volumen ~3× |
| 4 | `torch.from_numpy(roi).float().unsqueeze(0)` | → tensor [1, 64, 64, 64] |

### 5.3 Consideraciones especiales

- **Clip HU abdominal [-100, 400] (H3):** NO usar `[-1000, 400]` de LUNA16 — comprime contraste diagnóstico 7×. Parénquima pancreático: +30 a +150 HU. Tumor PDAC (hipodenso): -20 a +80 HU.
- **z-score por volumen (H3/Item-9):** Compensa diferencias sistemáticas entre fuentes multicéntricas (Radboudumc, MSD, NIH). Media=0, std=1 dentro de cada volumen.
- **Páncreas ocupa ~0.5–2% del volumen CT (H2):** Resize naïve a 64³ deja el páncreas en ~5 vóxeles. Opción A solo para FASE 0. Opción B recomendada para FASE 2.
- **Auditoría de fuentes (Item-9):** Radboudumc+UMCG (Holanda), MSD Task07 (NYC, ~50% PDAC), NIH Pancreas-CT (100% negativos → puede inflar accuracy por bias de dominio).
- **FocalLoss:** `alpha=0.75, gamma=2` (vs `alpha=0.25` de LUNA16) — más peso a PDAC+ para penalizar FN.
- **Dataset limitado:** ~281 volúmenes. k-fold CV (k=5) obligatorio.
- **Gradient checkpointing:** Obligatorio con 12 GB VRAM. CT abdominal 512×512×300 float32 = ~300 MB. batch_size=1–2 + FP16 + checkpoint.
- **Detección de convergencia trivial:** `check_trivial_convergence()` alerta si `prob_pos_mean < 0.4` o `frac_pred_pos < 0.05`.

---

## 6. Transformaciones Compartidas

### 6.1 Cadena 2D (datasets de imagen plana)

**Archivo fuente principal:** `src/pipeline/fase1/transform_2d.py`

La cadena estándar 2D se construye con `build_2d_transform()` y sigue el orden estricto definido en §6.2 de `arquitectura_moe.md`:

```
Resize → TVF → GammaCorrection → CLAHE → [guardar TransformRecord] → ToTensor → Normalize
```

**Detalle paso a paso con parámetros exactos:**

#### Paso 1: Resize
- **Función:** `transforms.Resize((img_size, img_size))`
- **Default:** `IMG_SIZE = 224` → `(224, 224)`
- **Interpolación:** Bilineal (default de torchvision)
- **Propósito:** Estandarizar resolución para el backbone ViT

#### Paso 2: Total Variation Filter (TVF)
- **Clase:** `TotalVariationFilter`
- **Backend:** `skimage.restoration.denoise_tv_chambolle` (algoritmo de Chambolle 2004)
- **Parámetros:**
  - `weight = TVF_WEIGHT = 10.0` (se escala internamente a `10.0 / 255.0 ≈ 0.0392` para rango [0,1])
  - `n_iter = TVF_N_ITER = 30` (iteraciones máximas del solver)
- **Operación:** PIL → numpy float64 / 255.0 → denoise_tv_chambolle → × 255 → uint8 → PIL
- **Referencia:** PMC9340712 — TVF + Gamma mejora accuracy y convergencia
- **Imágenes RGB:** usa `channel_axis=-1`; imágenes 2D (grises): sin channel_axis

#### Paso 3: Corrección Gamma
- **Clase:** `GammaCorrection`
- **Parámetro:** `gamma = DEFAULT_GAMMA = 1.0` (identidad por defecto)
- **Implementación:** LUT precalculada de 256 entradas uint8: `lut[i] = clip(((i/255)^γ) * 255, 0, 255)`
- **Rango clínico típico:** 0.8–1.2
  - `γ < 1` → imagen más brillante (realza sombras)
  - `γ = 1` → identidad
  - `γ > 1` → imagen más oscura (realza highlights)

#### Paso 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Clase wrapper:** `CLAHETransform`
- **Implementación real:** `transform_domain.apply_clahe()`
- **Parámetros:**
  - `clip_limit = CLAHE_CLIP_LIMIT = 2.0`
  - `tile_grid = CLAHE_TILE_GRID = (8, 8)`
- **Operación:** PIL RGB → `convert("L")` → `cv2.createCLAHE(clipLimit, tileGridSize)` → `clahe.apply()` → `cv2.cvtColor(COLOR_GRAY2RGB)` → PIL RGB
- **Nota:** Convierte a gris antes del realce — realzar R, G, B por separado alteraría el balance de color

#### Paso 5: Serialización del TransformRecord
- **Clase:** `TransformRecord` (dataclass serializable)
- **Archivos generados:** `<path>.pkl` (pickle, reconstrucción exacta) + `<path>.json` (legible, portabilidad)
- **Campos:** `img_size`, `tvf_weight`, `tvf_n_iter`, `gamma`, `clahe_clip_limit`, `clahe_tile_grid`, `imagenet_mean`, `imagenet_std`, `pipeline_order`
- **Propósito:** Reproducibilidad y mapeo para Grad-CAM (Funcionalidad 4 del dashboard)

#### Paso 6: ToTensor + Normalize
- **`transforms.ToTensor()`:** PIL Image HWC uint8 → tensor CHW float32 [0, 1]
- **`transforms.Normalize(mean, std)`:**
  - `mean = IMAGENET_MEAN = [0.485, 0.456, 0.406]`
  - `std = IMAGENET_STD = [0.229, 0.224, 0.225]`

**Datasets que usan la cadena 2D estándar completa:**
- **ChestXray14:** Usa `build_2d_transform()` directamente via `dataset_builder.py`
- **ISIC:** Usa pipeline simplificado propio (Resize + ToTensor + Normalize) en modo embedding; NO usa TVF/Gamma/CLAHE
- **OA:** Usa CLAHE manual en `__getitem__()` + resize BICUBIC + ToTensor + Normalize; NO usa TVF/Gamma via `build_2d_transform()`

### 6.2 Cadena 3D (LUNA16 y Páncreas)

**Archivos fuente:** `src/pipeline/fase1/transform_3d.py`, `src/pipeline/preprocessing.py`

Ambos datasets 3D comparten las funciones de normalización y proyección:

#### Paso 1: Normalización HU — `normalize_hu(volume, min_hu, max_hu)`
- **Fórmula:** `np.clip(volume, min_hu, max_hu)` → `(volume - min_hu) / (max_hu - min_hu)`
- **LUNA16:** `min_hu=-1000, max_hu=400` → `HU_LUNG_CLIP`
- **Páncreas:** `min_hu=-100, max_hu=400` → `HU_ABDOMEN_CLIP`
- **Salida:** float32 en rango [0, 1]

#### Paso 2: Resize 3D — `resize_volume_3d(volume, target)`
- **Target:** `PATCH_3D_SIZE = (64, 64, 64)`
- **Operación:** `numpy → torch.unsqueeze(0).unsqueeze(0)` → `F.interpolate(size=target, mode="trilinear", align_corners=False)` → `squeeze(0)` → [1, 64, 64, 64]

#### Paso 3: Proyección 3D→2D — `volume_to_vit_input(volume_3d_tensor, img_size)`
- **Entrada:** tensor [1, 64, 64, 64]
- **Operación:**
  1. Squeeze → [64, 64, 64]
  2. Extrae 3 cortes centrales: `axial = v[32,:,:]`, `coronal = v[:,32,:]`, `sagittal = v[:,:,32]`
  3. Stack como RGB: `torch.stack([axial, coronal, sagittal])` → [3, 64, 64]
  4. Resize bilineal: `F.interpolate(size=(224, 224), mode="bilinear", align_corners=False)` → [3, 224, 224]
  5. Normalización ImageNet: `(rgb - mean) / std`
- **Propósito:** Representación aproximada válida para routing. Para clasificación 3D real, los expertos usan el tensor volumétrico completo [1, 64, 64, 64].

#### Pipeline completo — `full_3d_pipeline(volume_np, min_hu, max_hu, target, img_size)`
- Conveniencia que encadena: `normalize_hu()` → `resize_volume_3d()` → `volume_to_vit_input()`

### 6.3 Transformaciones de dominio específicas

**Archivo fuente:** `src/pipeline/fase1/transform_domain.py`

| Transformación | Función | Uso | Parámetros |
|---------------|---------|-----|-----------|
| CLAHE | `apply_clahe()` | OA Rodilla (antes del resize), cadena 2D estándar (paso 4) | `clip_limit=2.0`, `tile_grid=(8, 8)` |
| Crop circular | `apply_circular_crop()` | ISIC BCN_20000 (antes del pipeline estándar) | Umbral binarización=10, idempotente si contenido >95% del frame |

### 6.4 Funciones de preprocesamiento global

**Archivo fuente:** `src/pipeline/preprocessing.py`

Este módulo contiene las implementaciones originales (usadas por los datasets de Familia 3):

| Función | Descripción | Parámetros |
|---------|-------------|-----------|
| `build_2d_transform(img_size=224)` | Pipeline básico: Resize + ToTensor + Normalize | Sin TVF/Gamma/CLAHE |
| `apply_clahe()` | CLAHE a resolución original | `clip_limit=2.0`, `tile_grid=(8, 8)` |
| `apply_circular_crop()` | Crop circular BCN_20000 | Umbral=10, idempotente >95% |
| `normalize_hu()` | Normalización HU lineal | `min_hu=-1000, max_hu=400` |
| `resize_volume_3d()` | Resize trilineal | `target=(64,64,64)` |
| `volume_to_vit_input()` | Proyección 3D→2D | 3 cortes centrales → RGB 224×224 |

---

## 7. Línea de Tiempo del Pipeline de Transformaciones

### Datasets 2D — ChestXray14 (usa cadena completa)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  FASE 0 — Preparación                    FASE 1 — Embedding                     │
│                                                                                  │
│  [Symlinks all_images/]                  [Resize 224×224]                        │
│  12 subcarpetas → links relativos              │                                │
│        │                                       ▼                                │
│        ▼                                [TotalVariation Filter]                  │
│  [Verificar split TXTs]                  w=10.0 (÷255→0.039)                    │
│  train_val_list.txt ≥80k                 n_iter=30                              │
│  test_list.txt ≥20k                            │                                │
│        │                                       ▼                                │
│        ▼                                [Gamma Correction]                      │
│  [Auditar Data_Entry_2017.csv]            γ=1.0 (identidad)                     │
│  5 columnas requeridas                    LUT 256 entradas                       │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Split por Patient ID]                  [CLAHE]                                │
│  file_list oficiales NIH                  clip=2.0, tile=8×8                    │
│        │                                  Gray→CLAHE→RGB                        │
│        ▼                                       │                                │
│  [Filtro View PA/AP]                           ▼                                │
│  (opcional)                              [ToTensor + Normalize]                  │
│        │                                  μ=[0.485, 0.456, 0.406]               │
│        ▼                                  σ=[0.229, 0.224, 0.225]               │
│  [Verificar leakage]                           │                                │
│  0 Patient IDs compartidos                     ▼                                │
│                                          [Backbone ViT (frozen)]                │
│                                           → CLS token = embedding               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Datasets 2D — ISIC 2019 (pipeline propio simplificado)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  FASE 0 — Preparación                    FASE 1 — Embedding                     │
│                                                                                  │
│  [Cargar GT CSV]                         [Circular Crop BCN]                    │
│  ISIC_2019_Training_GroundTruth            umbral=10                            │
│        │                                   idempotente >95%                     │
│        ▼                                       │                                │
│  [Eliminar KNOWN_DUPLICATES]                   ▼                                │
│  {ISIC_0067980, ISIC_0069013}            [Resize 224×224]                       │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Deduplicación MD5]                     [ToTensor]                             │
│  hash de cada .jpg                             │                                │
│        │                                       ▼                                │
│        ▼                                 [Normalize ImageNet]                   │
│  [Split por lesion_id]                    μ=[0.485, 0.456, 0.406]              │
│  80/20, random_state=42                   σ=[0.229, 0.224, 0.225]              │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Verificar UNK=0 en train]             [Backbone ViT (frozen)]                │
│        │                                  → CLS token = embedding              │
│        ▼                                                                        │
│  [Filtrar _downsampled]                                                         │
│  MSK no disponibles en disco                                                    │
│        │                                                                        │
│        ▼                                                                        │
│  [Auditoría fuentes]                                                            │
│  HAM/BCN/MSK por ID range                                                       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Datasets 2D — OA Rodilla (CLAHE antes del resize)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  FASE 0 — Preparación                    FASE 1 — Embedding                     │
│                                                                                  │
│  [Lectura por carpetas]                  [Image.open().convert("RGB")]           │
│  root/{train,val,test}/{0,1,2}/                │                                │
│        │                                       ▼                                │
│        ▼                                 [CLAHE — resolución ORIGINAL]          │
│  [Verificar 3 clases]                     clip=2.0, tile=8×8                    │
│  0=Normal, 1=Leve, 2=Severo               Gray→CLAHE→RGB                       │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Detectar aug offline]                  [Resize 224×224 BICUBIC]               │
│  ratio vs esperado >1.10x                      │                                │
│        │                                       ▼                                │
│        ▼                                 [ToTensor + Normalize]                  │
│  [Verificar canales R≈G≈B]               μ=[0.485, 0.456, 0.406]               │
│  muestra de 5 imgs                        σ=[0.229, 0.224, 0.225]               │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Verificar dimensiones]                [Backbone ViT (frozen)]                 │
│  muestra de 10 imgs                       → CLS token = embedding              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Datasets 3D — LUNA16 (parches nódulo pulmonar)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  FASE 0 — Extracción de Parches          FASE 1 — Embedding                     │
│                                                                                  │
│  [Descubrir CTs .mhd]                   [np.load(patch.npy)]                    │
│  subsets/ en ct_volumes/                  float32 [64,64,64]                    │
│        │                                  ya normalizado [0,1]                  │
│        ▼                                       │                                │
│  [Validar .raw]                                ▼                                │
│  ≥1 MB + header check                   [resize_volume_3d]                      │
│        │                                  trilineal → (64,64,64)                │
│        ▼                                  (identidad para LUNA)                 │
│  [Cargar candidates_V2.csv]                    │                                │
│  verificar V2 (no V1)                          ▼                                │
│        │                                [volume_to_vit_input]                   │
│        ▼                                  3 cortes centrales:                   │
│  [Neg sampling]                           axial  = v[32,:,:]                    │
│  ratio 10:1                               coronal = v[:,32,:]                   │
│        │                                  sagital = v[:,:,32]                   │
│        ▼                                       │                                │
│  [Splits por seriesuid]                        ▼                                │
│  luna_splits.json                        [Stack RGB → [3,64,64]]                │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Conversión world→vóxel]               [Resize bilineal]                       │
│  solve(direction*spacing,                 → [3, 224, 224]                       │
│   coord - origin) → [iz,iy,ix]                 │                                │
│        │                                       ▼                                │
│        ▼                                [Normalize ImageNet]                    │
│  [Extraer parche 64³]                     μ=[0.485, 0.456, 0.406]              │
│  centrado en vóxel                        σ=[0.229, 0.224, 0.225]              │
│  padding=-1000 HU                              │                                │
│        │                                       ▼                                │
│        ▼                                [Backbone ViT (frozen)]                 │
│  [Clip HU [-1000, 400]]                   → CLS token = embedding              │
│  → escala lineal [0, 1]                                                         │
│        │                                                                        │
│        ▼                                                                        │
│  [Guardar .npy 64³ float32]                                                     │
│  candidate_{idx:06d}.npy                                                        │
│  en patches/{train,val,test}/                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Datasets 3D — Páncreas PANORAMA (volúmenes CT abdominales)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  FASE 0 — Preparación                    FASE 1 — Embedding                     │
│                                                                                  │
│  [Clonar repo etiquetas]                 [sitk.ReadImage(.nii.gz)]              │
│  github.com/DIAGNijmegen/                 → GetArrayFromImage()                 │
│  panorama_labels                          float32 [D, H, W]                     │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Cross-match IDs]                       [Clip HU abdomen]                      │
│  FIX sufijo nnU-Net _XXXX                 np.clip(v, -100, 400)                 │
│        │                                  HU_ABDOMEN_CLIP                       │
│        ▼                                       │                                │
│  [Validar preprocesado]                        ▼                                │
│  muestra 10 vols                         [Z-score por volumen]                  │
│  shape (64³), rango [0,1]                 mean=0, std=1                         │
│        │                                  clip [-3, 3]                          │
│        ▼                                  → (v+3)/6 → [0, 1]                   │
│  [k-fold CV k=5]                               │                                │
│  StratifiedGroupKFold                          ▼                                │
│  por patient_id                          [resize_volume_3d]                      │
│  random_state=42                          trilineal → (64,64,64)                │
│        │                                       │                                │
│        ▼                                       ▼                                │
│  [Auditoría fuentes]                    [volume_to_vit_input]                   │
│  Radboudumc / MSD / NIH                   3 cortes centrales                    │
│                                            → RGB [3,64,64]                      │
│                                            → bilineal [3,224,224]               │
│                                            → Normalize ImageNet                 │
│                                                 │                                │
│                                                 ▼                                │
│                                          [Backbone ViT (frozen)]                │
│                                            → CLS token = embedding              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Resumen General del Flujo

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE COMPLETO: FASE 0 → FASE 1 → FASE 2                   │
│                                                                                        │
│  ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐    ┌────────────┐ │
│  │   DATOS      │    │    FASE 0             │    │    FASE 1        │    │  FASE 2    │ │
│  │   CRUDOS     │───▶│  Splits + Validación  │───▶│  Transformación  │───▶│ Entrenami- │ │
│  │              │    │  + Extracción 3D      │    │  + Embedding     │    │ ento MoE   │ │
│  └─────────────┘    └──────────────────────┘    └─────────────────┘    └────────────┘ │
│                                                                                        │
│  Experto 0 (Chest):   CSV + Splits TXT ──────▶ Resize→TVF→Gamma→CLAHE→Norm ──▶ CLS   │
│  Experto 1 (ISIC):    Lesion split + MD5 ────▶ CircCrop→Resize→Norm ──────────▶ CLS   │
│  Experto 2 (OA):      Carpetas predefinidas ─▶ CLAHE(orig)→Resize→Norm ──────▶ CLS   │
│  Experto 3 (LUNA):    Parches .npy 64³ ──────▶ 3Cortes→RGB→Resize→Norm ─────▶ CLS   │
│  Experto 4 (Panc):    NIfTI + etiquetas ─────▶ HU→zscore→Resize3D→3Cortes ──▶ CLS   │
│  Experto 5 (OOD):     (sin dataset — se inicializa en Fase 2 como MLP OOD)            │
│                                                                                        │
│  Backbones disponibles:                                                                │
│    vit_tiny_patch16_224  (d=192,  ~2 GB VRAM) ← DEFAULT                               │
│    swin_tiny_patch4_w7   (d=768,  ~4 GB VRAM)                                         │
│    cvt_13                (d=384,  ~3 GB VRAM)                                          │
│    densenet121_custom    (d=1024, ~3 GB VRAM)                                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

*Documento generado a partir del análisis exhaustivo de 14 archivos Python del proyecto. Todos los parámetros numéricos fueron extraídos directamente del código fuente.*
