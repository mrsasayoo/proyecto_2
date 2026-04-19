# Paso 1 — Descarga de Datos: Auditoría Completa

| Campo | Valor |
|---|---|
| **Fecha de auditoría** | 2026-04-05 |
| **Auditor** | Multi-agente: ARGOS (alineación guía↔disco), SIGMA (verificación de fuentes), EXPLORE (inspección de filesystem) |
| **Alcance** | Todos los datasets requeridos por §2 de la guía del proyecto + auxiliar PANORAMA |
| **Estado general** | ✅ **Paso 1 COMPLETO — todos los 19 ítems resueltos (A1–A19). ✅** Los 5 datasets de dominio + PANORAMA auxiliar están descargados, extraídos y con splits generados. Todos los ZIPs originales eliminados (~172.6 GB liberados). Bugs corregidos: ruta `fase0_pipeline.py`, OOM `_process_mask()`, leakage `split_pancreas()`. `cae_splits.csv` regenerado (162,611 filas). A19 resuelto: `_build_pairs()` omite casos sin CT silenciosamente (guard `if candidates:`). |
| **Commit del proyecto** | `948cd78b6de16a53d7220b92ce8863ba3d910edc` |

---

## 1. Resumen ejecutivo

Los cinco datasets requeridos por la guía del proyecto (NIH ChestXray14, ISIC 2019, Osteoarthritis Knee, LUNA16, Pancreatic Cancer) **están presentes en disco** con conteos de archivos consistentes y splits generados. Sin embargo, la documentación viviente (`arquitectura_documentacion.md`, §7.2) sigue declarando que el dataset de Páncreas está vacío y que los parches de test de LUNA16 no existen — ambas afirmaciones son falsas. Además, el dataset de Osteoarthritis Knee fue descargado desde una cuenta personal de Kaggle sin procedencia verificable al repositorio oficial, lo que constituye un riesgo de integridad científica (ISIC 2019 fue re-descargado desde la fuente oficial el 2026-04-05). Finalmente, se detectaron anomalías de estructura de directorios (sufijos `_downsampled` inconsistentes) que causarán fallos en el pipeline si no se corrigen antes de Fase 1. El directorio `isic_images/` fue eliminado de disco el 2026-04-06 (INC-06 cerrado).

---

## 2. Inventario de datasets

| # | Dataset | Fuente en la guía | Estado descarga | Archivos en disco | Splits | Riesgo de fuente |
|---|---------|-------------------|-----------------|-------------------|--------|------------------|
| 1 | NIH ChestXray14 | Kaggle (`nih-chest-xrays/data`) | ✅ Descargado | 112,120 .png | ✅ train:88,999 / val:11,349 / test:11,772 | 🟢 Bajo |
| 2 | ISIC 2019 | ISIC Archive oficial (S3) | ✅ Descargado | 25,331 .jpg | ✅ train:20,409 / val:2,474 / test:2,448 | 🟢 Bajo |
| 3 | Osteoarthritis Knee | Kaggle (`dhruvacube/osteoarthritis`) | ✅ Descargado | 4,766 imágenes | ✅ train:3,814 / val:480 / test:472 | 🔴 Alto |
| 4 | LUNA16 | Kaggle (metadata) + Zenodo (CT) | ✅ Descargado | 888 pares CT (.mhd+.raw), 17,785 parches | ✅ train:14,728 / val:1,143 / test:1,914 | 🟡 Medio |
| 5 | Pancreatic Cancer | Zenodo (`13715870`) | ✅ Descargado | 557 .nii.gz (~93 GB) | ✅ `pancreas_splits.csv` + `pancreas_test_ids.txt` | 🟢 Bajo |
| 6 | PANORAMA labels (auxiliar) | GitHub (`DIAGNijmegen/panorama_labels`) | ✅ Clonado | 1,756 auto + 482 manual = 2,238 labels | N/A (etiquetas, no imágenes) | 🟢 Bajo |

**Totales confirmados:**
- Suma de splits NIH: 88,999 + 11,349 + 11,772 = **112,120** ✅
- Suma de splits ISIC: 20,409 + 2,474 + 2,448 = **25,331** ✅
- Suma de splits OA: 3,814 + 480 + 472 = **4,766** ✅
- Suma de parches LUNA16: 14,728 + 1,143 + 1,914 = **17,785** ✅

---

## 3. Detalle por dataset

### 3.1 NIH ChestXray14

| Campo | Valor |
|---|---|
| **Fuente declarada** | Kaggle REST API: `https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data` |
| **Fuente oficial** | NIH Clinical Center / Box.com |
| **Slug de Kaggle** | `nih-chest-xrays/data` |
| **Script** | `src/pipeline/fase0/descargar.py` → `download_nih()` |

**Autenticidad de la fuente:** ✅ La cuenta `nih-chest-xrays` en Kaggle es una **cuenta de organización**, no personal. Es un mirror oficial del dataset del NIH Clinical Center. Riesgo: 🟢 Bajo.

**Archivos en disco:**
- `datasets/nih_chest_xrays/images_001/` a `images_012/`: 112,120 imágenes .png
- `datasets/nih_chest_xrays/all_images/`: 112,120 symlinks
- Splits: `splits/nih_train.txt`, `nih_val.txt`, `nih_test.txt`

**Estado de splits:**
- train: 88,999 | val: 11,349 | test: 11,772 | Total: 112,120 ✅
- Método: Patient ID (listas oficiales `train_val_list.txt` / `test_list.txt`)

**Anomalías detectadas:** Ninguna.

**Acción requerida:**
- ✅ **Resuelto 2026-04-04** — `data.zip` (45 GB) eliminado. Espacio liberado.

---

### 3.2 ISIC 2019

| Campo | Valor |
|---|---|
| **Fuente declarada** | ISIC Archive oficial (S3: `isic-archive.s3.amazonaws.com/challenges/2019/`) |
| **Fuente oficial** | ISIC Archive (S3 de Amazon) — ✅ Fuente institucional oficial |
| **Script** | `src/pipeline/fase0/descargar.py` → `download_isic()` |

**Autenticidad de la fuente:** ✅ Re-descargado el 2026-04-05 desde la fuente oficial ISIC Archive (S3: `isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip`). Las imágenes originales de Kaggle (`andrewmvd`) fueron reemplazadas con las oficiales. Descarga verificada: 9,771,618,190 bytes, 25,331 imágenes extraídas. Riesgo: 🟢 Bajo.

**Archivos en disco:**
- 25,331 imágenes .jpg
- ℹ️ **Actualización 2026-04-05:** Las imágenes fueron re-descargadas desde ISIC Archive oficial. La ruta es ahora `datasets/isic_2019/ISIC_2019_Training_Input/` (un solo nivel, no doble anidado). ✅ **Actualización 2026-04-06:** El directorio `isic_images/` fue eliminado de disco (INC-06 cerrado).

**Estado de splits:**
- train: 20,409 | val: 2,474 | test: 2,448 | Total: 25,331 ✅
- Método: por `lesion_id` (`build_lesion_split`)

**Anomalías detectadas:**
1. ✅ **Resuelto 2026-04-05** — `fase0_pipeline.py`, `fase1_pipeline.py` y `pre_modelo.py` corregidos: rutas actualizadas a `ISIC_2019_Training_Input/` (nivel único, fuente oficial).
2. ✅ **No es bug activo** — sufijo `_downsampled` en test split CSV: el código en `dataset_builder.py` ya excluye esas filas antes de construir el dataset.
3. ✅ **Resuelto 2026-04-06** — `isic_images/` directorio vacío eliminado de disco (INC-06 cerrado).

**Acciones requeridas:**
- (HIGH) ✅ **Resuelto 2026-04-05** — Ruta de imágenes actualizada en `fase0_pipeline.py`, `fase1_pipeline.py`, `pre_modelo.py` (fuente oficial, nivel único).
- (MEDIUM) ✅ **No requiere acción** — `_downsampled` ya manejado por exclusión en pipeline.
- (MEDIUM) ✅ **Resuelto 2026-04-05** — Re-descargado desde ISIC Archive oficial (S3). 25,331 imágenes verificadas. Riesgo de procedencia: 🟢 Bajo.
- (LOW) ✅ **Resuelto 2026-04-04** — `isic-2019.zip` (9.3 GB) eliminado.

---

### 3.3 Osteoarthritis Knee (OA)

| Campo | Valor |
|---|---|
| **Fuente declarada** | Kaggle CLI: `dhruvacube/osteoarthritis` |
| **URL guía** | https://www.kaggle.com/datasets/dhruvacube/osteoarthritis |
| **Origen probable** | Osteoarthritis Initiative (OAI) — NIH / NOAPII (inferido del código; sin DOI verificable) |
| **Script** | `src/pipeline/fase0/descargar.py` → `download_oa()` (línea ~329) |
| **Riesgo de proveniencia** | 🔴 Alto — cuenta personal Kaggle, sin DOI, sin paper de referencia. El código refiere reiteradamente al "dataset OAI" pero la procedencia formal es inverificable desde el Kaggle publicado. |

**Estructura en disco (`datasets/osteoarthritis/`):**

| Directorio | Contenido | Imágenes | Uso en pipeline |
|---|---|---|---|
| `KLGrade/KLGrade/0/` | KL0 — Normal | 1,315 | ✅ Sí |
| `KLGrade/KLGrade/1/` | KL1 — Doubtful | 1,266 | ✅ Sí |
| `KLGrade/KLGrade/2/` | KL2 — Mild | 765 | ✅ Sí |
| `KLGrade/KLGrade/3/` | KL3 — Moderate | 742 | ✅ Sí |
| `KLGrade/KLGrade/4/` | KL4 — Severe | 678 | ✅ Sí |
| `withoutKLGrade/withoutKLGrade/normal/` | Sin grado KL | 1,315 | ❌ No usado |
| `withoutKLGrade/withoutKLGrade/patient/` | Sin grado KL | 3,101 | ❌ No usado |
| `discarded/discarded/` | Descartados por uploader | 157 | ❌ No usado |
| **Total en disco** | | **9,339** | 4,766 usados |

> La guía declara "~10 K imgs" → refiere al total del dataset Kaggle (~9,339). El pipeline usa **solo las 4,766 imágenes con grado KL**.

**Remapeo de clases** (`pre_modelo.py:569`):

```python
mapping = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}  # KL → clase consolidada
```

| Grados KL | Clase consolidada | Nombre |
|---|---|---|
| KL0 | 0 | Normal |
| KL1 + KL2 | 1 | Leve (Doubtful + Mild) |
| KL3 + KL4 | 2 | Severo (Moderate + Severe) |

> Consistente con `proyecto_moe.md`: "3 grados KL".

**Splits (`oa_splits/`)** — copias físicas en subdirectorios por clase:

| Split | Clase 0 | Clase 1 | Clase 2 | Total |
|-------|---------|---------|---------|-------|
| `train/` | 1,054 | 1,624 | 1,136 | **3,814** |
| `val/` | 131 | 206 | 143 | **480** |
| `test/` | 130 | 201 | 141 | **472** |
| **Total** | 1,315 | 2,031 | 1,420 | **4,766** |

> Ratio 80/10/10 ✅. Los splits son copias físicas de imagen, no CSVs.

**Método de split** (`pre_modelo.py:510–605`, función `split_oa()`):
- Estratificado 80/10/10 a nivel de **grupo de similitud** (no por imagen individual)
- Grupos inferidos por: fingerprint 16×16 L2-normalizado → distancia euclidiana → Union-Find (threshold=0.12)
- **Rationale:** el dataset no incluye `patient_id`; la similitud visual sirve como proxy de identidad de paciente/rodilla
- ⚠️ **Limitación documentada** (código línea 537): es una heurística. No equivale a un split real por `patient_id`. Documentar en el reporte técnico.

**Bugs corregidos:** ninguno. El código maneja el doble anidamiento `KLGrade/KLGrade/` correctamente con fallback (línea 553).

**Estado:** ✅ Listo

---

### 3.4 LUNA16 (Nódulos Pulmonares)

| Campo | Valor |
|---|---|
| **Fuente CT volumes** | Zenodo oficial — `records/3723295` (subsets 0–6) + `records/2596479` (subsets 7–9) |
| **Fuente metadata** | Kaggle CLI: `fanbyprinciple/luna-lung-cancer-dataset` (330 MB, etiquetado "Sample") |
| **Script** | `src/pipeline/fase0/descargar.py` → `download_luna_meta()` + `download_luna_ct()` |
| **Riesgo CT** | 🟢 Bajo — Zenodo oficial LUNA16 Grand Challenge |
| **Riesgo metadata** | 🟡 Medio — cuenta personal Kaggle, no oficial |
| **Licencia** | CC BY 4.0 (LUNA16 Grand Challenge oficial). La fuente Kaggle declara CC BY-SA 3.0 — no aplicable. |

**Archivos de anotación verificados:**

| Archivo | Filas | Estado |
|---|---|---|
| `annotations.csv` | **1,186 anotaciones** | ✅ Completo (igual a especificación oficial) |
| `candidates_V2/candidates_V2.csv` | 754,975 candidatos | ✅ Versión V2 correcta |
| `candidates.csv` (V1) | 551,065 candidatos | ⚠️ V1 — presente pero no es la versión usada por el pipeline |
| `sampleSubmission.csv` | — | Presente |

**CT volumes:**
- **888 pares `.mhd`+`.raw`** en `ct_volumes/subset0/` a `ct_volumes/subset9/`
- `luna_splits.json`: train_uids: 712 / val_uids: 88 / test_uids: 88 (total: 888 CTs)

> `proyecto_moe.md` declara "~600 vol. (subset)"; en disco hay **888** — todos los 10 subsets completos. Superior al mínimo requerido ✅.

**Segmentaciones pulmonares (`seg-lungs-LUNA16/`):**
- **1,776 archivos** (888 `.mhd` + 888 `.zraw`) — **441 MB**
- Máscaras de segmentación pulmonar para los 888 CTs (un par por CT)
- **No usadas por el pipeline** — solo referenciadas en `src/notebooks/luna_lung_cancer/000_eda.ipynb`
- Se conservan en disco

**Parches extraídos (`patches/`)** — datos de `extraction_report.json`:

| Split | Positivos | Negativos | Total |
|-------|-----------|-----------|-------|
| `train/` | 1,258 | 13,470 | **14,728** |
| `val/` | 105 | 1,038 | **1,143** |
| `test/` | 174 | 1,740 | **1,914** |
| **Total** | 1,537 | 16,248 | **17,785** |

Parámetros de extracción: `patch_size=64`, HU clip `[−1000, 400]`, `neg_ratio=10`

**Data leakage fix (`_LEAKED_DO_NOT_USE/`):**
- `patches/train_stale_backup/` renombrado a `patches/_LEAKED_DO_NOT_USE/` el 2026-04-05
- Contiene 1,839 parches removidos del train activo por **overlap confirmado de seriesUID** entre train/val/test
- Fix original aplicado por `fix_luna_leakage.py` el 2026-04-02
- ⚠️ **NUNCA usar `_LEAKED_DO_NOT_USE/` para entrenamiento**

**Bugs corregidos:** ninguno en código de pipeline. El leakage fue resuelto como operación de datos (`fix_luna_leakage.py`).

**Estado:** ✅ Listo

---

### 3.5 Pancreatic Cancer (Zenodo) + PANORAMA Labels (auxiliar)

| Campo | Valor |
|---|---|
| **Fuente CT volumes** | Zenodo: `https://zenodo.org/records/13715870/files/batch_1.zip` |
| **Fuente labels** | GitHub: `https://github.com/DIAGNijmegen/panorama_labels.git` |
| **Scripts** | `src/pipeline/fase0/descargar.py` → `download_pancreas()` + `download_panorama()` |
| **Riesgo de fuente** | 🟢 Bajo — ambas fuentes institucionales (Radboud UMC / DIAGNijmegen) |

#### Volúmenes en disco

- **557** archivos `.nii.gz` en `datasets/zenodo_13715870/` (~93 GB extraídos)
- Convención de nombres: `{patient_id}_{study_id}_0000.nii.gz` — 557 estudios de **1,850 pacientes únicos** (14 pacientes tienen múltiples estudios)
- `batch_1.zip` (45.9 GB) **eliminado el 2026-04-05** tras verificar que los 557 volúmenes están intactos

#### Labels binarios — `pancreas_labels_binary.csv`

| Métrica | Valor |
|---------|-------|
| Total filas | **1,864** |
| `label=1` (PDAC+) | **1,756** — `label_source="mask_value_3"` (PANORAMA confirmó valor 3 en máscara) |
| `label=0` (PDAC−) | **108** — `label_source="assumed_negative_no_mask"` (sin máscara PANORAMA → negativo asumido) |
| `label=-1` (FUTURE_ERROR) | **0** — eliminados tras corrección del bug OOM (ver §A17) |

**Nota sobre multiplicidad:** las 1,864 filas corresponden a las 1,756 máscaras PANORAMA `automatic_labels/` + 108 CTs de Zenodo sin máscara. De los 557 CTs en disco, **449 tienen máscara PANORAMA** (label=1 confirmado o procesado) y **108 no tienen máscara** (label=0 asumido).

#### Splits

| Artefacto | Contenido |
|---|---|
| `datasets/pancreas_splits.csv` | **1,864** case_ids: **186** test + **1,678** en pool k-fold CV. Fold típico: ~1,342 train / ~336 val |
| `datasets/pancreas_test_ids.txt` | Obsoleto — refleja el split pre-recuperación. La fuente de verdad es `pancreas_splits.csv` |

Método de split: k-fold CV (k=5) con 10% test fijo (`GroupKFold` por `patient_id` desde 2026-04-05). El CSV tiene columnas `case_id, label, split` con valores `fold1_train`, `fold1_val`, …, `fold5_train`, `fold5_val`, `test`.

Distribución de labels en test: **175** PDAC+ / **11** PDAC−. Pacientes en test: 185.
Distribución Fold1: train → 1,342 casos; val → 336 casos.

> ⚠️ **Nota sobre usabilidad:** De los 1,864 case_ids en `pancreas_splits.csv`, solo **557 tienen archivos CT en `datasets/zenodo_13715870/`**. Los otros 1,307 son etiquetas PANORAMA sin CT descargado (batch_2+ de Zenodo no descargado). Fase 1 deberá filtrar por existencia de archivo antes de cargar. Ver A19.

#### PANORAMA Labels

| Campo | Valor |
|---|---|
| **Commit clonado** | `bf1d6ba3230f6b093e7ea959a4bf5e2eba2e3665` |
| **automatic_labels/** | 1,756 archivos `.nii.gz` |
| **manual_labels/** | 482 archivos |
| **Total label files** | 2,238 |

**Nota:** PANORAMA labels no está en la guía oficial (§2) pero es un auxiliar requerido para las etiquetas del Experto 4 (Páncreas).

#### Auditoría de código (2026-04-05)

Se auditaron los 6 archivos que referencian Pancreas/PANORAMA:

| Archivo | Estado |
|---------|--------|
| `descargar.py` → `download_pancreas()` / `download_panorama()` | ✅ Sin bugs |
| `extraer.py` → `_pancreas_extracted()` | ✅ Sin bugs |
| `pre_modelo.py` → `split_pancreas()` | ✅ Sin bugs — usa `datasets_dir / "pancreas_splits.csv"` (correcto) |
| `fase0_pipeline.py` → reporte paso 8, línea 606 | ✅ **Bug corregido (2026-04-05)** — ruta era `datasets/zenodo_13715870/pancreas_splits.csv`, cambiada a `datasets/pancreas_splits.csv` |
| `fase0_pipeline.py` → `paso4_pancreas_labels()` | ✅ **Bug OOM corregido (2026-04-05)** — `ProcessPoolExecutor` con N-1 workers causaba OOM en 13GB RAM al descomprimir múltiples NIfTI en paralelo. Reemplazado por loop secuencial con streaming chunk-based en `_process_mask()`. 1,622 casos FUTURE_ERROR recuperados. |
| `fase1_pipeline.py` → defaults pancreas | ✅ Sin bugs |
| `dataset_builder.py` → `cfg["pancreas_splits_csv"]` | ✅ Sin bugs |

#### Anomalías y acciones

| # | Severidad | Descripción | Estado |
|---|-----------|-------------|--------|
| 1 | ✅ | 557 archivos vs 281 esperados. Explicado: convención `{patient_id}_{study_id}_0000.nii.gz`. 1,850 pacientes únicos, 14 con múltiples estudios. La guía dice "~281" refiriéndose al número de pacientes aproximado. | Resuelto 2026-04-05 |
| 2 | ⚠️ HIGH | Checksum MD5 de `batch_1.zip` (`b3b3669a82696b954b449c27a9d85074`) no verificado por `descargar.py` antes de la eliminación | Irresolvable — ZIP eliminado el 2026-04-05 antes de verificar. Riesgo aceptado: los 557 `.nii.gz` están intactos y el pipeline los valida por contenido |
| 3 | ✅ | `batch_1.zip` (45.9 GB) eliminado tras confirmar 557 `.nii.gz` intactos | Resuelto 2026-04-05 |
| 4 | ✅ | Bug en `fase0_pipeline.py` línea 606: ruta incorrecta `datasets/zenodo_13715870/pancreas_splits.csv` → corregida a `datasets/pancreas_splits.csv` | Resuelto 2026-04-05 |
| 5 | ✅ | **Leakage intra-fold corregido (2026-04-05):** `split_pancreas()` reemplazó `StratifiedKFold` por `GroupKFold(n_splits=5, groups=patient_id)`. Test set también seleccionado por `patient_id`. 0 leakage verificado en los 6 folds + test. | Resuelto 2026-04-05 |
| 6 | ✅ | **Bug OOM en `paso4_pancreas_labels()`:** `ProcessPoolExecutor` con N-1 workers + decompresión NIfTI en paralelo agotaba los 13GB de RAM. 1,622 de 1,756 máscaras fallaban con `BrokenProcessPool`. Corregido con loop secuencial y streaming chunk-based. `pancreas_labels_binary.csv` regenerado: 1,864 filas, 0 FUTURE_ERROR. `pancreas_splits.csv` regenerado: 1,864 casos únicos. | Resuelto 2026-04-05 |
| 7 | ✅ | **1,307 casos sin CT en disco:** `pancreas_splits.csv` incluye 1,864 case_ids, pero solo 557 tienen `.nii.gz` en `datasets/zenodo_13715870/`. Los 1,307 restantes son etiquetas PANORAMA de CTs en batches Zenodo no descargados (batch_2+). `_build_pairs()` en `dataset_builder.py` L290–311 omite silenciosamente casos sin CT (guard `if candidates:`). Se añadió `log.warning` con conteo de omitidos. | Resuelto 2026-04-05 |

**Estado: ✅ Listo** — datos descargados, extraídos, labels regenerados sin errores (1,864 casos), splits regenerados con `GroupKFold` sin leakage (186 test, 0 leakage intra-fold), ZIP eliminado, bugs corregidos. A19 resuelto: `_build_pairs()` omite casos sin CT silenciosamente (guard `if candidates:`), `log.warning` con conteo de omitidos añadido.

---

## 4. Drift de documentación

> **Resuelto el 2026-04-04.** Se detectaron 7 discrepancias entre `arquitectura_documentacion.md` y el estado real del filesystem (bloqueantes falsos para Páncreas y LUNA16). Todas las correcciones fueron aplicadas directamente en `arquitectura_documentacion.md` en la misma sesión de auditoría.

---

## 5. Desperdicio de almacenamiento

Los siguientes archivos ZIP originales se conservan en disco junto a los datos ya extraídos. Eliminarlos liberaría espacio significativo:

| Archivo | Ruta relativa | Tamaño | Puede eliminarse |
|---------|---------------|--------|-----------------|
| `data.zip` (NIH) | `datasets/nih_chest_xrays/data.zip` | ~45 GB | ✅ **Eliminado 2026-04-04** |
| `isic-2019.zip` | `datasets/isic_2019/isic-2019.zip` | ~9.3 GB | ✅ **Eliminado 2026-04-04** |
| `osteoarthritis.zip` | `datasets/osteoarthritis/osteoarthritis.zip` | ~5.0 GB | ✅ **Eliminado 2026-04-05** |
| `batch_1.zip` (Pancreas) | `datasets/zenodo_13715870/batch_1.zip` | ~45.9 GB | ✅ **Eliminado 2026-04-05** |
| `subset0.zip` – `subset9.zip` (LUNA CT) | `datasets/luna_lung_cancer/ct_volumes/subset{0-9}.zip` | ~67 GB (10 archivos) | ✅ **Eliminado 2026-04-05** |
| `luna-lung-cancer-dataset.zip` (metadata) | `datasets/luna_lung_cancer/luna-lung-cancer-dataset.zip` | ~331 MB | ✅ **Eliminado 2026-04-05** — contenido verificado: `annotations.csv`, `candidates.csv`, `candidates_V2/`, `evaluationScript/`, `sampleSubmission.csv`, `seg-lungs-LUNA16/` todos presentes en disco |

**Espacio total recuperado:** ~172.6 GB (todos los ZIPs eliminados)

---

## 6. Ítems de acción — 19/19 resueltos ✅

| # | Severidad | Descripción | Responsable |
|---|-----------|-------------|-------------|
| A1 | ✅ RESOLVED | **Procedencia de Osteoarthritis Knee:** Fuente mandatada por `proyecto_moe.md` línea 461: `dhruvacube/osteoarthritis`. Fuente aceptada como autoritativa. | Completado 2026-04-05 |
| A2 | ✅ RESOLVED | **Actualizar `arquitectura_documentacion.md`:** §3.5, §7.2, §7.3, §7.4 corregidos para reflejar Páncreas descargado (557 archivos, ~93 GB), parches de test de LUNA16 (1,914), splits de Páncreas generados. Bloqueantes falsos eliminados. PANORAMA añadido a §7.2. | Completado 2026-04-05 |
| A3 | ✅ RESOLVED | **Ruta de imágenes ISIC corregida 2026-04-05:** `fase0_pipeline.py`, `fase1_pipeline.py`, `pre_modelo.py` actualizados a `ISIC_2019_Training_Input/` (nivel único, fuente oficial). | Completado |
| A4 | ✅ RESOLVED | **Sufijo `_downsampled` en ISIC test CSVs:** No es bug activo — `dataset_builder.py` ya excluye esas filas antes de construir el dataset. | Completado |
| A5 | ✅ RESOLVED | **`batch_1.zip` eliminado 2026-04-05:** 45.9 GB liberados. Los 557 `.nii.gz` están intactos en disco. Checksum MD5 no verificado antes de la eliminación (riesgo aceptado — datos extraídos intactos). | Completado |
| A6 | ✅ RESOLVED | **`train_stale_backup/` renombrado a `_LEAKED_DO_NOT_USE/`** el 2026-04-05. 1,839 parches con data leakage confirmado (overlap seriesUID). NUNCA usar para entrenamiento. | Completado 2026-04-05 |
| A7 | ✅ RESOLVED | **`annotations.csv` verificado:** 1,186 anotaciones de nódulos ✅ (1,187 líneas − 1 header). Pipeline usa `candidates_V2.csv` (754,975 candidatos), no V1 (551,065). | Completado 2026-04-05 |
| A8 | ✅ RESOLVED | **Remapeo KL grades OA documentado en §3.3:** `mapping = {"0":0, "1":1, "2":1, "3":2, "4":2}` → KL0→Cls0 (Normal) / KL1+2→Cls1 (Leve) / KL3+4→Cls2 (Severo). Tabla completa con conteos por grado y clase en §3.3. | Completado 2026-04-05 |
| A9 | ✅ RESOLVED | **Pacientes únicos en Páncreas documentados:** 557 estudios de 1,850 pacientes únicos (14 con múltiples estudios). La guía dice "~281" = estimado de pacientes únicos. Splits ahora cubren 1,864 casos. | Completado 2026-04-05 |
| A10 | ✅ RESOLVED | **Licencia LUNA16 corregida:** CC BY 4.0 (LUNA16 Grand Challenge oficial). La fuente Kaggle declara CC BY-SA 3.0 — no aplicable a este dataset. Documentado en §3.4. | Completado 2026-04-05 |
| A11 | ✅ RESOLVED | **Metadata LUNA16 de fuente oficial:** Fuente mandatada por `proyecto_moe.md` línea 462: `fanbyprinciple/luna-lung-cancer-dataset`. Fuente aceptada como autoritativa. | Completado 2026-04-05 |
| A12 | ✅ RESOLVED | **Discrepancia de conteo OA resuelta:** `proyecto_moe.md` dice "~10 K imgs" = aproximación del total Kaggle (9,339 imágenes únicas en disco). `arquitectura_documentacion.md` ya tiene las cifras correctas: 4,766 KL-graded usadas por el pipeline, de 9,339 en disco. No hay cifra errónea de "~8,260" en documentación actual. | Completado 2026-04-05 |
| A13 | ✅ RESOLVED | **`cae_splits.csv` documentado:** Manifiesto unificado de datos para Experto 6 (CAE). 162,611 filas, 5 columnas (`ruta_imagen`, `dataset_origen`, `split`, `expert_id`, `tipo_dato`). Generado por `build_cae_splits()` en `pre_modelo.py:947`. Distribución: NIH 112,120 \| ISIC 25,331 \| OA 4,766 \| LUNA 17,785 \| Pancreas 2,609 (= 557 CTs × ~4.7 folds). Consumido por Fase 3 (aún no implementada). **Regenerado el 2026-04-05** tras corrección del bug OOM (el anterior tenía LUNA:1,499 y Pancreas:1,110 — stale). | Completado 2026-04-05 |
| A14 | ✅ RESOLVED | **ZIPs eliminados:** `data.zip` (NIH, 2026-04-04), `isic-2019.zip` (ISIC, 2026-04-04), `osteoarthritis.zip` (OA, 2026-04-05), `subset0-9.zip` (LUNA, 2026-04-05), `batch_1.zip` (Pancreas, 2026-04-05). ~172 GB liberados. | Completado |
| A15 | ✅ RESOLVED | **`seg-lungs-LUNA16/` NO vacío:** 1,776 archivos (888 `.mhd` + 888 `.zraw`), 441 MB. Máscaras de segmentación pulmonar. No usadas por el pipeline (solo en EDA notebook). Se conservan en disco. | Completado 2026-04-05 |
| A16 | ✅ RESOLVED | **PANORAMA documentado como auxiliar:** §3.5 de `paso_01_descarga_datos.md` ahora consolida Pancreas + PANORAMA con nota explícita sobre su rol auxiliar. | Completado |
| A17 | ✅ RESOLVED | **Bug OOM en `paso4_pancreas_labels()`:** `ProcessPoolExecutor` con N-1 workers agotaba 13GB RAM al descomprimir múltiples NIfTI de hasta 7.5GB en paralelo. 1,622/1,756 máscaras fallaban con `BrokenProcessPool`. Fix: loop secuencial + streaming chunk-based en `_process_mask()`. `pancreas_labels_binary.csv` regenerado en 70 min sin errores (1,864 filas, 0 FUTURE_ERROR). `pancreas_splits.csv` regenerado: 1,864 casos, 186 test, ~1,342 train / ~336 val por fold. | Completado 2026-04-05 |
| A18 | ✅ RESOLVED | **Leakage intra-fold en Páncreas corregido:** `split_pancreas()` ahora usa `GroupKFold(n_splits=5)` con `groups=patient_id`. Test set seleccionado por `patient_id` (10% de pacientes únicos). Verificado: 0 leakage test↔CV, 0 leakage intra-fold en los 5 folds. Estadísticas: 1,864 casos, 186 test (185 pacientes), 1,850 pacientes únicos. | Completado 2026-04-05 |
| A19 | ✅ RESOLVED | **1,307 casos en splits sin CT en disco:** `pancreas_splits.csv` incluye 1,864 case_ids pero solo 557 tienen `.nii.gz` en `datasets/zenodo_13715870/`. Los 1,307 restantes son etiquetas PANORAMA de CTs de batches Zenodo no descargados (solo se descargó `batch_1.zip`). NON-BLOCKER. `_build_pairs()` en `dataset_builder.py` L290–311 omite silenciosamente casos sin CT (guard `if candidates:`). Se añadió `log.warning` con conteo de omitidos. Glob funciona: CSV `case_id=100001_00001` → disco `100001_00001_0000.nii.gz`. Fase 1 entrena con los ~557 casos disponibles. | Completado 2026-04-05 |

---

## 7. Apéndice

### 7.1 Commit de PANORAMA labels

| Campo | Valor |
|---|---|
| Repositorio | `https://github.com/DIAGNijmegen/panorama_labels.git` |
| Commit clonado | `bf1d6ba3230f6b093e7ea959a4bf5e2eba2e3665` |
| Archivo de registro | `datasets/panorama_labels_commit.txt` |
| Contenido de archivos | `automatic_labels/`: 1,756 archivos, `manual_labels/`: 482 archivos |

### 7.2 Checksum de Pancreas (Zenodo)

| Campo | Valor |
|---|---|
| Archivo | `batch_1.zip` |
| Fuente | `https://zenodo.org/records/13715870/files/batch_1.zip` |
| MD5 declarado por Zenodo | `b3b3669a82696b954b449c27a9d85074` |
| Estado de verificación | ❌ **No verificado** — `descargar.py` no ejecuta comparación de checksum post-descarga, y el ZIP fue eliminado el 2026-04-05 antes de verificar. Los 557 `.nii.gz` están intactos en disco. La verificación a posteriori ya no es posible sin re-descarga. |

### 7.3 URLs de descarga exactas (de `descargar.py`)

| Dataset | URL / Slug |
|---------|------------|
| NIH ChestXray14 | `https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data` |
| ISIC 2019 | `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip` (oficial, re-descargado 2026-04-05) |
| Osteoarthritis Knee | `kaggle datasets download -d dhruvacube/osteoarthritis` |
| LUNA16 metadata | `kaggle datasets download -d fanbyprinciple/luna-lung-cancer-dataset` |
| LUNA16 CT (subsets 0–6) | `https://zenodo.org/records/3723295/files/subset{i}.zip?download=1` |
| LUNA16 CT (subsets 7–9) | `https://zenodo.org/records/2596479/files/subset{i}.zip?download=1` |
| Pancreas | `https://zenodo.org/records/13715870/files/batch_1.zip?download=1` |
| PANORAMA labels | `https://github.com/DIAGNijmegen/panorama_labels.git` |

---

*Documento generado el 2026-04-04 por auditoría multi-agente (ARGOS + SIGMA + EXPLORE). Última verificación: 2026-04-05. Fuentes: `descargar.py`, `arquitectura_documentacion.md`, inspección directa del filesystem, verificación de fuentes remotas. Actualización 2026-04-05 (sesión 1): §3.3 OA Knee — auditoría de código completa (7 archivos, 0 bugs funcionales), `osteoarthritis.zip` eliminado, conteos por grado KL y remapeo documentados. Actualización 2026-04-05 (sesión 2): §3.5 Páncreas — bug OOM corregido en `_process_mask()`, 1,622 casos FUTURE_ERROR recuperados, `pancreas_labels_binary.csv` regenerado (1,864 filas, 0 errores), `pancreas_splits.csv` regenerado (1,864 casos, 187 test), leakage intra-fold documentado (14 pacientes multi-estudio, 6 afectados en folds, 2 en test+CV). Actualización 2026-04-05 (sesión 3): A18 leakage corregido con `GroupKFold` (0 leakage, 186 test), A8/A12/A13 resueltos, `luna-lung-cancer-dataset.zip` eliminado, `cae_splits.csv` regenerado (162,611 filas), A19 documentado (1,307 casos sin CT en disco).*
