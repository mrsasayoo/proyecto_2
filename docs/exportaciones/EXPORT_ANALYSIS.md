# Análisis de Exportación — Datos Mínimos para Entrenamiento MoE

> Generado por ATLAS Data Engineering Agent
> Fecha: 2026-04-14
> Proyecto: Sistema MoE Médico — 6 Expertos

---

## 1. Análisis por Dataloader — Archivos Requeridos en Entrenamiento

---

### 1.1 Expert 1 — NIH ChestXray14 (14 patologías, multilabel)

**Dataloader:** `src/pipeline/fase2/dataloader_expert1.py`
**Dataset class:** `src/pipeline/datasets/chest.py` → `ChestXray14Dataset`

| Archivo / Ruta | Líneas del código | Propósito |
|---|---|---|
| `datasets/nih_chest_xrays/Data_Entry_2017.csv` | `dataloader_expert1.py:399`, `chest.py:105` | Labels de 14 patologías (Finding Labels) + Patient ID |
| `datasets/nih_chest_xrays/all_images/` (symlinks → `images_001..012/images/`) | `dataloader_expert1.py:404`, `chest.py:95,228,322` | 112,120 imágenes PNG via symlinks |
| `datasets/nih_chest_xrays/splits/nih_train_list.txt` | `dataloader_expert1.py:411` | Nombres de imágenes del split train |
| `datasets/nih_chest_xrays/splits/nih_val_list.txt` | `dataloader_expert1.py:421` | Nombres de imágenes del split val |
| `datasets/nih_chest_xrays/splits/nih_test_list.txt` | `dataloader_expert1.py:431` | Nombres de imágenes del split test |

**Nota:** `all_images/` contiene symlinks relativos (`../images_001/images/...`). Para exportar, debemos incluir los directorios `images_001/` a `images_012/` (las imágenes reales) + la carpeta `all_images/` (los symlinks).

**EXCLUIR:** `data.zip` (42 GB), PDFs (`ARXIV_V5_CHESTXRAY.pdf`, `FAQ_CHESTXRAY.pdf`, `LOG_CHESTXRAY.pdf`, `README_CHESTXRAY.pdf`), `BBox_List_2017.csv` (solo para heatmap debugging, no entrenamiento), `test_list.txt` / `train_val_list.txt` (splits oficiales originales; el pipeline usa los 3 splits propios en `splits/`).

---

### 1.2 Expert 2 — ISIC 2019 (dermoscopía, 8 clases)

**Dataloader:** `src/pipeline/fase2/dataloader_expert2.py`
**Dataset class:** `src/pipeline/datasets/isic.py` → `ISICDataset`

| Archivo / Ruta | Líneas del código | Propósito |
|---|---|---|
| `datasets/isic_2019/ISIC_2019_Training_Input_preprocessed/` | `dataloader_expert2.py:55-56`, `isic.py:557-563` | 25,332 JPGs preprocesados (DullRazor + resize) — **cache preferido** |
| `datasets/isic_2019/ISIC_2019_Training_Input/` | `dataloader_expert2.py:53`, `isic.py:567-569` | Imágenes originales — **fallback** si cache no existe |
| `datasets/isic_2019/splits/isic_train.csv` | `dataloader_expert2.py:57` | Split train (image, label_idx, label_name, lesion_id) |
| `datasets/isic_2019/splits/isic_val.csv` | `dataloader_expert2.py:58` | Split val |
| `datasets/isic_2019/splits/isic_test.csv` | `dataloader_expert2.py:59` | Split test |
| `datasets/isic_2019/ISIC_2019_Training_GroundTruth.csv` | No cargado directamente por dataloader | Ground truth de referencia (usado en Fase 0 para build_lesion_split) |
| `datasets/isic_2019/ISIC_2019_Training_Metadata.csv` | No cargado directamente por dataloader | Metadata con lesion_id (usado en Fase 0) |

**Decisión:** El dataloader prefiere `_preprocessed/` (502 MB) sobre las originales (9.2 GB). Para máxima compresión, exportamos **solo el preprocessed**. Las originales son redundantes si el cache existe. Si se quisiera re-preprocessing, se re-descargan.

**EXCLUIR:** `isic_preprocesado.zip` (9.6 GB), `ISIC_2019_Training_Input/` (9.2 GB, redundante con preprocessed).

---

### 1.3 Expert OA — Osteoarthritis Knee (5 clases KL 0-4)

**Dataloader:** `src/pipeline/fase2/dataloader_expert_oa.py`
**Dataset class:** `src/pipeline/datasets/osteoarthritis.py` → `OAKneeDataset`

| Archivo / Ruta | Líneas del código | Propósito |
|---|---|---|
| `datasets/osteoarthritis/oa_splits/train/{0,1,2,3,4}/` | `dataloader_expert_oa.py:49,96-101`, `osteoarthritis.py:76,84,94` | 3,841 imágenes JPG/PNG de train organizadas por clase KL |
| `datasets/osteoarthritis/oa_splits/val/{0,1,2,3,4}/` | `dataloader_expert_oa.py:103-108` | 466 imágenes de validación |
| `datasets/osteoarthritis/oa_splits/test/{0,1,2,3,4}/` | `dataloader_expert_oa.py:110-115` | 459 imágenes de test |

**Nota:** No necesita CSVs separados — los labels se infieren de los nombres de carpeta (0-4).

**EXCLUIR:** `osteoarthritis.zip` (9.9 GB), `discarded/`, `KLGrade/`, `withoutKLGrade/`.

---

### 1.4 Expert 3 — LUNA16 (nódulos pulmonares 3D)

**Dataloader:** `src/pipeline/fase2/dataloader_expert3.py`
**Dataset class:** `LUNA16ExpertDataset` (en `dataloader_expert3.py:61-146`)

| Archivo / Ruta | Líneas del código | Propósito |
|---|---|---|
| `datasets/luna_lung_cancer/patches/train/candidate_*.npy` | `dataloader_expert3.py:99,132,242` | 13,880 parches 3D [64,64,64] entrenamiento |
| `datasets/luna_lung_cancer/patches/val/candidate_*.npy` | `dataloader_expert3.py:249` | 1,156 parches validación |
| `datasets/luna_lung_cancer/patches/test/candidate_*.npy` | `dataloader_expert3.py:256` | 2,014 parches test |
| `datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv` | `dataloader_expert3.py:48-54,185-186` | 754K filas → label map {index: class} |

**Nota:** El dataloader escanea `patches/{split}/candidate_*.npy` en disco y busca su label en `candidates_V2.csv` por index. **No necesita** `luna_splits.json` (ese es para Fase 0), ni `annotations.csv`, ni `ct_volumes/`, ni `seg-lungs-LUNA16/`, ni `evaluationScript/`.

**EXCLUIR:** `luna-lung-cancer-dataset.zip` (331 MB), `ct_volumes/`, `seg-lungs-LUNA16/`, `evaluationScript/`, `sampleSubmission.csv`, `candidates.csv` (V1), `annotations.csv`, `patches/train_aug/` (18 GB, generado aparte y no usado por el dataloader estándar), `patches/*.json`, `patches/*.txt`, `patches/*.md`, `patches/*.npy` (global_mean), `patches/*.csv` (manifest).

---

### 1.5 Expert 4 — Páncreas CT 3D (ResNet 3D / R3D-18)

**Dataloader:** `src/pipeline/fase2/dataloader_expert4.py`
**Dataset class:** `src/pipeline/datasets/pancreas.py` → `PancreasDataset`

| Archivo / Ruta | Líneas del código | Propósito |
|---|---|---|
| `datasets/pancreas_splits.csv` | `dataloader_expert4.py:44,72-73` | Mapeo case_id → label → split |
| `datasets/pancreas_labels_binary.csv` | `dataloader_expert4.py:45,78-79` | Fallback: labels binarias PDAC+/PDAC- |
| `datasets/zenodo_13715870/*.nii.gz` (raíz) | `dataloader_expert4.py:47,103,110`, `pancreas.py:664` | ~1,123 volúmenes NIfTI (el expert los lee directamente con SimpleITK) |

**Decisión CRÍTICA:** El expert4 dataloader lee los `.nii.gz` crudos directamente (NO los `.npy` preprocesados). Los preprocessed `.npy` en `zenodo_13715870/preprocessed/` son un fallback legacy (`_LegacyPancreasDataset`). Para el entrenamiento del Expert 4, **necesitamos todos los `.nii.gz`**.

**EXCLUIR:** `batch_3/` (46 GB), `batch_4/` (44 GB) — estos son datos descargados sin procesar; los `.nii.gz` en la raíz son los mismos volúmenes ya listos. `preprocessed/` se **incluye** solo como bonus (230 MB, útil para re-runs rápidos). `.gitkeep`.

---

### 1.6 Expert 5 / CAE — Autoencoder Multimodal (Fase 3)

**Dataloader:** `src/pipeline/fase3/dataloader_cae.py`
**Dataset class:** `src/pipeline/datasets/cae.py` → `MultimodalCAEDataset`

| Archivo / Ruta | Líneas del código | Propósito |
|---|---|---|
| `datasets/cae_splits.csv` | `dataloader_cae.py:19,38,44`, `cae.py:75` | 159,711 filas: ruta_imagen, dataset_origen, split, expert_id, tipo_dato |

**Nota:** El CAE **no tiene datos propios** — hace referencia cruzada a los archivos de los 5 datasets anteriores vía la columna `ruta_imagen` (paths relativos desde project_root). Por eso, el archivo `cae_splits.csv` es suficiente; los datos reales ya están incluidos en las exportaciones de los otros 5 datasets.

---

## 2. Mapeo Completo de Contenido Mínimo

| Dataset | Datos transformados | Labels/Splits | Tamaño estimado (sin comprimir) |
|---|---|---|---|
| **NIH** | `images_001..012/` (43 GB real) + `all_images/` (symlinks 3.7 MB) | `Data_Entry_2017.csv` (7.5 MB) + `splits/` (1.9 MB) | **~43 GB** |
| **ISIC** | `ISIC_2019_Training_Input_preprocessed/` (502 MB) | `splits/` (816 KB) + `GroundTruth.csv` (1.3 MB) + `Metadata.csv` (1.2 MB) | **~505 MB** |
| **OA** | `oa_splits/{train,val,test}/{0..4}/` (161 MB) | (implícito en estructura de carpetas) | **~161 MB** |
| **LUNA16** | `patches/{train,val,test}/candidate_*.npy` (17.2 GB) | `candidates_V2/candidates_V2.csv` (69 MB) | **~17.3 GB** |
| **Páncreas** | `zenodo_13715870/*.nii.gz` (raíz, ~92 GB) + `preprocessed/` (230 MB) | `pancreas_splits.csv` (224 KB) + `pancreas_labels_binary.csv` (88 KB) | **~92 GB** |
| **CAE** | (referencia cruzada a los anteriores) | `cae_splits.csv` (12 MB) | **~12 MB** |

### Espacio Total Estimado (sin comprimir): ~153 GB
### Espacio mínimo requerido para `.7z` comprimidos: ~80–100 GB (estimado)

> **Nota:** Las imágenes médicas PNG/JPG tienen baja redundancia (ya comprimidas). Los `.npy` y `.nii.gz` comprimen ~15-30%. Los CSV comprimen >90%.

---

## 3. Comandos 7z

```bash
#!/usr/bin/env bash
# ============================================================================
# export_training_data.sh — Comprime SOLO los datos de entrenamiento del MoE
# ============================================================================
# Uso: nohup bash docs/exportaciones/export_training_data.sh &
# Log: docs/exportaciones/export.log
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2"
EXPORT_DIR="${PROJECT_ROOT}/docs/exportaciones"

mkdir -p "${EXPORT_DIR}"

echo "==========================================="
echo "  EXPORT — Datos mínimos de entrenamiento"
echo "  Inicio: $(date)"
echo "==========================================="

# ── 1. NIH ChestXray14 ────────────────────────────────────────────────
# Incluye: images_001..012 (reales), all_images (symlinks), Data_Entry CSV, splits
# Excluye: data.zip, PDFs, BBox_List, test_list/train_val_list oficiales
echo ""
echo "[1/6] NIH ChestXray14 (~43 GB sin comprimir)..."
echo "  Inicio: $(date)"
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/nih_chest_xrays.7z" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/all_images/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_001/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_002/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_003/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_004/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_005/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_006/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_007/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_008/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_009/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_010/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_011/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/images_012/" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/Data_Entry_2017.csv" \
    "${PROJECT_ROOT}/datasets/nih_chest_xrays/splits/" \
    -xr!'*.zip' \
    -xr!'*.pdf'
echo "  Fin: $(date)"

# ── 2. ISIC 2019 ──────────────────────────────────────────────────────
# Incluye: preprocessed (502 MB), splits, GroundTruth, Metadata
# Excluye: Training_Input originals (9.2 GB), isic_preprocesado.zip (9.6 GB)
echo ""
echo "[2/6] ISIC 2019 (~505 MB sin comprimir)..."
echo "  Inicio: $(date)"
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/isic_2019.7z" \
    "${PROJECT_ROOT}/datasets/isic_2019/ISIC_2019_Training_Input_preprocessed/" \
    "${PROJECT_ROOT}/datasets/isic_2019/splits/" \
    "${PROJECT_ROOT}/datasets/isic_2019/ISIC_2019_Training_GroundTruth.csv" \
    "${PROJECT_ROOT}/datasets/isic_2019/ISIC_2019_Training_Metadata.csv" \
    -xr!'*.zip'
echo "  Fin: $(date)"

# ── 3. Osteoarthritis Knee ────────────────────────────────────────────
# Incluye: oa_splits/{train,val,test}/{0,1,2,3,4}/ (161 MB)
# Excluye: osteoarthritis.zip, discarded/, KLGrade/, withoutKLGrade/
echo ""
echo "[3/6] Osteoarthritis Knee (~161 MB sin comprimir)..."
echo "  Inicio: $(date)"
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/osteoarthritis.7z" \
    "${PROJECT_ROOT}/datasets/osteoarthritis/oa_splits/" \
    -xr!'*.zip'
echo "  Fin: $(date)"

# ── 4. LUNA16 ──────────────────────────────────────────────────────────
# Incluye: patches/{train,val,test}/ (solo .npy), candidates_V2.csv
# Excluye: patches/train_aug/ (18 GB), ct_volumes/, seg-lungs-LUNA16/,
#          evaluationScript/, luna zip, annotations.csv, candidates.csv (V1),
#          patches/*.json *.txt *.md *.csv *.npy (metadata/reports)
echo ""
echo "[4/6] LUNA16 (~17.3 GB sin comprimir)..."
echo "  Inicio: $(date)"
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/luna16.7z" \
    "${PROJECT_ROOT}/datasets/luna_lung_cancer/patches/train/" \
    "${PROJECT_ROOT}/datasets/luna_lung_cancer/patches/val/" \
    "${PROJECT_ROOT}/datasets/luna_lung_cancer/patches/test/" \
    "${PROJECT_ROOT}/datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv" \
    -xr!'*.zip' \
    -xr!'*.json' \
    -xr!'*.txt' \
    -xr!'*.md' \
    -xr!'*.csv' \
    -x!"${PROJECT_ROOT}/datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv"
echo "  Fin: $(date)"

# NOTA: El -xr!'*.csv' excluye CSVs dentro de patches/ (manifests), pero 
# también excluiría candidates_V2.csv que se añade explícitamente. 
# 7z procesa inclusiones antes que exclusiones para archivos listados 
# explícitamente, pero por seguridad usamos un comando separado:

# Corrección: Agregamos candidates_V2.csv por separado para evitar conflicto
# con la exclusión global de *.csv
echo "  Agregando candidates_V2.csv..."
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    "${EXPORT_DIR}/luna16.7z" \
    "${PROJECT_ROOT}/datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv"
echo "  Fin luna16 completo: $(date)"

# ── 5. Páncreas (Zenodo) ──────────────────────────────────────────────
# Incluye: zenodo_13715870/*.nii.gz (raíz) + preprocessed/*.npy + CSVs
# Excluye: batch_3/ (46 GB), batch_4/ (44 GB), .gitkeep
echo ""
echo "[5/6] Páncreas (~92 GB sin comprimir)..."
echo "  Inicio: $(date)"
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/pancreas.7z" \
    "${PROJECT_ROOT}/datasets/zenodo_13715870/*.nii.gz" \
    "${PROJECT_ROOT}/datasets/zenodo_13715870/preprocessed/" \
    "${PROJECT_ROOT}/datasets/pancreas_splits.csv" \
    "${PROJECT_ROOT}/datasets/pancreas_labels_binary.csv" \
    -xr!'*.zip' \
    -x!"${PROJECT_ROOT}/datasets/zenodo_13715870/batch_3" \
    -x!"${PROJECT_ROOT}/datasets/zenodo_13715870/batch_4"
echo "  Fin: $(date)"

# ── 6. CAE (splits CSV solamente) ─────────────────────────────────────
# Solo el CSV de referencia cruzada; los datos están en los 5 archivos anteriores
echo ""
echo "[6/6] CAE splits (~12 MB sin comprimir)..."
echo "  Inicio: $(date)"
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/cae_splits.7z" \
    "${PROJECT_ROOT}/datasets/cae_splits.csv"
echo "  Fin: $(date)"

echo ""
echo "==========================================="
echo "  EXPORT COMPLETADO: $(date)"
echo "  Archivos generados en: ${EXPORT_DIR}/"
echo "==========================================="
ls -lh "${EXPORT_DIR}/"*.7z 2>/dev/null || echo "  (ningún .7z encontrado — revisa errores)"
```

---

## 4. Tamaños Estimados por Archivo Comprimido

| Archivo `.7z` | Datos sin comprimir | Compresión esperada | Tamaño `.7z` estimado |
|---|---|---|---|
| `nih_chest_xrays.7z` | ~43 GB (PNG ya comprimidas) | ~5-8% reducción | **~40-41 GB** |
| `isic_2019.7z` | ~505 MB (JPG preprocesados) | ~5-10% reducción | **~455-480 MB** |
| `osteoarthritis.7z` | ~161 MB (JPG/PNG) | ~5-10% reducción | **~145-153 MB** |
| `luna16.7z` | ~17.3 GB (NPY float32) | ~20-30% reducción | **~12-14 GB** |
| `pancreas.7z` | ~92 GB (NIfTI comprimido) | ~3-8% reducción | **~85-89 GB** |
| `cae_splits.7z` | ~12 MB (CSV) | ~90%+ reducción | **~1-2 MB** |
| **TOTAL** | **~153 GB** | | **~138-148 GB** |

---

## 5. Espacio Mínimo Requerido

- **Datos transformados (sin comprimir):** ~153 GB
- **Archivos .7z estimados:** ~138-148 GB
- **Espacio libre necesario durante compresión:** ~153 GB (originales) + ~148 GB (destino) = **~301 GB**
- **Espacio final tras borrar originales:** ~148 GB

### Lo que se EXCLUYE (ahorro):
| Excluido | Tamaño |
|---|---|
| `data.zip` (NIH) | 42 GB |
| `isic_preprocesado.zip` | 9.6 GB |
| `ISIC_2019_Training_Input/` (originales) | 9.2 GB |
| `osteoarthritis.zip` | 9.9 GB |
| `luna-lung-cancer-dataset.zip` | 331 MB |
| `patches/train_aug/` | 18 GB |
| `ct_volumes/`, `seg-lungs-LUNA16/`, `evaluationScript/` | variable |
| `batch_3/` + `batch_4/` (páncreas) | 90 GB |
| PDFs, reports, metadata no esencial | ~few MB |
| **Total excluido** | **~179+ GB** |
