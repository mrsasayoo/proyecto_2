#!/usr/bin/env bash
# ============================================================================
# export_training_data.sh — Comprime SOLO los datos de entrenamiento del MoE
# ============================================================================
#
# Uso:
#   nohup bash docs/exportaciones/export_training_data.sh \
#       > docs/exportaciones/export.log 2>&1 &
#
# Requisitos:
#   - p7zip-full instalado (7z command)
#   - ~301 GB de espacio libre (153 GB datos + 148 GB destino .7z)
#
# Destino: docs/exportaciones/*.7z
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2"
EXPORT_DIR="${PROJECT_ROOT}/docs/exportaciones"
DS="${PROJECT_ROOT}/datasets"

mkdir -p "${EXPORT_DIR}"

echo "==========================================="
echo "  EXPORT — Datos mínimos de entrenamiento"
echo "  Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Destino: ${EXPORT_DIR}/"
echo "==========================================="
echo ""

# Verificar que 7z está instalado
if ! command -v 7z &>/dev/null; then
    echo "ERROR: 7z no encontrado. Instala con: sudo apt install p7zip-full"
    exit 1
fi

# ============================================================================
# 1. NIH ChestXray14
# ============================================================================
# Archivos requeridos por dataloader_expert1.py (líneas 396-438):
#   - Data_Entry_2017.csv (labels)
#   - images_001..012/ (imágenes reales apuntadas por symlinks)
#   - all_images/ (directorio de symlinks)
#   - splits/ (nih_train_list.txt, nih_val_list.txt, nih_test_list.txt)
# Excluir: data.zip (42 GB), PDFs, BBox_List_2017.csv, test_list.txt, train_val_list.txt
# ============================================================================
echo "[1/6] NIH ChestXray14 (~43 GB sin comprimir)..."
echo "  Inicio: $(date '+%H:%M:%S')"

nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    -snl \
    "${EXPORT_DIR}/nih_chest_xrays.7z" \
    "${DS}/nih_chest_xrays/all_images/" \
    "${DS}/nih_chest_xrays/images_001/" \
    "${DS}/nih_chest_xrays/images_002/" \
    "${DS}/nih_chest_xrays/images_003/" \
    "${DS}/nih_chest_xrays/images_004/" \
    "${DS}/nih_chest_xrays/images_005/" \
    "${DS}/nih_chest_xrays/images_006/" \
    "${DS}/nih_chest_xrays/images_007/" \
    "${DS}/nih_chest_xrays/images_008/" \
    "${DS}/nih_chest_xrays/images_009/" \
    "${DS}/nih_chest_xrays/images_010/" \
    "${DS}/nih_chest_xrays/images_011/" \
    "${DS}/nih_chest_xrays/images_012/" \
    "${DS}/nih_chest_xrays/Data_Entry_2017.csv" \
    "${DS}/nih_chest_xrays/splits/" \
    -xr!'*.zip' \
    -xr!'*.pdf'

echo "  Terminado: $(date '+%H:%M:%S')"
echo ""

# ============================================================================
# 2. ISIC 2019
# ============================================================================
# Archivos requeridos por dataloader_expert2.py (líneas 53-59):
#   - ISIC_2019_Training_Input_preprocessed/ (502 MB, cache preferido)
#   - splits/{isic_train.csv, isic_val.csv, isic_test.csv}
#   - ISIC_2019_Training_GroundTruth.csv (referencia)
#   - ISIC_2019_Training_Metadata.csv (lesion_id para rebuild de splits)
# Excluir: ISIC_2019_Training_Input/ (9.2 GB originales), isic_preprocesado.zip
# ============================================================================
echo "[2/6] ISIC 2019 (~505 MB sin comprimir)..."
echo "  Inicio: $(date '+%H:%M:%S')"

nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/isic_2019.7z" \
    "${DS}/isic_2019/ISIC_2019_Training_Input_preprocessed/" \
    "${DS}/isic_2019/splits/" \
    "${DS}/isic_2019/ISIC_2019_Training_GroundTruth.csv" \
    "${DS}/isic_2019/ISIC_2019_Training_Metadata.csv" \
    -xr!'*.zip'

echo "  Terminado: $(date '+%H:%M:%S')"
echo ""

# ============================================================================
# 3. Osteoarthritis Knee
# ============================================================================
# Archivos requeridos por dataloader_expert_oa.py (línea 49):
#   - oa_splits/{train,val,test}/{0,1,2,3,4}/ (imágenes JPG/PNG)
#   - Labels implícitos en estructura de carpetas (no CSVs)
# Excluir: osteoarthritis.zip, discarded/, KLGrade/, withoutKLGrade/
# ============================================================================
echo "[3/6] Osteoarthritis Knee (~161 MB sin comprimir)..."
echo "  Inicio: $(date '+%H:%M:%S')"

nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/osteoarthritis.7z" \
    "${DS}/osteoarthritis/oa_splits/" \
    -xr!'*.zip'

echo "  Terminado: $(date '+%H:%M:%S')"
echo ""

# ============================================================================
# 4. LUNA16
# ============================================================================
# Archivos requeridos por dataloader_expert3.py (líneas 47-56, 99, 185):
#   - patches/{train,val,test}/candidate_*.npy (17.2 GB total)
#   - candidates_V2/candidates_V2.csv (69 MB, label map)
# Excluir: patches/train_aug/ (18 GB), ct_volumes/, seg-lungs-LUNA16/,
#          evaluationScript/, luna zip, annotations.csv, candidates.csv V1,
#          patches/*.{json,txt,md,csv,npy} (reports/metadata)
# ============================================================================
echo "[4/6] LUNA16 (~17.3 GB sin comprimir)..."
echo "  Inicio: $(date '+%H:%M:%S')"

# Paso 4a: Comprimir parches .npy de train/val/test (solo .npy, sin metadata)
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/luna16.7z" \
    "${DS}/luna_lung_cancer/patches/train/candidate_*.npy" \
    "${DS}/luna_lung_cancer/patches/val/candidate_*.npy" \
    "${DS}/luna_lung_cancer/patches/test/candidate_*.npy" \
    -xr!'*.zip'

# Paso 4b: Agregar candidates_V2.csv (por separado para evitar wildcard conflicts)
nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    "${EXPORT_DIR}/luna16.7z" \
    "${DS}/luna_lung_cancer/candidates_V2/candidates_V2.csv"

echo "  Terminado: $(date '+%H:%M:%S')"
echo ""

# ============================================================================
# 5. Páncreas (Zenodo 13715870)
# ============================================================================
# Archivos requeridos por dataloader_expert4.py (líneas 44-47):
#   - zenodo_13715870/*.nii.gz (raíz, ~1,123 volúmenes, ~92 GB)
#   - zenodo_13715870/preprocessed/*.npy (230 MB, legacy/bonus)
#   - pancreas_splits.csv (224 KB)
#   - pancreas_labels_binary.csv (88 KB)
# Excluir: batch_3/ (46 GB), batch_4/ (44 GB), .gitkeep
# ============================================================================
echo "[5/6] Páncreas (~92 GB sin comprimir)..."
echo "  Inicio: $(date '+%H:%M:%S')"

nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/pancreas.7z" \
    "${DS}/zenodo_13715870/"*.nii.gz \
    "${DS}/zenodo_13715870/preprocessed/" \
    "${DS}/pancreas_splits.csv" \
    "${DS}/pancreas_labels_binary.csv" \
    -xr!'*.zip' \
    -x!"${DS}/zenodo_13715870/batch_3" \
    -x!"${DS}/zenodo_13715870/batch_4" \
    -xr!'.gitkeep'

echo "  Terminado: $(date '+%H:%M:%S')"
echo ""

# ============================================================================
# 6. CAE splits (solo CSV de referencia cruzada)
# ============================================================================
# Archivo requerido por dataloader_cae.py (línea 19) / cae.py (línea 75):
#   - cae_splits.csv (12 MB, 159,711 filas de paths relativos)
# Los datos reales están en los 5 archivos .7z anteriores
# ============================================================================
echo "[6/6] CAE splits (~12 MB sin comprimir)..."
echo "  Inicio: $(date '+%H:%M:%S')"

nice -n 19 ionice -c3 7z a \
    -t7z \
    -mx=9 \
    -mmt=4 \
    -ms=on \
    -m0=lzma2 \
    "${EXPORT_DIR}/cae_splits.7z" \
    "${DS}/cae_splits.csv"

echo "  Terminado: $(date '+%H:%M:%S')"
echo ""

# ============================================================================
# Resumen final
# ============================================================================
echo "==========================================="
echo "  EXPORT COMPLETADO: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="
echo ""
echo "Archivos generados:"
echo "-------------------"
ls -lh "${EXPORT_DIR}/"*.7z 2>/dev/null || echo "  (ningún .7z encontrado)"
echo ""
echo "Tamaño total:"
du -shc "${EXPORT_DIR}/"*.7z 2>/dev/null | tail -1 || echo "  N/A"
echo ""
echo "Para verificar integridad:"
echo "  7z t ${EXPORT_DIR}/nih_chest_xrays.7z"
echo "  7z t ${EXPORT_DIR}/isic_2019.7z"
echo "  7z t ${EXPORT_DIR}/osteoarthritis.7z"
echo "  7z t ${EXPORT_DIR}/luna16.7z"
echo "  7z t ${EXPORT_DIR}/pancreas.7z"
echo "  7z t ${EXPORT_DIR}/cae_splits.7z"
