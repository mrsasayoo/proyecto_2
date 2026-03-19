#!/usr/bin/env bash
# ==============================================================================
#  setup_datasets_fast.sh
#  Descarga todos los datasets en paralelo y los prepara para FASE 0.
#  Modo RÁPIDO: extracción a máxima velocidad, sin límites de RAM.
#
#  Uso:
#    cd /ruta/del/repo
#    bash scripts/setup_datasets_fast.sh
#
#  Prerequisito:
#    ~/.kaggle/kaggle.json — API key de Kaggle
#    (https://www.kaggle.com/settings → API → Create New Token)
# ==============================================================================
set -euo pipefail

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
ts() { date '+%H:%M:%S'; }
log_info()  { echo -e "$(ts) ${GREEN}[INFO]${NC}   $*"; }
log_warn()  { echo -e "$(ts) ${YELLOW}[WARN]${NC}   $*"; }
log_error() { echo -e "$(ts) ${RED}[ERROR]${NC}  $*" >&2; }
log_step()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${NC}"; \
              echo -e "${BOLD}${BLUE}  $*${NC}"; \
              echo -e "${BOLD}${BLUE}══════════════════════════════════════${NC}\n"; }

# ── Rutas (relativas al repo root) ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"
LOG_FILE="$DATASETS_DIR/setup_fast.log"

mkdir -p "$DATASETS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
log_step "Setup RÁPIDO iniciado — $(date)"
log_info "Repo: $REPO_ROOT"
log_info "Datasets: $DATASETS_DIR"

# ── Verificar prerequisitos ───────────────────────────────────────────────────
log_step "Verificando prerequisitos"
for cmd in python3 unzip git awk wget; do
    if command -v "$cmd" &>/dev/null; then
        log_info "$cmd ✓ ($(command -v "$cmd"))"
    else
        log_error "'$cmd' no encontrado. Instálalo antes de continuar."
        exit 1
    fi
done

if ! command -v kaggle &>/dev/null; then
    log_warn "kaggle CLI no encontrado. Instalando vía pip..."
    pip install --quiet kaggle || {
        log_error "No se pudo instalar kaggle. Instala manualmente: pip install kaggle"
        exit 1
    }
fi
log_info "kaggle ✓ ($(kaggle --version 2>/dev/null || echo 'versión desconocida'))"

if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    chmod 600 "$HOME/.kaggle/kaggle.json"
    log_info "~/.kaggle/kaggle.json ✓"
else
    log_warn "~/.kaggle/kaggle.json no encontrado — se intentará descargar solo si faltan ZIPs."
    log_warn "  Para obtenerlo: https://www.kaggle.com/settings → API → 'Create New Token'"
fi

# ── Crear estructura de directorios ───────────────────────────────────────────
mkdir -p \
    "$DATASETS_DIR/nih_chest_xrays" \
    "$DATASETS_DIR/isic_2019" \
    "$DATASETS_DIR/osteoarthritis" \
    "$DATASETS_DIR/luna_lung_cancer" \
    "$DATASETS_DIR/zenodo_13715870" \
    "$DATASETS_DIR/panorama_labels"

# ── Función de descarga Kaggle ────────────────────────────────────────────────
_kaggle_dl() {
    local dataset="$1" dest_dir="$2" expected_file="$3" display="$4"
    if [[ -f "$expected_file" ]]; then
        log_info "[$display] Ya existe: $(basename "$expected_file") ($(du -sh "$expected_file" | cut -f1)), saltando."
        return 0
    fi
    # Verificar credenciales solo cuando realmente se necesita descargar
    if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
        log_error "[$display] Falta ~/.kaggle/kaggle.json para descargar '$dataset'."
        log_error "  1. Ve a https://www.kaggle.com/settings → API → 'Create New Token'"
        log_error "  2. cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        return 1
    fi
    log_info "[$display] → Iniciando descarga: kaggle datasets download -d $dataset"
    kaggle datasets download -d "$dataset" -p "$dest_dir"
    if [[ -f "$expected_file" ]]; then
        log_info "[$display] ✓ Descarga OK: $(du -sh "$expected_file" | cut -f1)"
    else
        log_error "[$display] Archivo esperado no encontrado tras descarga: $expected_file"
        log_error "[$display] Contenido de $dest_dir: $(ls "$dest_dir" 2>/dev/null || echo 'vacío')"
        return 1
    fi
}

# ── Función de extracción RÁPIDA ──────────────────────────────────────────────
extract_archive() {
    local archive="$1" dest="$2" name="$3"
    local t0=$SECONDS
    log_info "[$name] Extrayendo: $(basename "$archive") → $dest"
    unzip -q "$archive" -d "$dest" || {
        log_error "[$name] ✗ Error en extracción de '$archive'"
        return 1
    }
    log_info "[$name] ✓ Extraído en $((SECONDS - t0))s"
}

# ──────────────────────────────────────────────────────────────────────────────
# FASE 1 — DESCARGAS EN PARALELO
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 1 — Descargas paralelas"
PIDS=()

# NIH ChestXray14 (~42 GB)
_kaggle_dl \
    "nih-chest-xrays/data" \
    "$DATASETS_DIR/nih_chest_xrays" \
    "$DATASETS_DIR/nih_chest_xrays/data.zip" \
    "NIH" &
PIDS+=($!)

# ISIC 2019 (~9 GB)
_kaggle_dl \
    "andrewmvd/isic-2019" \
    "$DATASETS_DIR/isic_2019" \
    "$DATASETS_DIR/isic_2019/isic-2019.zip" \
    "ISIC" &
PIDS+=($!)

# Osteoarthritis (~400 MB)
_kaggle_dl \
    "dhruvacube/osteoarthritis" \
    "$DATASETS_DIR/osteoarthritis" \
    "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" \
    "OA" &
PIDS+=($!)

# LUNA16 (~650 MB — solo CSVs y máscaras, sin CTs)
_kaggle_dl \
    "fanbyprinciple/luna-lung-cancer-dataset" \
    "$DATASETS_DIR/luna_lung_cancer" \
    "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" \
    "LUNA" &
PIDS+=($!)

# Pancreas PANORAMA — Zenodo batch_1.zip (~50 GB)
BATCH1="$DATASETS_DIR/zenodo_13715870/batch_1.zip"
if [[ ! -f "$BATCH1" ]]; then
    log_info "[PANCREAS] → Iniciando descarga desde Zenodo (~50 GB)..."
    wget --progress=dot:giga \
         -c \
         -O "$BATCH1" \
         "https://zenodo.org/records/13715870/files/batch_1.zip?download=1" &
    PIDS+=($!)
else
    log_info "[PANCREAS] batch_1.zip ya existe ($(du -sh "$BATCH1" | cut -f1)), saltando."
fi

# Esperar todas las descargas
log_info "Esperando ${#PIDS[@]} proceso(s) de descarga en paralelo..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        log_error "Proceso PID=$pid terminó con error."
        FAIL=$((FAIL + 1))
    fi
done
if [[ $FAIL -gt 0 ]]; then
    log_error "$FAIL descarga(s) fallaron. Revisa el log: $LOG_FILE"
    exit 1
fi
log_info "✓ Todas las descargas completadas."

# ──────────────────────────────────────────────────────────────────────────────
# FASE 2 — EXTRACCIÓN SECUENCIAL (máxima velocidad)
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 2 — Extracción secuencial (modo RÁPIDO)"

# NIH ChestXray14
if [[ ! -d "$DATASETS_DIR/nih_chest_xrays/images_001" ]]; then
    extract_archive \
        "$DATASETS_DIR/nih_chest_xrays/data.zip" \
        "$DATASETS_DIR/nih_chest_xrays" \
        "NIH"
else
    log_info "[NIH] Ya extraído (images_001/ existe), saltando."
fi

# ISIC 2019
if [[ ! -d "$DATASETS_DIR/isic_2019/ISIC_2019_Training_Input" ]]; then
    extract_archive \
        "$DATASETS_DIR/isic_2019/isic-2019.zip" \
        "$DATASETS_DIR/isic_2019" \
        "ISIC"
else
    log_info "[ISIC] Ya extraído (ISIC_2019_Training_Input/ existe), saltando."
fi

# Osteoarthritis
if [[ ! -d "$DATASETS_DIR/osteoarthritis/KLGrade" ]]; then
    extract_archive \
        "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" \
        "$DATASETS_DIR/osteoarthritis" \
        "OA"
else
    log_info "[OA] Ya extraído (KLGrade/ existe), saltando."
fi

# LUNA16
if [[ ! -d "$DATASETS_DIR/luna_lung_cancer/candidates_V2" ]]; then
    extract_archive \
        "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" \
        "$DATASETS_DIR/luna_lung_cancer" \
        "LUNA"
else
    log_info "[LUNA] Ya extraído (candidates_V2/ existe), saltando."
fi

# Pancreas — solo extraer si los NIfTI no están presentes
NII_COUNT=$(find "$DATASETS_DIR/zenodo_13715870" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l)
if [[ $NII_COUNT -eq 0 ]]; then
    extract_archive \
        "$DATASETS_DIR/zenodo_13715870/batch_1.zip" \
        "$DATASETS_DIR/zenodo_13715870" \
        "PANCREAS"
else
    log_info "[PANCREAS] Ya extraído ($NII_COUNT NIfTI files presentes), saltando."
fi

# ──────────────────────────────────────────────────────────────────────────────
# FASE 3 — PREPARACIÓN DE DATASETS
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 3 — Preparación de datasets"

# -- NIH: split lists + directorio all_images con symlinks --
NIH_DIR="$DATASETS_DIR/nih_chest_xrays"

# Mínimos esperados (86523 train_val, 25595 test en el dataset completo NIH).
declare -A NIH_TXT_MIN=( [train_val_list.txt]=80000 [test_list.txt]=20000 )
for txt in train_val_list.txt test_list.txt; do
    local_count=0
    [[ -f "$NIH_DIR/$txt" ]] && local_count=$(wc -l < "$NIH_DIR/$txt")
    if [[ ! -f "$NIH_DIR/$txt" ]]; then
        log_warn "[NIH] $txt no encontrado. Extrayendo de data.zip..."
        unzip -j -q "$NIH_DIR/data.zip" "$txt" -d "$NIH_DIR"
    elif [[ $local_count -lt ${NIH_TXT_MIN[$txt]} ]]; then
        log_warn "[NIH] $txt truncado ($local_count líneas < mínimo ${NIH_TXT_MIN[$txt]}). Re-extrayendo de data.zip..."
        unzip -j -q -o "$NIH_DIR/data.zip" "$txt" -d "$NIH_DIR"
    fi
    log_info "[NIH] $txt ✓ ($(wc -l < "$NIH_DIR/$txt") entradas)"
done

# Verificar all_images/ — detecta corrupción por I/O error (OOM kill previo)
NIH_ALL="$NIH_DIR/all_images"
NIH_ALL_OK=false
if [[ -d "$NIH_ALL" ]]; then
    N_LINKS=$(find "$NIH_ALL" -maxdepth 1 -type l 2>/dev/null | wc -l)
    if [[ $N_LINKS -gt 0 ]]; then
        NIH_ALL_OK=true
        log_info "[NIH] all_images/ ya existe ($N_LINKS symlinks), saltando."
    else
        log_warn "[NIH] all_images/ existe pero está vacío/corrupto (I/O error previo). Eliminando y recreando..."
        rm -rf "$NIH_ALL" 2>/dev/null || { log_error "[NIH] No se pudo eliminar all_images/ — ejecuta: sudo rm -rf $NIH_ALL"; }
    fi
fi
if [[ "$NIH_ALL_OK" == false ]]; then
    log_info "[NIH] Creando all_images/ con symlinks a todos los .png..."
    mkdir -p "$NIH_ALL"
    # --target-directory agrupa args en lotes (ARG_MAX) → ~10-20 invocaciones de ln en total.
    find "$NIH_DIR" -maxdepth 4 -path "*/images/*.png" -print0 | \
        xargs -0 ln -sf --target-directory="$NIH_ALL/"
    N_LINKS=$(find "$NIH_ALL" -maxdepth 1 -type l | wc -l)
    log_info "[NIH] ✓ $N_LINKS symlinks creados en all_images/"
fi

# -- OA Rodilla: splits train/val/test con consolidación KL 0-4 → 3 clases --
OA_DIR="$DATASETS_DIR/osteoarthritis"
if [[ ! -d "$OA_DIR/oa_splits/train" ]]; then
    log_info "[OA] Generando splits train/val/test (KL 0-4 → 3 clases)..."
    python3 - <<PYEOF
import os, shutil, random, sys
from pathlib import Path

os.chdir("$OA_DIR")
src = Path("KLGrade/KLGrade")
dst = Path("oa_splits")

if not src.exists():
    print(f"[OA] ERROR: directorio fuente no encontrado: {src.resolve()}", file=sys.stderr)
    sys.exit(1)

# Consolidar grados KL: 0→cls0, 1+2→cls1 (Leve), 3+4→cls2 (Severo)
mapping = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
all_files = {0: [], 1: [], 2: []}
for kl_str, cls in mapping.items():
    kl_dir = src / kl_str
    if kl_dir.exists():
        files = list(kl_dir.glob("*.jpg")) + list(kl_dir.glob("*.png"))
        all_files[cls].extend(files)
    else:
        print(f"[OA] Advertencia: KL{kl_str}/ no encontrada, saltando.")

random.seed(42)
total = 0
for cls, files in all_files.items():
    random.shuffle(files)
    n = len(files)
    total += n
    cuts = {"train": files[:int(.80*n)], "val": files[int(.80*n):int(.95*n)], "test": files[int(.95*n):]}
    for split_name, imgs in cuts.items():
        d = dst / split_name / str(cls)
        d.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, d / img.name)

report = {s: sum(len(list((dst/s/c).iterdir())) for c in "012" if (dst/s/c).exists()) for s in ["train","val","test"]}
print(f"[OA] ✓ Splits creados — train:{report['train']} | val:{report['val']} | test:{report['test']} | total:{total}")
PYEOF
    log_info "[OA] ✓ oa_splits/ listo."
else
    log_info "[OA] oa_splits/ ya existe, saltando."
fi

# -- LUNA: nota informativa --
log_warn "[LUNA] El dataset Kaggle contiene solo CSVs/máscaras, NO los volúmenes CT (.mhd/.raw)."
log_warn "[LUNA] Descarga los CTs desde: https://luna16.grand-challenge.org/data/"
log_warn "[LUNA] Hasta entonces, --luna_patches se omite en FASE 0."

# ──────────────────────────────────────────────────────────────────────────────
# FASE 4 — PANORAMA LABELS + COMPARACIÓN CON ZENODO
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 4 — Panorama Labels + Verificación de alineación"

PANORAMA_DIR="$DATASETS_DIR/panorama_labels"
COMMIT_FILE="$DATASETS_DIR/panorama_labels_commit.txt"

if [[ ! -d "$PANORAMA_DIR/.git" ]]; then
    log_info "[PANORAMA] Clonando DIAGNijmegen/panorama_labels..."
    git clone https://github.com/DIAGNijmegen/panorama_labels.git "$PANORAMA_DIR"
    COMMIT=$(cd "$PANORAMA_DIR" && git rev-parse HEAD)
    echo "$COMMIT" > "$COMMIT_FILE"
    log_info "[PANORAMA] ✓ Clonado. Commit: $COMMIT"
    log_info "[PANORAMA] Hash guardado en: $COMMIT_FILE"
else
    COMMIT=$(cd "$PANORAMA_DIR" && git rev-parse HEAD)
    log_info "[PANORAMA] Ya clonado. Commit: $COMMIT"
    echo "$COMMIT" > "$COMMIT_FILE"
fi

log_info "[PANORAMA] Comparando case IDs (zenodo_13715870 vs panorama_labels)..."
python3 - <<PYEOF
import sys, re
from pathlib import Path

zenodo_dir = Path("$DATASETS_DIR/zenodo_13715870")
labels_dir = Path("$DATASETS_DIR/panorama_labels")

# ─── IDs de los NIfTI en zenodo ──────────────────────────────────────────────
nii_files = sorted(zenodo_dir.glob("*.nii.gz"))
nii_ids = set()
for f in nii_files:
    m = re.match(r'^(\d+)_', f.name)
    if m:
        nii_ids.add(m.group(1))

print(f"[PANORAMA] NIfTI en zenodo_13715870: {len(nii_files)} archivos → {len(nii_ids)} patient IDs únicos")
if not nii_ids:
    print("[PANORAMA] ⚠  No se encontraron NIfTIs. ¿Se extrajo batch_1.zip?")
    sys.exit(0)

# ─── IDs en panorama_labels ───────────────────────────────────────────────────
label_ids = set()
source_desc = "desconocido"

# Intento 1: CSV con columna de ID
try:
    import pandas as pd
    candidate_cols = ['case_id', 'patient_id', 'PatientID', 'pid', 'id', 'subject_id',
                      'name', 'image_id', 'case']
    for csv_path in sorted(labels_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for col in candidate_cols:
                if col in df.columns:
                    ids_raw = df[col].astype(str)
                    ids_found = ids_raw.str.extract(r'(\d{5,6})')[0].dropna().unique()
                    if len(ids_found) > 0:
                        label_ids.update(ids_found.tolist())
                        source_desc = f"{csv_path.relative_to(labels_dir)} (col='{col}')"
                        break
            if label_ids:
                break
        except Exception:
            continue
except ImportError:
    pass

# Intento 2: NIfTI de segmentación en el repo de labels
if not label_ids:
    for f in labels_dir.rglob("*.nii.gz"):
        m = re.match(r'^(\d+)_', f.name)
        if m:
            label_ids.add(m.group(1))
    if label_ids:
        source_desc = "NIfTI files en panorama_labels"

# Intento 3: buscar números de 6 dígitos en cualquier archivo de texto/JSON
if not label_ids:
    for fpath in list(labels_dir.rglob("*.json"))[:5] + list(labels_dir.rglob("*.txt"))[:5]:
        try:
            content = fpath.read_text(errors='ignore')
            found = re.findall(r'\b(1\d{5})\b', content)
            if found:
                label_ids.update(found)
                source_desc = f"{fpath.relative_to(labels_dir)}"
        except Exception:
            continue

if not label_ids:
    print("[PANORAMA] ⚠  No se encontraron IDs en panorama_labels.")
    print("[PANORAMA]   Estructura encontrada:")
    for p in sorted(labels_dir.iterdir()):
        print(f"    {p.name}{'/' if p.is_dir() else ''}")
    sys.exit(0)

print(f"[PANORAMA] IDs en panorama_labels: {len(label_ids)} únicos (fuente: {source_desc})")

# ─── Comparación ─────────────────────────────────────────────────────────────
matched    = nii_ids & label_ids
only_nii   = nii_ids - label_ids
only_label = label_ids - nii_ids
pct = 100.0 * len(matched) / max(len(nii_ids), 1)

print(f"\n[PANORAMA] ─── Resultado de alineación ───")
print(f"  ✓ Con imagen Y etiqueta          : {len(matched):>5}  ({pct:.1f}% de los NIfTIs locales)")
print(f"  ⚠  Solo en zenodo (sin etiqueta) : {len(only_nii):>5}")
print(f"    Solo en labels (sin imagen)   : {len(only_label):>5}")

if pct == 100.0 and len(only_nii) == 0:
    print("\n[PANORAMA] ✅ ALINEACIÓN PERFECTA — todos los NIfTIs tienen etiqueta. Procediendo.")
elif pct >= 80.0:
    print(f"\n[PANORAMA] ✅ {pct:.0f}% de alineación — suficiente para FASE 0. Procediendo.")
    if only_nii:
        print(f"  IDs sin etiqueta (primeros 10): {sorted(only_nii)[:10]}")
else:
    print(f"\n[PANORAMA] ⚠  Solo {pct:.0f}% de alineación.")
    print("  Posible causa: batch_1.zip es solo una parte del dataset completo.")
    print("  Verifica si hay batch_2.zip, batch_3.zip... en Zenodo.")
PYEOF

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
log_step "SETUP COMPLETADO ✓"
echo ""
echo "Tamaño de datasets:"
du -sh "$DATASETS_DIR"/*/  2>/dev/null | sort -h
echo ""
log_info "Log completo: $LOG_FILE"
log_info ""
log_info "Siguiente paso — FASE 0 (ejecutar desde src/pipeline/):"
echo ""
cat <<CMD
  cd $REPO_ROOT/src/pipeline

  python3 fase0_extract_embeddings.py \\
    --backbone          vit_tiny_patch16_224 \\
    --batch_size        64 \\
    --output_dir        $REPO_ROOT/embeddings/vit_tiny \\
    --chest_csv         $DATASETS_DIR/nih_chest_xrays/Data_Entry_2017.csv \\
    --chest_imgs        $DATASETS_DIR/nih_chest_xrays/all_images \\
    --chest_train_list  $DATASETS_DIR/nih_chest_xrays/train_val_list.txt \\
    --chest_val_list    $DATASETS_DIR/nih_chest_xrays/test_list.txt \\
    --chest_view_filter PA \\
    --chest_bbox_csv    $DATASETS_DIR/nih_chest_xrays/BBox_List_2017.csv \\
    --isic_gt           $DATASETS_DIR/isic_2019/ISIC_2019_Training_GroundTruth.csv \\
    --isic_imgs         "$DATASETS_DIR/isic_2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input" \\
    --isic_metadata     $DATASETS_DIR/isic_2019/ISIC_2019_Training_Metadata.csv \\
    --oa_root           $DATASETS_DIR/osteoarthritis/oa_splits \\
    --pancreas_nii_dir       $DATASETS_DIR/zenodo_13715870 \\
    --pancreas_labels_dir    $DATASETS_DIR/panorama_labels \\
    --pancreas_labels_commit \$(cat $DATASETS_DIR/panorama_labels_commit.txt) \\
    --pancreas_roi_strategy  A
CMD
echo ""
