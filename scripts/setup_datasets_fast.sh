#!/usr/bin/env bash
# ==============================================================================
#  setup_datasets_fast.sh  (v3 — NIH OOM fix)
#  Descarga todos los datasets y los prepara para FASE 0.
#
#  Cambios v3:
#    - NIH (~42 GB): reemplazado kaggle CLI por wget con auth de la API de Kaggle.
#      kaggle CLI carga el ZIP completo en RAM antes de escribirlo a disco,
#      lo que provoca un OOM kill en máquinas con ≤ 20 GB de RAM.
#      wget escribe en streaming directo a disco sin buffer de memoria.
#    - Función nih_wget_download() extrae usuario/clave de ~/.kaggle/kaggle.json
#      y usa la Kaggle REST API v1 con --continue para reanudar descargas parciales.
#
#  Cambios v2:
#    - Eliminado set -e (causaba kills silenciosos en procesos paralelos)
#    - Descargas grandes (NIH+Zenodo) en paralelo; pequeñas secuenciales primero
#    - Detección automática del nombre real del ZIP descargado por Kaggle
#    - Eliminado --chunk-size (flag inexistente en kaggle CLI)
#    - Manejo explícito de errores sin abortar el script completo
#
#  Uso:
#    cd /ruta/del/repo
#    bash scripts/setup_datasets_fast.sh
#
#  Prerequisito:
#    ~/.kaggle/kaggle.json  (https://www.kaggle.com/settings → API → Create New Token)
# ==============================================================================
set -uo pipefail   # SIN -e: los errores de subprocesos no matan el script

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
ts()        { date '+%H:%M:%S'; }
log_info()  { echo -e "$(ts) ${GREEN}[INFO]${NC}   $*"; }
log_warn()  { echo -e "$(ts) ${YELLOW}[WARN]${NC}   $*"; }
log_error() { echo -e "$(ts) ${RED}[ERROR]${NC}  $*"; }
log_ok()    { echo -e "$(ts) ${GREEN}[OK]${NC}     $*"; }
log_step()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${NC}";
              echo -e "${BOLD}${BLUE}  $*${NC}";
              echo -e "${BOLD}${BLUE}══════════════════════════════════════${NC}\n"; }

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"
LOG_FILE="$DATASETS_DIR/setup_fast.log"

mkdir -p "$DATASETS_DIR"
# Redirigir stdout+stderr al log Y a la terminal
exec > >(tee -a "$LOG_FILE") 2>&1

log_step "Setup v2 iniciado — $(date)"
log_info "Repo:     $REPO_ROOT"
log_info "Datasets: $DATASETS_DIR"
log_info "Log:      $LOG_FILE"

# ── Verificar prerequisitos ───────────────────────────────────────────────────
log_step "Verificando prerequisitos"
PREREQ_OK=true
for cmd in python3 unzip git awk wget; do
    if command -v "$cmd" &>/dev/null; then
        log_info "$cmd ✓ ($(command -v "$cmd"))"
    else
        log_error "'$cmd' no encontrado.  sudo apt-get install -y $cmd"
        PREREQ_OK=false
    fi
done

if ! command -v kaggle &>/dev/null; then
    log_warn "kaggle CLI no encontrado. Instalando..."
    pip install --quiet kaggle
    if ! command -v kaggle &>/dev/null; then
        log_error "No se pudo instalar kaggle.  pip install kaggle"
        PREREQ_OK=false
    fi
fi
log_info "kaggle ✓ ($(kaggle --version 2>/dev/null | head -1))"

if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    chmod 600 "$HOME/.kaggle/kaggle.json"
    log_info "~/.kaggle/kaggle.json ✓"
else
    log_error "~/.kaggle/kaggle.json no encontrado."
    log_error "  1. Ve a https://www.kaggle.com/settings → API → 'Create New Token'"
    log_error "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    PREREQ_OK=false
fi

if [[ "$PREREQ_OK" == false ]]; then
    log_error "Prerequisitos no cumplidos. Corrige los errores anteriores y vuelve a ejecutar."
    exit 1
fi

# ── Crear directorios ─────────────────────────────────────────────────────────
mkdir -p \
    "$DATASETS_DIR/nih_chest_xrays" \
    "$DATASETS_DIR/isic_2019" \
    "$DATASETS_DIR/osteoarthritis" \
    "$DATASETS_DIR/luna_lung_cancer" \
    "$DATASETS_DIR/zenodo_13715870" \
    "$DATASETS_DIR/panorama_labels"

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: descarga Kaggle con detección automática de nombre de archivo
# Kaggle CLI a veces guarda como "archive.zip" en lugar del nombre esperado.
# ──────────────────────────────────────────────────────────────────────────────
kaggle_download() {
    # Args: dataset  dest_dir  expected_zip  display_name
    local dataset="$1"
    local dest_dir="$2"
    local expected_zip="$3"
    local name="$4"
    local t0=$SECONDS

    # Si ya existe el ZIP esperado, saltar
    if [[ -f "$expected_zip" ]]; then
        log_info "[$name] Ya existe: $(basename "$expected_zip") ($(du -sh "$expected_zip" | cut -f1)), saltando."
        return 0
    fi

    log_info "[$name] Descargando: kaggle datasets download -d $dataset"

    # Registrar ZIPs presentes antes de la descarga
    local before
    before=$(find "$dest_dir" -maxdepth 1 -name "*.zip" 2>/dev/null | sort)

    # Descargar — sin --chunk-size (no existe en kaggle CLI)
    if ! kaggle datasets download -d "$dataset" -p "$dest_dir"; then
        log_error "[$name] kaggle download falló para $dataset"
        return 1
    fi

    # Detectar el archivo ZIP recién descargado (puede ser "archive.zip" u otro nombre)
    local after new_zip
    after=$(find "$dest_dir" -maxdepth 1 -name "*.zip" 2>/dev/null | sort)
    new_zip=$(comm -13 <(echo "$before") <(echo "$after") | head -1)

    if [[ -z "$new_zip" ]]; then
        log_error "[$name] No se encontró ningún ZIP nuevo en $dest_dir tras la descarga."
        log_error "[$name] Contenido actual: $(ls -lah "$dest_dir")"
        return 1
    fi

    # Renombrar si el nombre no coincide con el esperado
    if [[ "$new_zip" != "$expected_zip" ]]; then
        log_warn "[$name] Kaggle descargó como '$(basename "$new_zip")' — renombrando a '$(basename "$expected_zip")'"
        mv "$new_zip" "$expected_zip"
    fi

    local size elapsed
    size=$(du -sh "$expected_zip" | cut -f1)
    elapsed=$((SECONDS - t0))
    log_ok "[$name] ✓ Descarga completada: $size en ${elapsed}s"
    return 0
}

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: descarga NIH via wget + Kaggle REST API
#
# Por qué NO usar kaggle CLI para NIH (42 GB):
#   kaggle CLI descarga el ZIP completo en RAM antes de escribirlo a disco.
#   42 GB en una máquina de 20 GB de RAM → el OOM killer de Linux lo mata
#   silenciosamente con "Killed" en el log, sin ningún mensaje de error útil.
#   wget escribe en streaming directo al disco sin buffer de memoria,
#   sin importar el tamaño del archivo, y soporta --continue para reanudar.
#
# La Kaggle REST API v1 acepta autenticación HTTP Basic (usuario:api_key).
# URL: https://www.kaggle.com/api/v1/datasets/download/<owner>/<dataset-slug>
# ──────────────────────────────────────────────────────────────────────────────
nih_wget_download() {
    local dest_zip="$1"
    local t0=$SECONDS

    if [[ -f "$dest_zip" ]]; then
        log_info "[NIH] Ya existe: data.zip ($(du -sh "$dest_zip" | cut -f1)), saltando."
        return 0
    fi

    # Leer credenciales de ~/.kaggle/kaggle.json
    local kaggle_user kaggle_key
    kaggle_user=$(python3 -c \
        "import json; d=json.load(open('$HOME/.kaggle/kaggle.json')); print(d['username'])")
    kaggle_key=$(python3 -c \
        "import json; d=json.load(open('$HOME/.kaggle/kaggle.json')); print(d['key'])")

    if [[ -z "$kaggle_user" || -z "$kaggle_key" ]]; then
        log_error "[NIH] No se pudieron leer las credenciales de ~/.kaggle/kaggle.json"
        return 1
    fi

    log_info "[NIH] Descargando via wget + Kaggle REST API (~42 GB, streaming a disco)..."
    log_info "[NIH] Usuario Kaggle: $kaggle_user"
    log_info "[NIH] Destino:        $dest_zip"

    wget \
        --progress=dot:giga \
        --continue \
        --tries=10 \
        --retry-connrefused \
        --waitretry=60 \
        --timeout=300 \
        --auth-no-challenge \
        --http-user="$kaggle_user" \
        --http-password="$kaggle_key" \
        -O "$dest_zip" \
        "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"

    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log_error "[NIH] wget terminó con código $rc."
        if [[ -f "$dest_zip" ]]; then
            local partial_size
            partial_size=$(du -sh "$dest_zip" | cut -f1)
            log_warn "[NIH] Archivo parcial en disco: $partial_size"
            log_warn "[NIH] Vuelve a ejecutar el script — wget reanudará desde donde quedó (--continue)."
        fi
        return 1
    fi

    local size elapsed
    size=$(du -sh "$dest_zip" | cut -f1)
    elapsed=$((SECONDS - t0))
    log_ok "[NIH] ✓ Descarga completada: $size en ${elapsed}s"
    return 0
}

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: extracción de ZIP
# ──────────────────────────────────────────────────────────────────────────────
extract_zip() {
    local archive="$1"
    local dest="$2"
    local name="$3"
    local t0=$SECONDS

    if [[ ! -f "$archive" ]]; then
        log_error "[$name] Archivo no encontrado para extraer: $archive"
        return 1
    fi

    log_info "[$name] Extrayendo $(basename "$archive") → $dest"
    if ! unzip -q "$archive" -d "$dest"; then
        log_error "[$name] Error en la extracción de $archive"
        return 1
    fi
    log_ok "[$name] ✓ Extraído en $((SECONDS - t0))s"
    return 0
}

# ──────────────────────────────────────────────────────────────────────────────
# FASE 1A — Datasets pequeños: OA y LUNA (secuenciales, rápidos)
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 1A — Descargas pequeñas (OA + LUNA, secuencial)"

# Osteoarthritis (~400 MB)
kaggle_download \
    "dhruvacube/osteoarthritis" \
    "$DATASETS_DIR/osteoarthritis" \
    "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" \
    "OA"
OA_DL_OK=$?

# LUNA16 (~330 MB — solo CSVs y scripts, sin volúmenes CT)
kaggle_download \
    "fanbyprinciple/luna-lung-cancer-dataset" \
    "$DATASETS_DIR/luna_lung_cancer" \
    "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" \
    "LUNA"
LUNA_DL_OK=$?

# ──────────────────────────────────────────────────────────────────────────────
# FASE 1B — Datasets medianos: ISIC (~9 GB, secuencial)
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 1B — Descarga mediana (ISIC 2019, ~9 GB, secuencial)"

kaggle_download \
    "andrewmvd/isic-2019" \
    "$DATASETS_DIR/isic_2019" \
    "$DATASETS_DIR/isic_2019/isic-2019.zip" \
    "ISIC"
ISIC_DL_OK=$?

# ──────────────────────────────────────────────────────────────────────────────
# FASE 1C — Datasets grandes: NIH (~42 GB) y Zenodo (~46 GB) en paralelo
# Tienen pesos similares → se reparten el ancho de banda equitativamente.
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 1C — Descargas grandes en paralelo (NIH ~42 GB + Zenodo ~46 GB)"

# NIH ChestXray14 — wget en background (NO kaggle CLI: 42 GB > 20 GB RAM → OOM kill)
(
    nih_wget_download "$DATASETS_DIR/nih_chest_xrays/data.zip"
) &
PID_NIH=$!

# Pancreas PANORAMA — Zenodo batch_1.zip — en background
BATCH1="$DATASETS_DIR/zenodo_13715870/batch_1.zip"
if [[ ! -f "$BATCH1" ]]; then
    log_info "[ZENODO] → Descargando batch_1.zip (~46 GB) desde Zenodo..."
    wget \
        --progress=dot:giga \
        --continue \
        --tries=5 \
        --retry-connrefused \
        --waitretry=30 \
        --timeout=120 \
        -O "$BATCH1" \
        "https://zenodo.org/records/13715870/files/batch_1.zip?download=1" &
    PID_ZENODO=$!
else
    log_info "[ZENODO] batch_1.zip ya existe ($(du -sh "$BATCH1" | cut -f1)), saltando."
    PID_ZENODO=""
fi

# Esperar NIH
NIH_DL_OK=0
if ! wait "$PID_NIH"; then
    log_error "[NIH] La descarga terminó con error."
    NIH_DL_OK=1
fi

# Esperar Zenodo
ZENODO_DL_OK=0
if [[ -n "${PID_ZENODO:-}" ]]; then
    if ! wait "$PID_ZENODO"; then
        log_error "[ZENODO] La descarga terminó con error."
        ZENODO_DL_OK=1
    else
        log_ok "[ZENODO] ✓ batch_1.zip descargado ($(du -sh "$BATCH1" | cut -f1))"
    fi
fi

# Resumen de descargas
log_step "Resumen de descargas"
DOWNLOAD_FAILURES=0
for result_name in "OA:$OA_DL_OK" "LUNA:$LUNA_DL_OK" "ISIC:$ISIC_DL_OK" "NIH:$NIH_DL_OK" "ZENODO:$ZENODO_DL_OK"; do
    ds="${result_name%%:*}"
    rc="${result_name##*:}"
    if [[ "$rc" -eq 0 ]]; then
        log_ok "  $ds ✓"
    else
        log_error "  $ds ✗ — FALLÓ"
        DOWNLOAD_FAILURES=$((DOWNLOAD_FAILURES + 1))
    fi
done

if [[ $DOWNLOAD_FAILURES -gt 0 ]]; then
    log_error "$DOWNLOAD_FAILURES descarga(s) fallaron. Corrige los errores y vuelve a ejecutar."
    log_warn "El script continuará con los datasets que sí están disponibles."
fi

# ──────────────────────────────────────────────────────────────────────────────
# FASE 2 — EXTRACCIÓN SECUENCIAL
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 2 — Extracción"

# NIH ChestXray14
if [[ -d "$DATASETS_DIR/nih_chest_xrays/images_001" ]]; then
    log_info "[NIH] Ya extraído (images_001/ existe), saltando."
elif [[ -f "$DATASETS_DIR/nih_chest_xrays/data.zip" ]]; then
    extract_zip \
        "$DATASETS_DIR/nih_chest_xrays/data.zip" \
        "$DATASETS_DIR/nih_chest_xrays" \
        "NIH"
else
    log_warn "[NIH] data.zip no encontrado — saltando extracción."
fi

# ISIC 2019
if [[ -d "$DATASETS_DIR/isic_2019/ISIC_2019_Training_Input" ]]; then
    log_info "[ISIC] Ya extraído (ISIC_2019_Training_Input/ existe), saltando."
elif [[ -f "$DATASETS_DIR/isic_2019/isic-2019.zip" ]]; then
    extract_zip \
        "$DATASETS_DIR/isic_2019/isic-2019.zip" \
        "$DATASETS_DIR/isic_2019" \
        "ISIC"
else
    log_warn "[ISIC] isic-2019.zip no encontrado — saltando extracción."
fi

# Osteoarthritis
if [[ -d "$DATASETS_DIR/osteoarthritis/KLGrade" ]]; then
    log_info "[OA] Ya extraído (KLGrade/ existe), saltando."
elif [[ -f "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" ]]; then
    extract_zip \
        "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" \
        "$DATASETS_DIR/osteoarthritis" \
        "OA"
else
    log_warn "[OA] osteoarthritis.zip no encontrado — saltando extracción."
fi

# LUNA16
if [[ -d "$DATASETS_DIR/luna_lung_cancer/candidates_V2" ]]; then
    log_info "[LUNA] Ya extraído (candidates_V2/ existe), saltando."
elif [[ -f "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" ]]; then
    extract_zip \
        "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" \
        "$DATASETS_DIR/luna_lung_cancer" \
        "LUNA"
else
    log_warn "[LUNA] luna-lung-cancer-dataset.zip no encontrado — saltando extracción."
fi

# Pancreas — extraer NIfTI de batch_1.zip
NII_COUNT=$(find "$DATASETS_DIR/zenodo_13715870" -maxdepth 2 -name "*.nii.gz" 2>/dev/null | wc -l)
if [[ $NII_COUNT -gt 0 ]]; then
    log_info "[PANCREAS] Ya extraído ($NII_COUNT archivos .nii.gz presentes), saltando."
elif [[ -f "$BATCH1" ]]; then
    extract_zip \
        "$BATCH1" \
        "$DATASETS_DIR/zenodo_13715870" \
        "PANCREAS"
else
    log_warn "[PANCREAS] batch_1.zip no encontrado — saltando extracción."
fi

# ──────────────────────────────────────────────────────────────────────────────
# FASE 3 — PREPARACIÓN DE DATASETS
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 3 — Preparación"

# ── NIH: archivos de texto + directorio all_images con symlinks ───────────────
NIH_DIR="$DATASETS_DIR/nih_chest_xrays"

declare -A NIH_TXT_MIN=( [train_val_list.txt]=80000 [test_list.txt]=20000 )
for txt in train_val_list.txt test_list.txt; do
    local_count=0
    [[ -f "$NIH_DIR/$txt" ]] && local_count=$(wc -l < "$NIH_DIR/$txt")
    if [[ ! -f "$NIH_DIR/$txt" ]]; then
        if [[ -f "$NIH_DIR/data.zip" ]]; then
            log_warn "[NIH] $txt no encontrado — extrayendo de data.zip..."
            unzip -j -q "$NIH_DIR/data.zip" "$txt" -d "$NIH_DIR" || true
        fi
    elif [[ $local_count -lt ${NIH_TXT_MIN[$txt]} ]]; then
        log_warn "[NIH] $txt truncado ($local_count < mínimo ${NIH_TXT_MIN[$txt]}) — re-extrayendo..."
        unzip -j -q -o "$NIH_DIR/data.zip" "$txt" -d "$NIH_DIR" || true
    fi
    if [[ -f "$NIH_DIR/$txt" ]]; then
        log_info "[NIH] $txt ✓ ($(wc -l < "$NIH_DIR/$txt") entradas)"
    else
        log_warn "[NIH] $txt aún no disponible (data.zip pendiente)."
    fi
done

NIH_ALL="$NIH_DIR/all_images"
if [[ -d "$NIH_ALL" ]]; then
    N_LINKS=$(find "$NIH_ALL" -maxdepth 1 -type l 2>/dev/null | wc -l)
    if [[ $N_LINKS -gt 0 ]]; then
        log_info "[NIH] all_images/ ya existe ($N_LINKS symlinks), saltando."
    else
        log_warn "[NIH] all_images/ existe pero vacío/corrupto — recreando..."
        rm -rf "$NIH_ALL"
    fi
fi
if [[ ! -d "$NIH_ALL" ]]; then
    PNG_COUNT=$(find "$NIH_DIR" -maxdepth 4 -path "*/images/*.png" 2>/dev/null | wc -l)
    if [[ $PNG_COUNT -gt 0 ]]; then
        log_info "[NIH] Creando all_images/ con $PNG_COUNT symlinks..."
        mkdir -p "$NIH_ALL"
        find "$NIH_DIR" -maxdepth 4 -path "*/images/*.png" -print0 | \
            xargs -0 ln -sf --target-directory="$NIH_ALL/"
        N_LINKS=$(find "$NIH_ALL" -maxdepth 1 -type l | wc -l)
        log_ok "[NIH] ✓ $N_LINKS symlinks creados en all_images/"
    else
        log_warn "[NIH] No se encontraron imágenes .png — all_images/ pendiente hasta extraer data.zip."
    fi
fi

# ── OA: splits train/val/test con consolidación KL 0-4 → 3 clases ────────────
OA_DIR="$DATASETS_DIR/osteoarthritis"
if [[ -d "$OA_DIR/oa_splits/train" ]]; then
    log_info "[OA] oa_splits/ ya existe, saltando."
elif [[ -d "$OA_DIR/KLGrade" ]]; then
    log_info "[OA] Generando splits train/val/test (KL 0-4 → 3 clases)..."
    python3 - <<PYEOF
import os, shutil, random, sys
from pathlib import Path

oa_dir = Path("$OA_DIR")
# Buscar el directorio KLGrade: puede estar en KLGrade/ o KLGrade/KLGrade/
for candidate in [oa_dir/"KLGrade"/"KLGrade", oa_dir/"KLGrade"]:
    if candidate.is_dir() and any(candidate.iterdir()):
        src = candidate
        break
else:
    print("[OA] ERROR: No se encontró la estructura KLGrade con imágenes", file=sys.stderr)
    sys.exit(1)

dst = oa_dir / "oa_splits"
print(f"[OA] Fuente: {src}")

# Consolidar: KL0→cls0, KL1+KL2→cls1 (Leve), KL3+KL4→cls2 (Severo)
mapping = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
all_files = {0: [], 1: [], 2: []}

for kl_str, cls in mapping.items():
    kl_dir = src / kl_str
    if kl_dir.exists():
        files = list(kl_dir.glob("*.jpg")) + list(kl_dir.glob("*.png"))
        all_files[cls].extend(files)
        print(f"[OA] KL{kl_str} → clase {cls}: {len(files)} imágenes")
    else:
        print(f"[OA] Advertencia: KL{kl_str}/ no encontrada, saltando.")

random.seed(42)
total = 0
for cls, files in all_files.items():
    random.shuffle(files)
    n = len(files)
    total += n
    splits = {
        "train": files[:int(.80 * n)],
        "val":   files[int(.80 * n):int(.95 * n)],
        "test":  files[int(.95 * n):]
    }
    for split_name, imgs in splits.items():
        d = dst / split_name / str(cls)
        d.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, d / img.name)

report = {}
for s in ["train", "val", "test"]:
    count = sum(
        len(list((dst / s / c).iterdir()))
        for c in "012"
        if (dst / s / c).exists()
    )
    report[s] = count

print(f"[OA] ✓ Splits — train:{report['train']} | val:{report['val']} | test:{report['test']} | total:{total}")
PYEOF
    log_ok "[OA] ✓ oa_splits/ listo."
else
    log_warn "[OA] KLGrade/ no encontrado — oa_splits/ pendiente hasta extraer osteoarthritis.zip."
fi

# ── LUNA: aviso sobre volúmenes CT ───────────────────────────────────────────
log_warn "[LUNA] El ZIP de Kaggle contiene solo CSVs/scripts, NO los volúmenes CT (.mhd/.zraw)."
log_warn "[LUNA] Para los CTs descarga desde: https://luna16.grand-challenge.org/data/"
log_warn "[LUNA] Los CSVs (annotations.csv, candidates_V2.csv) sí están disponibles para FASE 0."

# ──────────────────────────────────────────────────────────────────────────────
# FASE 4 — PANORAMA LABELS + ALINEACIÓN
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 4 — Panorama Labels + Verificación de alineación"

PANORAMA_DIR="$DATASETS_DIR/panorama_labels"
COMMIT_FILE="$DATASETS_DIR/panorama_labels_commit.txt"

if [[ ! -d "$PANORAMA_DIR/.git" ]]; then
    log_info "[PANORAMA] Clonando DIAGNijmegen/panorama_labels..."
    if git clone https://github.com/DIAGNijmegen/panorama_labels.git "$PANORAMA_DIR"; then
        COMMIT=$(cd "$PANORAMA_DIR" && git rev-parse HEAD)
        echo "$COMMIT" > "$COMMIT_FILE"
        log_ok "[PANORAMA] ✓ Clonado. Commit: $COMMIT"
    else
        log_error "[PANORAMA] git clone falló. Verifica tu conexión a GitHub."
    fi
else
    COMMIT=$(cd "$PANORAMA_DIR" && git rev-parse HEAD)
    log_info "[PANORAMA] Ya clonado. Commit: $COMMIT"
    echo "$COMMIT" > "$COMMIT_FILE"
fi

# Alineación de IDs zenodo ↔ panorama_labels
if [[ -d "$PANORAMA_DIR/.git" ]]; then
    log_info "[PANORAMA] Verificando alineación de IDs..."
    python3 - <<PYEOF
import sys, re
from pathlib import Path

zenodo_dir = Path("$DATASETS_DIR/zenodo_13715870")
labels_dir = Path("$DATASETS_DIR/panorama_labels")

# IDs de NIfTIs presentes localmente
nii_files = sorted(zenodo_dir.rglob("*.nii.gz"))
nii_ids = set()
for f in nii_files:
    m = re.match(r'^(\d+)', f.name)
    if m:
        nii_ids.add(m.group(1))

print(f"[PANORAMA] NIfTI locales: {len(nii_files)} archivos → {len(nii_ids)} IDs únicos")
if not nii_ids:
    print("[PANORAMA] ⚠  No se encontraron NIfTIs. ¿Se extrajo batch_1.zip?")
    sys.exit(0)

# IDs en panorama_labels
label_ids = set()
source_desc = "desconocido"

try:
    import pandas as pd
    candidate_cols = ['case_id', 'patient_id', 'PatientID', 'pid', 'id',
                      'subject_id', 'name', 'image_id', 'case']
    for csv_path in sorted(labels_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for col in candidate_cols:
                if col in df.columns:
                    ids_found = df[col].astype(str).str.extract(r'(\d{4,6})')[0].dropna().unique()
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

# Fallback: NIfTIs en el repo de labels
if not label_ids:
    for f in labels_dir.rglob("*.nii.gz"):
        m = re.match(r'^(\d+)', f.name)
        if m:
            label_ids.add(m.group(1))
    if label_ids:
        source_desc = "NIfTI en panorama_labels"

if not label_ids:
    print("[PANORAMA] ⚠  No se encontraron IDs en panorama_labels.")
    print("[PANORAMA]   Archivos en el repo:")
    for p in sorted(labels_dir.iterdir())[:15]:
        print(f"    {p.name}{'/' if p.is_dir() else ''}")
    sys.exit(0)

print(f"[PANORAMA] IDs en labels: {len(label_ids)} (fuente: {source_desc})")

matched    = nii_ids & label_ids
only_nii   = nii_ids - label_ids
only_label = label_ids - nii_ids
pct = 100.0 * len(matched) / max(len(nii_ids), 1)

print(f"\n[PANORAMA] ─── Alineación ───")
print(f"  ✓ Con imagen Y etiqueta:       {len(matched):>5}  ({pct:.1f}%)")
print(f"  ⚠  Solo en zenodo (sin label): {len(only_nii):>5}")
print(f"     Solo en labels (sin NIfTI): {len(only_label):>5}")

if pct >= 80.0:
    print(f"\n[PANORAMA] ✅ {pct:.0f}% de alineación — suficiente para continuar.")
else:
    print(f"\n[PANORAMA] ⚠  Solo {pct:.0f}% de alineación.")
    print("  Posible causa: batch_1.zip es un subconjunto del dataset completo.")
    print("  Verifica si existen batch_2.zip, batch_3.zip... en https://zenodo.org/records/13715870")
PYEOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
log_step "RESUMEN FINAL"
echo ""
echo "Tamaño de cada dataset:"
du -sh "$DATASETS_DIR"/*/  2>/dev/null | sort -h
echo ""
echo "Estado por dataset:"
declare -A MARKERS=(
    ["NIH"]="$DATASETS_DIR/nih_chest_xrays/data.zip"
    ["ISIC"]="$DATASETS_DIR/isic_2019/isic-2019.zip"
    ["OA"]="$DATASETS_DIR/osteoarthritis/oa_splits/train"
    ["LUNA"]="$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip"
    ["PANCREAS"]="$DATASETS_DIR/zenodo_13715870/batch_1.zip"
    ["PANORAMA"]="$DATASETS_DIR/panorama_labels/.git"
)
for ds in NIH ISIC OA LUNA PANCREAS PANORAMA; do
    marker="${MARKERS[$ds]}"
    if [[ -e "$marker" ]]; then
        log_ok "  $ds ✓"
    else
        log_warn "  $ds ✗ — pendiente (falta: $marker)"
    fi
done
echo ""
log_info "Log completo: $LOG_FILE"
log_info ""
log_info "Siguiente paso — FASE 0:"
echo ""
cat <<CMD
  cd $REPO_ROOT

  python3 src/pipeline/fase0_extract_embeddings.py \\
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
