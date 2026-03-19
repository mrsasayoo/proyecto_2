#!/usr/bin/env bash
# ==============================================================================
#  setup_datasets_smart.sh
#  Descarga todos los datasets en paralelo y los prepara para FASE 0.
#  Modo SMART: extracción secuencial con monitor de RAM.
#    - Si la RAM disponible cae bajo RAM_PAUSE_MB → suspende unzip con SIGSTOP.
#    - Cuando la RAM sube sobre RAM_RESUME_MB → reanuda con SIGCONT.
#    - Repite hasta terminar cada archivo.
#
#  Uso:
#    cd /ruta/del/repo
#    bash scripts/setup_datasets_smart.sh
#
#  Parámetros configurables (variables al inicio del script):
#    RAM_PAUSE_MB   — umbral para pausar  (default: 700 MB disponibles)
#    RAM_RESUME_MB  — umbral para reanudar (default: 1400 MB disponibles)
#    RAM_CHECK_S    — intervalo de chequeo mientras extrae (default: 3s)
#    RAM_SLEEP_S    — tiempo de pausa cuando RAM es baja   (default: 20s)
#
#  Prerequisito:
#    ~/.kaggle/kaggle.json — API key de Kaggle
# ==============================================================================
set -euo pipefail

# ── Configuración de RAM ──────────────────────────────────────────────────────
RAM_PAUSE_MB=700     # Pausar si RAM disponible < este valor  (MB)
RAM_RESUME_MB=1400   # Reanudar cuando RAM disponible > este valor (MB)
RAM_CHECK_S=3        # Segundos entre chequeos durante extracción activa
RAM_SLEEP_S=30       # Segundos a dormir cuando RAM está bajo el umbral

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ts()        { date '+%H:%M:%S'; }
log_info()  { echo -e "$(ts) ${GREEN}[INFO]${NC}   $*"; }
log_warn()  { echo -e "$(ts) ${YELLOW}[WARN]${NC}   $*"; }
log_error() { echo -e "$(ts) ${RED}[ERROR]${NC}  $*" >&2; }
log_ram()   { echo -e "$(ts) ${CYAN}[RAM]${NC}    $*"; }
log_step()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${NC}"; \
              echo -e "${BOLD}${BLUE}  $*${NC}"; \
              echo -e "${BOLD}${BLUE}══════════════════════════════════════${NC}\n"; }

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"
LOG_FILE="$DATASETS_DIR/setup_smart.log"

mkdir -p "$DATASETS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
log_step "Setup SMART iniciado — $(date)"
log_info "Repo: $REPO_ROOT  |  Datasets: $DATASETS_DIR"
log_info "RAM umbral pausa: ${RAM_PAUSE_MB}MB  |  reanuda: ${RAM_RESUME_MB}MB"

# ── Prerequisitos ─────────────────────────────────────────────────────────────
log_step "Verificando prerequisitos"
for cmd in python3 unzip git awk wget free; do
    command -v "$cmd" &>/dev/null && log_info "$cmd ✓" || { log_error "'$cmd' no encontrado."; exit 1; }
done

if ! command -v kaggle &>/dev/null; then
    log_warn "kaggle CLI no encontrado. Instalando..."
    pip install --quiet kaggle || { log_error "No se pudo instalar kaggle."; exit 1; }
fi
log_info "kaggle ✓"

if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    chmod 600 "$HOME/.kaggle/kaggle.json"
    log_info "kaggle.json ✓"
else
    log_warn "~/.kaggle/kaggle.json no encontrado — se intentará descargar solo si faltan ZIPs."
    log_warn "  Para obtenerlo: https://www.kaggle.com/settings → API → 'Create New Token'"
fi

# ── Crear directorios ─────────────────────────────────────────────────────────
mkdir -p \
    "$DATASETS_DIR/nih_chest_xrays" \
    "$DATASETS_DIR/isic_2019" \
    "$DATASETS_DIR/osteoarthritis" \
    "$DATASETS_DIR/luna_lung_cancer" \
    "$DATASETS_DIR/zenodo_13715870" \
    "$DATASETS_DIR/panorama_labels"

# ── Helper: RAM disponible en MB ──────────────────────────────────────────────
# Usa la columna "available" de `free -m` (incluye buffers/cache reciclables).
get_ram_avail_mb() {
    free -m | awk '/^Mem:/{print $7}'
}

# ── Helper: descarga Kaggle ────────────────────────────────────────────────────
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
    log_info "[$display] → Descargando kaggle dataset '$dataset' ..."
    kaggle datasets download -d "$dataset" -p "$dest_dir"
    if [[ -f "$expected_file" ]]; then
        log_info "[$display] ✓ Descarga OK: $(du -sh "$expected_file" | cut -f1)"
    else
        log_error "[$display] Archivo esperado no encontrado: $expected_file"
        log_error "[$display] Contenido: $(ls "$dest_dir" 2>/dev/null || echo 'vacío')"
        return 1
    fi
}

# ── Función de extracción SMART (monitor de RAM con SIGSTOP/SIGCONT) ──────────
#
# Lógica:
#   1. Antes de empezar: si RAM < PAUSE_MB → esperar hasta que suba.
#   2. Lanza unzip en background y guarda su PID.
#   3. Bucle de monitoreo cada RAM_CHECK_S segundos:
#        - RAM < PAUSE_MB  && no pausado → SIGSTOP (suspender), log_ram
#        - RAM > RESUME_MB && pausado    → SIGCONT (reanudar),  log_ram
#        - Si pausado: dormir RAM_SLEEP_S (más largo, dar tiempo al OS)
#        - Si activo:  dormir RAM_CHECK_S (chequeo frecuente)
#   4. Al terminar el bucle: enviar SIGCONT por si quedó suspendido, wait PID.
#   5. Trap SIGINT/SIGTERM: enviar SIGCONT+SIGKILL al unzip antes de salir.
#
extract_archive() {
    local archive="$1" dest="$2" name="$3"

    # ─ Esperar RAM suficiente antes de comenzar ───────────────────────────────
    local avail
    avail=$(get_ram_avail_mb)
    if [[ $avail -lt $RAM_PAUSE_MB ]]; then
        log_ram "[$name] RAM previa al inicio: ${avail}MB < ${RAM_PAUSE_MB}MB — esperando..."
        while true; do
            avail=$(get_ram_avail_mb)
            [[ $avail -ge $RAM_RESUME_MB ]] && break
            log_ram "[$name] RAM: ${avail}MB. Esperando ${RAM_SLEEP_S}s..."
            sleep "$RAM_SLEEP_S"
        done
        log_ram "[$name] RAM OK (${avail}MB). Iniciando."
    fi

    log_info "[$name] Extrayendo (SMART): $(basename "$archive") → $dest"
    log_ram  "[$name] RAM al inicio: ${avail}MB disponibles"
    local t0=$SECONDS

    # ─ Lanzar unzip en background ────────────────────────────────────────────
    unzip -q "$archive" -d "$dest" &
    local UNZIP_PID=$!

    # ─ Cleanup al recibir señales: asegurar que unzip no quede zombi/pausado ──
    trap "log_warn \"[$name] Señal recibida — terminando PID=$UNZIP_PID...\"; \
          kill -CONT $UNZIP_PID 2>/dev/null || true; \
          kill -TERM $UNZIP_PID 2>/dev/null || true; \
          exit 130" INT TERM

    # ─ Monitor de RAM ─────────────────────────────────────────────────────────
    local paused=false
    local pauses=0
    local total_pause_s=0
    local pause_start=0

    while kill -0 "$UNZIP_PID" 2>/dev/null; do
        avail=$(get_ram_avail_mb)

        if [[ $avail -lt $RAM_PAUSE_MB ]] && [[ "$paused" == false ]]; then
            # ── RAM baja → suspender unzip ──────────────────────────────────
            kill -STOP "$UNZIP_PID" 2>/dev/null || true
            paused=true
            pauses=$((pauses + 1))
            pause_start=$SECONDS
            log_ram "[$name] ⏸  Pausa #${pauses} — RAM: ${avail}MB < ${RAM_PAUSE_MB}MB. Unzip suspendido (PID=$UNZIP_PID)."
            sleep "$RAM_SLEEP_S"

        elif [[ $avail -gt $RAM_RESUME_MB ]] && [[ "$paused" == true ]]; then
            # ── RAM recuperada → reanudar ────────────────────────────────────
            total_pause_s=$((total_pause_s + SECONDS - pause_start))
            kill -CONT "$UNZIP_PID" 2>/dev/null || true
            paused=false
            log_ram "[$name] ▶  Reanudando — RAM: ${avail}MB > ${RAM_RESUME_MB}MB."
            sleep "$RAM_CHECK_S"

        elif [[ "$paused" == true ]]; then
            # ── Aún pausado, seguir esperando ────────────────────────────────
            log_ram "[$name] ⏸  Sigue pausado — RAM: ${avail}MB / necesita ${RAM_RESUME_MB}MB. Esperando ${RAM_SLEEP_S}s..."
            sleep "$RAM_SLEEP_S"

        else
            # ── Estado normal ─────────────────────────────────────────────────
            sleep "$RAM_CHECK_S"
        fi
    done

    # ─ Asegurar CONT antes de wait (edge case: STOPped y terminó por otro motivo)
    kill -CONT "$UNZIP_PID" 2>/dev/null || true
    trap - INT TERM

    wait "$UNZIP_PID"
    local rc=$?
    local elapsed=$((SECONDS - t0))
    local active_s=$((elapsed - total_pause_s))

    if [[ $rc -eq 0 ]]; then
        log_info "[$name] ✓ Extraído en ${elapsed}s total (activo: ${active_s}s | pausas: ${pauses}x / ${total_pause_s}s)"
    else
        log_error "[$name] ✗ Error en extracción (código=$rc)"
        return $rc
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# FASE 1 — DESCARGAS EN PARALELO
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 1 — Descargas paralelas"
PIDS=()

_kaggle_dl "nih-chest-xrays/data" \
    "$DATASETS_DIR/nih_chest_xrays" \
    "$DATASETS_DIR/nih_chest_xrays/data.zip" "NIH" &
PIDS+=($!)

_kaggle_dl "andrewmvd/isic-2019" \
    "$DATASETS_DIR/isic_2019" \
    "$DATASETS_DIR/isic_2019/isic-2019.zip" "ISIC" &
PIDS+=($!)

_kaggle_dl "dhruvacube/osteoarthritis" \
    "$DATASETS_DIR/osteoarthritis" \
    "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" "OA" &
PIDS+=($!)

_kaggle_dl "fanbyprinciple/luna-lung-cancer-dataset" \
    "$DATASETS_DIR/luna_lung_cancer" \
    "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" "LUNA" &
PIDS+=($!)

BATCH1="$DATASETS_DIR/zenodo_13715870/batch_1.zip"
if [[ ! -f "$BATCH1" ]]; then
    log_info "[PANCREAS] → Descargando desde Zenodo (~50 GB)..."
    wget --progress=dot:giga -c \
         -O "$BATCH1" \
         "https://zenodo.org/records/13715870/files/batch_1.zip?download=1" &
    PIDS+=($!)
else
    log_info "[PANCREAS] batch_1.zip ya existe ($(du -sh "$BATCH1" | cut -f1)), saltando."
fi

log_info "Esperando ${#PIDS[@]} descarga(s) en paralelo..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { log_error "Descarga PID=$pid falló."; FAIL=$((FAIL+1)); }
done
[[ $FAIL -gt 0 ]] && { log_error "$FAIL descarga(s) fallaron."; exit 1; }
log_info "✓ Todas las descargas completadas."

# ──────────────────────────────────────────────────────────────────────────────
# FASE 2 — EXTRACCIÓN SECUENCIAL (con monitor de RAM)
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 2 — Extracción secuencial (modo SMART — RAM: pausa<${RAM_PAUSE_MB}MB / reanuda>${RAM_RESUME_MB}MB)"

# Estado de RAM antes de comenzar
log_ram "RAM disponible antes de extracciones: $(get_ram_avail_mb)MB"

# NIH ChestXray14
if [[ ! -d "$DATASETS_DIR/nih_chest_xrays/images_001" ]]; then
    extract_archive \
        "$DATASETS_DIR/nih_chest_xrays/data.zip" \
        "$DATASETS_DIR/nih_chest_xrays" "NIH"
else
    log_info "[NIH] Ya extraído (images_001/ existe), saltando."
fi

# ISIC 2019
if [[ ! -d "$DATASETS_DIR/isic_2019/ISIC_2019_Training_Input" ]]; then
    extract_archive \
        "$DATASETS_DIR/isic_2019/isic-2019.zip" \
        "$DATASETS_DIR/isic_2019" "ISIC"
else
    log_info "[ISIC] Ya extraído, saltando."
fi

# Osteoarthritis
if [[ ! -d "$DATASETS_DIR/osteoarthritis/KLGrade" ]]; then
    extract_archive \
        "$DATASETS_DIR/osteoarthritis/osteoarthritis.zip" \
        "$DATASETS_DIR/osteoarthritis" "OA"
else
    log_info "[OA] Ya extraído, saltando."
fi

# LUNA16
if [[ ! -d "$DATASETS_DIR/luna_lung_cancer/candidates_V2" ]]; then
    extract_archive \
        "$DATASETS_DIR/luna_lung_cancer/luna-lung-cancer-dataset.zip" \
        "$DATASETS_DIR/luna_lung_cancer" "LUNA"
else
    log_info "[LUNA] Ya extraído, saltando."
fi

# Pancreas PANORAMA
NII_COUNT=$(find "$DATASETS_DIR/zenodo_13715870" -maxdepth 1 -name "*.nii.gz" 2>/dev/null | wc -l)
if [[ $NII_COUNT -eq 0 ]]; then
    extract_archive \
        "$DATASETS_DIR/zenodo_13715870/batch_1.zip" \
        "$DATASETS_DIR/zenodo_13715870" "PANCREAS"
else
    log_info "[PANCREAS] Ya extraído ($NII_COUNT NIfTIs), saltando."
fi

log_ram "RAM disponible tras extracciones: $(get_ram_avail_mb)MB"

# ──────────────────────────────────────────────────────────────────────────────
# FASE 3 — PREPARACIÓN DE DATASETS
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 3 — Preparación de datasets"

# -- NIH --
NIH_DIR="$DATASETS_DIR/nih_chest_xrays"
for txt in train_val_list.txt test_list.txt; do
    if [[ ! -f "$NIH_DIR/$txt" ]]; then
        log_warn "[NIH] $txt no encontrado. Extrayendo de data.zip..."
        unzip -j -q "$NIH_DIR/data.zip" "$txt" -d "$NIH_DIR"
    fi
    log_info "[NIH] $txt ✓ ($(wc -l < "$NIH_DIR/$txt") entradas)"
done

if [[ ! -d "$NIH_DIR/all_images" ]]; then
    log_info "[NIH] Creando all_images/ con symlinks..."
    mkdir -p "$NIH_DIR/all_images"
    # -P4 -I{} lanzaba 1 proceso por imagen (112 k procesos → OOM).
    # --target-directory agrupa todos los args en lotes grandes (~ARG_MAX):
    # ~10-20 invocaciones de ln en total, sin spike de memoria.
    find "$NIH_DIR" -maxdepth 4 -path "*/images/*.png" -print0 | \
        xargs -0 ln -sf --target-directory="$NIH_DIR/all_images/"
    log_info "[NIH] ✓ $(find "$NIH_DIR/all_images" -maxdepth 1 -type l | wc -l) symlinks creados."
else
    log_info "[NIH] all_images/ ya existe ($(find "$NIH_DIR/all_images" -maxdepth 1 -type l | wc -l) symlinks), saltando."
fi

# -- OA Rodilla --
OA_DIR="$DATASETS_DIR/osteoarthritis"
if [[ ! -d "$OA_DIR/oa_splits/train" ]]; then
    log_info "[OA] Generando splits (KL 0-4 → 3 clases)..."
    python3 - <<PYEOF
import os, shutil, random, sys
from pathlib import Path

os.chdir("$OA_DIR")
src = Path("KLGrade/KLGrade")
dst = Path("oa_splits")

if not src.exists():
    print(f"[OA] ERROR: {src.resolve()} no encontrado", file=sys.stderr)
    sys.exit(1)

mapping = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
all_files = {0: [], 1: [], 2: []}
for kl_str, cls in mapping.items():
    kl_dir = src / kl_str
    if kl_dir.exists():
        all_files[cls].extend(list(kl_dir.glob("*.jpg")) + list(kl_dir.glob("*.png")))
    else:
        print(f"[OA] Advertencia: KL{kl_str}/ no encontrada.")

random.seed(42)
total = 0
for cls, files in all_files.items():
    random.shuffle(files)
    n = len(files); total += n
    cuts = {"train": files[:int(.80*n)], "val": files[int(.80*n):int(.95*n)], "test": files[int(.95*n):]}
    for sname, imgs in cuts.items():
        d = dst / sname / str(cls)
        d.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, d / img.name)

rep = {s: sum(len(list((dst/s/c).iterdir())) for c in "012" if (dst/s/c).exists()) for s in ["train","val","test"]}
print(f"[OA] ✓ Splits — train:{rep['train']} | val:{rep['val']} | test:{rep['test']} | total:{total}")
PYEOF
    log_info "[OA] ✓ oa_splits/ listo."
else
    log_info "[OA] oa_splits/ ya existe, saltando."
fi

# -- LUNA --
log_warn "[LUNA] Dataset Kaggle contiene solo CSVs/máscaras — sin volúmenes CT."
log_warn "[LUNA] Para FASE 0 completa descarga CTs: https://luna16.grand-challenge.org/data/"

# ──────────────────────────────────────────────────────────────────────────────
# FASE 4 — PANORAMA LABELS + COMPARACIÓN
# ──────────────────────────────────────────────────────────────────────────────
log_step "FASE 4 — Panorama Labels + Verificación de alineación"

PANORAMA_DIR="$DATASETS_DIR/panorama_labels"
COMMIT_FILE="$DATASETS_DIR/panorama_labels_commit.txt"

if [[ ! -d "$PANORAMA_DIR/.git" ]]; then
    log_info "[PANORAMA] Clonando DIAGNijmegen/panorama_labels..."
    git clone https://github.com/DIAGNijmegen/panorama_labels.git "$PANORAMA_DIR"
    COMMIT=$(cd "$PANORAMA_DIR" && git rev-parse HEAD)
    echo "$COMMIT" > "$COMMIT_FILE"
    log_info "[PANORAMA] ✓ Clonado. Commit: $COMMIT  →  $COMMIT_FILE"
else
    COMMIT=$(cd "$PANORAMA_DIR" && git rev-parse HEAD)
    echo "$COMMIT" > "$COMMIT_FILE"
    log_info "[PANORAMA] Ya clonado. Commit: $COMMIT"
fi

log_info "[PANORAMA] Comparando IDs zenodo_13715870 vs panorama_labels..."
python3 - <<PYEOF
import sys, re
from pathlib import Path

zenodo_dir = Path("$DATASETS_DIR/zenodo_13715870")
labels_dir = Path("$DATASETS_DIR/panorama_labels")

# Case IDs de los NIfTI
nii_files = sorted(zenodo_dir.glob("*.nii.gz"))
nii_ids = {re.match(r'^(\d+)_', f.name).group(1) for f in nii_files if re.match(r'^(\d+)_', f.name)}
print(f"[PANORAMA] NIfTI en zenodo: {len(nii_files)} archivos → {len(nii_ids)} patient IDs únicos")
if not nii_ids:
    print("[PANORAMA] ⚠  Sin NIfTIS — ¿se extrajo batch_1.zip?")
    sys.exit(0)

# IDs en panorama_labels
label_ids, source = set(), "desconocido"
try:
    import pandas as pd
    for csv_path in sorted(labels_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for col in ['case_id','patient_id','PatientID','pid','id','name','image_id']:
                if col in df.columns:
                    found = df[col].astype(str).str.extract(r'(\d{5,6})')[0].dropna().unique()
                    if len(found) > 0:
                        label_ids.update(found.tolist())
                        source = f"{csv_path.relative_to(labels_dir)} (col='{col}')"
                        break
            if label_ids: break
        except Exception: continue
except ImportError: pass

if not label_ids:
    for f in labels_dir.rglob("*.nii.gz"):
        m = re.match(r'^(\d+)_', f.name)
        if m: label_ids.add(m.group(1))
    if label_ids: source = "NIfTI en panorama_labels"

if not label_ids:
    for fpath in list(labels_dir.rglob("*.json"))[:5] + list(labels_dir.rglob("*.txt"))[:5]:
        try:
            found = re.findall(r'\b(1\d{5})\b', fpath.read_text(errors='ignore'))
            if found: label_ids.update(found); source = str(fpath.relative_to(labels_dir)); break
        except Exception: continue

if not label_ids:
    print("[PANORAMA] ⚠  No se encontraron IDs en panorama_labels. Estructura:")
    for p in sorted(labels_dir.iterdir()): print(f"    {p.name}{'/' if p.is_dir() else ''}")
    sys.exit(0)

print(f"[PANORAMA] IDs en panorama_labels: {len(label_ids)} únicos (fuente: {source})")

matched    = nii_ids & label_ids
only_nii   = nii_ids - label_ids
only_label = label_ids - nii_ids
pct = 100.0 * len(matched) / max(len(nii_ids), 1)

print(f"\n[PANORAMA] ─── Alineación ───")
print(f"  ✓ Con imagen + etiqueta          : {len(matched):>5}  ({pct:.1f}%)")
print(f"  ⚠  Solo en zenodo (sin etiqueta) : {len(only_nii):>5}")
print(f"    Solo en labels (sin NIfTI)    : {len(only_label):>5}")

if pct == 100.0 and not only_nii:
    print("\n[PANORAMA] ✅ Alineación perfecta — procediendo.")
elif pct >= 80.0:
    print(f"\n[PANORAMA] ✅ {pct:.0f}% alineación — suficiente para FASE 0. Procediendo.")
else:
    print(f"\n[PANORAMA] ⚠  {pct:.0f}% alineación. Verifica si hay batch_2.zip en Zenodo.")
PYEOF

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
log_step "SETUP COMPLETADO ✓"
echo ""
echo "Tamaño de datasets:"
du -sh "$DATASETS_DIR"/*/  2>/dev/null | sort -h
echo ""
log_ram "RAM final disponible: $(get_ram_avail_mb)MB"
log_info "Log completo: $LOG_FILE"
log_info ""
log_info "Siguiente paso — FASE 0:"
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
