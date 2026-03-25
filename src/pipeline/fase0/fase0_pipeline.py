#!/usr/bin/env python3
"""
fase0_pipeline.py — Orquestador principal de Fase 0
=====================================================
Fase 0 — Preparación de Datos | Proyecto MoE Médico

Responsabilidad única: ejecutar los 8 pasos de Fase 0 en orden secuencial
con idempotencia, registro de tiempos, manejo de errores y reporte final.

Pasos:
  0 — Verificar prerequisites (7z, wget, git, kaggle, espacio disco)
  1 — Descargar datasets          [descargar.py]
  2 — Extraer archivos            [extraer.py]
  3 — Post-procesado NIH          [pre_chestxray14.py]
  4 — Etiquetas páncreas          [integrado aquí]
  5 — Splits 80/10/10             [pre_modelo.py]
  6 — Datos 3D (LUNA + páncreas)  [pre_embeddings.py]
  7 — CvT-13 instalar/verificar   [integrado aquí]
  8 — Reporte final               [integrado aquí]

Uso:
  python3 fase0/fase0_pipeline.py
  python3 fase0/fase0_pipeline.py --solo nih isic --solo_pasos 1 2 5
  python3 fase0/fase0_pipeline.py --skip pancreas --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Añadir project root y fase0 al path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Logging ──────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            str(LOGS_DIR / "fase0_pipeline.log"),
            mode="w",
        ),
    ],
)
log = logging.getLogger("fase0.pipeline")

# ── Constantes ───────────────────────────────────────────────────────────────
ALL_DATASETS = {"nih", "isic", "oa", "luna_meta", "luna_ct", "pancreas", "panorama"}
SEED = 42
PDAC_VALUE = 3


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: _process_mask (top-level para ser pickleable)
# ══════════════════════════════════════════════════════════════════════════════


def _process_mask(args):
    """Procesa una sola máscara .nii.gz y retorna fila de resultado."""
    lf_str, pdac_value = args
    import nibabel as nib

    lf = Path(lf_str)
    case_id = lf.name.replace(".nii.gz", "")
    try:
        data = np.asarray(nib.load(str(lf)).dataobj)
        unique_vals = sorted(np.unique(np.round(data)).astype(int).tolist())
        has_pdac = int(pdac_value in unique_vals)
        del data
        return {
            "case_id": case_id,
            "label": has_pdac,
            "label_source": "mask_value_3",
            "mask_values": str(unique_vals),
        }
    except Exception as e:
        return {
            "case_id": case_id,
            "label": -1,
            "label_source": "ERROR",
            "mask_values": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Resolve active datasets
# ══════════════════════════════════════════════════════════════════════════════


def resolve_active(solo=None, skip=None):
    # type: (list|None, list|None) -> set
    """Determina qué datasets están activos."""
    if solo:
        active = set()
        for s in solo:
            s = s.lower()
            if s == "luna":
                active.update({"luna_meta", "luna_ct"})
            elif s in ALL_DATASETS:
                active.add(s)
            else:
                log.warning("Dataset desconocido: %s", s)
        return active

    active = set(ALL_DATASETS)
    if skip:
        for s in skip:
            s = s.lower()
            if s == "luna":
                active -= {"luna_meta", "luna_ct"}
            else:
                active.discard(s)
    return active


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 0 — Prerequisites
# ══════════════════════════════════════════════════════════════════════════════


def paso0_prerequisites(active, dry_run=False):
    # type: (set, bool) -> dict
    """Verifica prerequisitos del sistema."""
    log.info("╔══ Paso 0: Verificar prerequisites ══╗")
    issues = []

    # 7z — OBLIGATORIO Y BLOQUEANTE
    has_7z = shutil.which("7z") is not None
    if not has_7z:
        msg = "7z no encontrado. Instalar: sudo apt-get install p7zip-full"
        log.error("  ✗ %s", msg)
        issues.append(("BLOQUEANTE", msg))
    else:
        log.info("  ✓ 7z disponible")

    # wget
    if shutil.which("wget"):
        log.info("  ✓ wget disponible")
    else:
        log.warning("  ⚠ wget no encontrado — descargas con wget fallarán")
        issues.append(("WARNING", "wget no disponible"))

    # git
    if shutil.which("git"):
        log.info("  ✓ git disponible")
    else:
        log.warning("  ⚠ git no encontrado — clone de panorama_labels fallará")
        issues.append(("WARNING", "git no disponible"))

    # kaggle CLI
    if any(d in active for d in {"isic", "oa", "luna_meta", "nih"}):
        if shutil.which("kaggle"):
            log.info("  ✓ kaggle CLI disponible")
            # Verificar credenciales
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_json.exists():
                log.info("  ✓ kaggle.json encontrado")
            else:
                log.warning("  ⚠ ~/.kaggle/kaggle.json no encontrado")
                issues.append(("WARNING", "kaggle.json no encontrado"))
        else:
            log.warning("  ⚠ kaggle CLI no disponible")
            issues.append(("WARNING", "kaggle CLI no disponible"))

    # Espacio en disco
    try:
        usage = shutil.disk_usage(str(DATASETS_DIR))
        free_gb = usage.free / 1e9
        log.info("  Espacio libre: %.1f GB", free_gb)
        if free_gb < 150:
            log.warning("  ⚠ Menos de 150 GB libres — puede ser insuficiente")
            issues.append(("WARNING", "Solo {:.0f} GB libres".format(free_gb)))
    except Exception:
        pass

    # Evaluar resultado
    bloqueantes = [i for i in issues if i[0] == "BLOQUEANTE"]
    if bloqueantes and not dry_run:
        log.error("  ✗ Prerequisites bloqueantes encontrados — abortando")
        return {"status": "❌", "issues": issues}

    status = "✅" if not issues else "⚠️"
    log.info("  Paso 0: %s (%d issues)", status, len(issues))
    return {"status": status, "issues": issues}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 1 — Descargar
# ══════════════════════════════════════════════════════════════════════════════


def paso1_descargar(active, luna_subsets=None, dry_run=False):
    # type: (set, list|None, bool) -> dict
    log.info("╔══ Paso 1: Descargar datasets ══╗")
    try:
        from descargar import run_downloads

        detail = run_downloads(
            DATASETS_DIR, active, luna_subsets=luna_subsets, dry_run=dry_run
        )
        all_ok = all(v for v in detail.values()) if detail else True
        return {"status": "✅" if all_ok else "⚠️", "detail": detail}
    except Exception as e:
        log.error("  Paso 1 falló: %s", e)
        return {"status": "❌", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 2 — Extraer
# ══════════════════════════════════════════════════════════════════════════════


def paso2_extraer(active, disco=False, luna_subsets=None, dry_run=False):
    # type: (set, bool, list|None, bool) -> dict
    log.info("╔══ Paso 2: Extraer archivos ══╗")
    try:
        from extraer import run_extractions

        detail = run_extractions(
            DATASETS_DIR, active, luna_subsets=luna_subsets, disco=disco
        )
        all_ok = all(v for v in detail.values()) if detail else True
        return {"status": "✅" if all_ok else "⚠️", "detail": detail}
    except Exception as e:
        log.error("  Paso 2 falló: %s", e)
        return {"status": "❌", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 3 — Post-procesado NIH
# ══════════════════════════════════════════════════════════════════════════════


def paso3_pre_chestxray14(active, dry_run=False):
    # type: (set, bool) -> dict
    if "nih" not in active:
        log.info("╔══ Paso 3: Saltado (NIH no activo) ══╗")
        return {"status": "⏭️", "skipped": True}

    log.info("╔══ Paso 3: Post-procesado NIH ChestXray14 ══╗")
    try:
        from pre_chestxray14 import run_pre_chestxray14

        return run_pre_chestxray14(DATASETS_DIR)
    except Exception as e:
        log.error("  Paso 3 falló: %s", e)
        return {"status": "❌", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 4 — Etiquetas páncreas
# ══════════════════════════════════════════════════════════════════════════════


def paso4_pancreas_labels(active, dry_run=False):
    # type: (set, bool) -> dict
    if "pancreas" not in active:
        log.info("╔══ Paso 4: Saltado (páncreas no activo) ══╗")
        return {"status": "⏭️", "skipped": True}

    log.info("╔══ Paso 4: Etiquetas binarias páncreas ══╗")

    labels_dir = DATASETS_DIR / "panorama_labels" / "automatic_labels"
    zenodo_dir = DATASETS_DIR / "zenodo_13715870"
    out_csv = DATASETS_DIR / "pancreas_labels_binary.csv"

    # Idempotencia
    if out_csv.exists() and out_csv.stat().st_size > 100:
        df_check = pd.read_csv(out_csv)
        valid = df_check[df_check["label"] >= 0]
        if len(valid) > 10:
            log.info(
                "  pancreas_labels_binary.csv ya existe (%d filas, %d válidas)",
                len(df_check),
                len(valid),
            )
            return {"status": "✅", "skipped": True, "rows": len(df_check)}

    if not labels_dir.is_dir():
        # Try to restore working tree if repo exists but files are missing
        repo_dir = DATASETS_DIR / "panorama_labels"
        if (repo_dir / ".git").is_dir():
            log.info(
                "  panorama_labels repo exists but automatic_labels/ missing — restoring working tree..."
            )
            try:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "checkout", "--", "automatic_labels/"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except Exception as e:
                log.error("  git checkout failed: %s", e)
        if not labels_dir.is_dir():
            log.warning("  panorama_labels/automatic_labels/ no encontrado")
            return {"status": "⚠️", "reason": "sin máscaras"}

    # Fix 7: derivar etiquetas binarias desde máscaras
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    n_total = len(label_files)
    log.info("  Máscaras a procesar: %d", n_total)

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    args_list = [(str(lf), PDAC_VALUE) for lf in label_files]
    rows = [None] * n_total
    done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_process_mask, arg): i for i, arg in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rows[idx] = future.result()
            except Exception as e:
                lf_str = args_list[idx][0]
                rows[idx] = {
                    "case_id": Path(lf_str).name.replace(".nii.gz", ""),
                    "label": -1,
                    "label_source": "FUTURE_ERROR",
                    "mask_values": str(e),
                }
            done += 1
            if done % 100 == 0 or done == n_total:
                elapsed = time.time() - t0
                log.info("  %d/%d máscaras [%.0fs]", done, n_total, elapsed)

    # Fix 8: volúmenes sin máscara → asumir PDAC negativo
    mask_ids = {r["case_id"] for r in rows if r is not None}
    if zenodo_dir.is_dir():
        for nf in sorted(zenodo_dir.glob("*.nii.gz")):
            stem = nf.name.replace(".nii.gz", "")
            parts = stem.rsplit("_", 1)
            case_id = parts[0] if len(parts) > 1 else stem
            if case_id not in mask_ids:
                rows.append(
                    {
                        "case_id": case_id,
                        "label": 0,
                        "label_source": "assumed_negative_no_mask",
                        "mask_values": "N/A",
                    }
                )

    df = pd.DataFrame([r for r in rows if r is not None])
    # Actualizar -1 sin máscara a 0
    no_mask = df["label_source"] == "no_mask_in_labels"
    df.loc[no_mask, "label"] = 0
    df.loc[no_mask, "label_source"] = "assumed_negative_no_mask"

    df.to_csv(out_csv, index=False)

    valid = df[df["label"] >= 0]
    pdac_pos = int((valid["label"] == 1).sum())
    pdac_neg = int((valid["label"] == 0).sum())
    unknown = int((df["label"] == -1).sum())

    log.info(
        "  PDAC+=%d  PDAC-=%d  unknown=%d  total=%d",
        pdac_pos,
        pdac_neg,
        unknown,
        len(df),
    )
    return {
        "status": "✅",
        "pdac_pos": pdac_pos,
        "pdac_neg": pdac_neg,
        "unknown": unknown,
        "total": len(df),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 5 — Splits 80/10/10
# ══════════════════════════════════════════════════════════════════════════════


def paso5_splits(active, dry_run=False):
    # type: (set, bool) -> dict
    log.info("╔══ Paso 5: Splits 80/10/10 ══╗")
    try:
        from pre_modelo import run_splits

        return run_splits(DATASETS_DIR, active)
    except Exception as e:
        log.error("  Paso 5 falló: %s", e)
        return {"status": "❌", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 6 — Datos 3D
# ══════════════════════════════════════════════════════════════════════════════


def paso6_pre_embeddings(
    active, workers=6, neg_ratio=10, max_neg=None, luna_subsets=None, dry_run=False
):
    # type: (set, int, int, int|None, list|None, bool) -> dict
    log.info("╔══ Paso 6: Datos 3D (LUNA + Páncreas) ══╗")
    try:
        from pre_embeddings import run_pre_embeddings

        return run_pre_embeddings(
            DATASETS_DIR,
            active,
            workers=workers,
            neg_ratio=neg_ratio,
            max_neg=max_neg,
            luna_subsets=luna_subsets,
            dry_run=dry_run,
        )
    except Exception as e:
        log.error("  Paso 6 falló: %s", e)
        return {"status": "❌", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 7 — CvT-13
# ══════════════════════════════════════════════════════════════════════════════

CVT13_MODULE_PATH = SCRIPTS_DIR / "cvt13_backbone.py"


def paso7_cvt13(dry_run=False):
    # type: (bool) -> dict
    log.info("╔══ Paso 7: CvT-13 ══╗")

    # 7a: Instalar dependencias
    for pkg in ["transformers", "einops"]:
        try:
            __import__(pkg)
            log.info("  ✓ %s ya instalado", pkg)
        except ImportError:
            log.info("  → Instalando %s...", pkg)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            )

    # 7b: Escribir cvt13_backbone.py si no existe o si es referencia vieja
    if CVT13_MODULE_PATH.exists():
        log.info("  ✓ cvt13_backbone.py ya existe")
    else:
        log.info(
            "  Paso 7: backbone CvT-13 gestionado nativamente en Fase 1 — script externo no requerido"
        )

    # 7c: Compatibilidad CvT-13 ahora es código nativo en Fase 1
    _cvt13_native = PROJECT_ROOT / "src" / "pipeline" / "fase1" / "backbone_cvt13.py"
    if _cvt13_native.exists():
        log.info(
            "  ✓ Compatibilidad CvT-13 → src/pipeline/fase1/backbone_cvt13.py (nativo)"
        )
    else:
        log.info(
            "  ⚠ backbone_cvt13.py no encontrado en src/pipeline/fase1/ — "
            "ejecutar la refactorización de Fase 1"
        )

    # 7d: Verificar CvT-13 con test de sanidad
    if dry_run:
        log.info("  [DRY-RUN] Saltando test de sanidad CvT-13")
        return {"status": "✅", "dry_run": True}

    if CVT13_MODULE_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(CVT13_MODULE_PATH)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                log.info("    %s", line)
            log.info("  ✓ Test de sanidad CvT-13 OK")
            return {"status": "✅"}
        else:
            log.error("  ✗ Test de sanidad CvT-13 FALLÓ")
            for line in (result.stdout + result.stderr).strip().splitlines():
                log.error("    %s", line)
            return {"status": "❌", "error": "test de sanidad falló"}

    return {"status": "⚠️", "reason": "cvt13_backbone.py no disponible"}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 8 — Reporte final
# ══════════════════════════════════════════════════════════════════════════════


def paso8_reporte(all_results, timings, active):
    # type: (dict, dict, set) -> dict
    log.info("╔══ Paso 8: Reporte final ══╗")

    import datetime

    report_path = Path(__file__).resolve().parent / "fase0_report.md"
    # Copia con timestamp en logs/ para historial de corridas
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_report_path = LOGS_DIR / "fase0_report_{}.md".format(ts)
    lines = [
        "# Fase 0 — Reporte de Preparación de Datos",
        "",
        "## Estado por paso",
        "",
        "| Paso | Descripción | Estado | Tiempo |",
        "|------|-------------|--------|--------|",
    ]

    paso_nombres = {
        0: "Prerequisites",
        1: "Descargar datasets",
        2: "Extraer archivos",
        3: "Post-procesado NIH",
        4: "Etiquetas páncreas",
        5: "Splits 80/10/10",
        6: "Datos 3D",
        7: "CvT-13",
        8: "Reporte",
    }

    for paso_num in range(9):
        key = "paso{}".format(paso_num)
        nombre = paso_nombres.get(paso_num, "?")
        r = all_results.get(key, {})
        status = r.get("status", "—")
        t = timings.get(key, 0)
        t_str = "{:.1f}s".format(t) if t > 0 else "—"
        lines.append("| {} | {} | {} | {} |".format(paso_num, nombre, status, t_str))

    lines.extend(["", "## Datasets activos", ""])
    for ds in sorted(active):
        lines.append("- {}".format(ds))

    # Conteos por dataset si están disponibles
    paso5 = all_results.get("paso5", {})
    if paso5 and not isinstance(paso5, str):
        lines.extend(["", "## Splits generados", ""])
        for ds_name, info in paso5.items():
            if isinstance(info, dict):
                lines.append("### {}".format(ds_name.upper()))
                for k, v in info.items():
                    lines.append("- {}: {}".format(k, v))
                lines.append("")

    # Comando de Fase 1
    fase1_path = PROJECT_ROOT / "src" / "pipeline" / "fase1" / "fase1_pipeline.py"

    lines.extend(
        [
            "",
            "## Comando para Fase 1",
            "",
            "> **Nota:** El backbone mostrado es el valor por defecto "
            "(`vit_tiny_patch16_224`). Editar antes de ejecutar si se desea "
            "usar `swin_tiny_patch4_window7_224` o `cvt_13`.",
            "",
            "```bash",
            "python3 {} \\".format(fase1_path),
            "    --backbone vit_tiny_patch16_224 \\",
            "    --batch_size 256 --workers 8 \\",
            "    --output_dir embeddings/vit_tiny \\",
            "    --chest_csv datasets/nih_chest_xrays/Data_Entry_2017.csv \\",
            "    --chest_imgs datasets/nih_chest_xrays/all_images \\",
            "    --nih_train_list datasets/nih_chest_xrays/splits/nih_train_list.txt \\",
            "    --nih_val_list datasets/nih_chest_xrays/splits/nih_val_list.txt \\",
            "    --nih_test_list datasets/nih_chest_xrays/splits/nih_test_list.txt \\",
            "    --chest_view_filter PA \\",
            "    --chest_bbox_csv datasets/nih_chest_xrays/BBox_List_2017.csv \\",
            "    --isic_train_csv datasets/isic_2019/splits/isic_train.csv \\",
            "    --isic_val_csv datasets/isic_2019/splits/isic_val.csv \\",
            "    --isic_test_csv datasets/isic_2019/splits/isic_test.csv \\",
            "    --isic_imgs datasets/isic_2019/isic_images \\",
            "    --oa_root datasets/osteoarthritis/oa_splits \\",
            "    --luna_patches_dir datasets/luna_lung_cancer/patches \\",
            "    --luna_csv datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv \\",
            "    --pancreas_splits_csv datasets/zenodo_13715870/pancreas_splits.csv \\",
            "    --pancreas_nii_dir datasets/zenodo_13715870 \\",
            "    --pancreas_fold 1 \\",
            "    --pancreas_roi_strategy A",
            "```",
            "",
            "---",
            "Generado automáticamente por fase0_pipeline.py",
        ]
    )

    content = "\n".join(lines)
    report_path.write_text(content, encoding="utf-8")
    log_report_path.write_text(content, encoding="utf-8")
    log.info("  ✓ Reporte: %s", report_path)
    log.info("  ✓ Copia en logs/: %s", log_report_path)
    return {"status": "✅", "report_path": str(report_path)}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Fase 0 — Orquestador de Preparación de Datos",
    )
    parser.add_argument(
        "--solo",
        nargs="+",
        default=None,
        help="Solo estos datasets (ej: --solo nih isic luna)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help="Saltar estos datasets (ej: --skip pancreas)",
    )
    parser.add_argument(
        "--solo_pasos",
        nargs="+",
        type=int,
        default=None,
        help="Solo ejecutar pasos específicos (0-8)",
    )
    parser.add_argument(
        "--disco",
        action="store_true",
        help="Modo disco — borrar ZIP tras extracción",
    )
    parser.add_argument(
        "--luna_subsets",
        nargs="+",
        type=int,
        default=None,
        help="Subsets CT de LUNA a procesar (0-9)",
    )
    parser.add_argument(
        "--max_neg",
        type=int,
        default=None,
        help="Máximo negativos LUNA para extracción de parches",
    )
    parser.add_argument(
        "--neg_ratio",
        type=int,
        default=10,
        help="Ratio negativos:positivos LUNA (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Workers paralelos para LUNA (default: 6)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Solo imprime lo que haría sin ejecutar",
    )
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  FASE 0 — Preparación de Datos | Proyecto MoE Médico")
    log.info("=" * 65)

    active = resolve_active(args.solo, args.skip)
    log.info("Datasets activos: %s", sorted(active))

    pasos = set(range(9))
    if args.solo_pasos is not None:
        pasos = set(args.solo_pasos)
    log.info("Pasos a ejecutar: %s", sorted(pasos))

    all_results = {}
    timings = {}

    def run_paso(num, func, *func_args, **func_kwargs):
        if num not in pasos:
            log.info("╔══ Paso %d: Saltado (--solo_pasos) ══╗", num)
            all_results["paso{}".format(num)] = {"status": "⏭️"}
            return
        t0 = time.time()
        try:
            result = func(*func_args, **func_kwargs)
        except Exception as e:
            log.error("Paso %d falló con excepción: %s", num, e)
            result = {"status": "❌", "error": str(e)}
        elapsed = time.time() - t0
        key = "paso{}".format(num)
        all_results[key] = result
        timings[key] = elapsed
        log.info(
            "Paso %d completado en %.1fs — %s", num, elapsed, result.get("status", "?")
        )

    # Ejecutar pasos
    run_paso(0, paso0_prerequisites, active, args.dry_run)

    # Si Paso 0 tiene bloqueantes, abortar
    p0 = all_results.get("paso0", {})
    if p0.get("status") == "❌" and not args.dry_run:
        log.error("Abortando — prerequisites bloqueantes.")
        return

    run_paso(1, paso1_descargar, active, args.luna_subsets, args.dry_run)
    run_paso(2, paso2_extraer, active, args.disco, args.luna_subsets, args.dry_run)
    run_paso(3, paso3_pre_chestxray14, active, args.dry_run)
    run_paso(4, paso4_pancreas_labels, active, args.dry_run)
    run_paso(5, paso5_splits, active, args.dry_run)
    run_paso(
        6,
        paso6_pre_embeddings,
        active,
        args.workers,
        args.neg_ratio,
        args.max_neg,
        args.luna_subsets,
        args.dry_run,
    )
    run_paso(7, paso7_cvt13, args.dry_run)
    run_paso(8, paso8_reporte, all_results, timings, active)

    # Resumen final
    log.info("")
    log.info("=" * 65)
    log.info("  FASE 0 COMPLETADA")
    log.info("=" * 65)
    for num in range(9):
        key = "paso{}".format(num)
        r = all_results.get(key, {})
        t = timings.get(key, 0)
        log.info("  Paso %d: %s (%.1fs)", num, r.get("status", "—"), t)
    total_time = sum(timings.values())
    log.info("  Tiempo total: %.1fs (%.1f min)", total_time, total_time / 60)


if __name__ == "__main__":
    main()
