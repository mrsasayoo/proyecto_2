#!/usr/bin/env python3
"""
descargar.py — Descarga de todos los datasets desde fuentes remotas
====================================================================
Fase 0 — Preparación de Datos | Proyecto MoE Médico

Responsabilidad única: descargar datasets al disco local.
No extrae, no valida estructura interna, no crea splits.

Origen: funciones de descarga de scripts/setup_datasets.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ── psutil: instalar si no está ──────────────────────────────────────────────
try:
    import psutil
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "psutil"]
    )
    import psutil

log = logging.getLogger("fase0.descargar")

# ── Constantes ────────────────────────────────────────────────────────────────

KAGGLE_API_URL = "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"

ZENODO_LUNA_URL_PART1 = (
    "https://zenodo.org/records/3723295/files/subset{i}.zip?download=1"
)
ZENODO_LUNA_URL_PART2 = (
    "https://zenodo.org/records/2596479/files/subset{i}.zip?download=1"
)
# Batches 1-4 son registros Zenodo independientes (CC-BY-NC 4.0)
# Batch 1: zenodo.org/records/13715870  Batch 2: zenodo.org/records/13742336
# Batch 3: zenodo.org/records/11034011  Batch 4: zenodo.org/records/10999754
ZENODO_PANCREAS_BATCHES = {
    1: {
        "url": "https://zenodo.org/records/13715870/files/batch_1.zip?download=1",
        "md5": "b3b3669a82696b954b449c27a9d85074",
        "size_bytes": 49_338_585_294,
    },
    2: {
        "url": "https://zenodo.org/records/13742336/files/batch_2.zip?download=1",
        "md5": "9668a43c24d5eb3473fbaa979b1dbaf8",
        "size_bytes": 49_284_974_263,
    },
    3: {
        "url": "https://zenodo.org/records/11034011/files/batch_3.zip?download=1",
        "md5": "9d852d09d750fd2e2a2e32a371d3bdd8",
        "size_bytes": 49_287_047_956,
    },
    4: {
        "url": "https://zenodo.org/records/10999754/files/batch_4.zip?download=1",
        "md5": "f2820a214aa24fa90daeedbaf99d0609",
        "size_bytes": 46_271_075_077,
    },
}

PANORAMA_REPO = "https://github.com/DIAGNijmegen/panorama_labels.git"

# Tamaños mínimos de ZIPs (bytes) — detectar archivos corruptos/incompletos
MIN_ZIP_SIZES = {
    "nih": 35 * 1024**3,
    "isic": 5 * 1024**3,
    "oa": 200 * 1024**2,
    "luna_meta": 100 * 1024**2,
    "luna_ct": 4 * 1024**3,
    "pancreas": 40
    * 1024**3,  # por batch; el más pequeño (batch_4) es 46.3 GB → 40 GB es safe
}

# Datasets que requieren kaggle CLI
KAGGLE_DATASETS = {"isic", "oa", "luna_meta"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def file_size_human(path):
    # type: (Path) -> str
    try:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return "{:.1f} {}".format(size, unit)
            size /= 1024.0
        return "{:.1f} PB".format(size)
    except Exception:
        return "?"


def read_kaggle_creds():
    # type: () -> tuple
    """Devuelve (username, key) o (None, None)."""
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        return None, None
    try:
        with open(path) as f:
            data = json.load(f)
        user = data.get("username")
        key = data.get("key")
        if user and key:
            return str(user), str(key)
    except Exception:
        pass
    return None, None


def check_prerequisites(active):
    # type: (set) -> bool
    """Verifica herramientas del sistema y credenciales Kaggle."""
    log.info("Verificando prerequisitos de descarga...")
    ok = True

    for cmd in ("wget", "git"):
        if shutil.which(cmd):
            log.info("  %s OK (%s)", cmd, shutil.which(cmd))
        else:
            log.error(
                "  %s NO encontrado — instalar con: sudo apt-get install %s", cmd, cmd
            )
            ok = False

    # kaggle CLI
    needs_kaggle = bool(active & KAGGLE_DATASETS)
    if needs_kaggle:
        if not shutil.which("kaggle"):
            log.warning("  kaggle CLI no encontrado — instalando via pip...")
            for pip_extra in (["--user"], ["--break-system-packages"], []):
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet"]
                        + pip_extra
                        + ["kaggle"]
                    )
                    break
                except subprocess.CalledProcessError:
                    continue
        if shutil.which("kaggle"):
            log.info("  kaggle OK")
        else:
            log.error(
                "  kaggle CLI necesario para %s pero no se pudo instalar.",
                ", ".join(sorted(active & KAGGLE_DATASETS)),
            )
            ok = False

    # Credenciales Kaggle
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if needs_kaggle:
        if not kaggle_json.exists():
            log.error("  ~/.kaggle/kaggle.json NO encontrado.")
            log.error(
                "  1. https://www.kaggle.com/settings -> API -> 'Create New Token'"
            )
            log.error(
                "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
            )
            log.error("  3. chmod 600 ~/.kaggle/kaggle.json")
            ok = False
        else:
            kaggle_json.chmod(0o600)
            user, key = read_kaggle_creds()
            if user and key:
                log.info("  kaggle.json OK (usuario: %s)", user)
            else:
                log.error("  kaggle.json existe pero falta 'username' o 'key'.")
                ok = False

    return ok


# ── Funciones de descarga ─────────────────────────────────────────────────────


def download_wget(url, dest, tag, extra_flags=None, min_size_bytes=None):
    # type: (str, Path, str, list|None, int|None) -> bool
    """Descarga con wget. Idempotente: salta si ya existe con tamaño suficiente."""
    if dest.exists() and dest.stat().st_size > 0:
        if min_size_bytes is not None and dest.stat().st_size < min_size_bytes:
            log.warning(
                "[%s] %s existe pero pesa solo %s (< %.0f MB mínimo) — continuando.",
                tag,
                dest.name,
                file_size_human(dest),
                min_size_bytes / 1024**2,
            )
        else:
            log.info(
                "[%s] Ya existe: %s (%s), saltando.",
                tag,
                dest.name,
                file_size_human(dest),
            )
            return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--continue",
        "--tries=10",
        "--retry-connrefused",
        "--waitretry=60",
        "--timeout=300",
        "--progress=dot:giga",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(["-O", str(dest), url])
    log.info("[%s] Descargando -> %s", tag, dest)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, timeout=86400)
        if proc.returncode != 0:
            log.error("[%s] wget terminó con código %d.", tag, proc.returncode)
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
            return False
    except subprocess.TimeoutExpired:
        log.error("[%s] wget timeout (24h). Reejecutar reanudará.", tag)
        return False
    except Exception as e:
        log.error("[%s] wget error: %s", tag, e)
        return False
    elapsed = time.time() - t0
    log.info("[%s] OK %s en %.0fs", tag, file_size_human(dest), elapsed)
    return True


def download_kaggle_cli(dataset_slug, dest_dir, expected_zip, tag, min_size_bytes=None):
    # type: (str, Path, Path, str, int|None) -> bool
    """Descarga con kaggle CLI. Detecta renombres automáticos."""
    if expected_zip.exists() and expected_zip.stat().st_size > 0:
        if min_size_bytes is not None and expected_zip.stat().st_size < min_size_bytes:
            log.warning(
                "[%s] %s existe pero pesa solo %s — re-descargando.",
                tag,
                expected_zip.name,
                file_size_human(expected_zip),
            )
            expected_zip.unlink()
        else:
            log.info(
                "[%s] Ya existe: %s (%s), saltando.",
                tag,
                expected_zip.name,
                file_size_human(expected_zip),
            )
            return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    before = set(dest_dir.glob("*.zip"))

    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir)]
    log.info("[%s] $ %s", tag, " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            log.error("[%s] kaggle CLI falló: %s", tag, proc.stderr.strip()[:300])
            return False
    except subprocess.TimeoutExpired:
        log.error("[%s] kaggle CLI timeout (2h).", tag)
        return False
    except Exception as e:
        log.error("[%s] kaggle CLI error: %s", tag, e)
        return False

    after = set(dest_dir.glob("*.zip"))
    new_zips = after - before

    if expected_zip.exists():
        if min_size_bytes is not None and expected_zip.stat().st_size < min_size_bytes:
            log.error(
                "[%s] %s descargado pero demasiado pequeño — corrupto.",
                tag,
                expected_zip.name,
            )
            return False
        elapsed = time.time() - t0
        log.info("[%s] OK %s en %.0fs", tag, file_size_human(expected_zip), elapsed)
        return True

    if new_zips:
        new_zip = sorted(new_zips)[0]
        log.warning(
            "[%s] Kaggle descargó '%s' — renombrando a '%s'",
            tag,
            new_zip.name,
            expected_zip.name,
        )
        new_zip.rename(expected_zip)
        if min_size_bytes is not None and expected_zip.stat().st_size < min_size_bytes:
            log.error(
                "[%s] %s descargado pero demasiado pequeño — corrupto.",
                tag,
                expected_zip.name,
            )
            return False
        elapsed = time.time() - t0
        log.info("[%s] OK %s en %.0fs", tag, file_size_human(expected_zip), elapsed)
        return True

    log.error("[%s] No se encontró ningún ZIP nuevo en %s", tag, dest_dir)
    return False


def download_nih(datasets_dir):
    # type: (Path) -> bool
    """NIH ChestXray14 via wget + Kaggle REST API."""
    user, key = read_kaggle_creds()
    if not user or not key:
        log.error("[NIH] Credenciales Kaggle no disponibles.")
        return False
    dest = datasets_dir / "nih_chest_xrays" / "data.zip"
    log.info("[NIH] Descargando via wget + Kaggle REST API (~42 GB)...")
    return download_wget(
        KAGGLE_API_URL,
        dest,
        "NIH",
        extra_flags=[
            "--auth-no-challenge",
            "--http-user=" + user,
            "--http-password=" + key,
        ],
        min_size_bytes=MIN_ZIP_SIZES["nih"],
    )


def download_isic(datasets_dir):
    # type: (Path) -> bool
    # Skip download if extracted data already present (BUG 2 fix)
    isic_img_dir = datasets_dir / "isic_2019" / "ISIC_2019_Training_Input"
    if isic_img_dir.is_dir():
        jpg_count = len(list(isic_img_dir.glob("*.jpg")))
        if jpg_count >= 25000:
            log.info(
                "[ISIC] Datos ya extraídos (%d JPGs en %s), saltando descarga.",
                jpg_count,
                isic_img_dir.name,
            )
            return True
    return download_kaggle_cli(
        "andrewmvd/isic-2019",
        datasets_dir / "isic_2019",
        datasets_dir / "isic_2019" / "isic-2019.zip",
        "ISIC",
        min_size_bytes=MIN_ZIP_SIZES["isic"],
    )


def download_oa(datasets_dir):
    # type: (Path) -> bool
    # Skip download if extracted data already present (mismo patrón que download_isic).
    # KLGrade/ es la salida directa de la extracción del ZIP — si tiene imágenes,
    # el dataset ya está en disco y no necesitamos volver a descargar los ~9.8 GB.
    oa_klgrade_dir = datasets_dir / "osteoarthritis" / "KLGrade"
    if oa_klgrade_dir.is_dir():
        img_count = sum(
            1
            for p in oa_klgrade_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if img_count >= 1000:
            log.info(
                "[OA] Datos ya extraídos (%d imágenes en %s), saltando descarga.",
                img_count,
                oa_klgrade_dir.name,
            )
            return True
    return download_kaggle_cli(
        "dhruvacube/osteoarthritis",
        datasets_dir / "osteoarthritis",
        datasets_dir / "osteoarthritis" / "osteoarthritis.zip",
        "OA",
        min_size_bytes=MIN_ZIP_SIZES["oa"],
    )


def download_luna_meta(datasets_dir):
    # type: (Path) -> bool
    return download_kaggle_cli(
        "fanbyprinciple/luna-lung-cancer-dataset",
        datasets_dir / "luna_lung_cancer",
        datasets_dir / "luna_lung_cancer" / "luna-lung-cancer-dataset.zip",
        "LUNA",
        min_size_bytes=MIN_ZIP_SIZES["luna_meta"],
    )


def download_luna_ct(datasets_dir, subsets):
    # type: (Path, list) -> bool
    if subsets is None:
        subsets = list(range(10))
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    ct_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for i in subsets:
        # Skip download if extracted subset already present (BUG 2 fix)
        subset_dir = ct_dir / "subset{}".format(i)
        if subset_dir.is_dir():
            mhd_count = len(list(subset_dir.glob("*.mhd")))
            raw_count = len(list(subset_dir.glob("*.raw")))
            if mhd_count >= 88 and mhd_count == raw_count:
                log.info(
                    "[LUNA-CT%d] Datos ya extraídos (%d .mhd + %d .raw), saltando descarga.",
                    i,
                    mhd_count,
                    raw_count,
                )
                continue
        url_template = ZENODO_LUNA_URL_PART1 if i <= 6 else ZENODO_LUNA_URL_PART2
        url = url_template.format(i=i)
        dest = ct_dir / "subset{}.zip".format(i)
        ok = download_wget(
            url, dest, "LUNA-CT{}".format(i), min_size_bytes=MIN_ZIP_SIZES["luna_ct"]
        )
        if not ok:
            all_ok = False
    return all_ok


def download_pancreas(datasets_dir, batches=None):
    # type: (Path, list|None) -> bool
    """Descarga uno o más batches de PANORAMA CT Pancreas desde Zenodo.

    Args:
        datasets_dir: directorio base de datasets
        batches: lista de enteros [1..4]. Por defecto [1] (compatibilidad hacia atrás).

    Returns:
        True si todos los batches solicitados descargaron correctamente.
    """
    if batches is None:
        batches = [1]
    all_ok = True
    dest_dir = datasets_dir / "zenodo_13715870"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for batch_num in batches:
        info = ZENODO_PANCREAS_BATCHES.get(batch_num)
        if info is None:
            log.error("[PANCREAS] Batch %d no reconocido (válidos: 1-4).", batch_num)
            all_ok = False
            continue
        dest = dest_dir / "batch_{}.zip".format(batch_num)
        tag = "PANCREAS-B{}".format(batch_num)

        # Skip download if ZIP absent but batch data already extracted (BUG 2 fix).
        # Read the ZIP manifest from a local copy if it exists; otherwise,
        # fall back to a conservative per-batch minimum file count.
        if not dest.exists() or dest.stat().st_size == 0:
            # ZIP not on disk — check if extracted .nii.gz are already present
            # by attempting a count.  Each batch has ~550-600 nii.gz files;
            # we use 100 as a safe per-batch minimum since without the ZIP
            # manifest we can't know the exact file list.  However, if we find
            # a substantial number, re-downloading is wasteful.
            #
            # To be more precise, we would need to keep a manifest on disk
            # after extraction, but the conservative count avoids the
            # 46+ GB re-download for the common case.
            #
            # We can't easily tell *which* .nii.gz belong to which batch
            # without the ZIP manifest.  So we check the global count vs.
            # what's expected for *all batches up to this one*.
            nii_count = (
                len(list(dest_dir.rglob("*.nii.gz"))) if dest_dir.is_dir() else 0
            )
            # Expected minimum: 100 files per batch for batches <= batch_num
            expected_min = 100 * batch_num
            if nii_count >= expected_min:
                log.info(
                    "[%s] ZIP ausente pero %d .nii.gz presentes (>= %d esperados) — datos ya extraídos, saltando descarga.",
                    tag,
                    nii_count,
                    expected_min,
                )
                continue

        ok = download_wget(
            info["url"],
            dest,
            tag,
            extra_flags=[
                "--tries=5",
                "--retry-connrefused",
                "--waitretry=30",
                "--timeout=120",
            ],
            min_size_bytes=MIN_ZIP_SIZES["pancreas"],
        )
        if ok:
            log.info(
                "[PANCREAS-B%d] Descarga OK — MD5 esperado: %s", batch_num, info["md5"]
            )
        else:
            all_ok = False
    return all_ok


def _panorama_is_valid_repo(repo_dir):
    # type: (Path) -> bool
    if not (repo_dir / ".git").is_dir():
        return False
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return False
    except Exception:
        return False
    # Verify working tree has actual label files (not just .git metadata)
    labels_dir = repo_dir / "automatic_labels"
    if not labels_dir.is_dir():
        return False
    if not any(labels_dir.glob("*.nii.gz")):
        return False
    return True


def download_panorama(datasets_dir):
    # type: (Path) -> bool
    """Clona panorama_labels y guarda commit hash."""
    repo_dir = datasets_dir / "panorama_labels"
    commit_file = datasets_dir / "panorama_labels_commit.txt"

    if _panorama_is_valid_repo(repo_dir):
        log.info("[PANORAMA] Ya clonado y válido.")
    else:
        if repo_dir.exists():
            log.warning("[PANORAMA] Repo existente pero inválido — re-clonando...")
            shutil.rmtree(repo_dir)
        log.info("[PANORAMA] Clonando %s ...", PANORAMA_REPO)
        try:
            subprocess.run(
                ["git", "clone", PANORAMA_REPO, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            log.error("[PANORAMA] git clone falló.")
            return False

    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = r.stdout.strip()
        commit_file.write_text(commit + "\n")
        log.info("[PANORAMA] Commit: %s", commit)
    except Exception as e:
        log.warning("[PANORAMA] No se pudo obtener commit hash: %s", e)
    return True


# ── Orquestador de descargas ──────────────────────────────────────────────────


def run_downloads(
    datasets_dir, active, luna_subsets=None, dry_run=False, pancreas_batches=None
):
    # type: (Path, set, list|None, bool, list|None) -> dict
    """Ejecuta descargas para los datasets activos. Retorna {ds_id: bool}."""
    results = {}
    if pancreas_batches is None:
        pancreas_batches = [1, 2, 3, 4]

    download_map = {
        "nih": lambda: download_nih(datasets_dir),
        "isic": lambda: download_isic(datasets_dir),
        "oa": lambda: download_oa(datasets_dir),
        "luna_meta": lambda: download_luna_meta(datasets_dir),
        "luna_ct": lambda: download_luna_ct(datasets_dir, luna_subsets),
        "pancreas": lambda: download_pancreas(datasets_dir, batches=pancreas_batches),
        "panorama": lambda: download_panorama(datasets_dir),
    }

    download_order = [
        "oa",
        "luna_meta",
        "isic",
        "panorama",
        "nih",
        "luna_ct",
        "pancreas",
    ]

    for ds_id in download_order:
        if ds_id not in active:
            continue
        if dry_run:
            log.info("[DRY-RUN] Descargaría: %s", ds_id)
            results[ds_id] = True
            continue
        fn = download_map.get(ds_id)
        if fn is None:
            continue
        try:
            ok = fn()
            results[ds_id] = ok
            if ok:
                log.info("[%s] Descarga OK", ds_id.upper())
            else:
                log.warning("[%s] Descarga FALLÓ — continuando.", ds_id.upper())
        except Exception as e:
            log.error("[%s] Error inesperado en descarga: %s", ds_id.upper(), e)
            results[ds_id] = False

    return results
