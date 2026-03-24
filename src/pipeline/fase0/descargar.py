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
from pathlib import Path

# ── psutil: instalar si no está ──────────────────────────────────────────────
try:
    import psutil
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "psutil"])
    import psutil

log = logging.getLogger("fase0.descargar")

# ── Constantes ────────────────────────────────────────────────────────────────

KAGGLE_API_URL = "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"

ZENODO_LUNA_URL_PART1 = "https://zenodo.org/records/3723295/files/subset{i}.zip?download=1"
ZENODO_LUNA_URL_PART2 = "https://zenodo.org/records/2596479/files/subset{i}.zip?download=1"
ZENODO_PANCREAS_URL = "https://zenodo.org/records/13715870/files/batch_1.zip?download=1"

PANORAMA_REPO = "https://github.com/DIAGNijmegen/panorama_labels.git"

# Tamaños mínimos de ZIPs (bytes) — detectar archivos corruptos/incompletos
MIN_ZIP_SIZES = {
    "nih":       35 * 1024**3,
    "isic":       5 * 1024**3,
    "oa":        200 * 1024**2,
    "luna_meta": 100 * 1024**2,
    "luna_ct":    4 * 1024**3,
    "pancreas":  40 * 1024**3,
}

# Datasets que requieren kaggle CLI
KAGGLE_DATASETS = {"isic", "oa", "luna"}


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
            log.error("  %s NO encontrado — instalar con: sudo apt-get install %s",
                      cmd, cmd)
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
                        + pip_extra + ["kaggle"])
                    break
                except subprocess.CalledProcessError:
                    continue
        if shutil.which("kaggle"):
            log.info("  kaggle OK")
        else:
            log.error("  kaggle CLI necesario para %s pero no se pudo instalar.",
                      ", ".join(sorted(active & KAGGLE_DATASETS)))
            ok = False

    # Credenciales Kaggle
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if needs_kaggle:
        if not kaggle_json.exists():
            log.error("  ~/.kaggle/kaggle.json NO encontrado.")
            log.error("  1. https://www.kaggle.com/settings -> API -> 'Create New Token'")
            log.error("  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
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
            log.warning("[%s] %s existe pero pesa solo %s (< %.0f MB mínimo) — continuando.",
                        tag, dest.name, file_size_human(dest), min_size_bytes / 1024 ** 2)
        else:
            log.info("[%s] Ya existe: %s (%s), saltando.",
                     tag, dest.name, file_size_human(dest))
            return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget", "--continue", "--tries=10", "--retry-connrefused",
        "--waitretry=60", "--timeout=300", "--progress=dot:giga",
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
            log.warning("[%s] %s existe pero pesa solo %s — re-descargando.",
                        tag, expected_zip.name, file_size_human(expected_zip))
            expected_zip.unlink()
        else:
            log.info("[%s] Ya existe: %s (%s), saltando.",
                     tag, expected_zip.name, file_size_human(expected_zip))
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
            log.error("[%s] %s descargado pero demasiado pequeño — corrupto.", tag, expected_zip.name)
            return False
        elapsed = time.time() - t0
        log.info("[%s] OK %s en %.0fs", tag, file_size_human(expected_zip), elapsed)
        return True

    if new_zips:
        new_zip = sorted(new_zips)[0]
        log.warning("[%s] Kaggle descargó '%s' — renombrando a '%s'",
                    tag, new_zip.name, expected_zip.name)
        new_zip.rename(expected_zip)
        if min_size_bytes is not None and expected_zip.stat().st_size < min_size_bytes:
            log.error("[%s] %s descargado pero demasiado pequeño — corrupto.", tag, expected_zip.name)
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
        KAGGLE_API_URL, dest, "NIH",
        extra_flags=[
            "--auth-no-challenge",
            "--http-user=" + user,
            "--http-password=" + key,
        ],
        min_size_bytes=MIN_ZIP_SIZES["nih"],
    )


def download_isic(datasets_dir):
    # type: (Path) -> bool
    return download_kaggle_cli(
        "andrewmvd/isic-2019",
        datasets_dir / "isic_2019",
        datasets_dir / "isic_2019" / "isic-2019.zip",
        "ISIC",
        min_size_bytes=MIN_ZIP_SIZES["isic"],
    )


def download_oa(datasets_dir):
    # type: (Path) -> bool
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
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    ct_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for i in subsets:
        url_template = ZENODO_LUNA_URL_PART1 if i <= 6 else ZENODO_LUNA_URL_PART2
        url = url_template.format(i=i)
        dest = ct_dir / "subset{}.zip".format(i)
        ok = download_wget(url, dest, "LUNA-CT{}".format(i),
                           min_size_bytes=MIN_ZIP_SIZES["luna_ct"])
        if not ok:
            all_ok = False
    return all_ok


def download_pancreas(datasets_dir):
    # type: (Path) -> bool
    dest = datasets_dir / "zenodo_13715870" / "batch_1.zip"
    return download_wget(
        ZENODO_PANCREAS_URL, dest, "PANCREAS",
        extra_flags=[
            "--tries=5", "--retry-connrefused",
            "--waitretry=30", "--timeout=120",
        ],
        min_size_bytes=MIN_ZIP_SIZES["pancreas"],
    )


def _panorama_is_valid_repo(repo_dir):
    # type: (Path) -> bool
    if not (repo_dir / ".git").is_dir():
        return False
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


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
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            log.error("[PANORAMA] git clone falló.")
            return False

    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        commit = r.stdout.strip()
        commit_file.write_text(commit + "\n")
        log.info("[PANORAMA] Commit: %s", commit)
    except Exception as e:
        log.warning("[PANORAMA] No se pudo obtener commit hash: %s", e)
    return True


# ── Orquestador de descargas ──────────────────────────────────────────────────

def run_downloads(datasets_dir, active, luna_subsets=None, dry_run=False):
    # type: (Path, set, list|None, bool) -> dict
    """Ejecuta descargas para los datasets activos. Retorna {ds_id: bool}."""
    results = {}

    download_map = {
        "nih":      lambda: download_nih(datasets_dir),
        "isic":     lambda: download_isic(datasets_dir),
        "oa":       lambda: download_oa(datasets_dir),
        "luna":     lambda: download_luna_meta(datasets_dir),
        "luna_ct":  lambda: download_luna_ct(datasets_dir, luna_subsets),
        "pancreas": lambda: download_pancreas(datasets_dir),
        "panorama": lambda: download_panorama(datasets_dir),
    }

    download_order = ["oa", "luna", "isic", "panorama", "nih", "luna_ct", "pancreas"]

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
