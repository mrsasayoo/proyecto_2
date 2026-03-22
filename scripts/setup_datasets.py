#!/usr/bin/env python3
"""
setup_datasets.py — Descarga, extracción y preparación de todos los datasets
==============================================================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II

Reemplaza:  setup_datasets_fast.sh, setup_datasets_smart.sh,
            download_luna.sh, y la parte de alineación de fix_alignment.py.

Datasets (5 expertos de dominio):
  0 → NIH ChestXray14   (~42 GB)     Kaggle REST API via wget
  1 → ISIC 2019         (~9 GB)      Kaggle CLI
  2 → OA Rodilla        (~400 MB)    Kaggle CLI
  3 → LUNA16 metadata   (~330 MB)    Kaggle CLI
      LUNA16 CT volumes  (subsets)    Zenodo wget
  4 → Pancreas PANORAMA  (~46 GB)    Zenodo wget
      Panorama Labels    (git repo)   GitHub clone

Uso:
  python3 scripts/setup_datasets.py
  python3 scripts/setup_datasets.py --only oa isic --no_extract
  python3 scripts/setup_datasets.py --skip nih pancreas --luna_subsets 0 1 2
  python3 scripts/setup_datasets.py --dry_run

Compatible con Python 3.8+.  Dependencias: psutil (se instala si falta).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# psutil: instalar si no está
# ─────────────────────────────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "psutil"])
    import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
DATASET_IDS = ("nih", "isic", "oa", "luna", "luna_ct", "pancreas", "panorama")

KAGGLE_API_URL = "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"

ZENODO_LUNA_URL_PART1 = "https://zenodo.org/records/3723295/files/subset{i}.zip?download=1"  # subsets 0-6
ZENODO_LUNA_URL_PART2 = "https://zenodo.org/records/2596479/files/subset{i}.zip?download=1"  # subsets 7-9
ZENODO_PANCREAS_URL = "https://zenodo.org/records/13715870/files/batch_1.zip?download=1"

# Tamaños mínimos de ZIPs (en bytes) — para detectar archivos corruptos/incompletos
MIN_ZIP_SIZES = {
    "nih":       35 * 1024**3,   # 35 GB mínimo (real ~42 GB)
    "isic":       5 * 1024**3,   # 5 GB mínimo (real ~9 GB)
    "oa":        200 * 1024**2,  # 200 MB mínimo (real ~400 MB)
    "luna_meta": 100 * 1024**2,  # 100 MB mínimo (real ~330 MB)
    "luna_ct":    4 * 1024**3,   # 4 GB mínimo por subset (real ~6 GB)
    "pancreas":  40 * 1024**3,   # 40 GB mínimo (real ~46 GB)
}

PANORAMA_REPO = "https://github.com/DIAGNijmegen/panorama_labels.git"

log = logging.getLogger("setup")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir):
    # type: (Path) -> logging.Logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "setup_datasets.log"
    fmt = "%(asctime)s | [%(levelname)-5s] | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger().handlers[1].setLevel(logging.INFO)
    logger = logging.getLogger("setup")
    logger.info("Log iniciado -> %s", log_path)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd, tag="", check=True):
    # type: (list, str, bool) -> subprocess.CompletedProcess
    """Ejecuta un comando con logging."""
    log.debug("[%s] $ %s", tag, " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("[%s] Código %d: %s", tag, result.returncode,
                  result.stderr.strip()[:500])
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def dir_size_human(path):
    # type: (Path) -> str
    """Tamaño en disco legible (du -sh)."""
    try:
        r = subprocess.run(["du", "-sh", str(path)],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.split()[0] if r.returncode == 0 else "?"
    except Exception:
        return "?"


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


def ram_available_mb():
    # type: () -> float
    return psutil.virtual_memory().available / (1024 * 1024)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — Prerequisitos
# ─────────────────────────────────────────────────────────────────────────────

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


# Datasets que requieren kaggle CLI para descarga
KAGGLE_DATASETS = {"isic", "oa", "luna"}


def check_prerequisites(active):
    # type: (set) -> bool
    """Verifica herramientas del sistema y credenciales Kaggle."""
    log.info("=" * 60)
    log.info("FASE 1 — Verificando prerequisitos")
    log.info("=" * 60)
    ok = True

    # Herramientas del sistema
    for cmd in ("python3", "unzip", "git", "wget"):
        if shutil.which(cmd):
            log.info("  %s OK (%s)", cmd, shutil.which(cmd))
        else:
            log.error("  %s NO encontrado — instalar con: sudo apt-get install %s",
                      cmd, cmd)
            ok = False

    # kaggle CLI — solo necesario para ISIC, OA, LUNA metadata
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
            log.error("  Instalar manualmente: pip install --user kaggle")
            ok = False
    else:
        if shutil.which("kaggle"):
            log.info("  kaggle OK (disponible, no requerido)")
        else:
            log.info("  kaggle no instalado (no requerido para datasets activos)")

    # 7z — extractor principal para ZIP64
    if not shutil.which("7z"):
        log.warning("  7z no encontrado — instalando p7zip-full...")
        try:
            subprocess.check_call(
                ["apt-get", "install", "-y", "p7zip-full"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    if shutil.which("7z"):
        log.info("  7z OK (%s)", shutil.which("7z"))
    else:
        log.warning(
            "  7z no disponible — se usará unzip como fallback (puede fallar con ZIP64)")

    # Credenciales
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        if needs_kaggle:
            log.error("  ~/.kaggle/kaggle.json NO encontrado.")
            log.error("  Instrucciones:")
            log.error("    1. Ve a https://www.kaggle.com/settings -> API -> 'Create New Token'")
            log.error("    2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
            log.error("    3. chmod 600 ~/.kaggle/kaggle.json")
            ok = False
        else:
            log.info("  kaggle.json ausente (no requerido para datasets activos)")
    else:
        kaggle_json.chmod(0o600)
        user, key = read_kaggle_creds()
        if user and key:
            log.info("  kaggle.json OK (usuario: %s)", user)
        else:
            msg = "kaggle.json existe pero falta 'username' o 'key'."
            if needs_kaggle:
                log.error("  %s", msg)
                ok = False
            else:
                log.warning("  %s", msg)

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — Descargas
# ─────────────────────────────────────────────────────────────────────────────

def download_wget(url, dest, tag, extra_flags=None, min_size_bytes=None):
    # type: (str, Path, str, list|None, int|None) -> bool
    """Descarga con wget. Idempotente: salta si ya existe con tamaño suficiente."""
    if dest.exists() and dest.stat().st_size > 0:
        if min_size_bytes is not None and dest.stat().st_size < min_size_bytes:
            log.warning("[%s] %s existe pero pesa solo %s (< %.0f MB mínimo) — continuando descarga.",
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
        # wget escribe progreso a stderr; lo dejamos pasar a consola
        proc = subprocess.run(cmd, timeout=86400)  # 24h máximo
        if proc.returncode != 0:
            log.error("[%s] wget terminó con código %d.", tag, proc.returncode)
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
                log.warning("[%s] Archivo de 0 bytes eliminado — la próxima corrida descargará de nuevo.", tag)
            elif dest.exists():
                log.warning("[%s] Archivo parcial: %s — reejecutar reanudará.",
                            tag, file_size_human(dest))
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
            log.warning("[%s] %s existe pero pesa solo %s (< %.0f MB mínimo) — re-descargando.",
                        tag, expected_zip.name, file_size_human(expected_zip), min_size_bytes / 1024 ** 2)
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

    # Si ya está el esperado (quizá kaggle lo nombró bien)
    if expected_zip.exists():
        if min_size_bytes is not None and expected_zip.stat().st_size < min_size_bytes:
            log.error("[%s] %s descargado pero pesa solo %s (< %.0f MB mínimo) — corrupto.",
                      tag, expected_zip.name, file_size_human(expected_zip), min_size_bytes / 1024 ** 2)
            return False
        elapsed = time.time() - t0
        log.info("[%s] OK %s en %.0fs", tag, file_size_human(expected_zip), elapsed)
        return True

    # Detectar ZIP nuevo y renombrar
    if new_zips:
        new_zip = sorted(new_zips)[0]
        log.warning("[%s] Kaggle descargó '%s' — renombrando a '%s'",
                    tag, new_zip.name, expected_zip.name)
        new_zip.rename(expected_zip)
        if min_size_bytes is not None and expected_zip.stat().st_size < min_size_bytes:
            log.error("[%s] %s descargado pero pesa solo %s (< %.0f MB mínimo) — corrupto.",
                      tag, expected_zip.name, file_size_human(expected_zip), min_size_bytes / 1024 ** 2)
            return False
        elapsed = time.time() - t0
        log.info("[%s] OK %s en %.0fs", tag, file_size_human(expected_zip), elapsed)
        return True

    log.error("[%s] No se encontró ningún ZIP nuevo en %s", tag, dest_dir)
    return False


def download_nih(datasets_dir):
    # type: (Path) -> bool
    """NIH ChestXray14 via wget + Kaggle REST API (NO kaggle CLI — OOM risk)."""
    user, key = read_kaggle_creds()
    if not user or not key:
        log.error("[NIH] Credenciales Kaggle no disponibles.")
        return False
    dest = datasets_dir / "nih_chest_xrays" / "data.zip"
    log.info("[NIH] Descargando via wget + Kaggle REST API (~42 GB, streaming a disco)...")
    log.info("[NIH] Usuario Kaggle: %s", user)
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
    """Verifica que el repo esté en estado válido (no solo que .git exista)."""
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
            log.warning("[PANORAMA] Repo existente pero inválido — eliminando y re-clonando...")
            shutil.rmtree(repo_dir)
        log.info("[PANORAMA] Clonando %s ...", PANORAMA_REPO)
        try:
            run_cmd(["git", "clone", PANORAMA_REPO, str(repo_dir)], tag="PANORAMA")
        except subprocess.CalledProcessError:
            log.error("[PANORAMA] git clone falló.")
            return False

    # Guardar commit hash
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


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — Extracción con monitor de RAM
# ─────────────────────────────────────────────────────────────────────────────

class RAMMonitor:
    """Hilo demonio que pausa/reanuda un proceso unzip según RAM disponible."""

    def __init__(self, pid, tag, pause_mb, resume_mb, check_interval=3):
        # type: (int, str, float, float, int) -> None
        self.pid = pid
        self.tag = tag
        self.pause_mb = pause_mb
        self.resume_mb = resume_mb
        self.check_interval = check_interval
        self._paused = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                avail = ram_available_mb()
                if not self._paused and avail < self.pause_mb:
                    os.kill(self.pid, signal.SIGSTOP)
                    self._paused = True
                    log.warning("[%s][RAM] SIGSTOP — RAM disponible: %.0f MB < %.0f MB",
                                self.tag, avail, self.pause_mb)
                elif self._paused and avail > self.resume_mb:
                    os.kill(self.pid, signal.SIGCONT)
                    self._paused = False
                    log.info("[%s][RAM] SIGCONT — RAM disponible: %.0f MB > %.0f MB",
                             self.tag, avail, self.resume_mb)
            except ProcessLookupError:
                break
            except Exception as e:
                log.debug("[%s][RAM] Error en monitor: %s", self.tag, e)
                break
            self._stop.wait(self.check_interval)


def wait_for_ram(tag, resume_mb):
    # type: (str, float) -> None
    """Espera hasta que haya suficiente RAM antes de empezar."""
    avail = ram_available_mb()
    if avail >= resume_mb:
        return
    log.warning("[%s] RAM disponible: %.0f MB < %.0f MB — esperando...",
                tag, avail, resume_mb)
    while avail < resume_mb:
        time.sleep(10)
        avail = ram_available_mb()
        log.debug("[%s] Esperando RAM: %.0f MB", tag, avail)
    log.info("[%s] RAM OK: %.0f MB", tag, avail)


def smart_extract(archive, dest, tag, pause_mb, resume_mb):
    # type: (Path, Path, str, float, float) -> bool
    """Extracción con monitor de RAM (SIGSTOP / SIGCONT). Usa 7z si está disponible."""
    if not archive.exists():
        log.error("[%s] Archivo no encontrado: %s", tag, archive)
        return False

    wait_for_ram(tag, resume_mb)

    dest.mkdir(parents=True, exist_ok=True)

    # Seleccionar extractor: 7z para ZIP64; unzip como fallback
    if shutil.which("7z"):
        # -o sin espacio antes de la ruta es obligatorio para 7z
        cmd = ["7z", "x", str(archive), "-o" + str(dest), "-y"]
        extractor = "7z"
    else:
        cmd = ["unzip", "-q", "-o", str(archive), "-d", str(dest)]
        extractor = "unzip"

    log.debug("[%s] Extractor: %s", tag, extractor)
    log.info("[%s] Extrayendo %s -> %s", tag, archive.name, dest)
    t0 = time.time()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    monitor = RAMMonitor(proc.pid, tag, pause_mb, resume_mb)
    monitor.start()

    try:
        _, stderr = proc.communicate()
        rc = proc.returncode
    finally:
        monitor.stop()

    elapsed = time.time() - t0
    stderr_text = stderr.decode(errors="replace")

    if extractor == "7z":
        # rc=0 OK; rc=1 warning (ZIP64 con errores conocidos en LUNA subsets 7/9); rc=2 error fatal
        if rc == 0:
            log.info("[%s] OK extraído en %.0fs", tag, elapsed)
            return True
        elif rc == 1:
            log.warning(
                "[%s] 7z terminó con warnings (rc=1) — posible ZIP64 con datos "
                "corruptos en origen (conocido en LUNA16 subsets 7 y 9). "
                "Verificar conteo de .mhd tras la extracción.", tag)
            return True  # los archivos válidos sí se extraen
        else:
            log.error("[%s] 7z falló (rc=%d): %s", tag, rc, stderr_text[:300])
            return False
    else:
        # unzip: rc=0 OK; rc=1 warning; rc>=2 error real
        if rc >= 2:
            log.error("[%s] unzip falló (rc=%d): %s", tag, rc, stderr_text[:300])
            return False
        elif rc == 1:
            log.warning("[%s] unzip terminó con warnings (rc=%d)", tag, rc)
        log.info("[%s] OK extraído en %.0fs", tag, elapsed)
        return True


def is_extracted(datasets_dir, ds_id, luna_subsets=None):
    # type: (Path, str, list|None) -> bool
    """Verifica si un dataset ya está extraído usando marcadores."""
    checks = {
        "nih":     lambda: (datasets_dir / "nih_chest_xrays" / "images_001").is_dir(),
        "isic":    lambda: (datasets_dir / "isic_2019" / "ISIC_2019_Training_Input").is_dir(),
        "oa":      lambda: (datasets_dir / "osteoarthritis" / "KLGrade").is_dir(),
        "luna":    lambda: (datasets_dir / "luna_lung_cancer" / "candidates_V2").is_dir(),
        "luna_ct": lambda: _luna_ct_extracted(datasets_dir, luna_subsets or [0]),
        "pancreas": lambda: _pancreas_extracted(datasets_dir),
    }
    checker = checks.get(ds_id)
    if checker is None:
        return False
    return checker()


def _luna_ct_extracted(datasets_dir, subsets):
    # type: (Path, list) -> bool
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    for i in subsets:
        subset_dir = ct_dir / "subset{}".format(i)
        if not subset_dir.is_dir():
            return False
        mhd_count = len(list(subset_dir.glob("*.mhd")))
        if mhd_count < 5:
            return False
    return True


def _pancreas_extracted(datasets_dir):
    # type: (Path) -> bool
    zenodo_dir = datasets_dir / "zenodo_13715870"
    nii_count = len(list(zenodo_dir.rglob("*.nii.gz")))
    return nii_count >= 1


def run_extractions(datasets_dir, active, args):
    # type: (Path, set, argparse.Namespace) -> dict
    """Ejecuta las extracciones en orden: menor a mayor."""
    results = {}
    pause_mb = args.ram_pause_mb
    resume_mb = args.ram_resume_mb
    luna_subsets = args.luna_subsets

    # Orden: OA -> LUNA meta -> ISIC -> NIH -> LUNA CTs -> Pancreas
    extraction_plan = [
        ("oa", datasets_dir / "osteoarthritis" / "osteoarthritis.zip",
         datasets_dir / "osteoarthritis", None),
        ("luna", datasets_dir / "luna_lung_cancer" / "luna-lung-cancer-dataset.zip",
         datasets_dir / "luna_lung_cancer", None),
        ("isic", datasets_dir / "isic_2019" / "isic-2019.zip",
         datasets_dir / "isic_2019", None),
        ("nih", datasets_dir / "nih_chest_xrays" / "data.zip",
         datasets_dir / "nih_chest_xrays", None),
    ]

    # LUNA CT subsets
    for i in luna_subsets:
        extraction_plan.append(
            ("luna_ct",
             datasets_dir / "luna_lung_cancer" / "ct_volumes" / "subset{}.zip".format(i),
             datasets_dir / "luna_lung_cancer" / "ct_volumes",
             [i]),
        )

    # Pancreas al final (el más grande)
    extraction_plan.append(
        ("pancreas", datasets_dir / "zenodo_13715870" / "batch_1.zip",
         datasets_dir / "zenodo_13715870",
         None),
    )

    for ds_id, archive, dest, subset_override in extraction_plan:
        if ds_id not in active:
            continue

        tag = ds_id.upper()
        if "luna_ct" in ds_id:
            tag = "LUNA-CT{}".format(archive.stem.replace("subset", ""))

        if is_extracted(datasets_dir, ds_id, subset_override if subset_override is not None else luna_subsets):
            log.info("[%s] Ya extraído, saltando.", tag)
            results[ds_id] = results.get(ds_id, True)
            continue

        if not archive.exists():
            log.warning("[%s] %s no encontrado — extracción pendiente.", tag, archive.name)
            results[ds_id] = False
            continue

        try:
            ok = smart_extract(archive, dest, tag, pause_mb, resume_mb)
            results[ds_id] = results.get(ds_id, True) and ok
            if ds_id == "luna_ct":
                subset_idx = int(archive.stem.replace("subset", ""))
                verify_luna_ct_subset(datasets_dir, subset_idx)
        except Exception as e:
            log.error("[%s] Error de extracción: %s", tag, e)
            results[ds_id] = False

    return results


def verify_luna_ct_subset(datasets_dir, subset_idx):
    # type: (Path, int) -> dict
    """Verifica un subset CT extraído. Retorna {'mhd': N, 'raw': N, 'ok': bool}."""
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    subset_dir = ct_dir / "subset{}".format(subset_idx)
    mhd_count = len(list(subset_dir.glob("*.mhd"))) if subset_dir.exists() else 0
    raw_count = len(list(subset_dir.glob("*.raw"))) if subset_dir.exists() else 0
    # subset0 tiene 178 archivos por ser doble; los demás ~88-89
    expected = 178 if subset_idx == 0 else 85
    ok = mhd_count >= expected and mhd_count == raw_count
    if ok:
        log.info("[LUNA-CT%d] Verificación OK: %d .mhd / %d .raw",
                 subset_idx, mhd_count, raw_count)
    else:
        log.warning(
            "[LUNA-CT%d] Extracción posiblemente parcial: %d .mhd / %d .raw "
            "(esperados >= %d pares). Puede ser normal para subsets con ZIPs "
            "corruptos en origen (subset7/9).",
            subset_idx, mhd_count, raw_count, expected)
    return {"mhd": mhd_count, "raw": raw_count, "ok": ok, "expected": expected}


def _extract_nih_split_txts(data_zip_path, nih_dir):
    # type: (Path, Path) -> None
    """
    Extrae train_val_list.txt y test_list.txt del data.zip de NIH antes de borrarlo.
    Solo se llama en modo --disco, donde el ZIP se borra después de la extracción.
    """
    txt_names = ["train_val_list.txt", "test_list.txt"]
    try:
        with zipfile.ZipFile(data_zip_path, "r") as zf:
            namelist = zf.namelist()
            for txt_name in txt_names:
                candidates = [n for n in namelist if n.endswith(txt_name)]
                if candidates:
                    data = zf.read(candidates[0])
                    out_path = nih_dir / txt_name
                    out_path.write_bytes(data)
                    line_count = data.count(b"\n")
                    log.info("[NIH][--disco] %s extraído preventivamente (%d líneas)",
                             txt_name, line_count)
                else:
                    log.warning("[NIH][--disco] %s no encontrado en data.zip", txt_name)
    except Exception as e:
        log.warning("[NIH][--disco] No se pudieron extraer txts de splits: %s", e)


def download_and_extract_one(ds_id, download_fn, archive_path, dest_path, tag,
                             args, extract_check_fn=None):
    # type: (str, object, Path, Path, str, object, object) -> tuple
    """
    Modo --disco: descarga, extrae y borra el ZIP en un solo paso.

    Returns:
        (download_ok: bool, extract_ok: bool)
    """
    # Verificar si ya está extraído (idempotencia)
    if extract_check_fn is not None and extract_check_fn():
        log.info("[%s] Ya extraído (modo --disco), saltando.", tag)
        # Si el ZIP también existe, borrarlo (limpieza de corridas anteriores)
        if archive_path.exists():
            log.info("[%s] Borrando ZIP residual: %s", tag, archive_path.name)
            archive_path.unlink()
        return True, True

    # Descargar
    dl_ok = download_fn()
    if not dl_ok:
        return False, False

    # Extraer inmediatamente
    log.info("[%s] --disco: extrayendo inmediatamente tras descarga.", tag)
    ex_ok = smart_extract(
        archive_path, dest_path, tag,
        args.ram_pause_mb, args.ram_resume_mb,
    )

    # Borrar ZIP independientemente del resultado de extracción
    if archive_path.exists():
        # En modo --disco con NIH, extraer preventivamente los archivos de splits
        # antes de borrar el ZIP, para que post_nih() los encuentre en FASE 4
        if ds_id == "nih" and ex_ok:
            _extract_nih_split_txts(archive_path, dest_path)
        
        zip_size = file_size_human(archive_path)
        archive_path.unlink()
        if ex_ok:
            log.info("[%s] --disco: ZIP eliminado (%s liberados).", tag, zip_size)
        else:
            log.warning(
                "[%s] --disco: ZIP eliminado aunque la extracción tuvo errores. "
                "Si los archivos extraídos son insuficientes, re-ejecutar "
                "sin --disco para reintentar.", tag)

    return dl_ok, ex_ok


# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 — Post-extracción
# ─────────────────────────────────────────────────────────────────────────────

def post_nih(datasets_dir):
    # type: (Path) -> None
    """Symlinks all_images/ + verificar splits txt."""
    nih_dir = datasets_dir / "nih_chest_xrays"
    tag = "NIH"

    # ── Verificar txt de splits ──────────────────────────────────────────
    min_lines = {"train_val_list.txt": 80000, "test_list.txt": 20000}
    for txt_name, min_count in min_lines.items():
        txt_path = nih_dir / txt_name
        needs_extract = False
        if not txt_path.exists():
            log.warning("[%s] %s no encontrado.", tag, txt_name)
            needs_extract = True
        else:
            count = sum(1 for _ in open(txt_path, encoding="utf-8"))
            if count < min_count:
                log.warning("[%s] %s truncado (%d < %d).", tag, txt_name, count, min_count)
                needs_extract = True
            else:
                log.info("[%s] %s OK (%d entradas)", tag, txt_name, count)

        if needs_extract:
            data_zip = nih_dir / "data.zip"
            if data_zip.exists():
                log.info("[%s] Extrayendo %s de data.zip...", tag, txt_name)
                try:
                    with zipfile.ZipFile(data_zip, "r") as zf:
                        # Buscar el archivo dentro del ZIP (puede estar en subdir)
                        candidates = [n for n in zf.namelist() if n.endswith(txt_name)]
                        if candidates:
                            member = candidates[0]
                            data = zf.read(member)
                            txt_path.write_bytes(data)
                            count = data.count(b"\n")
                            log.info("[%s] %s extraído (%d líneas)", tag, txt_name, count)
                        else:
                            log.warning("[%s] %s no encontrado dentro de data.zip.", tag, txt_name)
                except Exception as e:
                    log.error("[%s] Error extrayendo %s: %s", tag, txt_name, e)

    # ── Symlinks all_images/ ─────────────────────────────────────────────
    all_imgs = nih_dir / "all_images"
    if all_imgs.is_dir():
        n_links = sum(1 for p in all_imgs.iterdir() if p.is_symlink())
        if n_links > 0:
            log.info("[%s] all_images/ ya existe (%d symlinks), saltando.", tag, n_links)
            return
        else:
            log.warning("[%s] all_images/ vacío/corrupto — recreando...", tag)
            shutil.rmtree(all_imgs)

    # Buscar PNGs en images_001..012
    png_files = sorted(nih_dir.glob("images_*/images/*.png"))
    if not png_files:
        log.warning("[%s] No se encontraron .png — all_images pendiente.", tag)
        return

    all_imgs.mkdir(parents=True, exist_ok=True)
    created = 0
    for png in png_files:
        link = all_imgs / png.name
        if not link.exists():
            link.symlink_to(png.resolve())
            created += 1
    log.info("[%s] %d symlinks creados en all_images/", tag, created)


def post_oa(datasets_dir):
    # type: (Path) -> None
    """Consolidar KL grades y crear splits."""
    oa_dir = datasets_dir / "osteoarthritis"
    splits_dir = oa_dir / "oa_splits"
    tag = "OA"

    if splits_dir.is_dir() and any((splits_dir / "train").rglob("*.*")):
        log.info("[%s] oa_splits/ ya existe con imágenes, saltando.", tag)
        return

    # Buscar directorio KLGrade
    src = None
    for candidate in [oa_dir / "KLGrade" / "KLGrade", oa_dir / "KLGrade"]:
        if candidate.is_dir() and any(candidate.iterdir()):
            src = candidate
            break
    if src is None:
        # Buscar recursivamente
        kl_dirs = list(oa_dir.rglob("KLGrade"))
        for d in kl_dirs:
            if d.is_dir() and any(d.iterdir()):
                src = d
                break
    if src is None:
        log.warning("[%s] No se encontró KLGrade/ con imágenes.", tag)
        return

    log.info("[%s] Fuente: %s", tag, src)

    # Consolidar: KL0→0, KL1+KL2→1 (Leve), KL3+KL4→2 (Severo)
    mapping = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
    all_files = {0: [], 1: [], 2: []}

    for kl_str, cls in mapping.items():
        kl_dir = src / kl_str
        if not kl_dir.exists():
            log.warning("[%s] KL%s/ no encontrado, saltando.", tag, kl_str)
            continue
        files = list(kl_dir.glob("*.jpg")) + list(kl_dir.glob("*.png"))
        all_files[cls].extend(files)
        log.debug("[%s] KL%s -> clase %d: %d imágenes", tag, kl_str, cls, len(files))

    total = sum(len(v) for v in all_files.values())
    if total == 0:
        log.warning("[%s] No se encontraron imágenes en KLGrade/.", tag)
        return

    random.seed(42)
    counts = {"train": 0, "val": 0, "test": 0}
    for cls, files in all_files.items():
        random.shuffle(files)
        n = len(files)
        splits = {
            "train": files[:int(0.80 * n)],
            "val":   files[int(0.80 * n):int(0.95 * n)],
            "test":  files[int(0.95 * n):],
        }
        for split_name, imgs in splits.items():
            d = splits_dir / split_name / str(cls)
            d.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, d / img.name)
            counts[split_name] += len(imgs)

    log.info("[%s] oa_splits/ OK — train:%d | val:%d | test:%d | total:%d",
             tag, counts["train"], counts["val"], counts["test"], total)


def post_pancreas(datasets_dir):
    # type: (Path) -> None
    """Verificación de alineación NIfTI ↔ panorama_labels."""
    zenodo_dir = datasets_dir / "zenodo_13715870"
    labels_dir = datasets_dir / "panorama_labels"
    tag = "PANCREAS"

    nii_files = sorted(zenodo_dir.rglob("*.nii.gz"))
    nii_ids = set()
    for f in nii_files:
        m = re.match(r"^(\d+)", f.name)
        if m:
            nii_ids.add(m.group(1))

    log.info("[%s] NIfTI locales: %d archivos -> %d IDs únicos", tag, len(nii_files), len(nii_ids))
    if not nii_ids:
        log.warning("[%s] No se encontraron NIfTIs — batch_1.zip pendiente?", tag)
        return

    # Buscar IDs en panorama_labels
    label_ids = set()
    source_desc = "desconocido"

    # Intentar via pandas
    try:
        import pandas as pd
        candidate_cols = [
            "case_id", "patient_id", "PatientID", "pid", "id",
            "subject_id", "name", "image_id", "case",
        ]
        for csv_path in sorted(labels_dir.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
                for col in candidate_cols:
                    if col in df.columns:
                        extracted = df[col].astype(str).str.extract(r"(\d{4,6})")[0].dropna().unique()
                        if len(extracted) > 0:
                            label_ids.update(extracted.tolist())
                            source_desc = "{} (col='{}')".format(
                                csv_path.relative_to(labels_dir), col)
                            break
                if label_ids:
                    break
            except Exception:
                continue
    except ImportError:
        log.debug("[%s] pandas no disponible, usando fallback NIfTI.", tag)

    # Fallback: NIfTIs en el repo
    if not label_ids:
        for f in labels_dir.rglob("*.nii.gz"):
            m = re.match(r"^(\d+)", f.name)
            if m:
                label_ids.add(m.group(1))
        if label_ids:
            source_desc = "NIfTIs en panorama_labels"

    if not label_ids:
        log.warning("[%s] No se encontraron IDs en panorama_labels.", tag)
        return

    log.info("[%s] IDs en labels: %d (fuente: %s)", tag, len(label_ids), source_desc)

    matched = nii_ids & label_ids
    only_nii = nii_ids - label_ids
    only_label = label_ids - nii_ids
    pct = 100.0 * len(matched) / max(len(nii_ids), 1)

    log.info("[%s] --- Alineación ---", tag)
    log.info("[%s]   Con imagen Y etiqueta: %5d  (%.1f%%)", tag, len(matched), pct)
    log.info("[%s]   Solo zenodo (sin label): %5d", tag, len(only_nii))
    log.info("[%s]   Solo labels (sin NIfTI): %5d", tag, len(only_label))

    if pct >= 80.0:
        log.info("[%s] %.0f%% de alineación — suficiente para continuar.", tag, pct)
    else:
        log.warning("[%s] Solo %.0f%% de alineación.", tag, pct)
        log.warning("[%s] ¿Faltan batch_2.zip, batch_3.zip? "
                    "Ver https://zenodo.org/records/13715870", tag)


# ─────────────────────────────────────────────────────────────────────────────
# Tabla resumen
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(datasets_dir, download_results, extract_results, active, args):
    # type: (Path, dict, dict, set, argparse.Namespace) -> None
    """Tabla final de estado y comando sugerido de FASE 0."""
    log.info("")
    log.info("=" * 70)
    log.info("  RESUMEN FINAL")
    log.info("=" * 70)

    rows = [
        ("NIH",      "nih",      datasets_dir / "nih_chest_xrays",
         lambda: (datasets_dir / "nih_chest_xrays" / "all_images").is_dir()),
        ("ISIC",     "isic",     datasets_dir / "isic_2019",
         lambda: (datasets_dir / "isic_2019" / "ISIC_2019_Training_Input").is_dir()),
        ("OA",       "oa",       datasets_dir / "osteoarthritis",
         lambda: (datasets_dir / "osteoarthritis" / "oa_splits" / "train").is_dir()),
        ("LUNA",     "luna",     datasets_dir / "luna_lung_cancer",
         lambda: (datasets_dir / "luna_lung_cancer" / "candidates_V2").is_dir()),
        ("LUNA-CT",  "luna_ct",  datasets_dir / "luna_lung_cancer" / "ct_volumes",
         lambda: _luna_ct_extracted(datasets_dir, args.luna_subsets)),
        ("PANCREAS", "pancreas", datasets_dir / "zenodo_13715870",
         lambda: _pancreas_extracted(datasets_dir)),
        ("PANORAMA", "panorama", datasets_dir / "panorama_labels",
         lambda: (datasets_dir / "panorama_labels" / ".git").is_dir()),
    ]

    log.info("  %-12s %-8s %s", "Dataset", "Estado", "Tamaño")
    log.info("  " + "-" * 50)
    for label, ds_id, path, ready_fn in rows:
        skipped = ds_id not in active
        if skipped:
            status = " -  (saltado)"
        elif ready_fn():
            status = " OK"
        elif download_results.get(ds_id, False):
            status = " !!  (parcial)"
        else:
            status = " X  (faltante)"
        # Evitar du -sh en carpetas muy grandes que no están activas
        if skipped:
            size = "-"
        elif path.is_dir():
            size = dir_size_human(path)
        else:
            size = "-"
        log.info("  %-12s %-16s %s", label, status, size)
    log.info("")

    # Comando sugerido de FASE 0
    repo_root = datasets_dir.parent
    dd = datasets_dir
    commit_file = dd / "panorama_labels_commit.txt"
    commit_val = "$(cat {})".format(commit_file)

    log.info("Siguiente paso — FASE 0:")
    log.info("")
    cmd = textwrap.dedent("""\
        cd {repo}

        python3 src/pipeline/fase0_extract_embeddings.py \\
          --backbone          vit_tiny_patch16_224 \\
          --batch_size        64 \\
          --output_dir        {repo}/embeddings/vit_tiny \\
          --chest_csv         {dd}/nih_chest_xrays/Data_Entry_2017.csv \\
          --chest_imgs        {dd}/nih_chest_xrays/all_images \\
          --chest_train_list  {dd}/nih_chest_xrays/train_val_list.txt \\
          --chest_val_list    {dd}/nih_chest_xrays/test_list.txt \\
          --chest_view_filter PA \\
          --chest_bbox_csv    {dd}/nih_chest_xrays/BBox_List_2017.csv \\
          --isic_gt           {dd}/isic_2019/ISIC_2019_Training_GroundTruth.csv \\
          --isic_imgs         "{dd}/isic_2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input" \\
          --isic_metadata     {dd}/isic_2019/ISIC_2019_Training_Metadata.csv \\
          --oa_root           {dd}/osteoarthritis/oa_splits \\
          --pancreas_nii_dir       {dd}/zenodo_13715870 \\
          --pancreas_labels_dir    {dd}/panorama_labels \\
          --pancreas_labels_commit {commit} \\
          --pancreas_roi_strategy  A""").format(
        repo=repo_root, dd=dd, commit=commit_val)
    for line in cmd.splitlines():
        log.info("  %s", line)
    log.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador principal
# ─────────────────────────────────────────────────────────────────────────────

def resolve_active(args):
    # type: (argparse.Namespace) -> set
    """Calcula qué datasets están activos según --only / --skip."""
    all_ds = {"nih", "isic", "oa", "luna", "luna_ct", "pancreas", "panorama"}
    if args.only:
        active = set()
        for name in args.only:
            name = name.lower()
            if name in all_ds:
                active.add(name)
            else:
                log.warning("Dataset desconocido en --only: '%s' (opciones: %s)",
                            name, ", ".join(sorted(all_ds)))
        return active
    active = set(all_ds)
    if args.skip:
        for name in args.skip:
            name = name.lower()
            if name in active:
                active.discard(name)
            else:
                log.warning("Dataset desconocido en --skip: '%s'", name)
    return active


def main():
    parser = argparse.ArgumentParser(
        description="Setup completo de datasets — Proyecto MoE (5 expertos de dominio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              python3 setup_datasets.py
              python3 setup_datasets.py --only oa isic --no_extract
              python3 setup_datasets.py --skip nih pancreas --luna_subsets 0 1 2
              python3 setup_datasets.py --dry_run
        """),
    )
    parser.add_argument(
        "--repo_root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Ruta raíz del proyecto (default: padre de scripts/)",
    )
    parser.add_argument(
        "--skip", nargs="+", default=None,
        help="Datasets a saltar: nih isic oa luna luna_ct pancreas panorama",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Solo procesar estos datasets",
    )
    parser.add_argument(
        "--no_extract", action="store_true",
        help="Solo descargar, sin descomprimir",
    )
    parser.add_argument(
        "--no_download", action="store_true",
        help="Solo descomprimir (asume ZIPs ya presentes)",
    )
    parser.add_argument(
        "--luna_subsets", nargs="+", type=int, default=[0],
        help="Subsets CT de LUNA a descargar (default: 0)",
    )
    parser.add_argument(
        "--ram_pause_mb", type=float, default=700,
        help="Umbral de RAM para pausar extracción (MB, default: 700)",
    )
    parser.add_argument(
        "--ram_resume_mb", type=float, default=1400,
        help="Umbral de RAM para reanudar extracción (MB, default: 1400)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Solo verificar estado actual, sin descargar ni extraer",
    )
    parser.add_argument(
        "--disco",
        action="store_true",
        help=(
            "Modo ahorro de disco: tras descargar cada ZIP, extraerlo inmediatamente "
            "y borrarlo antes de descargar el siguiente. Reduce el pico de uso de disco "
            "a ~1 dataset a la vez en lugar de todos los ZIPs simultáneos. "
            "Útil cuando el disco libre es < suma de todos los ZIPs (~120 GB)."
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    datasets_dir = repo_root / "datasets"
    log_dir = repo_root / "logs"

    global log
    log = setup_logging(log_dir)

    log.info("repo_root:    %s", repo_root)
    log.info("datasets_dir: %s", datasets_dir)
    log.info("RAM actual:   %.0f MB disponible", ram_available_mb())

    # ── Crear directorios ────────────────────────────────────────────────
    for subdir in (
        "nih_chest_xrays", "isic_2019", "osteoarthritis",
        "luna_lung_cancer", "luna_lung_cancer/ct_volumes",
        "zenodo_13715870", "panorama_labels",
    ):
        (datasets_dir / subdir).mkdir(parents=True, exist_ok=True)

    active = resolve_active(args)
    log.info("Datasets activos: %s", ", ".join(sorted(active)))

    # ── Validaciones mutuamente excluyentes ─────────────────────────────
    if args.disco and args.no_extract:
        log.error("--disco y --no_extract son mutuamente excluyentes.")
        sys.exit(1)
    if args.disco and args.no_download:
        log.error("--disco y --no_download son mutuamente excluyentes.")
        sys.exit(1)

    # ── FASE 1 — Prerequisitos ───────────────────────────────────────────
    if not check_prerequisites(active):
        log.error("Prerequisitos no cumplidos. Corrige los errores y reintenta.")
        sys.exit(1)

    download_results = {}  # type: dict
    extract_results = {}   # type: dict

    # ── Dry run: solo estado ─────────────────────────────────────────────
    if args.dry_run:
        log.info("")
        log.info("[DRY RUN] Solo verificando estado — sin descargar ni extraer.")
        print_summary(datasets_dir, download_results, extract_results, active, args)
        return

    if args.disco:
        # ── MODO --disco: descarga → extrae → borra ZIP, dataset por dataset ──────
        log.info("")
        log.info("=" * 60)
        log.info("[--disco] Modo activo: cada dataset se extrae y borra inmediatamente.")
        log.info("[--disco] Orden: OA -> LUNA -> ISIC -> NIH -> LUNA-CT (uno a uno) -> PANCREAS")
        log.info("=" * 60)

        disco_plan = [
            ("oa",
             lambda: download_oa(datasets_dir),
             datasets_dir / "osteoarthritis" / "osteoarthritis.zip",
             datasets_dir / "osteoarthritis",
             "OA",
             lambda: is_extracted(datasets_dir, "oa")),
            ("luna",
             lambda: download_luna_meta(datasets_dir),
             datasets_dir / "luna_lung_cancer" / "luna-lung-cancer-dataset.zip",
             datasets_dir / "luna_lung_cancer",
             "LUNA",
             lambda: is_extracted(datasets_dir, "luna")),
            ("isic",
             lambda: download_isic(datasets_dir),
             datasets_dir / "isic_2019" / "isic-2019.zip",
             datasets_dir / "isic_2019",
             "ISIC",
             lambda: is_extracted(datasets_dir, "isic")),
            ("nih",
             lambda: download_nih(datasets_dir),
             datasets_dir / "nih_chest_xrays" / "data.zip",
             datasets_dir / "nih_chest_xrays",
             "NIH",
             lambda: is_extracted(datasets_dir, "nih")),
        ]

        for ds_id, dl_fn, archive, dest, tag_d, check_fn in disco_plan:
            if ds_id not in active:
                continue
            try:
                dl_ok, ex_ok = download_and_extract_one(
                    ds_id, dl_fn, archive, dest, tag_d, args, check_fn)
                download_results[ds_id] = dl_ok
                extract_results[ds_id] = ex_ok
            except Exception as e:
                log.error("[%s] Error inesperado en modo --disco: %s", ds_id.upper(), e)
                download_results[ds_id] = False
                extract_results[ds_id] = False

        # LUNA CT subsets uno por uno
        if "luna_ct" in active:
            ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
            ct_dir.mkdir(parents=True, exist_ok=True)
            all_ct_dl = True
            all_ct_ex = True
            for i in args.luna_subsets:
                url_template = ZENODO_LUNA_URL_PART1 if i <= 6 else ZENODO_LUNA_URL_PART2
                archive_ct = ct_dir / "subset{}.zip".format(i)
                tag_i = "LUNA-CT{}".format(i)
                check_i = lambda idx=i: _luna_ct_extracted(datasets_dir, [idx])
                try:
                    dl_ok, ex_ok = download_and_extract_one(
                        "luna_ct",
                        lambda idx=i, url_t=url_template, arch=archive_ct: download_wget(
                            url_t.format(i=idx), arch, "LUNA-CT{}".format(idx),
                            min_size_bytes=MIN_ZIP_SIZES["luna_ct"],
                        ),
                        archive_ct, ct_dir, tag_i, args, check_i,
                    )
                    if ex_ok:
                        verify_luna_ct_subset(datasets_dir, i)
                    if not dl_ok:
                        all_ct_dl = False
                    if not ex_ok:
                        all_ct_ex = False
                except Exception as e:
                    log.error("[LUNA-CT%d] Error inesperado en modo --disco: %s", i, e)
                    all_ct_dl = False
                    all_ct_ex = False
            download_results["luna_ct"] = all_ct_dl
            extract_results["luna_ct"] = all_ct_ex

        # PANCREAS
        if "pancreas" in active:
            try:
                dl_ok, ex_ok = download_and_extract_one(
                    "pancreas",
                    lambda: download_pancreas(datasets_dir),
                    datasets_dir / "zenodo_13715870" / "batch_1.zip",
                    datasets_dir / "zenodo_13715870",
                    "PANCREAS",
                    args,
                    lambda: _pancreas_extracted(datasets_dir),
                )
                download_results["pancreas"] = dl_ok
                extract_results["pancreas"] = ex_ok
            except Exception as e:
                log.error("[PANCREAS] Error inesperado en modo --disco: %s", e)
                download_results["pancreas"] = False
                extract_results["pancreas"] = False

        # PANORAMA nunca tiene ZIP — lógica idéntica al flujo normal
        if "panorama" in active:
            try:
                ok = download_panorama(datasets_dir)
                download_results["panorama"] = ok
            except Exception as e:
                log.error("[PANORAMA] Error inesperado en descarga: %s", e)
                download_results["panorama"] = False

        log.info("")
        log.info("[--disco] Descarga + extracción completada inline — saltando FASE 3.")

    else:
        # ── FASE 2 — Descargas ───────────────────────────────────────────────
        if not args.no_download:
            log.info("")
            log.info("=" * 60)
            log.info("FASE 2 — Descargas")
            log.info("=" * 60)

            download_map = {
                "nih":      lambda: download_nih(datasets_dir),
                "isic":     lambda: download_isic(datasets_dir),
                "oa":       lambda: download_oa(datasets_dir),
                "luna":     lambda: download_luna_meta(datasets_dir),
                "luna_ct":  lambda: download_luna_ct(datasets_dir, args.luna_subsets),
                "pancreas": lambda: download_pancreas(datasets_dir),
                "panorama": lambda: download_panorama(datasets_dir),
            }

            # Orden: pequeños primero, grandes después
            download_order = ["oa", "luna", "isic", "panorama", "nih", "luna_ct", "pancreas"]

            for ds_id in download_order:
                if ds_id not in active:
                    continue
                fn = download_map.get(ds_id)
                if fn is None:
                    continue
                try:
                    ok = fn()
                    download_results[ds_id] = ok
                    if ok:
                        log.info("[%s] Descarga OK", ds_id.upper())
                    else:
                        log.warning("[%s] Descarga FALLÓ — continuando.", ds_id.upper())
                except Exception as e:
                    log.error("[%s] Error inesperado en descarga: %s", ds_id.upper(), e)
                    download_results[ds_id] = False

        # ── FASE 3 — Extracción ─────────────────────────────────────────────
        if not args.no_extract:
            log.info("")
            log.info("=" * 60)
            log.info("FASE 3 — Extracción (monitor de RAM: pausa=%dMB / reanuda=%dMB)",
                     args.ram_pause_mb, args.ram_resume_mb)
            log.info("=" * 60)
            extract_results = run_extractions(datasets_dir, active, args)

    # ── FASE 4 — Post-extracción ────────────────────────────────────────
    if not args.no_extract:
        log.info("")
        log.info("=" * 60)
        log.info("FASE 4 — Preparación post-extracción")
        log.info("=" * 60)

        if "nih" in active:
            try:
                post_nih(datasets_dir)
            except Exception as e:
                log.error("[NIH] Error en post-extracción: %s", e)

        if "oa" in active:
            try:
                post_oa(datasets_dir)
            except Exception as e:
                log.error("[OA] Error en post-extracción: %s", e)

        if "pancreas" in active:
            try:
                post_pancreas(datasets_dir)
            except Exception as e:
                log.error("[PANCREAS] Error en post-extracción: %s", e)

    # ── Resumen ──────────────────────────────────────────────────────────
    print_summary(datasets_dir, download_results, extract_results, active, args)
    log.info("setup_datasets.py finalizado.")


if __name__ == "__main__":
    main()
