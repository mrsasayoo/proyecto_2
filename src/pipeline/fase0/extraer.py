#!/usr/bin/env python3
"""
extraer.py — Extracción de archivos comprimidos con 7z (sin fallback a unzip)
==============================================================================
Fase 0 — Preparación de Datos | Proyecto MoE Médico

Responsabilidad única: extraer todos los ZIPs descargados usando exclusivamente 7z.
Si 7z no está instalado, termina inmediatamente con instrucciones de instalación.

Origen: funciones de extracción de scripts/setup_datasets.py
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

try:
    import psutil
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "psutil"])
    import psutil

log = logging.getLogger("fase0.extraer")


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


def ram_available_mb():
    # type: () -> float
    return psutil.virtual_memory().available / (1024 * 1024)


def check_7z():
    # type: () -> bool
    """Verifica disponibilidad de 7z. OBLIGATORIO — no hay fallback."""
    if shutil.which("7z"):
        log.info("  7z OK (%s)", shutil.which("7z"))
        return True
    log.error(
        "  7z NO disponible. Es OBLIGATORIO para extraer los datasets.\n"
        "  Los ZIPs de LUNA16 subsets 7 y 9 son ZIP64 y unzip falla silenciosamente.\n"
        "  Instalar con: sudo apt-get install p7zip-full\n"
        "  No hay fallback a unzip en esta versión."
    )
    return False


# ── Monitor de RAM ────────────────────────────────────────────────────────────

class RAMMonitor:
    """Hilo demonio que pausa/reanuda un proceso 7z según RAM disponible."""

    def __init__(self, pid, tag, pause_mb, resume_mb, check_interval=3):
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
                    log.warning("[%s][RAM] SIGSTOP — RAM: %.0f MB < %.0f MB",
                                self.tag, avail, self.pause_mb)
                elif self._paused and avail > self.resume_mb:
                    os.kill(self.pid, signal.SIGCONT)
                    self._paused = False
                    log.info("[%s][RAM] SIGCONT — RAM: %.0f MB > %.0f MB",
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
    log.info("[%s] RAM OK: %.0f MB", tag, avail)


# ── Extracción ────────────────────────────────────────────────────────────────

def smart_extract(archive, dest, tag, pause_mb, resume_mb):
    # type: (Path, Path, str, float, float) -> bool
    """Extracción con 7z + monitor de RAM (SIGSTOP/SIGCONT)."""
    if not archive.exists():
        log.error("[%s] Archivo no encontrado: %s", tag, archive)
        return False

    wait_for_ram(tag, resume_mb)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["7z", "x", str(archive), "-o" + str(dest), "-y"]
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

    if rc == 0:
        log.info("[%s] OK extraído en %.0fs", tag, elapsed)
        return True
    elif rc == 1:
        log.warning(
            "[%s] 7z terminó con warnings (rc=1) — posible ZIP64 con datos "
            "corruptos en origen (LUNA16 subsets 7/9). Verificar conteo.", tag)
        return True
    else:
        log.error("[%s] 7z falló (rc=%d): %s", tag, rc, stderr_text[:300])
        return False


# ── Verificaciones de extracción ──────────────────────────────────────────────

def is_extracted(datasets_dir, ds_id, luna_subsets=None):
    # type: (Path, str, list|None) -> bool
    """Verifica si un dataset ya está extraído."""
    checks = {
        "nih":      lambda: (datasets_dir / "nih_chest_xrays" / "images_001").is_dir(),
        "isic":     lambda: (datasets_dir / "isic_2019" / "ISIC_2019_Training_Input").is_dir(),
        "oa":       lambda: (datasets_dir / "osteoarthritis" / "KLGrade").is_dir(),
        "luna":     lambda: (datasets_dir / "luna_lung_cancer" / "candidates_V2").is_dir(),
        "luna_ct":  lambda: _luna_ct_extracted(datasets_dir, luna_subsets or [0]),
        "pancreas": lambda: _pancreas_extracted(datasets_dir),
    }
    checker = checks.get(ds_id)
    return checker() if checker else False


def _luna_ct_extracted(datasets_dir, subsets):
    # type: (Path, list) -> bool
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    for i in subsets:
        subset_dir = ct_dir / "subset{}".format(i)
        if not subset_dir.is_dir():
            return False
        if len(list(subset_dir.glob("*.mhd"))) < 5:
            return False
    return True


def _pancreas_extracted(datasets_dir):
    # type: (Path) -> bool
    zenodo_dir = datasets_dir / "zenodo_13715870"
    return len(list(zenodo_dir.rglob("*.nii.gz"))) >= 1


def verify_luna_ct_subset(datasets_dir, subset_idx):
    # type: (Path, int) -> dict
    """Verifica un subset CT extraído."""
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    subset_dir = ct_dir / "subset{}".format(subset_idx)
    mhd_count = len(list(subset_dir.glob("*.mhd"))) if subset_dir.exists() else 0
    raw_count = len(list(subset_dir.glob("*.raw"))) if subset_dir.exists() else 0
    expected = 178 if subset_idx == 0 else 85
    ok = mhd_count >= expected and mhd_count == raw_count
    if ok:
        log.info("[LUNA-CT%d] Verificación OK: %d .mhd / %d .raw",
                 subset_idx, mhd_count, raw_count)
    else:
        log.warning(
            "[LUNA-CT%d] Extracción posiblemente parcial: %d .mhd / %d .raw "
            "(esperados >= %d pares).",
            subset_idx, mhd_count, raw_count, expected)
    return {"mhd": mhd_count, "raw": raw_count, "ok": ok, "expected": expected}


def _extract_nih_split_txts(data_zip_path, nih_dir):
    # type: (Path, Path) -> None
    """Extrae train_val_list.txt y test_list.txt del data.zip antes de borrarlo (modo --disco)."""
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
                    log.info("[NIH][--disco] %s extraído (%d líneas)", txt_name, line_count)
    except Exception as e:
        log.warning("[NIH][--disco] No se pudieron extraer txts de splits: %s", e)


# ── Orquestador de extracciones ───────────────────────────────────────────────

def run_extractions(datasets_dir, active, luna_subsets=None, pause_mb=700,
                    resume_mb=1400, disco=False):
    # type: (Path, set, list|None, float, float, bool) -> dict
    """Ejecuta las extracciones. Retorna {ds_id: bool}."""
    results = {}

    # Orden: menor a mayor
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

    for i in (luna_subsets or []):
        extraction_plan.append(
            ("luna_ct",
             datasets_dir / "luna_lung_cancer" / "ct_volumes" / "subset{}.zip".format(i),
             datasets_dir / "luna_lung_cancer" / "ct_volumes",
             [i]),
        )

    extraction_plan.append(
        ("pancreas", datasets_dir / "zenodo_13715870" / "batch_1.zip",
         datasets_dir / "zenodo_13715870", None),
    )

    for ds_id, archive, dest, subset_override in extraction_plan:
        if ds_id not in active:
            continue

        tag = ds_id.upper()
        if ds_id == "luna_ct":
            tag = "LUNA-CT{}".format(archive.stem.replace("subset", ""))

        check_subsets = subset_override if subset_override is not None else luna_subsets
        if is_extracted(datasets_dir, ds_id, check_subsets):
            log.info("[%s] Ya extraído, saltando.", tag)
            results[ds_id] = results.get(ds_id, True)
            continue

        if not archive.exists():
            log.warning("[%s] %s no encontrado — pendiente.", tag, archive.name)
            results[ds_id] = False
            continue

        try:
            ok = smart_extract(archive, dest, tag, pause_mb, resume_mb)
            results[ds_id] = results.get(ds_id, True) and ok
            if ds_id == "luna_ct":
                subset_idx = int(archive.stem.replace("subset", ""))
                verify_luna_ct_subset(datasets_dir, subset_idx)
            # Modo disco: borrar ZIP tras extracción
            if disco and ok and archive.exists():
                if ds_id == "nih":
                    _extract_nih_split_txts(archive, dest)
                zip_size = file_size_human(archive)
                archive.unlink()
                log.info("[%s] --disco: ZIP eliminado (%s liberados).", tag, zip_size)
        except Exception as e:
            log.error("[%s] Error de extracción: %s", tag, e)
            results[ds_id] = False

    return results
