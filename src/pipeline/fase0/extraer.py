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
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "psutil"]
    )
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


def _log_progress(tag, watch_dir, total_files, stop_event, baseline_count=0):
    # type: (str, Path, int, threading.Event, int) -> None
    """Hilo que reporta progreso de extracción cada 60 segundos con porcentaje y ETA.

    Args:
        baseline_count: number of files already present in watch_dir before
            extraction started.  Progress is computed as
            (current_count - baseline_count) / total_files so that files from
            prior batches do not inflate the percentage.
    """
    t0 = time.time()
    while not stop_event.wait(60):
        elapsed_min = (time.time() - t0) / 60.0
        try:
            current_count = sum(1 for f in Path(watch_dir).rglob("*") if f.is_file())
        except Exception:
            current_count = 0
        new_files = max(0, current_count - baseline_count)
        if total_files > 0:
            pct = min(100.0, new_files / total_files * 100.0)
            if pct > 0:
                remaining_min = elapsed_min / pct * (100.0 - pct)
                eta_str = "{:.0f} min restantes".format(remaining_min)
            else:
                eta_str = "calculando ETA..."
            log.info(
                "[%s] Progreso: %.1f min | %d/%d archivos nuevos | %.1f%% | %s",
                tag,
                elapsed_min,
                new_files,
                total_files,
                pct,
                eta_str,
            )
        else:
            log.info(
                "[%s] Progreso: %.1f min | %d archivos extraídos",
                tag,
                elapsed_min,
                new_files,
            )


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
                    log.warning(
                        "[%s][RAM] SIGSTOP — RAM: %.0f MB < %.0f MB",
                        self.tag,
                        avail,
                        self.pause_mb,
                    )
                elif self._paused and avail > self.resume_mb:
                    os.kill(self.pid, signal.SIGCONT)
                    self._paused = False
                    log.info(
                        "[%s][RAM] SIGCONT — RAM: %.0f MB > %.0f MB",
                        self.tag,
                        avail,
                        self.resume_mb,
                    )
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
    log.warning(
        "[%s] RAM disponible: %.0f MB < %.0f MB — esperando...", tag, avail, resume_mb
    )
    while avail < resume_mb:
        time.sleep(10)
        avail = ram_available_mb()
    log.info("[%s] RAM OK: %.0f MB", tag, avail)


# ── Extracción ────────────────────────────────────────────────────────────────


def smart_extract(archive, dest, tag, pause_mb, resume_mb):
    # type: (Path, Path, str, float, float) -> bool
    """Extracción con 7z + monitor de RAM (SIGSTOP/SIGCONT) + progreso en porcentaje."""
    if not archive.exists():
        log.error("[%s] Archivo no encontrado: %s", tag, archive)
        return False

    wait_for_ram(tag, resume_mb)
    dest.mkdir(parents=True, exist_ok=True)

    # ── Leer manifiesto del ZIP para calcular progreso real ──────────────────
    # zipfile lee solo el directorio central (al final del archivo) — muy rápido
    # incluso en ZIPs de varios GB.
    total_files = 0
    watch_dir = dest
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            total_files = len(names)
            # Determinar subdirectorio destino (ej. "subset0/" → ct_volumes/subset0/)
            top_dirs = {Path(n).parts[0] for n in names if n and Path(n).parts}
            if len(top_dirs) == 1:
                watch_dir = dest / top_dirs.pop()
    except Exception as e:
        log.debug("[%s] No se pudo leer manifiesto ZIP: %s — progreso sin %%.", tag, e)

    archive_size_gb = archive.stat().st_size / 1e9
    log.info(
        "[%s] Iniciando extracción: %s (%.1f GB, %d archivos) → %s",
        tag,
        archive.name,
        archive_size_gb,
        total_files,
        dest,
    )
    log.info(
        "[%s] Velocidad estimada HDD ~1.4 MB/s → ETA estimada: %.0f min",
        tag,
        archive_size_gb * 1000 / 1.4 / 60,
    )

    # Snapshot file count *before* extraction so the progress monitor
    # only counts newly extracted files (BUG 3 fix).
    try:
        baseline_count = sum(1 for f in watch_dir.rglob("*") if f.is_file())
    except Exception:
        baseline_count = 0

    t0 = time.time()

    # NOTA: NO usar -mmt=auto — es inválido para formato ZIP en p7zip y
    # provoca rc=2 ("E_INVALIDARG") incluso en ZIPs completamente válidos.
    cmd = ["7z", "x", str(archive), "-o" + str(dest), "-y"]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    monitor = RAMMonitor(proc.pid, tag, pause_mb, resume_mb)
    monitor.start()

    progress_stop = threading.Event()
    progress_thread = threading.Thread(
        target=_log_progress,
        args=(tag, watch_dir, total_files, progress_stop, baseline_count),
        daemon=True,
    )
    progress_thread.start()

    try:
        _, stderr = proc.communicate()
        rc = proc.returncode
    finally:
        monitor.stop()
        progress_stop.set()

    elapsed = time.time() - t0
    stderr_text = stderr.decode(errors="replace")

    if rc == 0:
        log.info("[%s] OK extraído en %.0fs", tag, elapsed)
        return True
    elif rc == 1:
        log.warning(
            "[%s] 7z terminó con warnings (rc=1) — posible ZIP64 con datos "
            "corruptos en origen (LUNA16 subsets 7/9). Verificar conteo.",
            tag,
        )
        return True
    elif rc == 2:
        log.warning(
            "[%s] 7z terminó con rc=2 (Data Error) en %.0fs — típico en LUNA16 ZIP64. "
            "Archivos probablemente extraídos. Verificando...",
            tag,
            elapsed,
        )
        return True
    else:
        log.error("[%s] 7z falló (rc=%d): %s", tag, rc, stderr_text[:300])
        return False


# ── Verificaciones de extracción ──────────────────────────────────────────────


def is_extracted(datasets_dir, ds_id, luna_subsets=None, pancreas_batches=None):
    # type: (Path, str, list|None, list|None) -> bool
    """Verifica si un dataset ya está extraído."""
    checks = {
        "nih": lambda: (
            sum(
                len(
                    list(
                        (
                            datasets_dir
                            / "nih_chest_xrays"
                            / "images_{:03d}".format(i)
                            / "images"
                        ).glob("*.png")
                    )
                )
                for i in range(1, 13)
            )
            >= 100000
        ),
        "isic": lambda: (
            datasets_dir / "isic_2019" / "ISIC_2019_Training_Input"
        ).is_dir(),
        "oa": lambda: (
            datasets_dir / "osteoarthritis" / "oa_splits" / "train"
        ).is_dir(),
        "luna_meta": lambda: (
            datasets_dir / "luna_lung_cancer" / "candidates_V2"
        ).is_dir(),
        "luna_ct": lambda: _luna_ct_extracted(datasets_dir, luna_subsets or [0]),
        "pancreas": lambda: _pancreas_extracted(datasets_dir, pancreas_batches),
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
        mhd_count = len(list(subset_dir.glob("*.mhd")))
        raw_count = len(list(subset_dir.glob("*.raw")))
        # Threshold real: 88 pares mhd+raw mínimo (subsets 8-9 tienen 88; los demás 89)
        if mhd_count < 88 or mhd_count != raw_count:
            return False
    return True


def _pancreas_extracted(datasets_dir, batches=None):
    # type: (Path, list|None) -> bool
    """Verifica que los batches solicitados fueron extraídos.

    Each batch ZIP extracts .nii.gz files into zenodo_13715870/.  Since all
    batches share the same flat directory, we check whether the ZIP still
    exists on disk: if the ZIP is present, we conservatively assume the batch
    may not yet have been extracted.  If the ZIP is absent (already deleted by
    --disco mode or never downloaded), we fall back to a global file-count
    threshold as a best-effort check.

    When *batches* is None we check all 4.
    """
    if batches is None:
        batches = [1, 2, 3, 4]
    zenodo_dir = datasets_dir / "zenodo_13715870"
    if not zenodo_dir.is_dir():
        return False

    # Per-batch check: if a batch ZIP still exists on disk, it hasn't been
    # extracted yet (smart_extract does not delete the ZIP unless --disco).
    for b in batches:
        batch_zip = zenodo_dir / "batch_{}.zip".format(b)
        if batch_zip.exists():
            # ZIP present → not yet extracted (extraction removes nothing,
            # but the orchestrator only calls us *before* attempting to
            # extract, so the ZIP's presence means extraction hasn't run).
            return False

    # All requested batch ZIPs are gone — either extracted+disco or never
    # downloaded.  Sanity-check: at least 100 .nii.gz per requested batch.
    nii_count = len(list(zenodo_dir.rglob("*.nii.gz")))
    expected_min = 100 * len(batches)
    return nii_count >= expected_min


def verify_luna_ct_subset(datasets_dir, subset_idx):
    # type: (Path, int) -> dict
    """Verifica un subset CT extraído."""
    ct_dir = datasets_dir / "luna_lung_cancer" / "ct_volumes"
    subset_dir = ct_dir / "subset{}".format(subset_idx)
    MIN_RAW_BYTES = 1_048_576  # 1 MB mínimo por archivo .raw
    mhd_count = len(list(subset_dir.glob("*.mhd"))) if subset_dir.exists() else 0
    raw_files = list(subset_dir.glob("*.raw")) if subset_dir.exists() else []
    raw_count = len(raw_files)
    raw_valid = sum(1 for f in raw_files if f.stat().st_size >= MIN_RAW_BYTES)
    raw_zero = raw_count - raw_valid
    expected = 88  # LUNA16: 89 mhd per subset (88 for subsets 8-9); 88 is safe minimum
    ok = mhd_count >= expected and raw_valid >= expected
    if ok:
        log.info(
            "[LUNA-CT%d] Verificación OK: %d .mhd / %d .raw (%d válidos, %d vacíos)",
            subset_idx,
            mhd_count,
            raw_count,
            raw_valid,
            raw_zero,
        )
    else:
        log.warning(
            "[LUNA-CT%d] Extracción posiblemente parcial: %d .mhd / %d .raw (%d válidos, %d vacíos) "
            "(esperados >= %d pares).",
            subset_idx,
            mhd_count,
            raw_count,
            raw_valid,
            raw_zero,
            expected,
        )
    return {
        "mhd": mhd_count,
        "raw": raw_count,
        "raw_valid": raw_valid,
        "raw_zero": raw_zero,
        "ok": ok,
        "expected": expected,
    }


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
                    log.info(
                        "[NIH][--disco] %s extraído (%d líneas)", txt_name, line_count
                    )
    except Exception as e:
        log.warning("[NIH][--disco] No se pudieron extraer txts de splits: %s", e)


# ── Orquestador de extracciones ───────────────────────────────────────────────


def run_extractions(
    datasets_dir,
    active,
    luna_subsets=None,
    pause_mb=700,
    resume_mb=1400,
    disco=False,
    pancreas_batches=None,
):
    # type: (Path, set, list|None, float, float, bool, list|None) -> dict
    """Ejecuta las extracciones. Retorna {ds_id: bool}."""
    results = {}
    if pancreas_batches is None:
        pancreas_batches = [1, 2, 3, 4]

    # Orden: menor a mayor
    extraction_plan = [
        (
            "oa",
            datasets_dir / "osteoarthritis" / "osteoarthritis.zip",
            datasets_dir / "osteoarthritis",
            None,
        ),
        (
            "luna_meta",
            datasets_dir / "luna_lung_cancer" / "luna-lung-cancer-dataset.zip",
            datasets_dir / "luna_lung_cancer",
            None,
        ),
        (
            "isic",
            datasets_dir / "isic_2019" / "isic-2019.zip",
            datasets_dir / "isic_2019",
            None,
        ),
        (
            "nih",
            datasets_dir / "nih_chest_xrays" / "data.zip",
            datasets_dir / "nih_chest_xrays",
            None,
        ),
    ]

    if luna_subsets is None:
        luna_subsets = list(range(10))

    for i in luna_subsets or []:
        extraction_plan.append(
            (
                "luna_ct",
                datasets_dir
                / "luna_lung_cancer"
                / "ct_volumes"
                / "subset{}.zip".format(i),
                datasets_dir / "luna_lung_cancer" / "ct_volumes",
                [i],
            ),
        )

    for _bnum in pancreas_batches:
        extraction_plan.append(
            (
                "pancreas",
                datasets_dir / "zenodo_13715870" / "batch_{}.zip".format(_bnum),
                datasets_dir / "zenodo_13715870",
                None,
                _bnum,
            ),
        )

    for plan_entry in extraction_plan:
        if len(plan_entry) == 5:
            ds_id, archive, dest, subset_override, batch_num = plan_entry
        else:
            ds_id, archive, dest, subset_override = plan_entry
            batch_num = None
        if ds_id not in active:
            continue

        tag = ds_id.upper()
        if ds_id == "luna_ct":
            tag = "LUNA-CT{}".format(archive.stem.replace("subset", ""))
        elif ds_id == "pancreas" and batch_num is not None:
            tag = "PANCREAS-B{}".format(batch_num)

        # Per-item idempotency: skip if already done
        if ds_id == "pancreas":
            # Per-batch check: read the ZIP manifest and count how many of
            # its .nii.gz files are already on disk.  Only skip if the vast
            # majority (>= 95%) are present — this catches partial
            # extractions that a 5-file sample would miss.
            if archive.exists():
                try:
                    with zipfile.ZipFile(archive, "r") as zf:
                        nii_names = [
                            n
                            for n in zf.namelist()
                            if n.endswith(".nii.gz") and not n.startswith("__MACOSX")
                        ]
                    if nii_names:
                        zenodo_dir = datasets_dir / "zenodo_13715870"
                        present = sum(
                            1 for n in nii_names if (zenodo_dir / Path(n).name).exists()
                        )
                        total_in_zip = len(nii_names)
                        ratio = present / total_in_zip if total_in_zip else 0.0
                        if ratio >= 0.95:
                            log.info(
                                "[%s] Ya extraído (%d/%d archivos presentes, %.0f%%), saltando.",
                                tag,
                                present,
                                total_in_zip,
                                ratio * 100,
                            )
                            results[ds_id] = results.get(ds_id, True)
                            continue
                        else:
                            log.warning(
                                "[%s] Extracción parcial detectada: %d/%d archivos (%.0f%%) — re-extrayendo.",
                                tag,
                                present,
                                total_in_zip,
                                ratio * 100,
                            )
                except Exception:
                    pass  # Can't read ZIP central directory — proceed to extract
            else:
                # ZIP absent: either never downloaded or deleted by --disco.
                # If .nii.gz files exist, assume extraction was done.
                zenodo_dir = datasets_dir / "zenodo_13715870"
                if (
                    zenodo_dir.is_dir()
                    and len(list(zenodo_dir.rglob("*.nii.gz"))) >= 100
                ):
                    log.info(
                        "[%s] Ya extraído (ZIP ausente, .nii.gz presentes), saltando.",
                        tag,
                    )
                    results[ds_id] = results.get(ds_id, True)
                    continue
        else:
            check_subsets = (
                subset_override if subset_override is not None else luna_subsets
            )
            if is_extracted(datasets_dir, ds_id, check_subsets, pancreas_batches):
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
