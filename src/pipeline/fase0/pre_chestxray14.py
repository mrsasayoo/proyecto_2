#!/usr/bin/env python3
"""
pre_chestxray14.py — Post-procesado específico de NIH ChestXray14
==================================================================
Fase 0 — Preparación de Datos | Proyecto MoE Médico

Responsabilidad única: preparación post-extracción de NIH ChestXray14.
- Crear directorio unificado all_images/ con symlinks relativos
- Verificar archivos de split oficiales
- Auditar Data_Entry_2017.csv

Origen: post_nih() de setup_datasets.py + fix1_nih_val_split() de fix_alignment.py
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("fase0.pre_chestxray14")


def crear_symlinks_all_images(nih_dir):
    # type: (Path) -> int
    """
    Crea all_images/ con symlinks relativos a los PNGs en las 12 subcarpetas.
    Symlinks relativos para portabilidad entre servidores.
    Retorna el número de symlinks creados.
    """
    all_imgs = nih_dir / "all_images"

    if all_imgs.is_dir():
        n_links = sum(1 for p in all_imgs.iterdir() if p.is_symlink())
        if n_links > 0:
            log.info("[NIH] all_images/ ya existe (%d symlinks), saltando.", n_links)
            return n_links

    # Buscar PNGs en images_001..012
    png_files = sorted(nih_dir.glob("images_*/images/*.png"))
    if not png_files:
        log.warning("[NIH] No se encontraron .png — all_images pendiente.")
        return 0

    if all_imgs.exists():
        shutil.rmtree(all_imgs)

    all_imgs.mkdir(parents=True, exist_ok=True)
    created = 0
    for png in png_files:
        link = all_imgs / png.name
        if not link.exists():
            # Symlink relativo para portabilidad
            target = os.path.relpath(png, all_imgs)
            link.symlink_to(target)
            created += 1
    log.info("[NIH] %d symlinks creados en all_images/", created)
    return created


def verificar_split_txts(nih_dir):
    # type: (Path) -> dict
    """
    Comprueba que train_val_list.txt y test_list.txt existen y no están truncados.
    Extrae desde data.zip si es necesario.
    Retorna dict con conteos.
    """
    min_lines = {"train_val_list.txt": 80000, "test_list.txt": 20000}
    result = {}

    for txt_name, min_count in min_lines.items():
        txt_path = nih_dir / txt_name
        needs_extract = False

        if not txt_path.exists():
            log.warning("[NIH] %s no encontrado.", txt_name)
            needs_extract = True
        else:
            count = sum(1 for _ in open(txt_path, encoding="utf-8"))
            if count < min_count:
                log.warning("[NIH] %s truncado (%d < %d).", txt_name, count, min_count)
                needs_extract = True
            else:
                log.info("[NIH] %s OK (%d entradas)", txt_name, count)
                result[txt_name] = count

        if needs_extract:
            data_zip = nih_dir / "data.zip"
            if data_zip.exists():
                log.info("[NIH] Extrayendo %s de data.zip...", txt_name)
                try:
                    with zipfile.ZipFile(data_zip, "r") as zf:
                        candidates = [n for n in zf.namelist() if n.endswith(txt_name)]
                        if candidates:
                            data = zf.read(candidates[0])
                            txt_path.write_bytes(data)
                            count = data.count(b"\n")
                            log.info("[NIH] %s extraído (%d líneas)", txt_name, count)
                            result[txt_name] = count
                except Exception as e:
                    log.error("[NIH] Error extrayendo %s: %s", txt_name, e)

    return result


def auditar_csv(nih_dir):
    # type: (Path) -> dict|None
    """
    Verifica Data_Entry_2017.csv: columnas requeridas y prevalencias por patología.
    Retorna dict de estadísticas o None si falta pandas.
    """
    csv_path = nih_dir / "Data_Entry_2017.csv"
    if not csv_path.exists():
        log.warning("[NIH] Data_Entry_2017.csv no encontrado.")
        return None

    try:
        import pandas as pd
    except ImportError:
        log.warning("[NIH] pandas no disponible — saltando auditoría CSV.")
        return None

    df = pd.read_csv(csv_path)
    required_cols = ["Image Index", "Finding Labels", "Patient ID",
                     "View Position", "Follow-up #"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error("[NIH] Columnas faltantes en CSV: %s", missing)
        return {"ok": False, "missing_cols": missing}

    # Prevalencia por patología
    all_labels = df["Finding Labels"].str.split("|").explode().str.strip()
    prevalence = all_labels.value_counts().to_dict()
    log.info("[NIH] CSV OK — %d imágenes, %d pacientes, %d patologías",
             len(df), df["Patient ID"].nunique(), len(prevalence))
    for label, count in sorted(prevalence.items(), key=lambda x: -x[1])[:5]:
        log.info("[NIH]   %s: %d (%.1f%%)", label, count, 100 * count / len(df))

    return {
        "ok": True,
        "total_images": len(df),
        "n_patients": int(df["Patient ID"].nunique()),
        "n_labels": len(prevalence),
        "prevalence": prevalence,
    }


def run_pre_chestxray14(datasets_dir):
    # type: (Path) -> dict
    """
    Ejecuta toda la preparación post-extracción de NIH ChestXray14.
    Retorna dict de estado.
    """
    nih_dir = datasets_dir / "nih_chest_xrays"
    if not nih_dir.exists():
        log.warning("[NIH] Directorio %s no existe — saltando.", nih_dir)
        return {"status": "⚠️", "reason": "directorio no existe"}

    result = {}

    # 1. Symlinks
    n_symlinks = crear_symlinks_all_images(nih_dir)
    result["symlinks"] = n_symlinks

    # 2. Verificar splits oficiales
    split_info = verificar_split_txts(nih_dir)
    result["splits"] = split_info

    # 3. Auditar CSV
    csv_info = auditar_csv(nih_dir)
    result["csv_audit"] = csv_info

    result["status"] = "✅" if n_symlinks > 0 and split_info else "⚠️"
    return result
