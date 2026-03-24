#!/usr/bin/env python3
"""
pre_modelo.py — Splits 80/10/10 para todos los datasets
=========================================================
Fase 0 — Preparación de Datos | Proyecto MoE Médico

Responsabilidad única: generar los splits definitivos train/val/test (80/10/10)
para los 5 datasets de dominio y el sexto experto (CAE).

La unidad de splitting es siempre paciente/sujeto, no imagen individual.
Semilla fija global (SEED=42). Resultado persiste en disco como fuente de verdad.

Origen: fix2_isic_splits, fix1_nih_val_split de fix_alignment.py,
        post_oa de setup_datasets.py, lógica de split de extract_luna_patches.py,
        PancreasDataset.build_kfold_splits
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

log = logging.getLogger("fase0.pre_modelo")

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#  NIH ChestXray14 — Split 80/10/10 por patient_id
# ══════════════════════════════════════════════════════════════════════════════

def split_nih(datasets_dir):
    # type: (Path) -> dict
    """
    Genera splits 80/10/10 por patient_id con estratificación por etiqueta más rara.
    Si el test oficial > 12% del total, reduce al 10% y mueve sobrantes al pool train_val.
    """
    nih = datasets_dir / "nih_chest_xrays"
    out = nih / "splits"
    out.mkdir(parents=True, exist_ok=True)

    # Idempotencia: verificar si ya existen
    expected = ["nih_train_list.txt", "nih_val_list.txt", "nih_test_list.txt"]
    if all((out / f).exists() and (out / f).stat().st_size > 100 for f in expected):
        log.info("[NIH] Splits ya existen en %s, saltando.", out)
        counts = {}
        for f in expected:
            counts[f] = sum(1 for _ in open(out / f))
        return {"status": "✅", "skipped": True, **counts}

    df = pd.read_csv(nih / "Data_Entry_2017.csv")
    tv_imgs = [l.strip() for l in
               (nih / "train_val_list.txt").read_text().split("\n") if l.strip()]
    te_imgs = [l.strip() for l in
               (nih / "test_list.txt").read_text().split("\n") if l.strip()]

    total = len(tv_imgs) + len(te_imgs)
    log.info("[NIH] train_val: %d imgs | test oficial: %d imgs | total: %d",
             len(tv_imgs), len(te_imgs), total)

    img2pid = dict(zip(df["Image Index"], df["Patient ID"]))
    img2labels = dict(zip(df["Image Index"], df["Finding Labels"]))

    # Verificar si el test oficial excede 12% del total
    test_pct = len(te_imgs) / total
    if test_pct > 0.12:
        log.info("[NIH] Test oficial = %.1f%% > 12%% — reduciendo a 10%%.", test_pct * 100)
        # Agrupar test por patient_id
        test_pid_imgs = defaultdict(list)
        for img in te_imgs:
            pid = img2pid.get(img)
            if pid is not None:
                test_pid_imgs[pid].append(img)

        test_pids = list(test_pid_imgs.keys())
        n_test_target = int(0.10 * total)

        # Seleccionar pacientes de test hasta llenar ~10%
        rng = np.random.default_rng(SEED)
        rng.shuffle(test_pids)
        keep_test_pids = set()
        test_img_count = 0
        for pid in test_pids:
            if test_img_count + len(test_pid_imgs[pid]) <= n_test_target * 1.05:
                keep_test_pids.add(pid)
                test_img_count += len(test_pid_imgs[pid])

        # Sobrantes -> pool train_val
        overflow_imgs = []
        for pid in test_pids:
            if pid not in keep_test_pids:
                overflow_imgs.extend(test_pid_imgs[pid])

        te_imgs = [img for img in te_imgs if img2pid.get(img) in keep_test_pids]
        tv_imgs = tv_imgs + overflow_imgs
        log.info("[NIH] Test reducido: %d imgs | Pool train_val: %d imgs",
                 len(te_imgs), len(tv_imgs))

    # Agrupar train_val por patient_id
    pid_imgs = defaultdict(list)
    for img in tv_imgs:
        pid = img2pid.get(img)
        if pid is not None:
            pid_imgs[pid].append(img)

    # Etiquetas por paciente
    pid_labels = defaultdict(set)
    for pid, imgs in pid_imgs.items():
        for img in imgs:
            for lbl in img2labels.get(img, "No Finding").split("|"):
                pid_labels[pid].add(lbl.strip())

    # Frecuencia de etiquetas
    lbl_freq = defaultdict(int)
    for labels in pid_labels.values():
        for lbl in labels:
            lbl_freq[lbl] += 1

    # Estratificación por etiqueta más rara
    pids = list(pid_imgs.keys())
    strat = [min(pid_labels[p], key=lambda l: lbl_freq[l]) for p in pids]

    # Separar val del pool train_val
    n_val_target = int(0.10 * total)
    val_fraction = n_val_target / sum(len(pid_imgs[p]) for p in pids)
    val_fraction = min(max(val_fraction, 0.05), 0.25)

    try:
        train_pids, val_pids = train_test_split(
            pids, test_size=val_fraction, stratify=strat, random_state=SEED
        )
    except ValueError:
        log.warning("[NIH] Stratified split falló — usando split aleatorio")
        train_pids, val_pids = train_test_split(
            pids, test_size=val_fraction, random_state=SEED
        )

    train_list = sorted(img for p in train_pids for img in pid_imgs[p])
    val_list = sorted(img for p in val_pids for img in pid_imgs[p])

    # Verificar no-solapamiento de patient_id
    tp, vp = set(train_pids), set(val_pids)
    tep = {img2pid[i] for i in te_imgs if i in img2pid}
    assert not (tp & vp), "Overlap pacientes train/val!"
    assert not (tp & tep), "Overlap pacientes train/test!"
    assert not (vp & tep), "Overlap pacientes val/test!"

    (out / "nih_train_list.txt").write_text("\n".join(train_list) + "\n")
    (out / "nih_val_list.txt").write_text("\n".join(val_list) + "\n")
    (out / "nih_test_list.txt").write_text("\n".join(te_imgs) + "\n")

    r = {
        "status": "✅",
        "train": len(train_list),
        "val": len(val_list),
        "test": len(te_imgs),
        "total": len(train_list) + len(val_list) + len(te_imgs),
    }
    log.info("[NIH] Splits: train=%d val=%d test=%d (overlap=0)", r["train"], r["val"], r["test"])
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  ISIC 2019 — Split 80/10/10 por lesion_id
# ══════════════════════════════════════════════════════════════════════════════

def split_isic(datasets_dir):
    # type: (Path) -> dict
    """Genera splits 80/10/10 por lesion_id con estratificación por clase."""
    CLASS_TO_IDX = {
        "MEL": 0, "NV": 1, "BCC": 2, "AK": 3,
        "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7,
    }

    isic = datasets_dir / "isic_2019"
    out = isic / "splits"
    out.mkdir(parents=True, exist_ok=True)

    expected = ["isic_train.csv", "isic_val.csv", "isic_test.csv"]
    if all((out / f).exists() and (out / f).stat().st_size > 100 for f in expected):
        log.info("[ISIC] Splits ya existen en %s, saltando.", out)
        return {"status": "✅", "skipped": True}

    gt = pd.read_csv(isic / "ISIC_2019_Training_GroundTruth.csv")
    md = pd.read_csv(isic / "ISIC_2019_Training_Metadata.csv")

    # Excluir UNK
    gt = gt[gt["UNK"] < 0.5].copy()

    class_cols = list(CLASS_TO_IDX.keys())
    gt["label_name"] = gt[class_cols].idxmax(axis=1)
    gt["label_idx"] = gt["label_name"].map(CLASS_TO_IDX)

    merged = gt[["image", "label_idx", "label_name"]].merge(
        md[["image", "lesion_id"]], on="image", how="left"
    )

    # Asignar lesion_id sintético para nulos
    null_mask = merged["lesion_id"].isna()
    merged.loc[null_mask, "lesion_id"] = merged.loc[null_mask, "image"].apply(
        lambda x: "SOLO_{}".format(x)
    )

    log.info("[ISIC] Total imágenes: %d | Lesiones únicas: %d",
             len(merged), merged["lesion_id"].nunique())

    # Clase por lesión (moda)
    lesion_class = (
        merged.groupby("lesion_id")["label_idx"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .reset_index()
    )
    lesion_class.columns = ["lesion_id", "cls"]

    les_ids = lesion_class["lesion_id"].to_numpy()
    les_cls = lesion_class["cls"].to_numpy()

    # 80/10/10
    try:
        train_les, temp_les, _, temp_cls = train_test_split(
            les_ids, les_cls, test_size=0.20,
            stratify=les_cls, random_state=SEED,
        )
        val_les, test_les = train_test_split(
            temp_les, test_size=0.50,
            stratify=temp_cls, random_state=SEED,
        )
    except ValueError:
        log.warning("[ISIC] Stratified split falló — usando split aleatorio")
        train_les, temp_les = train_test_split(
            les_ids, test_size=0.20, random_state=SEED,
        )
        val_les, test_les = train_test_split(
            temp_les, test_size=0.50, random_state=SEED,
        )

    sets = {
        "train": set(train_les),
        "val": set(val_les),
        "test": set(test_les),
    }

    # Verificar no-overlap
    assert not (sets["train"] & sets["val"]), "Overlap lesion train/val!"
    assert not (sets["train"] & sets["test"]), "Overlap lesion train/test!"
    assert not (sets["val"] & sets["test"]), "Overlap lesion val/test!"

    cols = ["image", "label_idx", "label_name", "lesion_id"]
    counts = {}
    for name, ids in sets.items():
        sub = merged[merged["lesion_id"].isin(ids)][cols]
        sub.to_csv(out / "isic_{}.csv".format(name), index=False)
        counts[name] = len(sub)

    total = sum(counts.values())
    active_classes = sorted(merged["label_name"].unique().tolist())

    # Verificar 8 clases en cada split
    for name, ids in sets.items():
        sub = merged[merged["lesion_id"].isin(ids)]
        n_classes = sub["label_name"].nunique()
        if n_classes < len(CLASS_TO_IDX):
            log.warning("[ISIC] Split %s solo tiene %d de %d clases.",
                        name, n_classes, len(CLASS_TO_IDX))

    # Verificar proporciones ±2%
    for name, count in counts.items():
        target = 0.80 if name == "train" else 0.10
        actual = count / total
        if abs(actual - target) > 0.02:
            log.warning("[ISIC] Split %s: %.1f%% (objetivo: %.0f%%)",
                        name, actual * 100, target * 100)

    r = {"status": "✅", **counts, "total": total, "classes": active_classes}
    log.info("[ISIC] Splits: train=%d val=%d test=%d (overlap=0)",
             counts["train"], counts["val"], counts["test"])
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  Osteoarthritis — Split 80/10/10 por clase consolidada
# ══════════════════════════════════════════════════════════════════════════════

def split_oa(datasets_dir):
    # type: (Path) -> dict
    """
    Genera splits 80/10/10 estratificados por clase consolidada (KL0→0, KL1+KL2→1, KL3+KL4→2).
    Copias físicas (no symlinks) porque las imágenes OA son pequeñas.

    DEUDA TÉCNICA CONOCIDA: el dataset OA no tiene metadatos de patient_id.
    El split se hace por imagen individual, lo que puede causar leakage implícito
    si hay imágenes de la misma rodilla en distintos splits. El plan original
    especificaba detección de imágenes similares (hash perceptual o distancia L2
    de píxeles) para agrupar imágenes del mismo paciente antes de hacer el split.
    Esto quedó pendiente y debe implementarse antes del entrenamiento final (Fase 3).
    """
    oa_dir = datasets_dir / "osteoarthritis"
    splits_dir = oa_dir / "oa_splits"

    # Idempotencia
    if splits_dir.is_dir() and any((splits_dir / "train").rglob("*.*")):
        log.info("[OA] oa_splits/ ya existe con imágenes, saltando.")
        return {"status": "✅", "skipped": True}

    # Buscar KLGrade
    src = None
    for candidate in [oa_dir / "KLGrade" / "KLGrade", oa_dir / "KLGrade"]:
        if candidate.is_dir() and any(candidate.iterdir()):
            src = candidate
            break
    if src is None:
        kl_dirs = list(oa_dir.rglob("KLGrade"))
        for d in kl_dirs:
            if d.is_dir() and any(d.iterdir()):
                src = d
                break
    if src is None:
        log.warning("[OA] No se encontró KLGrade/ con imágenes.")
        return {"status": "warning", "reason": "KLGrade no encontrado"}

    log.info("[OA] Fuente: %s", src)

    # Consolidar: KL0→0, KL1+KL2→1, KL3+KL4→2
    mapping = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
    all_files = {0: [], 1: [], 2: []}

    for kl_str, cls in mapping.items():
        kl_dir = src / kl_str
        if not kl_dir.exists():
            continue
        files = list(kl_dir.glob("*.jpg")) + list(kl_dir.glob("*.png"))
        all_files[cls].extend(files)

    total = sum(len(v) for v in all_files.values())
    if total == 0:
        log.warning("[OA] No se encontraron imágenes en KLGrade/.")
        return {"status": "warning", "reason": "sin imágenes"}

    # Limpieza previa
    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    random.seed(SEED)
    counts = {"train": 0, "val": 0, "test": 0}

    for cls, files in all_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(0.80 * n)
        n_val = int(0.10 * n)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split_name, imgs in splits.items():
            d = splits_dir / split_name / str(cls)
            d.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, d / img.name)
            counts[split_name] += len(imgs)

    r = {"status": "✅", **counts, "total": total}
    log.info("[OA] Splits: train=%d val=%d test=%d (total=%d)",
             counts["train"], counts["val"], counts["test"], total)
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  LUNA16 — Split 80/10/10 por seriesuid
# ══════════════════════════════════════════════════════════════════════════════

def split_luna(datasets_dir):
    # type: (Path) -> dict
    """Genera splits 80/10/10 por seriesuid y los persiste en luna_splits.json."""
    luna_dir = datasets_dir / "luna_lung_cancer"
    ct_dir = luna_dir / "ct_volumes"
    splits_file = luna_dir / "luna_splits.json"

    if splits_file.exists() and splits_file.stat().st_size > 50:
        log.info("[LUNA] luna_splits.json ya existe, saltando.")
        return {"status": "✅", "skipped": True}

    # Descubrir todos los seriesuids disponibles
    mhd_files = list(ct_dir.rglob("*.mhd"))
    if not mhd_files:
        log.warning("[LUNA] No se encontraron .mhd en %s — split pendiente.", ct_dir)
        return {"status": "warning", "reason": "sin CTs"}

    all_uids = sorted(set(p.stem for p in mhd_files))
    log.info("[LUNA] %d seriesuids encontrados en %d archivos .mhd",
             len(all_uids), len(mhd_files))

    rng = np.random.default_rng(SEED)
    uids_arr = np.array(all_uids)
    rng.shuffle(uids_arr)

    n_total = len(uids_arr)
    n_test = int(0.10 * n_total)
    n_val = int(0.10 * n_total)

    test_uids = uids_arr[:n_test].tolist()
    val_uids = uids_arr[n_test:n_test + n_val].tolist()
    train_uids = uids_arr[n_test + n_val:].tolist()

    # Verificar no-overlap
    assert not (set(train_uids) & set(val_uids)), "Overlap LUNA train/val!"
    assert not (set(train_uids) & set(test_uids)), "Overlap LUNA train/test!"
    assert not (set(val_uids) & set(test_uids)), "Overlap LUNA val/test!"

    splits_data = {
        "train_uids": train_uids,
        "val_uids": val_uids,
        "test_uids": test_uids,
    }
    with open(splits_file, "w") as f:
        json.dump(splits_data, f, indent=2)

    r = {
        "status": "✅",
        "train": len(train_uids),
        "val": len(val_uids),
        "test": len(test_uids),
        "total": n_total,
    }
    log.info("[LUNA] Splits: train=%d val=%d test=%d (total=%d)",
             r["train"], r["val"], r["test"], r["total"])
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  Pancreas PANORAMA — 10% test fijo + k-fold (k=5) sobre 90%
# ══════════════════════════════════════════════════════════════════════════════

def split_pancreas(datasets_dir):
    # type: (Path) -> dict
    """
    10% test fijo estratificado + k-fold CV (k=5) sobre el 90% restante.
    Requiere pancreas_labels_binary.csv del Paso 4.
    """
    csv_path = datasets_dir / "pancreas_labels_binary.csv"
    out_csv = datasets_dir / "pancreas_splits.csv"
    test_ids_file = datasets_dir / "pancreas_test_ids.txt"

    if out_csv.exists() and out_csv.stat().st_size > 100:
        log.info("[PANCREAS] pancreas_splits.csv ya existe, saltando.")
        return {"status": "✅", "skipped": True}

    if not csv_path.exists():
        log.warning("[PANCREAS] pancreas_labels_binary.csv no existe — split pendiente.")
        return {"status": "warning", "reason": "sin etiquetas binarias"}

    df = pd.read_csv(csv_path)
    # Filtrar solo los que tienen etiqueta válida
    df_valid = df[df["label"] >= 0].copy()
    log.info("[PANCREAS] %d volúmenes con etiqueta válida (de %d total)",
             len(df_valid), len(df))

    case_ids = df_valid["case_id"].values
    labels = df_valid["label"].values

    # 10% test fijo
    try:
        remain_ids, test_ids, remain_labels, _ = train_test_split(
            case_ids, labels, test_size=0.10,
            stratify=labels, random_state=SEED,
        )
    except ValueError:
        remain_ids, test_ids, remain_labels, _ = train_test_split(
            case_ids, labels, test_size=0.10, random_state=SEED,
        )

    # k-fold (k=5) sobre el 90% restante
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_assignments = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(remain_ids, remain_labels), 1):
        for i in train_idx:
            cid = remain_ids[i]
            if cid not in fold_assignments:
                fold_assignments[cid] = []
            fold_assignments[cid].append("fold{}_train".format(fold_idx))
        for i in val_idx:
            cid = remain_ids[i]
            if cid not in fold_assignments:
                fold_assignments[cid] = []
            fold_assignments[cid].append("fold{}_val".format(fold_idx))

    # Construir CSV de salida
    rows = []
    for cid in test_ids:
        lbl = int(df_valid[df_valid["case_id"] == cid]["label"].iloc[0])
        rows.append({"case_id": cid, "label": lbl, "split": "test"})

    for cid in remain_ids:
        lbl = int(df_valid[df_valid["case_id"] == cid]["label"].iloc[0])
        for split_tag in fold_assignments.get(cid, []):
            rows.append({"case_id": cid, "label": lbl, "split": split_tag})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    # Archivo de test IDs para referencia rápida
    test_ids_file.write_text("\n".join(str(x) for x in test_ids) + "\n")

    n_test = len(test_ids)
    n_remain = len(remain_ids)
    log.info("[PANCREAS] Splits: test=%d (fijo) | train/val=5-fold sobre %d",
             n_test, n_remain)
    return {
        "status": "✅",
        "test": n_test,
        "train_val_pool": n_remain,
        "total": n_test + n_remain,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CAE — Agregado de todos los datasets
# ══════════════════════════════════════════════════════════════════════════════

def build_cae_splits(datasets_dir):
    # type: (Path) -> dict
    """
    Construye cae_splits.csv agregando referencias de todos los datasets
    con sus splits ya definidos.
    """
    cae_csv = datasets_dir / "cae_splits.csv"
    if cae_csv.exists() and cae_csv.stat().st_size > 100:
        log.info("[CAE] cae_splits.csv ya existe, saltando.")
        return {"status": "✅", "skipped": True}

    rows = []
    nih_dir = datasets_dir / "nih_chest_xrays"
    isic_dir = datasets_dir / "isic_2019"
    oa_dir = datasets_dir / "osteoarthritis"
    luna_dir = datasets_dir / "luna_lung_cancer"

    # NIH
    for split_name, fname in [("train", "nih_train_list.txt"),
                               ("val", "nih_val_list.txt"),
                               ("test", "nih_test_list.txt")]:
        fpath = nih_dir / "splits" / fname
        if fpath.exists():
            for line in fpath.read_text().strip().split("\n"):
                img = line.strip()
                if img:
                    rows.append({
                        "ruta_imagen": str(nih_dir / "all_images" / img),
                        "dataset_origen": "nih",
                        "split": split_name,
                        "expert_id": 0,
                        "tipo_dato": "2d_image",
                    })

    # ISIC
    for split_name in ["train", "val", "test"]:
        fpath = isic_dir / "splits" / "isic_{}.csv".format(split_name)
        if fpath.exists():
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                img_name = row["image"]
                rows.append({
                    "ruta_imagen": str(isic_dir / "isic_images" / (img_name + ".jpg")),
                    "dataset_origen": "isic",
                    "split": split_name,
                    "expert_id": 1,
                    "tipo_dato": "2d_image",
                })

    # OA
    splits_path = oa_dir / "oa_splits"
    if splits_path.exists():
        for split_name in ["train", "val", "test"]:
            split_dir = splits_path / split_name
            if split_dir.exists():
                for img in sorted(split_dir.rglob("*.*")):
                    if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                        rows.append({
                            "ruta_imagen": str(img),
                            "dataset_origen": "oa",
                            "split": split_name,
                            "expert_id": 2,
                            "tipo_dato": "2d_image",
                        })

    # LUNA (parches .npy)
    patches_dir = luna_dir / "patches"
    for split_name in ["train", "val", "test"]:
        split_dir = patches_dir / split_name
        if split_dir.exists():
            for npy in sorted(split_dir.glob("candidate_*.npy")):
                rows.append({
                    "ruta_imagen": str(npy),
                    "dataset_origen": "luna",
                    "split": split_name,
                    "expert_id": 3,
                    "tipo_dato": "3d_patch_npy",
                })

    # Pancreas (volúmenes .nii.gz con preprocesado on-the-fly)
    pancreas_splits = datasets_dir / "pancreas_splits.csv"
    if pancreas_splits.exists():
        pdf = pd.read_csv(pancreas_splits)
        zenodo_dir = datasets_dir / "zenodo_13715870"
        for _, prow in pdf.iterrows():
            split_tag = prow["split"]
            # Normalizar: test, fold*_train -> train, fold*_val -> val
            if split_tag == "test":
                norm_split = "test"
            elif "train" in split_tag:
                norm_split = "train"
            else:
                norm_split = "val"

            # Buscar el archivo .nii.gz correspondiente
            nii_candidates = list(zenodo_dir.glob("{}*.nii.gz".format(prow["case_id"])))
            for nii in nii_candidates:
                rows.append({
                    "ruta_imagen": str(nii),
                    "dataset_origen": "pancreas",
                    "split": norm_split,
                    "expert_id": 4,
                    "tipo_dato": "3d_volume_nifti",
                })

    if not rows:
        log.warning("[CAE] No se encontraron datos para construir cae_splits.csv.")
        return {"status": "warning", "reason": "sin datos"}

    cae_df = pd.DataFrame(rows)
    cae_df.to_csv(cae_csv, index=False)

    by_split = cae_df.groupby("split").size().to_dict()
    by_dataset = cae_df.groupby("dataset_origen").size().to_dict()
    log.info("[CAE] cae_splits.csv: %d filas | Splits: %s | Datasets: %s",
             len(cae_df), by_split, by_dataset)
    return {"status": "✅", "total": len(cae_df), "by_split": by_split, "by_dataset": by_dataset}


# ══════════════════════════════════════════════════════════════════════════════
#  Orquestador de splits
# ══════════════════════════════════════════════════════════════════════════════

def run_splits(datasets_dir, active):
    # type: (Path, set) -> dict
    """Ejecuta la generación de splits para todos los datasets activos."""
    results = {}

    if "nih" in active:
        try:
            results["nih"] = split_nih(datasets_dir)
        except Exception as e:
            log.error("[NIH] Error en split: %s", e)
            results["nih"] = {"status": "error", "error": str(e)}

    if "isic" in active:
        try:
            results["isic"] = split_isic(datasets_dir)
        except Exception as e:
            log.error("[ISIC] Error en split: %s", e)
            results["isic"] = {"status": "error", "error": str(e)}

    if "oa" in active:
        try:
            results["oa"] = split_oa(datasets_dir)
        except Exception as e:
            log.error("[OA] Error en split: %s", e)
            results["oa"] = {"status": "error", "error": str(e)}

    if "luna_ct" in active or "luna" in active:
        try:
            results["luna"] = split_luna(datasets_dir)
        except Exception as e:
            log.error("[LUNA] Error en split: %s", e)
            results["luna"] = {"status": "error", "error": str(e)}

    if "pancreas" in active:
        try:
            results["pancreas"] = split_pancreas(datasets_dir)
        except Exception as e:
            log.error("[PANCREAS] Error en split: %s", e)
            results["pancreas"] = {"status": "error", "error": str(e)}

    # CAE — después de todos los splits individuales
    try:
        results["cae"] = build_cae_splits(datasets_dir)
    except Exception as e:
        log.error("[CAE] Error construyendo cae_splits.csv: %s", e)
        results["cae"] = {"status": "error", "error": str(e)}

    return results
