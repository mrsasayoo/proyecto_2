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
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    StratifiedKFold,
    GroupKFold,
)
from PIL import Image

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
    tv_imgs = [
        l.strip()
        for l in (nih / "train_val_list.txt").read_text().split("\n")
        if l.strip()
    ]
    te_imgs = [
        l.strip() for l in (nih / "test_list.txt").read_text().split("\n") if l.strip()
    ]

    total = len(tv_imgs) + len(te_imgs)
    log.info(
        "[NIH] train_val: %d imgs | test oficial: %d imgs | total: %d",
        len(tv_imgs),
        len(te_imgs),
        total,
    )

    img2pid = dict(zip(df["Image Index"], df["Patient ID"]))
    img2labels = dict(zip(df["Image Index"], df["Finding Labels"]))

    # Verificar si el test oficial excede 12% del total.
    # Umbral elegido: el test oficial NIH contiene ~12,000 imágenes de ~112,000 total
    # (~10.7%). Si en alguna ejecución el test > 12%, se reduce dinámicamente
    # moviendo pacientes sobrantes al pool train_val.
    # Referencia de splits esperados (ejecución canónica):
    #   train=88,999 | val=11,349 | test=11,772 (total=112,120)
    # La tolerancia 1.05× en la selección de pacientes de test evita cortes en
    # la mitad de un grupo de imágenes del mismo paciente.
    test_pct = len(te_imgs) / total
    if test_pct > 0.12:
        log.info(
            "[NIH] Test oficial = %.1f%% > 12%% — reduciendo a 10%%.", test_pct * 100
        )
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
        log.info(
            "[NIH] Test reducido: %d imgs | Pool train_val: %d imgs",
            len(te_imgs),
            len(tv_imgs),
        )

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
    log.info(
        "[NIH] Splits: train=%d val=%d test=%d (overlap=0)",
        r["train"],
        r["val"],
        r["test"],
    )
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  ISIC 2019 — Split 80/10/10 por lesion_id
# ══════════════════════════════════════════════════════════════════════════════


def build_lesion_split(datasets_dir):
    # type: (Path) -> dict
    """Genera splits 80/10/10 por lesion_id con estratificación por clase."""
    CLASS_TO_IDX = {
        "MEL": 0,
        "NV": 1,
        "BCC": 2,
        "AK": 3,
        "BKL": 4,
        "DF": 5,
        "VASC": 6,
        "SCC": 7,
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

    log.info(
        "[ISIC] Total imágenes: %d | Lesiones únicas: %d",
        len(merged),
        merged["lesion_id"].nunique(),
    )

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
            les_ids,
            les_cls,
            test_size=0.20,
            stratify=les_cls,
            random_state=SEED,
        )
        val_les, test_les = train_test_split(
            temp_les,
            test_size=0.50,
            stratify=temp_cls,
            random_state=SEED,
        )
    except ValueError:
        log.warning("[ISIC] Stratified split falló — usando split aleatorio")
        train_les, temp_les = train_test_split(
            les_ids,
            test_size=0.20,
            random_state=SEED,
        )
        val_les, test_les = train_test_split(
            temp_les,
            test_size=0.50,
            random_state=SEED,
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
            log.warning(
                "[ISIC] Split %s solo tiene %d de %d clases.",
                name,
                n_classes,
                len(CLASS_TO_IDX),
            )

    # Verificar proporciones ±2%
    for name, count in counts.items():
        target = 0.80 if name == "train" else 0.10
        actual = count / total
        if abs(actual - target) > 0.02:
            log.warning(
                "[ISIC] Split %s: %.1f%% (objetivo: %.0f%%)",
                name,
                actual * 100,
                target * 100,
            )

    r = {"status": "✅", **counts, "total": total, "classes": active_classes}
    log.info(
        "[ISIC] Splits: train=%d val=%d test=%d (overlap=0)",
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  Osteoarthritis — Helpers de similitud de imagen
# ══════════════════════════════════════════════════════════════════════════════


def _compute_fingerprint_oa(img_path, size=16):
    # type: (Path, int) -> np.ndarray|None
    """
    Genera un fingerprint compacto de una imagen OA redimensionándola a size×size
    en escala de grises y normalizando el vector resultante a norma L2 = 1.

    Tamaño 16×16 = 256 valores por imagen.
    La normalización L2 hace que la distancia euclidiana entre fingerprints
    equivalga a una medida de disimilitud entre 0 (idénticas) y 2 (opuestas).

    Imágenes de la misma rodilla (mismo paciente, misma visita o visitas
    consecutivas) producen fingerprints con distancia < 0.12 en la práctica
    para el dataset OAI. Imágenes de pacientes distintos producen distancias
    típicamente > 0.25.
    """
    try:
        img = Image.open(img_path).convert("L").resize((size, size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32).flatten()
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            return None
        return arr / norm
    except Exception:
        return None


def _union_find_groups(n):
    # type: (int) -> tuple
    """
    Retorna (parent, find, union) para Union-Find con path compression.
    Usado para agrupar índices de imágenes similares en el mismo grupo.
    """
    parent = list(range(n))

    def find(x):
        # type: (int) -> int
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        # type: (int, int) -> None
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    return parent, find, union


def _group_by_similarity(files, threshold=0.12, fingerprint_size=16):
    # type: (list, float, int) -> list
    """
    Agrupa imágenes por similitud visual para inferir pseudo-pacientes.

    Algoritmo:
      1. Calcula fingerprint de 16×16 L2-normalizado para cada imagen.
      2. Calcula todas las distancias par-a-par con operación matricial
         (vectorizado, sin bucle Python interno).
         d(i,j) = ||f_i - f_j||_2 = sqrt(2 - 2 * f_i · f_j)
         dado que ||f_i|| = ||f_j|| = 1.
      3. Aplica Union-Find: une i y j si d(i,j) < threshold.
      4. Devuelve lista de grupos (cada grupo = lista de rutas de imagen).

    Complejidad: O(n²) en tiempo, O(n²) en memoria para la matriz de distancias.
    Para n ≈ 4000 imágenes por clase: matriz de 4000×4000 float32 ≈ 64 MB.
    Aceptable para hardware con ≥ 4 GB de RAM.

    Parámetros:
      threshold: distancia L2 máxima para considerar dos imágenes del mismo
                 paciente. 0.12 calibrado empíricamente para el dataset OAI
                 de rodilla (imágenes 800×600 px en escala de grises).
                 Reducir para menos agrupaciones (más conservador).
                 Aumentar para más agrupaciones (más agresivo).

    Notas de robustez:
      - Imágenes con error de lectura se asignan a su propio grupo individual.
      - Si todas las imágenes fallan al leer, retorna un grupo por imagen
        (mismo comportamiento que el split naive sin agrupación).
    """
    n = len(files)
    if n == 0:
        return []

    # 1. Calcular fingerprints
    fingerprints = {}
    failed_indices = []
    for i, fp in enumerate(files):
        fprint = _compute_fingerprint_oa(fp, size=fingerprint_size)
        if fprint is not None:
            fingerprints[i] = fprint
        else:
            failed_indices.append(i)

    n_valid = len(fingerprints)
    if n_valid == 0:
        # Fallback: un grupo por imagen
        log.warning("[OA/similitud] Todos los fingerprints fallaron — sin agrupación.")
        return [[f] for f in files]

    # 2. Calcular todas las distancias par-a-par con numpy vectorizado
    valid_indices = sorted(fingerprints.keys())
    fps_matrix = np.array([fingerprints[i] for i in valid_indices], dtype=np.float32)

    # ||f_i - f_j||² = 2 - 2*(f_i · f_j)  para vectores L2-normalizados
    # Más eficiente que restar matrices directamente para n grande
    dot_products = fps_matrix @ fps_matrix.T  # [n_valid, n_valid]
    dot_products = np.clip(dot_products, -1.0, 1.0)  # por errores numéricos
    sq_dists = np.maximum(0.0, 2.0 - 2.0 * dot_products)
    distances = np.sqrt(sq_dists)  # [n_valid, n_valid]

    # 3. Union-Find
    parent, find, union = _union_find_groups(n)

    pairs_found = 0
    for ii in range(n_valid):
        for jj in range(ii + 1, n_valid):
            if distances[ii, jj] < threshold:
                union(valid_indices[ii], valid_indices[jj])
                pairs_found += 1

    # 4. Construir grupos
    groups_dict = defaultdict(list)
    for i in range(n):
        root = find(i)
        groups_dict[root].append(files[i])

    groups = list(groups_dict.values())

    # Estadísticas de agrupación
    n_groups = len(groups)
    sizes = [len(g) for g in groups]
    n_solo = sum(1 for s in sizes if s == 1)
    n_multi = n_groups - n_solo
    avg_multi = sum(s for s in sizes if s > 1) / max(n_multi, 1)

    log.info(
        "[OA/similitud] %d imágenes → %d grupos pseudo-paciente "
        "(umbral=%.2f) | solos=%d | multi=%d (avg %.1f imgs/grupo) | "
        "pares agrupados=%d | fallidos=%d",
        n,
        n_groups,
        threshold,
        n_solo,
        n_multi,
        avg_multi,
        pairs_found,
        len(failed_indices),
    )

    return groups


# ══════════════════════════════════════════════════════════════════════════════
#  Osteoarthritis — Split 80/10/10 por grupo de similitud
# ══════════════════════════════════════════════════════════════════════════════


def split_oa(datasets_dir, similarity_threshold=0.12, fingerprint_size=16):
    # type: (Path, float, int) -> dict
    """
    Genera splits 80/10/10 por grado KL directo (5 clases: 0–4, sin remapeo).

    Clases: 0=Normal, 1=Dudoso, 2=Leve, 3=Moderado, 4=Severo.

    Dado que el dataset OAI no incluye patient_id, la unidad de splitting se
    infiere mediante similitud de imagen (fingerprint 16×16 L2-normalizado +
    distancia euclidiana + Union-Find). Imágenes con distancia < threshold se
    consideran del mismo paciente/rodilla y van al mismo split.

    Pasos:
      1. Recopilar imágenes por grado KL directo (carpetas 0–4).
      2. Por cada clase, agrupar imágenes por similitud (_group_by_similarity).
      3. Aplicar train_test_split estratificado a nivel de GRUPO (no imagen).
         La clase de un grupo = clase de su primera imagen (todas son iguales
         dentro de la misma carpeta KL).
      4. Copiar imágenes al directorio de split correspondiente.
      5. Reportar estadísticas de agrupación y tamaños de split.

    Parámetros:
      similarity_threshold: distancia L2 máxima para inferir mismo paciente.
                            0.12 recomendado para OAI estándar.
      fingerprint_size:     lado del thumbnail cuadrado para el fingerprint.
                            16 → 256 valores por imagen.

    Copias físicas (no symlinks) porque las imágenes OA son pequeñas (~50 KB).

    LIMITACIÓN CONOCIDA: la detección de similitud es una heurística. Puede
    haber falsos positivos (pacientes distintos con rodillas muy similares) y
    falsos negativos (mismo paciente con posicionamiento muy distinto). Es una
    mejora significativa respecto al split naive por imagen, pero no equivale
    a un split por patient_id real. Documentar esta limitación en el reporte.
    """
    oa_dir = datasets_dir / "osteoarthritis"
    splits_dir = oa_dir / "oa_splits"

    # ── Idempotencia ──────────────────────────────────────────────────────────
    # ADVERTENCIA: si oa_splits/ contiene splits de una versión anterior
    # (3 clases consolidadas con carpetas {0,1,2}), hay que borrar el
    # directorio manualmente antes de re-ejecutar para regenerar con 5 clases
    # KL directas (carpetas {0,1,2,3,4}).
    if splits_dir.is_dir() and any((splits_dir / "train").rglob("*.*")):
        log.info("[OA] oa_splits/ ya existe con imágenes, saltando.")
        return {"status": "✅", "skipped": True}

    # ── Localizar KLGrade ─────────────────────────────────────────────────────
    src = None
    for candidate in [oa_dir / "KLGrade" / "KLGrade", oa_dir / "KLGrade"]:
        if candidate.is_dir() and any(candidate.iterdir()):
            src = candidate
            break
    if src is None:
        for d in oa_dir.rglob("KLGrade"):
            if d.is_dir() and any(d.iterdir()):
                src = d
                break
    if src is None:
        log.warning("[OA] No se encontró KLGrade/ con imágenes.")
        return {"status": "⚠️", "reason": "KLGrade no encontrado"}

    log.info("[OA] Fuente: %s", src)

    # ── Recopilar imágenes por grado KL directo (5 clases, sin remapeo) ──────
    kl_grades = ["0", "1", "2", "3", "4"]
    all_files = {i: [] for i in range(5)}

    for kl_str in kl_grades:
        cls = int(kl_str)
        kl_dir = src / kl_str
        if not kl_dir.exists():
            continue
        imgs = list(kl_dir.glob("*.jpg")) + list(kl_dir.glob("*.png"))
        all_files[cls].extend(sorted(imgs))
        log.info("[OA] KL%s → clase %d: %d imágenes", kl_str, cls, len(imgs))

    total_images = sum(len(v) for v in all_files.values())
    if total_images == 0:
        log.warning("[OA] No se encontraron imágenes en KLGrade/.")
        return {"status": "⚠️", "reason": "sin imágenes"}

    log.info("[OA] Total imágenes: %d en 5 clases KL directas", total_images)

    # ── Limpieza previa ───────────────────────────────────────────────────────
    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    # ── Split por grupos de similitud ─────────────────────────────────────────
    counts = {"train": 0, "val": 0, "test": 0}
    total_groups = 0
    group_report = {}

    for cls, files in all_files.items():
        if not files:
            continue

        log.info(
            "[OA] Clase %d — agrupando %d imágenes por similitud...", cls, len(files)
        )

        # Agrupar imágenes del mismo paciente/rodilla
        groups = _group_by_similarity(
            files,
            threshold=similarity_threshold,
            fingerprint_size=fingerprint_size,
        )

        n_groups = len(groups)
        total_groups += n_groups
        group_report[cls] = {
            "images": len(files),
            "groups": n_groups,
            "reduction_ratio": round(len(files) / max(n_groups, 1), 2),
        }

        # La "clase" de cada grupo es la de su carpeta (todas iguales aquí)
        group_classes = [cls] * n_groups

        if n_groups < 3:
            # Muy pocos grupos — split naive por imagen como fallback
            log.warning(
                "[OA] Clase %d: solo %d grupos — usando split naive por imagen.",
                cls,
                n_groups,
            )
            rng = np.random.default_rng(SEED)
            files_arr = list(files)
            rng.shuffle(files_arr)
            n = len(files_arr)
            n_tr = int(0.80 * n)
            n_va = int(0.10 * n)
            splits_files = {
                "train": files_arr[:n_tr],
                "val": files_arr[n_tr : n_tr + n_va],
                "test": files_arr[n_tr + n_va :],
            }
            for split_name, imgs in splits_files.items():
                d = splits_dir / split_name / str(cls)
                d.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    shutil.copy2(img, d / img.name)
                counts[split_name] += len(imgs)
            continue

        # Split ponderado por tamaño de grupo para alcanzar 80/10/10
        # en imágenes (no solo en grupos). Se barajan los grupos y se
        # asignan greedily al split que más lejos esté de su objetivo.
        total_cls_imgs = sum(len(g) for g in groups)
        targets = {
            "train": 0.80 * total_cls_imgs,
            "val": 0.10 * total_cls_imgs,
            "test": 0.10 * total_cls_imgs,
        }
        filled = {"train": 0, "val": 0, "test": 0}

        rng = np.random.default_rng(SEED)
        order = list(range(n_groups))
        rng.shuffle(order)

        assignment = {}
        for idx in order:
            g_size = len(groups[idx])
            # Elegir el split con mayor déficit relativo
            best = max(
                targets,
                key=lambda s: targets[s] - filled[s],
            )
            assignment[idx] = best
            filled[best] += g_size

        # Copiar imágenes según el split asignado a su grupo
        for group_idx, group_imgs in enumerate(groups):
            split_name = assignment[group_idx]
            d = splits_dir / split_name / str(cls)
            d.mkdir(parents=True, exist_ok=True)
            for img in group_imgs:
                shutil.copy2(img, d / img.name)
                counts[split_name] += 1

        log.info(
            "[OA] Clase %d — %d grupos → train=%d val=%d test=%d",
            cls,
            n_groups,
            sum(1 for v in assignment.values() if v == "train"),
            sum(1 for v in assignment.values() if v == "val"),
            sum(1 for v in assignment.values() if v == "test"),
        )

    # ── Verificar proporciones finales ────────────────────────────────────────
    total_out = sum(counts.values())
    for split_name, count in counts.items():
        target = 0.80 if split_name == "train" else 0.10
        actual = count / max(total_out, 1)
        if abs(actual - target) > 0.05:
            log.warning(
                "[OA] Split %s: %.1f%% (objetivo %.0f%%) — desviación > 5%%. "
                "Normal si hay pocos grupos por clase.",
                split_name,
                actual * 100,
                target * 100,
            )

    r = {
        "status": "✅",
        **counts,
        "total": total_images,
        "total_groups": total_groups,
        "group_report": group_report,
        "similarity_threshold": similarity_threshold,
        "note": (
            "Split por grupos de similitud inferidos (pseudo-patient_id). "
            "Umbral=%.2f. Reducción media: %.1f imgs/grupo."
            % (
                similarity_threshold,
                total_images / max(total_groups, 1),
            )
        ),
    }
    log.info(
        "[OA] Splits finales: train=%d val=%d test=%d | "
        "%d grupos pseudo-paciente inferidos de %d imágenes",
        counts["train"],
        counts["val"],
        counts["test"],
        total_groups,
        total_images,
    )
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
    log.info(
        "[LUNA] %d seriesuids encontrados en %d archivos .mhd",
        len(all_uids),
        len(mhd_files),
    )

    # Estratificación por presencia de nódulos (positivo = UID en annotations.csv)
    annotations_path = luna_dir / "annotations.csv"
    if annotations_path.exists():
        try:
            ann = pd.read_csv(annotations_path)
            positive_uids = set(ann["seriesuid"].unique())
        except Exception:
            positive_uids = set()
        log.info(
            "[LUNA] %d UIDs positivos (con nódulos), %d negativos",
            len([u for u in all_uids if u in positive_uids]),
            len([u for u in all_uids if u not in positive_uids]),
        )
    else:
        positive_uids = set()
        log.warning("[LUNA] annotations.csv no encontrado — split sin estratificación")

    # Estratificar si hay suficientes positivos; fallback a shuffle aleatorio
    labels = np.array([1 if u in positive_uids else 0 for u in all_uids])
    n_positive = int(labels.sum())
    min_needed = 3  # mínimo para StratifiedShuffleSplit

    if n_positive >= min_needed and (len(all_uids) - n_positive) >= min_needed:
        from sklearn.model_selection import StratifiedShuffleSplit

        # Test split estratificado (10%)
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
        trainval_idx, test_idx = next(sss_test.split(all_uids, labels))

        trainval_uids = [all_uids[i] for i in trainval_idx]
        trainval_labels = labels[trainval_idx]
        test_uids_list = [all_uids[i] for i in test_idx]

        # Val split estratificado (10% del total ≈ 11.1% de trainval)
        val_ratio = 0.10 / 0.90
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=SEED
        )
        train_idx_local, val_idx_local = next(
            sss_val.split(trainval_uids, trainval_labels)
        )

        train_uids = [trainval_uids[i] for i in train_idx_local]
        val_uids = [trainval_uids[i] for i in val_idx_local]
        test_uids = test_uids_list
        log.info("[LUNA] Split estratificado por presencia de nódulos aplicado.")
    else:
        log.warning(
            "[LUNA] Insuficientes positivos (%d) para estratificar — split aleatorio.",
            n_positive,
        )
        rng = np.random.default_rng(SEED)
        uids_arr = np.array(all_uids)
        rng.shuffle(uids_arr)
        n_total = len(uids_arr)
        n_test = int(0.10 * n_total)
        n_val = int(0.10 * n_total)
        test_uids = uids_arr[:n_test].tolist()
        val_uids = uids_arr[n_test : n_test + n_val].tolist()
        train_uids = uids_arr[n_test + n_val :].tolist()

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
    log.info(
        "[LUNA] Splits: train=%d val=%d test=%d (total=%d)",
        r["train"],
        r["val"],
        r["test"],
        r["total"],
    )
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  Pancreas PANORAMA — 10% test fijo + k-fold (k=5) sobre 90%
# ══════════════════════════════════════════════════════════════════════════════


def split_pancreas(datasets_dir):
    # type: (Path) -> dict
    """
    10% test fijo estratificado + k-fold CV (k=5) sobre el 90% restante.
    Requiere pancreas_labels_binary.csv del Paso 4.

    La unidad de splitting es patient_id (prefijo antes del último '_' en case_id),
    NO case_id individual. Esto evita data leakage cuando un paciente tiene
    múltiples estudios (e.g. 100047_00001, 100047_00002, ...).
    """
    csv_path = datasets_dir / "pancreas_labels_binary.csv"
    out_csv = datasets_dir / "pancreas_splits.csv"
    test_ids_file = datasets_dir / "pancreas_test_ids.txt"

    if out_csv.exists() and out_csv.stat().st_size > 100:
        log.info("[PANCREAS] pancreas_splits.csv ya existe, saltando.")
        return {"status": "✅", "skipped": True}

    if not csv_path.exists():
        log.warning(
            "[PANCREAS] pancreas_labels_binary.csv no existe — split pendiente."
        )
        return {"status": "warning", "reason": "sin etiquetas binarias"}

    df = pd.read_csv(csv_path)
    # Filtrar solo los que tienen etiqueta válida
    df_valid = df[df["label"] >= 0].copy()
    log.info(
        "[PANCREAS] %d volúmenes con etiqueta válida (de %d total)",
        len(df_valid),
        len(df),
    )

    # Extraer patient_id del case_id (formato: {patient_id}_{study_id})
    df_valid["patient_id"] = (
        df_valid["case_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
    )

    # ── Paso 1: Seleccionar 10% de PACIENTES como test ────────────────────────
    unique_patients = df_valid["patient_id"].unique()
    # Etiqueta por paciente: usar la moda (etiqueta más frecuente)
    patient_label = df_valid.groupby("patient_id")["label"].agg(
        lambda x: int(x.mode().iloc[0])
    )
    patient_ids_arr = patient_label.index.tolist()
    patient_labels_arr = patient_label.values.tolist()

    try:
        remain_patient_ids, test_patient_ids = train_test_split(
            patient_ids_arr,
            test_size=0.10,
            stratify=patient_labels_arr,
            random_state=SEED,
        )
    except ValueError:
        remain_patient_ids, test_patient_ids = train_test_split(
            patient_ids_arr,
            test_size=0.10,
            random_state=SEED,
        )

    test_patient_set = set(test_patient_ids)
    remain_patient_set = set(remain_patient_ids)

    # ── Paso 2: Separar cases en test y cv_pool ───────────────────────────────
    df_test = df_valid[df_valid["patient_id"].isin(test_patient_set)].copy()
    df_cv = df_valid[df_valid["patient_id"].isin(remain_patient_set)].copy()

    # ── Paso 3: GroupKFold(n_splits=5) sobre cv_pool, agrupando por patient_id
    gkf = GroupKFold(n_splits=5)
    cv_groups = df_cv["patient_id"].values
    fold_assignments = {}

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(df_cv, groups=cv_groups), 1
    ):
        for i in train_idx:
            cid = str(df_cv.iloc[i]["case_id"])
            if cid not in fold_assignments:
                fold_assignments[cid] = []
            fold_assignments[cid].append("fold{}_train".format(fold_idx))
        for i in val_idx:
            cid = str(df_cv.iloc[i]["case_id"])
            if cid not in fold_assignments:
                fold_assignments[cid] = []
            fold_assignments[cid].append("fold{}_val".format(fold_idx))

    # Construir CSV de salida
    rows = []
    for _, row in df_test.iterrows():
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "label": int(row["label"]),
                "split": "test",
            }
        )

    for _, row in df_cv.iterrows():
        cid = str(row["case_id"])
        lbl = int(row["label"])
        for split_tag in fold_assignments.get(cid, []):
            rows.append({"case_id": cid, "label": lbl, "split": split_tag})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    # Archivo de test IDs para referencia rápida (patient_ids de test)
    test_case_ids = df_test["case_id"].astype(str).tolist()
    test_ids_file.write_text("\n".join(test_case_ids) + "\n")

    n_test_cases = len(df_test)
    n_cv_cases = len(df_cv)
    log.info(
        "[PANCREAS] Splits: test=%d cases (%d patients) | "
        "train/val=5-fold sobre %d cases (%d patients)",
        n_test_cases,
        len(test_patient_ids),
        n_cv_cases,
        len(remain_patient_ids),
    )
    return {
        "status": "✅",
        "test": n_test_cases,
        "test_patients": len(test_patient_ids),
        "train_val_pool": n_cv_cases,
        "train_val_patients": len(remain_patient_ids),
        "total": n_test_cases + n_cv_cases,
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
    for split_name, fname in [
        ("train", "nih_train_list.txt"),
        ("val", "nih_val_list.txt"),
        ("test", "nih_test_list.txt"),
    ]:
        fpath = nih_dir / "splits" / fname
        if fpath.exists():
            for line in fpath.read_text().strip().split("\n"):
                img = line.strip()
                if img:
                    rows.append(
                        {
                            "ruta_imagen": str(nih_dir / "all_images" / img),
                            "dataset_origen": "nih",
                            "split": split_name,
                            "expert_id": 0,
                            "tipo_dato": "2d_image",
                        }
                    )

    # ISIC
    for split_name in ["train", "val", "test"]:
        fpath = isic_dir / "splits" / "isic_{}.csv".format(split_name)
        if fpath.exists():
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                img_name = row["image"]
                rows.append(
                    {
                        "ruta_imagen": str(
                            isic_dir / "ISIC_2019_Training_Input" / (img_name + ".jpg")
                        ),
                        "dataset_origen": "isic",
                        "split": split_name,
                        "expert_id": 1,
                        "tipo_dato": "2d_image",
                    }
                )

    # OA
    splits_path = oa_dir / "oa_splits"
    if splits_path.exists():
        for split_name in ["train", "val", "test"]:
            split_dir = splits_path / split_name
            if split_dir.exists():
                for img in sorted(split_dir.rglob("*.*")):
                    if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                        rows.append(
                            {
                                "ruta_imagen": str(img),
                                "dataset_origen": "oa",
                                "split": split_name,
                                "expert_id": 2,
                                "tipo_dato": "2d_image",
                            }
                        )

    # LUNA (parches .npy)
    patches_dir = luna_dir / "patches"
    for split_name in ["train", "val", "test"]:
        split_dir = patches_dir / split_name
        if split_dir.exists():
            for npy in sorted(split_dir.glob("candidate_*.npy")):
                rows.append(
                    {
                        "ruta_imagen": str(npy),
                        "dataset_origen": "luna",
                        "split": split_name,
                        "expert_id": 3,
                        "tipo_dato": "3d_patch_npy",
                    }
                )

    # Pancreas (volúmenes .nii.gz con preprocesado on-the-fly)
    # IMPORTANTE: pancreas_splits.csv tiene múltiples filas por case_id (k-fold).
    # Se usa SOLO fold1 (split='fold1_train' | 'fold1_val' | 'test') para evitar
    # que el mismo volumen aparezca como train Y val en el CAE.
    pancreas_splits = datasets_dir / "pancreas_splits.csv"
    if pancreas_splits.exists():
        pdf = pd.read_csv(pancreas_splits)
        zenodo_dir = datasets_dir / "zenodo_13715870"

        # Seleccionar solo fold1 y test (un único rol por volumen)
        fold1_mask = pdf["split"].isin(["fold1_train", "fold1_val", "test"])
        pdf_canonical = pdf[fold1_mask].copy()

        # Normalizar split names
        pdf_canonical["norm_split"] = pdf_canonical["split"].map(
            {"fold1_train": "train", "fold1_val": "val", "test": "test"}
        )

        seen_paths: set = set()
        for _, prow in pdf_canonical.iterrows():
            norm_split = prow["norm_split"]
            nii_candidates = list(
                zenodo_dir.rglob("{}*.nii.gz".format(prow["case_id"]))
            )
            for nii in nii_candidates:
                nii_str = str(nii)
                if nii_str in seen_paths:
                    continue  # deduplicar por ruta
                seen_paths.add(nii_str)
                rows.append(
                    {
                        "ruta_imagen": nii_str,
                        "dataset_origen": "pancreas",
                        "split": norm_split,
                        "expert_id": 4,
                        "tipo_dato": "3d_volume_nifti",
                    }
                )

    if not rows:
        log.warning("[CAE] No se encontraron datos para construir cae_splits.csv.")
        return {"status": "warning", "reason": "sin datos"}

    cae_df = pd.DataFrame(rows)
    cae_df.to_csv(cae_csv, index=False)

    by_split = cae_df.groupby("split").size().to_dict()
    by_dataset = cae_df.groupby("dataset_origen").size().to_dict()
    log.info(
        "[CAE] cae_splits.csv: %d filas | Splits: %s | Datasets: %s",
        len(cae_df),
        by_split,
        by_dataset,
    )
    return {
        "status": "✅",
        "total": len(cae_df),
        "by_split": by_split,
        "by_dataset": by_dataset,
    }


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
            results["isic"] = build_lesion_split(datasets_dir)
        except Exception as e:
            log.error("[ISIC] Error en split: %s", e)
            results["isic"] = {"status": "error", "error": str(e)}

    if "oa" in active:
        try:
            results["oa"] = split_oa(datasets_dir)
        except Exception as e:
            log.error("[OA] Error en split: %s", e)
            results["oa"] = {"status": "error", "error": str(e)}

    if "luna_ct" in active or "luna_meta" in active:
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
    # Verificar que no hay errores bloqueantes antes de construir el manifiesto
    failed_splits = [
        k
        for k, v in results.items()
        if isinstance(v, dict) and v.get("status") == "error"
    ]
    if failed_splits:
        log.warning(
            "[CAE] Los siguientes splits fallaron: %s — "
            "cae_splits.csv se generará con datos parciales.",
            failed_splits,
        )
    try:
        results["cae"] = build_cae_splits(datasets_dir)
    except Exception as e:
        log.error("[CAE] Error construyendo cae_splits.csv: %s", e)
        results["cae"] = {"status": "error", "error": str(e)}

    # Calcular status global
    has_error = any(
        v.get("status") == "error" for v in results.values() if isinstance(v, dict)
    )
    results["status"] = "⚠️" if has_error else "✅"

    return results
