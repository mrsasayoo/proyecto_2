"""
embeddings_storage.py — Persistencia en Disco
==============================================

Responsabilidad única: guardar los arrays de embeddings en disco en el
formato que Fase 2 espera, y cargarlos de vuelta de forma verificada.

backbone_meta.json es el contrato de interfaz entre Fase 1 y Fase 2.
"""

import json
import logging
from pathlib import Path

import numpy as np

from fase1_config import BACKBONE_CONFIGS, BACKBONE_META_KEYS

log = logging.getLogger("fase1")


def save_embeddings(output_dir, backbone_name,
                    Z_train, y_train, names_train,
                    Z_val, y_val, names_val,
                    Z_test, y_test, names_test):
    """
    Escribe todos los artefactos de embeddings en disco.

    Archivos:
        Z_train.npy, y_train.npy, Z_val.npy, y_val.npy,
        Z_test.npy, y_test.npy, names_train.txt, names_val.txt,
        names_test.txt, backbone_meta.json

    Etiquetas como int64 (PyTorch CrossEntropyLoss usa torch.long).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Arrays .npy
    np.save(out / "Z_train.npy", Z_train)
    np.save(out / "y_train.npy", y_train.astype(np.int64))
    np.save(out / "Z_val.npy",   Z_val)
    np.save(out / "y_val.npy",   y_val.astype(np.int64))
    np.save(out / "Z_test.npy",  Z_test)
    np.save(out / "y_test.npy",  y_test.astype(np.int64))

    log.debug("[Storage] Z_train: %s (%.1f MB)", Z_train.shape,
              Z_train.nbytes / 1e6)
    log.debug("[Storage] Z_val  : %s (%.1f MB)", Z_val.shape,
              Z_val.nbytes / 1e6)
    log.debug("[Storage] Z_test : %s (%.1f MB)", Z_test.shape,
              Z_test.nbytes / 1e6)

    # Nombres de archivo
    (out / "names_train.txt").write_text("\n".join(names_train), encoding="utf-8")
    (out / "names_val.txt").write_text("\n".join(names_val), encoding="utf-8")
    (out / "names_test.txt").write_text("\n".join(names_test), encoding="utf-8")

    # backbone_meta.json — contrato entre Fase 1 y Fase 2
    d_model = int(Z_train.shape[1]) if Z_train.shape[0] > 0 else 0
    meta = {
        "backbone": backbone_name,
        "d_model":  d_model,
        "n_train":  int(Z_train.shape[0]),
        "n_val":    int(Z_val.shape[0]),
        "n_test":   int(Z_test.shape[0]),
        "vram_gb":  BACKBONE_CONFIGS[backbone_name]["vram_gb"],
    }

    # Verificar que contiene exactamente las claves esperadas por Fase 2
    meta_keys = set(meta.keys())
    if meta_keys != BACKBONE_META_KEYS:
        missing = BACKBONE_META_KEYS - meta_keys
        extra   = meta_keys - BACKBONE_META_KEYS
        log.error("[Storage] backbone_meta.json no coincide con contrato. "
                  "Faltantes: %s | Sobrantes: %s", missing, extra)

    meta_path = out / "backbone_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("[Storage] backbone_meta.json escrito: %s", meta_path)

    # Verificación post-escritura — cargar y comparar formas
    Z_check = np.load(out / "Z_train.npy")
    if Z_check.shape != Z_train.shape:
        log.error("[Storage] ¡Verificación post-escritura fallida! "
                  "Shape en disco %s ≠ original %s", Z_check.shape, Z_train.shape)
    else:
        log.info("[Storage] Verificación post-escritura OK ✓")

    log.info("[Storage] Archivos guardados en: %s", out)
    return str(out)


def load_embeddings(embeddings_dir):
    """
    Carga embeddings desde disco con verificación de integridad.

    Returns:
        dict con claves: Z_train, y_train, Z_val, y_val, Z_test, y_test,
        names_train, names_val, names_test, meta
    """
    d = Path(embeddings_dir)

    meta_path = d / "backbone_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    result = {"meta": meta}

    for split in ("train", "val", "test"):
        Z = np.load(d / "Z_{}.npy".format(split))
        y = np.load(d / "y_{}.npy".format(split))
        names_file = d / "names_{}.txt".format(split)
        names = names_file.read_text(encoding="utf-8").strip().split("\n") \
            if names_file.exists() else []

        # Verificar dimensiones
        if Z.shape[0] != y.shape[0]:
            log.error("[Storage/load] %s: Z tiene %d filas, y tiene %d",
                      split, Z.shape[0], y.shape[0])
        if Z.shape[0] > 0 and Z.shape[1] != meta["d_model"]:
            log.error("[Storage/load] %s: d_model en array (%d) ≠ meta (%d)",
                      split, Z.shape[1], meta["d_model"])
        # Verificar NaN/Inf
        if np.isnan(Z).any():
            log.error("[Storage/load] %s: NaN detectado en embeddings", split)
        if np.isinf(Z).any():
            log.error("[Storage/load] %s: Inf detectado en embeddings", split)

        result["Z_{}".format(split)]     = Z
        result["y_{}".format(split)]     = y
        result["names_{}".format(split)] = names

    log.info("[Storage/load] Embeddings cargados: train=%d val=%d test=%d d=%d",
             result["Z_train"].shape[0], result["Z_val"].shape[0],
             result["Z_test"].shape[0], meta["d_model"])
    return result


def log_distribution(y_train, y_val, y_test, expert_ids):
    """
    Registra distribución de etiquetas por split.

    Args:
        expert_ids: dict {nombre: id_experto}
    """
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if len(y) == 0:
            continue
        log.info("Distribución de expertos en %s:", split_name)
        for exp_name, exp_id in expert_ids.items():
            count = int((y == exp_id).sum())
            pct   = count / len(y) * 100
            log.info("  Experto %d (%-10s): %6s  (%.1f%%)",
                     exp_id, exp_name, f"{count:,}", pct)
        log.info("  Experto 5 (ood       ):      0  (0.0%%)  "
                 "← sin dataset en Fase 1")

        counts  = [(y == i).sum() for i in range(5)]
        nonzero = [c for c in counts if c > 0]
        if len(nonzero) > 1:
            ratio = max(nonzero) / min(nonzero)
            if ratio > 10:
                log.warning("[Balance/%s] ratio max/min = %.1fx — muy desigual.",
                            split_name, ratio)
            else:
                log.info("[Balance/%s] ratio max/min = %.1fx", split_name, ratio)
