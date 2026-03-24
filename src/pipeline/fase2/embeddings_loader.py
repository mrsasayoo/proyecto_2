"""
embeddings_loader.py — Carga y validación de embeddings de Fase 1.

Responsabilidad única: cargar los archivos .npy y backbone_meta.json
producidos por Fase 1, verificar la integridad del contrato, y devolver
un dict con los arrays listos para usar.

Fase 2 es consumidora pura: este módulo NUNCA escribe ningún archivo en disco
ni modifica los artefactos de Fase 1.
"""

import json
import logging
from pathlib import Path

import numpy as np

from fase2_config import BACKBONE_META_REQUIRED_KEYS
from config import N_EXPERTS_DOMAIN

log = logging.getLogger("fase2")


def load_embeddings(emb_dir: str) -> dict:
    """
    Carga y valida los embeddings producidos por Fase 1.

    Verificaciones realizadas (en orden, para fallo rápido):
      1. Archivos obligatorios: Z_train.npy, y_train.npy, Z_val.npy, y_val.npy
      2. backbone_meta.json: claves requeridas y consistencia de d_model
      3. Ausencia de NaN e Inf en todos los arrays
      4. Clases presentes en y_train == {0, 1, ..., N_EXPERTS_DOMAIN-1}
      5. Carga opcional de Z_test.npy / y_test.npy

    Parameters
    ----------
    emb_dir : str | Path
        Directorio que contiene los artefactos de Fase 1.

    Returns
    -------
    dict con claves:
        Z_train, y_train : np.ndarray  — arrays de train
        Z_val,   y_val   : np.ndarray  — arrays de validación
        Z_test,  y_test  : np.ndarray  — arrays de test (vacíos si no existen)
        backbone_meta    : dict         — contenido de backbone_meta.json (o {})
        d_model          : int          — dimensión del embedding verificada
        has_test         : bool         — True si Z_test tiene muestras

    Raises
    ------
    FileNotFoundError
        Si algún archivo obligatorio no existe.
    ValueError
        Si alguna verificación de integridad falla.
    """
    emb_dir = Path(emb_dir)

    # ── 1. Verificar archivos obligatorios ──────────────────────────────
    required = ["Z_train.npy", "y_train.npy", "Z_val.npy", "y_val.npy"]
    for fname in required:
        fpath = emb_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"[Loader] Archivo obligatorio faltante: {fpath}\n"
                f"Ejecuta Fase 1 para generarlo:\n"
                f"  python src/pipeline/fase1/fase1_pipeline.py --output_dir {emb_dir}"
            )

    # ── 2. Cargar backbone_meta.json ────────────────────────────────────
    meta_path = emb_dir / "backbone_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            backbone_meta = json.load(f)
        log.info("[Loader] backbone_meta.json cargado: %s", backbone_meta)

        # Verificar claves requeridas
        missing_keys = BACKBONE_META_REQUIRED_KEYS - set(backbone_meta.keys())
        if missing_keys:
            raise ValueError(
                f"[Loader] backbone_meta.json incompleto. "
                f"Claves faltantes: {missing_keys}. "
                f"Ejecuta Fase 1 con la versión actualizada de embeddings_storage.py."
            )
    else:
        backbone_meta = {}
        log.warning(
            "[Loader] backbone_meta.json no encontrado en %s. "
            "d_model se inferirá de Z_train.shape[1]. "
            "Ejecuta Fase 1 con la versión actualizada para generarlo.",
            emb_dir,
        )

    # ── 3. Cargar arrays obligatorios ───────────────────────────────────
    Z_train = np.load(emb_dir / "Z_train.npy")
    y_train = np.load(emb_dir / "y_train.npy")
    Z_val = np.load(emb_dir / "Z_val.npy")
    y_val = np.load(emb_dir / "y_val.npy")
    d_model = int(Z_train.shape[1])

    log.info("[Loader] Z_train: %s  |  Z_val: %s", Z_train.shape, Z_val.shape)
    log.info("[Loader] d_model inferido de Z_train.shape[1]: %d", d_model)

    # ── 4. Verificar consistencia d_model vs backbone_meta ──────────────
    if backbone_meta.get("d_model") is not None:
        meta_d = int(backbone_meta["d_model"])
        if meta_d != d_model:
            raise ValueError(
                f"[Loader] d_model inconsistente: Z_train.shape[1]={d_model} "
                f"pero backbone_meta.json dice d_model={meta_d}. "
                f"Los embeddings pueden estar mezclados de distintas corridas de Fase 1."
            )

    # Verificar n_train y n_val si están disponibles
    if backbone_meta.get("n_train") is not None:
        if int(backbone_meta["n_train"]) != Z_train.shape[0]:
            log.warning(
                "[Loader] n_train en backbone_meta (%d) ≠ Z_train.shape[0] (%d). "
                "Puede indicar embeddings parciales.",
                backbone_meta["n_train"],
                Z_train.shape[0],
            )
    if backbone_meta.get("n_val") is not None:
        if int(backbone_meta["n_val"]) != Z_val.shape[0]:
            log.warning(
                "[Loader] n_val en backbone_meta (%d) ≠ Z_val.shape[0] (%d).",
                backbone_meta["n_val"],
                Z_val.shape[0],
            )

    # ── 5. Verificar ausencia de NaN e Inf ──────────────────────────────
    _check_nan_inf(Z_train, "Z_train")
    _check_nan_inf(Z_val, "Z_val")

    # ── 6. Verificar clases en y_train ──────────────────────────────────
    classes_in_train = set(np.unique(y_train).tolist())
    expected_classes = set(range(N_EXPERTS_DOMAIN))
    if classes_in_train != expected_classes:
        missing_cls = expected_classes - classes_in_train
        extra_cls = classes_in_train - expected_classes
        raise ValueError(
            f"[Loader] Clases en y_train: {classes_in_train} — "
            f"se esperaban {expected_classes}. "
            f"Clases faltantes: {missing_cls}. Clases extra: {extra_cls}. "
            f"El ablation study no será válido si algún experto no tiene muestras."
        )

    # ── 7. Cargar arrays opcionales de test ─────────────────────────────
    test_path_Z = emb_dir / "Z_test.npy"
    test_path_y = emb_dir / "y_test.npy"

    if test_path_Z.exists() and test_path_y.exists():
        Z_test = np.load(test_path_Z)
        y_test = np.load(test_path_y)
        has_test = len(Z_test) > 0
        if has_test:
            log.info("[Loader] Z_test cargado: %s", Z_test.shape)
            _check_nan_inf(Z_test, "Z_test")
            if backbone_meta.get("n_test") is not None:
                if int(backbone_meta["n_test"]) != Z_test.shape[0]:
                    log.warning(
                        "[Loader] n_test en backbone_meta (%d) ≠ Z_test.shape[0] (%d).",
                        backbone_meta["n_test"],
                        Z_test.shape[0],
                    )
        else:
            log.warning("[Loader] Z_test existe pero está vacío — ignorado.")
    else:
        Z_test = np.zeros((0, d_model), dtype=np.float32)
        y_test = np.zeros(0, dtype=np.int64)
        has_test = False
        log.warning("[Loader] Z_test no disponible — reporte final solo sobre val.")

    return {
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_val": Z_val,
        "y_val": y_val,
        "Z_test": Z_test,
        "y_test": y_test,
        "backbone_meta": backbone_meta,
        "d_model": d_model,
        "has_test": has_test,
    }


def _check_nan_inf(arr: np.ndarray, name: str) -> None:
    """Verifica ausencia de NaN e Inf en un array. Lanza ValueError si se detectan."""
    if np.isnan(arr).any():
        raise ValueError(
            f"[Loader] NaN detectado en {name}. "
            f"Regenera los embeddings con Fase 1 usando un backbone sin problemas numéricos."
        )
    if np.isinf(arr).any():
        raise ValueError(
            f"[Loader] Inf detectado en {name}. Regenera los embeddings con Fase 1."
        )
