"""
pre_isic.py — Preprocesamiento offline del dataset ISIC 2019.

Ejecutar UNA VEZ antes del entrenamiento. Genera caché de imágenes
preprocesadas con:
  1. DullRazor (hair removal)
  2. Resize aspect-ratio-preserving (lado corto = target_size)
  3. Guardado JPEG calidad 95

Color Constancy NO se aplica aquí (se aplica online con p=0.5 en
el training loop, para que ConvNeXt-Small vea variabilidad cromática).

Salida: {out_dir}/{isic_id}_pp_{target_size}.jpg + preprocess_report.json

Uso:
    python src/pipeline/fase0/pre_isic.py \\
        --img_dir datasets/isic_2019/ISIC_2019_Training_Input \\
        --out_dir datasets/isic_2019/ISIC_2019_Training_Input_preprocessed \\
        --gt_csv datasets/isic_2019/ISIC_2019_Training_GroundTruth.csv \\
        --target_size 224 \\
        --quality 95 \\
        --max_workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger("pre_isic")

# Intentar importar OpenCV; fallback si no disponible
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    warnings.warn(
        "OpenCV (cv2) no disponible. DullRazor hair removal desactivado.",
        stacklevel=2,
    )


# ── Funciones de preprocesamiento ──────────────────────────────────────


def shades_of_gray(img_array: np.ndarray, power: int = 6) -> np.ndarray:
    """
    Shades of Gray color constancy (Finlayson & Trezzi, 2004).

    Normaliza el iluminante estimado por la norma de Minkowski (power=6).
    Aplica corrección diagonal von Kries para simular luz blanca canónica.

    Args:
        img_array: imagen RGB uint8, shape (H, W, 3).
        power: exponente de la norma de Minkowski. 6 es el estándar.

    Returns:
        Imagen corregida, mismo shape y dtype que la entrada.
    """
    arr = img_array.astype(np.float32)
    illuminant = np.mean(arr**power, axis=(0, 1)) ** (1.0 / power)
    illuminant = illuminant / (np.mean(illuminant) + 1e-6)
    corrected = arr / (illuminant + 1e-6)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def remove_hair_dullrazor(img_array: np.ndarray) -> np.ndarray:
    """
    DullRazor hair removal (Lee et al., 1997).

    Pipeline:
      1. Convertir a escala de grises.
      2. Morphological closing (kernel 3x3) para estimar fondo sin pelo.
      3. Detectar píxeles con diferencia > threshold=10 (pelo oscuro).
      4. Inpainting con cv2.INPAINT_TELEA (radio=3).

    Si cv2 no está disponible, retorna la imagen original con warning.

    Args:
        img_array: imagen RGB uint8, shape (H, W, 3).

    Returns:
        Imagen con pelo eliminado, mismo shape y dtype.
    """
    if not _HAS_CV2:
        warnings.warn(
            "cv2 no disponible: retornando imagen sin hair removal.",
            stacklevel=2,
        )
        return img_array

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Morphological closing: estima el fondo sin pelo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Diferencia: píxeles de pelo tienen valores más bajos que el fondo
    diff = cv2.absdiff(closed, gray)

    # Umbral: píxeles con diferencia > 10 son pelo
    threshold = 10
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilatar la máscara para cubrir mejor los bordes del pelo
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpainting: rellenar las regiones de pelo
    result = cv2.inpaint(img_array, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result


def resize_shorter_side(img_array: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize manteniendo aspect ratio: lado corto = target_size.

    Usa LANCZOS interpolation (cv2.INTER_LANCZOS4) para máxima calidad.
    Fallback a PIL si cv2 no disponible.

    Args:
        img_array: imagen RGB uint8, shape (H, W, 3).
        target_size: tamaño deseado para el lado más corto.

    Returns:
        Imagen redimensionada, shape (new_H, new_W, 3).
    """
    h, w = img_array.shape[:2]
    scale = target_size / min(h, w)
    new_w = math.ceil(w * scale)
    new_h = math.ceil(h * scale)

    if _HAS_CV2:
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        pil_img = Image.fromarray(img_array)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(pil_img)


def audit_isic_dataset(
    img_dir: str | Path,
    gt_csv: str | Path,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Audita el dataset ISIC 2019: registra dimensiones, aspect ratio, fuente.

    Args:
        img_dir: directorio con imágenes .jpg originales.
        gt_csv: CSV ground truth ISIC 2019.
        out_csv: si se proporciona, guarda el audit a CSV.

    Returns:
        DataFrame con columnas: image_id, width, height, aspect_ratio,
        source, file_exists.
    """
    img_dir = Path(img_dir)
    df_gt = pd.read_csv(gt_csv)

    records: list[dict] = []
    for _, row in df_gt.iterrows():
        image_id = row["image"]
        fpath = img_dir / f"{image_id}.jpg"
        rec: dict = {"image_id": image_id, "file_exists": fpath.exists()}

        if fpath.exists():
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                rec["width"] = w
                rec["height"] = h
                rec["aspect_ratio"] = round(w / h, 3) if h > 0 else 0.0
            except Exception as e:
                rec["width"] = 0
                rec["height"] = 0
                rec["aspect_ratio"] = 0.0
                log.warning(f"Error leyendo '{fpath}': {e}")
        else:
            rec["width"] = 0
            rec["height"] = 0
            rec["aspect_ratio"] = 0.0

        # Inferir fuente
        if image_id.endswith("_downsampled"):
            rec["source"] = "MSK"
        else:
            try:
                num_id = int(image_id.replace("ISIC_", ""))
                rec["source"] = "HAM" if num_id <= 67977 else "BCN"
            except ValueError:
                rec["source"] = "UNKNOWN"

        records.append(rec)

    df_audit = pd.DataFrame(records)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_audit.to_csv(out_csv, index=False)
        log.info(f"[Audit] Guardado en {out_csv} ({len(df_audit):,} filas)")

    return df_audit


def _validate_isic_sample(
    out_dir: Path,
    target_size: int = 224,
    n_sample: int = 100,
) -> tuple[float, list[Path]]:
    """
    Valida una muestra aleatoria de imágenes preprocesadas en out_dir.

    Para cada archivo muestreado comprueba: apertura PIL sin excepción,
    modo RGB, lado corto == target_size, y varianza > 5.0 (no constante).

    Args:
        out_dir: directorio con imágenes *_pp_224.jpg.
        target_size: tamaño esperado del lado corto.
        n_sample: cantidad máxima de archivos a muestrear.

    Returns:
        Tupla (pass_rate, corrupt_paths) donde pass_rate es la fracción
        de archivos válidos y corrupt_paths lista las rutas que fallaron.
    """
    all_files = list(out_dir.glob(f"*_pp_{target_size}.jpg"))
    if not all_files:
        return 0.0, []

    sample = random.sample(all_files, min(n_sample, len(all_files)))
    corrupt: list[Path] = []

    for fpath in sample:
        try:
            with Image.open(fpath) as img:
                if img.mode != "RGB":
                    corrupt.append(fpath)
                    continue
                if min(img.width, img.height) != target_size:
                    corrupt.append(fpath)
                    continue
                arr = np.array(img)
                if arr.std() <= 5.0:
                    corrupt.append(fpath)
                    continue
        except Exception:
            corrupt.append(fpath)

    valid = len(sample) - len(corrupt)
    pass_rate = valid / len(sample)
    return pass_rate, corrupt


def _process_single_image(
    image_id: str,
    img_dir: Path,
    out_dir: Path,
    target_size: int,
    quality: int,
    apply_hair_removal: bool,
) -> dict:
    """Procesa una imagen individual. Retorna dict con resultado."""
    out_name = f"{image_id}_pp_{target_size}.jpg"
    out_path = out_dir / out_name

    # Idempotente: saltar si ya existe Y es válida
    if out_path.exists():
        try:
            with Image.open(out_path) as existing:
                if (
                    existing.mode == "RGB"
                    and min(existing.width, existing.height) == target_size
                ):
                    return {"image_id": image_id, "status": "skipped", "error": None}
            # Validación falló: eliminar archivo corrupto y reprocesar
            out_path.unlink()
            log.debug(f"Archivo corrupto eliminado para reproceso: {out_path.name}")
        except Exception:
            # PIL no pudo abrir: eliminar y reprocesar
            out_path.unlink(missing_ok=True)
            log.debug(f"Archivo ilegible eliminado para reproceso: {out_path.name}")

    src_path = img_dir / f"{image_id}.jpg"
    if not src_path.exists():
        return {"image_id": image_id, "status": "missing", "error": "file not found"}

    try:
        img = Image.open(src_path).convert("RGB")
        arr = np.array(img)

        # Hair removal
        if apply_hair_removal:
            arr = remove_hair_dullrazor(arr)

        # Resize shorter side
        arr = resize_shorter_side(arr, target_size)

        # Guardar como JPEG
        out_img = Image.fromarray(arr)
        out_img.save(out_path, "JPEG", quality=quality)

        return {"image_id": image_id, "status": "ok", "error": None}
    except Exception as e:
        return {"image_id": image_id, "status": "error", "error": str(e)}


def preprocess_isic_dataset(
    img_dir: str | Path,
    out_dir: str | Path,
    gt_csv: str | Path,
    target_size: int = 224,
    quality: int = 95,
    apply_hair_removal: bool = True,
    max_workers: int = 4,
    dry_run: bool = False,
) -> dict:
    """
    Orquestador de preprocesamiento offline ISIC 2019.

    Por cada imagen: load → DullRazor hair removal → resize shorter_side →
    guardar como {isic_id}_pp_{target_size}.jpg con calidad JPEG.

    Color Constancy NO se aplica aquí (se aplica online con p=0.5).

    Args:
        img_dir: directorio con imágenes .jpg originales.
        out_dir: directorio de salida para la caché.
        gt_csv: CSV ground truth ISIC 2019.
        target_size: tamaño del lado corto tras resize.
        quality: calidad JPEG (1-100).
        apply_hair_removal: aplicar DullRazor.
        max_workers: hilos para ThreadPoolExecutor.
        dry_run: si True, procesa solo 10 imágenes.

    Returns:
        dict con estadísticas del preprocesamiento.
    """
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_gt = pd.read_csv(gt_csv)
    image_ids = df_gt["image"].tolist()
    total_expected = len(image_ids)

    # ── Validación upfront de caché existente ──────────────────────────
    existing_files = list(out_dir.glob(f"*_pp_{target_size}.jpg"))
    n_existing = len(existing_files)

    if n_existing > 0:
        coverage = n_existing / total_expected if total_expected > 0 else 0.0
        log.info(
            f"[Preprocess] Caché existente: {n_existing:,}/{total_expected:,} "
            f"({100 * coverage:.1f}%) archivos encontrados"
        )

        if coverage >= 0.90:
            pass_rate, corrupt_paths = _validate_isic_sample(
                out_dir, target_size=target_size
            )
            log.info(
                f"[Preprocess] Validación de muestra: pass_rate={pass_rate:.2%} "
                f"({len(corrupt_paths)} corruptos en muestra)"
            )

            if pass_rate >= 0.95:
                log.info(
                    "[Preprocess] Caché válida (cobertura>=90%, "
                    f"pass_rate={pass_rate:.2%}>=95%). Skip completo."
                )
                return {
                    "total_images": total_expected,
                    "processed_ok": 0,
                    "skipped_existing": n_existing,
                    "missing_source": 0,
                    "errors": 0,
                    "target_size": target_size,
                    "jpeg_quality": quality,
                    "hair_removal": apply_hair_removal,
                    "elapsed_seconds": 0.0,
                    "error_details": [],
                    "validation_skip": True,
                    "pass_rate": pass_rate,
                }

            # pass_rate < 0.95: eliminar corruptos y continuar
            for p in corrupt_paths:
                p.unlink(missing_ok=True)
            log.warning(
                f"[Preprocess] Eliminados {len(corrupt_paths)} archivos corruptos "
                "de la muestra. Continuando con reproceso."
            )

            if pass_rate < 0.50:
                log.warning(
                    f"[Preprocess] ⚠ pass_rate MUY BAJO ({pass_rate:.2%}). "
                    "Posible corrupción masiva del caché. Revisión recomendada."
                )

    # ── Fin validación upfront ─────────────────────────────────────────

    if dry_run:
        image_ids = image_ids[:10]
        log.info(f"[DryRun] Procesando solo {len(image_ids)} imágenes")

    total = len(image_ids)
    log.info(
        f"[Preprocess] Inicio: {total:,} imágenes | "
        f"target_size={target_size} | quality={quality} | "
        f"hair_removal={apply_hair_removal} | workers={max_workers}"
    )

    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}
    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_image,
                image_id,
                img_dir,
                out_dir,
                target_size,
                quality,
                apply_hair_removal,
            ): image_id
            for image_id in image_ids
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "error":
                errors.append(result)

            if i % 500 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                log.info(
                    f"[Preprocess] {i:>6,}/{total:,} "
                    f"({100 * i / total:.1f}%) | "
                    f"{rate:.1f} img/s | "
                    f"ok={counts['ok']} skip={counts['skipped']} "
                    f"miss={counts['missing']} err={counts['error']}"
                )

    elapsed_total = time.time() - t0

    # Reporte
    report = {
        "total_images": total,
        "processed_ok": counts["ok"],
        "skipped_existing": counts["skipped"],
        "missing_source": counts["missing"],
        "errors": counts["error"],
        "target_size": target_size,
        "jpeg_quality": quality,
        "hair_removal": apply_hair_removal,
        "elapsed_seconds": round(elapsed_total, 1),
        "error_details": errors[:50],  # limitar a 50 errores
    }

    report_path = out_dir / "preprocess_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(
        f"[Preprocess] Finalizado en {elapsed_total:.1f}s | "
        f"ok={counts['ok']} | skip={counts['skipped']} | "
        f"miss={counts['missing']} | err={counts['error']}"
    )
    log.info(f"[Preprocess] Reporte guardado en {report_path}")

    return report


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Preprocesamiento offline ISIC 2019")
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Dir con imágenes .jpg originales",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Dir de salida para caché",
    )
    parser.add_argument(
        "--gt_csv",
        required=True,
        help="CSV ground truth ISIC 2019",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
    )
    parser.add_argument(
        "--no_hair_removal",
        action="store_true",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Procesar solo 10 imgs",
    )

    args = parser.parse_args()

    preprocess_isic_dataset(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        gt_csv=args.gt_csv,
        target_size=args.target_size,
        quality=args.quality,
        apply_hair_removal=not args.no_hair_removal,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
