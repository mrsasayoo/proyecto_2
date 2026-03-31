"""
embeddings_extractor.py — Extracción del CLS Token
===================================================

Responsabilidad única: recibir un backbone congelado y un DataLoader,
y devolver los arrays NumPy de embeddings, etiquetas y nombres de archivo.

Separado de dataset_builder y storage porque es el único paso que
requiere GPU y es el cuello de botella computacional del pipeline.
"""

import time
import logging

import numpy as np
import torch

from fase1_config import MIN_L2_NORM

log = logging.getLogger("fase1")


def extract_embeddings(model, dataloader, device, d_model, desc=""):
    """
    Pasa el dataloader completo por el backbone y acumula CLS tokens.

    Pre-allocación de memoria para predictibilidad de OOM.
    Detección activa de NaN/Inf por batch.
    Transferencia GPU→CPU batch a batch para minimizar uso de VRAM.

    Returns:
        Z         — np.ndarray [N, d_model] float32
        y_expert  — np.ndarray [N] int64
        img_names — list[str] de N nombres
    """
    n_total = len(dataloader.dataset)
    Z = np.zeros((n_total, d_model), dtype=np.float32)
    y_expert = np.zeros(n_total, dtype=np.int64)
    img_names = []
    cursor = 0

    nan_batches = 0
    inf_batches = 0
    t_start = time.time()
    # Determinar tipo de dispositivo una vez para el monitoreo VRAM
    _is_cuda = (hasattr(device, "type") and device.type == "cuda") or str(
        device
    ).startswith("cuda")

    log.info(
        "[Extract/%s] Iniciando: %s imágenes | d_model=%d | batch=%d",
        desc,
        f"{n_total:,}",
        d_model,
        dataloader.batch_size,
    )

    for batch_idx, (imgs, experts, names) in enumerate(dataloader):
        imgs = imgs.to(device)

        # OPT-3: inference_mode es más eficiente que no_grad (desactiva
        # autograd y versionado de tensores, reduciendo overhead en CPU)
        with torch.inference_mode():
            z = model(imgs)

        z_np = z.cpu().numpy()
        B = z_np.shape[0]

        # Detección de NaN/Inf por batch
        if np.isnan(z_np).any():
            nan_batches += 1
            log.error(
                "[Extract/%s] NaN en batch %d (imgs %d–%d)",
                desc,
                batch_idx,
                cursor,
                cursor + B,
            )
        if np.isinf(z_np).any():
            inf_batches += 1
            log.error(
                "[Extract/%s] Inf en batch %d (imgs %d–%d)",
                desc,
                batch_idx,
                cursor,
                cursor + B,
            )

        # Escritura directa en array pre-allocado (sin copias intermedias)
        Z[cursor : cursor + B] = z_np
        y_expert[cursor : cursor + B] = experts.numpy()
        img_names.extend(names)
        cursor += B

        # Progreso cada 50 batches
        if batch_idx % 50 == 0:
            elapsed = time.time() - t_start
            speed = cursor / elapsed if elapsed > 0 else 0
            eta_s = (n_total - cursor) / speed if speed > 0 else 0
            log.info(
                "[Extract/%s] %6s/%s (%.1f%%) | %.0f img/s | ETA %.1f min",
                desc,
                f"{cursor:,}",
                f"{n_total:,}",
                100 * cursor / n_total,
                speed,
                eta_s / 60,
            )

        # Monitoreo VRAM escalonado (cada 200 batches, no cada batch)
        if _is_cuda and batch_idx % 200 == 0:
            used_gb = torch.cuda.memory_allocated() / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            pct = 100 * used_gb / total_gb
            log.debug(
                "[Extract/%s] VRAM: %.2f/%.1f GB (%.1f%%)", desc, used_gb, total_gb, pct
            )
            if pct > 90:
                log.warning(
                    "[Extract/%s] VRAM al %.1f%% — riesgo de OOM. "
                    "Considera reducir --batch_size.",
                    desc,
                    pct,
                )

    elapsed_total = time.time() - t_start
    log.info(
        "[Extract/%s] Completado: %s embeddings en %.1fs (%.0f img/s)",
        desc,
        f"{cursor:,}",
        elapsed_total,
        cursor / elapsed_total if elapsed_total > 0 else 0,
    )

    # Reporte de anomalías
    if nan_batches or inf_batches:
        log.error(
            "[Extract/%s] ANOMALÍAS: NaN en %d batches | Inf en %d batches. "
            "Los embeddings no son confiables.",
            desc,
            nan_batches,
            inf_batches,
        )
    else:
        log.info("[Extract/%s] Sin NaN ni Inf detectados ✓", desc)

    # Verificación de normas L2
    norms = np.linalg.norm(Z[:cursor], axis=1)
    log.debug(
        "[Extract/%s] Norma L2 — media: %.2f | min: %.2f | max: %.2f",
        desc,
        norms.mean(),
        norms.min(),
        norms.max(),
    )
    if norms.mean() < MIN_L2_NORM:
        log.warning(
            "[Extract/%s] Norma L2 media muy baja (%.3f < %.1f). "
            "¿El backbone devuelve el CLS token correctamente?",
            desc,
            norms.mean(),
            MIN_L2_NORM,
        )

    return Z[:cursor], y_expert[:cursor], img_names
