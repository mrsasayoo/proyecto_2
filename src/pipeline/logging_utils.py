"""
Configuración de logging para el pipeline MoE.

Escribe simultáneamente a consola (INFO) y a archivo (DEBUG).
"""

import os
import logging
from pathlib import Path


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configura el sistema de logging del pipeline.
    Escribe simultáneamente a consola (INFO) y a archivo (DEBUG).

    Niveles usados en este script:
      DEBUG   → detalles internos útiles para depuración profunda
      INFO    → progreso normal del pipeline
      WARNING → situación anómala que no detiene la ejecución
      ERROR   → fallo grave que probablemente corrompe los embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / "fase0.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),   # consola
        ]
    )
    # La consola solo muestra INFO y superior — el archivo guarda todo
    logging.getLogger().handlers[1].setLevel(logging.INFO)

    log = logging.getLogger("fase0")
    log.info(f"Log iniciado → {log_path}")
    return log
