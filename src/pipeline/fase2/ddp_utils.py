"""
Utilidades para DistributedDataParallel (DDP) — Sistema MoE médico, Fase 2.

Módulo reutilizable que encapsula toda la lógica DDP para los 5 scripts de
entrenamiento de expertos. Diseñado para funcionar tanto en modo single-GPU
(fallback transparente) como en multi-GPU con torchrun.

Compatibilidad: PyTorch 2.3.0 + CUDA 12.1 (NO usa APIs de PyTorch >= 2.4).

Uso típico:

    from fase2.ddp_utils import (
        setup_ddp, cleanup_ddp, wrap_model_ddp,
        get_ddp_dataloader, is_main_process, save_checkpoint_ddp,
        get_rank, get_world_size, ddp_log,
    )

    setup_ddp()
    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    model = wrap_model_ddp(model, device)
    loader = get_ddp_dataloader(dataset, batch_size=32, shuffle=True)
    # ... training loop ...
    save_checkpoint_ddp(state_dict, path)
    cleanup_ddp()
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

log = logging.getLogger("ddp_utils")


# ─────────────────────────────────────────────────────────────────────────────
# Consulta de estado DDP
# ─────────────────────────────────────────────────────────────────────────────


def is_ddp_initialized() -> bool:
    """Devuelve True si el process group de DDP está inicializado."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Devuelve el rank local del proceso. 0 si DDP no está activo."""
    if is_ddp_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Devuelve el world_size (número de procesos). 1 si DDP no está activo."""
    if is_ddp_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Devuelve True solo en el proceso rank=0 (o si DDP no está activo).

    Usar para condicionar logging, checkpointing, y escritura de métricas.
    """
    return get_rank() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Setup y cleanup del process group
# ─────────────────────────────────────────────────────────────────────────────


def setup_ddp(backend: str = "auto") -> None:
    """Inicializa el process group de DDP si las variables de entorno lo indican.

    torchrun configura automáticamente RANK, WORLD_SIZE, LOCAL_RANK y
    MASTER_ADDR/MASTER_PORT. Si estas variables no están presentes, la
    función no hace nada (modo single-GPU transparente).

    Args:
        backend: backend de comunicación. "nccl" para GPU, "gloo" para CPU,
                 o "auto" para detectar automáticamente.
    """
    # torchrun define RANK y WORLD_SIZE; si no están, no estamos en modo DDP
    if "RANK" not in os.environ:
        log.info(
            "[DDP] Variables de entorno RANK/WORLD_SIZE no encontradas. "
            "Ejecutando en modo single-process (sin DDP)."
        )
        return

    # Auto-detectar backend si se especifica "auto"
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Asignar GPU al proceso
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # ── Guardia global de logging: silenciar ranks != 0 ────────────────
    # Los scripts configuran logging.basicConfig(level=INFO) antes de llamar
    # a setup_ddp(). Sin esta guardia, todos los ranks imprimen logs INFO,
    # duplicando la salida ×world_size. Elevar el nivel del root logger a
    # WARNING en workers deja pasar solo advertencias y errores reales.
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
        # Suprimir también warnings.warn() de PyTorch (CUDNN, scheduler,
        # torchvision) en workers. El módulo `warnings` de Python es
        # completamente independiente del módulo `logging`, así que la
        # guardia de logging de arriba no los cubre.
        warnings.filterwarnings("ignore")

    if rank == 0:
        log.info(
            f"[DDP] Process group inicializado: "
            f"backend={backend}, world_size={world_size}, "
            f"rank={rank}, local_rank={local_rank}"
        )


def cleanup_ddp() -> None:
    """Destruye el process group de DDP si está activo."""
    if is_ddp_initialized():
        dist.destroy_process_group()
        log.info("[DDP] Process group destruido.")


# ─────────────────────────────────────────────────────────────────────────────
# Wrapping de modelo
# ─────────────────────────────────────────────────────────────────────────────


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Envuelve el modelo en DDP si world_size > 1, si no lo devuelve tal cual.

    Args:
        model: modelo ya movido al dispositivo correcto (.to(device)).
        device: dispositivo del proceso actual (e.g., cuda:0, cuda:1).
        find_unused_parameters: True si el modelo tiene parámetros que no
            participan en todos los forwards (e.g., LP con backbone congelado).
            Tiene costo de rendimiento, usar solo cuando sea necesario.

    Returns:
        Modelo envuelto en DDP o el modelo original si no aplica.
    """
    if not is_ddp_initialized() or get_world_size() <= 1:
        return model

    device_ids = [device.index] if device.type == "cuda" else None
    wrapped = DDP(
        model,
        device_ids=device_ids,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )

    if is_main_process():
        log.info(
            f"[DDP] Modelo envuelto en DistributedDataParallel "
            f"(device_ids={device_ids}, "
            f"find_unused={find_unused_parameters})"
        )

    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader con DistributedSampler
# ─────────────────────────────────────────────────────────────────────────────


def get_ddp_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Crea un DataLoader compatible con DDP.

    Si DDP está activo (world_size > 1), usa DistributedSampler para particionar
    el dataset entre los procesos. Si no, crea un DataLoader estándar.

    Args:
        dataset: dataset de PyTorch.
        batch_size: batch size **por GPU**. El batch total será
            batch_size * world_size.
        shuffle: mezclar datos (manejado por DistributedSampler si DDP activo).
        drop_last: descartar último batch incompleto.
        num_workers: workers para carga de datos. Default: os.cpu_count() // 2
            dividido entre world_size para no saturar.
        pin_memory: fijar memoria en RAM para transferencia rápida a GPU.
            Default: True si hay CUDA.
        persistent_workers: mantener workers entre epochs.
            Default: True si num_workers > 0.

    Returns:
        Tupla (DataLoader, sampler). sampler es None si DDP no está activo.
        El sampler se devuelve para poder llamar sampler.set_epoch(epoch)
        en cada época.
    """
    world_size = get_world_size()

    # Defaults inteligentes
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = max(1, cpu_count // (2 * world_size))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    sampler: DistributedSampler | None = None

    if is_ddp_initialized() and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Cuando se usa DistributedSampler, shuffle del DataLoader debe ser False
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # sampler maneja el shuffle
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **(
                {"prefetch_factor": prefetch_factor}
                if prefetch_factor is not None
                else {}
            ),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **(
                {"prefetch_factor": prefetch_factor}
                if prefetch_factor is not None
                else {}
            ),
        )

    return loader, sampler


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing seguro con DDP
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint_ddp(state: dict[str, Any], path: str | Path) -> None:
    """Guarda un checkpoint de forma segura en modo DDP.

    Solo el proceso rank=0 escribe en disco. Incluye una barrera
    (dist.barrier) para sincronizar los demás procesos.

    Args:
        state: diccionario con el estado a guardar (model_state_dict, etc.).
        path: ruta del archivo .pt de destino.
    """
    path = Path(path)

    if is_main_process():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        log.info(f"[DDP] Checkpoint guardado: {path}")

    # Barrera para que los demás procesos esperen a que rank=0 termine
    if is_ddp_initialized():
        dist.barrier()


def load_checkpoint_ddp(
    path: str | Path,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any] | None:
    """Carga un checkpoint de forma segura para DDP.

    Todos los procesos cargan el checkpoint, pero se usa map_location
    para evitar conflictos de dispositivo.

    Args:
        path: ruta del checkpoint .pt.
        map_location: dispositivo destino para torch.load.

    Returns:
        Diccionario del checkpoint o None si no existe.
    """
    path = Path(path)
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de logging para DDP
# ─────────────────────────────────────────────────────────────────────────────


def ddp_log(logger: logging.Logger, level: int, msg: str) -> None:
    """Log condicional: solo imprime en rank=0.

    Args:
        logger: instancia de logging.Logger.
        level: nivel de logging (logging.INFO, logging.WARNING, etc.).
        msg: mensaje a loggear.
    """
    if is_main_process():
        logger.log(level, msg)


def get_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Obtiene el state_dict del modelo, desempaquetando DDP si es necesario.

    DDP envuelve el modelo en model.module. Esta función devuelve siempre
    el state_dict del modelo base, compatible con carga sin DDP.

    Args:
        model: modelo (posiblemente envuelto en DDP).

    Returns:
        state_dict del modelo base (sin prefijos 'module.').
    """
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Obtiene el modelo base, desempaquetando DDP si es necesario.

    Args:
        model: modelo (posiblemente envuelto en DDP).

    Returns:
        Modelo base sin wrapper DDP.
    """
    if isinstance(model, DDP):
        return model.module
    return model
