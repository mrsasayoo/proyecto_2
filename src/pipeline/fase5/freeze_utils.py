"""
Utilidades de congelamiento para fine-tuning por etapas (Paso 8).

Opera externamente sobre nn.Module — no requiere modificar los wrappers
de expertos existentes en fase2/models/ ni fase3/models/.

Funciones principales:
  - freeze_module / unfreeze_module: congela/descongela todos los params
  - freeze_except_head: congela todo excepto la cabeza clasificadora
  - apply_stage1_freeze / apply_stage2_freeze / apply_stage3_freeze:
    aplican el patron de congelamiento de cada etapa del fine-tuning
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch.nn as nn

log = logging.getLogger("fase5")


def freeze_module(module: nn.Module, name: str = "") -> int:
    """
    Congela todos los parametros de un modulo (requires_grad=False).

    Args:
        module: modulo PyTorch a congelar.
        name: nombre descriptivo para logging.

    Returns:
        Numero de parametros congelados.
    """
    n_frozen = 0
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad = False
            n_frozen += param.numel()
    if name:
        log.debug("[freeze] %s: %d params congelados", name, n_frozen)
    return n_frozen


def unfreeze_module(module: nn.Module, name: str = "") -> int:
    """
    Descongela todos los parametros de un modulo (requires_grad=True).

    Args:
        module: modulo PyTorch a descongelar.
        name: nombre descriptivo para logging.

    Returns:
        Numero de parametros descongelados.
    """
    n_unfrozen = 0
    for param in module.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            n_unfrozen += param.numel()
    if name:
        log.debug("[unfreeze] %s: %d params descongelados", name, n_unfrozen)
    return n_unfrozen


def freeze_except_head(
    expert_module: nn.Module,
    head_prefixes: list[str],
    expert_name: str = "",
) -> dict[str, int]:
    """
    Congela todo el modulo excepto los sub-modulos cuyo nombre de parametro
    comienza con alguno de los prefijos en head_prefixes.

    Ejemplo para Expert0 (ConvNeXt): head_prefixes=["model.classifier"]
      -> congela self.model.features, self.model.avgpool, etc.
      -> mantiene descongelado self.model.classifier

    Args:
        expert_module: el modulo experto completo.
        head_prefixes: lista de prefijos de nombres de parametros que
            corresponden a la cabeza clasificadora.
        expert_name: nombre descriptivo para logging.

    Returns:
        dict con conteos: {'frozen': N, 'trainable': M}
    """
    frozen_count = 0
    trainable_count = 0

    for param_name, param in expert_module.named_parameters():
        is_head = any(param_name.startswith(prefix) for prefix in head_prefixes)
        if is_head:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    if expert_name:
        log.debug(
            "[freeze_except_head] %s: %d frozen / %d trainable (prefixes=%s)",
            expert_name,
            frozen_count,
            trainable_count,
            head_prefixes,
        )

    return {"frozen": frozen_count, "trainable": trainable_count}


def count_trainable(module: nn.Module) -> int:
    """Cuenta parametros entrenables (requires_grad=True)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_frozen(module: nn.Module) -> int:
    """Cuenta parametros congelados (requires_grad=False)."""
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)


def log_freeze_state(modules: dict[str, nn.Module], stage: int) -> None:
    """
    Loguea el estado de congelamiento de todos los modulos del sistema MoE.

    Formato: [Stage N] ComponentName: X trainable / Y frozen params

    Args:
        modules: dict nombre -> modulo (ej: {"Expert0": model, "Router": router})
        stage: numero de etapa (1, 2 o 3)
    """
    log.info("--- Estado de congelamiento [Stage %d] ---", stage)
    total_trainable = 0
    total_frozen = 0
    for name, module in modules.items():
        t = count_trainable(module)
        f = count_frozen(module)
        total_trainable += t
        total_frozen += f
        log.info(
            "  [Stage %d] %s: %s trainable / %s frozen",
            stage,
            name,
            f"{t:,}",
            f"{f:,}",
        )
    log.info(
        "  [Stage %d] TOTAL: %s trainable / %s frozen",
        stage,
        f"{total_trainable:,}",
        f"{total_frozen:,}",
    )


def apply_stage1_freeze(
    router: nn.Module,
    experts: list[nn.Module],
    backbone: nn.Module | None = None,
) -> None:
    """
    Stage 1: congela TODO excepto el router.

    - backbone: congelado (si se pasa)
    - experts[0..5]: todos congelados
    - router: descongelado

    Args:
        router: modulo del router (LinearGatingHead).
        experts: lista de 6 modulos expertos.
        backbone: modulo backbone opcional (puede no existir como modulo separado).
    """
    # Congelar backbone si existe
    if backbone is not None:
        freeze_module(backbone, name="backbone")

    # Congelar todos los expertos
    for i, expert in enumerate(experts):
        freeze_module(expert, name=f"Expert{i}")

    # Descongelar router
    unfreeze_module(router, name="Router")

    log.info("[Stage 1] Freeze aplicado: solo Router entrenable")


def apply_stage2_freeze(
    router: nn.Module,
    experts: list[nn.Module],
    head_prefixes_map: dict[int, list[str]],
    backbone: nn.Module | None = None,
) -> None:
    """
    Stage 2: congela backbone + capas conv/feature de expertos;
    descongela router + cabezas clasificadoras.

    - backbone: congelado
    - experts: congelados excepto la cabeza clasificadora
    - router: descongelado

    Args:
        router: modulo del router.
        experts: lista de 6 modulos expertos.
        head_prefixes_map: dict {expert_id: [prefijos de cabeza]} de fase5_config.
        backbone: modulo backbone opcional.
    """
    # Congelar backbone si existe
    if backbone is not None:
        freeze_module(backbone, name="backbone")

    # Congelar expertos excepto cabezas
    for i, expert in enumerate(experts):
        prefixes = head_prefixes_map.get(i, [])
        if prefixes:
            freeze_except_head(expert, prefixes, expert_name=f"Expert{i}")
        else:
            freeze_module(expert, name=f"Expert{i}")

    # Descongelar router
    unfreeze_module(router, name="Router")

    log.info("[Stage 2] Freeze aplicado: Router + cabezas entrenables")


def apply_stage3_freeze(
    router: nn.Module,
    experts: list[nn.Module],
    backbone: nn.Module | None = None,
) -> None:
    """
    Stage 3: descongela TODO — fine-tuning global.

    Args:
        router: modulo del router.
        experts: lista de 6 modulos expertos.
        backbone: modulo backbone opcional.
    """
    # Descongelar backbone si existe
    if backbone is not None:
        unfreeze_module(backbone, name="backbone")

    # Descongelar todos los expertos
    for i, expert in enumerate(experts):
        unfreeze_module(expert, name=f"Expert{i}")

    # Descongelar router
    unfreeze_module(router, name="Router")

    log.info("[Stage 3] Freeze aplicado: TODO descongelado (fine-tuning global)")
