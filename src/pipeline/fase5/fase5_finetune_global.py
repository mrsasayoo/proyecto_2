"""
Paso 8 — Fine-tuning por etapas del sistema MoE completo.

Etapas:
  Stage 1: Router solo (backbone + expertos congelados). LR=1e-3. ~50 epocas.
  Stage 2: Router + cabezas clasificadoras. LR=1e-4. ~30 epocas.
  Stage 3: Fine-tuning global (todo descongelado). LR expertos=1e-6, router=1e-4. 7-10 epocas.

Prerrequisitos (checkpoints de Fases 2-4):
  - checkpoints/expert_0[0-5]_*/expert*_best.pt  (Pasos 5.1-5.6)
  - ablation output: best_router_*.{pt,joblib,faiss}  (Pasos 6-7)

PROHIBIDO ENTRENAR — usar --dry-run para verificacion.

Uso:
    # Dry-run completo (verifica las 3 etapas)
    python src/pipeline/fase5/fase5_finetune_global.py --dry-run

    # Dry-run de una etapa especifica
    python src/pipeline/fase5/fase5_finetune_global.py --stage 1 --dry-run

    # Entrenamiento real (NO ejecutar aun — solo verificar con dry-run)
    python src/pipeline/fase5/fase5_finetune_global.py --stage all
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase5.fase5_config import (
    STAGE1_LR_ROUTER,
    STAGE1_EPOCHS,
    STAGE1_PATIENCE,
    STAGE1_BATCH_SIZE,
    STAGE2_LR_ROUTER,
    STAGE2_LR_HEADS,
    STAGE2_EPOCHS,
    STAGE2_PATIENCE,
    STAGE3_LR_ROUTER,
    STAGE3_LR_EXPERTS,
    STAGE3_LR_BACKBONE,
    STAGE3_EPOCHS_MAX,
    STAGE3_PATIENCE,
    RESET_OPTIMIZER_BETWEEN_STAGES,
    ALPHA_L_AUX,
    BETA_L_ERROR,
    GAMMA_L_BALANCE,
    FP16_ENABLED,
    ACCUMULATION_STEPS,
    MIXED_BATCH_SIZE,
    CHECKPOINT_DIR,
    EXPERT_HEAD_PREFIXES,
    FASE5_CONFIG_SUMMARY,
)
from fase5.freeze_utils import (
    apply_stage1_freeze,
    apply_stage2_freeze,
    apply_stage3_freeze,
    log_freeze_state,
    count_trainable,
    count_frozen,
)
from fase5.moe_model import MoESystem, build_moe_system_dry_run
from fase5.dataloader_mixed import get_mixed_dataloader
from config import N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fase5")

# ── Constantes ─────────────────────────────────────────────────────────
_SEED = 42


def set_seed(seed: int = _SEED) -> None:
    """Fija todas las semillas para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_param_groups_stage1(moe: MoESystem) -> list[dict]:
    """Param groups para Stage 1: solo el router."""
    return [{"params": list(moe.router.parameters()), "lr": STAGE1_LR_ROUTER}]


def _build_param_groups_stage2(moe: MoESystem) -> list[dict]:
    """Param groups para Stage 2: router + cabezas de expertos."""
    groups = [{"params": list(moe.router.parameters()), "lr": STAGE2_LR_ROUTER}]

    # Recolectar parametros entrenables de cada experto (solo cabezas)
    expert_head_params = []
    for expert in moe.experts:
        for param in expert.parameters():
            if param.requires_grad:
                expert_head_params.append(param)

    if expert_head_params:
        groups.append({"params": expert_head_params, "lr": STAGE2_LR_HEADS})

    return groups


def _build_param_groups_stage3(moe: MoESystem) -> list[dict]:
    """
    Param groups para Stage 3: router + expertos + backbone, con LRs diferenciados.

    - Router: STAGE3_LR_ROUTER (1e-4)
    - Expertos: STAGE3_LR_EXPERTS (1e-6)
    - Backbone: STAGE3_LR_BACKBONE (1e-6)
    """
    groups = [{"params": list(moe.router.parameters()), "lr": STAGE3_LR_ROUTER}]

    expert_params = []
    for expert in moe.experts:
        expert_params.extend(list(expert.parameters()))

    if expert_params:
        groups.append({"params": expert_params, "lr": STAGE3_LR_EXPERTS})

    if moe.backbone is not None:
        backbone_params = list(moe.backbone.parameters())
        if backbone_params:
            groups.append({"params": backbone_params, "lr": STAGE3_LR_BACKBONE})

    return groups


def _compute_l_aux(gates: torch.Tensor) -> torch.Tensor:
    """
    Calcula la Auxiliary Loss del Switch Transformer con penalizacion de balance.

    L_aux = alpha * N * sum(f_i * P_i) + gamma * sum_k max(0, f_k - 2/K)^2

    donde:
        f_i = fraccion de muestras ruteadas al experto i
        P_i = probabilidad media del router para el experto i
        threshold = 2/K: ningun experto debe recibir mas del 40% de samples

    Args:
        gates: probabilidades del router [B, N_EXPERTS_DOMAIN].
               Se aplica softmax defensivo internamente.

    Returns:
        L_aux escalar (incluye termino base + penalizacion de umbral)
    """
    # Asegurar que gates sean probabilidades aunque LinearGatingHead ya aplique softmax
    gates = torch.softmax(gates, dim=-1)

    device = gates.device
    n_experts = gates.shape[1]  # N_EXPERTS_DOMAIN = 5
    batch_size = gates.shape[0]

    # f_i: fraccion de muestras ruteadas a cada experto (decision dura)
    routed = gates.detach().argmax(dim=-1)
    f_i = torch.zeros(n_experts, device=device)
    for eid in range(n_experts):
        f_i[eid] = (routed == eid).float().sum() / batch_size

    # P_i: probabilidad media del router por experto
    P_i = gates.mean(dim=0)  # [N_EXPERTS_DOMAIN]

    # Termino base Switch Transformer
    L_base = ALPHA_L_AUX * n_experts * (f_i * P_i).sum()

    # Penalizacion de umbral: ningun experto debe recibir mas de 2/K samples
    threshold = 2.0 / n_experts
    excess = torch.clamp(f_i - threshold, min=0.0)
    L_balance = GAMMA_L_BALANCE * (excess**2).sum()

    return L_base + L_balance


def _compute_l_error(gates: torch.Tensor, expert_ids: list[int]) -> torch.Tensor:
    # TODO: implementar en producción
    """
    Calcula L_error: penaliza enviar imagenes validas al CAE (Expert 5).

    En dry-run con router de 5 salidas (sin slot OOD), L_error es 0.
    En produccion con 6 salidas, seria la probabilidad media del slot OOD
    para muestras que NO son OOD.

    Args:
        gates: probabilidades del router [B, N]
        expert_ids: expert_ids del batch

    Returns:
        L_error escalar
    """
    # En el dry-run el router tiene N_EXPERTS_DOMAIN=5 salidas, sin slot OOD
    # L_error se calcula como 0 (placeholder para produccion)
    return torch.tensor(0.0, device=gates.device, requires_grad=False)


def _dry_run_stage(
    moe: MoESystem,
    stage: int,
    device: torch.device,
) -> dict[str, int]:
    """
    Ejecuta el dry-run de una etapa: aplica freeze, forward+backward sintetico.

    Returns:
        dict con 'trainable' y 'frozen' para esta etapa
    """
    log.info("")
    log.info("=" * 60)
    log.info("  DRY-RUN Stage %d", stage)
    log.info("=" * 60)

    # --- Aplicar freeze ---
    experts_list = list(moe.experts)
    if stage == 1:
        apply_stage1_freeze(moe.router, experts_list, moe.backbone)
    elif stage == 2:
        apply_stage2_freeze(
            moe.router, experts_list, EXPERT_HEAD_PREFIXES, moe.backbone
        )
    elif stage == 3:
        apply_stage3_freeze(moe.router, experts_list, moe.backbone)

    # --- Loguear estado de congelamiento ---
    modules = {}
    for i in range(6):
        modules[f"Expert{i}"] = moe.experts[i]
    modules["Router"] = moe.router
    if moe.backbone is not None:
        modules["Backbone"] = moe.backbone
    log_freeze_state(modules, stage)

    n_trainable = count_trainable(moe)
    n_frozen = count_frozen(moe)

    # --- Construir param groups ---
    if stage == 1:
        param_groups = _build_param_groups_stage1(moe)
    elif stage == 2:
        param_groups = _build_param_groups_stage2(moe)
    else:
        param_groups = _build_param_groups_stage3(moe)

    log.info("  Param groups: %d grupo(s)", len(param_groups))
    for i, pg in enumerate(param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        n_req_grad = sum(p.numel() for p in pg["params"] if p.requires_grad)
        log.info(
            "    Grupo %d: LR=%g | %s params totales | %s requires_grad",
            i,
            pg["lr"],
            f"{n_params:,}",
            f"{n_req_grad:,}",
        )

    # --- Forward + backward sintetico para cada tipo de experto ---
    # Agrupar: 2D experts (0,1,2,5) y 3D experts (3,4)
    expert_configs = {
        0: {"shape": (2, 3, 224, 224), "name": "Expert0/ConvNeXt"},
        1: {"shape": (2, 3, 224, 224), "name": "Expert1/EfficientNet"},
        2: {"shape": (2, 3, 224, 224), "name": "Expert2/EfficientNetB0"},
        3: {"shape": (2, 1, 64, 64, 64), "name": "Expert3/MC3-18"},
        4: {"shape": (1, 1, 64, 64, 64), "name": "Expert4/Swin3D"},
        5: {"shape": (2, 3, 224, 224), "name": "Expert5/CAE"},
    }

    moe.train()
    for eid, cfg in expert_configs.items():
        x = torch.randn(*cfg["shape"], device=device)

        try:
            out = moe(x, expert_id=eid)
            logits = out["logits"]
            gates = out["gates"]

            # Calcular loss sintetica
            if eid == 0:
                # Multilabel: BCEWithLogitsLoss
                target = torch.zeros_like(logits)
                loss_task = nn.functional.binary_cross_entropy_with_logits(
                    logits, target
                )
            elif eid == 5:
                # CAE: MSE reconstruccion
                recon = out["recon"]
                loss_task = nn.functional.mse_loss(recon, x)
            else:
                # Multiclase: CrossEntropyLoss
                n_classes = logits.shape[1]
                target = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
                loss_task = nn.functional.cross_entropy(logits, target)

            # L_aux
            l_aux = _compute_l_aux(gates)

            # L_total
            # NOTE: l_aux already includes ALPHA_L_AUX and GAMMA_L_BALANCE
            # internally. l_error desconectado (placeholder, ver TODO).
            l_total = loss_task + l_aux

            # Backward (sin optimizer.step)
            l_total.backward()

            log.info(
                "  [%s] input=%s | output=%s | L_task=%.4f | L_aux=%.4f | L_total=%.4f",
                cfg["name"],
                list(x.shape),
                list(logits.shape),
                loss_task.item(),
                l_aux.item(),
                l_total.item(),
            )

        except Exception as e:
            log.error("  [%s] FALLO en forward/backward: %s", cfg["name"], e)
            raise

        # Limpiar gradientes para el siguiente experto
        moe.zero_grad()

    return {"trainable": n_trainable, "frozen": n_frozen}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paso 8 — Fine-tuning por etapas del sistema MoE"
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Etapa a ejecutar: 1, 2, 3, o all (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifica el pipeline sin entrenar (datos sinteticos)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta al checkpoint de una etapa anterior para continuar",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MIXED_BATCH_SIZE,
        help=f"Batch size del DataLoader mixto (default: {MIXED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="./embeddings/vit_tiny_patch16_224",
        help="Directorio de embeddings para Stage 1 (router sobre embeddings)",
    )

    args = parser.parse_args()

    if not args.dry_run:
        log.error(
            "PROHIBIDO ENTRENAR sin --dry-run. "
            "Este paso solo se verifica con --dry-run."
        )
        sys.exit(1)

    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[Fase5] Dispositivo: %s", device)
    log.info("[Fase5] Config: %s", FASE5_CONFIG_SUMMARY)
    log.info("[Fase5] Stage solicitado: %s", args.stage)

    # ── Construir sistema MoE sintetico ────────────────────────────
    log.info("")
    log.info("[Fase5] Construyendo MoE sintetico (dry-run, sin checkpoints)...")
    moe = build_moe_system_dry_run(d_model=192)
    moe = moe.to(device)

    # Conteo total de parametros
    total_params = sum(p.numel() for p in moe.parameters())
    log.info("[Fase5] Total parametros del sistema MoE: %s", f"{total_params:,}")

    # ── Ejecutar dry-run por etapas ────────────────────────────────
    stages_to_run = [1, 2, 3] if args.stage == "all" else [int(args.stage)]

    stage_stats = {}
    for stage in stages_to_run:
        stats = _dry_run_stage(moe, stage, device)
        stage_stats[stage] = stats

    # ── Banner final ───────────────────────────────────────────────
    # Completar stats para etapas no ejecutadas
    for s in [1, 2, 3]:
        if s not in stage_stats:
            stage_stats[s] = {"trainable": "N/A", "frozen": "N/A"}

    def _fmt(val: int | str) -> str:
        if isinstance(val, int):
            return f"{val:,}"
        return str(val)

    log.info("")
    log.info("=" * 70)
    log.info("  DRY-RUN COMPLETADO — Paso 8: Fine-tuning por Etapas")
    log.info("  Stage solicitado: %s", args.stage)
    log.info("  Sistema MoE:")
    log.info("    Expertos: %d (%d dominio + 1 CAE)", N_EXPERTS_TOTAL, N_EXPERTS_DOMAIN)
    log.info("    Router:   LinearGatingHead")
    log.info("    Backbone: (no usado como modulo separado)")
    log.info("")
    log.info("  Parametros por etapa:")
    log.info(
        "    Stage 1: %s entrenables / %s congelados",
        _fmt(stage_stats[1]["trainable"]),
        _fmt(stage_stats[1]["frozen"]),
    )
    log.info(
        "    Stage 2: %s entrenables / %s congelados",
        _fmt(stage_stats[2]["trainable"]),
        _fmt(stage_stats[2]["frozen"]),
    )
    log.info(
        "    Stage 3: %s entrenables / %s congelados",
        _fmt(stage_stats[3]["trainable"]),
        _fmt(stage_stats[3]["frozen"]),
    )
    log.info("=" * 70)
    log.info(
        "[DRY-RUN] Pipeline verificado. Ejecuta sin --dry-run para fine-tuning real."
    )


if __name__ == "__main__":
    main()
