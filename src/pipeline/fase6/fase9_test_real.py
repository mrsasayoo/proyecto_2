"""
fase9_test_real.py
------------------
Orquestador principal del Paso 9: Prueba con datos reales.

Evalúa el sistema MoE sobre los test sets de los 5 datasets de dominio,
calcula métricas finales, genera Grad-CAM y reporte OOD.

Uso:
    python src/pipeline/fase6/fase9_test_real.py --dry-run
    python src/pipeline/fase6/fase9_test_real.py --checkpoint checkpoints/fase5/moe_final.pt
    python src/pipeline/fase6/fase9_test_real.py --experts 0,1,2 --no-gradcam
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Asegurar que el proyecto raíz esté en el path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Asegurar que pipeline/ y subdirectorios del router estén en sys.path
# para que linear.py pueda importar sus dependencias con bare imports
_PIPELINE_DIR = str(_PROJECT_ROOT / "src" / "pipeline")
_FASE2_DIR = str(_PROJECT_ROOT / "src" / "pipeline" / "fase2")
_ROUTERS_DIR = str(_PROJECT_ROOT / "src" / "pipeline" / "fase2" / "routers")
for _d in (_PIPELINE_DIR, _FASE2_DIR, _ROUTERS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from src.pipeline.logging_utils import setup_logging
from src.pipeline.fase6.fase6_config import (
    MOE_CHECKPOINT,
    RESULTS_DIR,
    FIGURES_DIR,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EXPERT_NAMES,
    N_EXPERTS_DOMAIN,
)
from src.pipeline.fase6.inference_engine import InferenceEngine
from src.pipeline.fase6.test_evaluator import TestEvaluator
from src.pipeline.fase6.ood_detector import OODDetector
from src.pipeline.fase6.gradcam_heatmap import generate_gradcam_samples

logger = logging.getLogger(__name__)


# === Configuración de datasets por experto ===
# num_classes y is_multilabel para cada experto de dominio.
# Ajustado a los constructores reales de cada experto:
#   Expert 0 (ConvNeXt): num_classes=14 (NIH ChestXray14, multilabel)
#   Expert 1 (EfficientNet): num_classes=9 (ISIC 2019, 8 clases + 1 UNK)
#   Expert 2 (VGG16BN): num_classes=3 (OA Knee, ordinal KL0/KL1-2/KL3-4)
#   Expert 3 (MC3-18): num_classes=2 (LUNA16, binario)
#   Expert 4 (Swin3D): num_classes=2 (Pancreas PDAC, binario)
EXPERT_DATASET_CONFIG = {
    0: {"name": "chest", "num_classes": 14, "is_multilabel": True},
    1: {"name": "isic", "num_classes": 9, "is_multilabel": False},
    2: {"name": "oa_knee", "num_classes": 3, "is_multilabel": False},
    3: {"name": "luna", "num_classes": 2, "is_multilabel": False},
    4: {"name": "pancreas", "num_classes": 2, "is_multilabel": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 9 — Prueba con datos reales del sistema MoE"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validar setup sin ejecutar inferencia real (EXIT 0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MOE_CHECKPOINT,
        help=f"Ruta al checkpoint MoESystem (default: {MOE_CHECKPOINT})",
    )
    parser.add_argument(
        "--experts",
        type=str,
        default="0,1,2,3,4",
        help="Expertos a evaluar (comma-separated, default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Deshabilitar generación de Grad-CAM",
    )
    parser.add_argument(
        "--no-ood",
        action="store_true",
        help="Deshabilitar evaluación OOD",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda / cpu (default: auto)",
    )
    return parser.parse_args()


def build_moe_system(
    checkpoint_path: str,
    device: torch.device,
    dry_run: bool,
) -> nn.Module:
    """
    Instancia el MoESystem con todos los expertos + router y carga checkpoint.

    MoESystem.__init__ signature (from moe_model.py):
        experts: nn.ModuleList  (6 expertos, indices 0-5)
        router:  nn.Module      (LinearGatingHead)
        backbone: Optional[nn.Module] = None

    En dry-run, instancia sin cargar pesos reales si el checkpoint no existe.
    """
    from src.pipeline.fase5.moe_model import MoESystem
    from src.pipeline.fase2.models.expert1_convnext import Expert1ConvNeXtTiny
    from src.pipeline.fase2.models.expert2_efficientnet import Expert2EfficientNetB3
    from src.pipeline.fase2.models.expert_oa_vgg16bn import ExpertOAVGG16BN
    from src.pipeline.fase2.models.expert3_r3d18 import Expert3MC318
    from src.pipeline.fase2.models.expert4_swin3d import ExpertPancreasSwin3D
    from src.pipeline.fase3.models.expert5_cae import ConvAutoEncoder
    from linear import LinearGatingHead

    logger.info("Instanciando expertos del MoESystem...")

    # Instanciar los 6 expertos con los mismos parámetros que
    # build_moe_system_dry_run() en moe_model.py
    experts = nn.ModuleList(
        [
            Expert1ConvNeXtTiny(
                fc_dropout_p=0.3, num_classes=14
            ),  # Expert 0: NIH Chest
            Expert2EfficientNetB3(
                fc_dropout_p=0.3, num_classes=9
            ),  # Expert 1: ISIC (8+UNK)
            ExpertOAVGG16BN(num_classes=3, dropout=0.5),  # Expert 2: OA Knee
            Expert3MC318(  # Expert 3: LUNA16
                spatial_dropout_p=0.15, fc_dropout_p=0.4, num_classes=2
            ),
            ExpertPancreasSwin3D(in_channels=1, num_classes=2),  # Expert 4: Páncreas
            ConvAutoEncoder(
                in_channels=3, latent_dim=512, img_size=224
            ),  # Expert 5: CAE
        ]
    )

    # Instanciar router: d_model=192 (ViT-Tiny), n_experts=5 (dominio)
    router = LinearGatingHead(d_model=192, n_experts=N_EXPERTS_DOMAIN)

    # Crear MoESystem con la signature correcta: experts, router, backbone
    moe = MoESystem(experts=experts, router=router, backbone=None)

    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        logger.info(f"Cargando checkpoint desde {checkpoint_path}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            moe.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            moe.load_state_dict(state["state_dict"])
        else:
            moe.load_state_dict(state)
        logger.info("Checkpoint cargado exitosamente")
    else:
        if dry_run:
            logger.warning(
                f"[DRY-RUN] Checkpoint no encontrado en {checkpoint_path}. "
                "Usando pesos aleatorios — normal en dry-run."
            )
        else:
            logger.error(
                f"Checkpoint NO encontrado: {checkpoint_path}. "
                "El entrenamiento (Paso 8) debe completarse primero."
            )

    moe = moe.to(device)
    moe.eval()
    return moe


def build_test_dataloaders(experts_to_eval: list, dry_run: bool) -> dict:
    """
    Construye DataLoaders de test para los expertos especificados.
    En dry-run retorna dict vacío.
    """
    if dry_run:
        logger.info("[DRY-RUN] Skipping DataLoader construction")
        return {}

    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    # Transforms para cada tipo de experto
    transform_2d = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataloaders = {}

    for expert_idx in experts_to_eval:
        try:
            if expert_idx == 0:
                from src.pipeline.datasets.chest import ChestXray14Dataset

                ds = ChestXray14Dataset(split="test", transform=transform_2d)
            elif expert_idx == 1:
                from src.pipeline.datasets.isic import ISICDataset

                ds = ISICDataset(split="test", transform=transform_2d)
            elif expert_idx == 2:
                from src.pipeline.datasets.osteoarthritis import OAKneeDataset

                ds = OAKneeDataset(split="test", transform=transform_2d)
            elif expert_idx == 3:
                from src.pipeline.datasets.luna import LUNADataset

                ds = LUNADataset(split="test")
            elif expert_idx == 4:
                from src.pipeline.datasets.pancreas import PancreasDataset

                ds = PancreasDataset(split="test")
            else:
                logger.warning(f"Expert index {expert_idx} not recognized, skipping")
                continue

            dataloaders[expert_idx] = DataLoader(
                ds,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=EVAL_NUM_WORKERS,
                pin_memory=True,
            )
            logger.info(
                f"DataLoader built for expert {expert_idx} "
                f"({EXPERT_NAMES[expert_idx]}): {len(ds)} samples"
            )

        except Exception as e:
            logger.warning(f"Failed to build DataLoader for expert {expert_idx}: {e}")

    return dataloaders


def run_evaluation(args: argparse.Namespace) -> int:
    """
    Función principal de evaluación.
    Returns: exit code (0 = OK, 1 = error)
    """
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    experts_to_eval = [int(e.strip()) for e in args.experts.split(",")]
    logger.info(f"Experts to evaluate: {experts_to_eval}")

    # Crear directorios de salida
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Construir MoESystem
    moe = build_moe_system(args.checkpoint, device, args.dry_run)

    # 2. Crear InferenceEngine
    engine = InferenceEngine(
        moe_system=moe,
        device=device,
    )
    logger.info("InferenceEngine creado")

    # 3. Construir DataLoaders de test
    test_dataloaders = build_test_dataloaders(experts_to_eval, args.dry_run)

    # 4. Evaluador batch
    evaluator = TestEvaluator(
        inference_engine=engine,
        device=device,
        dry_run=args.dry_run,
    )

    all_expert_metrics = []
    for expert_idx in experts_to_eval:
        cfg = EXPERT_DATASET_CONFIG[expert_idx]
        dl = test_dataloaders.get(expert_idx)

        if dl is None and not args.dry_run:
            logger.warning(f"No DataLoader for expert {expert_idx}, skipping")
            continue

        metrics = evaluator.evaluate_expert(
            expert_idx=expert_idx,
            dataloader=dl,
            expert_name=cfg["name"],
            num_classes=cfg["num_classes"],
            is_multilabel=cfg["is_multilabel"],
        )
        all_expert_metrics.append(metrics)
        logger.info(
            f"Expert {expert_idx} ({cfg['name']}): "
            f"F1={metrics['f1_macro']:.4f}, "
            f"AUC={metrics['auc_roc']:.4f}, "
            f"RoutingAcc={metrics['routing_accuracy']:.4f}"
        )

    # 5. Load balance
    lb_report = evaluator.compute_load_balance(all_expert_metrics)
    logger.info(f"Load balance ratio: {lb_report['max_min_ratio']:.3f}")

    # 6. Summary
    summary = evaluator.compute_summary(all_expert_metrics, lb_report)
    logger.info(
        f"Summary — F1 2D: {summary['f1_macro_2d_mean']:.4f}, "
        f"F1 3D: {summary['f1_macro_3d_mean']:.4f}, "
        f"Routing: {summary['routing_accuracy_mean']:.4f}"
    )

    # 7. OOD detection
    if not args.no_ood:
        ood_detector = OODDetector(
            inference_engine=engine,
            device=device,
            dry_run=args.dry_run,
        )
        # Calibrar sobre val set del chest (in-distribution proxy)
        # En dry-run usa umbral por defecto
        threshold = ood_detector.calibrate_threshold(
            val_dataloader=None if args.dry_run else _build_val_dataloader()
        )
        logger.info(f"OOD threshold calibrado: {threshold:.4f}")

        ood_report = ood_detector.compute_ood_auroc(
            in_dist_dataloader=None if args.dry_run else _build_in_dist_loader(),
            ood_dataloader=None if args.dry_run else _build_ood_loader(),
        )
        logger.info(f"OOD AUROC: {ood_report['ood_auroc']:.4f}")

    # 8. Grad-CAM
    if not args.no_gradcam:
        logger.info("Generando Grad-CAM samples...")
        for expert_idx in experts_to_eval:
            # 2D experts (0,1,2): [B, 3, 224, 224]
            # 3D experts (3,4): [B, 1, 64, 64, 64]
            if expert_idx < 3:
                dummy_img = torch.randn(2, 3, 224, 224).to(device)
            else:
                dummy_img = torch.randn(2, 1, 64, 64, 64).to(device)

            generate_gradcam_samples(
                moe_system=moe,
                expert_idx=expert_idx,
                expert_name=EXPERT_NAMES[expert_idx],
                sample_images=dummy_img if args.dry_run else dummy_img,
                dry_run=args.dry_run,
            )

    logger.info("=== Paso 9 completado ===")
    return 0


def _build_val_dataloader():
    """Placeholder — en producción carga el val set de chest."""
    from src.pipeline.datasets.chest import ChestXray14Dataset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = ChestXray14Dataset(split="val", transform=transform)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)


def _build_in_dist_loader():
    """Placeholder — devuelve val set chest como in-distribution."""
    return _build_val_dataloader()


def _build_ood_loader():
    """Placeholder — en producción carga muestras OOD del CAE dataset."""
    from src.pipeline.datasets.cae import CAEDataset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = CAEDataset(split="test", transform=transform)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)


def main() -> None:
    # setup_logging requires (output_dir, phase_name)
    setup_logging(output_dir=RESULTS_DIR, phase_name="fase9")
    args = parse_args()

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("PASO 9 — DRY-RUN MODE")
        logger.info("Validando imports, configuración y estructura de directorios")
        logger.info("=" * 60)

    exit_code = run_evaluation(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
