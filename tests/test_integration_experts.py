"""
Integration tests for MoE expert models (Experts 0-5) and the MoE system.

Tests cover:
    - Forward pass per expert (shape, no NaN/Inf).
    - count_parameters() returns a positive integer per expert.
    - build_moe_system_dry_run() assembles the full MoE system.

All tests run on CPU with batch_size ∈ {1, 2}, no real weights loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# sys.path setup — the project has no installable package, so we inject the
# paths needed by the expert modules and their internal imports.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
_PIPELINE = _SRC / "pipeline"
_FASE2 = _PIPELINE / "fase2"
_ROUTERS = _FASE2 / "routers"

for _p in (_SRC, _PIPELINE, _FASE2, _ROUTERS):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Imports — using the canonical class names from each expert file.
# ---------------------------------------------------------------------------
from pipeline.fase2.models.expert1_convnext import Expert1ConvNeXt  # noqa: E402
from pipeline.fase2.models.expert2_convnext_small import Expert2ConvNeXtSmall  # noqa: E402
from pipeline.fase2.models.expert_oa_efficientnet_b3 import ExpertOAEfficientNetB3  # noqa: E402
from pipeline.fase2.models.expert3_densenet3d import Expert3DenseNet3D  # noqa: E402
from pipeline.fase2.models.expert4_resnet3d import ExpertPancreasResNet3D  # noqa: E402
from pipeline.fase3.models.expert6_resunet import ConditionedResUNetAE  # noqa: E402

DEVICE = torch.device("cpu")
BATCH_SIZE = 2


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════


def _assert_no_nan_inf(t: torch.Tensor, label: str) -> None:
    """Fail if *t* contains NaN or Inf values."""
    assert not torch.isnan(t).any(), f"{label}: output contains NaN"
    assert not torch.isinf(t).any(), f"{label}: output contains Inf"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Forward-pass tests — one per expert
# ═══════════════════════════════════════════════════════════════════════════


class TestExpert0ConvNeXtTiny:
    """Expert 0 — ConvNeXt-Tiny (14-class multilabel, ChestXray14)."""

    def test_forward_shape_and_values(self) -> None:
        model = Expert1ConvNeXt(
            num_classes=14,
            dropout_fc=0.3,
            pretrained=False,
        ).to(DEVICE)
        model.eval()

        x = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH_SIZE, 14), (
            f"Expected ({BATCH_SIZE}, 14), got {out.shape}"
        )
        _assert_no_nan_inf(out, "Expert0/ConvNeXt-Tiny")


class TestExpert1ConvNeXtSmall:
    """Expert 1 — ConvNeXt-Small (8-class multiclass, ISIC 2019)."""

    def test_forward_shape_and_values(self) -> None:
        model = Expert2ConvNeXtSmall(
            num_classes=8,
            pretrained=False,
        ).to(DEVICE)
        model.eval()

        x = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH_SIZE, 8), (
            f"Expected ({BATCH_SIZE}, 8), got {out.shape}"
        )
        _assert_no_nan_inf(out, "Expert1/ConvNeXt-Small")


class TestExpert2EfficientNetB3:
    """Expert 2 — EfficientNet-B3 (5-class KL grading, OA Knee)."""

    def test_forward_shape_and_values(self) -> None:
        model = ExpertOAEfficientNetB3(
            num_classes=5,
            dropout=0.4,
        ).to(DEVICE)
        model.eval()

        x = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH_SIZE, 5), (
            f"Expected ({BATCH_SIZE}, 5), got {out.shape}"
        )
        _assert_no_nan_inf(out, "Expert2/EfficientNet-B3")


class TestExpert3DenseNet3D:
    """Expert 3 — DenseNet 3D (binary, LUNA16 nodules)."""

    def test_forward_shape_and_values(self) -> None:
        model = Expert3DenseNet3D(
            in_channels=1,
            num_classes=2,
            growth_rate=32,
            block_layers=[4, 8, 16, 12],
            init_features=64,
            spatial_dropout_p=0.15,
            fc_dropout_p=0.4,
        ).to(DEVICE)
        model.eval()

        # Typical 3D input: [B, 1, 64, 64, 64]
        x = torch.randn(BATCH_SIZE, 1, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH_SIZE, 2), (
            f"Expected ({BATCH_SIZE}, 2), got {out.shape}"
        )
        _assert_no_nan_inf(out, "Expert3/DenseNet3D")


class TestExpert4ResNet3D:
    """Expert 4 — R3D-18 (binary, PDAC pancreas CT)."""

    def test_forward_shape_and_values(self) -> None:
        model = ExpertPancreasResNet3D(
            in_channels=1,
            num_classes=2,
            dropout_p=0.5,
        ).to(DEVICE)
        model.eval()

        # Typical 3D input: [B, 1, 64, 64, 64]
        x = torch.randn(BATCH_SIZE, 1, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH_SIZE, 2), (
            f"Expected ({BATCH_SIZE}, 2), got {out.shape}"
        )
        _assert_no_nan_inf(out, "Expert4/R3D-18")


class TestExpert6ResUNetAutoencoder:
    """Expert 5 → Res-U-Net v6 (OOD detection, multimodal)."""

    def test_forward_shape_and_values(self) -> None:
        model = ConditionedResUNetAE(
            in_ch=3,
            base_ch=64,
            n_domains=6,
        ).to(DEVICE)
        model.eval()

        x = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
        domain_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            result = model(x, domain_ids)

        # ResUNetAutoencoder returns a tuple (x_hat, z)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"

        x_hat, z = result

        assert x_hat.shape == (BATCH_SIZE, 3, 224, 224), (
            f"x_hat shape: expected ({BATCH_SIZE}, 3, 224, 224), got {x_hat.shape}"
        )
        assert z.shape == (BATCH_SIZE, 512), (
            f"z shape: expected ({BATCH_SIZE}, 512), got {z.shape}"
        )
        _assert_no_nan_inf(x_hat, "Expert6/ResUNet x_hat")
        _assert_no_nan_inf(z, "Expert6/ResUNet z")


# ═══════════════════════════════════════════════════════════════════════════
# 2. count_parameters() tests — one per expert
# ═══════════════════════════════════════════════════════════════════════════


class TestCountParameters:
    """Verify count_parameters() returns a positive int for each expert."""

    def test_expert0_count_parameters(self) -> None:
        model = Expert1ConvNeXt(num_classes=14, dropout_fc=0.3, pretrained=False)
        n = model.count_parameters()
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive, got {n}"

    def test_expert1_count_parameters(self) -> None:
        model = Expert2ConvNeXtSmall(num_classes=8, pretrained=False)
        n = model.count_parameters()
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive, got {n}"

    def test_expert2_count_parameters(self) -> None:
        model = ExpertOAEfficientNetB3(num_classes=5, dropout=0.4)
        n = model.count_parameters()
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive, got {n}"

    def test_expert3_count_parameters(self) -> None:
        model = Expert3DenseNet3D(
            in_channels=1,
            num_classes=2,
            spatial_dropout_p=0.15,
            fc_dropout_p=0.4,
        )
        n = model.count_parameters()
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive, got {n}"

    def test_expert4_count_parameters(self) -> None:
        model = ExpertPancreasResNet3D(in_channels=1, num_classes=2, dropout_p=0.5)
        n = model.count_parameters()
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive, got {n}"

    def test_expert5_count_parameters(self) -> None:
        model = ConditionedResUNetAE(in_ch=3, base_ch=64)
        n = model.count_parameters()
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert 45_000_000 <= n <= 60_000_000, f"Expected 45M–60M params, got {n:,}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. MoE system dry-run test
# ═══════════════════════════════════════════════════════════════════════════


class TestMoESystemDryRun:
    """Test build_moe_system_dry_run() from fase5/moe_model.py."""

    def test_build_moe_system_dry_run(self) -> None:
        from pipeline.fase5.moe_model import build_moe_system_dry_run

        moe = build_moe_system_dry_run(d_model=192)

        # If we ever reach here (bug fixed), verify basic properties.
        assert hasattr(moe, "experts")
        assert hasattr(moe, "router")
        assert len(moe.experts) == 6
        params = moe.get_trainable_params_by_component()
        assert params["total"] > 0

    def test_manual_moe_assembly(self) -> None:
        """Assemble a MoE system manually with correct constructor args.

        This mirrors what build_moe_system_dry_run() *should* do, using the
        correct keyword arguments for each expert constructor.
        """
        from pipeline.fase2.routers.linear import LinearGatingHead
        from pipeline.fase5.moe_model import MoESystem

        experts = torch.nn.ModuleList(
            [
                Expert1ConvNeXt(num_classes=14, dropout_fc=0.3, pretrained=False),
                Expert2ConvNeXtSmall(num_classes=8, pretrained=False),
                ExpertOAEfficientNetB3(num_classes=5, dropout=0.4),
                Expert3DenseNet3D(
                    in_channels=1,
                    num_classes=2,
                    spatial_dropout_p=0.15,
                    fc_dropout_p=0.4,
                ),
                ExpertPancreasResNet3D(in_channels=1, num_classes=2, dropout_p=0.5),
                ConditionedResUNetAE(in_ch=3, base_ch=64, n_domains=6),
            ]
        )
        router = LinearGatingHead(d_model=192, n_experts=5)
        moe = MoESystem(experts=experts, router=router, backbone=None)

        assert len(moe.experts) == 6
        assert moe.n_experts_domain == 5
        assert moe.n_experts_total == 6

        params = moe.get_trainable_params_by_component()
        assert isinstance(params["total"], int)
        assert params["total"] > 0
        for i in range(6):
            assert params[f"expert_{i}"] > 0, f"expert_{i} has 0 trainable params"
