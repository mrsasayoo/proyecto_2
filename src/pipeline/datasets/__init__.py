"""
Re-exports de todos los datasets del pipeline MoE.

Uso:
    from src.pipeline.datasets import (
        ChestXray14Dataset, ISICDataset, OAKneeDataset,
        LUNA16Dataset, PancreasDataset,
        LUNA16PatchExtractor, LUNA16FROCEvaluator,
        PanoramaLabelLoader, PancreasROIExtractor,
    )
"""

from .chest import ChestXray14Dataset


# Lazy imports para evitar cargar dependencias innecesarias de otros datasets
def __getattr__(name):
    """Lazy import de datasets que no se usan en Expert 1."""
    if name == "ISICDataset":
        from .isic import ISICDataset

        return ISICDataset
    elif name == "OAKneeDataset":
        from .osteoarthritis import OAKneeDataset

        return OAKneeDataset
    elif name == "LUNA16Dataset":
        from .luna import LUNA16Dataset

        return LUNA16Dataset
    elif name == "LUNA16PatchExtractor":
        from .luna import LUNA16PatchExtractor

        return LUNA16PatchExtractor
    elif name == "LUNA16FROCEvaluator":
        from .luna import LUNA16FROCEvaluator

        return LUNA16FROCEvaluator
    elif name == "verify_hu_normalization":
        from .luna import verify_hu_normalization

        return verify_hu_normalization
    elif name == "PancreasDataset":
        from .pancreas import PancreasDataset

        return PancreasDataset
    elif name == "PanoramaLabelLoader":
        from .pancreas import PanoramaLabelLoader

        return PanoramaLabelLoader
    elif name == "PancreasROIExtractor":
        from .pancreas import PancreasROIExtractor

        return PancreasROIExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChestXray14Dataset",
    "ISICDataset",
    "OAKneeDataset",
    "LUNA16Dataset",
    "LUNA16PatchExtractor",
    "LUNA16FROCEvaluator",
    "verify_hu_normalization",
    "PancreasDataset",
    "PanoramaLabelLoader",
    "PancreasROIExtractor",
]

# Note: otros imports están disponibles via lazy __getattr__ para evitar
# cargar dependencias como torchvision que pueden falta liblzma en algunos
# entornos Python (especialmente Python 3.14 compilado sin liblzma-dev).
