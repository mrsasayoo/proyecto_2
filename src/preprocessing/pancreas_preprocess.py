"""
pancreas_preprocess.py — Pipeline de preprocesado para volúmenes CT pancreáticos.

Pasos:
  1. Leer .nii.gz → array HU
  2. Clip HU abdominal [-100, 400]
  3. Normalizar a [0, 1]
  4. Resampling isotrópico al spacing mínimo del volumen
  5. Resize a 64×64×64 con interpolación trilineal
"""

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

HU_MIN, HU_MAX = -100, 400
TARGET_SHAPE = (64, 64, 64)


def preprocess_pancreas_volume(nii_path: str) -> np.ndarray:
    """Retorna array float32 [64, 64, 64] normalizado a [0, 1]."""
    nii = nib.load(nii_path)
    vol = nii.get_fdata(dtype=np.float32)
    sp = np.array(nii.header.get_zooms()[:3], dtype=np.float64)

    # Clip + normalizar
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)

    # Resampling isotrópico
    min_sp = max(sp.min(), 1e-6)
    zoom_factors = sp / min_sp
    vol_iso = ndimage.zoom(vol, zoom_factors, order=1)

    # Resize al target
    final_zoom = (np.array(TARGET_SHAPE, dtype=np.float64)
                  / np.array(vol_iso.shape, dtype=np.float64))
    vol_final = ndimage.zoom(vol_iso, final_zoom, order=1)

    return vol_final.astype(np.float32)


if __name__ == "__main__":
    import sys
    import glob

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        files = [path]
    else:
        files = sorted(glob.glob("datasets/zenodo_13715870/*.nii.gz"))[:3]

    for nf in files:
        result = preprocess_pancreas_volume(nf)
        assert result.shape == TARGET_SHAPE, f"Shape: {result.shape}"
        assert 0.0 <= result.min() and result.max() <= 1.0, (
            f"Rango: [{result.min():.3f}, {result.max():.3f}]"
        )
        print(f"OK {nf}: shape={result.shape} "
              f"min={result.min():.4f} max={result.max():.4f}")
