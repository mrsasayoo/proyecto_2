"""Visualize the 26 False Negative 3D volumes from DenseNet3D evaluation.

Generates multi-slice views (axial, coronal, sagital) and optional 3D
volume renders using Plotly for each candidate volume.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "datasets" / "luna_lung_cancer" / "patches" / "test"
OUT_DIR = Path(__file__).resolve().parent

FN_IDS: list[str] = [
    "candidate_052458", "candidate_116807", "candidate_129835",
    "candidate_231094", "candidate_231127", "candidate_231311",
    "candidate_231678", "candidate_276894", "candidate_276900",
    "candidate_277052", "candidate_282052", "candidate_282084",
    "candidate_282464", "candidate_282945", "candidate_283333",
    "candidate_461962", "candidate_483326", "candidate_483861",
    "candidate_484086", "candidate_484446", "candidate_484530",
    "candidate_486947", "candidate_487073", "candidate_549524",
    "candidate_559242", "candidate_753549",
]

# ── Visual constants ─────────────────────────────────────────────────────
VMIN, VMAX = -0.1, 0.85
CMAP = "gray"
SLICE_FIG_DPI = 150


def load_volume(cid: str) -> np.ndarray:
    """Load a (64,64,64) volume from disk."""
    return np.load(DATA_DIR / f"{cid}.npy")


# ── 1. Tri-planar slice view ────────────────────────────────────────────
def plot_triplanar(
    vol: np.ndarray,
    cid: str,
    *,
    save: bool = True,
) -> None:
    """Axial / Coronal / Sagittal central slices + intensity profile."""
    mid = np.array(vol.shape) // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})

    titles = ["Axial (z-central)", "Coronal (y-central)", "Sagital (x-central)"]
    slices = [vol[mid[0], :, :], vol[:, mid[1], :], vol[:, :, mid[2]]]
    orientations = [("Anterior ↔ Posterior", "Izq ↔ Der"),
                    ("Superior ↔ Inferior", "Izq ↔ Der"),
                    ("Superior ↔ Inferior", "Anterior ↔ Posterior")]

    for i, (ax_img, ax_hist, title, slc, (ylabel, xlabel)) in enumerate(
        zip(axes[0], axes[1], titles, slices, orientations)
    ):
        im = ax_img.imshow(slc, cmap=CMAP, vmin=VMIN, vmax=VMAX, origin="lower")
        ax_img.set_title(title, fontsize=11, fontweight="bold")
        ax_img.set_xlabel(xlabel, fontsize=8)
        ax_img.set_ylabel(ylabel, fontsize=8)
        ax_img.tick_params(labelsize=7)

        # intensity histogram per slice
        ax_hist.hist(slc.ravel(), bins=60, color="#4a90d9", alpha=0.8, edgecolor="none")
        ax_hist.set_xlabel("Intensidad", fontsize=8)
        ax_hist.set_ylabel("Freq", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        ax_hist.axvline(0, color="red", lw=0.6, ls="--", alpha=0.5)

    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02, label="HU normalizado")

    # Stats annotation
    stats_text = (
        f"min={vol.min():.3f}  max={vol.max():.3f}  "
        f"mean={vol.mean():.3f}  std={vol.std():.3f}"
    )
    fig.suptitle(
        f"False Negative — {cid}\n{stats_text}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        fig.savefig(OUT_DIR / f"{cid}_triplanar.png", dpi=SLICE_FIG_DPI,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 2. Multi-slice montage (every 4th axial slice) ──────────────────────
def plot_montage(vol: np.ndarray, cid: str, *, step: int = 4) -> None:
    """Montage of axial slices to see nodule extent along z-axis."""
    indices = list(range(0, vol.shape[0], step))
    ncols = 8
    nrows = (len(indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.2 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else axes
    for ax in axes_flat:
        ax.axis("off")

    for i, idx in enumerate(indices):
        axes_flat[i].imshow(vol[idx, :, :], cmap=CMAP, vmin=VMIN, vmax=VMAX, origin="lower")
        axes_flat[i].set_title(f"z={idx}", fontsize=8)

    fig.suptitle(f"Montaje axial — {cid}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / f"{cid}_montage.png", dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 3. Interactive 3D volume render (Plotly isosurface) ──────────────────
def volume_render_plotly(vol: np.ndarray, cid: str) -> None:
    """Save an interactive Plotly isosurface HTML for 3D exploration."""
    z, y, x = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=vol.flatten(),
        isomin=0.05,
        isomax=float(vol.max()),
        surface_count=4,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale="Gray",
        opacity=0.4,
    ))
    fig.update_layout(
        title=dict(text=f"Volume Render — {cid}", font=dict(size=14)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        width=700, height=600,
    )
    fig.write_html(OUT_DIR / f"{cid}_3d.html")


# ── 4. Summary grid: all 26 FN central axial slices ─────────────────────
def plot_summary_grid() -> None:
    """Single figure with all 26 FN axial central slices for comparison."""
    ncols, nrows = 6, 5  # 30 slots for 26 volumes
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 15))
    for ax in axes.flatten():
        ax.axis("off")

    for i, cid in enumerate(FN_IDS):
        vol = load_volume(cid)
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.imshow(vol[32, :, :], cmap=CMAP, vmin=VMIN, vmax=VMAX, origin="lower")
        short_id = cid.replace("candidate_", "")
        ax.set_title(short_id, fontsize=9, fontweight="bold")
        ax.axis("on")
        ax.tick_params(labelsize=6)

    fig.suptitle(
        "26 Falsos Negativos — Corte axial central (z=32)\n"
        "DenseNet3D · Conjunto de test",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "summary_all_fn.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    print(f"Processing {len(FN_IDS)} false-negative volumes...\n")

    # Summary grid first
    print("Generating summary grid...")
    plot_summary_grid()
    print("  ✓ summary_all_fn.png\n")

    for i, cid in enumerate(FN_IDS, 1):
        print(f"[{i:02d}/{len(FN_IDS)}] {cid}")
        vol = load_volume(cid)

        plot_triplanar(vol, cid)
        print(f"  ✓ triplanar")

        plot_montage(vol, cid)
        print(f"  ✓ montage")

        volume_render_plotly(vol, cid)
        print(f"  ✓ 3D render")

    print("\n✅ All visualizations saved.")


if __name__ == "__main__":
    main()
