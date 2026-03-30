"""
dataset_builder.py — Lector de Splits y Constructor de Datasets
===============================================================

Responsabilidad única: leer los splits ya generados por Fase 0,
instanciar los 5 datasets de dominio con las transformaciones correctas,
y devolver tres ConcatDataset (train, val, test) listos para DataLoader.

Restricción fundamental: consumidor PURO de splits.
Nunca genera, re-divide, re-ordena ni re-mezcla datos.
"""

import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import ConcatDataset

# Datasets de Familia 3 (importados via sys.path → src/pipeline/)
from datasets.chest import ChestXray14Dataset
from datasets.isic import ISICDataset
from datasets.osteoarthritis import OAKneeDataset
from datasets.luna import LUNA16Dataset
from datasets.pancreas import PancreasDataset, PanoramaLabelLoader

# Transforms de este paquete (importados via sys.path → src/pipeline/fase1/)
from transform_2d import build_2d_transform
from transform_domain import apply_circular_crop, apply_clahe  # noqa: F401
from transform_3d import full_3d_pipeline, volume_to_vit_input  # noqa: F401
from fase1_config import MAX_IMBALANCE, PANCREAS_FOLD, IMG_SIZE

# ── Implementaciones canónicas de transform por dominio ─────────────────────
# Nota: LUNA16Dataset y PancreasDataset no aceptan transforms externos en su
# constructor actual (Familia 3 define sus propios pipelines de HU/3D).
# full_3d_pipeline / volume_to_vit_input están aquí como API canónica;
# se conectarán directamente cuando Familia 3 acepte transforms externos.
# TODO(fase1): pasar _make_3d_transform() a LUNA16/PancreasDataset tras
#              actualizar sus constructores para aceptar `transform_3d`.


class _BCNCrop:
    """Torchvision-compatible PIL transform que aplica el crop circular de ISIC."""

    def __call__(self, img_pil):
        return apply_circular_crop(img_pil)


def _make_isic_transform(img_size: int):
    """Transform canónico para ISIC: crop circular BCN + resize + normalizar.

    Combina apply_circular_crop de transform_domain con build_2d_transform.
    Se pasa como transform_standard a ISICDataset; en modo 'embedding' el
    dataset usa su propio tfs['embedding'], pero en modo 'expert' este
    transform se aplica a las imágenes de mayoría.
    """
    from torchvision import transforms as T
    from fase1_config import IMAGENET_MEAN, IMAGENET_STD

    return T.Compose(
        [
            _BCNCrop(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


log = logging.getLogger("fase1")


def build_datasets(cfg):
    """
    Lee los splits de Fase 0 y construye tres ConcatDataset (train, val, test).

    Cada muestra devuelve (tensor_imagen, expert_id, nombre_archivo) —
    contrato que embeddings_extractor.py espera.

    Args:
        cfg: dict con paths a artefactos de Fase 0 y opciones.
             Claves requeridas dependen de datasets activos.
    """
    img_size = cfg.get("img_size", IMG_SIZE)
    transform_2d = build_2d_transform(img_size=img_size)
    isic_transform = _make_isic_transform(img_size)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    # ── Experto 0: NIH ChestXray14 ──────────────────────────
    if cfg.get("nih_train_list") and cfg.get("chest_csv") and cfg.get("chest_imgs"):
        log.info("[Dataset] Cargando NIH ChestXray14 (Experto 0)...")
        view_filter = cfg.get("chest_view_filter")

        chest_train = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["nih_train_list"],
            transform=transform_2d,
            mode="embedding",
            filter_view=view_filter,
        )
        chest_val = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["nih_val_list"],
            transform=transform_2d,
            mode="embedding",
            filter_view=view_filter,
        )
        chest_test = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["nih_test_list"],
            transform=transform_2d,
            mode="embedding",
            filter_view=view_filter,
        )
        train_datasets.append(chest_train)
        val_datasets.append(chest_val)
        test_datasets.append(chest_test)
        log.info(
            "  → train: %s  val: %s  test: %s",
            f"{len(chest_train):,}",
            f"{len(chest_val):,}",
            f"{len(chest_test):,}",
        )

        if cfg.get("chest_bbox_csv"):
            try:
                ChestXray14Dataset.load_bbox_index(cfg["chest_bbox_csv"])
                log.info("[Chest] BBox index cargado.")
            except Exception as e:
                log.warning("[Chest] No se pudo cargar BBox: %s", e)
    else:
        log.warning("[Dataset] NIH ChestXray14 OMITIDO — faltan rutas")

    # ── Experto 1: ISIC 2019 ────────────────────────────────
    if cfg.get("isic_train_csv") and cfg.get("isic_imgs"):
        log.info("[Dataset] Cargando ISIC 2019 (Experto 1)...")

        isic_train_df = pd.read_csv(cfg["isic_train_csv"])
        isic_val_df = pd.read_csv(cfg["isic_val_csv"])
        isic_test_df = pd.read_csv(cfg["isic_test_csv"])

        # Filtrar imágenes MSK _downsampled: no están en disco como archivos independientes
        # (solo existen como parte de las imágenes originales). Sin este filtro, el Dataset
        # retorna tensores de ceros para estas entradas, contaminando los embeddings.
        for split_name, split_df in [
            ("train", isic_train_df),
            ("val", isic_val_df),
            ("test", isic_test_df),
        ]:
            antes = len(split_df)
            filtrado = split_df[
                ~split_df["image"].str.contains("_downsampled", na=False)
            ]
            eliminadas = antes - len(filtrado)
            if eliminadas > 0:
                log.info(
                    f"[ISIC/{split_name}] Filtradas {eliminadas} imágenes _downsampled no disponibles en disco"
                )
            # Reasignar al DataFrame correspondiente
            if split_name == "train":
                isic_train_df = filtrado
            elif split_name == "val":
                isic_val_df = filtrado
            else:
                isic_test_df = filtrado

        isic_train = ISICDataset(
            img_dir=cfg["isic_imgs"],
            split_df=isic_train_df,
            mode="embedding",
            transform_standard=isic_transform,
        )
        isic_val = ISICDataset(
            img_dir=cfg["isic_imgs"],
            split_df=isic_val_df,
            mode="embedding",
            transform_standard=isic_transform,
        )
        isic_test = ISICDataset(
            img_dir=cfg["isic_imgs"],
            split_df=isic_test_df,
            mode="embedding",
            transform_standard=isic_transform,
        )
        train_datasets.append(isic_train)
        val_datasets.append(isic_val)
        test_datasets.append(isic_test)
        log.info(
            "  → train: %s  val: %s  test: %s",
            f"{len(isic_train):,}",
            f"{len(isic_val):,}",
            f"{len(isic_test):,}",
        )
    else:
        log.warning("[Dataset] ISIC 2019 OMITIDO — faltan rutas")

    # ── Experto 2: OA Rodilla ────────────────────────────────
    if cfg.get("oa_root"):
        log.info("[Dataset] Cargando OA Rodilla (Experto 2)...")

        oa_train = OAKneeDataset(
            cfg["oa_root"], split="train", img_size=img_size, mode="embedding"
        )
        oa_val = OAKneeDataset(
            cfg["oa_root"], split="val", img_size=img_size, mode="embedding"
        )
        oa_test = OAKneeDataset(
            cfg["oa_root"], split="test", img_size=img_size, mode="embedding"
        )

        train_datasets.append(oa_train)
        val_datasets.append(oa_val)
        test_datasets.append(oa_test)
        log.info(
            "  → train: %s  val: %s  test: %s",
            f"{len(oa_train):,}",
            f"{len(oa_val):,}",
            f"{len(oa_test):,}",
        )
    else:
        log.warning("[Dataset] OA Rodilla OMITIDO — falta --oa_root")

    # ── Experto 3: LUNA16 ───────────────────────────────────
    if cfg.get("luna_patches_dir") and cfg.get("luna_csv"):
        log.info("[Dataset] Cargando LUNA16 parches 3D (Experto 3)...")
        _base = Path(cfg["luna_patches_dir"])

        # Lee los tres directorios directamente — sin re-dividir
        luna_train = LUNA16Dataset(
            str(_base / "train"),
            cfg["luna_csv"],
            mode="embedding",
        )
        luna_val = LUNA16Dataset(
            str(_base / "val"),
            cfg["luna_csv"],
            mode="embedding",
        )
        luna_test = LUNA16Dataset(
            str(_base / "test"),
            cfg["luna_csv"],
            mode="embedding",
        )
        train_datasets.append(luna_train)
        val_datasets.append(luna_val)
        test_datasets.append(luna_test)
        log.info(
            "[LUNA16] Splits respetan separación por seriesuid de Fase 0 "
            "(tres directorios leídos directamente, sin re-dividir)."
        )
        log.info(
            "  → train: %s  val: %s  test: %s",
            f"{len(luna_train):,}",
            f"{len(luna_val):,}",
            f"{len(luna_test):,}",
        )
    else:
        log.warning("[Dataset] LUNA16 OMITIDO — faltan rutas")

    # ── Experto 4: Páncreas PANORAMA ────────────────────────
    if cfg.get("pancreas_splits_csv") and cfg.get("pancreas_nii_dir"):
        log.info("[Dataset] Cargando Páncreas PANORAMA (Experto 4)...")

        fold = cfg.get("pancreas_fold", PANCREAS_FOLD)
        splits_df = pd.read_csv(cfg["pancreas_splits_csv"])

        train_mask = splits_df["split"] == "fold{}_train".format(fold)
        val_mask = splits_df["split"] == "fold{}_val".format(fold)
        test_mask = splits_df["split"] == "test"

        nii_dir = Path(cfg["pancreas_nii_dir"])

        def _build_pairs(mask):
            """Construye pares (ruta_nii, label) desde el CSV de splits."""
            pairs = []
            for _, row in splits_df[mask].iterrows():
                case_id = row["case_id"]
                label = int(row["label"])
                candidates = list(nii_dir.rglob("*{}*.nii.gz".format(case_id)))
                if not candidates:
                    candidates = list(nii_dir.rglob("*{}*.nii".format(case_id)))
                if candidates:
                    pairs.append((str(candidates[0]), label))
            return pairs

        roi = cfg.get("pancreas_roi_strategy", "A")
        panc_train_pairs = _build_pairs(train_mask)
        panc_val_pairs = _build_pairs(val_mask)
        panc_test_pairs = _build_pairs(test_mask)

        if panc_train_pairs:
            panc_train = PancreasDataset(
                panc_train_pairs,
                mode="embedding",
                roi_strategy=roi,
            )
            panc_val = PancreasDataset(
                panc_val_pairs,
                mode="embedding",
                roi_strategy=roi,
            )
            panc_test = PancreasDataset(
                panc_test_pairs,
                mode="embedding",
                roi_strategy=roi,
            )
            train_datasets.append(panc_train)
            val_datasets.append(panc_val)
            test_datasets.append(panc_test)
            log.info(
                "  → train: %s  val: %s  test: %s",
                f"{len(panc_train):,}",
                f"{len(panc_val):,}",
                f"{len(panc_test):,}",
            )
        else:
            log.error("[Pancreas] Sin pares válidos — verificar splits y NIfTIs")
    else:
        log.warning("[Dataset] Páncreas OMITIDO — faltan rutas")

    # ── Experto 5 (OOD) — sin dataset ───────────────────────
    log.info(
        "[Dataset] Experto 5 (OOD) — sin dataset en Fase 1. Se inicializa en Fase 2."
    )

    # ── Verificaciones transversales ────────────────────────
    for split_name, ds_list in [
        ("train", train_datasets),
        ("val", val_datasets),
        ("test", test_datasets),
    ]:
        sizes = [len(d) for d in ds_list]
        if sizes:
            log.info("[Dataset] Tamaños %s por experto: %s", split_name, sizes)
            nonzero = [s for s in sizes if s > 0]
            if len(nonzero) > 1:
                ratio = max(nonzero) / min(nonzero)
                if ratio > MAX_IMBALANCE:
                    log.warning(
                        "[Dataset/%s] Balance desigual: max/min = %.1fx",
                        split_name,
                        ratio,
                    )
                else:
                    log.info("[Dataset/%s] ratio max/min = %.1fx", split_name, ratio)
            zero_idx = [i for i, s in enumerate(sizes) if s == 0]
            if zero_idx:
                log.warning(
                    "[Dataset/%s] Expertos con 0 muestras: %s", split_name, zero_idx
                )

    if not train_datasets:
        log.error(
            "[Dataset] ¡Ningún dataset cargado! "
            "Verifica que al menos una ruta sea válida."
        )

    return (
        ConcatDataset(train_datasets),
        ConcatDataset(val_datasets),
        ConcatDataset(test_datasets),
    )
