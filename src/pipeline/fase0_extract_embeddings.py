"""
FASE 0 — Extracción de CLS tokens (Embeddings)
===============================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II

Requerimientos del proyecto (sección 4.1):
  - Backbone: ViT-Tiny preentrenado de timm (o CvT / Swin-Tiny)
  - Backbone CONGELADO — no se entrena, solo extrae
  - Pasar los 5 datasets completos → guardar CLS tokens en disco
  - Resultado: Z_train [N_train, 192], Z_val [N_val, 192]
  - Etiqueta de experto: 0=Chest, 1=ISIC, 2=OA, 3=LUNA, 4=Pancreas
  - Hardware objetivo: clúster con 2 GPUs × 12 GB VRAM

Uso:
  python fase0_extract_embeddings.py --backbone vit_tiny_patch16_224 \
                                     --batch_size 64 \
                                     --output_dir ./embeddings
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import SimpleITK as sitk   # para LUNA16 y Pancreas
import cv2                 # para CLAHE en OA

# ──────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────

EXPERT_IDS = {
    "chest":    0,   # NIH ChestXray14
    "isic":     1,   # ISIC 2019
    "oa":       2,   # Osteoarthritis Knee
    "luna":     3,   # LUNA16 / LIDC-IDRI
    "pancreas": 4,   # Pancreatic Cancer CT
}

# Normalización ImageNet — estándar para todos los datasets 2D
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────
# 2. PREPROCESADOR ADAPTATIVO (rank=4 → 2D, rank=5 → 3D)
# ──────────────────────────────────────────────────────────

def build_2d_transform(img_size=224):
    """Transform estándar para todos los datasets 2D."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def apply_clahe(img_pil):
    """CLAHE obligatorio para OA Rodilla (contraste óseo)."""
    img_array = np.array(img_pil.convert("L"))        # escala de grises
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

#  aca se aplica una normalizacion para los datasets mensionados, estos dos, uno de torax y pancreas, este ignora rango que no nos sirven
def normalize_hu(volume, min_hu=-1000, max_hu=400):
    """Normalización HU para CT 3D (LUNA16, Pancreas)."""
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume.astype(np.float32)


def resize_volume_3d(volume, target=(64, 64, 64)):
    """
    Redimensiona un volumen 3D [D, H, W] a target usando
    interpolación trilineal vía PyTorch.
    """
    t = torch.from_numpy(volume).float()
    # Añadir dims de batch y canal: [1, 1, D, H, W]
    t = t.unsqueeze(0).unsqueeze(0)
    t = nn.functional.interpolate(
        t, size=target, mode="trilinear", align_corners=False
    )
    return t.squeeze(0)  # → [1, 64, 64, 64]


def volume_to_vit_input(volume_3d_tensor):
    """
    Convierte [1, 64, 64, 64] a [3, 224, 224] para que ViT-2D
    lo procese: toma 3 slices centrales (axial, coronal, sagital)
    y los apila como canales RGB.

    Nota: esto es una aproximación válida para extracción de
    embeddings de routing. Para clasificación 3D real se usa
    la arquitectura 3D del experto (R3D-18, Swin3D-Tiny).
    """
    v = volume_3d_tensor.squeeze(0)           # [64, 64, 64]
    d, h, w = v.shape

    axial    = v[d // 2, :, :]                # corte central eje Z
    coronal  = v[:, h // 2, :]               # corte central eje Y
    sagittal = v[:, :, w // 2]               # corte central eje X

    # Stack como RGB → [3, 64, 64]
    rgb = torch.stack([axial, coronal, sagittal], dim=0)

    # Resize a 224×224
    rgb = nn.functional.interpolate(
        rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)                              # [3, 224, 224]

    # Normalizar como ImageNet (el volumen ya está en [0,1])
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (rgb - mean) / std


# ──────────────────────────────────────────────────────────
# 3. DATASETS
# ──────────────────────────────────────────────────────────

class ChestXray14Dataset(Dataset):
    """
    NIH ChestXray14 — Multi-label, 14 patologías.
    Para embedding: la etiqueta de experto es siempre 0.
    """
    def __init__(self, csv_path, img_dir, file_list, transform):
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.expert_id = EXPERT_IDS["chest"]

        df = pd.read_csv(csv_path)
        # Filtrar según train_val_list.txt o test_list.txt
        with open(file_list) as f:
            valid_files = set(f.read().splitlines())
        self.df = df[df["Image Index"].isin(valid_files)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image Index"]
        img      = Image.open(self.img_dir / img_name).convert("RGB")
        img      = self.transform(img)
        return img, self.expert_id, img_name


class ISICDataset(Dataset):
    """
    ISIC 2019 — Multiclase, 9 clases (8 en train).
    Para embedding: etiqueta de experto = 1.
    """
    def __init__(self, gt_csv, img_dir, transform, split="train"):
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.expert_id = EXPERT_IDS["isic"]

        df = pd.read_csv(gt_csv)
        # Separar por lesion_id (ver análisis dataset)
        # Aquí asumimos que ya tienes tu split hecho previamente
        if split == "train":
            self.df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        else:
            self.df = df.drop(df.sample(frac=0.8, random_state=42).index).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        img_path = self.img_dir / f"{img_name}.jpg"
        img      = Image.open(img_path).convert("RGB")
        img      = self.transform(img)
        return img, self.expert_id, img_name


class OAKneeDataset(Dataset):
    """
    Osteoarthritis Knee — Ordinal, 3 clases (0=Normal, 1=Leve, 2=Severo).
    Para embedding: etiqueta de experto = 2.
    CLAHE obligatorio antes del transform.
    """
    def __init__(self, root_dir, split="train", img_size=224):
        self.expert_id = EXPERT_IDS["oa"]
        self.img_size  = img_size
        self.samples   = []

        split_dir = Path(root_dir) / split
        # Inferir carpetas: 0/, 1/, 2/ (o verificar con os.listdir)
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, int(class_dir.name)))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, int(class_dir.name)))

        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        # CLAHE antes del resize (requisito del proyecto)
        img = apply_clahe(img)
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        img = self.final_transform(img)
        return img, self.expert_id, str(img_path.name)


class LUNA16Dataset(Dataset):
    """
    LUNA16 — Clasificación de candidatos 3D.
    Para embedding: convierte volumen 3D → representación 2D (3 slices).
    Etiqueta de experto = 3.

    NOTA IMPORTANTE: La conversión world→vóxel ya debe estar resuelta
    antes de llamar a este dataset. Este dataset asume que tienes
    parches 3D pre-extraídos guardados como .npy de [64,64,64].
    """
    def __init__(self, patches_dir, candidates_csv):
        """
        patches_dir: carpeta con archivos candidate_XXXXX.npy (64×64×64, HU)
        candidates_csv: candidates_V2.csv con columnas seriesuid, class
        """
        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["luna"]

        df = pd.read_csv(candidates_csv)
        self.samples = []
        for _, row in df.iterrows():
            patch_file = self.patches_dir / f"candidate_{row.name:06d}.npy"
            if patch_file.exists():
                self.samples.append((patch_file, int(row["class"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_file, _ = self.samples[idx]
        volume  = np.load(patch_file)                   # [64, 64, 64] HU crudo
        volume  = normalize_hu(volume)                  # → [0, 1]
        volume_t = resize_volume_3d(volume)             # [1, 64, 64, 64]
        img     = volume_to_vit_input(volume_t)         # [3, 224, 224]
        return img, self.expert_id, patch_file.stem


class PancreasDataset(Dataset):
    """
    Pancreatic Cancer CT 3D — Binario, 2 clases.
    Igual que LUNA16: asume parches 3D pre-extraídos.
    Etiqueta de experto = 4.
    """
    def __init__(self, patches_dir):
        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["pancreas"]
        self.samples = list(self.patches_dir.glob("*.npy"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_file = self.samples[idx]
        volume  = np.load(patch_file)
        volume  = normalize_hu(volume)
        volume_t = resize_volume_3d(volume)
        img     = volume_to_vit_input(volume_t)
        return img, self.expert_id, patch_file.stem


# ──────────────────────────────────────────────────────────
# 4. BACKBONE ViT — EXTRACTOR (CONGELADO)
# ──────────────────────────────────────────────────────────

def load_frozen_backbone(backbone_name="vit_tiny_patch16_224", device="cuda"):
    """
    Carga ViT-Tiny preentrenado de timm y lo congela completamente.
    Solo se usa para forward pass — sin gradientes.

    Opciones del proyecto:
      - vit_tiny_patch16_224      (d_model=192, más liviano)
      - cvt_13                    (CvT, mejor localidad espacial)
      - swin_tiny_patch4_window7_224 (Swin-Tiny, más eficiente)
    """
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0          # quita la cabeza de clasificación
    )

    # Congelar TODOS los parámetros
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    # Verificar d_model
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"[ViT] Backbone: {backbone_name}")
    print(f"[ViT] d_model (CLS token dim): {out.shape[1]}")
    print(f"[ViT] Parámetros congelados: {sum(p.numel() for p in model.parameters()):,}")

    return model, out.shape[1]   # (model, d_model)


# ──────────────────────────────────────────────────────────
# 5. EXTRACCIÓN DE EMBEDDINGS
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, dataloader, device, d_model, desc=""):
    """
    Pasa el dataloader completo por el backbone y acumula:
      - Z:           [N, d_model] — CLS tokens
      - y_expert:    [N]          — etiqueta de experto (0-4)
      - img_names:   [N]          — nombre de imagen para auditoría
    """
    Z          = np.zeros((len(dataloader.dataset), d_model), dtype=np.float32)
    y_expert   = np.zeros(len(dataloader.dataset), dtype=np.int64)
    img_names  = []
    cursor     = 0

    for batch_idx, (imgs, experts, names) in enumerate(dataloader):
        imgs = imgs.to(device)

        # Forward pass (sin gradientes — congelado)
        z = model(imgs)                          # [B, d_model]
        z_np = z.cpu().numpy()

        B = z_np.shape[0]
        Z[cursor:cursor + B]        = z_np
        y_expert[cursor:cursor + B] = experts.numpy()
        img_names.extend(names)
        cursor += B

        if batch_idx % 50 == 0:
            print(f"  {desc} [{cursor}/{len(dataloader.dataset)}]")

    return Z[:cursor], y_expert[:cursor], img_names


# ──────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def build_datasets(cfg):
    """
    Construye los datasets de train y val para los 5 expertos.
    Devuelve (train_dataset, val_dataset) como ConcatDataset.

    cfg: dict con rutas de cada dataset (ver argparse abajo)
    """
    transform_2d = build_2d_transform(img_size=224)

    train_datasets = []
    val_datasets   = []

    # ── Experto 1: NIH ChestXray14 ──
    if cfg.get("chest_csv") and cfg.get("chest_imgs"):
        print("[Dataset] Cargando NIH ChestXray14...")
        chest_train = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["chest_train_list"],
            transform=transform_2d
        )
        chest_val = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["chest_val_list"],
            transform=transform_2d
        )
        train_datasets.append(chest_train)
        val_datasets.append(chest_val)
        print(f"  train: {len(chest_train):,}  val: {len(chest_val):,}")

    # ── Experto 2: ISIC 2019 ──
    if cfg.get("isic_gt") and cfg.get("isic_imgs"):
        print("[Dataset] Cargando ISIC 2019...")
        isic_train = ISICDataset(cfg["isic_gt"], cfg["isic_imgs"], transform_2d, split="train")
        isic_val   = ISICDataset(cfg["isic_gt"], cfg["isic_imgs"], transform_2d, split="val")
        train_datasets.append(isic_train)
        val_datasets.append(isic_val)
        print(f"  train: {len(isic_train):,}  val: {len(isic_val):,}")

    # ── Experto 3: OA Rodilla ──
    if cfg.get("oa_root"):
        print("[Dataset] Cargando Osteoarthritis Knee...")
        oa_train = OAKneeDataset(cfg["oa_root"], split="train")
        oa_val   = OAKneeDataset(cfg["oa_root"], split="val")
        train_datasets.append(oa_train)
        val_datasets.append(oa_val)
        print(f"  train: {len(oa_train):,}  val: {len(oa_val):,}")

    # ── Experto 4: LUNA16 ──
    if cfg.get("luna_patches") and cfg.get("luna_csv"):
        print("[Dataset] Cargando LUNA16 (parches 3D pre-extraídos)...")
        luna_train = LUNA16Dataset(cfg["luna_patches"] + "/train", cfg["luna_csv"])
        luna_val   = LUNA16Dataset(cfg["luna_patches"] + "/val",   cfg["luna_csv"])
        train_datasets.append(luna_train)
        val_datasets.append(luna_val)
        print(f"  train: {len(luna_train):,}  val: {len(luna_val):,}")

    # ── Experto 5: Pancreas ──
    if cfg.get("pancreas_patches_train") and cfg.get("pancreas_patches_val"):
        print("[Dataset] Cargando Pancreas CT 3D...")
        panc_train = PancreasDataset(cfg["pancreas_patches_train"])
        panc_val   = PancreasDataset(cfg["pancreas_patches_val"])
        train_datasets.append(panc_train)
        val_datasets.append(panc_val)
        print(f"  train: {len(panc_train):,}  val: {len(panc_val):,}")

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Dispositivo: {device}")
    if device == "cuda":
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Cargar backbone congelado ──
    model, d_model = load_frozen_backbone(args.backbone, device)

    # ── Construir datasets ──
    cfg = {
        "chest_csv":        args.chest_csv,
        "chest_imgs":       args.chest_imgs,
        "chest_train_list": args.chest_train_list,
        "chest_val_list":   args.chest_val_list,
        "isic_gt":          args.isic_gt,
        "isic_imgs":        args.isic_imgs,
        "oa_root":          args.oa_root,
        "luna_patches":     args.luna_patches,
        "luna_csv":         args.luna_csv,
        "pancreas_patches_train": args.pancreas_patches_train,
        "pancreas_patches_val":   args.pancreas_patches_val,
    }

    train_dataset, val_dataset = build_datasets(cfg)
    print(f"\n[Dataset] Total train: {len(train_dataset):,}")
    print(f"[Dataset] Total val:   {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # NO mezclar — queremos reproducibilidad
        num_workers=args.workers,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda")
    )

    # ── Extraer embeddings ──
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[FASE 0] Extrayendo embeddings de TRAIN...")
    Z_train, y_train, names_train = extract_embeddings(
        model, train_loader, device, d_model, desc="train"
    )

    print("\n[FASE 0] Extrayendo embeddings de VAL...")
    Z_val, y_val, names_val = extract_embeddings(
        model, val_loader, device, d_model, desc="val"
    )

    # ── Guardar en disco ──
    out = Path(args.output_dir)
    np.save(out / "Z_train.npy",     Z_train)
    np.save(out / "y_train.npy",     y_train)
    np.save(out / "Z_val.npy",       Z_val)
    np.save(out / "y_val.npy",       y_val)

    # Guardar nombres para auditoría (opcional pero útil)
    with open(out / "names_train.txt", "w") as f:
        f.write("\n".join(names_train))
    with open(out / "names_val.txt", "w") as f:
        f.write("\n".join(names_val))

    # ── Reporte final ──
    print(f"\n{'='*55}")
    print(f"FASE 0 completada")
    print(f"{'='*55}")
    print(f"Z_train:  {Z_train.shape}  ({Z_train.nbytes / 1e6:.1f} MB)")
    print(f"Z_val:    {Z_val.shape}  ({Z_val.nbytes / 1e6:.1f} MB)")
    print(f"\nDistribución de expertos en train:")
    for exp_name, exp_id in EXPERT_IDS.items():
        count = (y_train == exp_id).sum()
        pct   = count / len(y_train) * 100
        print(f"  Experto {exp_id} ({exp_name:<10}): {count:>6,}  ({pct:.1f}%)")
    print(f"\nArchivos guardados en: {args.output_dir}")
    print(f"Siguiente paso: python fase1_ablation_router.py --embeddings {args.output_dir}")


# ──────────────────────────────────────────────────────────
# 7. ARGPARSE
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FASE 0 — Extracción de embeddings MoE")

    # Backbone
    parser.add_argument("--backbone",     default="vit_tiny_patch16_224",
                        choices=["vit_tiny_patch16_224", "cvt_13",
                                 "swin_tiny_patch4_window7_224"],
                        help="Backbone ViT a usar (proyecto permite ViT, CvT o Swin-Tiny)")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--output_dir",   default="./embeddings",
                        help="Carpeta donde se guardan Z_train.npy y Z_val.npy")

    # Rutas NIH ChestXray14
    parser.add_argument("--chest_csv",        default=None)
    parser.add_argument("--chest_imgs",       default=None)
    parser.add_argument("--chest_train_list", default=None,
                        help="train_val_list.txt oficial")
    parser.add_argument("--chest_val_list",   default=None,
                        help="test_list.txt oficial (como val para extracción)")

    # Rutas ISIC 2019
    parser.add_argument("--isic_gt",          default=None,
                        help="ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--isic_imgs",        default=None,
                        help="Carpeta con imágenes .jpg de ISIC")

    # Rutas OA Rodilla
    parser.add_argument("--oa_root",          default=None,
                        help="Raíz con subcarpetas train/val/test y clases 0,1,2")

    # Rutas LUNA16
    parser.add_argument("--luna_patches",     default=None,
                        help="Carpeta con parches 3D .npy pre-extraídos (train/ y val/)")
    parser.add_argument("--luna_csv",         default=None,
                        help="candidates_V2.csv")

    # Rutas Pancreas
    parser.add_argument("--pancreas_patches_train", default=None)
    parser.add_argument("--pancreas_patches_val",   default=None)

    args = parser.parse_args()
    main(args)
