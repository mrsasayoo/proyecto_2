"""
Osteoarthritis Knee — Experto 2: Ordinal, 3 clases (0=Normal, 1=Leve, 2=Severo).

Hallazgos implementados:
  H1 → clases ordinales: CrossEntropyLoss con pesos + QWK como métrica
  H2 → augmentation offline en el ZIP: detección por hash MD5 de muestra
  H3 → sin metadatos de paciente: split por carpeta (OAI predefinido)
  H4 → CLAHE ANTES del resize (espacio articular a alta resolución)
  H5 → KL1 contamina la clase "Leve" con alta ambigüedad inter-observador
"""

import logging
import random
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

from config import (
    EXPERT_IDS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OA_CLASS_NAMES,
    OA_N_CLASSES,
    OA_BASE_IMG_COUNT,
)

log = logging.getLogger("fase0")


class OAKneeDataset(Dataset):
    """
    Osteoarthritis Knee — Ordinal, 3 clases (0=Normal, 1=Leve, 2=Severo).

    Dos modos:
      mode="embedding" → FASE 0: (img, expert_id=2, img_name) — CLAHE aplicado
      mode="expert"    → FASE 2: (img, kl_label_int, img_name) — aug diferenciado
    """

    @staticmethod
    def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """QWK — Métrica principal para evaluación ordinal del Experto 2."""
        from sklearn.metrics import cohen_kappa_score

        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        return float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2])
        )

    @staticmethod
    def evaluate_boundary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        H5/Item-9 — Analiza la matriz de confusión con foco en la frontera
        Clase0↔Clase1 (Normal↔Leve), que es la más difícil por contaminación KL1.
        """
        from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_cm

        cm = sk_cm(y_true, y_pred, labels=[0, 1, 2])
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        n0 = cm[0].sum()
        n1 = cm[1].sum()
        n2 = cm[2].sum()

        b01 = (cm[0, 1] / max(n0, 1) + cm[1, 0] / max(n1, 1)) / 2
        b12 = (cm[1, 2] / max(n1, 1) + cm[2, 1] / max(n2, 1)) / 2
        severe_miss = cm[2, 0] / max(n2, 1)

        recall = {i: cm[i, i] / max(cm[i].sum(), 1) for i in range(3)}

        return {
            "qwk": float(qwk),
            "confusion_matrix": cm,
            "boundary_01_error_rate": float(b01),
            "boundary_12_error_rate": float(b12),
            "severe_misclass_rate": float(severe_miss),
            "per_class_recall": {i: float(r) for i, r in recall.items()},
        }

    @staticmethod
    def log_boundary_confusion(metrics: dict, epoch: int = -1) -> None:
        """H5/Item-9 — Imprime en el log un resumen de la matriz de confusión."""
        ep_str = f"Época {epoch} — " if epoch >= 0 else ""
        cm = metrics["confusion_matrix"]
        log.info(
            f"[OA] {ep_str}Evaluación Experto 2:\n"
            f"    QWK (métrica principal) : {metrics['qwk']:.4f}\n"
            f"    Recall por clase:\n"
            f"      Clase 0 Normal : {metrics['per_class_recall'][0]:.3f}\n"
            f"      Clase 1 Leve   : {metrics['per_class_recall'][1]:.3f}\n"
            f"      Clase 2 Severo : {metrics['per_class_recall'][2]:.3f}\n"
            f"    Matriz de confusión (filas=real, cols=pred):\n"
            f"      {cm[0]}\n"
            f"      {cm[1]}\n"
            f"      {cm[2]}\n"
            f"    H5 — Frontera 0↔1 (KL1 ambiguo): "
            f"error={metrics['boundary_01_error_rate']:.3f} "
            f"{'⚠ ALTA' if metrics['boundary_01_error_rate'] > 0.25 else '✓ aceptable'}\n"
            f"    Frontera 1↔2: error={metrics['boundary_12_error_rate']:.3f}\n"
            f"    Peor error: Severo→Normal = {metrics['severe_misclass_rate']:.3f} "
            f"{'⚠⚠ CRÍTICO (costo ordinal=4)' if metrics['severe_misclass_rate'] > 0.05 else '✓'}"
        )

    def __init__(self, root_dir, split="train", img_size=224, mode="embedding"):
        assert mode in ("embedding", "expert"), (
            f"[OA] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
        )

        self.expert_id = EXPERT_IDS["oa"]
        self.img_size = img_size
        self.mode = mode
        self.split = split
        self.samples = []

        split_dir = Path(root_dir) / split
        if not split_dir.exists():
            log.error(f"[OA] Directorio de split no encontrado: {split_dir}")
            return

        # ── Leer clases desde carpetas ──────────────────────────────────
        class_counts = {}
        found_classes = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                class_id = int(class_dir.name)
            except ValueError:
                log.warning(
                    f"[OA] Carpeta con nombre no numérico ignorada: '{class_dir.name}'."
                )
                continue
            imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_path in imgs:
                self.samples.append((img_path, class_id))
            class_counts[class_dir.name] = len(imgs)
            found_classes.append(class_dir.name)

        n_found = len(found_classes)
        log.info(
            f"[OA] split='{split}' mode='{mode}': {len(self.samples):,} imágenes | "
            f"clases: {found_classes}"
        )

        if len(self.samples) == 0:
            log.error(f"[OA] Dataset vacío en '{split_dir}'.")
            return

        # ── Item-1: verificar número de clases ────────────────────────────
        if n_found == 5:
            log.warning(
                f"[OA] Se encontraron 5 clases (grados KL 0-4 sin consolidar). "
                f"El proyecto usa 3 clases consolidadas."
            )
        elif n_found < 3:
            log.warning(f"[OA] Solo {n_found} clase(s) encontradas — se esperaban 3.")
        else:
            log.debug(f"[OA] 3 clases ordinales confirmadas ✓")

        # ── Item-2: distribución por clase + pesos ────────────────────────
        total = len(self.samples)
        counts_arr = np.zeros(OA_N_CLASSES, dtype=float)
        for k, v in class_counts.items():
            try:
                idx = int(k)
                if 0 <= idx < OA_N_CLASSES:
                    counts_arr[idx] = v
            except ValueError:
                pass

        log.info(f"[OA] Distribución por clase:")
        for i, (name, count) in enumerate(zip(OA_CLASS_NAMES, counts_arr)):
            bar = "█" * max(1, int(30 * count / total))
            log.info(
                f"    Clase {i} ({name:<18}): {int(count):>5,} ({100 * count / total:.1f}%) {bar}"
            )

        counts_safe = np.maximum(counts_arr, 1)
        cw = torch.tensor(total / (OA_N_CLASSES * counts_safe), dtype=torch.float32)
        self.class_weights = cw
        log.info(
            f"[OA] H1 — class_weights CrossEntropyLoss: "
            + " | ".join(f"Clase{i}:{cw[i]:.2f}" for i in range(OA_N_CLASSES))
        )
        log.info(
            "[OA] H1 — Opciones de loss para FASE 2:\n"
            "    Opción A (pragmática): CrossEntropyLoss(weight=dataset.class_weights)\n"
            "    Opción B (ordinal): OrdinalLoss(n_classes=3)\n"
            "    Métrica principal: QWK (cohen_kappa_score, weights='quadratic')."
        )
        log.info("[OA] H1 — ⚠ Augmentation: NUNCA usar RandomVerticalFlip.")

        # ── H2: detectar augmentation offline ────────────────────────────
        self._audit_augmentation_offline(split, total)

        # ── Item-3/H2: verificar canales de imagen ──────────────────────
        self._verify_image_channels()

        # ── Item-8: verificar consistencia de dimensiones ────────────────
        self._verify_image_dimensions()

        # ── H3: documentar limitación de split ───────────────────────────
        log.info(
            f"[OA] H3 — Sin metadatos de paciente: split '{split}' tomado tal cual "
            f"de las carpetas del ZIP."
        )

        # ── H4: documentar el orden CLAHE → resize ──────────────────────
        log.info(
            f"[OA] H4 — CLAHE aplicado ANTES del resize. "
            f"Orden: cargar → CLAHE → resize({self.img_size}×{self.img_size}) → normalize."
        )

        # ── H5: documentar contaminación KL1 ────────────────────────────
        log.info(
            "[OA] H5 — Contaminación KL1 en clase 'Leve' (Clase 1): "
            "la frontera Clase0↔Clase1 será la más difícil de aprender. "
            "HERRAMIENTA: evaluate_boundary_confusion() para monitorear."
        )

        # ── Transform pipeline ────────────────────────────────────────────
        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _audit_augmentation_offline(self, split: str, total_imgs: int):
        """H2 — Detecta si el ZIP contiene augmentation offline."""
        expected = {"train": 5778, "val": 826, "test": 1656}.get(
            split, OA_BASE_IMG_COUNT
        )
        ratio = total_imgs / max(expected, 1)

        if ratio > 1.10:
            log.warning(
                f"[OA] H2 — El split '{split}' tiene {total_imgs:,} imágenes "
                f"vs {expected:,} esperadas (ratio={ratio:.2f}x). "
                f"Sugiere augmentation offline."
            )
            self._aug_offline_detected = True
        else:
            log.info(
                f"[OA] H2 — Conteo consistente con base OAI "
                f"({total_imgs:,} vs ~{expected:,}, ratio={ratio:.2f}x)."
            )
            self._aug_offline_detected = False

        if len(self.samples) >= 20:
            sample_paths = [p for p, _ in random.sample(self.samples, 20)]
            hashes = {}
            dup_count = 0
            for p in sample_paths:
                try:
                    md5 = hashlib.md5(p.read_bytes()).hexdigest()
                    if md5 in hashes:
                        dup_count += 1
                    else:
                        hashes[md5] = p.name
                except Exception:
                    pass
            if dup_count:
                log.warning(
                    f"[OA] H2 — {dup_count}/20 imágenes de muestra con hash MD5 duplicado."
                )

    def _verify_image_channels(self):
        """Item-4 — Verifica que las imágenes son grises guardadas como RGB."""
        if not self.samples:
            return
        n_check = min(5, len(self.samples))
        sample_paths = [p for p, _ in random.sample(self.samples, n_check)]
        n_gray_rgb = 0
        n_true_rgb = 0
        for p in sample_paths:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                if np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2) and np.allclose(
                    arr[:, :, 0], arr[:, :, 2], atol=2
                ):
                    n_gray_rgb += 1
                else:
                    n_true_rgb += 1
            except Exception:
                pass
        if n_true_rgb == 0:
            log.info(
                f"[OA] Item-4 — {n_gray_rgb}/{n_check} imágenes son grises guardadas como RGB ✓"
            )
        else:
            log.warning(
                f"[OA] Item-4 — {n_true_rgb}/{n_check} imágenes tienen 3 canales distintos."
            )

    def _verify_image_dimensions(self):
        """Item-8 — Verifica consistencia de dimensiones."""
        if not self.samples:
            return
        n_check = min(10, len(self.samples))
        sizes = set()
        for p, _ in random.sample(self.samples, n_check):
            try:
                sizes.add(Image.open(p).size)
            except Exception:
                pass
        if len(sizes) == 1:
            log.debug(
                f"[OA] Item-8 — Dimensiones uniformes en muestra: {next(iter(sizes))} ✓"
            )
        else:
            log.info(
                f"[OA] Item-8 — Dimensiones variables en muestra: {sizes}. "
                f"Resize a {self.img_size}×{self.img_size} compensará las diferencias."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kl_class = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(
                f"[OA] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero."
            )
            dummy = torch.zeros(3, self.img_size, self.img_size)
            return (
                dummy,
                self.expert_id if self.mode == "embedding" else kl_class,
                str(img_path.name),
            )

        # H4 — CLAHE ANTES del resize
        from transform_domain import apply_clahe

        img = apply_clahe(img)
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        if self.mode == "embedding":
            return self.base_transform(img), self.expert_id, str(img_path.name)
        else:
            if (
                self.mode == "expert"
                and not getattr(self, "_aug_offline_detected", False)
                and self.split == "train"
            ):
                img_tensor = self.aug_transform(img)
            else:
                img_tensor = self.base_transform(img)
            return img_tensor, kl_class, str(img_path.name)
