# ISIC Pipeline Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reemplazar el pipeline de preprocesamiento y augmentación del dataset ISIC 2019 con la solución ganadora de ISIC 2020 (DullRazor + Color Constancy online + augmentaciones basadas en la literatura, CutMix/MixUp en el loop, FocalLoss multiclase).

**Architecture:**
- Fase offline (ejecutar una vez): auditoría → hair removal (DullRazor) → resize aspect-ratio-preserving → caché JPEG 95.
- Fase online (por epoch): RandomCrop → flips → rotation 360° → ColorJitter → RandomGamma → ShadesOfGray (p=0.5) → CoarseDropout → ToTensor → Normalize.
- CutMix/MixUp en el training loop (batch level). FocalLossMultiClass reemplaza CrossEntropyLoss.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, PIL/Pillow, NumPy, OpenCV (cv2), pandas. NO se añade albumentations como dependencia.

**Python:** `/home/mrsasayo_mesa/venv_global/bin/python`
**Directorio raíz:** `/mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/`
**Pipeline root:** `src/pipeline/`

---

## Contexto de referencia clave

```
EXPERT2_IMG_SIZE = 224       # target_size para crop cuadrado
EXPERT2_NUM_CLASSES = 8      # MEL, NV, BCC, AK, BKL, DF, VASC, SCC
EXPERT2_BATCH_SIZE = 32
EXPERT2_ACCUMULATION_STEPS = 3

ISIC classes y counts aprox:
  NV=12875, MEL=4522, BCC=3323, BKL=2624, AK=867, SCC=628, DF=239, VASC=142
  → ratio máximo ~91:1 (NV:VASC)

Rutas offline cache: datasets/isic_2019/ISIC_2019_Training_Input_preprocessed/
Rutas imágenes originales: datasets/isic_2019/ISIC_2019_Training_Input/
Splits: splits/isic_train.csv, splits/isic_val.csv, splits/isic_test.csv
```

Color Constancy se aplica ONLINE (p=0.5), NO offline, porque ConvNeXt-Small (>20M params) necesita la variabilidad cromática para generalizar.

---

## Tarea 1: `losses.py` — Añadir FocalLossMultiClass

**Files:**
- Modify: `src/pipeline/fase2/losses.py`

El `FocalLoss` existente es binario (para LUNA16). Hay que añadir un `FocalLossMultiClass` para ISIC multiclase.

**Implementación a añadir al final del archivo:**

```python
class FocalLossMultiClass(nn.Module):
    """
    Focal Loss multiclase para ISIC 2019 (Lin et al., ICCV 2017).

    Reduce el gradiente de ejemplos fáciles y concentra el aprendizaje
    en ejemplos difíciles de clases minoritarias (DF, VASC, AK, SCC).

    Parámetros ISIC:
        gamma=2.0: concentración estándar
        weight: tensor[8] con pesos inverse-frequency por clase
        label_smoothing=0.1: reduce overconfidence

    Uso:
        criterion = FocalLossMultiClass(gamma=2.0, weight=class_weights, label_smoothing=0.1)
        loss = criterion(logits, labels)   # logits: [B,8], labels: [B] int
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE por muestra con pesos y label smoothing
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = exp(-CE) ≈ probabilidad de la clase correcta
        pt = torch.exp(-ce)
        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()
```

**Verificación:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python -c "
import sys; sys.path.insert(0, 'src/pipeline')
import ast; ast.parse(open('src/pipeline/fase2/losses.py').read()); print('Sintaxis OK')
import torch
from fase2.losses import FocalLossMultiClass
crit = FocalLossMultiClass(gamma=2.0)
logits = torch.randn(4, 8)
labels = torch.randint(0, 8, (4,))
loss = crit(logits, labels)
print(f'FocalLossMultiClass OK: loss={loss.item():.4f}')
"
```

---

## Tarea 2: `isic.py` — Reescribir con el nuevo pipeline

**Files:**
- Rewrite: `src/pipeline/datasets/isic.py`

### Qué se elimina del archivo actual:
- `build_isic_transforms()` (función con 3 variantes: embedding/standard/minority) → ELIMINAR
- `MINORITY_IDX` → ELIMINAR
- Lógica `transform_standard` / `transform_minority` en `__getitem__` → ELIMINAR
- Todas las referencias a UNK (slot 8, H5) → ELIMINAR (ahora son 8 clases exactas)
- `RandAugment`, `RandomGrayscale`, `RandomErasing` inline → ELIMINAR
- `RandomRotation(30)` → REEMPLAZAR por 360°
- `RandomVerticalFlip(0.3)` → REEMPLAZAR por p=0.5
- `Resize((256,256))` → ELIMINAR (las imágenes vienen precacheadas con shorter_side=224)

### Qué se mantiene / refactoriza:
- `ISICDataset` clase: mantener con modo `"embedding"` y `"expert"`, simplificar
- `build_lesion_split()`: mantener (H2+H4, split por lesion_id + dedup MD5)
- `get_weighted_sampler()`: mantener
- `_source_audit()`: mantener (útil para logging)
- `KNOWN_DUPLICATES`: mantener (todavía usado en build_lesion_split)
- Import de `apply_circular_crop` desde `fase1.transform_domain`: MANTENER (se aplica a imágenes originales si no hay caché)

### Nuevas clases a implementar:

**`RandomGamma`:**
```python
class RandomGamma:
    """Corrección gamma aleatoria: I' = I^γ con γ ∈ gamma_range."""
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p
    def __call__(self, img):  # img: PIL Image
        if np.random.random() < self.p:
            gamma = np.random.uniform(*self.gamma_range)
            return transforms.functional.adjust_gamma(img, gamma)
        return img
```

**`ShadesOfGray`:**
```python
class ShadesOfGray:
    """Color Constancy: normalización del iluminante por norma de Minkowski (p=6).
    Aplica corrección von Kries diagonal para simular luz blanca canónica.
    Usar con p=0.5 online para ConvNeXt-Small (>20M params)."""
    def __init__(self, power=6):
        self.power = power
    def __call__(self, img):  # img: PIL Image → PIL Image
        arr = np.array(img, dtype=np.float32)
        illuminant = np.mean(arr ** self.power, axis=(0, 1)) ** (1.0 / self.power)
        illuminant = illuminant / (np.mean(illuminant) + 1e-6)
        corrected = arr / (illuminant + 1e-6)
        return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))
```

**`CoarseDropout`:**
```python
class CoarseDropout:
    """Elimina 1–3 parches rectangulares aleatorios (CutOut / CoarseDropout).
    Opera sobre PIL Image ANTES de ToTensor."""
    def __init__(self, n_holes_range=(1, 3), size_range=(32, 96),
                 fill_value=(int(0.485*255), int(0.456*255), int(0.406*255)), p=0.5):
        self.n_holes_range = n_holes_range
        self.size_range = size_range
        self.fill_value = fill_value
        self.p = p
    def __call__(self, img):  # img: PIL Image
        if np.random.random() >= self.p:
            return img
        arr = np.array(img)
        h, w = arr.shape[:2]
        n_holes = np.random.randint(self.n_holes_range[0], self.n_holes_range[1] + 1)
        for _ in range(n_holes):
            hole_h = np.random.randint(self.size_range[0], self.size_range[1] + 1)
            hole_w = np.random.randint(self.size_range[0], self.size_range[1] + 1)
            y = np.random.randint(0, max(1, h - hole_h))
            x = np.random.randint(0, max(1, w - hole_w))
            arr[y:y+hole_h, x:x+hole_w] = self.fill_value
        return Image.fromarray(arr)
```

### Nuevos TRANSFORM_TRAIN, TRANSFORM_VAL, TRANSFORM_EMBEDDING:

```python
TARGET_SIZE = 224  # debe coincidir con EXPERT2_IMG_SIZE

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(TARGET_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(
        degrees=360,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0,
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomGamma(gamma_range=(0.7, 1.5), p=0.5),
    transforms.RandomApply([ShadesOfGray(power=6)], p=0.5),
    CoarseDropout(n_holes_range=(1, 3), size_range=(32, 96), p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_VAL = transforms.Compose([
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Para modo "embedding" (Fase 0) — sin augmentación
TRANSFORM_EMBEDDING = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**NOTA importante sobre TRANSFORM_TRAIN:** Las imágenes cacheadas tienen el lado corto = 224 px (son rectangulares). `RandomCrop(224)` extrae el cuadrado cuadrado. Si la imagen NO está cacheada (se carga la original de alta resolución), el crop podría ser enorme o muy pequeño. Para manejar fallback a imágenes originales, `ISICDataset.__getitem__` debe hacer primero un resize (shorter_side → 224) antes de aplicar TRANSFORM_TRAIN si la imagen no está en caché. Ver ISICDataset para el manejo de cache_dir.

### Funciones helper CutMix / MixUp (a añadir en isic.py):

```python
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: recorta región rectangular de imagen A y la pega en imagen B.
    Returns: mixed_x, y_a, y_b, lam
    Usar en training loop como: loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    rand_index = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[rand_index]
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x = x.clone()
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    return x, y_a, y_b, lam


def mixup_data(x, y, alpha=0.4):
    """
    MixUp: combina linealmente dos imágenes y sus etiquetas.
    Returns: mixed_x, y_a, y_b, lam
    Usar en training loop como: loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    rand_index = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[rand_index]
    mixed_x = lam * x + (1.0 - lam) * x[rand_index]
    return mixed_x, y_a, y_b, lam
```

### ISICDataset — simplificación:

La clase `ISICDataset` se simplifica así:
- Constructor: recibe `img_dir` (imágenes originales), `cache_dir` (opcional, imágenes preprocesadas), `split_df`, `mode`, `split`, `transform`, `apply_bcn_crop`.
- Si `cache_dir` está configurado y la imagen existe en caché → cargar desde caché (ya tiene shorter_side=224, no necesita Resize previo).
- Si la imagen NO está en caché o `cache_dir` es None → cargar desde `img_dir` y hacer `Resize` a shorter_side=224 antes de aplicar el transform.
- `mode="embedding"`: usa `TRANSFORM_EMBEDDING` → devuelve `(tensor, expert_id, img_name)`
- `mode="expert"` + `split="train"`: usa `TRANSFORM_TRAIN` → devuelve `(tensor, label_int, img_name)`
- `mode="expert"` + `split in ("val","test")`: usa `TRANSFORM_VAL` → devuelve `(tensor, label_int, img_name)`
- Eliminar toda la lógica de `transform_standard`, `transform_minority`, `is_minority`, `MINORITY_IDX`.
- Eliminar manejo de `UNK` (ya no hay clase 9).
- Mantener `class_weights` (inverse-frequency, tensor[8]).
- Mantener `_source_audit()`, `build_lesion_split()`, `get_weighted_sampler()`.
- Añadir `KNOWN_DUPLICATES` (usado en build_lesion_split).

**Helper interno para resize:**
```python
@staticmethod
def _resize_shorter_side(img: Image.Image, target_size: int) -> Image.Image:
    """Resize manteniendo aspecto: lado corto = target_size."""
    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)
```

**Verificación final:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python -c "
import sys; sys.path.insert(0, 'src/pipeline')
import ast; ast.parse(open('src/pipeline/datasets/isic.py').read()); print('Sintaxis OK')
from datasets.isic import (ISICDataset, TRANSFORM_TRAIN, TRANSFORM_VAL,
                            TRANSFORM_EMBEDDING, cutmix_data, mixup_data,
                            RandomGamma, ShadesOfGray, CoarseDropout)
import torch
# Test transforms con tensor dummy PIL
from PIL import Image; import numpy as np
img = Image.fromarray(np.random.randint(0,255,(280,224,3), dtype=np.uint8))
t = TRANSFORM_TRAIN(img)
assert t.shape == (3, 224, 224), f'Train shape error: {t.shape}'
img2 = Image.fromarray(np.random.randint(0,255,(280,224,3), dtype=np.uint8))
v = TRANSFORM_VAL(img2)
assert v.shape == (3, 224, 224), f'Val shape error: {v.shape}'
# Test cutmix/mixup
x = torch.randn(4,3,224,224); y = torch.randint(0,8,(4,))
mx, ya, yb, lam = cutmix_data(x, y)
assert mx.shape == x.shape
mx2, ya2, yb2, lam2 = mixup_data(x, y)
assert mx2.shape == x.shape
print('All checks passed')
"
```

---

## Tarea 3: `pre_isic.py` — Script de preprocesamiento offline

**Files:**
- Create: `src/pipeline/fase0/pre_isic.py`

Script standalone que preprocesa el dataset ISIC 2019 una sola vez y guarda en caché.

### Funciones a implementar:

**1. `audit_isic_dataset(img_dir, gt_csv, out_csv)`**
- Lee gt_csv y por cada imagen registra: image_id, width, height, aspect_ratio, source (HAM/BCN/MSK), has_circular_frame (si aspect ~1:1 y tiene borde oscuro).
- Guarda en out_csv.
- Devuelve DataFrame.

**2. `shades_of_gray(img_array, power=6)` → np.ndarray**
- Implementación Shades of Gray como función pura sobre arrays numpy.

**3. `remove_hair_dullrazor(img_array)` → np.ndarray**
- DullRazor: grayscale → morphological closing (kernel 3x3) → detección píxeles con diferencia > threshold=10 → inpainting con cv2.INPAINT_TELEA (radio=3).
- Fallback si cv2 no disponible: retornar imagen original con warning.

**4. `resize_shorter_side(img_array, target_size=224)` → np.ndarray**
- Resize manteniendo aspect ratio: lado corto = target_size con LANCZOS (cv2.INTER_LANCZOS4).

**5. `preprocess_isic_dataset(img_dir, out_dir, gt_csv, target_size=224, quality=95, apply_hair_removal=True, max_workers=4)`**
- Orquestador: lee gt_csv → por cada imagen: load → hair removal → resize → guardar como `{isic_id}_pp_{target_size}.jpg` con calidad 95.
- Usa `concurrent.futures.ThreadPoolExecutor(max_workers)` para paralelismo.
- Progreso con tqdm o conteo simple cada 500 imágenes.
- Idempotente: si archivo ya existe en out_dir, lo salta.
- Guarda `preprocess_report.json` en out_dir al finalizar.
- NOTA: Color Constancy NO se aplica aquí (se aplica online con p=0.5).

**CLI:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesamiento offline ISIC 2019")
    parser.add_argument("--img_dir", required=True, help="Dir con imágenes .jpg originales")
    parser.add_argument("--out_dir", required=True, help="Dir de salida para caché")
    parser.add_argument("--gt_csv", required=True, help="CSV ground truth ISIC 2019")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--no_hair_removal", action="store_true")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true", help="Procesar solo 10 imgs")
    args = parser.parse_args()
    ...
```

**Verificación:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python -c "
import sys; sys.path.insert(0, 'src/pipeline')
import ast; ast.parse(open('src/pipeline/fase0/pre_isic.py').read()); print('Sintaxis OK')
from fase0.pre_isic import shades_of_gray, remove_hair_dullrazor, resize_shorter_side
import numpy as np
img = np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
out1 = shades_of_gray(img)
assert out1.shape == img.shape, 'shades_of_gray shape error'
out2 = remove_hair_dullrazor(img)
assert out2.shape == img.shape, 'hair_removal shape error'
out3 = resize_shorter_side(img, 224)
assert min(out3.shape[:2]) == 224, f'resize error: {out3.shape}'
print('pre_isic.py: All checks passed')
"
```

---

## Tarea 4: `dataloader_expert2.py` — WeightedRandomSampler + cache_dir

**Files:**
- Rewrite: `src/pipeline/fase2/dataloader_expert2.py`

### Cambios principales:

1. **Eliminar** `transform_standard` / `transform_minority` como parámetros → ya no existen en ISICDataset.
2. **Añadir** `cache_dir` parámetro: si existe y tiene imágenes preprocesadas, se usa; si no, fallback a `img_dir`.
3. **Usar `WeightedRandomSampler`** en el train DataLoader (no `shuffle=True`).
4. ISICDataset en mode="expert": ya maneja la selección de TRANSFORM_TRAIN vs TRANSFORM_VAL internamente según `split`.

```python
# Rutas con caché
_ISIC_CACHE_DIR = _PROJECT_ROOT / "datasets" / "isic_2019" / "ISIC_2019_Training_Input_preprocessed"

def build_dataloaders_expert2(..., cache_dir=None, ...):
    ...
    # Determinar directorio de imágenes
    effective_img_dir = ...  # cache_dir si existe, sino img_dir
    
    train_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,   # nuevo parámetro
        split_df=train_df,
        mode="expert",
        split="train",
        apply_bcn_crop=True,
    )
    
    # WeightedRandomSampler para train
    sampler = ISICDataset.get_weighted_sampler(train_df, ISICDataset.CLASSES)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # reemplaza shuffle=True
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    # Val y test: sin sampler, shuffle=False
    ...
```

**Verificación:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python -c "
import sys; sys.path.insert(0, 'src/pipeline')
import ast; ast.parse(open('src/pipeline/fase2/dataloader_expert2.py').read()); print('Sintaxis OK')
"
```

---

## Tarea 5: `train_expert2.py` — CutMix/MixUp en training loop + FocalLossMultiClass

**Files:**
- Modify: `src/pipeline/fase2/train_expert2.py`

### Cambios en `train_one_epoch()`:

Añadir CutMix (p=0.3) y MixUp (p=0.2) como augmentaciones de batch:

```python
from datasets.isic import cutmix_data, mixup_data

# En train_one_epoch, parámetros nuevos:
def train_one_epoch(
    ...,
    cutmix_prob: float = 0.3,
    mixup_prob: float = 0.2,
    ...
):
    ...
    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.long().to(device)
        
        # Selección de augmentación de batch (mutuamente excluyentes)
        r = np.random.random()
        use_cutmix = r < cutmix_prob
        use_mixup = (not use_cutmix) and (r < cutmix_prob + mixup_prob)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            if use_cutmix:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            elif use_mixup:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels)
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        ...
```

### Cambio de loss en `train()`:

Reemplazar `nn.CrossEntropyLoss` con `FocalLossMultiClass`:

```python
from fase2.losses import FocalLossMultiClass
...
criterion = FocalLossMultiClass(
    gamma=2.0,
    weight=class_weights,
    label_smoothing=EXPERT2_LABEL_SMOOTHING,
)
```

### Firma actualizada de `_run_phase()`:
El parámetro `cutmix_prob` y `mixup_prob` se pasan a `train_one_epoch()`. Los valores por defecto (0.3 / 0.2) pueden hardcodearse en `_run_phase()`.

**Verificación:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python -c "
import sys; sys.path.insert(0, 'src/pipeline')
import ast; ast.parse(open('src/pipeline/fase2/train_expert2.py').read()); print('Sintaxis OK')
"
# Dry-run completo:
cd /mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/src/pipeline/fase2
/home/mrsasayo_mesa/venv_global/bin/python train_expert2.py --dry-run 2>&1 | head -30
```

---

## Orden de ejecución

1. Tarea 1: `losses.py` (independiente, sin deps)
2. Tarea 2: `isic.py` (depende de nada; provee TRANSFORM_TRAIN etc.)
3. Tarea 3: `pre_isic.py` (independiente; usa numpy/cv2/PIL)
4. Tarea 4: `dataloader_expert2.py` (depende de isic.py Tarea 2)
5. Tarea 5: `train_expert2.py` (depende de losses.py Tarea 1 + isic.py Tarea 2)

Verificación final de integración: sintaxis AST de todos los archivos modificados.

---

## ✅ Completación

**Fecha de completación:** 2026-04-14

**Estado:** Todas las 5 tareas implementadas y verificadas.

| Tarea | Archivo | Estado |
|---|---|---|
| 1 | `src/pipeline/fase2/losses.py` | ✅ `FocalLossMultiClass` añadida. Verificada con forward pass sintético |
| 2 | `src/pipeline/datasets/isic.py` | ✅ Reescrito con `RandomGamma`, `ShadesOfGray`, `CoarseDropout`, `cutmix_data`, `mixup_data`, `TRANSFORM_TRAIN/VAL/EMBEDDING`, soporte `cache_dir` |
| 3 | `src/pipeline/fase0/pre_isic.py` | ✅ Creado: auditoría + DullRazor + resize aspect-ratio + caché JPEG |
| 4 | `src/pipeline/fase2/dataloader_expert2.py` | ✅ `WeightedRandomSampler` reemplaza `shuffle=True`; soporte `cache_dir` |
| 5 | `src/pipeline/fase2/train_expert2.py` | ✅ CutMix (p=0.3) + MixUp (p=0.2) en training loop; `FocalLossMultiClass` reemplaza `CrossEntropyLoss` |

**Cambio adicional:** `src/pipeline/fase0/fase0_pipeline.py` — nuevo paso 6 "ISIC Preprocesamiento" insertado; pasos renumerados (viejo 6→7, 7→8, 8→9); ahora 10 pasos (0–9).

**Verificación:** Sintaxis AST de los 5 archivos modificados/creados pasó sin errores. Dry-run de `train_expert2.py` verificado.
