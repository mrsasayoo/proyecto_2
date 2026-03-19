# Pipeline de Embeddings — Proyecto MoE
## Incorporar Elementos de IA — Unidad II

---

## Arquitectura general

```
┌──────────────┐     ┌──────────────┐     ┌────────────────┐
│  5 Datasets  │ ──► │  FASE 0      │ ──► │  FASE 1        │
│  médicos     │     │  Embeddings  │     │  Ablation Study│
└──────────────┘     └──────────────┘     └────────────────┘
                       backbone ViT          4 routers
                       congelado             comparados
```

**6 expertos del MoE:**

| Experto | Dataset | Modalidad | Tarea clínica |
|---------|---------|-----------|---------------|
| 0 | NIH ChestXray14 | Rx 2D | Multi-label 14 patologías torácicas |
| 1 | ISIC 2019 | Dermatoscopia 2D | Multiclass 9 lesiones (8+UNK) |
| 2 | OA Rodilla | Rx 2D | Ordinal KL 0–4 (osteoartritis) |
| 3 | LUNA16 | CT 3D (parches) | Detección nódulo pulmonar |
| 4 | Pancreas PANORAMA | CT 3D (volúmenes) | Clasificación PDAC (k-fold) |
| 5 | OOD / Error | — | Fallback por entropía del router |

---

## Archivos

| Archivo | Qué hace |
|---|---|
| `fase0_extract_embeddings.py` | Extrae CLS tokens del backbone ViT y los guarda en disco |
| `fase1_ablation_router.py` | Carga los embeddings y compara 4 routers (ablation study) |

---

## Dependencias

```bash
pip install timm torch torchvision scikit-learn faiss-cpu SimpleITK opencv-python-headless pandas numpy
```

Con GPU CUDA:
```bash
pip install faiss-gpu   # en vez de faiss-cpu
```

---

## FASE 0 — Extracción de embeddings

### Descripción

Carga los 5 datasets médicos, pasa cada imagen/volumen por un backbone ViT
**congelado** (`requires_grad=False`) y guarda los CLS tokens (embeddings)
como arrays NumPy. Este paso se ejecuta **una sola vez** por configuración de backbone.

### Backbones disponibles

| Backbone | `--backbone` | d_model | VRAM estimada |
|----------|-------------|---------|---------------|
| ViT-Tiny | `vit_tiny_patch16_224` | 192 | ~2 GB |
| CvT-13 | `cvt_13` | 384 | ~3 GB |
| Swin-Tiny | `swin_tiny_patch4_window7_224` | 768 | ~4 GB |

> **Recomendación:** empezar con `vit_tiny_patch16_224` para iteración rápida.
> Para el ablation final del backbone, correr los 3 y comparar en FASE 1.

### Argumentos CLI

#### Generales

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--backbone` | `vit_tiny_patch16_224` | Backbone ViT (ver tabla arriba) |
| `--batch_size` | `64` | Tamaño de batch para extracción |
| `--workers` | `4` | Workers del DataLoader |
| `--output_dir` | `./embeddings` | Carpeta de salida |

#### NIH ChestXray14 (Experto 0)

| Argumento | Descripción |
|-----------|-------------|
| `--chest_csv` | `Data_Entry_2017.csv` con etiquetas multi-label |
| `--chest_imgs` | Carpeta con las imágenes `.png` |
| `--chest_train_list` | `train_val_list.txt` (split oficial) |
| `--chest_val_list` | `test_list.txt` (split oficial) |
| `--chest_view_filter` | Filtrar por vista: `PA`, `AP` o sin filtro (H4) |
| `--chest_bbox_csv` | `BBox_List_2017.csv` para heatmaps cualitativos (H5, opcional) |

#### ISIC 2019 (Experto 1)

| Argumento | Descripción |
|-----------|-------------|
| `--isic_gt` | `ISIC_2019_Training_GroundTruth.csv` |
| `--isic_imgs` | Carpeta con imágenes `.jpg` |
| `--isic_metadata` | `ISIC_2019_Training_Metadata.csv` — split por lesion_id (H2, recomendado) |

> **IMPORTANTE (H2):** sin `--isic_metadata`, el split es aleatorio por imagen y
> puede producir **leakage** (misma lesión con augmentación distinta en train y val).

#### OA Rodilla (Experto 2)

| Argumento | Descripción |
|-----------|-------------|
| `--oa_root` | Raíz con subcarpetas `train/`, `val/`, `test/` y subclases `0/`, `1/`, `2/` |

> Internamente aplica CLAHE a resolución original **antes** del resize (H4).
> Las augmentaciones no incluyen flip vertical (H3) para preservar orientación clínica.

#### LUNA16 (Experto 3)

| Argumento | Descripción |
|-----------|-------------|
| `--luna_patches` | Carpeta con parches 3D `.npy` pre-extraídos (`train/` y `val/`) |
| `--luna_csv` | `candidates_V2.csv` con coordenadas y etiquetas |

#### Pancreas PANORAMA (Experto 4) — Modo completo (recomendado)

| Argumento | Descripción |
|-----------|-------------|
| `--pancreas_nii_dir` | Directorio con los `.nii.gz` del ZIP de Zenodo (H1) |
| `--pancreas_labels_dir` | Directorio del repo clonado `panorama_labels` (H1) |
| `--pancreas_labels_commit` | Hash SHA del commit para reproducibilidad (H1) |
| `--pancreas_roi_strategy` | `A` (resize completo, default) o `B` (recorte Z[120:220]) (H2) |

```bash
# Preparar labels del Pancreas PANORAMA
git clone https://github.com/DIAGNijmegen/panorama_labels.git
cd panorama_labels && git rev-parse HEAD   # guardar el hash
```

#### Pancreas — Modo legado (parches pre-procesados)

| Argumento | Descripción |
|-----------|-------------|
| `--pancreas_patches_train` | Carpeta con `.npy` pre-procesados de train |
| `--pancreas_patches_val` | Carpeta con `.npy` pre-procesados de val |

> Si se proveen `--pancreas_nii_dir` y `--pancreas_labels_dir`, el modo completo
> tiene prioridad sobre el legado.

### Ejemplo de ejecución mínima (solo datos 2D)

```bash
python fase0_extract_embeddings.py \
  --backbone vit_tiny_patch16_224 \
  --batch_size 64 \
  --output_dir ./embeddings/vit_tiny \
  \
  --chest_csv     /data/chestxray14/Data_Entry_2017.csv \
  --chest_imgs    /data/chestxray14/images/ \
  --chest_train_list /data/chestxray14/train_val_list.txt \
  --chest_val_list   /data/chestxray14/test_list.txt \
  --chest_view_filter PA \
  \
  --isic_gt       /data/isic2019/ISIC_2019_Training_GroundTruth.csv \
  --isic_imgs     /data/isic2019/ISIC_2019_Training_Input/ \
  --isic_metadata /data/isic2019/ISIC_2019_Training_Metadata.csv \
  \
  --oa_root       /data/osteoarthritis/
```

### Ejemplo de ejecución completa (5 datasets)

```bash
python fase0_extract_embeddings.py \
  --backbone vit_tiny_patch16_224 \
  --batch_size 64 \
  --output_dir ./embeddings/vit_tiny \
  \
  --chest_csv     /data/chestxray14/Data_Entry_2017.csv \
  --chest_imgs    /data/chestxray14/images/ \
  --chest_train_list /data/chestxray14/train_val_list.txt \
  --chest_val_list   /data/chestxray14/test_list.txt \
  --chest_view_filter PA \
  --chest_bbox_csv   /data/chestxray14/BBox_List_2017.csv \
  \
  --isic_gt       /data/isic2019/ISIC_2019_Training_GroundTruth.csv \
  --isic_imgs     /data/isic2019/ISIC_2019_Training_Input/ \
  --isic_metadata /data/isic2019/ISIC_2019_Training_Metadata.csv \
  \
  --oa_root       /data/osteoarthritis/ \
  \
  --luna_patches  /data/luna16/patches_3d/ \
  --luna_csv      /data/luna16/candidates_V2.csv \
  \
  --pancreas_nii_dir     /data/pancreas/panorama_images/ \
  --pancreas_labels_dir  /data/pancreas/panorama_labels/ \
  --pancreas_labels_commit abc123def456 \
  --pancreas_roi_strategy A
```

### Ablation del backbone (3 corridas)

```bash
for bb in vit_tiny_patch16_224 swin_tiny_patch4_window7_224 cvt_13; do
  python fase0_extract_embeddings.py \
    --backbone $bb \
    --output_dir ./embeddings/$bb \
    --chest_csv /data/chestxray14/Data_Entry_2017.csv \
    --chest_imgs /data/chestxray14/images/ \
    --chest_train_list /data/chestxray14/train_val_list.txt \
    --chest_val_list /data/chestxray14/test_list.txt \
    --isic_gt /data/isic2019/ISIC_2019_Training_GroundTruth.csv \
    --isic_imgs /data/isic2019/ISIC_2019_Training_Input/ \
    --oa_root /data/osteoarthritis/ \
    --luna_patches /data/luna16/patches_3d/ \
    --luna_csv /data/luna16/candidates_V2.csv \
    --pancreas_nii_dir /data/pancreas/panorama_images/ \
    --pancreas_labels_dir /data/pancreas/panorama_labels/
done
```

### Output generado

```
<output_dir>/
├── Z_train.npy        ← [N_train, d_model]  embeddings de train
├── Z_val.npy          ← [N_val,   d_model]  embeddings de val
├── y_train.npy        ← [N_train]            etiqueta de experto (0-4)
├── y_val.npy          ← [N_val]              etiqueta de experto (0-4)
├── names_train.txt    ← nombres de imagen/volumen (para auditoría)
├── names_val.txt
└── backbone_meta.json ← {backbone, d_model, vram_gb, timestamp}
```

> `d_model` depende del backbone elegido: 192 (ViT-Tiny), 384 (CvT-13) o 768 (Swin-Tiny).

---

## FASE 1 — Ablation Study del Router

### Descripción

Carga los embeddings de FASE 0 y entrena/evalúa 4 mecanismos de routing distintos.
Produce una tabla comparativa y exporta resultados a JSON.

### Routers comparados

| Router | Método | Características |
|--------|--------|-----------------|
| **Linear+Softmax** | `nn.Linear(d_model, 6)` + Softmax | Baseline entrenable, SGD con LR decay |
| **GMM** | Gaussian Mixture Model (5 componentes) | Estadístico, sin entrenamiento iterativo |
| **Naive Bayes** | Gaussian NB | Ultra-rápido, baseline probabilístico |
| **kNN-FAISS** | k-Nearest Neighbors con índice FAISS | Escalable, sin entrenamiento |

> Los 4 routers clasifican en **6 clases** (5 dominios + 1 OOD).
> El Experto 5 (OOD) se asigna por **umbral de entropía** calibrado sobre val.

### Argumentos CLI

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--embeddings` | `./embeddings` | Carpeta con `Z_train.npy`, `Z_val.npy` y `backbone_meta.json` |
| `--epochs` | `50` | Épocas para el router Linear+Softmax |
| `--knn_k` | `5` | Número de vecinos para kNN-FAISS |

### Ejemplo de ejecución

```bash
python fase1_ablation_router.py \
  --embeddings ./embeddings/vit_tiny \
  --epochs 50 \
  --knn_k 5
```

### Output

- **Tabla comparativa** (impresa en terminal): accuracy, F1-macro, latencia, balance de carga por experto
- **`ablation_results.json`**: resultados exportados para análisis posterior

---

## Prerequisito para datasets 3D: extracción de parches

LUNA16 y Pancreas requieren extraer parches 3D **antes** de FASE 0 (si se usa modo legado).

### Conversión world → vóxel (CRÍTICO)

```python
import SimpleITK as sitk
import numpy as np

def extract_patch(mhd_path, coord_world, patch_size=64):
    image  = sitk.ReadImage(mhd_path)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    array  = sitk.GetArrayFromImage(image)   # [Z, Y, X] en HU

    # Conversión world → vóxel con dirección (¡NO solo dividir por spacing!)
    coord_shifted = coord_world - origin
    coord_voxel = np.linalg.solve(direction * spacing, coord_shifted)
    z, y, x = np.round(coord_voxel).astype(int)

    half = patch_size // 2
    patch = array[
        max(0, z-half):z+half,
        max(0, y-half):y+half,
        max(0, x-half):x+half
    ]

    # Padding si el candidato está cerca del borde
    if patch.shape != (patch_size, patch_size, patch_size):
        pad_z = patch_size - patch.shape[0]
        pad_y = patch_size - patch.shape[1]
        pad_x = patch_size - patch.shape[2]
        patch = np.pad(patch,
                       ((0, pad_z), (0, pad_y), (0, pad_x)),
                       constant_values=-1000)   # -1000 HU = aire
    return patch   # [64, 64, 64] en HU crudo
```

### Normalización HU por tejido

| Dataset | Rango HU | Concepto |
|---------|----------|----------|
| LUNA16 (pulmón) | [-1000, 400] | Aire → tejido blando |
| Pancreas (abdomen) | [-100, 400] | Grasa → hueso suave |

### Volúmenes 3D → input ViT 2D

`volume_to_vit_input()` toma 3 slices centrales (axial, coronal, sagital) y los
apila como canales RGB. Esto es suficiente para el routing. Los expertos 3D
usarán arquitecturas 3D reales (R3D-18, Swin3D-Tiny) en FASE 2.

---

## Hallazgos clave implementados

### ChestXray14 (Experto 0)
- **H1:** Multi-label con BCEWithLogitsLoss (14 sigmoides, no softmax)
- **H4:** Filtro de vista PA/AP via `--chest_view_filter`
- **H5:** BBox para validación cualitativa via `--chest_bbox_csv`

### ISIC 2019 (Experto 1)
- **H1:** 9 neuronas softmax (8 clases + UNK slot)
- **H2:** Split por lesion_id via `--isic_metadata` (anti-leakage)
- **H5:** class_weights inversamente proporcionales a frecuencia

### OA Rodilla (Experto 2)
- **H3:** Sin flip vertical en augmentación (preservar orientación clínica)
- **H4:** CLAHE a resolución original antes de resize
- **H5:** OrdinalLoss y evaluación con QWK (Quadratic Weighted Kappa)

### LUNA16 (Experto 3)
- **H3:** Conversión world→vóxel con matrix de dirección
- **H4:** FocalLoss para desbalance nódulo/no-nódulo
- **H5:** FROC + CPM como métrica de evaluación

### Pancreas PANORAMA (Experto 4)
- **H1:** Pipeline completo NIfTI + labels de GitHub (cross-match por case_id)
- **H2:** Estrategia ROI configurable (A: resize, B: recorte Z)
- **H3:** k-fold CV por centro (anti sesgo multicéntrico)
- **H4:** Normalización HU abdominal clip → z-score → [0,1]

---

## Notas técnicas

1. **El backbone siempre congelado en FASE 0** — `requires_grad=False` en todos
   los parámetros. Solo extrae, no aprende.

2. **La etiqueta de experto es supervisión débil** — indica "esta imagen viene del
   dataset del Experto N", no la patología. El router aprende el *tipo de imagen*.

3. **Z_train y Z_val se guardan una sola vez** — FASE 1 trabaja sobre estos archivos
   sin tocar el backbone.

4. **backbone_meta.json** almacena el nombre del backbone, d_model, VRAM estimada
   y timestamp. FASE 1 lo usa para dimensionar automáticamente los routers.

5. **FASE 1 detecta d_model automáticamente** desde `Z_train.shape[1]`, así que
   no necesita saber qué backbone se usó.
