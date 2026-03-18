# Pipeline de Embeddings — Proyecto MoE
## Incorporar Elementos de IA — Unidad II

---

## Archivos

| Archivo | Qué hace |
|---|---|
| `fase0_extract_embeddings.py` | Extrae CLS tokens del backbone ViT y los guarda en disco |
| `fase1_ablation_router.py` | Carga los embeddings y compara los 4 routers (ablation study) |

---

## Orden de ejecución

### Paso 0: Instalar dependencias

```bash
pip install timm torch torchvision scikit-learn faiss-cpu SimpleITK opencv-python-headless pandas numpy
```

Si tienes GPU con CUDA:
```bash
pip install faiss-gpu   # en vez de faiss-cpu
```

---

### Paso 1: Extraer embeddings (FASE 0)

```bash
python fase0_extract_embeddings.py \
  --backbone vit_tiny_patch16_224 \
  --batch_size 64 \
  --output_dir ./embeddings \
  \
  --chest_csv     /data/chestxray14/Data_Entry_2017.csv \
  --chest_imgs    /data/chestxray14/images/ \
  --chest_train_list /data/chestxray14/train_val_list.txt \
  --chest_val_list   /data/chestxray14/test_list.txt \
  \
  --isic_gt       /data/isic2019/ISIC_2019_Training_GroundTruth.csv \
  --isic_imgs     /data/isic2019/ISIC_2019_Training_Input/ \
  \
  --oa_root       /data/osteoarthritis/ \
  \
  --luna_patches  /data/luna16/patches_3d/ \
  --luna_csv      /data/luna16/candidates_V2.csv \
  \
  --pancreas_patches_train /data/pancreas/patches_3d/train/ \
  --pancreas_patches_val   /data/pancreas/patches_3d/val/
```

**Output generado en `./embeddings/`:**
```
Z_train.npy     ← [N_train, 192]  embeddings de train
Z_val.npy       ← [N_val,   192]  embeddings de val
y_train.npy     ← [N_train]       etiqueta de experto (0-4)
y_val.npy       ← [N_val]         etiqueta de experto (0-4)
names_train.txt ← nombres de imagen (para auditoría)
names_val.txt
```

---

### Paso 2: Ablation study del router (FASE 1)

```bash
python fase1_ablation_router.py \
  --embeddings ./embeddings \
  --epochs 50
```

**Output:** tabla comparativa de los 4 routers + `ablation_results.json`

---

## Prerequisito para LUNA16 y Pancreas: extracción de parches 3D

Los datasets 3D requieren extraer parches centrados en cada candidato
**antes** de correr FASE 0. Pasos básicos:

```python
import SimpleITK as sitk
import numpy as np

def extract_patch(mhd_path, coord_world, patch_size=64):
    image  = sitk.ReadImage(mhd_path)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    array  = sitk.GetArrayFromImage(image)   # [Z, Y, X] en HU

    # Convertir world → vóxel (¡CRÍTICO! no dividir solo por spacing)
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

Guardar cada parche como `candidate_XXXXXX.npy` en la carpeta
`patches_3d/train/` o `patches_3d/val/` según el split por `seriesuid`.

---

## Notas importantes del proyecto

1. **El backbone siempre congelado en FASE 0** — `requires_grad=False` en todos
   los parámetros. Solo extrae, no aprende.

2. **La etiqueta de experto es supervisión débil** — no es la etiqueta de
   patología sino simplemente "esta imagen viene del dataset del Experto N".
   El router aprende a reconocer el *tipo de imagen*, no la enfermedad.

3. **Z_train y Z_val se guardan una sola vez** — el ablation study (4 routers)
   trabaja sobre estos archivos sin tocar el ViT de nuevo.

4. **Para volúmenes 3D**, `volume_to_vit_input()` toma 3 slices centrales
   (axial, coronal, sagital) y los apila como canales RGB. Esto es suficiente
   para el routing — los expertos 3D usan arquitecturas 3D reales (R3D-18,
   Swin3D-Tiny) para la clasificación.

5. **Backbone recomendado:** `vit_tiny_patch16_224` (d_model=192, ~5.7M params).
   Si el clúster tiene más VRAM disponible: `swin_tiny_patch4_window7_224`.
