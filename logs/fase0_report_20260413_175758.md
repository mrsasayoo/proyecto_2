# Fase 0 — Reporte de Preparación de Datos

## Estado por paso

| Paso | Descripción | Estado | Tiempo |
|------|-------------|--------|--------|
| 0 | Prerequisites | ✅ | 0.1s |
| 1 | Descargar datasets | ✅ | 1430.6s |
| 2 | Extraer archivos | ✅ | 1.0s |
| 3 | Post-procesado NIH | ✅ | 5.6s |
| 4 | Etiquetas páncreas | ✅ | 0.0s |
| 5 | Splits 80/10/10 | ✅ | 1.8s |
| 6 | Datos 3D | — | 1560.9s |
| 7 | DenseNet3D | ✅ | 2.6s |
| 8 | Reporte | — | — |

## Datasets activos

- isic
- luna_ct
- luna_meta
- nih
- oa
- pancreas
- panorama

## Splits generados

### NIH
- status: ✅
- skipped: True
- nih_train_list.txt: 88999
- nih_val_list.txt: 11349
- nih_test_list.txt: 11772

### ISIC
- status: ✅
- skipped: True

### OA
- status: ✅
- skipped: True

### LUNA
- status: ✅
- skipped: True

### PANCREAS
- status: ✅
- skipped: True

### CAE
- status: ✅
- skipped: True


## Comando para Fase 1

> **Nota:** El backbone mostrado es el valor por defecto (`vit_tiny_patch16_224`). Editar antes de ejecutar si se desea usar `swin_tiny_patch4_window7_224` o `cvt_13`.

```bash
python3 /mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/src/pipeline/fase1/fase1_pipeline.py \
    --backbone vit_tiny_patch16_224 \
    --batch_size 256 --workers 8 \
    --output_dir embeddings/vit_tiny \
    --chest_csv datasets/nih_chest_xrays/Data_Entry_2017.csv \
    --chest_imgs datasets/nih_chest_xrays/all_images \
    --nih_train_list datasets/nih_chest_xrays/splits/nih_train_list.txt \
    --nih_val_list datasets/nih_chest_xrays/splits/nih_val_list.txt \
    --nih_test_list datasets/nih_chest_xrays/splits/nih_test_list.txt \
    --chest_view_filter PA \
    --chest_bbox_csv datasets/nih_chest_xrays/BBox_List_2017.csv \
    --isic_train_csv datasets/isic_2019/splits/isic_train.csv \
    --isic_val_csv datasets/isic_2019/splits/isic_val.csv \
    --isic_test_csv datasets/isic_2019/splits/isic_test.csv \
    --isic_imgs datasets/isic_2019/ISIC_2019_Training_Input \
    --oa_root datasets/osteoarthritis/oa_splits \
    --luna_patches_dir datasets/luna_lung_cancer/patches \
    --luna_csv datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv \
    --pancreas_splits_csv datasets/pancreas_splits.csv \
    --pancreas_nii_dir datasets/zenodo_13715870 \
    --pancreas_fold 1 \
    --pancreas_roi_strategy A
```

---
Generado automáticamente por fase0_pipeline.py