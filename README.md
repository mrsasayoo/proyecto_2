# proyecto_2 — Sistema MoE Médico

Sistema de Mixture of Experts (MoE) para clasificación de imágenes médicas. El router, que decide qué experto activar, se evalúa experimentalmente a través de cuatro mecanismos: Vision Transformer, GMM, Naive Bayes y k-NN. El Expert 1 implementa clasificación multilabel de 14 patologías torácicas sobre el dataset NIH ChestX-ray14 usando una arquitectura custom Hybrid-Deep-Vision entrenada desde cero.

El pipeline se divide en dos fases principales: Fase 0 (preprocesamiento offline de imágenes a `.npy` con CLAHE y normalización) y Fase 2 (entrenamiento distribuido con DDP, FocalLoss, gradient accumulation y TTA). El entrenamiento soporta de forma transparente desde CPU hasta clusters multi-GPU sin cambios de código.

## Arquitectura General

```
Fase 0: Preprocesamiento Offline
  raw PNG → grayscale → CLAHE → resize 256×256 → float32 [0,1] → .npy + metadata.csv

Fase 2: Entrenamiento DDP
  .npy → Albumentations (8 augmentaciones) → Hybrid-Deep-Vision → FocalLoss → AdamW + CosineAnnealingLR
  Evaluación: AUC-ROC por clase + Macro AUC + TTA (original + HorizontalFlip)
```

## Requisitos de Hardware

| Hardware | Config recomendada | VRAM por GPU | Tiempo estimado/época |
|----------|-------------------|--------------|----------------------|
| 2× Titan Xp (12 GB) | batch=48, accum=2, workers=8, FP16 | ~8 GB | ~3-5 min |
| 1× RTX 4090 (24 GB) | batch=96, accum=1, workers=12, FP16 | ~12 GB | ~2-3 min |
| CPU only | batch=16, accum=4, workers=4, FP32 | N/A | ~45+ min |

## Instalación

```bash
# Clonar y entrar al proyecto
cd proyecto_2

# Crear entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python-headless pandas scikit-learn tqdm numpy

# Verificar GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## Dataset

Descargar NIH ChestX-ray14 desde https://nihcc.app.box.com/v/ChestXray-NIHCC y colocar en:

```
datasets/nih_chest_xrays/
├── images_001/ ... images_012/    # Imágenes originales
├── Data_Entry_2017.csv            # Etiquetas
└── splits/
    ├── nih_train_list.txt
    ├── nih_val_list.txt
    └── nih_test_list.txt
```

## Ejecución

### 1. Preprocesamiento (Fase 0) — una sola vez

```bash
python src/pipeline/fase0/pre_chestxray14.py
```

Genera `datasets/nih_chest_xrays/preprocessed/{train,val,test}/` con `.npy` + `metadata.csv` + `stats.json`.

### 2. Dry-run (verificar pipeline)

```bash
bash run_expert.sh 1 --dry-run
```

Ejecuta 2 batches de train + 1 de validación (~1.5s). Verifica que todo funciona sin entrenar.

### 3. Entrenamiento completo

```bash
# Multi-GPU (detecta automáticamente)
bash run_expert.sh 1

# Con batch size custom
bash run_expert.sh 1 --batch-per-gpu 24

# Reanudar desde checkpoint
bash run_expert.sh 1 --resume checkpoints/expert_01_hybrid_deep_vision/best.pt
```

### 4. Monitoreo

```bash
# VRAM y temperatura
watch -n 2 nvidia-smi

# Training log (JSON)
cat checkpoints/expert_01_hybrid_deep_vision/expert1_ddp_training_log.json | python -m json.tool
```

## Estructura del Proyecto

```
src/pipeline/
├── fase0/pre_chestxray14.py        # Preprocesamiento offline
├── fase2/
│   ├── train_expert1_ddp.py        # Entrenamiento DDP principal
│   ├── expert1_config.py           # Hiperparámetros centralizados
│   ├── dataloader_expert1.py       # Data loading + Albumentations
│   ├── ddp_utils.py                # Utilidades DDP reutilizables
│   └── models/expert1_convnext.py  # Arquitectura Hybrid-Deep-Vision
├── datasets/chest.py               # Dataset class ChestXray14
run_expert.sh                       # Lanzador DDP con detección de GPUs
docs/                               # Documentación técnica
checkpoints/                        # Modelos guardados + logs
```

## Documentación

- [Arquitectura del Modelo](docs/ARCHITECTURE.md)
- [Guía de Configuración](docs/CONFIG_GUIDE.md)
- [Optimizaciones](docs/OPTIMIZATION.md)
- [Métricas](docs/METRICS.md)
- [Code Review](docs/CODE_REVIEW.md)
- [Checklist de Calidad](QUALITY_CHECKLIST.md)
