"""
fase1_config.py — Constantes EXCLUSIVAS de Fase 1
==================================================

Fuente de verdad para todos los valores numéricos, rutas y parámetros
del proceso de extracción de embeddings.

Las constantes que Familia 2 y 3 necesitan permanecen en
src/pipeline/config.py global.
"""

# ── Configuración de backbones disponibles ──────────────────
# ┌──────────────────────────────────┬─────────┬───────┬─────────────────────────────┐
# │ Backbone                         │ d_model │ VRAM  │ Cuándo usarlo               │
# ├──────────────────────────────────┼─────────┼───────┼─────────────────────────────┤
# │ vit_tiny_patch16_224  (DEFAULT)  │   192   │  ~2GB │ Primera corrida. Más rápido │
# │ swin_tiny_patch4_window7_224     │   768   │  ~4GB │ Ablation study final        │
# │ cvt_13                           │   384   │  ~3GB │ Balance intermedio          │
# └──────────────────────────────────┴─────────┴───────┴─────────────────────────────┘
BACKBONE_CONFIGS = {
    "vit_tiny_patch16_224": {"d_model": 192, "vram_gb": 2.0},
    "swin_tiny_patch4_window7_224": {"d_model": 768, "vram_gb": 4.0},
    "cvt_13": {"d_model": 384, "vram_gb": 3.0},
    "densenet121_custom": {"d_model": 1024, "vram_gb": 3.0},
}

# ── Normalización ImageNet ──────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Ventanas HU por tipo de CT ──────────────────────────────
# LUNA16: aire puro (-1000) hasta límite óseo (+400)
HU_LUNG_CLIP = (-1000, 400)
# Pancreas: grasa peripancreática (-100) hasta límite óseo (+400)
# NO usar [-1000, 400] para abdomen — comprime contraste diagnóstico 7x
HU_ABDOMEN_CLIP = (-100, 400)

# ── Parámetros por defecto del extractor ────────────────────
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 4
IMG_SIZE = 224
PATCH_3D_SIZE = (64, 64, 64)

# ── Umbrales de sanidad ────────────────────────────────────
# Norma L2 media mínima antes de emitir alerta sobre CLS token
MIN_L2_NORM = 1.0
# Ratio max/min de muestras entre expertos antes de advertencia
MAX_IMBALANCE = 10.0

# ── Preprocesamiento 2D (§6.2) ─────────────────────────────
# Total Variation Filter: intensidad del suavizado (λ en la formulación TV)
TVF_WEIGHT = 10.0
TVF_N_ITER = 30
# Corrección gamma: rango clínico 0.8–1.2, 1.0 = identidad
DEFAULT_GAMMA = 1.0
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ── Páncreas k-fold ────────────────────────────────────────
# Fold del k-fold CV (generado por Fase 0) usado para train/val en Fase 1
PANCREAS_FOLD = 1

# ── Claves esperadas en backbone_meta.json ──────────────────
# Contrato de interfaz entre Fase 1 (escritor) y Fase 2 (lector)
BACKBONE_META_KEYS = frozenset(
    {
        "backbone",
        "d_model",
        "n_train",
        "n_val",
        "n_test",
        "vram_gb",
    }
)

# ── Entrenamiento end-to-end de backbones (Paso 4.1) ──────────
# Hiperparámetros para la tarea proxy de clasificación de dominio
TRAIN_EPOCHS = 20  # épocas — suficiente para proxy de dominio
TRAIN_LR = 3e-4  # AdamW default
TRAIN_WEIGHT_DECAY = 0.01  # L2 regularización
TRAIN_WARMUP_EPOCHS = 2  # épocas de warm-up lineal del LR
TRAIN_BATCH_SIZE = 64  # igual que DEFAULT_BATCH_SIZE
TRAIN_WORKERS = 4  # igual que DEFAULT_WORKERS
TRAIN_GRAD_CLIP = 1.0  # gradient clipping max norm

# Mapeo backbone_name → subdirectorio en checkpoints/
BACKBONE_TO_CHECKPOINT_DIR = {
    "vit_tiny_patch16_224": "backbone_01_vit_tiny",
    "cvt_13": "backbone_02_cvt13",
    "swin_tiny_patch4_window7_224": "backbone_03_swin_tiny",
    "densenet121_custom": "backbone_04_densenet121",
}

# Nombre del archivo de checkpoint dentro del subdirectorio
BACKBONE_CHECKPOINT_FILENAME = "backbone.pth"
