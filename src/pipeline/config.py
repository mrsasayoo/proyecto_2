"""
Configuración global del pipeline MoE.

Constantes compartidas entre FASE 0, FASE 1 y FASE 2:
  - EXPERT_IDS: mapeo nombre → ID de experto
  - BACKBONE_CONFIGS: backbones disponibles con d_model y VRAM
  - Constantes HU para LUNA16 y Pancreas
  - Constantes de normalización ImageNet
  - Constantes de datasets (patologías, clases, etc.)
"""

import numpy as np

# ── Expertos del sistema MoE ────────────────────────────────
EXPERT_IDS = {
    "chest":    0,   # NIH ChestXray14      — radiografía tórax 2D
    "isic":     1,   # ISIC 2019            — dermatoscopía 2D
    "oa":       2,   # Osteoarthritis Knee  — radiografía rodilla 2D
    "luna":     3,   # LUNA16 / LIDC-IDRI   — CT pulmón 3D (parches)
    "pancreas": 4,   # Pancreatic Cancer CT — CT abdomen 3D (volumen)
    # "ood": 5   ← se activa en FASE 1, no en FASE 0
}

N_EXPERTS_TOTAL  = 6   # Expertos 0–4 de dominio + Experto 5 OOD
N_EXPERTS_DOMAIN = 5   # Solo los expertos con dataset propio (FASE 0)


# ── Normalización ImageNet ──────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Configuración de backbones disponibles ──────────────────
# ┌──────────────────────────────────┬─────────┬───────┬─────────────────────────────┐
# │ Backbone                         │ d_model │ VRAM  │ Cuándo usarlo               │
# ├──────────────────────────────────┼─────────┼───────┼─────────────────────────────┤
# │ vit_tiny_patch16_224  (DEFAULT)  │   192   │  ~2GB │ Primera corrida. Más rápido │
# │ swin_tiny_patch4_window7_224     │   768   │  ~4GB │ Ablation study final        │
# │ cvt_13                           │   384   │  ~3GB │ Balance intermedio          │
# └──────────────────────────────────┴─────────┴───────┴─────────────────────────────┘
BACKBONE_CONFIGS = {
    "vit_tiny_patch16_224":        {"d_model": 192,  "vram_gb": 2.0},
    "swin_tiny_patch4_window7_224":{"d_model": 768,  "vram_gb": 4.0},
    "cvt_13":                      {"d_model": 384,  "vram_gb": 3.0},
}


# ── NIH ChestXray14 ────────────────────────────────────────
CHEST_PATHOLOGIES = [
    "Atelectasis",        # 0
    "Cardiomegaly",       # 1
    "Effusion",           # 2
    "Infiltration",       # 3
    "Mass",               # 4
    "Nodule",             # 5
    "Pneumonia",          # 6
    "Pneumothorax",       # 7
    "Consolidation",      # 8
    "Edema",              # 9
    "Emphysema",          # 10
    "Fibrosis",           # 11
    "Pleural_Thickening", # 12
    "Hernia",             # 13
]
N_CHEST_CLASSES = len(CHEST_PATHOLOGIES)   # = 14

CHEST_BBOX_CLASSES = {
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax"
}


# ── Osteoarthritis Knee ────────────────────────────────────
OA_CLASS_NAMES   = ["Normal (KL0)", "Leve (KL1-2)", "Severo (KL3-4)"]
OA_N_CLASSES     = 3
# Reservado para FASE 2 — WeightedLoss con penalización ordinal
OA_COST_MATRIX   = np.array([[0, 1, 4],
                              [1, 0, 1],
                              [4, 1, 0]], dtype=np.float32)
OA_BASE_IMG_COUNT = 8260


# ── Constantes HU ──────────────────────────────────────────
# LUNA16: aire puro (-1000) hasta límite óseo (+400)
HU_LUNG_CLIP = (-1000, 400)
# Pancreas: grasa peripancreática (-100) hasta límite óseo (+400)
# NO usar [-1000, 400] para abdomen — comprime contraste diagnóstico 7x
HU_ABDOMEN_CLIP = (-100, 400)
