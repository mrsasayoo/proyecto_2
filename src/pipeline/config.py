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


# ── Nombres y notas por experto ─────────────────────────────
# EXPERT_NAMES: etiquetas para logs y tablas
# EXPERT_NOTES: notas clínicas por experto, exportadas a ablation_results.json para guiar FASE 2
EXPERT_NAMES = {
    0: "Chest",
    1: "ISIC",
    2: "OA-Knee",
    3: "LUNA16",
    4: "Pancreas",
    # 5: "OOD" ← activado por entropía, no por routing directo
}

EXPERT_NOTES = {
    0: (
        "NIH ChestXray14 — MULTI-LABEL (14 patologías). "
        "H1: Loss=BCEWithLogitsLoss con pos_weight; NO CrossEntropyLoss. "
        "H2: splits por Patient ID verificados (train_val_list.txt / test_list.txt). "
        "H3: etiquetas por NLP, benchmark DenseNet-121 ≈ 0.81 AUC macro; "
        "AUC > 0.85 significativo → revisar confounding. "
        "H4: filtro View Position PA recomendado para estudios controlados (--chest_view_filter PA). "
        "H5: BBox disponible para 8/14 clases (~1000 imgs) — usar en dashboard "
        "para validar heatmaps ViT con ChestXray14Dataset.load_bbox_index(). "
        "H6: ⚠ NUNCA usar Accuracy (~54% prediciendo siempre 'No Finding'). "
        "Métricas: AUC-ROC por clase + F1 Macro + AUPRC. "
        "FocalLossMultiLabel(gamma=2) disponible como alternativa a BCEWithLogitsLoss."
    ),
    1: ("ISIC 2019 — MULTICLASE (8 clases en train, UNK solo en test). "
        "H1: Loss: CrossEntropyLoss con pesos class_weights. NO BCEWithLogitsLoss. "
        "Capa de salida: 9 neuronas softmax (incluye slot UNK para inferencia). "
        "H2: Split por lesion_id (build_lesion_split). Sin metadata_csv → riesgo de leakage. "
        "H3: 3 fuentes (HAM10000/BCN_20000/MSK) con bias de dominio — ColorJitter agresivo. "
        "Métricas: BMCA (oficial) + AUC-ROC por clase. "
        "UNK en inferencia → H(g) alta → Experto 5 OOD la captura."),
    2: ("OA Rodilla — ORDINAL (3 clases: Normal/Leve/Severo, consolidadas de KL 0-4). "
        "H1: Opción A (pragmática): CrossEntropyLoss con class_weights. "
        "Opción B: OrdinalLoss(n_classes=3) — salida [B,2] logits. "
        "Métrica principal: QWK — llamar OAKneeDataset.compute_qwk(y_true, y_pred) "
        "en cada época de validación. ⚠ NO usar Accuracy ni F1 como métricas principales para OA. "
        "H2: Aug offline detectado → NO aug adicional en runtime. "
        "H3: Sin Patient ID — documentar como limitación en el reporte. "
        "H4: CLAHE SIEMPRE antes del resize (apply_clahe() → resize(224×224)). "
        "H5: KL1 contamina Clase1 — frontera 0↔1 será la más difícil. "
        "Usar OAKneeDataset.evaluate_boundary_confusion() + log_boundary_confusion() "
        "en cada época de validación. ⚠ NUNCA RandomVerticalFlip."),
    3: ("LUNA16 — BINARIO parches 3D (NO volúmenes completos). "
        "H1: Tarea = clasificar parche 64×64×64 centrado en candidato (x,y,z). "
        "Tensor FASE 2: [B, 1, 64, 64, 64] — arquitectura 3D (R3D-18, Swin3D). "
        "H2: Desbalance ~490:1 — FocalLoss(gamma=2, alpha=0.25) obligatoria. "
        "BCELoss → modelo trivial (siempre class=0). "
        "H3: Conversión world→vóxel con LUNA16PatchExtractor.world_to_voxel(). "
        "Validar con LUNA16PatchExtractor.validate_extraction() antes de producción. "
        "Item-2: SIEMPRE candidates_V2.csv (V1 obsoleto, 24 nódulos menos). "
        "Item-4: HU clip[-1000,400]→[0,1] en LUNA16PatchExtractor.extract(). "
        "Item-7: gradient checkpointing obligatorio (batch=4, FP16, 12GB VRAM). "
        "Item-8: CPM+FROC con noduleCADEvaluationLUNA16.py. "
        "Techo teórico: 94.4% (LUNA16Dataset.SENSITIVITY_CEILING). "
        "Item-9: spacing variable en volúmenes — verificado en __init__."),
    4: ("Pancreas PANORAMA — BINARIO volumen 3D (PDAC+ / PDAC−). "
        "H1: etiquetas en repo GitHub SEPARADO — git clone panorama_labels; "
        "fijar hash commit; cruzar case_ids con .nii.gz del ZIP. "
        "H2: páncreas ~1% del volumen → resize naïve destruye señal. "
        "Opción A (FASE 0): resize completo. Opción B (FASE 2): recorte Z[120:220]. "
        "H3: HU clip HU_ABDOMEN_CLIP=(-100,400) — NO HU_LUNG_CLIP de LUNA16. "
        "z-score por volumen para bias multicéntrico (Radboudumc/MSD/NIH). "
        "Loss: FocalLoss(alpha=0.75, gamma=2). "
        "Item-6: batch_size=1–2 + FP16 + gradient checkpointing (más restrictivo que LUNA16). "
        "Item-8: k-fold CV k=5 (solo ~281 volúmenes). "
        "Métrica: AUC-ROC > 0.85 (bueno); baseline nnU-Net ~0.88."),
}
