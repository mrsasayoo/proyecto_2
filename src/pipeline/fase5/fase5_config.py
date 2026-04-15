"""
fase5_config.py — Constantes exclusivas del fine-tuning por etapas (Paso 8).

Fuente de verdad para todos los hiperparametros del fine-tuning global del
sistema MoE. Las decisiones de LR y epocas siguen la arquitectura_documentacion.md
(seccion 5.6) como fuente de verdad para las incongruencias con la guia.

Etapas:
  Stage 1: Router solo (backbone + expertos congelados)
  Stage 2: Router + cabezas clasificadoras (backbone + conv layers congelados)
  Stage 3: Fine-tuning global (todo descongelado)

REGLA CRITICA (seccion 5.6): NO reiniciar optimizer entre etapas.
"""

# =====================================================================
# Stage 1 — Router solo (expertos + backbone congelados)
# =====================================================================
STAGE1_LR_ROUTER = 1e-3
"""Learning rate del router en Stage 1. Solo el router entrena."""

STAGE1_EPOCHS = 50
"""Epocas maximas para Stage 1 (seccion 5.6)."""

STAGE1_PATIENCE = 10
"""Early stopping patience para Stage 1."""

STAGE1_BATCH_SIZE = 64
"""Batch size de embeddings para Stage 1 (router sobre embeddings)."""


# =====================================================================
# Stage 2 — Router + cabezas clasificadoras
# (backbone + capas conv/feature de expertos congelados)
# =====================================================================
STAGE2_LR_ROUTER = 1e-4
"""Learning rate del router en Stage 2."""

STAGE2_LR_HEADS = 1e-4
"""Learning rate de las cabezas clasificadoras en Stage 2."""

STAGE2_EPOCHS = 30
"""Epocas maximas para Stage 2 (seccion 5.6)."""

STAGE2_PATIENCE = 8
"""Early stopping patience para Stage 2."""


# =====================================================================
# Stage 3 — Fine-tuning global (nada congelado)
# =====================================================================
STAGE3_LR_ROUTER = 1e-4
"""Learning rate del router en Stage 3 (Metodo B de la guia)."""

STAGE3_LR_EXPERTS = 1e-6
"""Learning rate de los expertos en Stage 3.
arch_doc seccion 5.6: LR=1e-6 (decision deliberada, mas conservador que la
guia que indica 1e-5). Justificacion: prevenir catastrofic forgetting en
expertos ya convergidos en Fases 2-3."""

STAGE3_LR_BACKBONE = 1e-6
"""Learning rate del backbone en Stage 3. Igual que expertos."""

STAGE3_EPOCHS_MAX = 10
"""Epocas maximas para Stage 3 (seccion 5.6: 7-10 epocas con early stop)."""

STAGE3_PATIENCE = 5
"""Early stopping patience para Stage 3."""


# =====================================================================
# Regla critica (seccion 5.6)
# =====================================================================
RESET_OPTIMIZER_BETWEEN_STAGES = False
"""NO reiniciar optimizer entre etapas. El momentum acumulado en Stage 1-2
ayuda a la convergencia de Stage 3."""


# =====================================================================
# Loss coefficients
# =====================================================================
ALPHA_L_AUX = 0.01
"""Coeficiente de la Auxiliary Loss del Switch Transformer.
Mismo valor que fase2_config.ALPHA_L_AUX."""

BETA_L_ERROR = 0.1
"""Coeficiente de L_error: penaliza enviar imagenes validas (in-distribution)
al Expert 5 CAE. Extension aprobada en arch_doc.
L_total = L_task + alpha * L_aux + beta * L_error."""

GAMMA_L_BALANCE = 1.0
"""Coeficiente de la penalizacion cuadratica de balance de carga.
Penaliza expertos que reciben mas de 2/K de los samples.
Termino: gamma * sum_k max(0, f_k - 2/K)^2."""


# =====================================================================
# Mixed precision
# =====================================================================
FP16_ENABLED = True
"""FP16 para expertos 0-4 y router. Expert 5 CAE usa FP32 siempre (manejado
por separado en el forward pass)."""


# =====================================================================
# Gradient
# =====================================================================
ACCUMULATION_STEPS = 4
"""Gradient accumulation steps. Batch efectivo minimo del proyecto."""

GRAD_CHECKPOINT_3D = True
"""Gradient checkpointing obligatorio para expertos 3 (MC3-18) y 4 (Swin3D).
Reduce uso de VRAM a cambio de velocidad."""


# =====================================================================
# DataLoader mixto — proporciones por dataset
# =====================================================================
MIXED_BATCH_SIZE = 32
"""Batch size real del DataLoader mixto."""

MIXED_SAMPLES_PER_DATASET = {
    "chest": 88999,
    "isic": 20409,
    "oa": 3814,
    "luna": 14728,
    "pancreas": 2052,
}
"""Numero de muestras de entrenamiento por dataset (de cae_splits.csv).
Usado para calcular pesos de muestreo proporcional."""


# =====================================================================
# Checkpoints de salida
# =====================================================================
CHECKPOINT_DIR = "checkpoints/fase5"
CHECKPOINT_STAGE1 = "checkpoints/fase5/moe_stage1.pt"
CHECKPOINT_STAGE2 = "checkpoints/fase5/moe_stage2.pt"
CHECKPOINT_STAGE3 = "checkpoints/fase5/moe_final.pt"


# =====================================================================
# Checkpoints de entrada (prerequisitos de Fases 2-4)
# =====================================================================
EXPERT_CHECKPOINTS = {
    0: "checkpoints/expert_00_convnext_tiny/expert1_best.pt",
    1: "checkpoints/expert_01_efficientnet_b3/expert2_best.pt",
    2: "checkpoints/expert_02_vgg16_bn/expert_oa_best.pt",
    3: "checkpoints/expert_03_vivit_tiny/expert3_best.pt",
    4: "checkpoints/expert_04_swin3d_tiny/expert4_best.pt",
    5: "checkpoints/expert_05_cae/cae_best.pt",
}
"""Rutas a los checkpoints de cada experto entrenado en Fases 2-3.
Estos checkpoints deben existir antes de ejecutar el fine-tuning real."""


# =====================================================================
# Mapping de atributos de cabeza por experto
# =====================================================================
# Mapeo del atributo que contiene la cabeza clasificadora de cada experto.
# Se usa en freeze_utils para congelar todo excepto la cabeza.
#
# NOTA sobre la estructura interna real de cada modelo:
#   Expert0 (ConvNeXt): self.model.classifier  (wrapper: self.model es el backbone)
#   Expert1 (ConvNeXt-Small): self.head  (head personalizado nn.Sequential)
#   Expert2 (EfficientNet-B0): self.model.classifier  (wrapper: self.model es el backbone)
#   Expert3 (MC3-18): self.classifier  (atributo directo)
#   Expert4 (Swin3D): self.backbone.head  (wrapper: self.backbone)
#   Expert5 (CAE): self.decoder_fc + self.decoder_conv  (no tiene "cabeza" clasica)
#
# Para freeze_except_head necesitamos saber los nombres de los modulos a
# mantener descongelados. Se usa una lista de prefijos por experto.
EXPERT_HEAD_PREFIXES = {
    0: ["model.classifier"],  # Expert1ConvNeXtTiny
    1: ["head"],  # Expert2ConvNeXtSmall
    2: ["classifier"],  # ExpertOAEfficientNetB3
    3: ["classifier"],  # Expert3MC318
    4: ["backbone.head"],  # ExpertPancreasSwin3D
    5: ["decoder_fc", "decoder_conv"],  # ConvAutoEncoder
}
"""Prefijos de nombres de parametros que corresponden a la cabeza de cada
experto. Usado por freeze_utils.freeze_except_head()."""


# =====================================================================
# Resumen ejecutivo para logs
# =====================================================================
FASE5_CONFIG_SUMMARY = (
    "Fase 5 Fine-tuning: "
    "Stage1(router, LR=1e-3, 50ep) | "
    "Stage2(router+heads, LR=1e-4, 30ep) | "
    "Stage3(global, LR_router=1e-4, LR_experts=1e-6, 10ep) | "
    "L_total = L_task + 0.01*L_aux + 0.1*L_error"
)
