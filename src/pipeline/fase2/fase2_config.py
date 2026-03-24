"""
fase2_config.py — Constantes exclusivas de Fase 2.

Responsabilidad única: ser la única fuente de verdad para todos los valores
numéricos y parámetros que son propios del ablation study del router, sin
mezclar constantes de otras fases.

Las constantes compartidas entre fases (N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL,
EXPERT_IDS, EXPERT_NAMES, EXPERT_NOTES, CHEST_PATHOLOGIES, OA_COST_MATRIX)
permanecen en src/pipeline/config.py global y NO se duplican aquí.

Un investigador que quiera reproducir solo el ablation study puede abrir este
archivo y entender todos los parámetros del proceso sin leer configuración de
otras fases ni inferir valores del código.
"""

# ── Router Linear (paramétrico, gradiente) ──────────────────
ALPHA_L_AUX = 0.01
"""Coeficiente α de la Auxiliary Loss del Switch Transformer.
L_aux = α · N · Σ f_i · P_i
Si α=0 el router lineal no tiene penalización de balance de carga.
"""

LINEAR_EPOCHS = 50
"""Épocas de entrenamiento del router Linear."""

LINEAR_LR = 1e-3
"""Learning rate del optimizador Adam para el router Linear."""

LINEAR_BATCH_SIZE = 512
"""Batch size durante el entrenamiento del router Linear."""

# ── Router GMM (paramétrico, EM sin supervisión) ─────────────
GMM_COV_TYPE = "full"
"""Tipo de covarianza del GaussianMixture (default='full').
'full' modela correlaciones entre dimensiones — apropiado para embeddings
de ViT donde la varianza no es diagonal. Si EM no converge con 'full',
el módulo hace fallback automático a 'diag'.
"""

GMM_MAX_ITER = 200
"""Iteraciones máximas del algoritmo EM para el GMM."""

# ── Router kNN-FAISS (no paramétrico, distancia coseno) ─────
KNN_K = 5
"""Número de vecinos más cercanos para el router kNN-FAISS."""

LAPLACE_EPSILON = 0.01
"""Suavizado de Laplace para probabilidades de voto del kNN.
Con k=5, las probabilidades sin suavizar son discretas {0, 0.2, …, 1.0},
produciendo entropías artificialmente discretas que hacen el umbral OOD
no comparable con los otros routers (continuos). El suavizado hace las
probabilidades continuas sin alterar materialmente las predicciones.
"""

# ── Ablation study ───────────────────────────────────────────
LOAD_BALANCE_THRESHOLD = 1.30
"""Umbral de balance de carga: max(f_i)/min(f_i) debe ser ≤ 1.30.
Un router que supera este umbral tiene penalización del 40% en la
evaluación final del proyecto y no puede ser el ganador del ablation,
independientemente de su routing accuracy.
"""

ENTROPY_PERCENTILE = 95
"""Percentil para calibrar el umbral OOD sobre el set de validación.
El umbral es el percentil ENTROPY_PERCENTILE de H(g) sobre las entropías
del set completo de validación. El 5% de las muestras más confusas del
router serán tratadas como OOD en inferencia.
"""

# ── Contrato del backbone_meta.json (producido por Fase 1) ──
BACKBONE_META_REQUIRED_KEYS = {
    "backbone",
    "d_model",
    "n_train",
    "n_val",
    "n_test",
    "vram_gb",
}
"""Conjunto de claves que backbone_meta.json DEBE contener.
Si alguna clave falta, embeddings_loader.py emite un error descriptivo.
Esta constante es la fuente de verdad para el contrato entre Fase 1 y Fase 2.
Un cambio en el contrato (por ejemplo, añadir 'n_classes') solo requiere
modificar este archivo.
"""
