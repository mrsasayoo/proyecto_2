"""
FASE 0 — Extracción de CLS tokens (Embeddings)
===============================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II
 
Arquitectura MoE (diseño propio — 6 expertos):
  - Experto 0 → NIH ChestXray14   (radiografía tórax, multi-label BCE)
  - Experto 1 → ISIC 2019         (dermatoscopía, multiclase CE)
  - Experto 2 → OA Rodilla        (radiografía rodilla, ordinal QWK)
  - Experto 3 → LUNA16            (CT pulmón 3D, parches, FocalLoss FROC)
  - Experto 4 → Pancreas PANORAMA (CT abdomen 3D, volumen, FocalLoss AUC)
  - Experto 5 → OOD / Error       (MLP sin dominio clínico — ver nota abajo)
 
  ⚠ EXPERTO 6 (ID=5, OOD/Error) NO genera embeddings en FASE 0.
    No existe un "dataset de imágenes OOD" que extraer. El Experto 6
    es un MLP simple que se define e inicializa en FASE 1, cuando se
    entrena el router. Su rol es recibir imágenes que el router no sabe
    clasificar (entropía alta del gating score H(g)) y devolver una
    señal de alerta. La pérdida L_error penaliza al router si delega
    imágenes médicas válidas (Expertos 0–4) al Experto 6.
    → y_train/y_val solo contendrán etiquetas 0–4 al finalizar FASE 0.
 
Requerimientos del proyecto (sección 4.1):
  - Backbone: ViT-Tiny preentrenado de timm (o CvT / Swin-Tiny)
  - Backbone CONGELADO — no se entrena, solo extrae
  - Pasar los 5 datasets de dominio → guardar CLS tokens en disco
  - Resultado: Z_train [N_train, 192], Z_val [N_val, 192]
  - Etiquetas de experto en disco: 0=Chest, 1=ISIC, 2=OA, 3=LUNA, 4=Pancreas
    (Experto 5=OOD no aparece — no tiene datos propios)
  - Hardware objetivo: clúster con 2 GPUs × 12 GB VRAM
 
Uso:
  python fase0_extract_embeddings.py --backbone vit_tiny_patch16_224 \
                                     --batch_size 64 \
                                     --output_dir ./embeddings
"""
 
import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import SimpleITK as sitk   # para LUNA16 y Pancreas
import cv2                 # para CLAHE en OA
 

# ──────────────────────────────────────────────────────────
# 0. LOGGING
# ──────────────────────────────────────────────────────────
 
 
def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configura el sistema de logging del pipeline.
    Escribe simultáneamente a consola (INFO) y a archivo (DEBUG).
 
    Niveles usados en este script:
      DEBUG   → detalles internos útiles para depuración profunda
      INFO    → progreso normal del pipeline
      WARNING → situación anómala que no detiene la ejecución
                (dataset vacío, d_model inesperado, batch con NaN, etc.)
      ERROR   → fallo grave que probablemente corrompe los embeddings
                (backbone con gradientes activos, NaN sistémico, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / "fase0.log"
 
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
 
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),   # consola
        ]
    )
    # La consola solo muestra INFO y superior — el archivo guarda todo
    logging.getLogger().handlers[1].setLevel(logging.INFO)
 
    log = logging.getLogger("fase0")
    log.info(f"Log iniciado → {log_path}")
    return log
 
  
# Logger global — se inicializa en main() con setup_logging()
# Las clases Dataset lo obtienen con getLogger para no depender del orden de importación
log = logging.getLogger("fase0")


# ──────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────
 

EXPERT_IDS = {
    "chest":    0,   # NIH ChestXray14      — radiografía tórax 2D
    "isic":     1,   # ISIC 2019            — dermatoscopía 2D
    "oa":       2,   # Osteoarthritis Knee  — radiografía rodilla 2D
    "luna":     3,   # LUNA16 / LIDC-IDRI   — CT pulmón 3D (parches)
    "pancreas": 4,   # Pancreatic Cancer CT — CT abdomen 3D (volumen)
    # ──────────────────────────────────────────────────────────────────
    # Experto 5 — OOD / Error  (ID=5)
    # ──────────────────────────────────────────────────────────────────
    # INTENCIONALMENTE AUSENTE de los datasets de FASE 0.
    #
    # El Experto 6 (OOD) es un MLP sin conocimiento de dominio clínico.
    # No aprende de imágenes médicas: aprende a recibir lo que el router
    # no sabe dónde enviar (entropía alta del gating score H(g) > umbral).
    #
    # Por qué no tiene embeddings aquí:
    #   • No existe un "dataset de imágenes OOD" curado para extraer.
    #   • Su entrenamiento ocurre en FASE 1 mediante L_error:
    #       L_error penaliza al router cada vez que envía una imagen
    #       médica válida (dominio 0–4) al Experto 6. Con esto, el router
    #       aprende activamente a NO delegar imágenes conocidas al OOD.
    #   • En inferencia: H(g) = -sum(g_i * log(g_i)) > ENTROPY_THRESHOLD
    #       → imagen se delega a Experto 6 → alerta OOD en dashboard.
    #
    # Referencia en el diagrama de arquitectura:
    #   Router → [Exp0..Exp4] si confianza alta (H(g) < umbral)
    #   Router → Exp5 OOD    si confianza baja  (H(g) ≥ umbral)
    #
    # "ood": 5   ← se activa en FASE 1, no en FASE 0
    # ──────────────────────────────────────────────────────────────────
}
 
# Número total de expertos en la arquitectura final (incluye OOD)
N_EXPERTS_TOTAL  = 6   # Expertos 0–4 de dominio + Experto 5 OOD
N_EXPERTS_DOMAIN = 5   # Solo los expertos con dataset propio (FASE 0)
 
# ── NIH ChestXray14 — 14 patologías en orden canónico ───────
# Orden idéntico al de las columnas en BBox_List_2017.csv y a la literatura.
# CRÍTICO: el vector binario de etiquetas SIEMPRE debe respetar este orden.
# En modo expert, __getitem__ devuelve un tensor float32 de longitud 14
# donde 1.0 indica presencia de la patología.
# Loss: BCEWithLogitsLoss (NO CrossEntropyLoss — las clases NO son mutuamente excluyentes).
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
    # "No Finding" se excluye del vector — indica ausencia de todas las anteriores.
    # Si "No Finding" aparece solo, el vector será todo ceros.
]
N_CHEST_CLASSES = len(CHEST_PATHOLOGIES)   # = 14

# ── H5: Clases CON bounding boxes en BBox_List_2017.csv ─────
# Solo 8 de 14 patologías tienen anotaciones de localización (~1000 imágenes).
# NO sirven para entrenar detección supervisada (muy pocas muestras).
# SÍ sirven para validar cualitativamente los Attention Heatmaps del dashboard:
#   si el heatmap del ViT se superpone con el BBox → el backbone aprendió features reales.
# Uso en FASE 2/dashboard: cargar con ChestXray14Dataset.load_bbox_index(bbox_csv_path)
CHEST_BBOX_CLASSES = {
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax"
}   # Las 6 restantes (Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia)
    # no tienen bounding boxes disponibles en BBox_List_2017.csv.

# ── H6: Estrategia de pérdida para ChestXray14 ──────────────
# Opción A (especificada en el proyecto): BCEWithLogitsLoss con pos_weight por clase.
#   pos_weight[i] = N_neg[i] / N_pos[i]  → amplifica patologías raras
# Opción B (válida pero fuera del documento): FocalLoss — reduce el peso de ejemplos
#   fáciles, útil especialmente para Hernia (~0.2%) y Pneumonia.
#   Ver clase FocalLossMultiLabel abajo. Justificarla en el reporte si se usa.
#
# ⚠ LO QUE NUNCA SE DEBE USAR: Accuracy como métrica principal.
#   Un modelo que siempre predice "No Finding" tiene ~54% de Accuracy
#   sin detectar ninguna patología. Usar: AUC-ROC por clase + F1 Macro + AUPRC.
#   Benchmark: DenseNet-121 ≈ 0.81 AUC macro. SOTA ≈ 0.85.


class FocalLossMultiLabel(nn.Module):
    """
    Focal Loss para clasificación multi-label (alternativa a BCEWithLogitsLoss).

    H6 — Hallazgo 6: útil cuando BCEWithLogitsLoss con pesos no converge bien
    en patologías muy raras (Hernia ~0.2%, Pneumonia ~1.2%) porque FocalLoss
    reduce dinámicamente el gradiente de los ejemplos fáciles (alta confianza)
    y focaliza el entrenamiento en los difíciles.

    NOTA: El proyecto especifica BCEWithLogitsLoss como loss principal para
    ChestXray14. Si se usa FocalLoss, documentarlo en el Reporte Técnico
    (sección 3 — decisiones de diseño) con la justificación técnica.

    Uso en FASE 2:
        criterion = FocalLossMultiLabel(gamma=2.0)
        loss = criterion(logits, labels)   # logits: [B,14], labels: [B,14] float
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE por elemento sin reducción
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t: probabilidad de la clase correcta
        p_t  = torch.exp(-bce)
        loss = (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # "none"


# Normalización ImageNet — estándar para todos los datasets 2D
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Configuración de backbones disponibles ──────────────────
# Alineado con sección 4.1 del proyecto: "ViT como backbone compartido".
# La guía permite ViT, CvT o Swin-Tiny. Cada uno produce un d_model distinto,
# lo que cambia el tamaño de Z_train.npy y los parámetros de los routers.
#
# ┌──────────────────────────────────┬─────────┬───────┬─────────────────────────────┐
# │ Backbone                         │ d_model │ VRAM  │ Cuándo usarlo               │
# ├──────────────────────────────────┼─────────┼───────┼─────────────────────────────┤
# │ vit_tiny_patch16_224  (DEFAULT)  │   192   │  ~2GB │ Primera corrida. Más rápido │
# │                                  │         │       │ y estable. Ideal para       │
# │                                  │         │       │ verificar que la lógica MoE │
# │                                  │         │       │ funciona antes de escalar.  │
# ├──────────────────────────────────┼─────────┼───────┼─────────────────────────────┤
# │ swin_tiny_patch4_window7_224     │   768   │  ~4GB │ Recomendado para el ablation│
# │                                  │         │       │ study final. Embeddings de  │
# │                                  │         │       │ mayor calidad por su sesgo  │
# │                                  │         │       │ inductivo jerárquico. Muy   │
# │                                  │         │       │ probable que mejore accuracy│
# │                                  │         │       │ de todos los routers.       │
# ├──────────────────────────────────┼─────────┼───────┼─────────────────────────────┤
# │ cvt_13                           │   384   │  ~3GB │ Balance intermedio. Mejor   │
# │                                  │         │       │ localidad espacial que ViT  │
# │                                  │         │       │ puro (convolución en tokens)│
# └──────────────────────────────────┴─────────┴───────┴─────────────────────────────┘
#
# CONSEJO PARA EL ABLATION STUDY DEL BACKBONE:
#   Corre FASE 0 tres veces con --backbone distinto → guarda en carpetas separadas:
#     ./embeddings/vit_tiny/   ./embeddings/swin_tiny/   ./embeddings/cvt/
#   Luego corre fase1_ablation_router.py sobre cada carpeta y compara los 4 routers
#   en cada espacio de embeddings. Esto da una tabla 3×4 = 12 experimentos que
#   enriquece significativamente la sección 4.3 del reporte técnico.
#
# NOTA: Si cambias de backbone, Z_train.npy tendrá distinto d_model.
#   fase1_ablation_router.py lo detecta automáticamente desde Z_train.shape[1].
BACKBONE_CONFIGS = {
    "vit_tiny_patch16_224":        {"d_model": 192,  "vram_gb": 2.0},
    "swin_tiny_patch4_window7_224":{"d_model": 768,  "vram_gb": 4.0},
    "cvt_13":                      {"d_model": 384,  "vram_gb": 3.0},
}


# ──────────────────────────────────────────────────────────
# 2. PREPROCESADOR ADAPTATIVO (rank=4 → 2D, rank=5 → 3D)
# ──────────────────────────────────────────────────────────
 
# Esta función crea una tubería (pipeline) de preprocesamiento estándar que convierte cualquier imagen cruda 
# de los datasets 2D al formato matemático exacto que el backbone ViT-Tiny espera recibir para operar correctamente.
def build_2d_transform(img_size=224):
    """Transform estándar para todos los datasets 2D."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
 
 
# Esta función mejora la visibilidad de los detalles óseos en las radiografías de rodilla mediante CLAHE 
# (Contrast Limited Adaptive Histogram Equalization), técnica indispensable para que el backbone del modelo 
# (ViT) pueda detectar cambios sutiles en el cartílago y el hueso que indican osteoartritis.
def apply_clahe(img_pil):
    """CLAHE obligatorio para OA Rodilla (contraste óseo)."""
    img_array = np.array(img_pil.convert("L"))        # escala de grises
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


# Esta función normaliza las Unidades Hounsfield (HU) del volumen CT al rango [0, 1] utilizando 
# clipping para descartar densidades extremas (como metales) y escalado lineal para resaltar tejidos 
# de interés (pulmón, partes blandas, tumores), facilitando la convergencia del entrenamiento.
def normalize_hu(volume, min_hu=-1000, max_hu=400):
    """Normalización HU para CT 3D (LUNA16, Pancreas)."""
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume.astype(np.float32)
 
 
# Esta función estandariza la geometría de los escáneres CT 3D (volúmenes de densidad variable) a una cuadrícula 
# fija de 64x64x64 utilizando interpolación trilineal. Esto es indispensable porque los modelos 3D (como Swin3D-Tiny o R3D-18) 
# requieren un tamaño de entrada constante para realizar las operaciones matriciales de las capas convolucionales o de atención.
def resize_volume_3d(volume, target=(64, 64, 64)):
    """
    Redimensiona un volumen 3D [D, H, W] a target usando
    interpolación trilineal vía PyTorch.
    """
    t = torch.from_numpy(volume).float()
    # Añadir dims de batch y canal: [1, 1, D, H, W]
    t = t.unsqueeze(0).unsqueeze(0)
    t = nn.functional.interpolate(
        t, size=target, mode="trilinear", align_corners=False
    )
    return t.squeeze(0)  # → [1, 64, 64, 64]


# Esta función implementa una proyección ortogonal del volumen 3D: extrae los tres cortes centrales (axial, coronal y sagital) 
# de una tomografía y los apila como si fueran los canales de color (R, G, B) de una imagen normal. Esto comprime la 
# información tridimensional en una representación 2D que el ViT puede procesar sin necesidad de modificar su arquitectura.
def volume_to_vit_input(volume_3d_tensor):
    """
    Convierte [1, 64, 64, 64] a [3, 224, 224] para que ViT-2D
    lo procese: toma 3 slices centrales (axial, coronal, sagital)
    y los apila como canales RGB.
 
    Nota: esto es una aproximación válida para extracción de
    embeddings de routing. Para clasificación 3D real se usa
    la arquitectura 3D del experto (R3D-18, Swin3D-Tiny).
    """
    v = volume_3d_tensor.squeeze(0)           # [64, 64, 64]
    d, h, w = v.shape
 
    axial    = v[d // 2, :, :]                # corte central eje Z
    coronal  = v[:, h // 2, :]               # corte central eje Y
    sagittal = v[:, :, w // 2]               # corte central eje X
 
    # Stack como RGB → [3, 64, 64]
    rgb = torch.stack([axial, coronal, sagittal], dim=0)
 
    # Resize a 224×224
    rgb = nn.functional.interpolate(
        rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)                              # [3, 224, 224]
 
    # Normalizar como ImageNet (el volumen ya está en [0,1])
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (rgb - mean) / std


# ──────────────────────────────────────────────────────────
# 3. DATASETS
# ──────────────────────────────────────────────────────────


# ── Experto 0 — NIH ChestXray14 ────────────────────────────────────────────
# Cargador con auditoría de integridad completa. Implementa los 6 hallazgos:
#   H1 → modo expert: vector multi-label [14] + BCEWithLogitsLoss
#   H2 → verificación cruzada de Patient ID entre splits (sin leakage)
#   H3 → warning de ruido NLP + benchmark AUC en logs
#   H4 → filtro opcional por View Position (PA/AP) para evitar bias de dominio
#   H5 → load_bbox_index() para validar heatmaps del dashboard
#   H6 → pos_weight automático + FocalLossMultiLabel disponible como alternativa
class ChestXray14Dataset(Dataset):
    """
    NIH ChestXray14 — Multi-label, 14 patologías.

    Hallazgos implementados:
      H1 — Multi-label: BCEWithLogitsLoss obligatoria, salida [14] sigmoides.
      H2 — Leakage: splits por Patient ID verificados cruzadamente.
      H3 — Ruido NLP: benchmark DenseNet-121 ≈ 0.81 AUC macro en logs.
      H4 — Vista PA/AP: filtro opcional por View Position (filter_view="PA").
      H5 — BBox: load_bbox_index() para validación de heatmaps en dashboard.
      H6 — Pérdida: pos_weight por clase calculado automáticamente.
           FocalLossMultiLabel disponible como alternativa (ver clase arriba).
           ⚠ NUNCA usar Accuracy como métrica principal (~54% prediciendo siempre
             "No Finding" sin detectar ninguna patología).

    Dos modos de uso:
      mode="embedding" → FASE 0: devuelve (img, expert_id=0, img_name)
      mode="expert"    → FASE 2: devuelve (img, label_vector_14, img_name)
    """

    def __init__(self, csv_path, img_dir, file_list, transform,
                 mode="embedding", patient_ids_other=None, filter_view=None):
        """
        Args:
            csv_path           : ruta a Data_Entry_2017.csv
            img_dir            : directorio con los .png
            file_list          : train_val_list.txt o test_list.txt (split oficial)
            transform          : torchvision transform pipeline
            mode               : "embedding" (FASE 0) o "expert" (FASE 2)
            patient_ids_other  : set de Patient IDs del OTRO split — verifica leakage.
                                 Si None, se omite la verificación cruzada (H2).
            filter_view        : "PA", "AP" o None (sin filtro).
                                 H4: usar "PA" para estudios controlados y evitar
                                 el sesgo visual de las vistas AP (pacientes encamados).
                                 Documentar en el reporte si se filtra.
        """
        assert mode in ("embedding", "expert"), \
            f"[Chest] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"

        self.img_dir     = Path(img_dir)
        self.transform   = transform
        self.expert_id   = EXPERT_IDS["chest"]
        self.mode        = mode
        self.filter_view = filter_view

        # ── H2+H3: Verificación de columnas del CSV ───────────────────────
        df_full = pd.read_csv(csv_path)
        log.debug(f"[Chest] CSV cargado: {len(df_full):,} filas en {csv_path}")

        required_cols = {"Image Index", "Finding Labels", "Patient ID",
                         "View Position", "Follow-up #"}
        missing_cols = required_cols - set(df_full.columns)
        if missing_cols:
            log.error(f"[Chest] Columnas faltantes en el CSV: {missing_cols}. "
                      f"Verifica que sea Data_Entry_2017.csv oficial.")
        else:
            log.debug(f"[Chest] Columnas verificadas: {sorted(required_cols)} ✓")

        # ── H2: Aplicar split oficial (por Patient ID, no aleatorio) ──────
        with open(file_list) as f:
            valid_files = set(f.read().splitlines())
        log.debug(f"[Chest] file_list '{Path(file_list).name}': {len(valid_files):,} nombres")

        self.df       = df_full[df_full["Image Index"].isin(valid_files)].reset_index(drop=True)
        n_before_view = len(self.df)
        n_missing_csv = len(valid_files) - n_before_view

        if n_before_view == 0:
            log.error("[Chest] ¡Dataset vacío tras aplicar file_list! "
                      "Verifica que chest_csv y chest_train_list sean del mismo release.")
        elif n_missing_csv > 0:
            log.warning(f"[Chest] {n_missing_csv:,} nombres del file_list no están en el CSV "
                        f"(aceptable en subsets). Imágenes cargadas: {n_before_view:,}")
        else:
            log.info(f"[Chest] {n_before_view:,} imágenes cargadas (split '{Path(file_list).name}')")

        # ── H4: Filtro por View Position (PA vs AP) ───────────────────────
        if "View Position" in self.df.columns:
            view_counts = self.df["View Position"].value_counts()
            log.info(f"[Chest] Distribución View Position ANTES del filtro: "
                     + " | ".join(f"{v}: {c:,}" for v, c in view_counts.items()))

            if filter_view is not None:
                n_antes = len(self.df)
                self.df = self.df[
                    self.df["View Position"] == filter_view
                ].reset_index(drop=True)
                n_filtradas = n_antes - len(self.df)
                log.info(f"[Chest] H4 — Filtro View Position='{filter_view}' aplicado: "
                         f"{n_filtradas:,} imágenes eliminadas → "
                         f"{len(self.df):,} imágenes restantes.")
                log.info(f"[Chest] H4 — Documenta este filtro en el Reporte Técnico "
                         f"(sección 3 — decisiones de preprocesado).")
                if len(self.df) == 0:
                    log.error(f"[Chest] ¡Dataset vacío tras filtrar View Position='{filter_view}'! "
                              f"Verifica que la columna tenga ese valor. "
                              f"Valores disponibles: {list(view_counts.index)}")
            else:
                if "AP" in view_counts and "PA" in view_counts:
                    ap_pct = 100 * view_counts.get("AP", 0) / n_before_view
                    log.warning(
                        f"[Chest] H4 — {ap_pct:.1f}% de imágenes son vistas AP (pacientes "
                        f"encamados). Las AP tienen corazón aparentemente más grande y mayor "
                        f"distorsión geométrica — sesgo de dominio potencial. "
                        f"Para estudios controlados: usa filter_view='PA' o "
                        f"--chest_view_filter PA en CLI."
                    )

        n_matched = len(self.df)

        # ── H2: Verificación cruzada de Patient ID (leakage) ──────────────
        self.patient_ids = set(self.df["Patient ID"].unique())
        log.info(f"[Chest] Pacientes únicos en este split: {len(self.patient_ids):,}")

        if patient_ids_other is not None:
            leaked = self.patient_ids & patient_ids_other
            if leaked:
                log.error(
                    f"[Chest] ¡DATA LEAKAGE! {len(leaked)} Patient IDs aparecen en AMBOS "
                    f"splits. Primeros 5: {list(leaked)[:5]}. "
                    f"Usa EXCLUSIVAMENTE train_val_list.txt y test_list.txt oficiales."
                )
            else:
                log.info(f"[Chest] H2 — Verificación de leakage: 0 Patient IDs compartidos ✓")
        else:
            log.warning("[Chest] patient_ids_other=None — verificación de leakage H2 OMITIDA. "
                        "Pasa el set de Patient IDs del split complementario para validar.")

        # ── H2: Estadísticas de seguimiento longitudinal ──────────────────
        if "Follow-up #" in self.df.columns:
            followup_count = (self.df["Follow-up #"] > 0).sum()
            pct_followup   = 100 * followup_count / n_matched if n_matched > 0 else 0
            log.info(
                f"[Chest] H2 — Imágenes con Follow-up # > 0 (visitas repetidas): "
                f"{followup_count:,} ({pct_followup:.1f}%) — "
                f"refuerza la importancia de separar por Patient ID en el split."
            )

        # ── Archivos en disco ─────────────────────────────────────────────
        missing_disk = [r for r in self.df["Image Index"]
                        if not (self.img_dir / r).exists()]
        if missing_disk:
            log.warning(f"[Chest] {len(missing_disk):,} archivos no encontrados en disco "
                        f"'{self.img_dir}'. Primeros 5: {missing_disk[:5]}")

        # ── H1 + H6: Modo expert — vector multi-label + pesos de clase ───
        self.class_weights = None
        if self.mode == "expert":
            log.info("[Chest] Modo 'expert': preparando vectores multi-label de 14 patologías.")
            log.info("[Chest] H1  — Loss obligatoria: BCEWithLogitsLoss (NO CrossEntropyLoss).")
            log.info("[Chest] H6  — Alternativa: FocalLossMultiLabel(gamma=2.0) — "
                     "útil para Hernia/Pneumonia. Justificar en reporte si se usa.")
            log.info("[Chest] H6  — ⚠ NUNCA usar Accuracy como métrica principal. "
                     "Usar: AUC-ROC por clase + F1 Macro + AUPRC. "
                     "Benchmark DenseNet-121 ≈ 0.81 AUC macro. SOTA ≈ 0.85.")
            log.info("[Chest] H3  — Etiquetas generadas por NLP (~>90% precisión). "
                     "AUC > 0.85 significativo → revisar confounding antes de reportar.")

            # H1: Construir vector binario desde "Finding Labels"
            self.df = self.df.copy()
            self.df["label_vector"] = self.df["Finding Labels"].apply(
                self._parse_finding_labels
            )

            # Estadísticas de prevalencia por patología
            all_labels = np.stack(self.df["label_vector"].values)   # [N, 14]
            prevalence = all_labels.mean(axis=0)
            log.info("[Chest] Prevalencia por patología:")
            for name, prev in zip(CHEST_PATHOLOGIES, prevalence):
                bbox_tag = " [BBox✓]" if name in CHEST_BBOX_CLASSES else ""
                bar      = "█" * max(1, int(prev * 30))
                log.info(f"    {name:<22}: {prev:.3f}  {bar}{bbox_tag}")

            # H6: pos_weight = N_neg / N_pos por clase (para BCEWithLogitsLoss)
            n_pos  = all_labels.sum(axis=0).clip(min=1)
            n_neg  = len(all_labels) - n_pos
            self.class_weights = torch.tensor(n_neg / n_pos, dtype=torch.float32)
            log.info(f"[Chest] H6  — pos_weight BCEWithLogitsLoss — "
                     f"min: {self.class_weights.min():.1f} ({CHEST_PATHOLOGIES[self.class_weights.argmin()]}) | "
                     f"max: {self.class_weights.max():.1f} ({CHEST_PATHOLOGIES[self.class_weights.argmax()]})")
            log.debug("[Chest] pos_weight por clase: "
                      + " | ".join(f"{n}:{w:.1f}"
                                   for n, w in zip(CHEST_PATHOLOGIES,
                                                   self.class_weights.tolist())))

            # Verificar "No Finding" → vector todo-ceros
            no_finding_only = self.df["Finding Labels"].str.strip() == "No Finding"
            nf_count = no_finding_only.sum()
            log.info(f"[Chest] Imágenes 'No Finding' (vector todo-ceros): "
                     f"{nf_count:,} ({100*nf_count/n_matched:.1f}%) — "
                     f"la clase dominante que inflaría Accuracy.")

    @staticmethod
    def _parse_finding_labels(label_str: str) -> np.ndarray:
        """
        H1 — Convierte "Cardiomegaly|Effusion" → vector binario float32 [14].
        "No Finding" o strings con solo etiquetas no en CHEST_PATHOLOGIES → todo-ceros.
        """
        vec    = np.zeros(N_CHEST_CLASSES, dtype=np.float32)
        labels = [l.strip() for l in label_str.split("|")]
        for label in labels:
            if label in CHEST_PATHOLOGIES:
                vec[CHEST_PATHOLOGIES.index(label)] = 1.0
        return vec

    @staticmethod
    def get_patient_ids_from_file_list(csv_path: str, file_list: str) -> set:
        """
        H2 — Extrae Patient IDs de un split sin construir el dataset completo.
        Uso:
            ids_val   = ChestXray14Dataset.get_patient_ids_from_file_list(csv, val_list)
            train_ds  = ChestXray14Dataset(..., patient_ids_other=ids_val)
        """
        df = pd.read_csv(csv_path, usecols=["Image Index", "Patient ID"])
        with open(file_list) as f:
            valid = set(f.read().splitlines())
        return set(df[df["Image Index"].isin(valid)]["Patient ID"].unique())

    @staticmethod
    def load_bbox_index(bbox_csv_path: str) -> dict:
        """
        H5 — Carga BBox_List_2017.csv como índice {image_name: [bbox_list]}.

        Uso en FASE 2 / Dashboard para validar Attention Heatmaps:
            bbox_index = ChestXray14Dataset.load_bbox_index(bbox_csv_path)
            bboxes = bbox_index.get("00000001_000.png", [])
            # bboxes = [{"label": "Cardiomegaly", "x": 100, "y": 150, "w": 80, "h": 60}, ...]

        Si el heatmap del ViT se superpone con el BBox → el backbone aprendió
        features clínicamente relevantes (no artefactos de dominio).

        Solo 8 de 14 patologías tienen BBoxes (ver CHEST_BBOX_CLASSES).
        ~1,000 imágenes (~0.9% del total) tienen anotaciones.
        NO usar como supervisión de entrenamiento — insuficiente para detección.
        """
        df  = pd.read_csv(bbox_csv_path)
        idx = {}
        for _, row in df.iterrows():
            img = row["Image Index"]
            if img not in idx:
                idx[img] = []
            idx[img].append({
                "label": row["Finding Label"],
                "x":     int(row["Bbox [x"]),
                "y":     int(row["y"]),
                "w":     int(row["w"]),
                "h":     int(row["h]"]),
            })
        log.info(f"[Chest] H5 — BBox index cargado: {len(idx):,} imágenes con anotaciones | "
                 f"clases cubiertas: {sorted(CHEST_BBOX_CLASSES)}")
        return idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image Index"]
        img_path = self.img_dir / img_name
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[Chest] Error abriendo '{img_path}': {e}. "
                        f"Reemplazando con tensor cero.")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img = self.transform(img)

        if self.mode == "embedding":
            # FASE 0: el router aprende "soy imagen de ChestXray14"
            return img, self.expert_id, img_name
        else:
            # FASE 2: el Experto 0 aprende las 14 patologías con BCEWithLogitsLoss
            label_vec = torch.tensor(
                self.df.loc[idx, "label_vector"], dtype=torch.float32
            )
            return img, label_vec, img_name

 
# Esta clase es un DataLoader polimórfico: cambia su comportamiento dinámicamente según la fase del entrenamiento. 
# En modo embedding, oculta la patología y solo muestra la "etiqueta de experto" para que el Router se especialice 
# en identificar la modalidad; en modo expert, oculta la etiqueta de experto y muestra la patología real para que 
# el especialista clínico pueda realizar el diagnóstico.
class ISICDataset(Dataset):
    """
    ISIC 2019 — Multiclase, 9 clases (8 en train).
 
    Dos modos de uso:
      mode="embedding" → FASE 0: devuelve (img, expert_id=1, img_name)
                         El router aprende "esta imagen es de ISIC".
      mode="expert"    → FASE 2: devuelve (img, class_label, img_name)
                         El Experto 2 aprende la patología real (MEL/NV/BCC/...)
                         con CrossEntropyLoss sobre 8 clases.
 
    Nota sobre el split:
      El split correcto para ISIC es por lesion_id (ver análisis del dataset).
      Aquí se usa sample() como aproximación. Para producción: pre-generar
      splits por lesion_id y pasar el CSV ya filtrado.
    """
 
    # Orden canónico de las 8 clases de train (excluye UNK que es solo test)
    CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
 
    def __init__(self, gt_csv, img_dir, transform, split="train", mode="embedding"):
        """
        Args:
            gt_csv    : ruta a ISIC_2019_Training_GroundTruth.csv
            img_dir   : directorio con los .jpg
            transform : torchvision transform pipeline
            split     : "train" o "val"
            mode      : "embedding" (FASE 0) o "expert" (FASE 2)
        """
        assert mode in ("embedding", "expert"), \
            f"mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
 
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.expert_id = EXPERT_IDS["isic"]
        self.mode      = mode
 
        df = pd.read_csv(gt_csv)
        log.debug(f"[ISIC] CSV cargado: {len(df):,} filas en {gt_csv}")
 
        # Eliminar duplicados conocidos antes de cualquier split
        before = len(df)
        df = df[~df["image"].isin(["ISIC_0067980", "ISIC_0069013"])].reset_index(drop=True)
        removed = before - len(df)
        if removed:
            log.info(f"[ISIC] {removed} duplicados conocidos eliminados (ISIC_0067980, ISIC_0069013)")
 
        # Split aproximado — en producción usar lesion_id del metadata CSV
        if split == "train":
            self.df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        else:
            self.df = df.drop(df.sample(frac=0.8, random_state=42).index).reset_index(drop=True)
 
        log.info(f"[ISIC] split='{split}' mode='{mode}': {len(self.df):,} imágenes")
 
        # En modo expert: convertir one-hot → índice de clase (0–7)
        if self.mode == "expert":
            self.df = self.df.copy()
            self.df["class_label"] = self.df[self.CLASSES].values.argmax(axis=1)
            dist = self.df["class_label"].value_counts().sort_index()
            log.debug(f"[ISIC] Distribución de clases (modo expert):\n"
                      + "\n".join(f"    {self.CLASSES[i]:>4}: {dist.get(i,0):>5,}" for i in range(len(self.CLASSES))))
 
            # Detectar filas donde todas las clases son 0 (no debería ocurrir en train)
            all_zero = (self.df[self.CLASSES].sum(axis=1) == 0).sum()
            if all_zero:
                log.warning(f"[ISIC] {all_zero} filas con todas las columnas de clase en 0 "
                            f"(posiblemente clase UNK). Serán asignadas a clase 0 por argmax.")
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        img_path = self.img_dir / f"{img_name}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[ISIC] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero.")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img = self.transform(img)
 
        if self.mode == "embedding":
            return img, self.expert_id, img_name
        else:
            label = int(self.df.loc[idx, "class_label"])
            return img, label, img_name


# Esta clase es un cargador de datos jerárquico que infiere las etiquetas de grado de osteoartritis (0, 1, 2) 
# directamente desde la estructura de directorios (0/, 1/, 2/). Además, cumple con el requerimiento médico del 
# proyecto al aplicar un filtro CLAHE para resaltar detalles óseos antes de estandarizar la imagen al tamaño 
# requerido por el modelo.
class OAKneeDataset(Dataset):
    """
    Osteoarthritis Knee — Ordinal, 3 clases (0=Normal, 1=Leve, 2=Severo).
    Para embedding: etiqueta de experto = 2.
    CLAHE obligatorio antes del transform.
    """
    def __init__(self, root_dir, split="train", img_size=224):
        self.expert_id = EXPERT_IDS["oa"]
        self.img_size  = img_size
        self.samples   = []
 
        split_dir = Path(root_dir) / split
        if not split_dir.exists():
            log.error(f"[OA] Directorio de split no encontrado: {split_dir}")
            return
 
        class_counts = {}
        found_classes = []
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                found_classes.append(class_dir.name)
                imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                for img_path in imgs:
                    self.samples.append((img_path, int(class_dir.name)))
                class_counts[class_dir.name] = len(imgs)
 
        log.info(f"[OA] split='{split}': {len(self.samples):,} imágenes | "
                 f"clases encontradas: {found_classes}")
        log.debug(f"[OA] Distribución por clase: " +
                  " | ".join(f"clase {k}: {v:,}" for k, v in class_counts.items()))
 
        if len(self.samples) == 0:
            log.error(f"[OA] Dataset vacío en '{split_dir}'. "
                      f"Verifica que existan subcarpetas 0/, 1/, 2/ con imágenes.")
 
        # Advertir si solo hay 2 clases (KL5 puede estar ausente en algunos subsets)
        if len(found_classes) < 3:
            log.warning(f"[OA] Solo {len(found_classes)} clases encontradas ({found_classes}). "
                        f"Se esperaban 3 (Normal/Leve/Severo). "
                        f"¿El ZIP incluye las 3 carpetas?")
 
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        img_path, kl_class = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[OA] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero.")
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            img = self.final_transform(img)
            return img, self.expert_id, str(img_path.name)
 
        img = apply_clahe(img)
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        img = self.final_transform(img)
        return img, self.expert_id, str(img_path.name)


# Esta clase es el procesador de volumen a imagen del sistema. Toma parches 3D de tomografías pulmonares (LUNA16), 
# normaliza su densidad radiológica, estandariza su tamaño volumétrico y los proyecta como imágenes RGB 2D, 
# permitiendo que un modelo entrenado en 2D pueda "ver" la estructura tridimensional de los tejidos pulmonares.
class LUNA16Dataset(Dataset):
    """
    LUNA16 — Clasificación de candidatos 3D.
    Para embedding: convierte volumen 3D → representación 2D (3 slices).
    Etiqueta de experto = 3.
    """
    def __init__(self, patches_dir, candidates_csv):
        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["luna"]
 
        if not self.patches_dir.exists():
            log.error(f"[LUNA16] Directorio de parches no encontrado: {self.patches_dir}")
            self.samples = []
            return
 
        df = pd.read_csv(candidates_csv)
        log.debug(f"[LUNA16] candidates_V2.csv: {len(df):,} candidatos totales")
 
        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            patch_file = self.patches_dir / f"candidate_{row.name:06d}.npy"
            if patch_file.exists():
                self.samples.append((patch_file, int(row["class"])))
            else:
                missing += 1
 
        n_pos = sum(1 for _, c in self.samples if c == 1)
        n_neg = sum(1 for _, c in self.samples if c == 0)
        ratio = n_neg / n_pos if n_pos > 0 else float("inf")
 
        log.info(f"[LUNA16] {len(self.samples):,} parches cargados | "
                 f"positivos (nódulo): {n_pos:,} | negativos: {n_neg:,} | ratio neg/pos: {ratio:.0f}:1")
 
        if missing > 0:
            log.warning(f"[LUNA16] {missing:,} parches en el CSV no encontrados en disco '{self.patches_dir}'. "
                        f"¿Ejecutaste la extracción de parches 3D previa?")
        if n_pos == 0:
            log.error(f"[LUNA16] ¡Cero candidatos positivos! El modelo no podrá aprender nódulos. "
                      f"Verifica que candidates_V2.csv tenga filas con class=1.")
        if ratio > 1000:
            log.warning(f"[LUNA16] Ratio neg/pos extremo ({ratio:.0f}:1). "
                        f"Considera FocalLoss(gamma=2) o subsampling de negativos en el DataLoader.")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        patch_file, _ = self.samples[idx]
        try:
            volume = np.load(patch_file)          # [64, 64, 64] HU crudo
        except Exception as e:
            log.warning(f"[LUNA16] Error cargando parche '{patch_file}': {e}. "
                        f"Reemplazando con tensor cero.")
            return torch.zeros(3, 224, 224), self.expert_id, patch_file.stem
 
        # Detectar volumen completamente vacío (padding excesivo o error de extracción)
        if volume.max() == volume.min():
            log.warning(f"[LUNA16] Parche '{patch_file.name}' es constante "
                        f"(valor={volume.max():.1f} HU). Posible error de extracción.")
 
        volume   = normalize_hu(volume)
        volume_t = resize_volume_3d(volume)
        img      = volume_to_vit_input(volume_t)
        return img, self.expert_id, patch_file.stem
# Sugerencia:
# Dado que este dataset carga archivos .npy (que suelen ser rápidos de leer), el entrenamiento debería ser fluido. 
# Sin embargo, si notas que el entrenamiento es lento, el cuello de botella será volume_to_vit_input debido 
# a la interpolación bilinear constante. Si eso ocurre, podrías pre-procesar y guardar los tensores resultantes 
# [3, 224, 224] en disco una sola vez, y en esta clase simplemente cargarlos ya listos. Pero por ahora, esta 
# implementación es la correcta y más limpia para empezar. ¡Excelente trabajo!


# Esta clase es el cargador especializado para el dataset de cáncer de páncreas. Su rol es transformar volúmenes 
# CT 3D crudos en representaciones 2D compatibles con tu Backbone ViT, etiquetando cada muestra como "Experto 4" 
# para que el Router pueda gestionar el balanceo de carga y dirigir estas imágenes exclusivamente al experto 
# especializado en anatomía abdominal.
class PancreasDataset(Dataset):
    """
    Pancreatic Cancer CT 3D — Binario, 2 clases.
    Igual que LUNA16: asume parches 3D pre-extraídos.
    Etiqueta de experto = 4.
    """
    def __init__(self, patches_dir):
        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["pancreas"]
 
        if not self.patches_dir.exists():
            log.error(f"[Pancreas] Directorio no encontrado: {self.patches_dir}")
            self.samples = []
            return
 
        self.samples = list(self.patches_dir.glob("*.npy"))
        log.info(f"[Pancreas] {len(self.samples):,} volúmenes encontrados en '{self.patches_dir}'")
 
        if len(self.samples) == 0:
            log.error(f"[Pancreas] ¡Sin archivos .npy! Verifica que hayas preprocesado "
                      f"los volúmenes PANORAMA y que estén en '{self.patches_dir}'.")
        elif len(self.samples) < 50:
            log.warning(f"[Pancreas] Solo {len(self.samples)} volúmenes — dataset muy pequeño. "
                        f"Considera k-fold CV (k=5) para estimaciones más estables.")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        patch_file = self.samples[idx]
        try:
            volume = np.load(patch_file)
        except Exception as e:
            log.warning(f"[Pancreas] Error cargando '{patch_file}': {e}. "
                        f"Reemplazando con tensor cero.")
            return torch.zeros(3, 224, 224), self.expert_id, patch_file.stem
 
        if volume.max() == volume.min():
            log.warning(f"[Pancreas] Volumen '{patch_file.name}' es constante "
                        f"(valor={volume.max():.1f} HU). ¿HU clip mal aplicado?")
 
        # Pancreas usa rango HU abdominal [-100, 400], distinto a LUNA [-1000, 400]
        volume   = normalize_hu(volume, min_hu=-100, max_hu=400)
        volume_t = resize_volume_3d(volume)
        img      = volume_to_vit_input(volume_t)
        return img, self.expert_id, patch_file.stem
# Consejo:
# Como este experto es el último (Experto 4), el balance de carga 
# (max(f_i)/min(f_i) < 1.30) será tu mayor desafío aquí. Los datasets 3D suelen 
# ser más pequeños que los 2D (NIH/ISIC). Asegúrate de que durante el 
# entrenamiento del Router (FASE 1), tu DataLoader mixto use una estrategia de 
# oversampling para que el experto 4 reciba suficientes muestras y no sea 
# ignorado por el Router en favor de los datasets 2D más grandes.
 
 
# ──────────────────────────────────────────────────────────
# 4. BACKBONE ViT — EXTRACTOR (CONGELADO)
# ──────────────────────────────────────────────────────────
#
# ¿Qué hace esta sección?
#   Carga un Vision Transformer preentrenado en ImageNet (via timm),
#   congela TODOS sus parámetros (requires_grad=False) y lo convierte
#   en un extractor puro de representaciones: recibe una imagen [B,3,224,224]
#   y devuelve su CLS token [B, d_model] — el vector z que resume la imagen.
#
# ¿Dónde se alinea con la guía?
#   Sección 4.1 — "Backbone ViT compartido":
#     "El backbone ViT extrae representaciones de cada imagen.
#      El CLS token resultante alimenta al router para decidir qué experto activa."
#   El backbone está CONGELADO en FASE 0 (y también en FASE 1).
#   Solo se descongela parcialmente en FASE 3 (fine-tuning global, LR=1e-6).
#
# ¿Por qué ViT-Tiny por defecto y no Swin o CvT?
#   ViT-Tiny (d_model=192) es el "safe choice" para la primera corrida:
#   consume menos VRAM (~2 GB), es más rápido y verifica que toda la lógica
#   MoE funciona antes de escalar. Para el ablation study del reporte técnico,
#   se recomienda correr también con swin_tiny (d_model=768) — los embeddings
#   de Swin tienen sesgo inductivo jerárquico que probablemente mejore la
#   accuracy de todos los routers. Ver BACKBONE_CONFIGS para la comparativa.


# Esta función actúa como un "Selector de Extracción de Características". Carga modelos preentrenados del 
# ecosistema timm, los prepara para actuar exclusivamente como extractores de features 
# (eliminando la capa de clasificación final) y los congela para garantizar que el backbone no aprenda durante 
# la FASE 0, asegurando la reproducibilidad de los embeddings.
def load_frozen_backbone(backbone_name="vit_tiny_patch16_224", device="cuda"):
    """
    Carga un backbone ViT/Swin/CvT preentrenado (timm) y lo congela completamente.
    Solo se usa para forward pass — sin gradientes (requires_grad=False en todo).
 
    Retorna:
      model   — backbone listo para inferencia, en `device`
      d_model — dimensión del CLS token (varía según backbone, ver BACKBONE_CONFIGS)
 
    Backbones disponibles y sus d_model:
      vit_tiny_patch16_224           → d_model=192  (default, más liviano)
      swin_tiny_patch4_window7_224   → d_model=768  (recomendado para ablation final)
      cvt_13                         → d_model=384  (balance localidad/atención global)
 
    El d_model determina el tamaño de Z_train.npy y los parámetros del router.
    fase1_ablation_router.py lo detecta automáticamente desde Z_train.shape[1].
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Backbone '{backbone_name}' no reconocido.\n"
            f"Opciones válidas: {list(BACKBONE_CONFIGS.keys())}"
        )
 
    expected_d = BACKBONE_CONFIGS[backbone_name]["d_model"]
    vram_est   = BACKBONE_CONFIGS[backbone_name]["vram_gb"]
    log.info(f"[Backbone] Seleccionado  : {backbone_name}")
    log.info(f"[Backbone] d_model esp.  : {expected_d}")
    log.info(f"[Backbone] VRAM estimada : ~{vram_est} GB")
 
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0          # elimina la cabeza de clasificación → devuelve CLS token
    )
 
    # Congelar TODOS los parámetros — FASE 0 es extracción pura, no aprendizaje
    for param in model.parameters():
        param.requires_grad = False
 
    model.eval()
    model.to(device)
 
    # Verificar que el congelamiento fue efectivo — cualquier grad activo es un ERROR grave
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        log.error(f"[Backbone] ¡{len(trainable)} parámetros con requires_grad=True tras el congelamiento! "
                  f"Los embeddings NO serán reproducibles. Primeros 3: {trainable[:3]}")
    else:
        log.debug(f"[Backbone] Congelamiento verificado: 0 parámetros entrenables.")
 
    # Verificar d_model con un forward pass dummy
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
 
    actual_d = out.shape[1]
    if actual_d != expected_d:
        log.warning(f"[Backbone] d_model real ({actual_d}) difiere del esperado ({expected_d}). "
                    f"Actualiza BACKBONE_CONFIGS si agregaste un backbone nuevo.")
    else:
        log.debug(f"[Backbone] d_model verificado: {actual_d} ✓")
 
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"[Backbone] d_model real (CLS dim) : {actual_d}")
    log.info(f"[Backbone] Parámetros congelados  : {total_params:,}")
 
    return model, actual_d
# Consejo para el reporte:
# Puedes incluir una sección llamada "Evaluación de Backbones" donde muestres 
# una tabla comparando los 3 backbones. Al dejar tu código así de flexible, 
# puedes demostrar que realizaste una exploración exhaustiva para elegir el 
# mejor extractor de características para tu sistema MoE. 


# ──────────────────────────────────────────────────────────
# 5. EXTRACCIÓN DE EMBEDDINGS
# ──────────────────────────────────────────────────────────
 

# Esta función actúa como el motor de extracción de características del sistema. Recorre los dataloaders de 
# todos los datasets (2D y 3D) y utiliza el backbone (previamente congelado) para transformar imágenes y 
# volúmenes en vectores numéricos compactos (embeddings), almacenándolos eficientemente en memoria mediante 
# pre-asignación (pre-allocation) antes de guardarlos en disco.
@torch.no_grad()
def extract_embeddings(model, dataloader, device, d_model, desc=""):
    """
    Pasa el dataloader completo por el backbone y acumula CLS tokens.
    Detecta activamente: NaN/Inf en embeddings, VRAM excesiva, batches vacíos.
    """
    n_total   = len(dataloader.dataset)
    Z         = np.zeros((n_total, d_model), dtype=np.float32)
    y_expert  = np.zeros(n_total, dtype=np.int64)
    img_names = []
    cursor    = 0
 
    nan_batches  = 0    # batches con NaN en el embedding
    inf_batches  = 0    # batches con Inf en el embedding
    t_start      = time.time()
 
    log.info(f"[Extract/{desc}] Iniciando extracción: {n_total:,} imágenes | "
             f"d_model={d_model} | batch_size={dataloader.batch_size}")
 
    for batch_idx, (imgs, experts, names) in enumerate(dataloader):
        imgs = imgs.to(device)
        z    = model(imgs)           # [B, d_model]
        z_np = z.cpu().numpy()
        B    = z_np.shape[0]
 
        # ── Detección de NaN / Inf — señal de problema grave ──────────────
        if np.isnan(z_np).any():
            nan_batches += 1
            log.error(f"[Extract/{desc}] NaN detectado en batch {batch_idx} "
                      f"(imágenes {cursor}–{cursor+B}). "
                      f"Causa probable: imagen corrupta o overflow en el backbone.")
        if np.isinf(z_np).any():
            inf_batches += 1
            log.error(f"[Extract/{desc}] Inf detectado en batch {batch_idx} "
                      f"(imágenes {cursor}–{cursor+B}). "
                      f"Causa probable: normalización incorrecta de píxeles.")
 
        Z[cursor:cursor + B]        = z_np
        y_expert[cursor:cursor + B] = experts.numpy()
        img_names.extend(names)
        cursor += B
 
        # ── Progreso cada 50 batches ───────────────────────────────────────
        if batch_idx % 50 == 0:
            elapsed  = time.time() - t_start
            speed    = cursor / elapsed if elapsed > 0 else 0
            eta_s    = (n_total - cursor) / speed if speed > 0 else 0
            log.info(f"[Extract/{desc}] {cursor:>6,}/{n_total:,} "
                     f"({100*cursor/n_total:.1f}%) | "
                     f"{speed:.0f} img/s | ETA {eta_s/60:.1f} min")
 
        # ── VRAM cada 200 batches (solo GPU) ──────────────────────────────
        if device == "cuda" and batch_idx % 200 == 0:
            used_gb  = torch.cuda.memory_allocated() / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            pct      = 100 * used_gb / total_gb
            log.debug(f"[Extract/{desc}] VRAM: {used_gb:.2f}/{total_gb:.1f} GB ({pct:.1f}%)")
            if pct > 90:
                log.warning(f"[Extract/{desc}] VRAM al {pct:.1f}% — riesgo de OOM. "
                            f"Considera reducir --batch_size.")
 
    elapsed_total = time.time() - t_start
    log.info(f"[Extract/{desc}] Completado: {cursor:,} embeddings en {elapsed_total:.1f}s "
             f"({cursor/elapsed_total:.0f} img/s)")
 
    # ── Resumen de anomalías ───────────────────────────────────────────────
    if nan_batches or inf_batches:
        log.error(f"[Extract/{desc}] RESUMEN ANOMALÍAS: "
                  f"NaN en {nan_batches} batches | Inf en {inf_batches} batches. "
                  f"Los embeddings resultantes no son confiables.")
    else:
        log.info(f"[Extract/{desc}] Sin NaN ni Inf detectados ✓")
 
    # Verificar norma L2 media — embeddings ViT sanos tienen norma ~10–30
    norms = np.linalg.norm(Z[:cursor], axis=1)
    log.debug(f"[Extract/{desc}] Norma L2 embeddings — "
              f"media: {norms.mean():.2f} | min: {norms.min():.2f} | max: {norms.max():.2f}")
    if norms.mean() < 1.0:
        log.warning(f"[Extract/{desc}] Norma L2 media muy baja ({norms.mean():.3f}). "
                    f"¿El backbone devuelve el CLS token correctamente?")
 
    return Z[:cursor], y_expert[:cursor], img_names
# Un pequeño consejo "pro" para tu reporte:
# Cuando describas esta función en tu reporte técnico, menciona que la pre-asignación de memoria (NumPy zeros) 
# y el uso de torch.no_grad() fueron decisiones estratégicas para optimizar la eficiencia del pipeline en el 
# hardware limitado (2 GPUs de 12GB), asegurando que el proceso de extracción escalara correctamente con 
# datasets masivos como NIH ChestXray14 (112k imágenes).


# ──────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────
 

# Esta función actúa como una Fábrica de Datasets con Auditoría de Leakage. Su función es orquestar la carga 
# de los 5 datasets, pero con una capa crítica de seguridad: antes de instanciar el dataset del NIH, extrae 
# los identificadores de pacientes de ambos splits (train/val) y los cruza para garantizar que no haya filtración 
# de información entre ellos. Finalmente, fusiona todo en un ConcatDataset para que el Router entrene con una 
# distribución equilibrada de las 5 modalidades.
def build_datasets(cfg):
    """
    Construye los datasets de train y val para los 5 expertos de dominio.
    Devuelve (train_dataset, val_dataset) como ConcatDataset.
 
    cfg: dict con rutas de cada dataset (ver argparse abajo)
 
    Expertos incluidos aquí (ID 0–4):
      0 → ChestXray14, 1 → ISIC, 2 → OA Rodilla, 3 → LUNA16, 4 → Pancreas
 
    Experto 5 (OOD/Error, ID=5) — NO incluido intencionalmente:
      No tiene dataset propio. Se define como MLP en FASE 1 y aprende
      su comportamiento mediante L_error durante el entrenamiento del router.
      Ver constante N_EXPERTS_TOTAL = 6 si necesitas el conteo completo.
    """
    transform_2d = build_2d_transform(img_size=224)
 
    train_datasets = []
    val_datasets   = []
 
    # ── Experto 1: NIH ChestXray14 ──
    if cfg.get("chest_csv") and cfg.get("chest_imgs"):
        log.info("[Dataset] Cargando NIH ChestXray14 (Experto 0)...")

        # H2: obtener Patient IDs de cada split antes de construir los datasets
        if cfg.get("chest_train_list") and cfg.get("chest_val_list"):
            ids_train = ChestXray14Dataset.get_patient_ids_from_file_list(
                cfg["chest_csv"], cfg["chest_train_list"])
            ids_val   = ChestXray14Dataset.get_patient_ids_from_file_list(
                cfg["chest_csv"], cfg["chest_val_list"])
            log.debug(f"[Chest] Patient IDs pre-cargados: "
                      f"train={len(ids_train):,} | val={len(ids_val):,}")
        else:
            ids_train, ids_val = None, None
            log.warning("[Chest] No se pueden verificar Patient IDs — faltan rutas de file_list.")

        # H4: filter_view controla el filtro PA/AP (None = sin filtro)
        view_filter = cfg.get("chest_view_filter", None)

        chest_train = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["chest_train_list"],
            transform=transform_2d,
            mode="embedding",
            patient_ids_other=ids_val,
            filter_view=view_filter,
        )
        chest_val = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["chest_val_list"],
            transform=transform_2d,
            mode="embedding",
            patient_ids_other=ids_train,
            filter_view=view_filter,
        )
        train_datasets.append(chest_train)
        val_datasets.append(chest_val)
        log.info(f"  → train: {len(chest_train):,}  val: {len(chest_val):,}")

        # H5: cargar BBox index si se proporciona (para auditoría de heatmaps en dashboard)
        if cfg.get("chest_bbox_csv"):
            try:
                _ = ChestXray14Dataset.load_bbox_index(cfg["chest_bbox_csv"])
                log.info("[Chest] H5 — BBox index cargado. "
                         "Disponible para validación de heatmaps en FASE 2/dashboard.")
            except Exception as e:
                log.warning(f"[Chest] H5 — No se pudo cargar BBox_List_2017.csv: {e}")
        else:
            log.info("[Chest] H5 — BBox no cargado (pasa --chest_bbox_csv para activar). "
                     "Opcional: útil para validación cualitativa de heatmaps del dashboard.")
    else:
        log.warning("[Dataset] NIH ChestXray14 OMITIDO — falta --chest_csv o --chest_imgs")
 
    # ── Experto 2: ISIC 2019 ──
    if cfg.get("isic_gt") and cfg.get("isic_imgs"):
        log.info("[Dataset] Cargando ISIC 2019 (Experto 1)...")
        isic_train = ISICDataset(cfg["isic_gt"], cfg["isic_imgs"], transform_2d, split="train")
        isic_val   = ISICDataset(cfg["isic_gt"], cfg["isic_imgs"], transform_2d, split="val")
        train_datasets.append(isic_train)
        val_datasets.append(isic_val)
        log.info(f"  → train: {len(isic_train):,}  val: {len(isic_val):,}")
    else:
        log.warning("[Dataset] ISIC 2019 OMITIDO — falta --isic_gt o --isic_imgs")
 
    # ── Experto 3: OA Rodilla ──
    if cfg.get("oa_root"):
        log.info("[Dataset] Cargando Osteoarthritis Knee (Experto 2)...")
        oa_train = OAKneeDataset(cfg["oa_root"], split="train")
        oa_val   = OAKneeDataset(cfg["oa_root"], split="val")
        train_datasets.append(oa_train)
        val_datasets.append(oa_val)
        log.info(f"  → train: {len(oa_train):,}  val: {len(oa_val):,}")
    else:
        log.warning("[Dataset] OA Rodilla OMITIDO — falta --oa_root")
 
    # ── Experto 4: LUNA16 ──
    if cfg.get("luna_patches") and cfg.get("luna_csv"):
        log.info("[Dataset] Cargando LUNA16 parches 3D (Experto 3)...")
        luna_train = LUNA16Dataset(cfg["luna_patches"] + "/train", cfg["luna_csv"])
        luna_val   = LUNA16Dataset(cfg["luna_patches"] + "/val",   cfg["luna_csv"])
        train_datasets.append(luna_train)
        val_datasets.append(luna_val)
        log.info(f"  → train: {len(luna_train):,}  val: {len(luna_val):,}")
    else:
        log.warning("[Dataset] LUNA16 OMITIDO — falta --luna_patches o --luna_csv")
 
    # ── Experto 5: Pancreas ──
    if cfg.get("pancreas_patches_train") and cfg.get("pancreas_patches_val"):
        log.info("[Dataset] Cargando Pancreas CT 3D (Experto 4)...")
        panc_train = PancreasDataset(cfg["pancreas_patches_train"])
        panc_val   = PancreasDataset(cfg["pancreas_patches_val"])
        train_datasets.append(panc_train)
        val_datasets.append(panc_val)
        log.info(f"  → train: {len(panc_train):,}  val: {len(panc_val):,}")
    else:
        log.warning("[Dataset] Pancreas OMITIDO — falta --pancreas_patches_train o --pancreas_patches_val")
 
    # ── Experto 6 (OOD) — ausente intencionalmente ──
    log.info("[Dataset] Experto 5 (OOD) — sin dataset en FASE 0. Se inicializa en FASE 1.")
 
    if not train_datasets:
        log.error("[Dataset] ¡Ningún dataset cargado! Verifica que al menos una ruta sea válida.")
 
    # Advertir si el balance entre expertos es muy desigual (objetivo: max/min < 1.30)
    sizes = [len(d) for d in train_datasets]
    if sizes:
        ratio = max(sizes) / min(sizes)
        log.info(f"[Dataset] Tamaños train por experto: {sizes}")
        if ratio > 10:
            log.warning(f"[Dataset] Balance muy desigual: max/min = {ratio:.1f}x. "
                        f"Considera WeightedRandomSampler en el DataLoader de FASE 1.")
 
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)
# Sugerencia:
# Cuando ejecutes esto, asegúrate de que el orden de los datasets en train_datasets coincida con el orden de 
# EXPERT_IDS. Como lo has hecho de forma secuencial y ordenada, no deberías tener problemas, pero mantén esa 
# consistencia si añades más expertos en el futuro


# Esta función es el "Orquestador de la FASE 0". Su rol es gestionar el ciclo de vida completo de la extracción: 
# desde la validación del hardware y la carga del modelo congelado, pasando por la creación de los dataloaders 
# multimodales, hasta la ejecución del forward pass del backbone y la persistencia de los resultados 
# (embeddings) y auditorías (nombres de archivo) en el almacenamiento.
def main(args):
    # ── Inicializar logging lo primero de todo ──
    global log
    log = setup_logging(args.output_dir)
 
    log.info("=" * 60)
    log.info("FASE 0 — Extracción de CLS tokens (Embeddings)")
    log.info("=" * 60)
    log.info(f"Argumentos recibidos: {vars(args)}")
 
    # ── Setup de dispositivo ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[Setup] Dispositivo: {device}")
    if device == "cuda":
        log.info(f"[Setup] GPU           : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"[Setup] VRAM total    : {total_vram:.1f} GB")
        expected_vram = BACKBONE_CONFIGS[args.backbone]["vram_gb"]
        if expected_vram > total_vram * 0.8:
            log.warning(f"[Setup] El backbone '{args.backbone}' requiere ~{expected_vram} GB "
                        f"pero solo hay {total_vram:.1f} GB. Considera reducir --batch_size.")
    else:
        log.warning("[Setup] Corriendo en CPU — la extracción será muy lenta. "
                    "Se recomienda GPU para un dataset de esta escala.")
 
    # ── Cargar backbone congelado ──
    log.info("[Setup] Cargando backbone...")
    model, d_model = load_frozen_backbone(args.backbone, device)
 
    # ── Construir datasets ──
    cfg = {
        "chest_csv":         args.chest_csv,
        "chest_imgs":        args.chest_imgs,
        "chest_train_list":  args.chest_train_list,
        "chest_val_list":    args.chest_val_list,
        "chest_view_filter": args.chest_view_filter,  # H4: "PA", "AP" o None
        "chest_bbox_csv":    args.chest_bbox_csv,      # H5: BBox_List_2017.csv (opcional)
        "isic_gt":           args.isic_gt,
        "isic_imgs":         args.isic_imgs,
        "oa_root":           args.oa_root,
        "luna_patches":      args.luna_patches,
        "luna_csv":          args.luna_csv,
        "pancreas_patches_train": args.pancreas_patches_train,
        "pancreas_patches_val":   args.pancreas_patches_val,
    }
 
    log.info("[Setup] Construyendo datasets...")
    train_dataset, val_dataset = build_datasets(cfg)
    log.info(f"[Dataset] Total train: {len(train_dataset):,}")
    log.info(f"[Dataset] Total val  : {len(val_dataset):,}")
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda")
    )
 
    # ── Extraer embeddings ──
    os.makedirs(args.output_dir, exist_ok=True)
 
    log.info("[FASE 0] Extrayendo embeddings de TRAIN...")
    t0 = time.time()
    Z_train, y_train, names_train = extract_embeddings(
        model, train_loader, device, d_model, desc="train"
    )
    log.info(f"[FASE 0] TRAIN completado en {time.time()-t0:.1f}s")
 
    log.info("[FASE 0] Extrayendo embeddings de VAL...")
    t0 = time.time()
    Z_val, y_val, names_val = extract_embeddings(
        model, val_loader, device, d_model, desc="val"
    )
    log.info(f"[FASE 0] VAL completado en {time.time()-t0:.1f}s")
 
    # ── Guardar en disco ──
    out = Path(args.output_dir)
    log.info(f"[Guardado] Escribiendo archivos en '{out}'...")
    np.save(out / "Z_train.npy", Z_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "Z_val.npy",   Z_val)
    np.save(out / "y_val.npy",   y_val)
    log.debug(f"[Guardado] Z_train.npy: {Z_train.shape}  ({Z_train.nbytes/1e6:.1f} MB)")
    log.debug(f"[Guardado] Z_val.npy  : {Z_val.shape}  ({Z_val.nbytes/1e6:.1f} MB)")
 
    with open(out / "names_train.txt", "w") as f:
        f.write("\n".join(names_train))
    with open(out / "names_val.txt", "w") as f:
        f.write("\n".join(names_val))
 
    backbone_meta = {
        "backbone": args.backbone,
        "d_model":  int(Z_train.shape[1]),
        "n_train":  int(Z_train.shape[0]),
        "n_val":    int(Z_val.shape[0]),
        "vram_gb":  BACKBONE_CONFIGS[args.backbone]["vram_gb"],
    }
    with open(out / "backbone_meta.json", "w") as f:
        json.dump(backbone_meta, f, indent=2)
    log.info(f"[Guardado] backbone_meta.json escrito")
 
    # ── Reporte final ──
    log.info("=" * 55)
    log.info("FASE 0 completada")
    log.info("=" * 55)
    log.info(f"Z_train : {Z_train.shape}  ({Z_train.nbytes/1e6:.1f} MB)")
    log.info(f"Z_val   : {Z_val.shape}  ({Z_val.nbytes/1e6:.1f} MB)")
 
    log.info("Distribución de expertos en train (dominios clínicos 0–4):")
    for exp_name, exp_id in EXPERT_IDS.items():
        count = (y_train == exp_id).sum()
        pct   = count / len(y_train) * 100
        log.info(f"  Experto {exp_id} ({exp_name:<10}): {count:>6,}  ({pct:.1f}%)")
    log.info(f"  Experto 5 (ood       ):      0  (0.0%)  ← esperado: sin dataset en FASE 0")
    log.info(f"  [Experto 5 se inicializa en FASE 1 como MLP OOD — entrenado via L_error]")
    log.info(f"  N_EXPERTS_DOMAIN = {N_EXPERTS_DOMAIN} | N_EXPERTS_TOTAL = {N_EXPERTS_TOTAL}")
 
    # Verificar balance entre expertos (objetivo del proyecto: max/min < 1.30 en FASE 1)
    counts = [(y_train == i).sum() for i in range(N_EXPERTS_DOMAIN)]
    if min(counts) > 0:
        balance = max(counts) / min(counts)
        if balance > 10:
            log.warning(f"[Balance] ratio max/min = {balance:.1f}x — muy desigual. "
                        f"Usa WeightedRandomSampler en FASE 1 para compensar.")
        else:
            log.info(f"[Balance] ratio max/min = {balance:.1f}x")
 
    log.info(f"Archivos guardados en: {args.output_dir}")
    log.info(f"Siguiente paso: python fase1_ablation_router.py --embeddings {args.output_dir}")


# ──────────────────────────────────────────────────────────
# 7. ARGPARSE
# ──────────────────────────────────────────────────────────


# Este bloque es el "Panel de Control" del pipeline. Utiliza la librería argparse para permitir que tú, como 
# investigador, definas los hiperparámetros (como batch_size) y las rutas de los datos (de los 5 datasets) 
# desde la consola, garantizando que el experimento sea flexible, documentado y fácil de replicar para tu 
# ablation study.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FASE 0 — Extracción de embeddings MoE")
 
    # ── Backbone ──────────────────────────────────────────────────────────────
    # Usa --backbone para el ablation study del backbone (ver BACKBONE_CONFIGS).
    # CONSEJO: corre FASE 0 tres veces con distintos backbones y guarda en
    # carpetas separadas para comparar la calidad de embeddings en FASE 1:
    #   python fase0_extract_embeddings.py --backbone vit_tiny_patch16_224 --output_dir ./embeddings/vit_tiny
    #   python fase0_extract_embeddings.py --backbone swin_tiny_patch4_window7_224 --output_dir ./embeddings/swin_tiny
    #   python fase0_extract_embeddings.py --backbone cvt_13 --output_dir ./embeddings/cvt
    parser.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(BACKBONE_CONFIGS.keys()),
        help=(
            "Backbone para extraer CLS tokens. Opciones:\n"
            "  vit_tiny_patch16_224        → d_model=192, ~2GB VRAM (DEFAULT: primera corrida)\n"
            "  swin_tiny_patch4_window7_224→ d_model=768, ~4GB VRAM (recomendado: ablation final)\n"
            "  cvt_13                      → d_model=384, ~3GB VRAM (balance intermedio)\n"
            "El d_model resultante determina el tamaño de Z_train.npy y los parámetros del router."
        )
    )
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--output_dir",   default="./embeddings",
                        help="Carpeta donde se guardan Z_train.npy y Z_val.npy")
 
    # Rutas NIH ChestXray14
    parser.add_argument("--chest_csv",        default=None)
    parser.add_argument("--chest_imgs",       default=None)
    parser.add_argument("--chest_train_list", default=None,
                        help="train_val_list.txt oficial")
    parser.add_argument("--chest_val_list",   default=None,
                        help="test_list.txt oficial (como val para extracción)")
    parser.add_argument(
        "--chest_view_filter", default=None, choices=["PA", "AP", None],
        help=(
            "H4 — Filtrar por vista radiográfica. Opciones: PA, AP o None (sin filtro). "
            "Recomendado: 'PA' para estudios controlados (evita sesgo de dominio AP). "
            "Documenta en el Reporte Técnico si se aplica."
        )
    )
    parser.add_argument(
        "--chest_bbox_csv", default=None,
        help=(
            "H5 — Ruta a BBox_List_2017.csv (opcional). "
            "Carga el índice de bounding boxes para validación cualitativa de "
            "Attention Heatmaps en FASE 2/dashboard. "
            "Solo ~1,000 imágenes (~8 clases) tienen BBoxes disponibles."
        )
    )
 
    # Rutas ISIC 2019
    parser.add_argument("--isic_gt",          default=None,
                        help="ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--isic_imgs",        default=None,
                        help="Carpeta con imágenes .jpg de ISIC")
 
    # Rutas OA Rodilla
    parser.add_argument("--oa_root",          default=None,
                        help="Raíz con subcarpetas train/val/test y clases 0,1,2")
 
    # Rutas LUNA16
    parser.add_argument("--luna_patches",     default=None,
                        help="Carpeta con parches 3D .npy pre-extraídos (train/ y val/)")
    parser.add_argument("--luna_csv",         default=None,
                        help="candidates_V2.csv")
 
    # Rutas Pancreas
    parser.add_argument("--pancreas_patches_train", default=None)
    parser.add_argument("--pancreas_patches_val",   default=None)
 
    args = parser.parse_args()
    main(args)