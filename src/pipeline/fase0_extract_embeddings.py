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
 
 
# Esta función aplica CLAHE sobre la imagen a su RESOLUCIÓN ORIGINAL antes de cualquier
# resize. Este orden es crítico (H4): si se aplica CLAHE después del resize a 224×224,
# el espacio articular —región diagnósticamente más importante— puede haber quedado
# reducido a unos pocos píxeles, limitando severamente el efecto del realce de contraste.
#
# Detalle de canales (H4): las radiografías OA son imágenes en escala de grises guardadas
# como RGB con los tres canales idénticos (Image.mode == "RGB" pero canales duplicados).
# CLAHE opera sobre un único canal uint8, así que:
#   1. Se extrae el canal L (luminancia) → array uint8 monocanal
#   2. Se aplica CLAHE
#   3. Se replica a 3 canales idénticos para compatibilidad con el backbone ImageNet
# Esta secuencia es idempotente si la imagen ya es genuinamente monocanal.
def apply_clahe(img_pil: Image.Image,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> Image.Image:
    """
    H4 — CLAHE sobre imagen a resolución ORIGINAL (antes del resize).

    SIEMPRE llamar ANTES de img.resize() — nunca después.

    Flujo interno:
      img_pil (RGB, canales iguales) → canal L uint8 → CLAHE → RGB (canales iguales)

    Args:
        img_pil    : imagen PIL en modo RGB (gris guardado como RGB o genuinamente RGB)
        clip_limit : umbral de clipping del histograma (default 2.0 — conservador)
        tile_grid  : tamaño de celda para ecualización local (default 8×8)

    Returns:
        imagen PIL RGB con contraste óseo mejorado, misma resolución que la entrada
    """
    # Extraer canal L — correcto tanto para gris-como-RGB como para genuinamente RGB
    img_gray = np.array(img_pil.convert("L"), dtype=np.uint8)
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    # Replicar a 3 canales para compatibilidad con normalización ImageNet del backbone
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

 

def apply_circular_crop(img_pil):
    """
    H6/Item-8 — Elimina los bordes negros circulares de las imágenes BCN_20000.

    Las imágenes de dermoscopio de BCN tienen un campo de visión circular
    con fondo negro. Recortar esos bordes evita que el modelo aprenda a
    detectar el círculo (artefacto de adquisición) en lugar de la lesión.

    Estrategia: umbral de brillo (>10 = contenido) → bounding box del área
    no negra → recorte. Si no hay bordes negros (HAM10000, MSK) la imagen
    se devuelve sin modificar — operación idempotente.
    """
    img_np  = np.array(img_pil)
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords  = cv2.findNonZero(mask)
    if coords is None:
        return img_pil
    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = img_np.shape[:2]
    if w >= 0.95 * img_w and h >= 0.95 * img_h:
        return img_pil
    return Image.fromarray(img_np[y:y + h, x:x + w])


# ── Experto 1 — ISIC 2019 ──────────────────────────────────────────────────
# H1 → multiclase: CrossEntropyLoss, 9 neuronas softmax, pesos de clase
# H2 → split por lesion_id sin leakage (build_lesion_split)
# H3 → bias 3 fuentes: _source_audit() + apply_circular_crop() para BCN_20000
# H4 → deduplicación MD5: elimina duplicados silenciosos ISIC 2018/2019
# H5 → UNK: validación todo-ceros en train, slot 8 para OOD en inferencia
# H6 → augmentation diferenciado: estándar vs agresivo (Gessert et al.) por clase
class ISICDataset(Dataset):
    """
    ISIC 2019 — Multiclase, 9 clases (8 en train + UNK solo en test).

    Hallazgos implementados:
      H1 — Multiclase: CrossEntropyLoss, 9 neuronas softmax, class_weights.
           NUNCA usar Accuracy global. Métrica oficial: BMCA.
      H2 — Split por lesion_id sin leakage (build_lesion_split).
      H3 — Bias 3 fuentes: _source_audit() + apply_circular_crop() BCN_20000.
      H4 — Deduplicación MD5 antes del split (ISIC 2019 ⊇ ISIC 2018).
      H5 — UNK validado como todo-ceros en train. Slot 8 en softmax para OOD.
      H6 — Augmentation diferenciado por clase (estándar vs minoría).

    Dos modos:
      mode="embedding" → FASE 0: (img, expert_id=1, img_name)
      mode="expert"    → FASE 2: (img, class_label_int, img_name)
    """

    CLASSES       = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    N_CLASSES     = 9
    N_TRAIN_CLS   = 8
    MINORITY_IDX  = {3, 5, 6, 7}       # AK=3, DF=5, VASC=6, SCC=7
    MINORITY_NAMES= {"AK", "DF", "VASC", "SCC"}
    KNOWN_DUPLICATES = {"ISIC_0067980", "ISIC_0069013"}

    @staticmethod
    def build_isic_transforms(img_size=224):
        """
        H6 — Tres pipelines de transform:
          "embedding" : sin augmentation (FASE 0).
          "standard"  : augmentation moderado para clases mayoritarias (FASE 2).
          "minority"  : augmentation agresivo Gessert et al. 2020 para
                        DF(~239)/VASC(~253)/SCC(~628)/AK(~867) (FASE 2).
        """
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        embedding_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        standard_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])

        # Gessert et al. 2020 — ganador ISIC 2019 Challenge
        minority_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation((90,  90))],  p=0.33),
            transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.33),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])

        return {"embedding": embedding_tf,
                "standard":  standard_tf,
                "minority":  minority_tf}

    @staticmethod
    def get_weighted_sampler(split_df, classes):
        """
        H6 — WeightedRandomSampler para compensar desbalance NV>>DF/VASC en FASE 2.
        Uso: DataLoader(dataset, sampler=ISICDataset.get_weighted_sampler(train_df, ISICDataset.CLASSES))
        """
        from torch.utils.data import WeightedRandomSampler
        labels  = split_df[classes].values.argmax(axis=1)
        counts  = np.bincount(labels, minlength=len(classes)).astype(float)
        counts  = np.maximum(counts, 1)
        w_cls   = 1.0 / counts
        w_samp  = w_cls[labels]
        return WeightedRandomSampler(
            weights=torch.tensor(w_samp, dtype=torch.float32),
            num_samples=len(w_samp), replacement=True)

    def __init__(self, img_dir, split_df, mode="embedding",
                 transform_standard=None, transform_minority=None,
                 apply_bcn_crop=True):
        """
        Args:
            img_dir            : directorio con los .jpg de ISIC
            split_df           : DataFrame de build_lesion_split()
            mode               : "embedding" (FASE 0) o "expert" (FASE 2)
            transform_standard : transform mayoría en FASE 2 (None → build_isic_transforms)
            transform_minority : transform minoría en FASE 2 (None → build_isic_transforms)
            apply_bcn_crop     : H3/Item-8 — circular crop para BCN_20000 (default True)
        """
        assert mode in ("embedding", "expert"), \
            f"[ISIC] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"

        self.img_dir        = Path(img_dir)
        self.expert_id      = EXPERT_IDS["isic"]
        self.mode           = mode
        self.apply_bcn_crop = apply_bcn_crop
        self.df             = split_df.reset_index(drop=True)
        self.class_weights  = None

        tfs = self.build_isic_transforms()
        if mode == "embedding":
            self.transform = tfs["embedding"]
        else:
            self.transform_standard = transform_standard or tfs["standard"]
            self.transform_minority = transform_minority or tfs["minority"]
            self.transform = self.transform_standard

        log.info(f"[ISIC] Dataset: {len(self.df):,} imgs | mode='{mode}' | "
                 f"apply_bcn_crop={apply_bcn_crop}")

        # H1 — verificar one-hot
        if all(c in self.df.columns for c in self.CLASSES):
            bad = (self.df[self.CLASSES].sum(axis=1) != 1).sum()
            if bad:
                log.error(f"[ISIC] H1 — {bad} filas sin exactamente un 1.0. "
                          f"ISIC es MULTICLASE (no multi-label). row_sum=0=UNK; row_sum>1=error CSV.")
            else:
                log.debug("[ISIC] H1 — Ground truth one-hot ✓")

        # H5 — verificar UNK = todo-ceros en train
        if "UNK" in self.df.columns:
            n_unk = (self.df["UNK"] != 0).sum()
            if n_unk:
                log.warning(f"[ISIC] H5 — {n_unk} filas con UNK≠0. "
                            f"UNK solo debería estar en el test set oficial.")
            else:
                log.info("[ISIC] H5 — UNK=0 en train ✓ | "
                         "Slot 8 (softmax) disponible para OOD en inferencia → Experto 5")

        # H3/Item-6 — detectar _downsampled y verificar en disco
        ds_names = self.df["image"][self.df["image"].str.endswith("_downsampled")].tolist()
        if ds_names:
            pct = 100 * len(ds_names) / len(self.df)
            log.warning(f"[ISIC] H3/Item-6 — {len(ds_names):,} ({pct:.1f}%) imgs _downsampled (MSK). "
                        f"Verificando en disco...")
            missing_ds = [n for n in ds_names if not (self.img_dir / f"{n}.jpg").exists()]
            if missing_ds:
                log.error(f"[ISIC] Item-6 — {len(missing_ds)} archivos _downsampled NO en disco. "
                          f"Primeros 5: {missing_ds[:5]}")
            else:
                log.info(f"[ISIC] Item-6 — Todos los _downsampled verificados en disco ✓")
        else:
            log.debug("[ISIC] Item-6 — Sin imágenes _downsampled en este split.")

        # H3/Item-7 — source audit (HAM/BCN/MSK)
        self._source_audit()

        # H6/Item-9 — preparar modo expert
        if mode == "expert":
            log.info("[ISIC] H1  — Loss: CrossEntropyLoss con pesos (NO BCEWithLogitsLoss).")
            log.info("[ISIC] H1  — 9 neuronas softmax. Metrica: BMCA (no Accuracy global).")
            log.info("[ISIC] H6/Item-9 — Augmentation diferenciado:\n"
                     "    Mayoría: flip H, rot±30°, ColorJitter moderado\n"
                     "    Minoría (DF/VASC/SCC/AK): flip H+V, rot 0/90/180/270°, "
                     "ColorJitter agresivo, RandomAffine, RandomErasing (Cutout)\n"
                     "    Usar get_weighted_sampler() en DataLoader de FASE 2.")

            self.df = self.df.copy()
            self.df["class_label"] = self.df[self.CLASSES].values.argmax(axis=1)
            self.df["is_minority"]  = self.df["class_label"].isin(self.MINORITY_IDX)

            dist  = self.df["class_label"].value_counts().sort_index()
            total = len(self.df)
            log.info("[ISIC] Distribución de clases:")
            for i, cls in enumerate(self.CLASSES):
                count = dist.get(i, 0)
                bar   = "█" * max(1, int(30 * count / total))
                tag   = " ⚠ MINORÍA" if i in self.MINORITY_IDX else ""
                log.info(f"    {cls:<5}(idx {i}): {count:>6,} ({100*count/total:.1f}%) {bar}{tag}")

            counts  = np.array([dist.get(i, 1) for i in range(self.N_TRAIN_CLS)], dtype=float)
            weights = total / (self.N_TRAIN_CLS * counts)
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            log.info(f"[ISIC] H1 — class_weights: "
                     f"min={self.class_weights.min():.2f}({self.CLASSES[self.class_weights.argmin()]}) | "
                     f"max={self.class_weights.max():.2f}({self.CLASSES[self.class_weights.argmax()]})")

            all_zero = (self.df[self.CLASSES].sum(axis=1) == 0).sum()
            if all_zero:
                log.warning(f"[ISIC] {all_zero} filas sum=0 → argmax asignará clase 0 (MEL).")

    def _source_audit(self):
        """H3/Item-7 — Infiere fuente por naming e imprime estadísticas de bias de dominio."""
        names    = self.df["image"]
        msk_mask = names.str.endswith("_downsampled")
        try:
            numeric_ids = names.str.extract(r"ISIC_(\d+)")[0].astype(float)
            ham_mask    = (~msk_mask) & (numeric_ids <= 67977)
            bcn_mask    = (~msk_mask) & (numeric_ids > 67977)
        except Exception:
            ham_mask = pd.Series([False] * len(names), index=names.index)
            bcn_mask = ~msk_mask & ~ham_mask

        n_ham, n_bcn, n_msk = ham_mask.sum(), bcn_mask.sum(), msk_mask.sum()
        total = len(names)
        log.info(
            f"[ISIC] H3/Item-7 — Auditoría de fuentes (heurística por ID range):\n"
            f"    HAM10000 (Viena, 600×450, recortado)      : {n_ham:>5,} ({100*n_ham/total:.1f}%)\n"
            f"    BCN_20000 (Barcelona, 1024×1024, dermoscop): {n_bcn:>5,} ({100*n_bcn/total:.1f}%)\n"
            f"    MSK (NYC, variable, _downsampled)          : {n_msk:>5,} ({100*n_msk/total:.1f}%)\n"
            f"    Contramedida activa: apply_circular_crop() (BCN) + ColorJitter (todas).\n"
            f"    ⚠ Para inspección visual: samplea 5-10 imgs de cada fuente antes de FASE 2."
        )

    @staticmethod
    def build_lesion_split(gt_csv, img_dir=None, metadata_csv=None,
                           frac_train=0.8, random_state=42):
        """
        H2 + H4 — Split por lesion_id + deduplicación MD5.

        Pasos:
          1. Eliminar KNOWN_DUPLICATES por ID.
          2. H4: Si img_dir → calcular hash MD5 y eliminar duplicados exactos.
          3. H2: Split por lesion_id. Sin lesion_id → ID único por imagen.
          4. Verificar leakage (cero lesion_id compartidos).
          5. H5: Verificar UNK=0 en ambos splits.

        Args:
            gt_csv       : ISIC_2019_Training_GroundTruth.csv
            img_dir      : carpeta .jpg para hashing MD5 (None = omitir H4)
            metadata_csv : ISIC_2019_Training_Metadata.csv (None = split aleatorio)
            frac_train   : fracción de lesiones para train
            random_state : semilla reproducibilidad
        """
        import hashlib

        df_gt = pd.read_csv(gt_csv)
        log.info(f"[ISIC/split] GT CSV: {len(df_gt):,} filas")

        # H4: eliminar duplicados conocidos por ID
        before = len(df_gt)
        df_gt  = df_gt[~df_gt["image"].isin(ISICDataset.KNOWN_DUPLICATES)].reset_index(drop=True)
        if before - len(df_gt):
            log.info(f"[ISIC/split] H4 — {before-len(df_gt)} KNOWN_DUPLICATES eliminados.")

        # H4: deduplicación por hash MD5 de archivos reales
        if img_dir is not None:
            img_path_obj = Path(img_dir)
            log.info(f"[ISIC/split] H4 — Calculando hashes MD5 de {len(df_gt):,} imgs "
                     f"(puede tardar ~1-2 min)...")
            hashes, dup_ids, missing = {}, set(), 0
            for _, row in df_gt.iterrows():
                fpath = img_path_obj / f"{row['image']}.jpg"
                if not fpath.exists():
                    missing += 1
                    continue
                try:
                    md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
                    if md5 in hashes:
                        dup_ids.add(row["image"])
                        log.debug(f"[ISIC/split] H4 — Dup: '{row['image']}' == '{hashes[md5]}'")
                    else:
                        hashes[md5] = row["image"]
                except Exception as e:
                    log.warning(f"[ISIC/split] H4 — No se pudo hashear '{fpath}': {e}")
            if missing:
                log.warning(f"[ISIC/split] H4 — {missing} archivos no encontrados para hashing.")
            if dup_ids:
                df_gt = df_gt[~df_gt["image"].isin(dup_ids)].reset_index(drop=True)
                log.info(f"[ISIC/split] H4 — {len(dup_ids)} duplicados MD5 eliminados. "
                         f"Quedan {len(df_gt):,} imgs únicas.")
            else:
                log.info("[ISIC/split] H4 — Sin duplicados MD5 detectados ✓")
        else:
            log.warning("[ISIC/split] H4 — img_dir=None: dedup MD5 OMITIDA. "
                        "Pasa --isic_imgs a build_lesion_split para deduplicación completa.")

        # H2: split por lesion_id
        if metadata_csv is not None:
            df_meta = pd.read_csv(metadata_csv, usecols=["image", "lesion_id"])
            df_gt   = df_gt.merge(df_meta, on="image", how="left")
            n_with  = df_gt["lesion_id"].notna().sum()
            n_wo    = df_gt["lesion_id"].isna().sum()
            log.info(f"[ISIC/split] H2 — lesion_id: {n_with:,} con ID | {n_wo:,} sin ID (→ únicos)")

            mask_na = df_gt["lesion_id"].isna()
            df_gt.loc[mask_na, "lesion_id"] = df_gt.loc[mask_na, "image"].apply(
                lambda x: f"_unique_{x}")

            unique_lesions = df_gt["lesion_id"].unique()
            rng = np.random.RandomState(random_state)
            rng.shuffle(unique_lesions)
            n_tr          = int(len(unique_lesions) * frac_train)
            train_lesions = set(unique_lesions[:n_tr])
            val_lesions   = set(unique_lesions[n_tr:])

            train_df = df_gt[df_gt["lesion_id"].isin(train_lesions)].copy()
            val_df   = df_gt[df_gt["lesion_id"].isin(val_lesions)].copy()

            shared = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
            if shared:
                log.error(f"[ISIC/split] H2 — ¡LEAKAGE! {len(shared)} lesion_ids compartidos.")
            else:
                log.info(f"[ISIC/split] H2 — Split por lesion_id ✓ | "
                         f"train: {len(train_lesions):,} lesiones ({len(train_df):,} imgs) | "
                         f"val: {len(val_lesions):,} lesiones ({len(val_df):,} imgs)")
        else:
            log.warning("[ISIC/split] H2 — metadata_csv=None: split aleatorio por imagen. "
                        "Pasa --isic_metadata para split correcto por lesion_id.")
            idx_all = np.arange(len(df_gt))
            rng     = np.random.RandomState(random_state)
            rng.shuffle(idx_all)
            n_tr     = int(len(idx_all) * frac_train)
            train_df = df_gt.iloc[idx_all[:n_tr]].copy()
            val_df   = df_gt.iloc[idx_all[n_tr:]].copy()

        # H5: verificar UNK=0 en ambos splits
        for sname, sdf in [("train", train_df), ("val", val_df)]:
            if "UNK" in sdf.columns:
                n_unk = (sdf["UNK"] != 0).sum()
                if n_unk:
                    log.warning(f"[ISIC/split] H5 — {n_unk} filas UNK≠0 en '{sname}'. "
                                f"UNK solo debería estar en el test set oficial.")
                else:
                    log.debug(f"[ISIC/split] H5 — UNK=0 en '{sname}' ✓")

        log.info(f"[ISIC/split] Final — train: {len(train_df):,} | val: {len(val_df):,}")
        return train_df, val_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        img_path = self.img_dir / f"{img_name}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[ISIC] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero.")
            dummy = torch.zeros(3, 224, 224)
            if self.mode == "embedding":
                return dummy, self.expert_id, img_name
            label = int(self.df.loc[idx, "class_label"]) if "class_label" in self.df.columns else 0
            return dummy, label, img_name

        # H3/Item-8 — eliminar bordes negros BCN_20000 (idempotente para HAM/MSK)
        if self.apply_bcn_crop:
            img = apply_circular_crop(img)

        if self.mode == "embedding":
            return self.transform(img), self.expert_id, img_name
        else:
            # H6/Item-9 — augmentation diferenciado por clase
            is_minority = bool(self.df.loc[idx, "is_minority"])
            tf  = self.transform_minority if is_minority else self.transform_standard
            return tf(img), int(self.df.loc[idx, "class_label"]), img_name





# ── Experto 2 — Osteoarthritis Knee ────────────────────────────────────────
# Cargador con auditoría de integridad completa. Implementa los hallazgos:
#   H1 → clases ordinales: CrossEntropyLoss con pesos + QWK como métrica;
#         OrdinalLoss disponible como alternativa (Chen et al., 2019);
#         pesos por clase calculados automáticamente para el desbalance;
#         NO usar RandomVerticalFlip — anatómicamente incorrecto.
#   H2 → augmentation offline en el ZIP: detección por hash MD5 de muestra;
#         warning si el dataset es ~54% más grande que el base OAI (8,260 imgs);
#         recomendación de NO aplicar aug en runtime sobre imágenes ya augmentadas.
#   H3 → sin metadatos de paciente: split por carpeta (OAI predefinido en Kaggle);
#         documentado como limitación del proyecto; verificación de canales RGB.

# Constantes globales de OA — usadas por OAKneeDataset y exportadas para FASE 2
OA_CLASS_NAMES   = ["Normal (KL0)", "Leve (KL1-2)", "Severo (KL3-4)"]
OA_N_CLASSES     = 3
# Matriz de costos ordinales: cost[i][j] = |i - j|² (penaliza distancias grandes)
# Confundir Normal(0) con Severo(2) → costo 4; con Leve(1) → costo 1
OA_COST_MATRIX   = np.array([[0, 1, 4],
                              [1, 0, 1],
                              [4, 1, 0]], dtype=np.float32)
# Imágenes esperadas en la versión base OAI (sin augmentation offline)
OA_BASE_IMG_COUNT = 8260


class OrdinalLoss(nn.Module):
    """
    H1 — OrdinalLoss (Chen et al., 2019 — PMC doi:10.1016/j.compmedimag.2019.05.007).

    Reformula la clasificación ordinal como K-1 clasificaciones binarias
    acumulativas: P(y > k) para k = 0, ..., K-2.

    Para K=3 clases (Normal/Leve/Severo):
      P(y > 0) = P(Leve o Severo)
      P(y > 1) = P(Severo)

    La loss es BCEWithLogitsLoss sumada sobre los K-1 umbrales.

    Ventaja sobre CrossEntropyLoss: penaliza confundir Normal↔Severo más
    que Normal↔Leve, porque la predicción incorrecta en el umbral 0 ya
    está mal, y en el umbral 1 también.

    Uso en FASE 2 (opción B — ver H1 en reporte técnico):
        criterion = OrdinalLoss(n_classes=3)
        # model output: [B, K-1] logits (NO softmax)
        loss = criterion(logits, labels)   # labels ∈ {0, 1, 2}

    Para CrossEntropyLoss estándar con pesos (opción A — pragmática):
        criterion = nn.CrossEntropyLoss(weight=dataset.class_weights)
        # model output: [B, K] logits
        loss = criterion(logits, labels)
    """
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, K-1] — logits de los K-1 umbrales binarios
        targets : [B]      — etiquetas ordinales enteras {0, ..., K-1}
        """
        K = self.n_classes
        # Construir targets binarios: para target=k, P(y>j)=1 si j<k, 0 si j>=k
        # Ejemplo K=3: target=2 → [1, 1]; target=1 → [1, 0]; target=0 → [0, 0]
        bin_targets = torch.zeros(targets.size(0), K - 1, device=logits.device)
        for k in range(K - 1):
            bin_targets[:, k] = (targets > k).float()
        return self.bce(logits, bin_targets)


class OAKneeDataset(Dataset):
    """
    Osteoarthritis Knee — Ordinal, 3 clases (0=Normal, 1=Leve, 2=Severo).

    Hallazgos implementados:
      H1 — Ordinalidad: CrossEntropyLoss con pesos por clase (opción pragmática);
           OrdinalLoss disponible como alternativa (Chen et al., 2019);
           QWK (cohen_kappa_score, weights='quadratic') como métrica principal;
           pesos de clase calculados en modo expert;
           ⚠ NUNCA usar RandomVerticalFlip (anatómicamente incorrecto para rodilla).
      H2 — Augmentation offline: detección por hash MD5 de muestra;
           warning si total imágenes > OA_BASE_IMG_COUNT × 1.1;
           decisión documentada en logs (usar splits del ZIP tal cual vs re-aug).
      H3 — Sin metadatos: split por carpeta predefinida (Kaggle/OAI);
           documentado como limitación — no verificable separación por paciente;
           verificación de canales RGB (¿grises duplicados o genuinamente RGB?).
      H4 — CLAHE ANTES del resize (crítico para el espacio articular):
           apply_clahe() opera sobre la resolución original de la imagen.
           Las radiografías OA son grises guardadas como RGB (3 canales idénticos).
           El pipeline correcto es: cargar → CLAHE → resize → normalize.
           Aplicar CLAHE después del resize reduciría el espacio articular a
           pocos píxeles antes de realzar, limitando severamente el efecto.
      H5 — KL1 contamina la clase "Leve" con alta ambigüedad inter-observador:
           La consolidación KL1+KL2 → Clase 1 (Leve) mezcla casos con certeza
           diagnóstica (KL2: osteofitos definitivos) con casos donde el ground
           truth original es incierto (KL1: hallazgos dudosos). La frontera
           Clase0↔Clase1 será sistemáticamente la más difícil de aprender.
           evaluate_boundary_confusion() la monitorea en FASE 2.

    Dos modos:
      mode="embedding" → FASE 0: (img, expert_id=2, img_name) — CLAHE aplicado
      mode="expert"    → FASE 2: (img, kl_label_int, img_name) — aug diferenciado
    """

    @staticmethod
    def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        QWK pendiente — Función callable lista para el loop de entrenamiento de FASE 2.

        Quadratic Weighted Kappa — métrica principal del Experto 2.
        Penaliza confusiones proporcionalmente al cuadrado de la distancia ordinal:
          confundir Normal↔Severo (distancia=2, costo=4) penaliza 4× más que
          confundir Normal↔Leve   (distancia=1, costo=1).

        Uso en loop de validación de FASE 2:
            y_pred_np = logits.argmax(dim=1).cpu().numpy()
            y_true_np = labels.cpu().numpy()
            qwk = OAKneeDataset.compute_qwk(y_true_np, y_pred_np)
            log.info(f"Época {e} | QWK: {qwk:.4f}")

        Returns:
            float en [-1, 1] — interpretación:
              < 0.20  : sin acuerdo útil
              0.20-0.40: acuerdo débil
              0.40-0.60: acuerdo moderado (mínimo aceptable)
              0.60-0.80: acuerdo sustancial (objetivo del proyecto)
              > 0.80  : acuerdo casi perfecto (benchmark literatura OAI)
        """
        from sklearn.metrics import cohen_kappa_score
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        # labels forzado para que funcione aunque no aparezcan todas las clases en el batch
        return float(cohen_kappa_score(y_true, y_pred,
                                       weights="quadratic",
                                       labels=[0, 1, 2]))

    @staticmethod
    def evaluate_boundary_confusion(y_true: np.ndarray,
                                    y_pred: np.ndarray) -> dict:
        """
        H5/Item-9 — Analiza la matriz de confusión con foco en la frontera
        Clase0↔Clase1 (Normal↔Leve), que es sistemáticamente la más difícil
        por la contaminación de KL1 en la clase "Leve".

        Uso en el loop de validación de FASE 2:
            metrics = OAKneeDataset.evaluate_boundary_confusion(y_val, y_pred)
            log.info(f"QWK: {metrics['qwk']:.4f} | "
                     f"Error frontera 0↔1: {metrics['boundary_01_error_rate']:.3f}")

        Returns: dict con:
            qwk                   : Quadratic Weighted Kappa (métrica principal)
            confusion_matrix      : np.ndarray [3, 3]
            boundary_01_error_rate: fracción de Clase0 mal clasificada como Clase1
                                    + fracción de Clase1 mal clasificada como Clase0
            boundary_12_error_rate: fracción de Clase1↔Clase2 confundidas (esperado menor)
            severe_misclass_rate  : fracción de Clase2 clasificada como Clase0 (peor error)
            per_class_recall      : recall por clase {0: r0, 1: r1, 2: r2}
        """
        from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_cm

        cm  = sk_cm(y_true, y_pred, labels=[0, 1, 2])
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        # Frontera 0↔1: errores en la zona ambigua KL1
        n0 = cm[0].sum()   # total real Clase 0
        n1 = cm[1].sum()   # total real Clase 1
        n2 = cm[2].sum()   # total real Clase 2

        b01 = (cm[0, 1] / max(n0, 1) + cm[1, 0] / max(n1, 1)) / 2   # tasa de error promedio
        b12 = (cm[1, 2] / max(n1, 1) + cm[2, 1] / max(n2, 1)) / 2
        severe_miss = cm[2, 0] / max(n2, 1)   # Severo predicho como Normal

        recall = {i: cm[i, i] / max(cm[i].sum(), 1) for i in range(3)}

        return {
            "qwk":                   float(qwk),
            "confusion_matrix":      cm,
            "boundary_01_error_rate": float(b01),
            "boundary_12_error_rate": float(b12),
            "severe_misclass_rate":  float(severe_miss),
            "per_class_recall":      {i: float(r) for i, r in recall.items()},
        }

    @staticmethod
    def log_boundary_confusion(metrics: dict, epoch: int = -1) -> None:
        """
        H5/Item-9 — Imprime en el log un resumen de la matriz de confusión
        con énfasis en la frontera Clase0↔Clase1.

        Uso en el loop de validación de FASE 2:
            metrics = OAKneeDataset.evaluate_boundary_confusion(y_val, y_pred)
            OAKneeDataset.log_boundary_confusion(metrics, epoch=epoch)
        """
        ep_str = f"Época {epoch} — " if epoch >= 0 else ""
        cm     = metrics["confusion_matrix"]
        log.info(
            f"[OA] {ep_str}Evaluación Experto 2:\n"
            f"    QWK (métrica principal) : {metrics['qwk']:.4f}\n"
            f"    Recall por clase:\n"
            f"      Clase 0 Normal : {metrics['per_class_recall'][0]:.3f}\n"
            f"      Clase 1 Leve   : {metrics['per_class_recall'][1]:.3f}\n"
            f"      Clase 2 Severo : {metrics['per_class_recall'][2]:.3f}\n"
            f"    Matriz de confusión (filas=real, cols=pred):\n"
            f"      {cm[0]}\n"
            f"      {cm[1]}\n"
            f"      {cm[2]}\n"
            f"    H5 — Frontera 0↔1 (KL1 ambiguo): "
            f"error={metrics['boundary_01_error_rate']:.3f} "
            f"{'⚠ ALTA' if metrics['boundary_01_error_rate'] > 0.25 else '✓ aceptable'}\n"
            f"    Frontera 1↔2: error={metrics['boundary_12_error_rate']:.3f}\n"
            f"    Peor error: Severo→Normal = {metrics['severe_misclass_rate']:.3f} "
            f"{'⚠⚠ CRÍTICO (costo ordinal=4)' if metrics['severe_misclass_rate'] > 0.05 else '✓'}"
        )

    def __init__(self, root_dir, split="train", img_size=224, mode="embedding"):
        """
        Args:
            root_dir : raíz con subcarpetas train/val/test, cada una con 0/, 1/, 2/
            split    : "train", "val" o "test"
            img_size : tamaño de resize (default 224)
            mode     : "embedding" (FASE 0) o "expert" (FASE 2)
        """
        assert mode in ("embedding", "expert"), \
            f"[OA] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"

        self.expert_id = EXPERT_IDS["oa"]
        self.img_size  = img_size
        self.mode      = mode
        self.samples   = []

        split_dir = Path(root_dir) / split
        if not split_dir.exists():
            log.error(f"[OA] Directorio de split no encontrado: {split_dir}")
            return

        # ── Leer clases desde carpetas (os.listdir equivalente) ──────────────
        class_counts  = {}
        found_classes = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                class_id = int(class_dir.name)
            except ValueError:
                log.warning(f"[OA] Carpeta con nombre no numérico ignorada: '{class_dir.name}'. "
                            f"Se esperan carpetas '0', '1', '2' (o '0'–'4' si son 5 grados KL).")
                continue
            imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_path in imgs:
                self.samples.append((img_path, class_id))
            class_counts[class_dir.name] = len(imgs)
            found_classes.append(class_dir.name)

        n_found = len(found_classes)
        log.info(
            f"[OA] split='{split}' mode='{mode}': {len(self.samples):,} imágenes | "
            f"clases: {found_classes}"
        )

        if len(self.samples) == 0:
            log.error(f"[OA] Dataset vacío en '{split_dir}'. "
                      f"Verifica subcarpetas 0/, 1/, 2/ con imágenes.")
            return

        # ── Item-1: verificar número de clases (3 consolidadas o 5 originales) ─
        if n_found == 5:
            log.warning(
                f"[OA] Se encontraron 5 clases (grados KL 0-4 sin consolidar). "
                f"El proyecto usa 3 clases consolidadas: "
                f"KL0→Clase0(Normal), KL1+2→Clase1(Leve), KL3+4→Clase2(Severo). "
                f"Verifica que el root_dir apunta al dataset ya consolidado."
            )
        elif n_found < 3:
            log.warning(
                f"[OA] Solo {n_found} clase(s) encontradas — se esperaban 3. "
                f"¿El ZIP está completo?"
            )
        else:
            log.debug(f"[OA] 3 clases ordinales confirmadas ✓")

        # ── Item-2: distribución por clase + pesos para CrossEntropyLoss ──────
        log.info(f"[OA] Distribución por clase:")
        total = len(self.samples)
        counts_arr = np.zeros(OA_N_CLASSES, dtype=float)
        for k, v in class_counts.items():
            try:
                idx = int(k)
                if 0 <= idx < OA_N_CLASSES:
                    counts_arr[idx] = v
            except ValueError:
                pass

        for i, (name, count) in enumerate(zip(OA_CLASS_NAMES, counts_arr)):
            bar = "█" * max(1, int(30 * count / total))
            log.info(f"    Clase {i} ({name:<18}): {int(count):>5,} ({100*count/total:.1f}%) {bar}")

        # Pesos de clase para CrossEntropyLoss (opción pragmática H1)
        counts_safe = np.maximum(counts_arr, 1)
        cw = torch.tensor(total / (OA_N_CLASSES * counts_safe), dtype=torch.float32)
        self.class_weights = cw
        log.info(
            f"[OA] H1 — class_weights CrossEntropyLoss: "
            + " | ".join(f"Clase{i}:{cw[i]:.2f}" for i in range(OA_N_CLASSES))
        )
        log.info(
            "[OA] H1 — Opciones de loss para FASE 2:\n"
            "    Opción A (pragmática): CrossEntropyLoss(weight=dataset.class_weights)\n"
            "      → salida: [B, 3] logits\n"
            "    Opción B (ordinal): OrdinalLoss(n_classes=3)\n"
            "      → salida: [B, 2] logits (un umbral binario por par adyacente)\n"
            "    Documenta la elección en el Reporte Técnico (sección 3).\n"
            "    Métrica principal: QWK (cohen_kappa_score, weights='quadratic')."
        )

        # ── H1: advertir sobre RandomVerticalFlip ─────────────────────────────
        log.info(
            "[OA] H1 — ⚠ Augmentation: NUNCA usar RandomVerticalFlip. "
            "Una rodilla con tibia arriba es anatómicamente imposible y el modelo "
            "aprendería una distribución que no existe en inferencia real. "
            "Válidos: RandomHorizontalFlip(0.5), RandomRotation(±10°), "
            "ColorJitter(brightness=0.2)."
        )

        # ── H2: detectar augmentation offline ────────────────────────────────
        self._audit_augmentation_offline(split, total)

        # ── Item-3/H2: verificar canales de imagen (gris guardado como RGB) ──
        self._verify_image_channels()

        # ── Item-8: verificar consistencia de dimensiones ─────────────────────
        self._verify_image_dimensions()

        # ── H3: documentar limitación de split sin Patient ID ─────────────────
        log.info(
            f"[OA] H3 — Sin metadatos de paciente: split '{split}' tomado tal cual "
            f"de las carpetas del ZIP (predefinido por OAI/Kaggle). "
            f"No es posible verificar separación por Patient ID sin datos de "
            f"nda.nih.gov/oai. Documentar como limitación en el Reporte Técnico."
        )

        # ── H4: documentar el orden CLAHE → resize ───────────────────────────
        log.info(
            "[OA] H4 — CLAHE aplicado en apply_clahe() ANTES del resize. "
            "Orden en __getitem__: cargar (resolución original) → CLAHE → "
            f"resize({self.img_size}×{self.img_size}) → normalize ImageNet. "
            "Aplicar CLAHE después del resize reduciría el espacio articular a "
            "pocos píxeles antes del realce — resultado diagnósticamente incorrecto. "
            "Las radiografías OA son grises guardadas como RGB (3 canales idénticos): "
            "apply_clahe() extrae canal L → CLAHE → replica a RGB."
        )

        # ── H5: documentar contaminación KL1 y frontera 0↔1 ──────────────────
        log.info(
            "[OA] H5 — Contaminación KL1 en clase 'Leve' (Clase 1): "
            "KL1 tiene el menor acuerdo inter-observador del sistema KL — "
            "radiólogos expertos discrepan frecuentemente en este grado. "
            "Al consolidar KL1+KL2 → Clase1, se mezclan casos con certeza "
            "(KL2: osteofitos definitivos) con casos inciertos (KL1: hallazgos dudosos). "
            "CONSECUENCIA: la frontera Clase0↔Clase1 será la más difícil de aprender. "
            "Anticipar en el Reporte Técnico como característica del dataset, "
            "NO como fallo del modelo. "
            "HERRAMIENTA: OAKneeDataset.evaluate_boundary_confusion(y_true, y_pred) "
            "y log_boundary_confusion() para monitorear esta frontera en cada época de FASE 2."
        )

        # ── Transform pipeline ────────────────────────────────────────────────
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        # Augmentation para modo expert — respeta las restricciones anatómicas
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip AUSENTE — H1 anatómicamente incorrecto
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _audit_augmentation_offline(self, split: str, total_imgs: int):
        """
        H2 — Detecta si el ZIP contiene augmentation offline comparando
        el conteo total contra el esperado del split base OAI.

        Split base OAI esperado (aprox.):
          train ~5,778 | val ~826 | test ~1,656 | total ~8,260

        Si total_imgs > esperado × 1.1 → probablemente hay aug offline.
        En ese caso: NO aplicar aug adicional en runtime para evitar
        distorsiones anatómicas y sobre-augmentation.
        """
        expected = {"train": 5778, "val": 826, "test": 1656}.get(split, OA_BASE_IMG_COUNT)
        ratio    = total_imgs / max(expected, 1)

        if ratio > 1.10:
            log.warning(
                f"[OA] H2 — El split '{split}' tiene {total_imgs:,} imágenes "
                f"vs {expected:,} esperadas del base OAI (ratio={ratio:.2f}x). "
                f"Esto sugiere augmentation offline en el ZIP. "
                f"DECISIÓN recomendada: usar estas imágenes SIN augmentation adicional "
                f"en runtime para evitar distorsiones anatómicas. "
                f"Si se aplica aug en runtime, excluir RandomVerticalFlip y "
                f"rotaciones > ±15°. Documentar esta decisión en el Reporte Técnico."
            )
            self._aug_offline_detected = True
        else:
            log.info(
                f"[OA] H2 — Conteo de imágenes consistente con base OAI "
                f"({total_imgs:,} vs ~{expected:,}, ratio={ratio:.2f}x). "
                f"Augmentation en runtime es seguro."
            )
            self._aug_offline_detected = False

        # Hash MD5 de muestra aleatoria de 20 imágenes para detectar duplicados
        # (augmentation offline frecuentemente produce duplicados con nombres distintos)
        if len(self.samples) >= 20:
            import hashlib, random
            sample_paths = [p for p, _ in random.sample(self.samples, 20)]
            hashes = {}
            dup_count = 0
            for p in sample_paths:
                try:
                    md5 = hashlib.md5(p.read_bytes()).hexdigest()
                    if md5 in hashes:
                        dup_count += 1
                        log.debug(f"[OA] H2 — Hash dup: '{p.name}' == '{hashes[md5]}'")
                    else:
                        hashes[md5] = p.name
                except Exception as e:
                    log.debug(f"[OA] H2 — No se pudo hashear '{p}': {e}")
            if dup_count:
                log.warning(
                    f"[OA] H2 — {dup_count}/20 imágenes de muestra tienen hash MD5 duplicado. "
                    f"Hay imágenes con píxeles idénticos pero nombres distintos — "
                    f"confirmación de augmentation offline. Aplica aug en runtime con moderación."
                )
            else:
                log.debug("[OA] H2 — Sin duplicados MD5 en la muestra de 20 imágenes.")

    def _verify_image_channels(self):
        """
        Item-4 — Verifica que las imágenes son grises guardadas como RGB.
        OA Knee son radiografías monocromáticas convertidas a 3 canales idénticos.
        Si los 3 canales son idénticos → CLAHE se aplica correctamente en canal gray.
        Si los 3 canales difieren → la imagen es genuinamente RGB (poco probable en OA).
        Muestra 5 imágenes aleatorias para diagnóstico.
        """
        if not self.samples:
            return
        import random
        n_check     = min(5, len(self.samples))
        sample_paths = [p for p, _ in random.sample(self.samples, n_check)]
        n_gray_rgb  = 0
        n_true_rgb  = 0
        for p in sample_paths:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                if np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2) and \
                   np.allclose(arr[:, :, 0], arr[:, :, 2], atol=2):
                    n_gray_rgb += 1
                else:
                    n_true_rgb += 1
            except Exception:
                pass
        if n_true_rgb == 0:
            log.info(
                f"[OA] Item-4 — {n_gray_rgb}/{n_check} imágenes muestreadas son "
                f"grises guardadas como RGB (3 canales idénticos) ✓ "
                f"CLAHE se aplica correctamente."
            )
        else:
            log.warning(
                f"[OA] Item-4 — {n_true_rgb}/{n_check} imágenes muestreadas tienen "
                f"3 canales distintos (genuinamente RGB). Inusual para radiografías OA. "
                f"Verifica que el dataset no está mezclado con imágenes de otra modalidad."
            )

    def _verify_image_dimensions(self):
        """
        Item-8 — Verifica consistencia de dimensiones en una muestra de imágenes.
        Las imágenes del OAI pueden mostrar una o ambas rodillas en la misma toma.
        Si las dimensiones son muy variables → posibles problemas de resize uniforme.
        """
        if not self.samples:
            return
        import random
        n_check = min(10, len(self.samples))
        sizes   = set()
        for p, _ in random.sample(self.samples, n_check):
            try:
                sizes.add(Image.open(p).size)   # (width, height)
            except Exception:
                pass
        if len(sizes) == 1:
            log.debug(f"[OA] Item-8 — Dimensiones uniformes en muestra: {next(iter(sizes))} ✓")
        else:
            log.info(
                f"[OA] Item-8 — Dimensiones variables en muestra: {sizes}. "
                f"Resize a {self.img_size}×{self.img_size} px compensará las diferencias. "
                f"Si hay imágenes muy anchas (ambas rodillas en una toma), "
                f"considerar recortar cada rodilla individualmente antes del resize."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kl_class = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[OA] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero.")
            dummy = torch.zeros(3, self.img_size, self.img_size)
            return dummy, self.expert_id if self.mode == "embedding" else kl_class, str(img_path.name)

        # H4 — CLAHE ANTES del resize (obligatorio — ver apply_clahe() y H4 en docstring).
        # Flujo: imagen a resolución original → CLAHE → resize → normalize.
        # Nunca invertir este orden — el espacio articular debe estar a alta resolución
        # cuando CLAHE lo realza para que el efecto sea clínicamente significativo.
        img = apply_clahe(img)                                     # resolución original
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)  # 224×224

        if self.mode == "embedding":
            # FASE 0: router aprende "soy imagen de OA Rodilla"
            return self.base_transform(img), self.expert_id, str(img_path.name)
        else:
            # FASE 2: Experto 2 aprende el grado KL con loss ordinal o CE
            # Augmentation en runtime SOLO si no hay aug offline detectado,
            # o de forma muy conservadora si lo hay (ver H2)
            if self.mode == "expert" and not getattr(self, "_aug_offline_detected", False):
                img_tensor = self.aug_transform(img)
            else:
                img_tensor = self.base_transform(img)
            return img_tensor, kl_class, str(img_path.name)



# ── Experto 3 — LUNA16 / LIDC-IDRI ────────────────────────────────────────
# Cargador + utilitarios para clasificación de PARCHES 3D (track 2 del challenge).
# Implementa los hallazgos:
#   H1 → tarea: parches 64×64×64, NO volúmenes completos; tensor [B,1,64,64,64]
#   H2 → desbalance ~490:1: FocalLoss implementada; submuestreo como alternativa
#   H3 → conversión world→vóxel correcta: LUNA16PatchExtractor con validación
#         de al menos 10 candidatos class=1 antes de producción
#   Item-2  → verificación de candidates_V2.csv vs V1 obsoleto
#   Item-4  → normalización HU clip[-1000,400] implementada y documentada
#   Item-5  → verificación de ratio class=1/class=0 en __init__
#   Item-7  → gradient checkpointing documentado (obligatorio con 12 GB VRAM)
#   Item-8  → configuración FROC documentada; techo teórico 94.4% explicado
#   Item-9  → verificación de spacing variable en muestra de volúmenes



# ── Constantes HU para LUNA16 ─────────────────────────────────────────────
# H4 — El rango [-1000, 400] HU no es arbitrario: cada valor tiene justificación clínica.
#
# Tejido pulmonar          : -900 a -500 HU (aire + tejido alveolar)
# Nódulos sólidos          : -100 a +100 HU (mismo rango que tejido blando)
# Nódulos part-sólidos     :  -700 a +100 HU (combinan ambos)
# Pared torácica / músculo :  +30 a +80 HU
# Hueso                    :  +400 a +1800 HU  ← EXCLUIDO
# Metales / artefactos     : >+1800 HU         ← EXCLUIDO
#
# Por qué se excluye el hueso (>400 HU):
#   El hueso domina la dinámica de la red si no se elimina. Las capas iniciales
#   gastarían capacidad representacional en distinguir tipos de hueso
#   en lugar de tipos de tejido pulmonar. El clip a 400 HU hace que toda
#   densidad ≥400 HU colapse al mismo valor, neutralizando esa información.
#
# Por qué -1000 HU como mínimo (no un valor más alto):
#   -1000 HU es la densidad del aire puro. Preservarlo asegura que el fondo
#   del parche (cavidades de aire alrededor del pulmón) tenga siempre el
#   mismo valor absoluto = 0.0 tras la normalización → el padding en bordes
#   es consistente con el fondo real de la imagen.
HU_LUNG_CLIP = (-1000, 400)   # (min_hu, max_hu) — no cambiar sin justificación clínica


def verify_hu_normalization(patches_dir: str, n_sample: int = 3) -> None:
    """
    H4/Item-4 — Verifica visualmente que los parches están correctamente normalizados.

    Muestra estadísticas de intensidad en n parches aleatorios.
    Un parche bien normalizado tiene:
      - min ≈ 0.0 (aire = -1000 HU tras el clip)
      - max ≈ 1.0 (tejido muy denso = 400 HU)
      - mean entre 0.0 y 0.3 (predomina el aire en el contexto pulmonar)

    Si mean > 0.5 → el clip HU puede estar mal aplicado (rango incorrecto).
    Si min < 0 o max > 1 → el parche NO está normalizado (está en HU crudo).

    Uso antes de FASE 0:
        verify_hu_normalization("./embeddings/luna_patches/train")
    """
    import random
    patches = list(Path(patches_dir).glob("*.npy"))
    if not patches:
        log.warning(f"[LUNA16/verify_hu] Sin parches .npy en '{patches_dir}'")
        return
    sample = random.sample(patches, min(n_sample, len(patches)))
    log.info(f"[LUNA16/verify_hu] H4 — Verificando normalización HU en {len(sample)} parches:")
    for p in sample:
        arr = np.load(p)
        log.info(
            f"    {p.name}: min={arr.min():.4f} | max={arr.max():.4f} | "
            f"mean={arr.mean():.4f} | shape={arr.shape}"
        )
        if arr.min() < -0.01:
            log.error(f"    ⚠ min < 0 → parche probablemente en HU crudo (no normalizado). "
                      f"Aplica HU_LUNG_CLIP=(-1000,400) y normaliza a [0,1].")
        if arr.max() > 1.01:
            log.error(f"    ⚠ max > 1 → parche en HU crudo. "
                      f"Normalización: (HU - (-1000)) / (400 - (-1000)).")
        if arr.mean() > 0.5:
            log.warning(f"    ⚠ mean > 0.5 → densidad inusualmente alta. "
                        f"¿El clip HU usa el rango correcto {HU_LUNG_CLIP}?")


# ── LUNA16 FROC Evaluator ──────────────────────────────────────────────────

class LUNA16FROCEvaluator:
    """
    H6 — Interfaz para calcular CPM y curva FROC oficial de LUNA16.

    AUC-ROC NO es la métrica correcta para LUNA16:
      - Mide a nivel de candidato individual, no de nódulo
      - No captura que el mismo nódulo puede tener múltiples candidatos cercanos
      - No refleja el costo clínico de los FP (que se mide por escaneo, no por candidato)

    CPM (Competition Performance Metric) — métrica oficial:
      Promedio de sensitividad en los 7 puntos FP/scan del challenge:
        FP_THRESHOLDS = {1/8, 1/4, 1/2, 1, 2, 4, 8}
      Punto clínicamente más relevante: sensitividad @ 1 FP/scan.
      Un radiólogo puede revisar ~1 FP/scan antes de que el sistema pierda utilidad.

    Referencia: Setio et al., Medical Image Analysis 2017 (arXiv:1612.08012)

    Uso:
        evaluator = LUNA16FROCEvaluator(
            seriesuid_list   = ["uid1", "uid2", ...],
            annotations_csv  = "annotations.csv",
            candidates_v2_csv= "candidates_V2.csv",
        )
        # Generar submission CSV con probabilidades del modelo
        evaluator.write_submission(
            predictions  = {patch_stem: probability, ...},
            output_csv   = "submission.csv"
        )
        # Ejecutar script oficial de evaluación
        evaluator.run_froc_script(
            submission_csv      = "submission.csv",
            evaluation_script   = "noduleCADEvaluationLUNA16.py",
            output_dir          = "./froc_results/"
        )
    """

    # Los 7 puntos FP/scan del CPM — no modificar (definidos por el challenge)
    FP_THRESHOLDS = [1/8, 1/4, 1/2, 1, 2, 4, 8]

    def __init__(self, seriesuid_list: list,
                 annotations_csv: str,
                 candidates_v2_csv: str):
        """
        Args:
            seriesuid_list   : lista de seriesuid de los escaneos del dataset
            annotations_csv  : ruta a annotations.csv (ground truth de nódulos)
            candidates_v2_csv: ruta a candidates_V2.csv
        """
        self.seriesuid_list    = seriesuid_list
        self.annotations_csv   = annotations_csv
        self.candidates_v2_csv = candidates_v2_csv

        # Cargar anotaciones para calcular el techo teórico
        self.annotations = pd.read_csv(annotations_csv)
        self.candidates  = pd.read_csv(candidates_v2_csv)

        n_nodules   = len(self.annotations)
        n_positives = (self.candidates["class"] == 1).sum()
        sensitivity_ceil = n_positives / max(n_nodules, 1)

        log.info(
            f"[FROC] Evaluador LUNA16 inicializado:\n"
            f"    Nódulos en annotations.csv : {n_nodules}\n"
            f"    Candidatos positivos (V2)  : {n_positives}\n"
            f"    Techo teórico sensitividad : {sensitivity_ceil:.4f} "
            f"({sensitivity_ceil*100:.1f}%) — {n_nodules - n_positives} nódulos sin candidato\n"
            f"    FP_THRESHOLDS (CPM)        : {self.FP_THRESHOLDS}\n"
            f"    Punto clínico clave        : sensitividad @ 1 FP/scan"
        )

    def write_submission(self, predictions: dict, output_csv: str) -> None:
        """
        H6 — Genera el CSV de submission en el formato requerido por noduleCADEvaluationLUNA16.py.

        Formato requerido:
            seriesuid, coordX, coordY, coordZ, probability

        Args:
            predictions : dict {patch_stem: probability} donde patch_stem es
                          el nombre del archivo .npy sin extensión
                          (ej: "candidate_000042" → probability = 0.87)
            output_csv  : ruta de salida del CSV
        """
        rows = []
        # Mapear patch_stem → fila del candidates_V2.csv
        for i, row in self.candidates.iterrows():
            stem = f"candidate_{i:06d}"
            prob = predictions.get(stem, 0.0)
            rows.append({
                "seriesuid":   row["seriesuid"],
                "coordX":      row["coordX"],
                "coordY":      row["coordY"],
                "coordZ":      row["coordZ"],
                "probability": float(prob),
            })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_csv, index=False)
        log.info(
            f"[FROC] Submission CSV escrito: {output_csv} | "
            f"{len(df_out):,} candidatos | "
            f"prob media: {df_out['probability'].mean():.4f} | "
            f"prob máx: {df_out['probability'].max():.4f}"
        )

    def run_froc_script(self, submission_csv: str,
                        evaluation_script: str,
                        output_dir: str) -> None:
        """
        H6 — Ejecuta noduleCADEvaluationLUNA16.py con subprocess.

        Requiere que el script esté en el ZIP del challenge LUNA16.
        Genera: curva FROC (.png), datos FROC (.txt), CADAnalysis.txt con CPM.

        Args:
            submission_csv    : CSV generado por write_submission()
            evaluation_script : ruta a noduleCADEvaluationLUNA16.py
            output_dir        : directorio de salida para los resultados FROC
        """
        import subprocess
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "python", evaluation_script,
            "-s", submission_csv,
            "-a", self.annotations_csv,
            "-o", output_dir,
        ]
        log.info(f"[FROC] Ejecutando: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                log.info(f"[FROC] Evaluación FROC completada ✓ → resultados en '{output_dir}'")
                # Mostrar el CPM del output
                for line in result.stdout.split("\n"):
                    if "CPM" in line or "cpm" in line.lower() or "sensitivity" in line.lower():
                        log.info(f"[FROC] {line.strip()}")
            else:
                log.error(f"[FROC] Error en el script FROC (returncode={result.returncode}):\n"
                          f"{result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            log.error("[FROC] El script FROC tardó más de 5 minutos. "
                      "Verifica que el submission CSV no esté vacío.")
        except FileNotFoundError:
            log.error(f"[FROC] Script no encontrado: '{evaluation_script}'. "
                      f"Extrae noduleCADEvaluationLUNA16.py del ZIP del challenge LUNA16.")

    @staticmethod
    def log_cpm_interpretation(cpm_value: float) -> None:
        """H6 — Interpreta el CPM en el contexto del challenge LUNA16."""
        if cpm_value >= 0.90:
            level = "EXCELENTE — nivel SOTA (ensemble de sistemas)"
        elif cpm_value >= 0.80:
            level = "BUENO — nivel sistema individual SOTA (~0.831)"
        elif cpm_value >= 0.70:
            level = "ACEPTABLE — objetivo mínimo del proyecto"
        elif cpm_value >= 0.50:
            level = "BAJO — el modelo detecta nódulos pero con muchos FP"
        else:
            level = "MUY BAJO — revisar FocalLoss, conversión world→vóxel y normalización HU"
        log.info(
            f"[FROC] CPM = {cpm_value:.4f} → {level}\n"
            f"    Referencia: LUNA16 sistema individual SOTA ≈ 0.831\n"
            f"    Referencia: LUNA16 ensemble SOTA ≈ 0.946\n"
            f"    Punto clave: sensitividad @ 1 FP/scan (objetivo >0.60)"
        )


class FocalLoss(nn.Module):
    """
    H2 — FocalLoss (Lin et al., ICCV 2017 — arXiv:1708.02002).

    Reduce el gradiente de los ejemplos negativos fáciles (los ~549,000 que el
    modelo ya predice bien) y concentra el aprendizaje en los positivos (~1,186)
    y los negativos difíciles cercanos a nódulos reales.

    Con desbalance ~490:1, BCELoss estándar converge a predecir siempre class=0
    con >99.7% de accuracy. FocalLoss evita ese mínimo trivial.

    Parámetros recomendados para LUNA16:
      gamma=2   : factor de modulación estándar (Lin et al.)
      alpha=0.25: peso para la clase positiva (nódulo)
                  alpha alto (~0.75) para el positivo si el desbalance es extremo

    Uso en FASE 2:
        criterion = FocalLoss(gamma=2, alpha=0.25)
        loss = criterion(logits, labels.float())  # logits: [B], labels: [B] ∈ {0,1}

    Alternativa: submuestreo de negativos 1:10 a 1:20 máximo.
    Combinar ambas estrategias da los mejores resultados en la literatura.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B] — logits sin sigmoid (BCEWithLogitsLoss internamente)
        targets : [B] — etiquetas binarias float {0.0, 1.0}
        """
        bce    = nn.functional.binary_cross_entropy_with_logits(
                     logits, targets, reduction="none")
        p_t    = torch.exp(-bce)                      # probabilidad del target real
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss   = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class LUNA16PatchExtractor:
    """
    H3 — Extrae parches 3D centrados en coordenadas world del candidates_V2.csv.

    PROBLEMA CRÍTICO: las coordenadas en candidates_V2.csv están en milímetros
    en el espacio del mundo (world coordinates), NO en índices de vóxel.

    BUG CLÁSICO (NO hacer esto):
        coord_voxel = coord_world / spacing          # INCORRECTO — ignora origin y direction

    SOLUCIÓN CORRECTA:
        coord_shifted = coord_world - origin
        coord_voxel   = np.linalg.solve(direction * spacing, coord_shifted)

    Con volúmenes típicos de 512×512×400 vóxeles y spacing ~0.7×0.7×2.5 mm,
    un error de 10 mm desplaza el parche ~14 vóxeles en XY — suficiente para
    que el parche no contenga el nódulo y el modelo se entrene con etiquetas incorrectas.

    Uso:
        extractor = LUNA16PatchExtractor()
        patch = extractor.extract(mhd_path, coord_world=[x, y, z], patch_size=64)
        np.save(f"candidate_000001.npy", patch)   # guardar como .npy antes de FASE 0
    """

    @staticmethod
    def world_to_voxel(coord_world: np.ndarray,
                       origin: np.ndarray,
                       spacing: np.ndarray,
                       direction: np.ndarray) -> np.ndarray:
        """
        H3 — Conversión world → vóxel correcta.

        Args:
            coord_world : [x, y, z] en milímetros (del candidates_V2.csv)
            origin      : imagen.GetOrigin()   — [x0, y0, z0] en mm
            spacing     : imagen.GetSpacing()  — [dx, dy, dz] en mm/vóxel
            direction   : imagen.GetDirection().reshape(3,3) — matriz de orientación

        Returns:
            índices de vóxel [iz, iy, ix] (orden Z,Y,X para indexar el array numpy)
        """
        coord_shifted = np.array(coord_world) - np.array(origin)
        # direction * spacing = Jacobiano de la transformación vóxel→world
        coord_voxel   = np.linalg.solve(
            np.array(direction).reshape(3, 3) * np.array(spacing),
            coord_shifted
        )
        # SimpleITK devuelve coordenadas en orden X,Y,Z; numpy usa Z,Y,X
        return np.round(coord_voxel[::-1]).astype(int)   # [iz, iy, ix]

    @staticmethod
    def extract(mhd_path: str,
                coord_world: list,
                patch_size: int = 64,
                clip_hu: tuple = (-1000, 400)) -> np.ndarray:
        """
        H3 — Lee un volumen .mhd y extrae un parche centrado en coord_world.

        Normalización HU incluida (clip + escala a [0,1]).
        Padding con -1000 HU → 0.0 normalizado si el candidato está en el borde.

        Args:
            mhd_path   : ruta al archivo .mhd (el .zraw debe estar en el mismo directorio)
            coord_world: [x, y, z] en milímetros (del candidates_V2.csv)
            patch_size : tamaño del parche cúbico (default 64 → [64,64,64])
            clip_hu    : rango HU para clip (default [-1000, 400] — pulmón/nódulos)

        Returns:
            np.ndarray float32 [patch_size, patch_size, patch_size] normalizado [0,1]
        """
        image   = sitk.ReadImage(mhd_path)
        origin  = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())
        direc   = np.array(image.GetDirection())
        array   = sitk.GetArrayFromImage(image).astype(np.float32)   # [Z, Y, X] en HU

        iz, iy, ix = LUNA16PatchExtractor.world_to_voxel(
            coord_world, origin, spacing, direc)

        half = patch_size // 2
        z1, z2 = max(0, iz - half), min(array.shape[0], iz + half)
        y1, y2 = max(0, iy - half), min(array.shape[1], iy + half)
        x1, x2 = max(0, ix - half), min(array.shape[2], ix + half)

        patch = array[z1:z2, y1:y2, x1:x2]

        # Padding con -1000 HU (aire) si el candidato está cerca del borde
        if patch.shape != (patch_size, patch_size, patch_size):
            pad_z = patch_size - patch.shape[0]
            pad_y = patch_size - patch.shape[1]
            pad_x = patch_size - patch.shape[2]
            patch = np.pad(patch,
                           ((0, pad_z), (0, pad_y), (0, pad_x)),
                           constant_values=clip_hu[0])

        # Normalización HU a [0, 1]
        patch = np.clip(patch, clip_hu[0], clip_hu[1])
        patch = (patch - clip_hu[0]) / (clip_hu[1] - clip_hu[0])
        return patch.astype(np.float32)

    @staticmethod
    def validate_extraction(mhd_paths: list,
                            candidates_df,
                            n_positives: int = 10) -> bool:
        """
        H3/Item-3 — Valida la conversión world→vóxel en n candidatos class=1.

        OBLIGATORIO ejecutar antes de extraer el dataset completo.
        Verifica que los parches de candidatos positivos tengan intensidad
        media plausible para tejido blando/nódulo (no fondo de aire).

        Args:
            mhd_paths     : lista de rutas a archivos .mhd disponibles
            candidates_df : DataFrame de candidates_V2.csv
            n_positives   : número de candidatos positivos a verificar (default 10)

        Returns:
            True si todos los parches tienen intensidad media > 0.1 (plausible)
            False si algún parche parece solo aire (error de conversión probable)
        """
        positives = candidates_df[candidates_df["class"] == 1].head(n_positives)
        if len(positives) == 0:
            log.error("[LUNA16/validate] Sin candidatos positivos en el CSV. "
                      "¿Estás usando candidates_V2.csv?")
            return False

        mhd_map  = {Path(p).stem: p for p in mhd_paths}
        all_ok   = True

        log.info(f"[LUNA16/validate] Validando conversión world→vóxel "
                 f"en {len(positives)} candidatos positivos...")

        for i, (_, row) in enumerate(positives.iterrows()):
            mhd_path = mhd_map.get(row["seriesuid"])
            if mhd_path is None:
                log.warning(f"  [{i+1}] seriesuid '{row['seriesuid']}' no encontrado "
                            f"en la lista de .mhd")
                continue
            try:
                patch = LUNA16PatchExtractor.extract(
                    mhd_path, [row["coordX"], row["coordY"], row["coordZ"]])
                mean_val = patch.mean()
                status   = "✓" if mean_val > 0.1 else "⚠ POSIBLE ERROR"
                log.info(f"  [{i+1}] {Path(mhd_path).stem[:30]:30s} | "
                         f"coord=({row['coordX']:.1f},{row['coordY']:.1f},{row['coordZ']:.1f}) | "
                         f"patch_mean={mean_val:.3f} {status}")
                if mean_val <= 0.1:
                    all_ok = False
                    log.error(f"  [{i+1}] Parche con intensidad media muy baja ({mean_val:.3f}). "
                              f"El parche es casi todo aire — posible error en la conversión "
                              f"world→vóxel. Verificar origen, spacing y direction del header .mhd.")
            except Exception as e:
                log.error(f"  [{i+1}] Error extrayendo parche: {e}")
                all_ok = False

        if all_ok:
            log.info("[LUNA16/validate] Conversión world→vóxel VÁLIDA en todos los candidatos ✓")
        else:
            log.error("[LUNA16/validate] ¡VALIDACIÓN FALLIDA! Revisar la conversión "
                      "world→vóxel antes de extraer el dataset completo.")
        return all_ok


# ── LUNA16Dataset ────────────────────────────────────────────────────────────

class LUNA16Dataset(Dataset):
    """
    LUNA16 — Clasificación de PARCHES 3D de candidatos a nódulo.

    ⚠ H1 — La tarea NO es clasificar volúmenes completos.
      Cada candidato en candidates_V2.csv es una posición (x,y,z) en mm.
      Se extrae un parche 64×64×64 vóxeles centrado ahí y se clasifica.
      Tensor de entrada al Experto 3 en FASE 2: [B, 1, 64, 64, 64]
      (un solo canal de intensidad HU normalizada).

    ⚠ H2 — Desbalance ~490:1: BCELoss estándar → modelo trivial (siempre class=0).
      Usar FocalLoss(gamma=2, alpha=0.25) obligatoriamente en FASE 2.
      Clase FocalLoss implementada en este módulo.

    ⚠ H3 — Conversión world→vóxel: fuente de bugs #1 en LUNA16.
      LUNA16PatchExtractor implementa la conversión correcta.
      validate_extraction() verifica 10 candidatos antes de producción.

    Item-2  — Siempre usar candidates_V2.csv (V1 está obsoleto).
    Item-4  — Normalización HU clip[-1000,400] → [0,1] aplicada en FASE 0.
    Item-7  — gradient checkpointing obligatorio con 12 GB VRAM (documentado).
    Item-8  — CPM y curva FROC calculados con noduleCADEvaluationLUNA16.py.
    Item-9  — Verificación de spacing en __init__.
    Item-10 — Techo teórico: 94.4% (66/1,186 nódulos sin candidato en V2).

    Para FASE 0 (embedding): el volumen se convierte a representación 2D
    (3 slices centrales axial/coronal/sagital apilados como RGB).
    Para FASE 2 (expert): el tensor es [B, 1, 64, 64, 64] — arquitectura 3D real.

    Dos modos:
      mode="embedding" → FASE 0: (img_2d, expert_id=3, patch_stem)
      mode="expert"    → FASE 2: (volume_3d, label_binary, patch_stem)
    """

    # Techo teórico de sensitividad en LUNA16 (66 nódulos sin candidato en V2)
    SENSITIVITY_CEILING = 1120 / 1186   # ≈ 0.9443

    def __init__(self, patches_dir: str, candidates_csv: str, mode: str = "embedding"):
        """
        Args:
            patches_dir    : carpeta con candidate_XXXXXX.npy (64×64×64, HU normalizado)
            candidates_csv : SIEMPRE candidates_V2.csv — nunca candidates.csv (V1)
            mode           : "embedding" (FASE 0) o "expert" (FASE 2)
        """
        assert mode in ("embedding", "expert"), \
            f"[LUNA16] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"

        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["luna"]
        self.mode        = mode

        if not self.patches_dir.exists():
            log.error(f"[LUNA16] Directorio de parches no encontrado: {self.patches_dir}")
            self.samples = []
            return

        # ── Item-2 / H7: verificar V2 vs V1 ─────────────────────────────────
        csv_path = Path(candidates_csv)
        csv_lower = csv_path.name.lower()
        if "candidates_v2" in csv_lower or "_v2" in csv_lower:
            log.debug(f"[LUNA16] H7/Item-2 — candidates_V2.csv verificado ✓")
        elif csv_path.name == "candidates.csv" or "candidates.csv" == csv_path.name:
            log.error(
                f"[LUNA16] H7/Item-2 — ¡CSV OBSOLETO! Se detectó 'candidates.csv' (V1). "
                f"V1 es el conjunto original del ISBI 2016, explícitamente deprecado. "
                f"candidates_V2.csv detecta 1,162/1,186 nódulos (+24 que V1). "
                f"Usa: --luna_csv /ruta/candidates_V2/candidates_V2.csv "
                f"Impacto: usar V1 reduce artificialmente el techo de sensitividad alcanzable."
            )
        else:
            log.warning(
                f"[LUNA16] H7/Item-2 — El CSV '{csv_path.name}' no parece ser candidates_V2.csv. "
                f"Verifica que estás usando V2 y no V1 (obsoleto). "
                f"V2 detecta 24 nódulos más que V1 y es el estándar del challenge."
            )

        df = pd.read_csv(candidates_csv)
        log.info(f"[LUNA16] CSV cargado: {len(df):,} candidatos totales | "
                 f"columnas: {list(df.columns)}")

        # ── Item-5: verificar ratio class=1 / class=0 ─────────────────────────
        n_pos_csv = (df["class"] == 1).sum()
        n_neg_csv = (df["class"] == 0).sum()
        ratio_csv = n_neg_csv / max(n_pos_csv, 1)
        log.info(
            f"[LUNA16] Item-5 — Distribución en CSV: "
            f"positivos={n_pos_csv:,} | negativos={n_neg_csv:,} | "
            f"ratio={ratio_csv:.0f}:1 (esperado ~490:1)"
        )
        if ratio_csv < 100:
            log.warning(f"[LUNA16] Ratio neg/pos={ratio_csv:.0f}:1 inusualmente bajo. "
                        f"¿El CSV está completo?")
        elif ratio_csv > 1000:
            log.warning(f"[LUNA16] Ratio neg/pos={ratio_csv:.0f}:1 extremo. "
                        f"FocalLoss obligatoria — BCELoss convergirá al mínimo trivial.")

        # ── Cargar parches disponibles en disco ───────────────────────────────
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
        ratio = n_neg / max(n_pos, 1)

        log.info(
            f"[LUNA16] {len(self.samples):,} parches cargados del disco | "
            f"positivos: {n_pos:,} | negativos: {n_neg:,} | ratio: {ratio:.0f}:1"
        )
        if missing > 0:
            log.warning(
                f"[LUNA16] {missing:,} parches del CSV no encontrados en disco '{self.patches_dir}'. "
                f"Ejecuta LUNA16PatchExtractor.extract() para generarlos."
            )
        if n_pos == 0:
            log.error("[LUNA16] ¡Cero candidatos positivos cargados! "
                      "El modelo no podrá aprender nódulos. Verifica el CSV y los .npy.")

        # ── H4/Item-4: normalización HU — justificación fisiológica ──────────
        log.info(
            "[LUNA16] H4/Item-4 — Normalización HU clip[-1000, 400] → [0, 1]:\n"
            f"    HU_LUNG_CLIP = {HU_LUNG_CLIP} — constante exportada para FASE 2.\n"
            "    Justificación del rango:\n"
            "      -1000 HU = aire puro (fondo del parche, padding consistente)\n"
            "       -900 a -500 HU = tejido pulmonar\n"
            "       -100 a +100 HU = nódulos sólidos (lo que queremos detectar)\n"
            "       +400 HU = límite: hueso EXCLUIDO — dominaría la dinámica de la red\n"
            "    Sin este clip, las capas iniciales gastan capacidad representacional\n"
            "    en distinguir tipos de hueso en lugar de tipos de tejido pulmonar.\n"
            f"    Verificar normalización con: verify_hu_normalization(patches_dir)"
        )

        # ── Item-2 + H2: log de FocalLoss ────────────────────────────────────
        log.info(
            "[LUNA16] H2 — Loss obligatoria para FASE 2: FocalLoss(gamma=2, alpha=0.25). "
            "BCELoss estándar convergirá a predecir siempre class=0 (accuracy >99.7% trivial). "
            "FocalLoss implementada en este módulo. "
            "Alternativa complementaria: submuestreo de negativos (ratio máx. 1:20)."
        )

        # ── H5/Item-7: gradient checkpointing — cálculo de VRAM ─────────────
        log.info(
            "[LUNA16] H5/Item-7 — Gradient checkpointing OBLIGATORIO con 12 GB VRAM.\n"
            "    Cálculo de VRAM por batch:\n"
            "      Entrada  : 1 parche 64³ float32 = 1 MB → batch_size=4 → 4 MB\n"
            "      Activaciones red 3D (5+ capas) sin checkpointing: 4-8 GB por muestra\n"
            "      Con FP16 + checkpointing: ~1-2 GB por muestra → batch_size=4 viable\n"
            "    ⚠ Las 3 condiciones deben estar activas SIMULTÁNEAMENTE:\n"
            "      1. FP16 (torch.cuda.amp.autocast)\n"
            "      2. gradient checkpointing (torch.utils.checkpoint)\n"
            "      3. batch_size ≤ 4\n"
            "    Sin cualquiera de las tres → OOM en el forward pass inicial.\n"
            "    Ejemplo de implementación en FASE 2:\n"
            "        from torch.utils.checkpoint import checkpoint_sequential\n"
            "        scaler = torch.cuda.amp.GradScaler()\n"
            "        with torch.cuda.amp.autocast():\n"
            "            output = checkpoint_sequential(model.encoder, segments=4, input=x)\n"
            "        loss = criterion(output, labels)\n"
            "        scaler.scale(loss).backward()\n"
            "        scaler.step(optimizer)\n"
            "        scaler.update()"
        )

        # ── H6/Item-8: FROC/CPM — por qué no AUC-ROC ─────────────────────────
        log.info(
            f"[LUNA16] H6/Item-8 — Métrica oficial: CPM (Competition Performance Metric).\n"
            f"    ⚠ AUC-ROC NO es la métrica correcta:\n"
            f"      AUC-ROC es a nivel de candidato individual, no de nódulo.\n"
            f"      No captura que un nódulo puede generar múltiples candidatos cercanos.\n"
            f"      No refleja el costo clínico de FP medidos por escaneo.\n"
            f"    CPM = promedio de sensitividad en los 7 puntos FP/scan:\n"
            f"      {LUNA16FROCEvaluator.FP_THRESHOLDS}\n"
            f"    Punto clínicamente más relevante: sensitividad @ 1 FP/scan.\n"
            f"    Un radiólogo puede revisar ~1 FP/scan antes de que el sistema\n"
            f"    pierda utilidad clínica.\n"
            f"    Techo teórico: {self.SENSITIVITY_CEILING:.4f} ({self.SENSITIVITY_CEILING*100:.1f}%)\n"
            f"    — 66 nódulos sin candidato en V2: nunca detectables.\n"
            f"    Documentar en el Reporte Técnico (sección 3).\n"
            f"    Cómo evaluar:\n"
            f"        evaluator = LUNA16FROCEvaluator(seriesuid_list, annotations_csv, candidates_v2_csv)\n"
            f"        evaluator.write_submission(predictions_dict, 'submission.csv')\n"
            f"        evaluator.run_froc_script('submission.csv', 'noduleCADEvaluationLUNA16.py', './froc/')"
        )

        # ── Item-9: verificar spacing variable en muestra ────────────────────
        self._verify_spacing_sample()

    def _verify_spacing_sample(self):
        """
        Item-9 — Verifica si el spacing es uniforme o variable en los .npy cargados.
        Los parches ya extraídos son todos [64,64,64] pero el spacing original
        del CT afecta la escala física. Si es muy variable, considerar re-muestreo
        isométrico antes de la extracción.

        Como los .npy no guardan metadata de spacing, este check mide las
        dimensiones del parche en disco — si alguno difiere de [64,64,64]
        hay un bug en la extracción.
        """
        import random
        n_check = min(5, len(self.samples))
        if n_check == 0:
            return
        shapes = set()
        for patch_file, _ in random.sample(self.samples, n_check):
            try:
                arr = np.load(patch_file)
                shapes.add(arr.shape)
            except Exception:
                pass
        if len(shapes) == 1 and next(iter(shapes)) == (64, 64, 64):
            log.debug("[LUNA16] Item-9 — Todos los parches muestreados son [64,64,64] ✓")
        elif len(shapes) == 1:
            log.warning(f"[LUNA16] Item-9 — Parches con forma {next(iter(shapes))} "
                        f"— se esperaba (64,64,64). Verifica la extracción.")
        else:
            log.error(f"[LUNA16] Item-9 — Formas inconsistentes en los parches: {shapes}. "
                      f"Todos los .npy deben ser [64,64,64]. Re-extrae con "
                      f"LUNA16PatchExtractor.extract(patch_size=64).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_file, label = self.samples[idx]
        try:
            volume = np.load(patch_file)   # [64, 64, 64] normalizado [0,1]
        except Exception as e:
            log.warning(f"[LUNA16] Error cargando '{patch_file}': {e}. "
                        f"Reemplazando con tensor cero.")
            if self.mode == "embedding":
                return torch.zeros(3, 224, 224), self.expert_id, patch_file.stem
            else:
                return torch.zeros(1, 64, 64, 64), label, patch_file.stem

        # Detectar parche todo-constante (error de extracción o padding excesivo)
        if volume.max() == volume.min():
            log.warning(f"[LUNA16] Parche '{patch_file.name}' es constante "
                        f"(valor={volume.max():.3f}). Posible error en la extracción "
                        f"world→vóxel o parche fuera del volumen.")

        if self.mode == "embedding":
            # FASE 0: proyectar 3D → 2D (3 slices centrales como RGB) para ViT
            volume_t = resize_volume_3d(volume)         # [1, 64, 64, 64]
            img      = volume_to_vit_input(volume_t)    # [3, 224, 224]
            return img, self.expert_id, patch_file.stem
        else:
            # FASE 2: tensor 3D real para arquitectura 3D (R3D-18, Swin3D, etc.)
            # ⚠ H1: el tensor es [1, 64, 64, 64] — un solo canal de intensidad HU
            # El Experto 3 recibe PARCHES, no volúmenes completos
            volume_t = torch.from_numpy(volume).float().unsqueeze(0)  # [1, 64, 64, 64]
            return volume_t, label, patch_file.stem


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

        # H7 — Advertir asimetría si filter_view se aplica solo a uno de los splits.
        # Ambos splits deben recibir el mismo view_filter para que el dominio
        # del espacio de validación coincida con el de entrenamiento.
        # Como ambas instancias reciben view_filter del mismo cfg, son simétricas.
        # El riesgo ocurre si el usuario instancia ChestXray14Dataset manualmente
        # con filter_view diferente — este comentario lo documenta explícitamente.
        if view_filter:
            log.info(
                f"[Chest] H7 — filter_view='{view_filter}' aplicado SIMÉTRICAMENTE "
                f"a train y val. Si alguna vez instancias ChestXray14Dataset manualmente, "
                f"asegúrate de pasar el mismo filter_view a ambos splits para evitar "
                f"que el modelo vea distribuciones distintas de View Position en train vs val."
            )

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

        # H4: pasar img_dir para deduplicación por hash MD5 antes del split
        isic_train_df, isic_val_df = ISICDataset.build_lesion_split(
            gt_csv       = cfg["isic_gt"],
            img_dir      = cfg.get("isic_imgs", None),   # H4: hashing MD5
            metadata_csv = cfg.get("isic_metadata", None),
            frac_train   = 0.8,
            random_state = 42,
        )

        isic_train = ISICDataset(
            img_dir  = cfg["isic_imgs"],
            split_df = isic_train_df,
            mode     = "embedding",
        )
        isic_val = ISICDataset(
            img_dir  = cfg["isic_imgs"],
            split_df = isic_val_df,
            mode     = "embedding",
        )
        train_datasets.append(isic_train)
        val_datasets.append(isic_val)
        log.info(f"  → train: {len(isic_train):,}  val: {len(isic_val):,}")
    else:
        log.warning("[Dataset] ISIC 2019 OMITIDO — falta --isic_gt o --isic_imgs")
 
    # ── Experto 3: OA Rodilla ──
    if cfg.get("oa_root"):
        log.info("[Dataset] Cargando Osteoarthritis Knee (Experto 2)...")
        oa_train = OAKneeDataset(cfg["oa_root"], split="train", mode="embedding")
        oa_val   = OAKneeDataset(cfg["oa_root"], split="val",   mode="embedding")
        train_datasets.append(oa_train)
        val_datasets.append(oa_val)
        log.info(f"  → train: {len(oa_train):,}  val: {len(oa_val):,}")
    else:
        log.warning("[Dataset] OA Rodilla OMITIDO — falta --oa_root")
 
    # ── Experto 4: LUNA16 ──
    if cfg.get("luna_patches") and cfg.get("luna_csv"):
        log.info("[Dataset] Cargando LUNA16 parches 3D (Experto 3)...")
        luna_train = LUNA16Dataset(cfg["luna_patches"] + "/train",
                                   cfg["luna_csv"], mode="embedding")
        luna_val   = LUNA16Dataset(cfg["luna_patches"] + "/val",
                                   cfg["luna_csv"], mode="embedding")
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
        "isic_metadata":     args.isic_metadata,   # H2: ISIC_2019_Training_Metadata.csv
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
    parser.add_argument("--isic_gt", default=None,
                        help="ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--isic_imgs", default=None,
                        help="Carpeta con imágenes .jpg de ISIC")
    parser.add_argument(
        "--isic_metadata", default=None,
        help=(
            "H2 — ISIC_2019_Training_Metadata.csv. Necesario para hacer el split "
            "por lesion_id y evitar leakage (misma lesión en train y val). "
            "Sin este archivo, el split cae back a aleatorio por imagen con WARNING."
        )
    )
 
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