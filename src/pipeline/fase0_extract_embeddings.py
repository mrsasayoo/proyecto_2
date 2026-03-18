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
 
# Esta clase define el contenedor de datos especializado para el dataset NIH ChestXray14. Su función es filtrar
# los archivos de imagen mediante las listas oficiales de división (train/val), asignar automáticamente la identidad 
# del experto (0) y aplicar las transformaciones de imagen necesarias para que el backbone del sistema pueda procesarlas.
class ChestXray14Dataset(Dataset):
    """
    NIH ChestXray14 — Multi-label, 14 patologías.
    Para embedding: la etiqueta de experto es siempre 0.
    """
    def __init__(self, csv_path, img_dir, file_list, transform):
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.expert_id = EXPERT_IDS["chest"]
 
        df = pd.read_csv(csv_path)
        # Filtrar según train_val_list.txt o test_list.txt
        with open(file_list) as f:
            valid_files = set(f.read().splitlines())
        self.df = df[df["Image Index"].isin(valid_files)].reset_index(drop=True)
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image Index"]
        img      = Image.open(self.img_dir / img_name).convert("RGB")
        img      = self.transform(img)
        return img, self.expert_id, img_name
 
 
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
 
        # Eliminar duplicados conocidos antes de cualquier split
        df = df[~df["image"].isin(["ISIC_0067980", "ISIC_0069013"])].reset_index(drop=True)
 
        # Split aproximado — en producción usar lesion_id del metadata CSV
        if split == "train":
            self.df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        else:
            self.df = df.drop(df.sample(frac=0.8, random_state=42).index).reset_index(drop=True)
 
        # En modo expert: convertir one-hot → índice de clase (0–7)
        if self.mode == "expert":
            self.df = self.df.copy()
            self.df["class_label"] = self.df[self.CLASSES].values.argmax(axis=1)
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        img_path = self.img_dir / f"{img_name}.jpg"
        img      = Image.open(img_path).convert("RGB")
        img      = self.transform(img)
 
        if self.mode == "embedding":
            # FASE 0: el router aprende "soy imagen de ISIC"
            return img, self.expert_id, img_name
        else:
            # FASE 2: el Experto 2 aprende la patología real
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
        # Inferir carpetas: 0/, 1/, 2/ (o verificar con os.listdir)
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, int(class_dir.name)))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, int(class_dir.name)))
 
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        # CLAHE antes del resize (requisito del proyecto)
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
 
    NOTA IMPORTANTE: La conversión world→vóxel ya debe estar resuelta
    antes de llamar a este dataset. Este dataset asume que tienes
    parches 3D pre-extraídos guardados como .npy de [64,64,64].
    """
    def __init__(self, patches_dir, candidates_csv):
        """
        patches_dir: carpeta con archivos candidate_XXXXX.npy (64×64×64, HU)
        candidates_csv: candidates_V2.csv con columnas seriesuid, class
        """
        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["luna"]
 
        df = pd.read_csv(candidates_csv)
        self.samples = []
        for _, row in df.iterrows():
            patch_file = self.patches_dir / f"candidate_{row.name:06d}.npy"
            if patch_file.exists():
                self.samples.append((patch_file, int(row["class"])))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        patch_file, _ = self.samples[idx]
        volume  = np.load(patch_file)                   # [64, 64, 64] HU crudo
        volume  = normalize_hu(volume)                  # → [0, 1]
        volume_t = resize_volume_3d(volume)             # [1, 64, 64, 64]
        img     = volume_to_vit_input(volume_t)         # [3, 224, 224]
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
        self.samples = list(self.patches_dir.glob("*.npy"))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        patch_file = self.samples[idx]
        volume  = np.load(patch_file)
        volume  = normalize_hu(volume)
        volume_t = resize_volume_3d(volume)
        img     = volume_to_vit_input(volume_t)
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
    print(f"[ViT] Backbone seleccionado : {backbone_name}")
    print(f"[ViT] d_model esperado      : {expected_d}")
    print(f"[ViT] VRAM estimada (extrac): ~{vram_est} GB")
 
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0          # elimina la cabeza de clasificación → devuelve CLS token
    )
 
    # Congelar TODOS los parámetros — FASE 0 es extracción pura, no aprendizaje
    # (alineado con sección 4.1: "backbone congelado en FASE 0 y FASE 1")
    for param in model.parameters():
        param.requires_grad = False
 
    model.eval()
    model.to(device)
 
    # Verificar d_model con un forward pass dummy
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
 
    actual_d = out.shape[1]
    if actual_d != expected_d:
        print(f"[ViT] ⚠ d_model real ({actual_d}) difiere del esperado ({expected_d}). "
              f"Actualiza BACKBONE_CONFIGS si agregaste un backbone nuevo.")
 
    print(f"[ViT] d_model real (CLS dim): {actual_d}")
    print(f"[ViT] Parámetros congelados : {sum(p.numel() for p in model.parameters()):,}")
 
    return model, actual_d   # (model, d_model)
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
    Pasa el dataloader completo por el backbone y acumula:
      - Z:           [N, d_model] — CLS tokens
      - y_expert:    [N]          — etiqueta de experto (0–4, dominios clínicos)
      - img_names:   [N]          — nombre de imagen para auditoría
 
    Nota sobre el Experto 5 (OOD):
      El Experto 6 (ID=5) NO aparece en y_expert porque no tiene dataset
      propio. y_expert solo contendrá valores en {0, 1, 2, 3, 4}.
      El Experto 6 se inicializa y entrena en FASE 1 mediante L_error.
    """
    Z          = np.zeros((len(dataloader.dataset), d_model), dtype=np.float32)
    y_expert   = np.zeros(len(dataloader.dataset), dtype=np.int64)
    img_names  = []
    cursor     = 0
 
    for batch_idx, (imgs, experts, names) in enumerate(dataloader):
        imgs = imgs.to(device)
 
        # Forward pass (sin gradientes — congelado)
        z = model(imgs)                          # [B, d_model]
        z_np = z.cpu().numpy()
 
        B = z_np.shape[0]
        Z[cursor:cursor + B]        = z_np
        y_expert[cursor:cursor + B] = experts.numpy()
        img_names.extend(names)
        cursor += B
 
        if batch_idx % 50 == 0:
            print(f"  {desc} [{cursor}/{len(dataloader.dataset)}]")
 
    return Z[:cursor], y_expert[:cursor], img_names
# Un pequeño consejo "pro" para tu reporte:
# Cuando describas esta función en tu reporte técnico, menciona que la pre-asignación de memoria (NumPy zeros) 
# y el uso de torch.no_grad() fueron decisiones estratégicas para optimizar la eficiencia del pipeline en el 
# hardware limitado (2 GPUs de 12GB), asegurando que el proceso de extracción escalara correctamente con 
# datasets masivos como NIH ChestXray14 (112k imágenes).


# ──────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────
 

# Esta función actúa como una "Fábrica de Datasets". Su rol es verificar qué datasets están disponibles 
# según la configuración, instanciar las clases específicas para cada modalidad (2D o 3D) y unificar todos 
# estos conjuntos heterogéneos en dos estructuras únicas (ConcatDataset) para entrenamiento y validación, 
# permitiendo que el sistema procese un flujo mixto de imágenes médicas sin distinción de origen.
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
        print("[Dataset] Cargando NIH ChestXray14...")
        chest_train = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["chest_train_list"],
            transform=transform_2d
        )
        chest_val = ChestXray14Dataset(
            csv_path=cfg["chest_csv"],
            img_dir=cfg["chest_imgs"],
            file_list=cfg["chest_val_list"],
            transform=transform_2d
        )
        train_datasets.append(chest_train)
        val_datasets.append(chest_val)
        print(f"  train: {len(chest_train):,}  val: {len(chest_val):,}")
 
    # ── Experto 2: ISIC 2019 ──
    if cfg.get("isic_gt") and cfg.get("isic_imgs"):
        print("[Dataset] Cargando ISIC 2019...")
        isic_train = ISICDataset(cfg["isic_gt"], cfg["isic_imgs"], transform_2d, split="train")
        isic_val   = ISICDataset(cfg["isic_gt"], cfg["isic_imgs"], transform_2d, split="val")
        train_datasets.append(isic_train)
        val_datasets.append(isic_val)
        print(f"  train: {len(isic_train):,}  val: {len(isic_val):,}")
 
    # ── Experto 3: OA Rodilla ──
    if cfg.get("oa_root"):
        print("[Dataset] Cargando Osteoarthritis Knee...")
        oa_train = OAKneeDataset(cfg["oa_root"], split="train")
        oa_val   = OAKneeDataset(cfg["oa_root"], split="val")
        train_datasets.append(oa_train)
        val_datasets.append(oa_val)
        print(f"  train: {len(oa_train):,}  val: {len(oa_val):,}")
 
    # ── Experto 4: LUNA16 ──
    if cfg.get("luna_patches") and cfg.get("luna_csv"):
        print("[Dataset] Cargando LUNA16 (parches 3D pre-extraídos)...")
        luna_train = LUNA16Dataset(cfg["luna_patches"] + "/train", cfg["luna_csv"])
        luna_val   = LUNA16Dataset(cfg["luna_patches"] + "/val",   cfg["luna_csv"])
        train_datasets.append(luna_train)
        val_datasets.append(luna_val)
        print(f"  train: {len(luna_train):,}  val: {len(luna_val):,}")
 
    # ── Experto 5: Pancreas ──
    if cfg.get("pancreas_patches_train") and cfg.get("pancreas_patches_val"):
        print("[Dataset] Cargando Pancreas CT 3D...")
        panc_train = PancreasDataset(cfg["pancreas_patches_train"])
        panc_val   = PancreasDataset(cfg["pancreas_patches_val"])
        train_datasets.append(panc_train)
        val_datasets.append(panc_val)
        print(f"  train: {len(panc_train):,}  val: {len(panc_val):,}")
 
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Dispositivo: {device}")
    if device == "cuda":
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
    # ── Cargar backbone congelado ──
    model, d_model = load_frozen_backbone(args.backbone, device)
 
    # ── Construir datasets ──
    cfg = {
        "chest_csv":        args.chest_csv,
        "chest_imgs":       args.chest_imgs,
        "chest_train_list": args.chest_train_list,
        "chest_val_list":   args.chest_val_list,
        "isic_gt":          args.isic_gt,
        "isic_imgs":        args.isic_imgs,
        "oa_root":          args.oa_root,
        "luna_patches":     args.luna_patches,
        "luna_csv":         args.luna_csv,
        "pancreas_patches_train": args.pancreas_patches_train,
        "pancreas_patches_val":   args.pancreas_patches_val,
    }
 
    train_dataset, val_dataset = build_datasets(cfg)
    print(f"\n[Dataset] Total train: {len(train_dataset):,}")
    print(f"[Dataset] Total val:   {len(val_dataset):,}")
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # NO mezclar — queremos reproducibilidad
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
 
    print("\n[FASE 0] Extrayendo embeddings de TRAIN...")
    Z_train, y_train, names_train = extract_embeddings(
        model, train_loader, device, d_model, desc="train"
    )
 
    print("\n[FASE 0] Extrayendo embeddings de VAL...")
    Z_val, y_val, names_val = extract_embeddings(
        model, val_loader, device, d_model, desc="val"
    )
 
    # ── Guardar en disco ──
    out = Path(args.output_dir)
    np.save(out / "Z_train.npy",     Z_train)
    np.save(out / "y_train.npy",     y_train)
    np.save(out / "Z_val.npy",       Z_val)
    np.save(out / "y_val.npy",       y_val)
 
    # Guardar nombres para auditoría (opcional pero útil)
    with open(out / "names_train.txt", "w") as f:
        f.write("\n".join(names_train))
    with open(out / "names_val.txt", "w") as f:
        f.write("\n".join(names_val))
 
    # ── Reporte final ──
    print(f"\n{'='*55}")
    print(f"FASE 0 completada")
    print(f"{'='*55}")
    print(f"Z_train:  {Z_train.shape}  ({Z_train.nbytes / 1e6:.1f} MB)")
    print(f"Z_val:    {Z_val.shape}  ({Z_val.nbytes / 1e6:.1f} MB)")
    print(f"\nDistribución de expertos en train (dominios clínicos 0–4):")
    for exp_name, exp_id in EXPERT_IDS.items():
        count = (y_train == exp_id).sum()
        pct   = count / len(y_train) * 100
        print(f"  Experto {exp_id} ({exp_name:<10}): {count:>6,}  ({pct:.1f}%)")
    print(f"\n  Experto 5 (ood       ):      0  (0.0%)  ← esperado: sin dataset en FASE 0")
    print(f"  [Experto 5 se inicializa en FASE 1 como MLP OOD — entrenado via L_error]")
    print(f"\n  N_EXPERTS_DOMAIN = {N_EXPERTS_DOMAIN}  (con dataset propio)")
    print(f"  N_EXPERTS_TOTAL  = {N_EXPERTS_TOTAL}  (incluye Experto 5 OOD)")
    print(f"\nArchivos guardados en: {args.output_dir}")
    print(f"Siguiente paso: python fase1_ablation_router.py --embeddings {args.output_dir}")


# ──────────────────────────────────────────────────────────
# 7. ARGPARSE
# ──────────────────────────────────────────────────────────
#Este bloque es el "Panel de Control" del pipeline. Utiliza la librería argparse para permitir que tú, como 
# investigador, definas los hiperparámetros (como batch_size) y las rutas de los datos (de los 5 datasets) 
# desde la consola, garantizando que el experimento sea flexible, documentado y fácil de replicar para tu 
# ablation study.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FASE 0 — Extracción de embeddings MoE")
 
    # Backbone
    parser.add_argument("--backbone",     default="vit_tiny_patch16_224",
                        choices=["vit_tiny_patch16_224", "cvt_13",
                                 "swin_tiny_patch4_window7_224"],
                        help="Backbone ViT a usar (proyecto permite ViT, CvT o Swin-Tiny)")
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
