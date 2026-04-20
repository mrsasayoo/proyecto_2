# exp1v21: ConvNeXt-V2-Base 384 + Single Head + Fisher + ASL + Multi-scale TTA
# Basado en exp1v20, aplicando diffs congelados de handoff 06 (sección D):
#   B5  ASL (gamma+=0, gamma-=4, prob_shift=0.05) + clip_grad_norm=0.5
#   B6  Backbone convnextv2_base.fcmae_ft_in22k_in1k_384
#   B7  drop_path_rate=0.1
#   B15 Multi-scale TTA (320, 384, 512) + hflip
#   B18 RandomResizedCrop(scale=(0.5, 1.0))
#   B19 ColorJitter(brightness=0.3, contrast=0.3)
#   B20 Logit adjustment post-hoc (tau=0.7) antes de threshold optimization
import os
import sys
_need_install = any(k in os.environ for k in ('COLAB_RELEASE_TAG', 'KAGGLE_DATA_PROXY_TOKEN'))
if _need_install:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'iterative-stratification', 'timm', 'kaggle',
                           'psutil', 'huggingface_hub',
                           'scikit-learn', 'seaborn',
                           'albumentations', 'opencv-python-headless'])
else:
    print("Deps: usando entorno pre-configurado (uv venv o sistema)")
import os, json, time, copy, random, math, signal, shutil, glob
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from timm.utils import ModelEmaV2

from sklearn.metrics import roc_auc_score, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

try:
    from huggingface_hub import upload_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import time as _time

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 67
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

IS_COLAB = 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ
IS_KAGGLE = os.path.exists('/kaggle/input') and not IS_COLAB
IS_VASTAI = os.path.exists('/workspace') and not IS_KAGGLE and not IS_COLAB
IS_CLUSTER = (
    not IS_COLAB and not IS_KAGGLE and not IS_VASTAI
    and 'uaodeepia' in os.path.expanduser('~')
)

WORKING_DIR = (
    '/kaggle/working' if IS_KAGGLE else
    '/content' if IS_COLAB else
    '/workspace' if IS_VASTAI else
    os.path.expanduser('~/martin-moe/outputs') if IS_CLUSTER else
    './output'
)
os.makedirs(WORKING_DIR, exist_ok=True)

# === CONFIGURACION exp1v21 ===
LP_EPOCHS = 10
FT_EPOCHS = 37
LP_LR = 1e-3
FT_HEAD_LR = 2e-4
FT_BACKBONE_LR = 3e-5
WEIGHT_DECAY = 0.05
PATIENCE = 5
EMA_DECAY = 0.9995

# Scheduler FT: CosineAnnealingLR monotónico (handoff 03/06)
COSINE_ETA_MIN = 1e-6

IMG_SIZE = 384
N_FOLDS = 3
RUN_FOLDS = 1            # single fold para esta iteracion

BATCH_SIZE = 32          # RTX 4090 24 GB: batch=48 causo OOM en FT (22.67 GB) 2026-04-18. GPU ya saturada con 32.
VAL_BATCH_SIZE = 96
ACCUM_STEPS = 1          # batch efectivo = 32 (sin acumulacion)
TTA_SCALES = (320, 384, 512)
LOGIT_ADJ_TAU = 0.7
LOGIT_TAU_SWEEP = (0.0, 0.3, 0.5, 0.7, 1.0)
CLASS_REPORT_EVERY = 5
TIMING_SYNC_CUDA = os.environ.get("TIMING_SYNC_CUDA", "0").strip().lower() in (
    "1", "true", "yes", "on"
)

# 6 clases (suma Pneumothorax respecto a v17 que tenia 5)
PATOLOGIAS = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax']
NUM_CLASSES = len(PATOLOGIAS)

# Submuestreo No Finding antes del split (decision confirmada 2026-04-17)
NO_FINDING_FRAC = 0.30
NO_FINDING_SAMPLE_SEED = 444

HF_REPO_ID = 'mitgar14/moe-medical-experts'
EXP_PREFIX = 'exp1v21'
MODEL_NAME = 'convnextv2_base.fcmae_ft_in22k_in1k_384'

_tmp_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
_data_cfg = timm.data.resolve_data_config({}, model=_tmp_model)
MODEL_MEAN = list(_data_cfg['mean'])
MODEL_STD = list(_data_cfg['std'])
MODEL_INTERPOLATION = _data_cfg['interpolation']
del _tmp_model, _data_cfg

env_name = (
    'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else
    'Vast.ai' if IS_VASTAI else 'Cluster UAO' if IS_CLUSTER else 'local'
)
print(f"=== exp1v21: ConvNeXt-V2-Base 384 + Single Head + Fisher + ASL + CosineLR ===")
print(f"Normalizacion via timm: mean={MODEL_MEAN}, std={MODEL_STD}")
n_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
mode_str = f"DP {n_gpus} GPUs" if n_gpus > 1 else "single GPU"
print(f"Device: {device} ({mode_str})")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, 'total_memory', 0)
        print(f"  GPU {i}: {props.name} ({mem / 1e9:.1f} GB)")
print(f"Entorno: {env_name}")
print(f"PyTorch: {torch.__version__}, timm: {timm.__version__}")
print(f"Patologias ({NUM_CLASSES}): {', '.join(PATOLOGIAS)}")
print(f"No Finding submuestreo: {NO_FINDING_FRAC:.2f} (seed={NO_FINDING_SAMPLE_SEED})")
print(f"LP: {LP_EPOCHS} ep, FT: {FT_EPOCHS} ep (patience={PATIENCE}, monitor=F1 macro)")
print(f"Loss: BCEWithLogitsLoss (rollback a v20 baseline; ASL causo collapse en LP)")
print(f"EMA: decay={EMA_DECAY}")
print(f"Scheduler FT: CosineAnnealingLR(T_max={FT_EPOCHS}, eta_min={COSINE_ETA_MIN})")
print(f"LR: LP={LP_LR}, FT_head={FT_HEAD_LR}, FT_backbone={FT_BACKBONE_LR}, wd={WEIGHT_DECAY}")
print(f"Modelo: {MODEL_NAME} + LSE(r=10) + Single Head")
eff_batch = BATCH_SIZE * ACCUM_STEPS
print(f"Batch: {BATCH_SIZE} x {ACCUM_STEPS} accum = {eff_batch} effective")
print(f"channels_last: ON | AMP: FP16 | fused AdamW: ON | Fisher iters: 1x len(dataset)")
print(f"TTA (OOF only): multi-scale={TTA_SCALES} + hflip | LP/FT val: scale={IMG_SIZE} single | logit_adjustment tau={LOGIT_ADJ_TAU}")
print(f"Timing sync CUDA (diagnóstico): {'ON' if TIMING_SYNC_CUDA else 'OFF'}")
print(f"Folds: {RUN_FOLDS}/{N_FOLDS} (fold 0 single)")


# === FAULT TOLERANCE (copia estable de v17) ===

class PreemptionHandler:
    def __init__(self):
        self.preempted = False
        if os.name != 'nt':
            signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        print(f"\n[SIGTERM] Guardando checkpoint de emergencia...")
        self.preempted = True

    @property
    def should_stop(self):
        return self.preempted


class NaNLossDetector:
    def __init__(self, max_streak=3):
        self.max_streak = max_streak
        self.streak = 0

    def check(self, loss_val):
        if math.isnan(loss_val) or math.isinf(loss_val):
            self.streak += 1
            print(f"[NaN] Loss={loss_val}. Streak: {self.streak}/{self.max_streak}")
            return self.streak < self.max_streak
        self.streak = 0
        return True


class DiskGuard:
    def __init__(self, directory, min_free_gb=5.0, keep_last_n=2):
        self.directory = directory
        self.min_free_gb = min_free_gb
        self.keep_last_n = keep_last_n

    def has_space(self):
        free_gb = shutil.disk_usage(self.directory).free / (1024**3)
        return free_gb >= self.min_free_gb

    def cleanup_old_checkpoints(self, pattern="ckpt_*.pt"):
        files = sorted(glob.glob(f"{self.directory}/{pattern}"),
                       key=os.path.getmtime)
        while len(files) > self.keep_last_n and not self.has_space():
            old = files.pop(0)
            os.remove(old)
            print(f"[DiskGuard] Eliminado {old}")


class Heartbeat:
    def __init__(self, path, interval_steps=50):
        self.path = path
        self.interval = interval_steps
        self.step_count = 0

    def beat(self, epoch, step, loss, extra=None):
        self.step_count += 1
        if self.step_count % self.interval != 0:
            return
        entry = {
            "ts": _time.time(), "epoch": epoch, "step": step,
            "loss": float(loss), "pid": os.getpid(),
        }
        if torch.cuda.is_available():
            entry["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        if extra:
            entry.update(extra)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def atomic_save(state_dict, path):
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


def upload_with_retry(local_path, repo_path, repo_id, token, max_retries=3):
    if not HF_AVAILABLE:
        return False
    for attempt in range(max_retries):
        try:
            upload_file(path_or_fileobj=local_path, path_in_repo=repo_path,
                        repo_id=repo_id, token=token)
            print(f"  HF: subido {repo_path}")
            return True
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"  HF retry {attempt+1}/{max_retries}: {e}. {wait}s...")
            _time.sleep(wait)
    print(f"  HF: fallo para {repo_path}")
    return False


preemption = PreemptionHandler()
nan_detector = NaNLossDetector(max_streak=3)
disk_guard = DiskGuard(WORKING_DIR, min_free_gb=5.0, keep_last_n=2)
heartbeat = Heartbeat(f"{WORKING_DIR}/heartbeat.jsonl", interval_steps=50)
print("Fault tolerance inicializado")


# === DATA LOADING ===

import subprocess as _sp

def _kaggle_download(dest):
    _sp.run(['kaggle', 'datasets', 'download', '-d', 'nih-chest-xrays/data',
             '-p', dest, '--unzip'], check=True)

if IS_COLAB:
    DATA_PATH = Path("/content/chestxray")
    if not list(DATA_PATH.glob("**/Data_Entry*.csv")):
        try:
            from google.colab import userdata
            os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
        except Exception:
            pass
        _kaggle_download("/content/chestxray")
elif IS_KAGGLE:
    DATA_PATH = Path("/kaggle/input/nih-chest-xrays/data")
elif IS_VASTAI:
    DATA_PATH = Path("/workspace/chestxray")
    if not list(DATA_PATH.glob("**/Data_Entry*.csv")):
        _kaggle_download("/workspace/chestxray")
elif IS_CLUSTER:
    DATA_PATH = Path(os.path.expanduser("~/martin-moe/data/cxr14_raw"))
else:
    DATA_PATH = Path("./data/chestxray")

print(f"DATA_PATH: {DATA_PATH}")
print(f"Existe: {DATA_PATH.exists()}")

POSSIBLE_ROOTS = [
    Path('/kaggle/input/data'),
    Path('/kaggle/input/nih-chest-xrays/data'),
    Path('/workspace/chestxray'),
    Path('/workspace/chestxray/data'),
    Path('/content/chestxray'),
    Path(os.path.expanduser('~/martin-moe/data/cxr14_raw')),
]

data_root = None
for p in POSSIBLE_ROOTS:
    if p.exists():
        data_root = p
        break

if data_root is None:
    raise FileNotFoundError("Dataset no encontrado en ninguna ruta conocida")

csv_candidates = list(data_root.glob('Data_Entry*.csv'))
if not csv_candidates:
    csv_candidates = list(data_root.parent.glob('Data_Entry*.csv'))
    if not csv_candidates:
        csv_candidates = list(data_root.rglob('Data_Entry*.csv'))
df = pd.read_csv(csv_candidates[0])
print(f"Registros CSV originales: {len(df)}")

# === SUBMUESTREO No Finding al 30% ANTES del split ===
# Justificacion (hoja-de-ruta-post-ferro-gabriel.md, seccion opcion b):
#   Submuestreo aleatorio del 30% de las filas No Finding antes del split
#   train/val/test. La reduccion se propaga al split, dejando val/test con
#   distribucion comparable a Gabriel (F1 macro 0,5155). La metodologia queda
#   documentada como limitacion en el writeup final.
n_nf_before = int((df['Finding Labels'] == 'No Finding').sum())
df_no_finding = df[df['Finding Labels'] == 'No Finding'].copy()
df_con_hallazgos = df[df['Finding Labels'] != 'No Finding'].copy()
df_no_finding = df_no_finding.sample(frac=NO_FINDING_FRAC, random_state=NO_FINDING_SAMPLE_SEED)
df = pd.concat([df_con_hallazgos, df_no_finding], ignore_index=True)
n_nf_after = int((df['Finding Labels'] == 'No Finding').sum())
print(f"No Finding antes: {n_nf_before} -> despues ({NO_FINDING_FRAC:.0%}): {n_nf_after}")
print(f"Total tras submuestreo: {len(df)} "
      f"(-{100*(1 - len(df) / (n_nf_before + len(df_con_hallazgos))):.1f}% vs original)")

for pat in PATOLOGIAS:
    df[pat] = df['Finding Labels'].apply(lambda x, p=pat: 1 if p in x.split('|') else 0)

print("Indexando imagenes...")
t0 = time.time()
all_paths = {}
for dp, _, fns in os.walk(str(data_root)):
    for f in fns:
        if f.endswith(('.png', '.jpg', '.jpeg')):
            all_paths[f] = os.path.join(dp, f)
print(f"  {len(all_paths)} imagenes en {time.time()-t0:.1f}s")

df['path'] = df['Image Index'].map(all_paths)
df = df.dropna(subset=['path']).reset_index(drop=True)
print(f"Imagenes con match: {len(df)}")

pos_counts = df[PATOLOGIAS].sum().values.astype(np.float32)
print(f"\nPrevalencia por clase (tras submuestreo):")
for i, pat in enumerate(PATOLOGIAS):
    print(f"  {pat:22s}: {int(pos_counts[i]):>6d} ({pos_counts[i]/len(df)*100:.2f}%)")
nf_pct = (df['Finding Labels'] == 'No Finding').mean() * 100
print(f"  {'No Finding':22s}: {int((df['Finding Labels'] == 'No Finding').sum()):>6d} ({nf_pct:.2f}%)")

# Split patient-level (evita leakage)
patient_df = df.groupby('Patient ID')[PATOLOGIAS].max().reset_index()
patient_labels = patient_df[PATOLOGIAS].values

mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pfm = {}
for fi, (_, vp) in enumerate(mskf.split(patient_df, patient_labels)):
    for pid in patient_df.iloc[vp]['Patient ID']:
        pfm[pid] = fi
df['fold'] = df['Patient ID'].map(pfm)
for fi in range(N_FOLDS):
    n = (df['fold'] == fi).sum()
    print(f"Fold {fi}: {n} imgs ({n/len(df)*100:.1f}%)")

del all_paths, patient_df, patient_labels, pfm


# === FISHER GEOMETRIC MEAN SAMPLER ===
# Fisher 2025 §6.3: w_i = exp(mean(log(n/count_c for c in active)))
# Si no hay clases activas (muestra No Finding), w_i = 1.0

def compute_fisher_weights(label_matrix):
    n_samples = len(label_matrix)
    freq = label_matrix.sum(axis=0).astype(np.float64)
    inv_freq = n_samples / np.maximum(freq, 1.0)

    weights = np.ones(n_samples, dtype=np.float64)
    for i in range(n_samples):
        active = np.where(label_matrix[i] > 0)[0]
        if len(active) > 0:
            log_w = np.log(inv_freq[active]).mean()
            weights[i] = np.exp(log_w)
    return weights


def log_sampler_stats(weights, name="sampler"):
    w = np.array(weights)
    print(f"  {name}: max_w={w.max():.2f}, min_w={w.min():.2f}, "
          f"mean_w={w.mean():.2f}, std_w={w.std():.2f}, "
          f"median_w={np.median(w):.2f}")
    # Threshold 50 alinea con Fisher geometric mean (v21 max_w~12) vs el umbral
    # historico 10 que daba falso positivo. v11 inestable tenia max_w=517 con
    # WeightedRandomSampler 1/freq (no geometric mean).
    if w.max() > 50:
        print(f"  [ALERTA] max_w > 50: riesgo de inestabilidad (exp1v11 tuvo max_w=517)")


# === PREPROCESSING ===

def multistage_resize(img, target):
    h, w = img.shape[:2]
    while h > target * 2 and w > target * 2:
        h, w = h // 2, w // 2
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if h != target or w != target:
        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    return img


# Augmentations adoptadas de v20 (F1=0.4830 probado). Las palancas B18/B19 originales
# de v21 (RandomResizedCrop(0.5-1.0) + ColorJitter(0.3)) causaron collapse en LP con
# head frozen (3 epochs con AUROC~0.5, run asl_ls 2026-04-18): el crop agresivo descarta
# patologias pequenas (Nodule/Mass) en ~50% de samples y ColorJitter sobre grayscale
# distorsiona CLAHE. Rot limit=10 + BrightContrast 0.2 son suaves y validados.
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
    ToTensorV2(),
])

val_transform_hflip = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
    ToTensorV2(),
])


def make_eval_transform(img_size: int, hflip: bool = False):
    ops = []
    if img_size != IMG_SIZE:
        ops.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC))
    if hflip:
        ops.append(A.HorizontalFlip(p=1.0))
    ops.extend([
        A.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
        ToTensorV2(),
    ])
    return A.Compose(ops)


# === OFFLINE CACHE (reusa pipeline v17) ===
# precompute_cache_v21.py genera *_cache384.jpg con CLAHE + multistage_resize 1024->384
# + grayscale->RGB aplicados. Mismo formato y nombre que v17 (cache compartible si
# el dataset ya fue precomputado; el setup de Vast.ai regenera por host nuevo).
CACHE_SUFFIX = f"_cache{IMG_SIZE}.jpg"


def _build_cache_on_miss(src_path, cache_path):
    g = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        g = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        g = multistage_resize(g, IMG_SIZE)
    img_rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(cache_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img_rgb


class ChestXray14DatasetCached(Dataset):
    """Lee cache JPEG pre-computado. Target multilabel 6 clases (single head)."""

    def __init__(self, dataframe, transform):
        self.paths = dataframe["path"].values
        self.labels = dataframe[PATOLOGIAS].values.astype(np.float32)
        self.transform = transform

    @staticmethod
    def cache_path_for(src):
        return src.rsplit(".", 1)[0] + CACHE_SUFFIX

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        src = self.paths[idx]
        cp = self.cache_path_for(src)
        img = cv2.imread(cp, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = _build_cache_on_miss(src, cp)
        img = self.transform(image=img)["image"]
        labels = torch.tensor(self.labels[idx])
        return img, labels


class CXRInferDataset(Dataset):
    """Dataset de inferencia para TTA multi-escala sobre cache precomputado."""

    def __init__(self, dataframe, img_size: int = IMG_SIZE, hflip: bool = False):
        self.paths = dataframe["path"].values
        self.labels = dataframe[PATOLOGIAS].values.astype(np.float32)
        self.transform = make_eval_transform(img_size=img_size, hflip=hflip)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        src = self.paths[idx]
        cp = ChestXray14DatasetCached.cache_path_for(src)
        img = cv2.imread(cp, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = _build_cache_on_miss(src, cp)
        img = self.transform(image=img)["image"]
        labels = torch.tensor(self.labels[idx])
        return img, labels


print("Probando pipeline con 384x384...")
_test_path = df['path'].iloc[0]
_img = cv2.imread(_test_path, cv2.IMREAD_GRAYSCALE)
print(f"  Original: {_img.shape}, dtype={_img.dtype}, range=[{_img.min()}, {_img.max()}]")
_clahe_test = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_img_c = _clahe_test.apply(_img)
_img_r = multistage_resize(_img_c, IMG_SIZE)
print(f"  CLAHE+Resize: {_img_r.shape}")
_img_rgb = cv2.cvtColor(_img_r, cv2.COLOR_GRAY2RGB)
_out = train_transform(image=_img_rgb)['image']
print(f"  Tensor: {_out.shape}, dtype={_out.dtype}")
del _img, _img_c, _img_r, _img_rgb, _out, _clahe_test


# === MODEL: Single Head ===

class LSEPool2d(nn.Module):
    """Log-Sum-Exp Pooling (Wang et al., CVPR 2017). r=10 optimo empirico CXR14."""
    def __init__(self, r=10.0):
        super().__init__()
        self.r = r

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        x_max = x_flat.max(dim=2, keepdim=True).values
        exp_sum = torch.exp(self.r * (x_flat - x_max)).mean(dim=2, keepdim=True)
        out = x_max + (1.0 / self.r) * torch.log(exp_sum + 1e-8)
        return out.view(b, c, 1, 1)


class CXRExpertSingleHead(nn.Module):
    """ConvNeXt-V2-Base 384 con single head (6 clases, ASL)."""
    def __init__(self, model_name, num_classes=6, drop_path=0.1):
        super().__init__()
        _tmp = timm.create_model(model_name, pretrained=False, features_only=True)
        last_idx = len(_tmp.feature_info) - 1
        del _tmp
        self.backbone = timm.create_model(
            model_name, pretrained=True,
            features_only=True, out_indices=(last_idx,),
            drop_path_rate=drop_path,
        )
        feat_dim = self.backbone.feature_info[-1]['num_chs']

        self.pool = LSEPool2d(r=10.0)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )
        self.feat_dim = feat_dim
        self._init_new_weights()

    def _init_new_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_head_bias_log_prior(self, prevalence):
        """Inicializa el bias de la sigmoide al logit de la prevalencia de train."""
        eps = 1e-3
        with torch.no_grad():
            final = self.head[-1]
            for i, p in enumerate(prevalence):
                p = float(np.clip(p, eps, 1.0 - eps))
                final.bias[i] = float(np.log(p / (1.0 - p)))

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.pool(feat).flatten(1)
        logits = self.head(feat)
        return logits


def create_model():
    return CXRExpertSingleHead(MODEL_NAME, NUM_CLASSES)


# === LP/FT HELPERS ===

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Backbone congelado: {trainable:,}/{total:,} params entrenables")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    t = sum(p.numel() for p in model.parameters())
    print(f"  Todo descongelado: {t:,} params entrenables")


class EarlyStopping:
    def __init__(self, patience, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.triggered = False

    def check(self, score):
        if self.best is None:
            self.best = score
            return
        improved = (score > self.best) if self.mode == 'max' \
            else (score < self.best)
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True


class AsymmetricLoss(nn.Module):
    """ASL para multilabel con negative probability shift y label smoothing.

    Label smoothing se aplica SOLO a los targets usados en log-loss (los_pos/los_neg),
    no al focusing weight ni a pt. Esto preserva la semantica original de ASL y evita
    que xs_pos saturate a 0 (collapse observado en v21 con head frozen + gn=4 + shift=0.05).
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps=1e-8, prob_shift=0.05,
                 label_smoothing=0.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps
        self.prob_shift = prob_shift
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = (xs_pos - self.prob_shift).clamp(min=0)

        if self.label_smoothing > 0:
            ls = self.label_smoothing
            targets_loss = targets * (1 - 2 * ls) + ls
        else:
            targets_loss = targets

        los_pos = targets_loss * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets_loss) * torch.log((1 - xs_neg).clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = (
                self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            )
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


# === HF UPLOAD ===

def _get_hf_token():
    token = None
    if IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            pass
    elif IS_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except Exception:
            pass
    if token is None:
        token = os.environ.get('HF_TOKEN')
    return token


def upload_to_hf(local_path, repo_path):
    token = _get_hf_token()
    if not token:
        return
    upload_with_retry(local_path, repo_path, HF_REPO_ID, token)


_m = create_model()
n_total = sum(p.numel() for p in _m.parameters())
n_head = sum(p.numel() for p in _m.head.parameters())
n_backbone = n_total - n_head
print(f"\nCXRExpertSingleHead: {n_total:,} params total")
print(f"  backbone: {n_backbone:,}")
print(f"  head: {n_head:,} ({NUM_CLASSES} clases, ASL)")
print(f"  pooling: LSE(r=10), 0 params")
del _m


# === EVALUATION ===

@torch.no_grad()
def get_logits(model, loader, criterion=None):
    model.eval()
    all_logits, all_labels = [], []
    running_loss = 0.0
    n_batches = 0
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labs = labs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(imgs)
        if criterion is not None:
            running_loss += criterion(out.float(), labs).item()
            n_batches += 1
        all_logits.append(out.float().cpu().numpy())
        all_labels.append(labs.cpu().numpy())
    mean_loss = running_loss / max(n_batches, 1) if criterion is not None else None
    return np.concatenate(all_logits), np.concatenate(all_labels), mean_loss


def build_single_val_loader(df_val, num_workers=4):
    """Val loader single-scale (IMG_SIZE, sin flip) para monitoreo rapido durante LP/FT.
    El multi-scale TTA queda reservado a la evaluacion OOF final (ver build_multiscale_val_loaders).
    """
    ds = CXRInferDataset(df_val, img_size=IMG_SIZE, hflip=False)
    return DataLoader(
        ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_multiscale_val_loaders(df_val, num_workers=4):
    loaders = {}
    for scale in TTA_SCALES:
        for flip in (False, True):
            ds = CXRInferDataset(df_val, img_size=scale, hflip=flip)
            batch_size = 8 if scale > IMG_SIZE else VAL_BATCH_SIZE
            loaders[(scale, flip)] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=4 if num_workers > 0 else None,
            )
    return loaders


def eval_with_multiscale_tta(model, val_loaders, criterion=None):
    all_logits = []
    labels_ref = None
    val_loss = None
    for key in sorted(val_loaders.keys()):
        logits_i, labels_i, loss_i = get_logits(model, val_loaders[key], criterion=criterion)
        all_logits.append(logits_i)
        if labels_ref is None:
            labels_ref = labels_i
        if key == (IMG_SIZE, False):
            val_loss = loss_i
    return np.mean(np.stack(all_logits, axis=0), axis=0), labels_ref, val_loss


def apply_logit_adjustment(logits, prevalence_train, tau=LOGIT_ADJ_TAU):
    log_prior = np.log(np.asarray(prevalence_train, dtype=np.float64) + 1e-12)
    return logits - tau * log_prior


def _optimize_thresholds(probs, labels):
    thresholds = np.arange(0.05, 0.951, 0.01)
    best_thr = np.full(NUM_CLASSES, 0.5)
    for i in range(NUM_CLASSES):
        bf, bt = 0.0, 0.5
        y_true = labels[:, i]
        y_prob = probs[:, i]
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > bf:
                bf, bt = f, t
        best_thr[i] = bt
    return best_thr


def compute_metrics(logits, labels, prevalence_train=None, tau=LOGIT_ADJ_TAU,
                    tau_candidates=None, timing_out=None):
    t_metrics_start = time.perf_counter()
    selected_tau = tau
    adjusted_logits = logits
    tau_results = []
    tau_sweep_elapsed = 0.0

    if prevalence_train is not None and tau_candidates:
        t_tau_sweep_start = time.perf_counter()
        best_f1 = -1.0
        for tau_c in tau_candidates:
            logits_c = apply_logit_adjustment(logits, prevalence_train, tau=tau_c)
            probs_c = 1.0 / (1.0 + np.exp(-logits_c))
            thr_c = _optimize_thresholds(probs_c, labels)
            preds_c = (probs_c >= thr_c).astype(int)
            f1_c = f1_score(labels, preds_c, average='macro', zero_division=0)
            tau_results.append((tau_c, f1_c))
            if f1_c > best_f1:
                best_f1 = f1_c
                selected_tau = tau_c
        adjusted_logits = apply_logit_adjustment(logits, prevalence_train, tau=selected_tau)
        tau_sweep_elapsed = time.perf_counter() - t_tau_sweep_start
    elif prevalence_train is not None:
        adjusted_logits = apply_logit_adjustment(logits, prevalence_train, tau=tau)

    probs = 1.0 / (1.0 + np.exp(-adjusted_logits))
    auroc = np.mean([roc_auc_score(labels[:, i], probs[:, i])
                     for i in range(NUM_CLASSES) if labels[:, i].sum() > 0])

    f1_05 = f1_score(labels, (probs >= 0.5).astype(int),
                     average='macro', zero_division=0)

    best_thr = _optimize_thresholds(probs, labels)
    preds_opt = (probs >= best_thr).astype(int)
    f1_opt = f1_score(labels, preds_opt, average='macro', zero_division=0)

    if timing_out is not None:
        timing_out["metric_post_s"] = time.perf_counter() - t_metrics_start
        timing_out["tau_sweep_s"] = tau_sweep_elapsed

    return auroc, f1_05, f1_opt, best_thr, selected_tau, tau_results


def format_epoch_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mins, secs = divmod(total, 60)
    return f"{mins}:{secs:02d}"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.1f}ms"


def _maybe_sync_cuda():
    if TIMING_SYNC_CUDA and torch.cuda.is_available():
        torch.cuda.synchronize()


def print_epoch_summary(phase_tag, epoch, total_epochs, elapsed, train_loss, val_loss,
                        f1_opt, auroc, lr_info, es_counter=None, es_patience=None):
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"\n{phase_tag} {epoch}/{total_epochs} ({format_epoch_duration(elapsed)})")
    print(f"LOSS: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
    print(f"EXPERTO (multi-etiqueta): F1@opt={f1_opt:.4f} | AUROC={auroc:.4f}")
    print(f"SISTEMA: {lr_info} | VRAM={vram:.1f} GB")
    if es_counter is not None and es_patience is not None:
        print(f"EARLY STOPPING: {es_counter}/{es_patience}")


def print_detailed_metrics(logits, labels, prevalence_train, epoch, phase, loss_val,
                           lr_info, selected_tau, tau_results=None, sampler_weights=None,
                           timing_stats=None):
    auroc, f1_05, f1_opt, best_thr, _, _ = compute_metrics(
        logits, labels, prevalence_train=prevalence_train, tau=selected_tau
    )
    logits_adj = apply_logit_adjustment(logits, prevalence_train, tau=selected_tau)
    probs = 1 / (1 + np.exp(-logits_adj))
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    print(f"\n=== Epoch {epoch} ({phase}) ===")
    print(f"F1@opt (macro): {f1_opt:.4f} | F1@0.5: {f1_05:.4f} | AUROC: {auroc:.4f}")
    print(f"Logit adjustment: tau={selected_tau:.2f}")
    if tau_results:
        txt = ", ".join([f"{t:.1f}:{f1v:.4f}" for t, f1v in tau_results])
        print(f"Tau sweep (F1@opt): {txt}")
    print(f"\nREPORTE POR CLASE (columnas explícitas):")
    print(f"  {'Clase':22s} | {'Prev':>5s} | {'F1':>6s} | {'Prec':>6s} | "
          f"{'Rec':>6s} | {'AUC':>6s} | {'Thr':>5s} | {'TP':>5s} | {'FP':>5s} | {'FN':>5s}")
    print(f"  {'-'*95}")
    for i, pat in enumerate(PATOLOGIAS):
        auc_i = roc_auc_score(labels[:, i], probs[:, i]) if labels[:, i].sum() > 0 else 0
        pred_i = (probs[:, i] >= best_thr[i]).astype(int)
        tp = int(((pred_i == 1) & (labels[:, i] == 1)).sum())
        fp = int(((pred_i == 1) & (labels[:, i] == 0)).sum())
        fn = int(((pred_i == 0) & (labels[:, i] == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1_i = 2 * prec * rec / max(prec + rec, 1e-8)
        prev = labels[:, i].mean() * 100
        print(f"  {pat:22s} | {prev:4.1f}% | {f1_i:.4f} | {prec:.4f} | "
              f"{rec:.4f} | {auc_i:.4f} | {best_thr[i]:.2f}  | {tp:>5d} | {fp:>5d} | {fn:>5d}")

    if sampler_weights is not None:
        w = np.array(sampler_weights)
        print(f"\nSampler stats: max_w={w.max():.2f}, min_w={w.min():.2f}, "
              f"mean_w={w.mean():.2f}")

    if timing_stats:
        print(
            "TIMING (extendido, host-side aprox): "
            f"data_time={_fmt_ms(timing_stats.get('data_time_mean_s', 0.0))} | "
            f"step_time={_fmt_ms(timing_stats.get('step_time_mean_s', 0.0))} | "
            f"val={format_epoch_duration(timing_stats.get('val_s', 0.0))} | "
            f"metric_post={_fmt_ms(timing_stats.get('metric_post_s', 0.0))} | "
            f"tau_sweep={_fmt_ms(timing_stats.get('tau_sweep_s', 0.0))}"
        )

    print(f"LR: {lr_info} | Loss: {loss_val:.4f} | VRAM: {vram:.1f} GB")
    return auroc, f1_05, f1_opt, best_thr


# === TRAINING LOOP ===

t_total = time.time()

oof_logits_all = {}
oof_labels_all = {}
fold_results = {}

for fold in range(RUN_FOLDS):
    if preemption.should_stop:
        break

    print(f"\n{'='*60}")
    print(f"FOLD {fold} (exp1v21: ConvNeXt-V2 + ASL + multi-scale TTA)")
    print(f"{'='*60}")
    t_fold = time.time()

    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    print(f"  train: {len(train_df)} | val: {len(val_df)}")

    # Rollback a BCE (pattern v20 probado F1=0.4830): ASL con head frozen + 6 clases
    # causo collapse determinista (val_loss=0.0219 exacto, AUROC~0.5) en 2 iteraciones.
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Datasets
    train_ds = ChestXray14DatasetCached(train_df, train_transform)
    # Val single-scale para monitoreo por epoch; multi-scale TTA se construye al final del fold.
    val_loader = build_single_val_loader(val_df, num_workers=4)

    # Fisher geometric mean sampler (solo usado en FT; LP usa shuffle standard para
    # evitar train/val mismatch con head frozen, confirmado en ronda 4 BCE run: LP con
    # Fisher sobrepredice minoritarias (Infiltration recall 0.93 precision 0.30))
    train_labels = train_df[PATOLOGIAS].values.astype(np.float32)
    fisher_weights = compute_fisher_weights(train_labels)
    log_sampler_stats(fisher_weights, "Fisher geometric mean")
    fisher_sampler = WeightedRandomSampler(
        weights=fisher_weights, num_samples=len(fisher_weights), replacement=True
    )

    # train_loader se instancia por fase (LP=shuffle, FT=Fisher) para evitar duplicar workers

    raw_model = create_model().to(device)
    # channels_last: ConvNeXt se beneficia +10-25% sobre Ampere/Ada.
    raw_model = raw_model.to(memory_format=torch.channels_last)

    # Log-prior bias init: prevalencias del split de train
    ml_prev = train_df[PATOLOGIAS].sum().values.astype(np.float32) / max(len(train_df), 1)
    raw_model.init_head_bias_log_prior(ml_prev)
    print(f"  Head bias init: prev=[{', '.join(f'{p:.4f}' for p in ml_prev)}]")
    selected_tau_fold = LOGIT_ADJ_TAU

    freeze_backbone(raw_model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model
    ema = ModelEmaV2(raw_model, decay=EMA_DECAY)

    best_f1 = 0.0
    best_path = None
    epoch_global = 0

    # ===================== LP: heads only (shuffle, sin Fisher) =====================
    print(f"\n  === LP ({LP_EPOCHS} ep, shuffle, head only) ===", flush=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    lp_params = [p for p in raw_model.parameters() if p.requires_grad]
    lp_opt = optim.AdamW(lp_params, lr=LP_LR, weight_decay=WEIGHT_DECAY, fused=True)
    lp_scaler = torch.cuda.amp.GradScaler()

    for ep in range(LP_EPOCHS):
        if preemption.should_stop:
            break
        model.train()
        running = 0.0
        t0 = time.time()
        lp_opt.zero_grad()
        n_batches = len(train_loader)
        processed_batches = 0
        data_time_sum = 0.0
        step_time_sum = 0.0
        _maybe_sync_cuda()
        iter_end = time.perf_counter()

        for step, (imgs, labs) in enumerate(train_loader):
            _maybe_sync_cuda()
            iter_start = time.perf_counter()
            data_time_sum += max(0.0, iter_start - iter_end)
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labs = labs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(imgs)
            loss = criterion(out.float(), labs)
            lp_scaler.scale(loss).backward()
            lp_scaler.unscale_(lp_opt)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=0.5)
            loss_value = float(loss.item())

            if not nan_detector.check(loss_value):
                print("  [WARN] NaN/Inf detectado en LP; se salta step y se conserva estado.")
                lp_opt.zero_grad(set_to_none=True)
                _maybe_sync_cuda()
                iter_end = time.perf_counter()
                step_time_sum += max(0.0, iter_end - iter_start)
                continue
            running += loss_value
            processed_batches += 1

            heartbeat.beat(ep, step, loss_value)

            lp_scaler.step(lp_opt)
            lp_scaler.update()
            lp_opt.zero_grad()
            ema.update(raw_model)
            _maybe_sync_cuda()
            iter_end = time.perf_counter()
            step_time_sum += max(0.0, iter_end - iter_start)

        avg_loss = running / max(processed_batches, 1)

        _maybe_sync_cuda()
        t_val_start = time.perf_counter()
        logits_v, labels_v, val_loss = get_logits(
            ema.module, val_loader, criterion=criterion
        )
        _maybe_sync_cuda()
        val_elapsed = time.perf_counter() - t_val_start
        tau_candidates = LOGIT_TAU_SWEEP if ((ep + 1) % CLASS_REPORT_EVERY == 0) else None
        metric_timing = {}
        auroc, f1_05, f1_opt, best_thr, selected_tau, tau_results = compute_metrics(
            logits_v,
            labels_v,
            prevalence_train=ml_prev,
            tau=selected_tau_fold,
            tau_candidates=tau_candidates,
            timing_out=metric_timing,
        )
        selected_tau_fold = selected_tau

        elapsed = time.time() - t0
        lr_lp = f"lr={lp_opt.param_groups[0]['lr']:.2e}"
        print_epoch_summary(
            phase_tag="LP",
            epoch=ep + 1,
            total_epochs=LP_EPOCHS,
            elapsed=elapsed,
            train_loss=avg_loss,
            val_loss=val_loss if val_loss is not None else avg_loss,
            f1_opt=f1_opt,
            auroc=auroc,
            lr_info=lr_lp,
            es_counter=0,
            es_patience=PATIENCE,
        )

        if (ep + 1) % CLASS_REPORT_EVERY == 0 or (ep + 1) == LP_EPOCHS:
            timing_stats = {
                "data_time_mean_s": data_time_sum / max(processed_batches, 1),
                "step_time_mean_s": step_time_sum / max(processed_batches, 1),
                "val_s": val_elapsed,
                "metric_post_s": metric_timing.get("metric_post_s", 0.0),
                "tau_sweep_s": metric_timing.get("tau_sweep_s", 0.0),
            }
            print_detailed_metrics(
                logits_v,
                labels_v,
                prevalence_train=ml_prev,
                epoch=ep + 1,
                phase="LP",
                loss_val=val_loss if val_loss is not None else avg_loss,
                lr_info=lr_lp,
                selected_tau=selected_tau_fold,
                tau_results=tau_results,
                sampler_weights=fisher_weights,
                timing_stats=timing_stats,
            )

        if f1_opt > best_f1:
            best_f1 = f1_opt
            best_path = f"{WORKING_DIR}/ckpt_{EXP_PREFIX}_f{fold}_best.pt"
            if disk_guard.has_space():
                atomic_save({
                    'model_state_dict': raw_model.state_dict(),
                    'ema_state_dict': ema.module.state_dict(),
                    'epoch': ep, 'phase': 'LP',
                    'best_val_metric': best_f1,
                }, best_path)
                print(f"    >> Mejor F1@opt: {best_f1:.4f}")

        epoch_global = ep + 1

    del lp_opt, lp_scaler, train_loader

    # ===================== FT: full unfreeze + CosineAnnealingLR + Fisher sampler =====================
    print(f"\n  === FT ({FT_EPOCHS} ep, patience={PATIENCE}, full unfreeze + CosineLR + Fisher) ===", flush=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=fisher_sampler,
        num_workers=8, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    unfreeze_all(raw_model)

    # Recargar best LP ckpt antes de FT: FT arranca desde el head optimo de LP,
    # no desde el head al final de LP (que pudo oscilar tras el mejor). Kumar 2022
    # LP-FT strict. Idempotente sobre backbone (frozen durante LP). Relevante si
    # LP tiene 15+ epochs donde F1 puede fluctuar.
    if best_path and os.path.exists(best_path):
        ckpt_lp = torch.load(best_path, map_location='cpu', weights_only=False)
        state_lp = ckpt_lp['model_state_dict']
        raw_model.load_state_dict(state_lp)
        raw_model = raw_model.to(device).to(memory_format=torch.channels_last)
        ema.module.load_state_dict(ckpt_lp.get('ema_state_dict', state_lp))
        print(f"  FT arranca desde LP best: ep {ckpt_lp.get('epoch', '?') + 1}, "
              f"F1={ckpt_lp.get('best_val_metric', 0):.4f}")

    # Param groups con LR diferenciado: head (aprende rapido) y backbone (aprende lento)
    head_params = list(raw_model.head.parameters())
    backbone_params = [p for p in raw_model.backbone.parameters() if p.requires_grad]
    ft_opt = optim.AdamW([
        {'params': head_params, 'lr': FT_HEAD_LR, 'name': 'head'},
        {'params': backbone_params, 'lr': FT_BACKBONE_LR, 'name': 'backbone'},
    ], weight_decay=WEIGHT_DECAY, fused=True)
    ft_sched = optim.lr_scheduler.CosineAnnealingLR(
        ft_opt, T_max=FT_EPOCHS, eta_min=COSINE_ETA_MIN
    )
    ft_scaler = torch.cuda.amp.GradScaler()
    es = EarlyStopping(patience=PATIENCE, mode='max')
    es.best = best_f1

    for ep in range(FT_EPOCHS):
        if preemption.should_stop:
            print(f"  [SIGTERM] Interrumpiendo FT en epoch {ep+1}")
            break

        model.train()
        running = 0.0
        t0 = time.time()
        ft_opt.zero_grad()
        n_batches = len(train_loader)
        processed_batches = 0
        data_time_sum = 0.0
        step_time_sum = 0.0
        _maybe_sync_cuda()
        iter_end = time.perf_counter()

        for step, (imgs, labs) in enumerate(train_loader):
            _maybe_sync_cuda()
            iter_start = time.perf_counter()
            data_time_sum += max(0.0, iter_start - iter_end)
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labs = labs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(imgs)
            loss = criterion(out.float(), labs)
            ft_scaler.scale(loss).backward()
            ft_scaler.unscale_(ft_opt)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=0.5)
            loss_value = float(loss.item())

            if not nan_detector.check(loss_value):
                if best_path and os.path.exists(best_path):
                    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
                    raw_model.load_state_dict(ckpt['model_state_dict'])
                    ema.module.load_state_dict(
                        ckpt.get('ema_state_dict', ckpt['model_state_dict']))
                    raw_model.to(device)
                    raw_model = raw_model.to(memory_format=torch.channels_last)
                nan_detector.streak = 0
                print("  [WARN] NaN/Inf en FT; se restaura mejor checkpoint y se corta época.")
                break
            running += loss_value
            processed_batches += 1

            heartbeat.beat(epoch_global + ep, step, loss_value)

            ft_scaler.step(ft_opt)
            ft_scaler.update()
            ft_opt.zero_grad()
            ema.update(raw_model)
            _maybe_sync_cuda()
            iter_end = time.perf_counter()
            step_time_sum += max(0.0, iter_end - iter_start)

        ft_sched.step()

        avg_loss = running / max(processed_batches, 1)

        _maybe_sync_cuda()
        t_val_start = time.perf_counter()
        logits_v, labels_v, val_loss = get_logits(
            ema.module, val_loader, criterion=criterion
        )
        _maybe_sync_cuda()
        val_elapsed = time.perf_counter() - t_val_start
        tau_candidates = LOGIT_TAU_SWEEP if ((ep + 1) % CLASS_REPORT_EVERY == 0) else None
        metric_timing = {}
        auroc, f1_05, f1_opt_v, _, selected_tau, tau_results = compute_metrics(
            logits_v,
            labels_v,
            prevalence_train=ml_prev,
            tau=selected_tau_fold,
            tau_candidates=tau_candidates,
            timing_out=metric_timing,
        )
        selected_tau_fold = selected_tau

        lr_head = ft_opt.param_groups[0]['lr']
        lr_bb = ft_opt.param_groups[1]['lr']
        elapsed = time.time() - t0
        lr_info = f"lr_head={lr_head:.2e} | lr_bb={lr_bb:.2e}"
        print_epoch_summary(
            phase_tag="FT",
            epoch=ep + 1,
            total_epochs=FT_EPOCHS,
            elapsed=elapsed,
            train_loss=avg_loss,
            val_loss=val_loss if val_loss is not None else avg_loss,
            f1_opt=f1_opt_v,
            auroc=auroc,
            lr_info=lr_info,
            es_counter=es.counter,
            es_patience=PATIENCE,
        )

        es.check(f1_opt_v)

        if (ep + 1) % CLASS_REPORT_EVERY == 0 or (ep + 1) == FT_EPOCHS or es.triggered:
            timing_stats = {
                "data_time_mean_s": data_time_sum / max(processed_batches, 1),
                "step_time_mean_s": step_time_sum / max(processed_batches, 1),
                "val_s": val_elapsed,
                "metric_post_s": metric_timing.get("metric_post_s", 0.0),
                "tau_sweep_s": metric_timing.get("tau_sweep_s", 0.0),
            }
            print_detailed_metrics(
                logits_v,
                labels_v,
                prevalence_train=ml_prev,
                epoch=epoch_global + ep + 1,
                phase="FT",
                loss_val=val_loss if val_loss is not None else avg_loss,
                lr_info=lr_info,
                selected_tau=selected_tau_fold,
                tau_results=tau_results,
                sampler_weights=fisher_weights,
                timing_stats=timing_stats,
            )

        if f1_opt_v > best_f1:
            best_f1 = f1_opt_v
            best_path = f"{WORKING_DIR}/ckpt_{EXP_PREFIX}_f{fold}_best.pt"
            if disk_guard.has_space():
                atomic_save({
                    'model_state_dict': raw_model.state_dict(),
                    'ema_state_dict': ema.module.state_dict(),
                    'optimizer_state_dict': ft_opt.state_dict(),
                    'epoch': epoch_global + ep, 'phase': 'FT',
                    'best_val_metric': best_f1,
                }, best_path)
                print(f"    >> Mejor F1@opt: {best_f1:.4f}")
            else:
                disk_guard.cleanup_old_checkpoints()

        if es.triggered:
            print(f"    >> Early stopping en FT {ep+1}")
            break

    epoch_global += ep + 1
    del ft_opt, ft_sched, ft_scaler

    # === UPLOAD checkpoint a HF Hub ===
    if best_path and os.path.exists(best_path):
        hf_ckpt_path = f"{EXP_PREFIX}/ckpt_{EXP_PREFIX}_f{fold}_best.pt"
        print(f"\n  [UPLOAD] Subiendo checkpoint fold {fold} a HF Hub...")
        upload_to_hf(best_path, hf_ckpt_path)
    else:
        print(f"\n  [WARN] No hay checkpoint para fold {fold}")

    # Evaluacion final del fold
    try:
        if best_path and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
            eval_model = create_model().to(device)
            eval_model = eval_model.to(memory_format=torch.channels_last)
            state = ckpt.get('ema_state_dict') or ckpt['model_state_dict']
            eval_model.load_state_dict(state)

            # Multi-scale TTA solo para OOF final (no durante LP/FT por coste).
            print(f"\n  [OOF] Construyendo multi-scale TTA val loaders ({TTA_SCALES} x hflip)...")
            val_loaders_oof = build_multiscale_val_loaders(val_df, num_workers=4)
            logits_oof, labels_oof, _ = eval_with_multiscale_tta(
                eval_model, val_loaders_oof, criterion=None
            )
            auroc_f, f1_05_f, f1_f, thr_f, tau_f, tau_results_f = compute_metrics(
                logits_oof,
                labels_oof,
                prevalence_train=ml_prev,
                tau=selected_tau_fold,
                tau_candidates=LOGIT_TAU_SWEEP,
            )
            logits_oof_adj = apply_logit_adjustment(logits_oof, ml_prev, tau=tau_f)

            oof_logits_all[fold] = logits_oof
            oof_labels_all[fold] = labels_oof
            fold_results[fold] = {
                'auroc': auroc_f, 'f1_opt': f1_f,
                'f1_05': f1_05_f, 'thresholds': thr_f.tolist(),
                'tau': tau_f,
                'tau_results': [(float(t), float(v)) for t, v in tau_results_f],
                'phase_best': ckpt.get('phase', 'unknown'),
                'best_epoch': ckpt.get('epoch', -1),
            }

            probs_f = 1 / (1 + np.exp(-logits_oof_adj))
            print(f"\n  Fold {fold} resultados finales:")
            print(f"  AUROC: {auroc_f:.4f} | F1@opt: {f1_f:.4f} | "
                  f"Best phase: {ckpt.get('phase', '?')} ep {ckpt.get('epoch', '?')}")
            print(f"  Tau seleccionado: {tau_f:.2f}")
            print(f"\n  REPORTE POR CLASE (columnas explícitas)")
            print(f"  {'Patología':22s} | {'thr':>5s} | {'F1@opt':>6s} "
                  f"| {'AUROC':>6s} | {'n_pos':>6s}")
            print(f"  {'-'*60}")
            for i, pat in enumerate(PATOLOGIAS):
                auc_i = roc_auc_score(labels_oof[:, i], probs_f[:, i]) \
                    if labels_oof[:, i].sum() > 0 else 0
                pred_opt = (probs_f[:, i] >= thr_f[i]).astype(int)
                f1_opt_i = f1_score(labels_oof[:, i], pred_opt, zero_division=0)
                n_pos = int(labels_oof[:, i].sum())
                print(f"  {pat:22s} | {thr_f[i]:5.3f} | {f1_opt_i:6.4f} "
                      f"| {auc_i:6.4f} | {n_pos:>6d}")

            thr_path = f"{WORKING_DIR}/{EXP_PREFIX}_f{fold}_thresholds.json"
            thr_dict = {pat: float(thr_f[i]) for i, pat in enumerate(PATOLOGIAS)}
            with open(thr_path, 'w') as f:
                json.dump(thr_dict, f, indent=2)
            upload_to_hf(thr_path, f"{EXP_PREFIX}/{EXP_PREFIX}_f{fold}_thresholds.json")

            # Export OOF artefactos para post-hoc local (calibracion + threshold opt)
            oof_probs_adj = probs_f.astype(np.float32)
            oof_logits_raw = logits_oof.astype(np.float32)
            oof_labels_f32 = labels_oof.astype(np.int8)
            np.save(f"{WORKING_DIR}/oof_val_fold{fold}_probs.npy", oof_probs_adj)
            np.save(f"{WORKING_DIR}/oof_val_fold{fold}_logits_raw.npy", oof_logits_raw)
            np.save(f"{WORKING_DIR}/oof_val_fold{fold}_labels.npy", oof_labels_f32)
            oof_meta = {
                "fold": fold,
                "tau_logit_adj": float(tau_f),
                "tau_sweep_results": [(float(t), float(v)) for t, v in tau_results_f],
                "thresholds_per_class": thr_dict,
                "prevalence_train": ml_prev.tolist() if hasattr(ml_prev, "tolist") else list(ml_prev),
                "patologias": PATOLOGIAS,
                "n_samples": int(oof_probs_adj.shape[0]),
                "model": MODEL_NAME,
                "exp_prefix": EXP_PREFIX,
            }
            with open(f"{WORKING_DIR}/oof_val_fold{fold}_meta.json", "w", encoding="utf-8") as fmeta:
                json.dump(oof_meta, fmeta, indent=2)
            upload_to_hf(f"{WORKING_DIR}/oof_val_fold{fold}_probs.npy",
                         f"{EXP_PREFIX}/oof_val_fold{fold}_probs.npy")
            upload_to_hf(f"{WORKING_DIR}/oof_val_fold{fold}_logits_raw.npy",
                         f"{EXP_PREFIX}/oof_val_fold{fold}_logits_raw.npy")
            upload_to_hf(f"{WORKING_DIR}/oof_val_fold{fold}_labels.npy",
                         f"{EXP_PREFIX}/oof_val_fold{fold}_labels.npy")
            upload_to_hf(f"{WORKING_DIR}/oof_val_fold{fold}_meta.json",
                         f"{EXP_PREFIX}/oof_val_fold{fold}_meta.json")
            print(f"  [OOF] Exportados probs/logits/labels/meta a HF Hub ({EXP_PREFIX}/oof_val_fold{fold}_*)")

            del eval_model, val_loaders_oof
    except Exception as e:
        print(f"  [ERROR] Evaluacion fold {fold} fallo: {e}")
        print(f"  El checkpoint ya fue subido a HF Hub.")

    fold_time = time.time() - t_fold
    print(f"\nFold {fold} completado en {fold_time/60:.1f} min "
          f"(best F1@opt: {best_f1:.4f})")

    del model, raw_model, ema
    del train_loader
    del train_ds
    del val_loader
    torch.cuda.empty_cache()


# === RESULTADOS FINALES ===

total_time = time.time() - t_total

print(f"\n{'='*60}")
print(f"RESULTADOS exp1v21 ({len(fold_results)} fold(s))")
print(f"{'='*60}")
for fi, r in fold_results.items():
    print(f"  Fold {fi}: AUROC={r['auroc']:.4f} | F1@opt={r['f1_opt']:.4f} "
          f"| phase={r['phase_best']} ep={r['best_epoch']}")

# Comparacion con runs anteriores
print(f"\n  Comparacion con runs anteriores:")
print(f"    Gabriel  (DenseNet-121 224 + subsample 30% + 6cls):    0,5155")
print(f"    exp1v16  (Tiny-224, ASL+LSE+ls, 5cls, 3f):             0,4263")
print(f"    exp1v17  (Tiny-384, DualHead+Fisher+cRT, 5cls, f0):    0,4316")
for fi, r in fold_results.items():
    print(f"    >> exp1v21 (V2-384, SingleHead+Fisher+ASL+MS-TTA,"
          f" 6cls+subsample, f{fi}): {r['f1_opt']:.4f}  << ESTE")

summary = {
    'experiment': EXP_PREFIX,
    'model': MODEL_NAME,
    'architecture': 'CXRExpertSingleHead (ConvNeXt-V2-Base 384)',
    'loss': 'AsymmetricLoss(gamma_pos=0,gamma_neg=4,prob_shift=0.05)',
    'n_classes': NUM_CLASSES,
    'classes': PATOLOGIAS,
    'no_finding_frac': NO_FINDING_FRAC,
    'no_finding_seed': NO_FINDING_SAMPLE_SEED,
    'lp_epochs': LP_EPOCHS,
    'ft_epochs': FT_EPOCHS,
    'patience': PATIENCE,
    'batch_size': BATCH_SIZE,
    'accum_steps': ACCUM_STEPS,
    'lr_lp': LP_LR,
    'lr_ft_head': FT_HEAD_LR,
    'lr_ft_backbone': FT_BACKBONE_LR,
    'weight_decay': WEIGHT_DECAY,
    'ema_decay': EMA_DECAY,
    'img_size': IMG_SIZE,
    'sampler': 'fisher_geometric_mean_1x',
    'scheduler_ft': f'CosineAnnealingLR(T_max={FT_EPOCHS}, eta_min={COSINE_ETA_MIN})',
    'tta_scales': list(TTA_SCALES),
    'logit_adjustment_tau_default': LOGIT_ADJ_TAU,
    'logit_adjustment_tau_sweep': list(LOGIT_TAU_SWEEP),
    'channels_last': True,
    'fused_adamw': True,
    'amp': 'fp16',
    'n_folds': len(fold_results),
    'fold_results': {str(k): v for k, v in fold_results.items()},
    'total_time_hours': total_time / 3600,
}
summary_path = f"{WORKING_DIR}/{EXP_PREFIX}_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
upload_to_hf(summary_path, f"{EXP_PREFIX}/{EXP_PREFIX}_summary.json")

print(f"\nTiempo total: {total_time/60:.1f} min ({total_time/3600:.1f}h)")
print("\n=== exp1v21 FINALIZADO ===")
