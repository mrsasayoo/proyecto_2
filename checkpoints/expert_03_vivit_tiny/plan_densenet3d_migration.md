# PROMPT — Migración a DenseNet3D + Corrección de 10 Problemas
## Notebook: Expert 3 / LUNA16 (Clasificación Binaria de Nódulos Pulmonares)

---

## CONTEXTO OBLIGATORIO — LEE ANTES DE TOCAR CUALQUIER CELDA

Se te entrega el notebook `luna16-online-aug.ipynb`. Es un notebook de Kaggle en Python
que entrena un modelo 3D sobre parches `.npy` de 64×64×64 voxeles del dataset LUNA16 para
clasificación binaria (nódulo = 1 / falso positivo = 0).

El notebook tiene **37 celdas** (índices 0–36, mix de `markdown` y `code`).

### Problemas diagnosticados que DEBES corregir

| # | Celda afectada | Problema | Fix |
|---|---|---|---|
| P1 | 20, 23, 24, 25, 26 | `rpn_reg = 0.0` exacto en 33 epochs; `rpn_cls ≈ 0.00001`; el RPN genera gradientes parásitos que degradan el backbone | Eliminar RPN por completo — cambiar arquitectura |
| P2 | 16, 17, 18 | Arquitectura Faster R-CNN diseñada para detección multi-objeto, usada para clasificación binaria de parche único | Reemplazar por DenseNet3D (spec abajo) |
| P3 | 4, 13 | Se usa `TRAIN_DIR` (10:1 neg:pos, oversampling efectivo = 1×) en lugar de `TRAIN_AUG_DIR` (2.48:1, 4× más positivos) | Cambiar a `TRAIN_AUG_DIR` + `TRAIN_AUG_MANIFEST` |
| P4 | 26 | Checkpoint guardado por `val_loss` mínima (epoch 13: F1=0.604) ignorando epoch 31 (F1=0.654) | `is_best` por `val_f1_macro` máximo |
| P5 | 5 | `SCHEDULER_T0=15` → reset destructivo en epoch 15: FP salta de 6 a 471 (78×), F1 cae de 0.567 a 0.459 | `SCHEDULER_T0 = 30` |
| P6 | 5 | `WEIGHT_DECAY=0.03` (valor para ViT-Base en ImageNet-21k, excesivo aquí) | `WEIGHT_DECAY = 0.01` |
| P7 | 5 | `FOCAL_ALPHA=0.85` → ratio de peso 5.67:1 insuficiente para compensar ratio de clases 10:1 | `FOCAL_ALPHA = 0.90` |
| P8 | 22, 25, 26 | `EarlyStopping` monitorea `val_loss` (métrica trivialmente minimizable con predicción todo-negativo) | Cambiar a modo `max` monitoreando `val_f1_macro` |
| P9 | 5 | `MIN_DELTA=0.001` demasiado grueso para F1 que oscila en décimas | `MIN_DELTA = 0.0005` |
| P10 | 5 | `FPN_CHANNELS` y `ROI_POOL_SIZE` son constantes del Faster R-CNN que ya no existen | Eliminar estas dos líneas |

---

## ESPECIFICACIÓN COMPLETA — DenseNet3D

### Parámetros de configuración
```
num_init_features = 32
growth_rate       = 32
block_config      = (6, 12, 24, 16)   # DenseNet-121 style, 3D
compression_rate  = 0.5
bn_size           = 4                  # bottleneck = bn_size × growth_rate
dropout_rate      = 0.2                # dentro de cada DenseLayer
fc_dropout        = DROPOUT_FC (= 0.4) # antes de la FC final
num_classes       = NUM_CLASSES (= 2)
```
**Parámetros esperados: ~11.14M** (dentro del rango 8–12M solicitado).
Si al instanciar el conteo real difiere de este rango, ajusta `growth_rate` a 28
(resultado ~8.6M). No cambies `block_config`.

### Código completo de la arquitectura — pégalo EXACTAMENTE en la celda 16

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer3D(nn.Module):
    """
    Capa densa 3D con bottleneck BN-ReLU-Conv1x1-BN-ReLU-Conv3x3x3.
    Implementa dense connectivity: la salida se concatena con la entrada.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, dropout_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, padding=1, bias=False)
        self.drop  = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        out = self.drop(out)
        return torch.cat([x, out], dim=1)


class _DenseBlock3D(nn.Module):
    """Bloque denso 3D: num_layers de _DenseLayer3D con concatenación progresiva."""
    def __init__(self, num_layers, num_input_features,
                 growth_rate, bn_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer3D(
                    num_input_features + i * growth_rate,
                    growth_rate, bn_size, dropout_rate,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition3D(nn.Sequential):
    """Capa de transición: BN-ReLU-Conv1x1-AvgPool3d(2). Reduce spatial y canales."""
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    """
    DenseNet-121 adaptado para clasificación binaria de parches 3D CT (LUNA16).

    Arquitectura base: DenseNet (Huang et al., CVPR 2017) — versión 3D.
    Configuración: DenseNet-121 style con convoluciones 3D.

    Entrada:  [B, 1, 64, 64, 64]  — parche CT monocanal, float32
    Salida:   [B, 2]               — logits binarios

    Reducción espacial:
      Stem        : 64 → 32  (stride-2 conv + maxpool)
      Transition1 : 32 → 16
      Transition2 : 16 → 8
      Transition3 : 8  → 4
      Block4+Pool :      → 1  (AdaptiveAvgPool3d)

    Parámetros: ~11.14M (growth_rate=32, block_config=(6,12,24,16))
    """

    def __init__(
        self,
        growth_rate       = 32,
        block_config      = (6, 12, 24, 16),
        num_init_features = 32,
        bn_size           = 4,
        dropout_rate      = 0.2,
        fc_dropout        = 0.4,
        num_classes       = 2,
        compression_rate  = 0.5,
    ):
        super().__init__()
        self.growth_rate       = growth_rate
        self.block_config      = block_config
        self.num_init_features = num_init_features

        # ── Stem ──────────────────────────────────────────────────────────────
        # Conv3d(1→init, 7×7×7, stride=2, pad=3) + BN + ReLU + MaxPool3d(3,2,1)
        # 64³ → 32³
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # ── Dense Blocks + Transitions ─────────────────────────────────────────
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(
                num_layers        = num_layers,
                num_input_features= num_features,
                growth_rate       = growth_rate,
                bn_size           = bn_size,
                dropout_rate      = dropout_rate,
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            if i < len(block_config) - 1:
                out_features = int(num_features * compression_rate)
                trans = _Transition3D(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features

        # ── Final BN ──────────────────────────────────────────────────────────
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(num_features, num_classes),
        )
        self._num_features = num_features

        # ── Inicialización de pesos ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward: [B, 1, 64, 64, 64] → [B, 2]"""
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


print('✅ DenseNet3D definido')
print(f'   Configuración: growth={32}, blocks=(6,12,24,16), init_features={32}')
print(f'   Parámetros esperados: ~11.14M')
```

---

## CAMBIOS CELDA POR CELDA

### Celda 0 — markdown (MODIFICAR)

Reemplaza el bloque de tabla con:
```markdown
# Expert 3 — DenseNet3D / LUNA16: Clasificación de Nódulos Pulmonares 3D

**Proyecto MoE Médico — Fase 2: Entrenamiento de Experto 3**

Este notebook entrena un modelo **DenseNet3D** (DenseNet-121 con convoluciones 3D)
para clasificación binaria de parches 3D de candidatos a nódulo pulmonar del dataset LUNA16.

| Propiedad | Valor |
|---|---|
| **Tarea** | Clasificación binaria: nódulo (1) vs falso positivo (0) |
| **Entrada** | Parches 64×64×64, float32, zero-centrado (global_mean=0.0992) |
| **Modelo** | DenseNet3D (growth_rate=32, blocks=(6,12,24,16), ~11.14M parámetros) |
| **Loss** | FocalLoss BCE (γ=2.0, α=0.90) con label smoothing=0.05 |
| **Optimizer** | AdamW (lr=3e-4, wd=0.01) |
| **Scheduler** | CosineAnnealingWarmRestarts (T_0=30, T_mult=2) |
| **Dataset train** | train_aug/ (ratio 2.48:1, 17,571 parches) + aug ONLINE |
| **Checkpoint** | Guardado por val_f1_macro máximo |
| **Early stopping** | 20 épocas sin mejora en val_f1_macro |
```

---

### Celda 4 — RUTAS (MODIFICAR)

Añade estas dos líneas inmediatamente después de `TRAIN_DIR`:
```python
TRAIN_AUG_DIR      = os.path.join(LUNA_ROOT, 'train_aug')
TRAIN_AUG_MANIFEST = os.path.join(MANIFEST_ROOT, 'train_aug', 'manifest.csv')
```

Actualiza el comentario del bloque de rutas para reflejar que `train_aug/` es la
fuente de entrenamiento recomendada.

En el bucle de verificación, añade `('train_aug/', TRAIN_AUG_DIR)` a la lista.

---

### Celda 5 — HIPERPARÁMETROS (MODIFICAR 5 valores + eliminar 2 líneas)

Cambios exactos de valores:
```python
WEIGHT_DECAY    = 0.01     # era 0.03  — P6
FOCAL_ALPHA     = 0.90     # era 0.85  — P7
SCHEDULER_T0    = 30       # era 15    — P5
MIN_DELTA       = 0.0005   # era 0.001 — P9
```

Elimina estas dos líneas (ya no existen en DenseNet3D):
```python
FPN_CHANNELS  = 128   # ELIMINAR
ROI_POOL_SIZE = 2     # ELIMINAR
```

---

### Celda 13 — BUILD DATASETS (MODIFICAR 1 línea)

Cambia ÚNICAMENTE la línea de `train_ds`:
```python
# ANTES:
train_ds = LUNAPatchDataset(TRAIN_DIR, split='train', manifest_path=TRAIN_MANIFEST)

# DESPUÉS:
train_ds = LUNAPatchDataset(TRAIN_AUG_DIR, split='train', manifest_path=TRAIN_AUG_MANIFEST)
```

Actualiza el comentario de encabezado de la celda:
```python
# Fuente: train_aug/ (2.48:1, aug ONLINE)  |  val/ y test/ sin aug
```

Actualiza el print final:
```python
print(f'  train : {len(train_ds):,} parches  {len(train_loader):,} batches  (train_aug/, aug ONLINE)')
```

---

### Celda 16 — MODELO (REEMPLAZAR COMPLETAMENTE)

Elimina todo el contenido actual (clases `FPN3D`, `RPN3DHead`, `roi_align_3d`,
`Expert3FasterRCNN3D`) y reemplaza con el código completo de `DenseNet3D`
especificado en la sección anterior de este prompt.

---

### Celda 17 — INSTANCIAR MODELO (REEMPLAZAR COMPLETAMENTE)

```python
# =============================================================================
# INSTANCIAR DenseNet3D
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')
if device.type == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

model = DenseNet3D(
    growth_rate       = 32,
    block_config      = (6, 12, 24, 16),
    num_init_features = 32,
    bn_size           = 4,
    dropout_rate      = 0.2,
    fc_dropout        = DROPOUT_FC,
    num_classes       = NUM_CLASSES,
    compression_rate  = 0.5,
).to(device)

if torch.cuda.device_count() > 1:
    print(f'  Usando {torch.cuda.device_count()} GPUs con DataParallel')
    model = nn.DataParallel(model)
_core = model.module if hasattr(model, 'module') else model

n_params = _core.count_parameters()
print(f'\nParámetros entrenables: {n_params:,}  ({n_params/1e6:.2f}M)')
print(f'Ratio params/datos    : {n_params / max(len(train_ds),1):.0f}:1')
assert 8_000_000 <= n_params <= 12_000_000, (
    f'Parámetros fuera del rango [8M, 12M]: {n_params:,}. '
    f'Ajustar growth_rate a 28 si > 12M, o a 36 si < 8M.'
)

# Verificación forward pass
model.eval()
with torch.no_grad():
    dummy  = torch.zeros(2, 1, 64, 64, 64, device=device)
    logits = _core(dummy)
    print(f'\nVerificación forward pass:')
    print(f'  input  : {list(dummy.shape)}')
    print(f'  logits : {list(logits.shape)}   ← clasificación binaria')
    assert logits.shape == (2, 2), f'Shape inesperado: {logits.shape}'
    print('  ✅ Verificación OK')
```

---

### Celda 18 — RESUMEN DE CAPAS (REEMPLAZAR COMPLETAMENTE)

```python
# --- Resumen de bloques del modelo DenseNet3D ---
print('Arquitectura DenseNet3D:')
print('=' * 60)
total_params     = sum(p.numel() for p in _core.parameters())
trainable_params = sum(p.numel() for p in _core.parameters() if p.requires_grad)

blocks_summary = {
    'stem (conv0+pool0)'  : sum(p.numel() for n, p in _core.named_parameters() if 'conv0' in n or 'norm0' in n),
    'denseblock1'         : sum(p.numel() for n, p in _core.named_parameters() if 'denseblock1' in n),
    'transition1'         : sum(p.numel() for n, p in _core.named_parameters() if 'transition1' in n),
    'denseblock2'         : sum(p.numel() for n, p in _core.named_parameters() if 'denseblock2' in n),
    'transition2'         : sum(p.numel() for n, p in _core.named_parameters() if 'transition2' in n),
    'denseblock3'         : sum(p.numel() for n, p in _core.named_parameters() if 'denseblock3' in n),
    'transition3'         : sum(p.numel() for n, p in _core.named_parameters() if 'transition3' in n),
    'denseblock4'         : sum(p.numel() for n, p in _core.named_parameters() if 'denseblock4' in n),
    'norm_final'          : sum(p.numel() for n, p in _core.named_parameters() if 'norm_final' in n),
    'classifier'          : sum(p.numel() for n, p in _core.named_parameters() if 'classifier' in n),
}
for name, n in blocks_summary.items():
    pct = n / total_params * 100
    print(f'  {name:<24s}: {n:>10,} params ({pct:5.1f}%)')
print(f'  {"TOTAL":<24s}: {total_params:>10,} params')
print(f'  Entrenables         : {trainable_params:>10,}')
```

---

### Celda 20 — LOSS (REEMPLAZAR COMPLETAMENTE)

Elimina `FocalLossBCE` y `FasterRCNN3DLoss`. Reemplaza con:

```python
# =============================================================================
# FOCAL LOSS — Clasificación binaria (Lin et al., ICCV 2017)
#
# FocalLoss es la ÚNICA pérdida del modelo DenseNet3D.
# No hay RPN, no hay bbox regression. Solo clasificación binaria.
# =============================================================================
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss binaria para clasificación de nódulos LUNA16.

    L = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Con label smoothing: labels {0,1} → {ls/2, 1 - ls/2}.

    Args:
        gamma          : exponente de modulación. Default: 2.0
        alpha          : peso de clase positiva (nódulo). Default: 0.90
        label_smoothing: suavizado de etiquetas. Default: 0.05
    """

    def __init__(self, gamma=2.0, alpha=0.90, label_smoothing=0.05):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha
        self.ls     = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits  : tensor [B, 2] — logits crudos (antes de softmax)
            targets : tensor [B]    — labels enteros {0, 1}
        Returns:
            scalar loss
        """
        targets_f = targets.float()
        # Label smoothing
        targets_s = targets_f * (1 - self.ls) + self.ls / 2.0
        bce   = F.binary_cross_entropy_with_logits(
            logits[:, 1], targets_s, reduction='none'
        )
        p_t   = torch.exp(-bce)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        loss  = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


print('✅ FocalLoss definida (γ=2.0, α=0.90, label_smoothing=0.05)')
```

---

### Celda 21 — GRADIENT CHECKPOINTING (REEMPLAZAR COMPLETAMENTE)

```python
# =============================================================================
# GRADIENT CHECKPOINTING — Reduce VRAM para volúmenes 3D
# Se aplica a denseblock2, denseblock3 y denseblock4 del DenseNet3D.
# =============================================================================
from torch.utils.checkpoint import checkpoint


def enable_gradient_checkpointing(core_model):
    """Activa gradient checkpointing en denseblock2-4 del DenseNet3D."""
    for block_name in ('denseblock2', 'denseblock3', 'denseblock4'):
        block = getattr(core_model.features, block_name, None)
        if block is None:
            print(f'  ⚠️  {block_name} no encontrado — saltando')
            continue
        orig = block.forward

        def make_ckpt(fwd):
            def ckpt_fwd(x):
                return checkpoint(fwd, x, use_reentrant=False)
            return ckpt_fwd

        block.forward = make_ckpt(orig)
    print('Gradient checkpointing HABILITADO en denseblock2, denseblock3, denseblock4')


if device.type == 'cuda':
    enable_gradient_checkpointing(_core)
```

---

### Celda 22 — EARLY STOPPING (REEMPLAZAR COMPLETAMENTE)

```python
# =============================================================================
# EARLY STOPPING — modo configurable (min para loss, max para F1/AUC)
# =============================================================================


class EarlyStopping:
    """
    Early stopping con soporte para minimización (val_loss) y maximización (val_f1).

    Args:
        patience   : épocas sin mejora antes de detener
        min_delta  : umbral mínimo para considerar una mejora
        mode       : 'min' (menor es mejor) o 'max' (mayor es mejor)
    """

    def __init__(self, patience, min_delta=0.0005, mode='max'):
        assert mode in ('min', 'max'), f"mode debe ser 'min' o 'max', recibido: {mode}"
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter    = 0
        self.should_stop = False

    def step(self, score):
        """Retorna True si se debe detener el entrenamiento."""
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False
```

---

### Celda 23 — TRAIN / VALIDATE (REEMPLAZAR COMPLETAMENTE)

```python
from torch.amp import GradScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import time


def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, accum_steps, use_fp16):
    """
    Entrena una época con gradient accumulation y AMP (FP16).

    Returns:
        train_loss (float) — loss promedio de la época
    """
    model.train()
    total_loss = 0.0
    n_steps    = 0
    optimizer.zero_grad()

    for step, (volumes, labels, _fnames) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(volumes)           # [B, 2]
            loss   = criterion(logits, labels)
            loss   = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_steps    += 1

    # Flush gradientes residuales al final del epoch
    if n_steps % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_fp16):
    """
    Validación completa con cálculo de métricas.

    Returns:
        dict con val_loss, val_f1_macro, val_auc, val_acc,
              confusion_matrix, all_labels, all_probs, all_preds
    """
    model.eval()
    total_loss = 0.0
    n_steps    = 0
    all_labels, all_probs, all_preds = [], [], []

    for volumes, labels, _fnames in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(volumes)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        n_steps    += 1

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    la  = np.array(all_labels)
    pr  = np.array(all_probs)
    pd_ = np.array(all_preds)

    f1  = (f1_score(la, pd_, average='macro', zero_division=0)
           if len(np.unique(la)) >= 2 else 0.0)
    try:
        auc_val = roc_auc_score(la, pr)
    except Exception:
        auc_val = 0.0
    cm = confusion_matrix(la, pd_, labels=[0, 1])

    return {
        'val_loss'      : total_loss / max(n_steps, 1),
        'val_f1_macro'  : f1,
        'val_auc'       : auc_val,
        'val_acc'       : float((la == pd_).mean()),
        'confusion_matrix': cm.tolist(),
        'all_labels'    : la,
        'all_probs'     : pr,
        'all_preds'     : pd_,
    }
```

---

### Celda 24 — TTA (MODIFICAR — solo los forwards)

La lógica de los 8 flips se mantiene intacta. Únicamente cambia el forward:

En `predict_with_tta`: reemplaza el bloque `with torch.amp.autocast...` por:
```python
with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
    logits = _fwd(aug)              # DenseNet3D retorna [B, 2], no tupla
p = torch.softmax(logits, dim=1)[:, 1].item()
```

En `evaluate_with_tta`: reemplaza la línea de bboxes y el bucle interior:
```python
for volumes, labels, _fnames in loader:      # 3 valores, no 4
    for i in range(volumes.size(0)):
        vol_i = volumes[i:i+1].to(device)
        prob  = predict_with_tta(model, vol_i, device, use_fp16, core=core)
        all_probs.append(prob)
        all_labels.append(labels[i].item())
```

Actualiza el docstring para reflejar que el modelo es DenseNet3D.

Elimina el parámetro `core` de `predict_with_tta` y `evaluate_with_tta` ya que
DenseNet3D devuelve un tensor simple, no una tupla. Si existe DataParallel, usar
`model.module` directamente donde sea necesario.

---

### Celda 25 — SETUP DE ENTRENAMIENTO (REEMPLAZAR COMPLETAMENTE)

```python
# =============================================================================
# SETUP DE ENTRENAMIENTO — DenseNet3D
# =============================================================================

criterion = FocalLoss(
    gamma           = FOCAL_GAMMA,
    alpha           = FOCAL_ALPHA,
    label_smoothing = LABEL_SMOOTHING,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT, eta_min=SCHEDULER_ETA_MIN
)
use_fp16   = USE_FP16 and device.type == 'cuda'
scaler     = GradScaler(device=device.type, enabled=use_fp16)

# EarlyStopping monitorea val_f1_macro (mode='max') — P8
early_stop = EarlyStopping(
    patience  = EARLY_STOPPING_PATIENCE,
    min_delta = MIN_DELTA,
    mode      = 'max',
)

print(f'Loss       : FocalLoss(γ={FOCAL_GAMMA}, α={FOCAL_ALPHA}, ls={LABEL_SMOOTHING})')
print(f'Optimizer  : AdamW(lr={LR}, wd={WEIGHT_DECAY})')
print(f'Scheduler  : CosineAnnealingWarmRestarts(T0={SCHEDULER_T0}, T_mult={SCHEDULER_T_MULT})')
print(f'FP16       : {use_fp16}')
print(f'Batch eff. : {BATCH_SIZE}×{ACCUMULATION_STEPS}={BATCH_SIZE*ACCUMULATION_STEPS}')
print(f'EarlyStop  : patience={EARLY_STOPPING_PATIENCE}, min_delta={MIN_DELTA}, mode=max(F1)')
print(f'\n{"="*70}')
print(f'  ENTRENAMIENTO — DenseNet3D / LUNA16')
print(f'  Parámetros: {n_params:,}  |  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}')
print(f'{"="*70}')
```

---

### Celda 26 — LOOP DE ENTRENAMIENTO (REEMPLAZAR COMPLETAMENTE)

```python
# =============================================================================
# LOOP DE ENTRENAMIENTO — DenseNet3D
# Checkpoint por val_f1_macro máximo (no por val_loss) — P4
# EarlyStopping monitorea val_f1_macro                  — P8
# =============================================================================

best_val_f1  = 0.0      # P4: tracking por F1, no por loss
training_log = []

for epoch in range(MAX_EPOCHS):
    t0 = time.time()

    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer,
        scaler, device, ACCUMULATION_STEPS, use_fp16,
    )
    val_res  = validate(model, val_loader, criterion, device, use_fp16)
    scheduler.step()
    lr_now   = optimizer.param_groups[0]['lr']

    val_loss = val_res['val_loss']
    val_f1   = val_res['val_f1_macro']
    val_auc  = val_res['val_auc']
    val_acc  = val_res['val_acc']
    cm       = val_res['confusion_matrix']

    # P4: is_best por val_f1_macro máximo
    is_best = val_f1 > best_val_f1 + MIN_DELTA

    print(
        f'[Ep {epoch+1:3d}/{MAX_EPOCHS}] '
        f'trL={train_loss:.4f} | '
        f'vlL={val_loss:.4f} | '
        f'F1={val_f1:.4f} | '
        f'AUC={val_auc:.4f} | '
        f'lr={lr_now:.1e} | '
        f'{time.time()-t0:.1f}s'
        f'{" ★ BEST" if is_best else ""}',
        flush=True,
    )
    print(
        f'           CM: TN={cm[0][0]:>5} FP={cm[0][1]:>5} | '
        f'FN={cm[1][0]:>5} TP={cm[1][1]:>5}',
        flush=True,
    )

    if is_best:
        best_val_f1 = val_f1
        torch.save({
            'epoch'              : epoch + 1,
            'model_state_dict'   : _core.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss'           : val_loss,
            'val_f1'             : val_f1,
            'val_auc'            : val_auc,
            'config': {
                'arch'            : 'DenseNet3D',
                'growth_rate'     : 32,
                'block_config'    : (6, 12, 24, 16),
                'num_init_features': 32,
                'n_params'        : n_params,
                'focal_gamma'     : FOCAL_GAMMA,
                'focal_alpha'     : FOCAL_ALPHA,
                'seed'            : SEED,
            },
        }, BEST_MODEL_PATH)
        print(f'  → Checkpoint guardado: {BEST_MODEL_PATH}', flush=True)

    training_log.append({
        'epoch'         : epoch + 1,
        'train_loss'    : train_loss,
        'val_loss'      : val_loss,
        'val_f1_macro'  : val_f1,
        'val_auc'       : val_auc,
        'val_acc'       : val_acc,
        'confusion_matrix': cm,
        'lr'            : lr_now,
        'is_best'       : is_best,
    })
    with open(TRAINING_LOG_PATH, 'w') as f_:
        json.dump(training_log, f_, indent=2, default=str)

    if device.type == 'cuda' and (epoch == 0 or (epoch + 1) % 10 == 0):
        print(
            f'           VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB alloc | '
            f'{torch.cuda.memory_reserved()/1e9:.2f} GB reserved',
            flush=True,
        )

    # P8: early stopping por val_f1_macro
    if early_stop.step(val_f1):
        print(
            f'\n[EarlyStopping] Detenido en epoch {epoch+1}. '
            f'Mejor val_f1={best_val_f1:.4f}',
            flush=True,
        )
        break

print(f'\n{"="*70}', flush=True)
print(f'  ENTRENAMIENTO FINALIZADO — DenseNet3D / LUNA16', flush=True)
if training_log:
    best = max(training_log, key=lambda x: x['val_f1_macro'])
    print(f'  Mejor época (F1): {best["epoch"]} | '
          f'F1={best["val_f1_macro"]:.4f} | AUC={best["val_auc"]:.4f}', flush=True)
print(f'  Checkpoint: {BEST_MODEL_PATH}', flush=True)
print(f'{"="*70}', flush=True)
```

---

### Celda 27 — CURVAS (MODIFICAR)

Elimina las referencias a `rpn_cls` y `rpn_reg` en el training_log. El log ya no
tiene estas claves. Adapta la lectura para usar solo:
`train_loss`, `val_loss`, `val_f1_macro`, `val_auc`, `lr`.

Añade una línea vertical en el gráfico de F1 que marque la mejor época por F1:
```python
best_f1_idx = val_f1s.index(max(val_f1s))
ax.axvline(x=epochs_range[best_f1_idx], color='red', linestyle='--',
           alpha=0.6, label=f'Best F1 (epoch {epochs_range[best_f1_idx]})')
```

Actualiza el `suptitle`:
```python
plt.suptitle('Expert 3 (DenseNet3D / LUNA16) — Curvas de Entrenamiento', ...)
```

---

### Celda 29 — CARGAR MODELO (MODIFICAR)

El checkpoint ahora tiene `val_f1` en lugar de solo `val_loss` como referencia.
Actualiza el print de verificación:
```python
print(f"Modelo cargado de época {best_ckpt['epoch']} | "
      f"val_f1={best_ckpt['val_f1']:.4f} | val_loss={best_ckpt['val_loss']:.4f}")
```

Elimina el parámetro `core=_core` de la llamada a `validate`:
```python
test_results = validate(model, test_loader, criterion, device, use_fp16)
```

---

### Celda 34 — TTA EVALUACIÓN (MODIFICAR)

Elimina `core=_core` de la llamada a `evaluate_with_tta`:
```python
tta_results = evaluate_with_tta(model, test_loader, device, use_fp16)
```

---

### Celda 35 — SECCIÓN 7 markdown (REEMPLAZAR)

```markdown
## Sección 7 — Arquitectura y Preprocesado

### Arquitectura: DenseNet3D (DenseNet-121 con convoluciones 3D)

```
Input [B, 1, 64, 64, 64]
  ↓ Stem: Conv3d(1→32, 7³, stride=2) + BN + ReLU + MaxPool3d(3,2,1)
    → [B, 32, 16, 16, 16]
  ↓ DenseBlock1 (6 layers, growth=32)  → [B, 224, 16, 16, 16]
  ↓ Transition1 (compression=0.5)      → [B, 112, 8, 8, 8]
  ↓ DenseBlock2 (12 layers, growth=32) → [B, 496, 8, 8, 8]
  ↓ Transition2 (compression=0.5)      → [B, 248, 4, 4, 4]
  ↓ DenseBlock3 (24 layers, growth=32) → [B, 1016, 4, 4, 4]
  ↓ Transition3 (compression=0.5)      → [B, 508, 2, 2, 2]
  ↓ DenseBlock4 (16 layers, growth=32) → [B, 1020, 2, 2, 2]
  ↓ BN + ReLU + AdaptiveAvgPool3d(1)   → [B, 1020]
  ↓ Dropout(0.4) + Linear(1020, 2)
  → logits [B, 2]
```

**Loss:** `FocalLoss(γ=2.0, α=0.90)` — única loss, sin RPN ni bbox regression.
**Parámetros:** ~11.14M
**Checkpoint:** guardado por `val_f1_macro` máximo.

### Correcciones aplicadas vs versión anterior (Faster R-CNN 3D)

| Problema | Fix aplicado |
|---|---|
| RPN losses = 0 exacto (gradientes parásitos) | Arquitectura eliminada completa |
| Faster R-CNN para clasificación binaria | Reemplazado por DenseNet3D |
| Dataset train/ (10:1 sin oversampling efectivo) | Cambiado a train_aug/ (2.48:1) |
| Checkpoint por val_loss (min) | Cambiado a val_f1_macro (max) |
| SCHEDULER_T0=15 (reset destructivo en epoch 15) | T0=30 |
| WEIGHT_DECAY=0.03 (excesivo) | 0.01 |
| FOCAL_ALPHA=0.85 (insuficiente para 10:1) | 0.90 |
| EarlyStopping por val_loss | Cambiado a val_f1_macro (mode=max) |
| MIN_DELTA=0.001 (grueso para F1) | 0.0005 |
| FPN_CHANNELS, ROI_POOL_SIZE (variables huérfanas) | Eliminadas |
```

---

## CELDAS QUE NO SE DEBEN TOCAR

Las siguientes celdas deben quedar **exactamente igual** que en el notebook original.
No las modifiques bajo ninguna circunstancia:

`1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 14, 15, 19, 28, 30, 31, 32, 33, 36`

En particular:
- **Celda 12**: `LUNAPatchDataset` con las 7 augmentaciones online — NO TOCAR.
- **Celda 19**: separador markdown de Sección 5 — NO TOCAR.
- **Celdas 30–33**: clasificación, matriz de confusión, ROC/PR, threshold — NO TOCAR
  (funcionan con el mismo formato de `test_results` que devuelve `validate`).

---

## RESTRICCIONES ABSOLUTAS

1. **No cambies el formato de retorno de `validate`**. Las celdas 29–33 dependen de
   que `test_results` tenga exactamente las claves:
   `val_loss`, `val_f1_macro`, `val_auc`, `val_acc`, `all_labels`, `all_probs`, `all_preds`.

2. **No cambies `LUNAPatchDataset` (celda 12)**. El dataset retorna tuplas de 3 elementos
   `(volume_tensor, label, fname)`. El training loop ya usa este formato.

3. **El assert de parámetros en celda 17 es obligatorio**:
   ```python
   assert 8_000_000 <= n_params <= 12_000_000
   ```
   Si el conteo real cae fuera del rango, ajusta `growth_rate` antes de continuar.

4. **No importes librerías nuevas**. Todo lo necesario (`torch`, `torch.nn`, `scipy`,
   `numpy`, `sklearn`) ya está importado en celdas anteriores.

5. **El notebook debe ser ejecutable de arriba a abajo sin errores** con las rutas
   de Kaggle correctas.

---

## CHECKLIST DE VERIFICACIÓN FINAL

Antes de entregar el notebook modificado, verifica cada uno de estos puntos:

- [ ] Celda 5: `WEIGHT_DECAY=0.01`, `FOCAL_ALPHA=0.90`, `SCHEDULER_T0=30`, `MIN_DELTA=0.0005`
- [ ] Celda 5: `FPN_CHANNELS` y `ROI_POOL_SIZE` NO aparecen
- [ ] Celda 13: `train_ds` usa `TRAIN_AUG_DIR` y `TRAIN_AUG_MANIFEST`
- [ ] Celda 16: contiene exactamente las clases `_DenseLayer3D`, `_DenseBlock3D`, `_Transition3D`, `DenseNet3D` y NO contiene `FPN3D`, `RPN3DHead`, `roi_align_3d`, `Expert3FasterRCNN3D`
- [ ] Celda 17: `assert 8_000_000 <= n_params <= 12_000_000` presente y activo
- [ ] Celda 17: forward devuelve tensor `[B, 2]`, no tupla
- [ ] Celda 20: contiene exactamente la clase `FocalLoss` y NO contiene `FocalLossBCE`, `FasterRCNN3DLoss`
- [ ] Celda 21: gradient checkpointing en `denseblock2`, `denseblock3`, `denseblock4`
- [ ] Celda 22: `EarlyStopping.__init__` tiene parámetro `mode` con default `'max'`
- [ ] Celda 23: `train_one_epoch` NO tiene parámetros `core`, `bboxes`, `rpn_outs`
- [ ] Celda 23: `validate` NO tiene parámetros `core`, `bboxes`, `rpn_outs`
- [ ] Celda 25: `EarlyStopping` instanciado con `mode='max'`
- [ ] Celda 26: `is_best = val_f1 > best_val_f1 + MIN_DELTA`
- [ ] Celda 26: `early_stop.step(val_f1)` (no `val_loss`)
- [ ] Celda 26: training_log NO tiene claves `rpn_cls`, `rpn_reg`, `det_cls`
- [ ] Celda 29: `validate(model, test_loader, criterion, device, use_fp16)` sin `core=`
- [ ] Celda 34: `evaluate_with_tta(model, test_loader, device, use_fp16)` sin `core=`
- [ ] Celdas 12, 30, 31, 32, 33, 36 idénticas al original

---

## EVIDENCIA DE LOS PROBLEMAS (para referencia)

Extraída del `training_log.json` del entrenamiento previo:

| Métrica | Valor | Interpretación |
|---|---|---|
| `rpn_reg` en 33 epochs | `0.000000` exacto siempre | RPN sin bounding boxes GT → nunca hay gradiente |
| `rpn_cls` rango | `1.2e-05` – `3.0e-04` | 1000× menor que `det_cls ≈ 0.030` → señal irrelevante |
| `det_cls` vs `train_loss` | Diferencia < 0.001 siempre | 100% de la señal viene de det_cls |
| Mejor epoch por `val_loss` | Epoch 13: F1=0.604, AUC=0.743 | Checkpoint guardado aquí |
| Mejor epoch por `val_f1` | Epoch 31: F1=0.654, AUC=0.747 | Nunca guardado con el criterio actual |
| Mejor epoch por `val_auc` | Epoch 23: F1=0.602, AUC=0.769 | Nunca guardado con el criterio actual |
| Epoch 15 → 16 (LR reset) | FP: 6 → 471 (+78×); F1: 0.567 → 0.459 | Reset destructivo por T0=15 |
