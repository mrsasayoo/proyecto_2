# Paso 4.1 — Entrenar Backbones End-to-End

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-05 |
| **Alcance** | Entrenamiento end-to-end de los 4 backbones sobre los 5 datasets de dominio, usando clasificación de dominio como tarea proxy. Genera checkpoints consumidos por Paso 4.2 |
| **Estado general** | ✅ **Paso 4.1 IMPLEMENTADO — módulos `backbone_trainer.py` y `fase1_train_pipeline.py` creados, constantes en `fase1_config.py` añadidas, `backbone_loader.py` extendido con `load_trainable_backbone()` y `load_frozen_backbone_from_checkpoint()`.** |

---

## 1. Objetivo

Entrenar cada backbone **desde cero** (sin pesos preentrenados) usando una **tarea proxy de clasificación de dominio médico** (5 clases, una por dataset/experto). El objetivo no es obtener un clasificador de dominio perfecto, sino que el backbone aprenda representaciones visuales generales de imágenes médicas antes de congelarlo para la extracción de embeddings (Paso 4.2).

El entrenamiento desde cero es un requisito explícito del proyecto: no se permite usar pesos de ImageNet, HuggingFace, timm u otra fuente externa. La resolución de INC-P4-01 (2026-04-05) confirmó esta restricción.

---

## 2. Qué hace este paso en el pipeline

### 2.1 ¿Por qué clasificación de dominio como tarea proxy?

El sistema MoE necesita que los CLS tokens del backbone codifiquen diferencias semánticas entre las 5 modalidades médicas. Un backbone con pesos aleatorios produce embeddings cuasi-aleatorios que no permiten distinguir dominios (ver INC-P4-01). La tarea proxy fuerza al backbone a aprender representaciones que separan los 5 dominios — exactamente lo que el router necesita downstream.

La etiqueta de entrenamiento es el **expert_id** (0–4), asignado por el dataset de origen:

| expert_id | Dataset | Modalidad |
|---|---|---|
| 0 | NIH ChestXray14 | Rayos X tórax |
| 1 | ISIC 2019 | Dermatoscopía |
| 2 | OsteoArthritis Knee | Rayos X rodilla |
| 3 | LUNA16 | CT pulmonar (parches 3D→2D) |
| 4 | Pancreas PDAC | CT abdominal (volumen 3D→2D) |

### 2.2 ¿Qué se entrena y qué se descarta?

Se entrena un sistema `backbone + LinearHead(d_model → 5)` con `CrossEntropyLoss`. Al finalizar, **solo se guardan los pesos del backbone**. La cabeza lineal se descarta — su único propósito fue proveer la señal de gradiente durante el entrenamiento.

---

## 3. Inputs

### 3.1 Datasets

Los 5 datasets de dominio, construidos por `dataset_builder.build_datasets()`, idénticos a los usados en Paso 4.2. Todos se instancian con `mode="embedding"` (sin augmentaciones, deterministas).

| # | Dataset | Train | Val | Test |
|---|---------|-------|-----|------|
| 0 | NIH ChestXray14 | 88,999 | 11,349 | 11,772 |
| 1 | ISIC 2019 | 20,409 | 2,474 | 2,448 |
| 2 | OA Knee | 3,814 | 480 | 472 |
| 3 | LUNA16 | 14,728 | 1,143 | 1,914 |
| 4 | Pancreas | ~1,342* | ~336* | 186 |

> \* Pancreas usa k-fold CV (k=5); train/val dependen del fold seleccionado (`PANCREAS_FOLD=1` por defecto en `fase1_config.py`).

### 3.2 Qué retornan los loaders

Cada batch del DataLoader devuelve una tupla `(imgs, expert_ids, names)`:

| Elemento | Tipo | Shape | Descripción |
|---|---|---|---|
| `imgs` | `torch.Tensor` float32 | `[B, 3, 224, 224]` | Imágenes preprocesadas (transforms base, sin augmentaciones) |
| `expert_ids` | `torch.Tensor` int64 | `[B]` | Etiqueta de dominio (0–4) — usada como label de clasificación |
| `names` | `list[str]` | `[B]` | Nombres de archivo (no usado en entrenamiento) |

### 3.3 Prerequisitos

| # | Prerrequisito | Detalle |
|---|---|---|
| 1 | Datos descargados (Paso 1) | 5/5 datasets descargados y verificados |
| 2 | Datos preparados (Paso 3) | Splits generados, transforms configurados |
| 3 | Parches LUNA16 extraídos | Directorio `luna_lung_cancer/patches/` con 17,785 parches |
| 4 | VRAM disponible | Mínimo 4 GB para `swin_tiny`, 8 GB recomendado |
| 5 | `timm` + `transformers` instalados | Para instanciar arquitecturas de backbone |

---

## 4. Outputs

### 4.1 Checkpoints

Cada backbone entrenado produce un único archivo `backbone.pth` en su subdirectorio canónico dentro de `checkpoints/`:

```
checkpoints/
├── backbone_01_vit_tiny/
│   ├── backbone.pth
│   └── fase1_train_report.md
├── backbone_02_cvt13/
│   ├── backbone.pth
│   └── fase1_train_report.md
├── backbone_03_swin_tiny/
│   ├── backbone.pth
│   └── fase1_train_report.md
└── backbone_04_densenet121/
    ├── backbone.pth
    └── fase1_train_report.md
```

El mapeo `backbone_name → subdirectorio` está definido en `fase1_config.py:BACKBONE_TO_CHECKPOINT_DIR`.

### 4.2 Estructura del checkpoint (`backbone.pth`)

Archivo PyTorch serializado con `torch.save()`. Contiene únicamente los pesos del backbone (sin la cabeza lineal):

```python
{
    "backbone_name": str,      # e.g. "vit_tiny_patch16_224"
    "epoch": int,              # época del mejor val_acc
    "val_acc": float,          # accuracy de clasificación de dominio en val
    "state_dict": OrderedDict, # pesos del backbone
}
```

### 4.3 Reporte de entrenamiento (`fase1_train_report.md`)

Generado automáticamente al finalizar el entrenamiento. Incluye:

- Fecha, backbone, dispositivo, tiempo total
- Mejor `val_acc` y época correspondiente
- Historia completa: `train_loss`, `train_acc`, `val_loss`, `val_acc` por época
- Estado de idempotencia (si el entrenamiento fue omitido)

---

## 5. Arquitectura de entrenamiento

### 5.1 Modelo

```
imgs [B, 3, 224, 224]
    │
    ▼
backbone(imgs)  →  z [B, d_model]     ← CLS token
    │
    ▼
LinearHead(z)   →  logits [B, 5]      ← se descarta post-entrenamiento
    │
    ▼
CrossEntropyLoss(logits, expert_ids)
```

### 5.2 LinearHead

Cabeza de clasificación lineal definida en `backbone_trainer.py`:

```python
class LinearHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        self.fc = nn.Linear(d_model, n_classes)
        # Xavier uniform + bias=0
```

- `n_classes = N_EXPERTS_DOMAIN = 5` (importado de `config.py` global)
- Inicialización: `xavier_uniform_` para pesos, zeros para bias
- Se descarta al guardar el checkpoint — no se persiste

### 5.3 Optimizador y Scheduler

| Componente | Configuración | Constante en `fase1_config.py` |
|---|---|---|
| **Optimizador** | AdamW | — |
| **Learning rate** | 3×10⁻⁴ | `TRAIN_LR` |
| **Weight decay** | 0.01 | `TRAIN_WEIGHT_DECAY` |
| **Scheduler** | CosineAnnealingLR | — |
| **T_max** | `epochs` (20) | `TRAIN_EPOCHS` |
| **eta_min** | `lr × 0.01` = 3×10⁻⁶ | — |
| **Warm-up** | Lineal, 2 épocas | `TRAIN_WARMUP_EPOCHS` |
| **Gradient clipping** | `clip_grad_norm_`, max_norm=1.0 | `TRAIN_GRAD_CLIP` |

### 5.4 Estrategia de warm-up

Durante las primeras `TRAIN_WARMUP_EPOCHS` épocas, el LR escala linealmente desde 0 hasta `TRAIN_LR`:

```
scale(epoch) = min(1.0, (epoch + 1) / warmup_epochs)
lr_efectivo = TRAIN_LR × scale
```

El scheduler `CosineAnnealingLR` solo se activa después del warm-up (`epoch >= warmup_epochs`).

### 5.5 Criterio de guardado del checkpoint

Se guarda el checkpoint cuando `val_acc` del epoch actual **supera** el mejor `val_acc` registrado hasta el momento (best model selection). Solo se persiste un checkpoint por backbone — el del mejor epoch.

---

## 6. Cómo ejecutar

### 6.1 Script principal

```bash
python src/pipeline/fase1/fase1_train_pipeline.py --backbone <nombre> [opciones]
```

### 6.2 Flags disponibles

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--backbone` | `str` | `vit_tiny_patch16_224` | Backbone a entrenar. Opciones: `vit_tiny_patch16_224`, `cvt_13`, `swin_tiny_patch4_window7_224`, `densenet121_custom` |
| `--epochs` | `int` | 20 | Número de épocas de entrenamiento |
| `--lr` | `float` | 3e-4 | Learning rate inicial |
| `--weight_decay` | `float` | 0.01 | Weight decay (L2) para AdamW |
| `--warmup_epochs` | `int` | 2 | Épocas de warm-up lineal del LR |
| `--batch_size` | `int` | 64 | Tamaño del batch |
| `--workers` | `int` | 4 | Número de workers para DataLoader |
| `--checkpoint_dir` | `str` | `./checkpoints` | Directorio base donde guardar checkpoints |
| `--force` | `flag` | `False` | Reentrenar aunque el checkpoint ya exista |
| `--dry-run` | `flag` | `False` | Verificar configuración + un forward pass sintético sin entrenar ni modificar disco |

Adicionalmente acepta todos los flags de rutas de datasets (idénticos a `fase1_pipeline.py`): `--chest_csv`, `--chest_imgs`, `--isic_train_csv`, `--oa_root`, `--luna_patches_dir`, `--pancreas_splits_csv`, etc.

### 6.3 Comandos de ejecución

```bash
# ── Dry-run: verificar configuración sin entrenar ──
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone vit_tiny_patch16_224 --dry-run

# ── Entrenar los 4 backbones (orden recomendado por VRAM ascendente) ──

# 1. ViT-Tiny (d_model=192, ~2 GB VRAM)
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone vit_tiny_patch16_224

# 2. CvT-13 (d_model=384, ~3 GB VRAM)
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone cvt_13

# 3. DenseNet-121 custom (d_model=1024, ~3 GB VRAM)
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone densenet121_custom

# 4. Swin-Tiny (d_model=768, ~4 GB VRAM)
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone swin_tiny_patch4_window7_224
```

```bash
# ── Forzar reentrenamiento de un backbone ──
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone vit_tiny_patch16_224 --force

# ── Entrenamiento con hiperparámetros custom ──
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone cvt_13 --epochs 30 --lr 1e-4 --warmup_epochs 3
```

### 6.4 Dry-run

El modo `--dry-run` ejecuta:

1. Detección de dispositivo (GPU/CPU)
2. Instanciación del backbone + cabeza lineal
3. Un forward pass sintético con un batch de 4 imágenes aleatorias
4. Impresión de configuración completa (parámetros, d_model, checkpoint path, etc.)
5. Verificación de que el checkpoint existe o se crearía

No entrena, no modifica disco, no construye datasets reales.

---

## 7. Idempotencia

El pipeline verifica si el checkpoint ya existe antes de entrenar:

```
Si checkpoint existe Y --force NO activo → SKIP (log + reporte de omisión)
Si checkpoint existe Y --force activo    → REENTRENAR (sobrescribir checkpoint)
Si checkpoint no existe                  → ENTRENAR normalmente
```

Cuando se omite por idempotencia, se genera igualmente un `fase1_train_report.md` con `Estado: OMITIDO (idempotencia)` para dejar registro.

La verificación usa `backbone_trainer.backbone_checkpoint_exists()` que simplemente comprueba si el archivo `backbone.pth` existe en la ruta canónica.

---

## 8. Restricciones del proyecto

| Restricción | Detalle |
|---|---|
| **Sin pesos preentrenados** | Todos los backbones se inicializan con `pretrained=False`. Prohibido usar pesos de timm, HuggingFace, torchvision u otra fuente externa. |
| **Sin oclusión** | Prohibido usar `RandomErasing`, `Cutout`, `CutMix`, o cualquier transformación que oculte partes de la imagen. |
| **Sin augmentaciones en entrenamiento del backbone** | Los datasets se instancian con `mode="embedding"` (transforms base únicamente, sin flips, rotaciones ni color jitter). La señal de entrenamiento proviene exclusivamente del gradiente de la tarea proxy. |
| **Fuente de verdad** | `proyecto_moe.md` es la referencia canónica del proyecto. |

---

## 9. Conexión con Paso 4.2 (Extracción de Embeddings)

El checkpoint generado por este paso es consumido automáticamente por `fase1_pipeline.py` (Paso 4.2):

1. `fase1_pipeline.py` busca el checkpoint en la ruta canónica `checkpoints/<subdir>/backbone.pth` (definida por `BACKBONE_TO_CHECKPOINT_DIR`).
2. Si el checkpoint existe, lo carga mediante `load_frozen_backbone_from_checkpoint()` — que reconstruye la arquitectura, carga los pesos del `state_dict`, y congela todos los parámetros (`.eval()` + `requires_grad_(False)`).
3. Si el checkpoint no existe, emite un **warning** y carga el backbone con pesos aleatorios (los embeddings resultantes no serán significativos para routing).
4. El flag `--checkpoint_path` en `fase1_pipeline.py` permite especificar una ruta distinta a la canónica.

**Flujo completo:**

```
Paso 4.1 (fase1_train_pipeline.py)
    │
    │  Entrena backbone end-to-end
    │  Guarda: checkpoints/<subdir>/backbone.pth
    │
    ▼
Paso 4.2 (fase1_pipeline.py)
    │
    │  Carga backbone.pth → congela → extrae CLS tokens
    │  Guarda: embeddings/<backbone>/Z_train.npy, Z_val.npy, Z_test.npy
    │
    ▼
Fase 2 (ablation study — routers estadísticos sobre embeddings)
```

---

## 10. Constantes relevantes de `fase1_config.py`

### 10.1 Hiperparámetros de entrenamiento (Paso 4.1)

| Constante | Valor | Descripción |
|---|---|---|
| `TRAIN_EPOCHS` | 20 | Épocas de entrenamiento |
| `TRAIN_LR` | 3×10⁻⁴ | Learning rate inicial (AdamW) |
| `TRAIN_WEIGHT_DECAY` | 0.01 | Regularización L2 |
| `TRAIN_WARMUP_EPOCHS` | 2 | Épocas de warm-up lineal |
| `TRAIN_BATCH_SIZE` | 64 | Tamaño del batch |
| `TRAIN_WORKERS` | 4 | Workers del DataLoader |
| `TRAIN_GRAD_CLIP` | 1.0 | Max norm para gradient clipping |

### 10.2 Mapeo de checkpoints

| Constante | Valor |
|---|---|
| `BACKBONE_TO_CHECKPOINT_DIR` | `{"vit_tiny_patch16_224": "backbone_01_vit_tiny", "cvt_13": "backbone_02_cvt13", "swin_tiny_patch4_window7_224": "backbone_03_swin_tiny", "densenet121_custom": "backbone_04_densenet121"}` |
| `BACKBONE_CHECKPOINT_FILENAME` | `"backbone.pth"` |

### 10.3 Backbones disponibles

| Backbone | `d_model` | VRAM estimada | Subdirectorio checkpoint |
|---|---|---|---|
| `vit_tiny_patch16_224` | 192 | ~2 GB | `backbone_01_vit_tiny` |
| `cvt_13` | 384 | ~3 GB | `backbone_02_cvt13` |
| `swin_tiny_patch4_window7_224` | 768 | ~4 GB | `backbone_03_swin_tiny` |
| `densenet121_custom` | 1024 | ~3 GB | `backbone_04_densenet121` |

> **Fuente de verdad:** `fase1_config.py:BACKBONE_CONFIGS` (líneas 20–25) y `BACKBONE_TO_CHECKPOINT_DIR` (líneas 88–93).

---

## 11. Módulos involucrados

| Módulo | Responsabilidad |
|---|---|
| `fase1_train_pipeline.py` | Orquestador CLI: argparse, guard clauses, construcción de DataLoaders, invocación del entrenamiento, reporte |
| `backbone_trainer.py` | Bucle de entrenamiento (`train_one_epoch`, `validate`, `train_backbone`), `LinearHead`, utilidades de checkpoint |
| `backbone_loader.py` | `load_trainable_backbone()` — construye backbone en modo `.train()` con `requires_grad=True` |
| `backbone_loader.py` | `load_frozen_backbone_from_checkpoint()` — carga checkpoint y congela (usado por Paso 4.2) |
| `fase1_config.py` | Constantes `TRAIN_*`, `BACKBONE_TO_CHECKPOINT_DIR`, `BACKBONE_CHECKPOINT_FILENAME` |
| `dataset_builder.py` | Construcción de los 5 datasets combinados (reutilizado de Paso 4.2) |

---

*Documento generado el 2026-04-05. Fuentes: `src/pipeline/fase1/fase1_train_pipeline.py`, `backbone_trainer.py`, `backbone_loader.py`, `fase1_config.py`, `proyecto_moe.md` §4.1, `docs/plans/2026-04-05-paso41-backbone-training.md`.*
