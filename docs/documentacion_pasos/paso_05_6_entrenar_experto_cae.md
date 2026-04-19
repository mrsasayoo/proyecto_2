# Paso 5.6 — Entrenar Experto 5: CAE Multimodal (AutoEncoder 2D)

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-05 |
| **Alcance** | Entrenamiento del Experto 5 (CAE — Convolutional AutoEncoder) para detección OOD vía error de reconstrucción. Entrenado con los 5 datasets combinados. Genera checkpoint consumido por Fase 4 (ablation) y Fase 5 (fine-tuning global) |
| **Estado general** | ✅ **Script implementado y verificado con dry-run** — pendiente de ejecutar entrenamiento completo |

---

## 1. Objetivo

Entrenar un **Convolutional AutoEncoder 2D (ConvAutoEncoder)** para detección de distribución fuera de dominio (OOD) mediante error de reconstrucción. Este experto es el Experto 5 del sistema MoE (`expert_id=5`).

**El CAE NO es un experto de routing directo.** No clasifica patologías. Se activa únicamente cuando la entropía del gating vector `H(g)` supera el percentil 95 calculado sobre el set de validación. Su función es distinguir imágenes médicas válidas (in-distribution) de imágenes basura o defectuosas (OOD).

> **⚠️ Extensión verbal:** El CAE no está definido en `proyecto_moe.md`. Fue aprobado verbalmente por el profesor como extensión del sistema MoE de 5 a 6 expertos. La función de pérdida total incluye un tercer término `β·L_error` no contemplado en la guía original.

---

## 2. Dataset

| Campo | Valor |
|---|---|
| **Manifiesto** | `datasets/cae_splits.csv` — 162,611 filas |
| **Columnas** | `ruta_imagen`, `dataset_origen`, `split`, `expert_id`, `tipo_dato` |
| **Datasets origen** | NIH / ISIC / OA / LUNA / Pancreas (los 5 dominios) |
| **Tipos de dato** | `2d_image`, `3d_patch_npy`, `3d_volume_nifti` |

### 2.1 Distribución por split

| Split | Total | NIH | ISIC | LUNA | OA | Pancreas |
|---|---|---|---|---|---|---|
| **Train** | 130,002 | 88,999 | 20,409 | 14,728 | 3,814 | 2,052 |
| **Val** | 15,959 | — | — | — | — | — |

### 2.2 Comportamiento de `__getitem__`

```python
def __getitem__(self, idx) -> Tuple[Tensor, str]:
    # Returns: (tensor [3, 224, 224], path_str)
    # NO label — autoencoder (unsupervised)
```

- **Sin etiquetas:** el CAE es un autoencoder — la "etiqueta" es la propia imagen de entrada
- **Salida uniforme:** independientemente de la modalidad de origen, `__getitem__` siempre retorna un tensor `[3, 224, 224]`

### 2.3 Dispatch para datos 3D

Los datos 3D se convierten a slices 2D para el autoencoder:

| Tipo | Estrategia |
|---|---|
| **LUNA patches** (`.npy`) | Se extrae la **slice axial central** del parche 3D |
| **Páncreas NIfTI** (`.nii.gz`) | HU clip `[-100, 400]` → normalización → **slice central** del volumen |

En ambos casos, el slice 2D resultante se replica a 3 canales y se redimensiona a `224×224` para mantener uniformidad con las imágenes 2D nativas (NIH, ISIC, OA).

---

## 3. Arquitectura del modelo

```
imagen [B, 3, 224, 224]
    │
    ▼
Encoder:
    Conv2d(3, 32, k=3, s=2, p=1) → BN2d → ReLU   → [B, 32, 112, 112]
    Conv2d(32, 64, k=3, s=2, p=1) → BN2d → ReLU  → [B, 64, 56, 56]
    Conv2d(64, 128, k=3, s=2, p=1) → BN2d → ReLU → [B, 128, 28, 28]
    Flatten → [B, 200704]
    Linear(200704, 512)
    │
    ▼
Latent: z ∈ ℝ^512   [B, 512]
    │
    ▼
Decoder:
    Linear(512, 200704)
    Unflatten → [B, 128, 28, 28]
    ConvTranspose2d(128, 64, k=3, s=2, p=1, op=1) → BN2d → ReLU → [B, 64, 56, 56]
    ConvTranspose2d(64, 32, k=3, s=2, p=1, op=1) → BN2d → ReLU  → [B, 32, 112, 112]
    ConvTranspose2d(32, 3, k=3, s=2, p=1, op=1) → Sigmoid         → [B, 3, 224, 224]
    │
    ▼
reconstrucción [B, 3, 224, 224]
```

| Campo | Valor |
|---|---|
| **Clase** | `ConvAutoEncoder` (2D) |
| **Parámetros totales** | ~206M (206,464,771) |
| **Entrada** | `[B, 3, 224, 224]` — imagen multimodal normalizada, float32 |
| **Salida** | `(recon, z)` — reconstrucción `[B, 3, 224, 224]` + latente `[B, 512]` |
| **Latent dim** | 512 |
| **Activation final** | Sigmoid (salida en rango `[0, 1]`) |

### 3.1 Métodos del modelo

| Método | Firma | Descripción |
|---|---|---|
| `forward(x)` | `(Tensor) → (Tensor, Tensor)` | Retorna `(reconstrucción, z_latente)` |
| `reconstruction_error(x)` | `(Tensor) → Tensor` | Retorna MSE per-sample: `[B]` — usado en inferencia para decisión OOD |

### 3.2 Nota sobre el tamaño del modelo

La mayor parte de los ~206M parámetros se concentra en las capas lineales del bottleneck:
- `Linear(200704, 512)` → ~102.8M params (encoder)
- `Linear(512, 200704)` → ~102.8M params (decoder)

Las capas convolucionales aportan ~0.1M parámetros. Este diseño prioriza la capacidad de reconstrucción sobre la compresión del latente.

---

## 4. Configuración de entrenamiento

| Hiperparámetro | Valor | Constante en `expert5_cae_config.py` |
|---|---|---|
| **Optimizador** | AdamW | — |
| **Learning rate** | 1×10⁻³ | `CAE_LR` |
| **Weight decay** | 1×10⁻⁵ | `CAE_WEIGHT_DECAY` |
| **Batch size** | 32 | `CAE_BATCH_SIZE` |
| **Mixed precision (FP16)** | **No** (FP32 obligatorio) | `CAE_FP16 = False` |
| **Max épocas** | 100 | `CAE_MAX_EPOCHS` |
| **Early stopping patience** | 15 | `CAE_EARLY_STOPPING_PATIENCE` |
| **Early stopping monitor** | `val_mse` | — |
| **Lambda L1** | 0.1 | `CAE_LAMBDA_L1` |

### 4.1 FP32 obligatorio

A diferencia de los expertos de dominio (que usan FP16), el CAE **debe** entrenarse en FP32. La loss MSE en FP16 sufre inestabilidad numérica: cuando los errores de reconstrucción son pequeños (< 1e-4), FP16 pierde precisión en las diferencias y los gradientes se vuelven ruidosos o se saturan a cero. Esto impide la convergencia del autoencoder.

---

## 5. Criterio de pérdida

```
L_total = MSE(recon, x) + λ · L1(recon, x)
```

| Término | Peso | Función |
|---|---|---|
| **MSE** | 1.0 | Pérdida principal — penaliza errores cuadráticos, sensible a outliers |
| **L1** | λ = 0.1 | Regularizador — promueve reconstrucciones más nítidas (menos blurring) |

La combinación MSE + L1 produce reconstrucciones con mejor balance entre fidelidad global (MSE) y preservación de bordes (L1).

---

## 6. Scheduler

`ReduceLROnPlateau(factor=0.5, patience=5)`

- **factor=0.5:** reduce LR a la mitad cuando `val_mse` deja de mejorar
- **patience=5:** espera 5 épocas sin mejora antes de reducir
- Más conservador que CosineAnnealing, adecuado para un autoencoder que converge de forma más suave que un clasificador

---

## 7. Archivos clave

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase3/train_cae.py` | Script de entrenamiento completo: bucle de épocas, EarlyStopping, checkpointing, soporte `--dry-run` |
| `src/pipeline/fase3/expert5_cae_config.py` | Constantes de hiperparámetros (prefijo `CAE_`). Fuente de verdad |
| `src/pipeline/fase3/models/expert5_cae.py` | Modelo ConvAutoEncoder 2D con `forward()` y `reconstruction_error()` |
| `src/pipeline/fase3/dataloader_cae.py` | Función de construcción de dataloaders — retorna `(train_loader, val_loader)` |
| `src/pipeline/datasets/cae.py` | Dataset class multimodal con dispatch 3D→2D slice |

---

## 8. Checkpoint

| Campo | Valor |
|---|---|
| **Directorio** | `checkpoints/expert_05_cae/` |
| **Archivo** | `cae_best.pt` |
| **Ruta completa** | `checkpoints/expert_05_cae/cae_best.pt` |
| **Criterio de guardado** | Mejor `val_mse` durante el entrenamiento |

---

## 9. Cómo ejecutar

### 9.1 Dry-run (verificación sin datos)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase3/train_cae.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Importación de todos los módulos (config, modelo, dataloader)
2. Instanciación del modelo ConvAutoEncoder
3. Forward pass sintético con un batch de imágenes aleatorias `[2, 3, 224, 224]`
4. Verificación de la shape de salida (reconstrucción `(2, 3, 224, 224)` + latente `(2, 512)`)
5. Verificación de `reconstruction_error()` — retorna shape `(2,)`
6. Conteo de parámetros
7. Exit code 0 si todo pasa correctamente

### 9.2 Entrenamiento real

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase3/train_cae.py
```

---

## 10. Dry-run verificado

| Verificación | Resultado |
|---|---|
| Importaciones | ✅ OK |
| Forward pass del modelo | ✅ OK — recon shape `(2, 3, 224, 224)`, latent shape `(2, 512)` |
| `reconstruction_error()` | ✅ OK — shape `(2,)` |
| Conteo de parámetros | ✅ 206,464,771 (~206M) |
| Dataset | ✅ train=130,002 / val=15,959 samples |
| Dry-run exit code | ✅ 0 |
| Entrenamiento real | ⏳ Pendiente de ejecutar |

---

## 11. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **FP32 obligatorio** | MSE en FP16 sufre inestabilidad numérica con errores pequeños — **NO activar FP16** para el CAE |
| **Sin augmentaciones de oclusión** | Prohibido `RandomErasing`, `Cutout`, `CutMix` — destruirían la señal de reconstrucción |
| **expert_id=5 excluido de load balancing** | El CAE no participa en el cálculo `max(f_i)/min(f_i)` — solo los expertos 0–4 |
| **No routing directo** | El CAE se activa por entropía `H(g) > P95`, no por `argmax(g)` |
| **Extensión verbal** | No está en `proyecto_moe.md` — aprobado verbalmente por el profesor |

---

## 12. Notas

- **expert_id=5:** este experto se registra como `EXPERT_IDS["cae"] = 5` en el sistema MoE. Es el sexto experto (IDs 0–5).
- **Activación por entropía:** durante inferencia, si `H(g) > threshold` (percentil 95 sobre val set), la imagen se envía al CAE. El threshold se calibra en Fase 4 y se persiste en `entropy_threshold.pkl`.
- **No participa en load balancing:** la restricción `max(f_i)/min(f_i) ≤ 1.30` aplica solo a los expertos de dominio (0–4). El CAE está explícitamente excluido.
- **Riesgo del "router perezoso":** si el CAE reconstruye demasiado bien, el router podría delegar todo al Experto 5. Esto se previene con `L_error` en la función de pérdida total (`L_total = L_task + α·L_aux + β·L_error`).
- **Fase del pipeline:** Fase 3 — se entrena **después** de los 5 expertos de dominio (Fase 2). Durante el entrenamiento, el CAE no interactúa con el router; solo en inferencia.

---

*Documento generado el 2026-04-05. Fuentes: `src/pipeline/fase3/expert5_cae_config.py`, `models/expert5_cae.py`, `dataloader_cae.py`, `train_cae.py`, `src/pipeline/datasets/cae.py`, `arquitectura_documentacion.md` §4.*
