# Paso 5.5 — Entrenar Experto 4: Páncreas CT (ResNet 3D)

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-14 |
| **Alcance** | Entrenamiento del Experto 4 (clasificación binaria PDAC en CT abdominal 3D) con ResNet 3D desde cero. Genera checkpoint consumido por Fase 4 (ablation) y Fase 5 (fine-tuning global) |
| **Responsable** | Luz |
| **Estado general** | ✅ **Script implementado y verificado con dry-run** — pendiente de ejecutar entrenamiento completo |

---

## 1. Objetivo

Entrenar un modelo **ResNet 3D desde cero** (sin pesos preentrenados, `weights=None`) para clasificación **binaria** de volúmenes CT abdominales: PDAC+ (adenocarcinoma ductal de páncreas) / PDAC−. Este experto es el Experto 4 del sistema MoE (`expert_id=4`), responsable del dominio de CT abdominal 3D.

El dataset PANORAMA tiene un desbalance significativo (~16.3:1 PDAC+ vs. PDAC−), lo que requiere FocalLoss con alpha=0.75 y validación cruzada k-fold (k=5) obligatoria dado el volumen reducido de datos.

---

## 2. Dataset

| Campo | Valor |
|---|---|
| **Dataset** | PANORAMA Challenge (Zenodo 13715870, 13742336, 11034011, 10999754) |
| **Modalidad** | CT abdominal 3D (volúmenes NIfTI) |
| **Tarea** | Clasificación binaria de volumen (PDAC+ / PDAC−) |
| **Archivos en disco** | 557 `.nii.gz` en `datasets/zenodo_13715870/` |
| **Labels** | `datasets/pancreas_labels_binary.csv` — 1,864 filas (1,756 PDAC+, 108 PDAC−) |
| **Splits** | `datasets/pancreas_splits.csv` — k-fold CV (k=5) por `patient_id` (`GroupKFold`) |
| **Fold 0 (ejemplo)** | Train: 1,491 / Val: 373 samples |
| **Desbalance** | ~16.3:1 (PDAC+ vs. PDAC−) |
| **Mecanismo valid_pairs** | `_build_pairs()` omite silenciosamente los casos sin CT en disco (1,307/1,864), `log.warning` añadido |

### 2.1 Validación cruzada obligatoria

Dado el volumen reducido (~557 volúmenes disponibles en disco), se usa k-fold CV (k=5) obligatorio. Cada fold produce un checkpoint independiente. El rendimiento final se reporta como la media ± desviación estándar sobre los 5 folds.

---

## 3. Arquitectura del modelo

```
volumen CT abdominal [B, 1, 64, 64, 64]
    │
    ▼
ResNet 3D (weights=None, desde cero)
    │
    ├── conv1: Conv3d(1→64, k=7, s=2) → BN3d → ReLU → MaxPool3d
    ├── layer1: ResBlock3d × 2 → [B, 64, ...]
    ├── layer2: ResBlock3d × 2 → [B, 128, ...]
    ├── layer3: ResBlock3d × 2 → [B, 256, ...]
    ├── layer4: ResBlock3d × 2 → [B, 512, 2, 2, 2]
    │
    ├── avgpool: AdaptiveAvgPool3d(1) → [B, 512]
    └── fc: Dropout(0.4) → Linear(512, 2)
    │
    ▼
logits [B, 2]  ← FocalLoss(alpha=0.75, gamma=2.0)
```

| Campo | Valor |
|---|---|
| **Arquitectura** | ResNet 3D (desde cero, weights=None) |
| **Parámetros totales** | ~15M |
| **Entrada** | `[B, 1, 64, 64, 64]` — volumen CT abdominal, float32 |
| **Salida** | `[B, 2]` — logits binarios (PDAC+ / PDAC−) |
| **Gradient checkpointing** | ✅ Habilitado (obligatorio para VRAM < 20 GB) |

---

## 4. Pipeline de preprocesado

### FASE 1 OFFLINE (dataset completo — determinista — se ejecuta una vez)

| Paso | Técnica | Justificación |
|---|---|---|
| 1 | Carga `.nii.gz` → verificación orientación/dirección + alineación máscara | Lectura nativa del formato NIfTI y validación de consistencia geométrica |
| 2 | Remuestreo isotrópico → 0.8×0.8×0.8 mm³ (Bspline para imagen, NN para máscara) | Estandariza spacing preservando detalle en tejido blando pancreático |
| 3 | Clipping HU → [−150, +250] para CECT abdominal / páncreas | Ventana CECT abdominal para páncreas (tejido blando, HU ~30–50 con contraste) |
| 4 | [ETAPA COARSE] Segmentación del páncreas (nnUNet 3d_lowres) → genera máscara ROI | Localiza la región pancreática para evitar destrucción de señal por resize global |
| 5 | Recorte al bounding box del páncreas + margen de 20 mm por eje | Aísla la región de interés preservando contexto peripancreático |
| 6 | Normalización Z-score sobre foreground (percentil 0.5–99.5) | Compensa bias multicéntrico entre centros y protocolos de adquisición |
| 7 | Extracción de parches 3D (96³ o 192³ vóxeles) → `.nii.gz` / `.npy` | Parches volumétricos a resolución suficiente para detectar PDAC |

### FASE 2 ONLINE / ON-THE-FLY (solo entrenamiento — estocástica — cada epoch)

| Paso | Técnica | Justificación |
|---|---|---|
| 8 | Oversampling de parches positivos (PDAC) → 33% del batch | Compensa el desbalance extremo durante el muestreo |
| 9 | Flips aleatorios en ejes X, Y, Z (p=0.5 c/u) | Augmentación geométrica básica que explota la simetría anatómica |
| 10 | Rotaciones 3D: ±30° en X/Y, ±180° en Z (interpolación trilineal) | Simula variabilidad en orientación y posición del paciente |
| 11 | Escalado uniforme aleatorio ([0.7, 1.4] × tamaño del parche) | Simula variabilidad en tamaño del páncreas y resolución del escáner |
| 12 | Deformación elástica 3D (σ=5–8, magnitud=50–150 vóxeles, p=0.2) | Simula variabilidad morfológica natural del tejido abdominal |
| 13 | Ajuste Gamma [0.7, 1.5] + multiplicación de brillo [0.75, 1.25] | Simula variabilidad en protocolos de contraste y calidad de imagen |
| 14 | Ruido Gaussiano aditivo (σ ∈ [0, 50 HU equivalente]) | Simula ruido de adquisición variable entre escáneres |
| 15 | Blur axial / simulación de movimiento respiratorio (p=0.1) | Simula artefactos de movimiento respiratorio comunes en CT abdominal |

> **⚠️ Nota:** HU clip [−150, +250] es diferente de LUNA16 [−1000, +400]. Ventana CECT abdominal para páncreas (tejido blando, HU ~30–50 con contraste).

---

## 5. Configuración de entrenamiento

| Hiperparámetro | Valor | Constante en `expert4_config.py` |
|---|---|---|
| **Optimizador** | AdamW | — |
| **Learning rate** | 5×10⁻⁵ | `EXPERT4_LR` |
| **Weight decay** | 0.05 | `EXPERT4_WEIGHT_DECAY` |
| **Batch size (real)** | 2 | `EXPERT4_BATCH_SIZE` |
| **Accumulation steps** | 8 | `EXPERT4_ACCUMULATION_STEPS` |
| **Batch efectivo** | 16 (2 × 8) | — |
| **Mixed precision (FP16)** | Sí | `EXPERT4_FP16` |
| **Max épocas** | 100 | `EXPERT4_MAX_EPOCHS` |
| **Early stopping patience** | 15 | `EXPERT4_EARLY_STOPPING_PATIENCE` |
| **Early stopping monitor** | `val_loss` | — |

### 5.1 Nota sobre batch size

El batch size real es 2, limitado por el tamaño de los volúmenes 3D completos (64³) en combinación con la arquitectura ResNet 3D. La acumulación de 8 pasos proporciona un batch efectivo de 16, suficiente para estabilizar los gradientes con FocalLoss y desbalance extremo.

---

## 6. Scheduler

`CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`

- **T_0=10:** primer ciclo de 10 épocas antes del primer reinicio
- **T_mult=2:** cada ciclo siguiente dura el doble (10 → 20 → 40 épocas)
- Permite exploración periódica del espacio de parámetros, útil para datasets pequeños donde el riesgo de atrapamiento en mínimos locales es alto

---

## 7. Criterio de pérdida y métricas

### 7.1 Loss

`FocalLoss(alpha=0.75, gamma=2.0)` — obligatoria por el desbalance extremo del dataset.

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `gamma=2.0`: reduce la contribución de ejemplos fáciles
- `alpha=0.75`: pondera la clase positiva (PDAC+, mayoritaria en labels pero la clase diagnósticamente relevante)

### 7.2 Métricas

| Métrica | Uso | Target |
|---|---|---|
| **AUC-ROC** | Evaluación principal | > 0.85 (bueno); baseline nnU-Net ≈ 0.88 |
| **F1** | Evaluación secundaria | Requerida por la rúbrica (§10.1: F1 Macro > 0.65 para full marks en expertos 3D) |

---

## 8. Archivos clave

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase2/train_expert4.py` | Script de entrenamiento completo: bucle de épocas, EarlyStopping, checkpointing, soporte `--dry-run`, soporte `--fold` |
| `src/pipeline/fase2/expert4_config.py` | Constantes de hiperparámetros (prefijo `EXPERT4_`). Fuente de verdad |
| `src/pipeline/fase2/models/expert4_resnet3d.py` | Modelo ResNet 3D con conv1 monocanal y gradient checkpointing |
| `src/pipeline/fase2/dataloader_expert4.py` | Función de construcción de dataloaders por fold — retorna `(train_loader, val_loader)` |
| `src/pipeline/datasets/pancreas.py` | Dataset class con `valid_pairs` mechanism y preprocesado HU |

---

## 9. Checkpoint

| Campo | Valor |
|---|---|
| **Directorio** | `checkpoints/expert_04_swin3d_tiny/` |
| **Archivo** | `expert4_best.pt` |
| **Ruta completa** | `checkpoints/expert_04_swin3d_tiny/expert4_best.pt` |
| **Descripción** | Pesos Experto 4 (Páncreas, ResNet 3D) |
| **Criterio de guardado** | Mejor `val_loss` durante el entrenamiento |

> **Nota:** El directorio se llama `expert_04_swin3d_tiny` por compatibilidad con la estructura de checkpoints preexistente, aunque el modelo implementado es ResNet 3D.

---

## 10. Cómo ejecutar

### 10.1 Dry-run (verificación sin datos)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert4.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Importación de todos los módulos (config, modelo, dataloader)
2. Instanciación del modelo ResNet 3D
3. Forward pass sintético con un batch de volúmenes aleatorios `[2, 1, 64, 64, 64]`
4. Verificación de la shape de salida `(2, 2)`
5. Conteo de parámetros
6. Exit code 0 si todo pasa correctamente

### 10.2 Entrenamiento real (un fold)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert4.py --fold 0
```

### 10.3 Entrenamiento completo (5 folds)

```bash
for fold in 0 1 2 3 4; do
    PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert4.py --fold $fold
done
```

---

## 11. Dry-run verificado

| Verificación | Resultado |
|---|---|
| Importaciones | ✅ OK |
| Forward pass del modelo | ✅ OK — shape `(2, 2)` |
| Conteo de parámetros | ✅ ~15M |
| Dataset (fold 0) | ✅ train=1,491 / val=373 samples |
| Dry-run exit code | ✅ 0 |
| Entrenamiento real | ⏳ Pendiente de ejecutar (5 folds) |

---

## 12. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin pesos preentrenados** | ResNet 3D con `weights=None` — entrenamiento from scratch obligatorio |
| **Sin augmentaciones de oclusión** | Prohibido `RandomErasing`, `Cutout`, `CutMix` o cualquier transformación que oculte partes del volumen |
| **FocalLoss obligatoria** | Desbalance ~16.3:1 requiere loss adaptativa |
| **FP16 obligatorio** | Volúmenes 3D requieren mixed precision para caber en VRAM |
| **Gradient checkpointing** | Obligatorio — sin él, batch=2 con ResNet 3D excede la VRAM de 20 GB |
| **k-fold CV obligatorio** | Con ~557 volúmenes, un solo split 80/10/10 no proporciona estimación fiable del rendimiento |
| **HU clip [−150, +250]** | Rango CECT abdominal — NO usar el rango pulmonar [−1000, +400] |

---

## 13. Notas

- **expert_id=4:** este experto se registra como `EXPERT_IDS["pancreas"] = 4` en el sistema MoE.
- **Mecanismo `valid_pairs`:** el dataset Pancreas usa `_build_pairs()` para emparejar labels con archivos NIfTI disponibles en disco. Los 1,307 casos sin CT se omiten silenciosamente con un `log.warning`.
- **k-fold mandatory:** a diferencia de los otros expertos que usan splits fijos 80/10/10, Páncreas requiere k-fold por el tamaño reducido del dataset. El argumento `--fold` es obligatorio para el entrenamiento real.

---

*Documento actualizado el 2026-04-14. Fuentes: `src/pipeline/fase2/expert4_config.py`, `models/expert4_resnet3d.py`, `dataloader_expert4.py`, `train_expert4.py`, `src/pipeline/datasets/pancreas.py`, `proyecto_moe.md` §5.5.*
