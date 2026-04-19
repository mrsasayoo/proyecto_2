# Paso 5.4 — Entrenar Experto 4 (LUNA16 / DenseNet 3D)

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-14 |
| **Alcance** | Entrenamiento del Experto 3 (detección de nódulos pulmonares en CT 3D) con DenseNet 3D desde cero. Genera checkpoint consumido por Fase 4 (ablation) y Fase 5 (fine-tuning global) |
| **Responsable** | Nicolás |
| **Estado general** | ✅ **Script corregido y verificado con dry-run** — pendiente de ejecutar entrenamiento completo |

---

## 1. Objetivo

Entrenar un modelo **DenseNet 3D desde cero** (sin pesos preentrenados, `weights=None`) para clasificación **binaria** de parches de CT pulmonar: nódulo sí / nódulo no. Este experto es el Experto 3 del sistema MoE (`expert_id=3`), responsable del dominio de CT pulmonar 3D.

El dataset LUNA16 tiene un desbalance significativo (~10.7:1 negativos vs. positivos tras el fix de leakage), lo que requiere FocalLoss obligatoria y regularización agresiva.

---

## 2. Conflictos de documentación resueltos

Durante la implementación se identificaron divergencias entre la documentación original (`arquitectura_documentacion.md` §3.4) y la implementación real. La arquitectura definitiva es **DenseNet 3D**:

| Aspecto | Doc. original (§3.4) | Implementación real | Justificación |
|---|---|---|---|
| **Arquitectura** | ViViT-Tiny (~25M params) | DenseNet 3D (~7M params) | Ratio params/datos con ViViT: ~1,700:1 (alto riesgo overfitting). Con DenseNet 3D: ~467:1 (aceptable). Ver `expert3_config.py` para análisis completo |
| **FocalLoss alpha** | 0.25 | 0.85 | alpha=0.25 pesa la clase minoritaria al 25% — contra-intuitivo. alpha=0.85 pondera la clase positiva (nódulos, minoritaria) acorde al ratio de desbalance real (~10.7:1). Calibrado para penalizar falsos negativos |

> La arquitectura definitiva para el Experto 3 (LUNA16) es DenseNet 3D (desde cero, weights=None), seleccionada por su capacidad de aprendizaje eficiente de características 3D con conexiones densas que facilitan el gradiente en datasets médicos de tamaño moderado.

> **Nota:** El directorio de checkpoint mantiene el nombre `expert_03_vivit_tiny/` por compatibilidad con la estructura existente, aunque el modelo implementado es DenseNet 3D.

---

## 3. Archivos modificados

| Archivo | Tipo | Descripción |
|---|---|---|
| `src/pipeline/fase2/train_expert3.py` | Corregido | **BUG-1 corregido:** paths de checkpoint y log apuntaban a la raíz del proyecto en vez de `checkpoints/expert_03_vivit_tiny/`. Corregido para que `expert3_best.pt` y los logs se escriban en el directorio correcto |
| `src/pipeline/fase2/expert3_config.py` | Existente | Hiperparámetros del experto — fuente de verdad. Sin modificaciones |

---

## 4. Arquitectura del modelo

```
parche CT [B, 1, 64, 64, 64]
    │
    ▼
DenseNet 3D (weights=None, desde cero)
    │
    ├── initial_conv: Conv3d(1→64, k=7, s=2) → BN3d → ReLU → MaxPool3d
    ├── dense_block_1 → transition_1
    ├── dense_block_2 → transition_2
    ├── dense_block_3 → transition_3
    ├── dense_block_4
    │   → [B, 1024, 2, 2, 2]
    │
    ├── norm: BN3d → ReLU
    ├── avgpool: AdaptiveAvgPool3d(1) → [B, 1024]
    └── fc: Dropout(0.4) → Linear(1024, 2)
    │
    ▼
logits [B, 2]  ← FocalLoss(alpha=0.85, gamma=2.0)
```

| Campo | Valor |
|---|---|
| **Arquitectura** | DenseNet 3D (desde cero) |
| **Backbone** | 3D DenseNet con conexiones densas entre capas |
| **Parámetros totales** | ~7M |
| **Entrada** | `[B, 1, 64, 64, 64]` — parche CT centrado en candidato a nódulo, float32 |
| **Salida** | `[B, 2]` — logits binarios (nódulo / no-nódulo) |

### 4.1 Por qué DenseNet 3D

DenseNet 3D usa conexiones residuales densas (cada capa recibe feature maps de todas las capas anteriores), lo que facilita el gradiente y la reutilización de features. Para datasets médicos 3D de tamaño moderado (~17K parches), esto ofrece mejor convergencia que modelos basados en Transformers 3D o redes puramente convolucionales sin skip connections. El desbalance 10.7:1 se maneja con FocalLoss(α=0.85, γ=2.0).

| Modelo | Params | Ratio params/datos | Riesgo overfitting |
|---|---|---|---|
| ViViT-Tiny | ~25M | ~1,700:1 | 🔴 Alto |
| DenseNet 3D | ~7M | ~467:1 | 🟡 Moderado (mitigable con regularización) |

---

## 5. Configuración de entrenamiento

| Hiperparámetro | Valor | Constante en `expert3_config.py` |
|---|---|---|
| **Optimizador** | AdamW | — |
| **Learning rate** | 3×10⁻⁴ | `EXPERT3_LR` |
| **Weight decay** | 0.03 | `EXPERT3_WEIGHT_DECAY` |
| **Dropout FC** | 0.4 | `EXPERT3_DROPOUT_FC` |
| **Spatial Dropout 3D** | 0.15 | `EXPERT3_SPATIAL_DROPOUT_3D` |
| **Label smoothing** | 0.05 | `EXPERT3_LABEL_SMOOTHING` |
| **Batch size (real)** | 4 | `EXPERT3_BATCH_SIZE` |
| **Accumulation steps** | 8 | `EXPERT3_ACCUMULATION_STEPS` |
| **Batch efectivo** | 32 (4 × 8) | — |
| **Mixed precision (FP16)** | Sí | `EXPERT3_FP16` |
| **Max épocas** | 100 | `EXPERT3_MAX_EPOCHS` |
| **Early stopping patience** | 20 | `EXPERT3_EARLY_STOPPING_PATIENCE` |
| **Early stopping monitor** | `val_loss` | `EXPERT3_EARLY_STOPPING_MONITOR` |

### 5.1 Nota sobre batch size y accumulation

El batch size real es 4 (limitado por VRAM con volúmenes 64³) y la acumulación es 8 (el doble del mínimo obligatorio de 4). La acumulación mayor proporciona estabilidad adicional del gradiente, que es crítica cuando se usa FocalLoss con desbalance extremo.

---

## 6. Pipeline de preprocesado

### FASE 1 OFFLINE (todo el dataset — se ejecuta una sola vez)

| Paso | Técnica | Justificación |
|---|---|---|
| 1 | Carga `.mhd/.raw` → conversión a HU (slope/intercept) | Lectura nativa del formato LUNA16 y conversión a unidades radiológicas estándar |
| 2 | Corrección de píxeles fuera de FOV (−2000 → 0) | Los escáneres marcan el exterior del FOV con valores extremos que corrompen la normalización |
| 3 | Remuestreo isotrópico → 1×1×1 mm³ | Estandariza el spacing variable entre escáneres para que los filtros 3D operen en escala real |
| 4 | Segmentación pulmonar (máscara 3D + dilatación morfológica) | Aísla el parénquima pulmonar, eliminando estructuras no relevantes (hueso, mediastino) |
| 5 | Clipping HU → [−1000, +400] | Rango de Unidades Hounsfield relevante para tejido pulmonar y nódulos |
| 6 | Normalización → [0.0, 1.0] | Escala el rango HU clipped a float32 normalizado |
| 7 | Zero-centering → restar media global (≈ 0.25 en LUNA16) | Centra la distribución para mejor convergencia del optimizador |
| 8 | Extracción de parches 3D (64³ o 128³ vóxeles) → `.npy` | Parche volumétrico centrado en las coordenadas (x, y, z) del candidato |

### FASE 2 ONLINE / ON-THE-FLY (solo entrenamiento — cada epoch)

| Paso | Técnica | Justificación |
|---|---|---|
| 9 | Oversampling de muestras positivas (ratio ~1:10) | Compensa el desbalance extremo del dataset durante el muestreo |
| 10 | Flips aleatorios en ejes X, Y, Z (p=0.5 c/u) | Augmentación geométrica básica que explota la simetría anatómica |
| 11 | Rotaciones 3D continuas (±15°) o discretas (90°/180°/270°) | Simula variabilidad en la orientación del escáner y posición del paciente |
| 12 | Escalado uniforme aleatorio ([0.8, 1.2] × tamaño) | Simula variabilidad en el tamaño de nódulos y resolución del escáner |
| 13 | Traslación aleatoria (±3–5 mm por eje) | Simula imprecisión en el centrado del candidato |
| 14 | Deformación elástica 3D (σ=1–3, α=0–5 mm) | Simula variabilidad morfológica natural del tejido pulmonar |
| 15 | Ruido Gaussiano + ajuste de brillo/contraste | Simula variabilidad en la calidad de imagen entre escáneres |

> **Conversión de coordenadas:** Las coordenadas de `candidates_V2.csv` están en espacio mundo (mm). Se convierten a espacio vóxel con `LUNA16PatchExtractor.world_to_voxel()` considerando el spacing variable de cada volumen.

---

## 7. Criterio de pérdida y métricas

### 7.1 Loss

`FocalLoss(gamma=2.0, alpha=0.85)` — obligatoria por el desbalance extremo del dataset.

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `gamma=2.0`: reduce la contribución de negativos fáciles en 400× (cuando p_t=0.95)
- `alpha=0.85`: pondera la clase positiva (nódulos, minoritaria) para penalizar falsos negativos
- `label_smoothing=0.05`: convierte labels {0,1} → {0.025, 0.975}, previniendo sobreconfianza

> **⚠️ BCELoss está prohibida:** Con BCELoss, el modelo converge a predecir siempre "no-nódulo" (accuracy 99.8%, AUC ~0.50, utilidad clínica nula).

### 7.2 Métricas

| Métrica | Uso | Justificación |
|---|---|---|
| **AUC-ROC** | Evaluación principal | Capacidad discriminativa independiente del threshold, robusta ante desbalance |
| **F1** | Evaluación secundaria | Requerida por la rúbrica (§10.1: F1 Macro > 0.65 para full marks en expertos 3D) |
| **Sensitivity** | Monitoreo clínico | Proporción de nódulos correctamente detectados — la métrica clínicamente más relevante |
| **Specificity** | Monitoreo clínico | Proporción de negativos correctamente clasificados |
| **Accuracy** | **NUNCA** | ~99.8% prediciendo siempre "no-nódulo" — métrica engañosa |

---

## 8. Dataset

| Campo | Valor |
|---|---|
| **Dataset** | LUNA16 / LIDC-IDRI |
| **Modalidad** | CT pulmonar 3D (parches, no volúmenes completos) |
| **Tarea** | Clasificación binaria (nódulo sí / nódulo no) |
| **Train** | 14,728 parches |
| **Val** | 1,143 parches |
| **Test** | 1,914 parches |
| **Desbalance** | ~10.7:1 (negativos vs. positivos) |
| **Split** | Por `seriesuid` (leakage fix aplicado 2026-04-02) |
| **Candidatos** | `candidates_V2.csv` (754,975 filas) — **SIEMPRE V2, nunca V1** |
| **Anotaciones** | `annotations.csv` (1,186 nódulos) |
| **Techo de sensibilidad** | 94.4% (5.6% de nódulos con acuerdo ≤2/4 radiólogos) |

### 8.1 ⚠️ Data leakage — directorio prohibido

**NO usar** `datasets/luna_lung_cancer/patches/_LEAKED_DO_NOT_USE/`

Este directorio (anteriormente `train_stale_backup/`) contiene 1,839 parches con data leakage detectado. Fue renombrado para prevenir uso accidental. El pipeline de entrenamiento usa exclusivamente los parches de `luna_splits.json`.

---

## 9. Bug corregido (BUG-1)

| Campo | Detalle |
|---|---|
| **Archivo** | `src/pipeline/fase2/train_expert3.py` |
| **Bug** | Los paths de checkpoint y log apuntaban a la raíz del proyecto (`expert3_best.pt`, `train_expert3.log`) en vez del directorio correcto |
| **Fix** | Paths actualizados para apuntar a `checkpoints/expert_03_vivit_tiny/expert3_best.pt` y logs al directorio correspondiente |
| **Impacto** | Sin el fix, el checkpoint se sobrescribiría en la raíz y no sería encontrado por Fase 4/5 |

---

## 10. Checkpoint

| Campo | Valor |
|---|---|
| **Directorio** | `checkpoints/expert_03_vivit_tiny/` |
| **Archivo** | `expert3_best.pt` |
| **Ruta completa** | `checkpoints/expert_03_vivit_tiny/expert3_best.pt` |
| **Descripción** | Pesos Experto 3 (LUNA16, DenseNet 3D) |
| **Criterio de guardado** | Mejor `val_loss` durante el entrenamiento |

> **Nota:** El directorio se llama `expert_03_vivit_tiny` por compatibilidad con la estructura de checkpoints preexistente, aunque el modelo implementado es DenseNet 3D.

---

## 11. Cómo ejecutar

### 11.1 Entrenamiento real

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert3.py
```

### 11.2 Dry-run (verificación sin datos)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert3.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Importación de todos los módulos (config, modelo, dataloader)
2. Instanciación del modelo DenseNet 3D
3. Forward pass sintético con un batch de volúmenes aleatorios `[2, 1, 64, 64, 64]`
4. Verificación de la shape de salida `(2, 2)`
5. Conteo de parámetros
6. Exit code 0 si todo pasa correctamente

---

## 12. Estado

| Verificación | Resultado |
|---|---|
| Importaciones | ✅ OK |
| Forward pass del modelo | ✅ OK — shape `(2, 2)` |
| Conteo de parámetros | ✅ ~7M |
| BUG-1 (paths) | ✅ Corregido |
| Dry-run | ✅ Exitoso (exit_code=0, 0 errores) |
| Entrenamiento real | ⏳ Pendiente de ejecutar |

---

## 13. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin pesos preentrenados** | DenseNet 3D con `weights=None` — entrenamiento from scratch obligatorio |
| **Sin augmentaciones de oclusión** | Prohibido `RandomErasing`, `Cutout`, `CutMix` o cualquier transformación que oculte partes del volumen |
| **FocalLoss obligatoria** | BCELoss produce modelo trivial (siempre clase 0) con desbalance ~10.7:1 |
| **candidates_V2.csv** | SIEMPRE V2 — V1 tiene 24 nódulos mal etiquetados/ausentes |
| **No usar `_LEAKED_DO_NOT_USE/`** | Directorio con data leakage — uso prohibido |
| **FP16 obligatorio** | Volúmenes 3D requieren mixed precision para caber en VRAM |
| **Gradient checkpointing** | Recomendado para mantener VRAM < 12 GB con batch=4 |

---

*Documento actualizado el 2026-04-14. Fuentes: `src/pipeline/fase2/expert3_config.py`, `train_expert3.py`, `checkpoints/expert_03_vivit_tiny/model_card.md`, `proyecto_moe.md` §5.4.*
