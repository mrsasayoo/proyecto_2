# Arquitectura del Sistema MoE — Clasificación Médica Multimodal

*Última actualización: 2026-04-06 (Pasos 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6, 8, 9, 10, 11 y 12 implementados — dry-run verificado) | Autor: MNEMON (agente de memoria) | Proyecto: Incorporar Elementos de IA — Unidad II, Bloque Visión | Universidad Autónoma de Occidente — Pregrado Ingeniería de Datos / IA*

---

## Tabla de contenido

1. [Visión general del sistema](#1-visión-general-del-sistema)
2. [Hoja de ruta de desarrollo](#2-hoja-de-ruta-de-desarrollo)
3. [Expertos de dominio (1–5)](#3-expertos-de-dominio-15)
4. [Experto 6 — CAE (Experto de Incertidumbre)](#4-experto-6--cae-experto-de-incertidumbre)
5. [Diseño del router](#5-diseño-del-router)
6. [Pipeline de preprocesamiento de imagen](#6-pipeline-de-preprocesamiento-de-imagen)
7. [Pipeline de datos (Fase 0)](#7-pipeline-de-datos-fase-0)
8. [Dashboard interactivo (Fase 6)](#8-dashboard-interactivo-fase-6)
9. [Optimización de hardware](#9-optimización-de-hardware)
10. [Métricas de evaluación y umbrales mínimos](#10-métricas-de-evaluación-y-umbrales-mínimos)
11. [Restricciones de penalización](#11-restricciones-de-penalización)

---

## 1. Visión general del sistema

### 1.1 Propósito

El sistema es un **Mixture of Experts (MoE)** para clasificación de imágenes médicas multimodales. Recibe como única entrada una imagen o volumen médico (PNG, JPEG o NIfTI) y produce un diagnóstico clasificándolo a través del experto especializado correspondiente.

La pregunta científica central del proyecto:

> ¿Justifica el Vision Transformer su costo computacional como router frente a métodos estadísticos clásicos operando sobre los mismos embeddings?
> 

### 1.2 Composición: 5 + 1 expertos

El sistema contiene **6 expertos**, organizados en dos categorías:

| ID | Nombre | Tipo | Modalidad | Función |
| --- | --- | --- | --- | --- |
| 0 | Chest (NIH ChestXray14) | Dominio | Radiografía 2D | 14 patologías torácicas |
| 1 | ISIC 2019 | Dominio | Dermatoscopía 2D | 9 clases de lesiones cutáneas |
| 2 | OA-Knee | Dominio | Radiografía 2D | 3 grados osteoartritis (KL) |
| 3 | LUNA16 | Dominio | CT pulmonar 3D | Detección nódulos (binario) |
| 4 | Pancreas | Dominio | CT abdominal 3D | Detección PDAC (binario) |
| 5 | CAE (OOD) | Incertidumbre | Todas las modalidades | Reconstrucción + filtro de basura/defectos |

Los expertos 0–4 se entrenan con su dataset propio. El experto 5 (CAE) se entrena con **todos los datasets combinados** y se activa por entropía del router, no por routing directo.

> **Nota de aprobación:** La incorporación del Experto 6 (CAE) fue aprobada verbalmente por el profesor. La guía oficial define 5 expertos, pero esta extensión amplía la propuesta con un mecanismo de filtro de calidad de imagen. Como consecuencia, el vector de gating es `g ∈ ℝ^6` en lugar de `g ∈ ℝ^5`, y la función de pérdida incorpora un tercer término `β·L_error` que la guía no contempla pero el profesor autorizó.

> **Nota sobre los backbones de los expertos:**
> - Los modelos **ViT, CvT y Swin** se construyen **desde cero** con arquitectura predefinida (no son pesos preentrenados de ImageNet ni HuggingFace como punto de partida).
> - **DenseNet** es una arquitectura **propia/custom** diseñada desde cero para este proyecto.
> - **Paso 4.1 (Entrenar backbones — ✅ completado):** antes de extraer embeddings, cada backbone se entrena end-to-end con una **tarea proxy de clasificación de dominio médico** (5 clases = `expert_id` 0–4, `CrossEntropyLoss`). La cabeza lineal (`LinearHead(d_model→5)`) se descarta tras el entrenamiento; solo los pesos del backbone se guardan en `checkpoints/backbone_0X_<nombre>/backbone.pth`. El Paso 4.2 carga ese checkpoint y congela el backbone antes de extraer embeddings.
> - **Fix en `transform_2d.py` (2026-04-05):** durante el dry-run de Pasos 5.1/5.2, se detectó que el import de `fase1_config` en `src/pipeline/fase1/transform_2d.py` fallaba cuando se invocaba desde el contexto de Fase 2. Corregido con `try/except` que intenta `from . import fase1_config` y como fallback `from fase1 import fase1_config`, preservando compatibilidad hacia atrás.

### 1.3 Restricción absoluta: solo píxeles

El sistema **nunca** recibe metadatos, etiquetas de modalidad, nombres de archivo ni texto como entrada. La única entrada es el tensor de la imagen.

- Detección automática 2D/3D por el rango del tensor: `rank=4` → 2D, `rank=5` → 3D.
- Violación de esta restricción: **−20% en la nota del proyecto**.

### 1.4 Flujo general

```
Imagen/Volumen (sin metadatos)
        │
        ▼
Preprocesador Adaptativo
  • rank=4 → resize 224×224, normalización ImageNet
  • rank=5 → resize 64×64×64, normalización HU
        │
        ▼
Backbone compartido (ViT-Tiny / CvT-13 / Swin-Tiny)
  • Patch Embedding → tokens [B, N, d_model]
  • Transformer Blocks + Self-Attention
  • Extracción del CLS token: z ∈ ℝ^d_model
        │
        ▼
Router: softmax(W·z + b) → distribución sobre 6 expertos
        │
        ├── H(g) baja (alta confianza) → Experto argmax (0–4)
        │
        └── H(g) alta (incertidumbre) → Experto 5 (CAE)
```

> **Nota importante:** Además de ViT-Tiny, CvT-13 y Swin-Tiny, también se incluye **DenseNet custom** como backbone alternativo de embeddings propios, recomendado por el profesor para imágenes médicas 2D.

---

## 2. Hoja de ruta de desarrollo

### 2.1 Secuencia de 12 pasos

| # | Paso | Descripción | Fase pipeline | Estado |
| --- | --- | --- | --- | --- |
| 1 | Descargar datos | Descarga de los 5 datasets desde fuentes públicas | Fase 0 – Paso 1 | ✅ Completado (100% — 19/19 ítems resueltos, 2026-04-05) |
| 2 | Extraer archivos | Descompresión de ZIPs, tarballs, NIfTI | Fase 0 – Paso 2 | ✅ Completado |
| 3 | Preparar datos | Splits 80/10/10, preprocesado por dominio, etiquetas | Fase 0 – Pasos 3–5 | 🔄 En progreso — verificar con re-ejecución (Pancreas previamente con error, datos ya disponibles) |
| 4.1 | Entrenar backbones | Entrenamiento end-to-end de los 4 backbones desde cero con tarea proxy (clasificación de dominio médico, 5 clases). Produce `backbone.pth` en `checkpoints/backbone_0X_<nombre>/` | Fase 1 | ✅ Completado (2026-04-05) |
| 4.2 | Generar embeddings | Backbone congelado (cargado desde checkpoint de Paso 4.1) extrae CLS tokens → `.npy` | Fase 1 | ⏳ Pendiente |
| 5.1 | Entrenar Experto 1 | Chest — ConvNeXt-Tiny (~27.8M params), BCEWithLogitsLoss(pos_weight), 14 clases multilabel. Archivos: `expert1_config.py`, `models/expert1_convnext.py`, `dataloader_expert1.py`, `train_expert1.py`. Checkpoint: `checkpoints/expert_00_convnext_tiny/expert1_best.pt` | Fase 2 | ✅ Script implementado — dry-run verificado |
| 5.2 | Entrenar Experto 2 | ISIC — EfficientNet-B3 (~10.7M params), CrossEntropyLoss(class_weights), 9 clases (8 train + UNK). Archivos: `expert2_config.py`, `models/expert2_efficientnet.py`, `dataloader_expert2.py`, `train_expert2.py`. Checkpoint: `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` | Fase 2 | ✅ Script implementado — dry-run verificado |
| 5.3 | Entrenar Experto 3 | OA — VGG16-BN (~134M params), CrossEntropyLoss(class_weights), 3 clases ordinales (KL→Normal/Leve/Severo). Archivos: `expert_oa_config.py`, `models/expert_oa_vgg16bn.py`, `dataloader_expert_oa.py`, `train_expert_oa.py`. Checkpoint: `checkpoints/expert_02_vgg16_bn/expert_oa_best.pt` | Fase 2 | ✅ Script implementado — dry-run verificado |
| 5.4 | Entrenar Experto 4 | LUNA16 — MC3-18 (~11.4M params), FocalLoss(alpha=0.85, gamma=2.0), 2 clases. Script `train_expert3.py` corregido (BUG-1: paths). Checkpoint: `checkpoints/expert_03_vivit_tiny/expert3_best.pt` | Fase 2 | ✅ Script corregido — dry-run verificado |
| 5.5 | Entrenar Experto 5 | Páncreas — Swin3D-Tiny (~6.94M params), FocalLoss(alpha=0.75, gamma=2.0), 2 clases, k-fold CV (k=5). Archivos: `expert4_config.py`, `models/expert4_swin3d.py`, `dataloader_expert4.py`, `train_expert4.py`. Checkpoint: `checkpoints/expert_04_swin3d_tiny/expert4_best.pt` | Fase 2 | ✅ Script implementado — dry-run verificado |
| 5.6 | Entrenar Experto 6 | CAE — ConvAutoEncoder 2D (~206M params), MSE+L1, 5 datasets combinados (162,611 muestras). Activado por entropía H(g)>P95, NO routing directo. Archivos: `expert5_cae_config.py`, `models/expert5_cae.py`, `dataloader_cae.py`, `train_cae.py`. Checkpoint: `checkpoints/expert_05_cae/cae_best.pt` | Fase 3 | ✅ Script implementado — dry-run verificado |
| 6 | Ablation study | 4 routers comparados sobre embeddings congelados: Linear (L_aux Switch Transformer), GMM (EM full/diag), NaiveBayes (MLE analítico), kNN-FAISS (coseno, k=5). Winner: load balance filter → max accuracy. Input: embeddings Fase 1, d_model=192, N_EXPERTS_DOMAIN=5. Output: `ablation_results.json` + best router model. `--dry-run` disponible (embeddings sintéticos, sin artefactos a disco) | Fase 2 | ✅ Implementado — dry-run verificado |
| 7 | Escoger router | Seleccionar el mejor método de routing (integrado en Paso 6 — selección automática por criterio balance+accuracy) | Fase 2 | ✅ Implementado (parte del Paso 6) |
| 8 | Fine-tuning por etapas | Fine-tuning 3 etapas: Stage1 (router LR=1e-3, ~50ep) → Stage2 (router+cabezas LR=1e-4, ~30ep) → Stage3 (global, LR router=1e-4/expertos=1e-6, 7-10ep). Script: `fase5_finetune_global.py`. `--dry-run` disponible. | Fase 5 | ✅ Implementado — dry-run verificado |
| 9 | Prueba con datos reales | Evaluación end-to-end del MoE sobre test sets reales. Inferencia sin expert_id (router decide): backbone→router→entropy→experto. Métricas F1/AUC por experto, routing accuracy, load balance, OOD AUROC, Grad-CAM. Script: `fase9_test_real.py`. `--dry-run` disponible. | Fase 6 | ✅ Implementado — dry-run verificado |
| 10 | Verificación | Verificación funcional completa del sistema. 9 categorías de checks (A–I): imports, no-metadata, load balance, métricas §10, reproducibilidad, artefactos, augmentaciones prohibidas, VRAM, arquitectura. Detecta penalizaciones −40%/−20%. Script: `paso10_verificacion.py`. `--dry-run` disponible. | Fase 6 | ✅ Implementado — dry-run verificado |
| 11 | Página web | Dashboard Interactivo MoE con 8 funcionalidades obligatorias (§11 guía). Gradio app: carga imagen, inferencia en tiempo real, Grad-CAM, panel experto, ablation study, load balance, OOD alert. Scripts: `paso11_webapp.py`, `webapp_helpers.py`. Degradación graceful sin checkpoints. `--dry-run` disponible. | Fase 6 | ✅ Implementado — dry-run verificado |
| 12 | Dashboard | Dashboard de reporte final con 6 tabs (arquitectura, métricas, ablation, load balance, estado, figuras) + 5 figuras obligatorias §12.2 (matplotlib Agg, 150 dpi). Scripts: `paso12_dashboard.py`, `dashboard_figures.py`. `--dry-run` disponible. | Fase 6 | ✅ Implementado — dry-run verificado |

### 2.2 Mapa de fases del pipeline

| Fase | Nombre | Descripción | Script orquestador |
| --- | --- | --- | --- |
| Fase 0 | Preparación de datos | Descarga, extracción, splits, parches 3D | `fase0_pipeline.py` |
| Fase 1 | Entrenamiento de backbones + extracción de embeddings | **Paso 4.1:** entrenamiento end-to-end desde cero (proxy: clasificación de dominio) → `backbone.pth`. **Paso 4.2:** backbone congelado → vectores `.npy` | `fase1_train_pipeline.py` (4.1) / `fase1_pipeline.py` (4.2) |
| Fase 2 | Entrenamiento de los 5 expertos | Fine-tuning individual de cada experto de dominio | `fase2_train_experts.py` |
| Fase 3 | Entrenamiento del Experto 6 (CAE) | Autoencoder convolucional para filtro OOD/basura | `fase3_train_cae.py` |
| Fase 4 | Ablation study del router | 4 métodos de routing (Linear, GMM, NaiveBayes, kNN-FAISS) comparados sobre embeddings congelados. ✅ Implementado — dry-run verificado | `fase2_pipeline.py` (en `src/pipeline/fase2/`) |
| Fase 5 | Fine-tuning global | Sistema MoE completo, descongelamiento progresivo en 3 stages: Stage1 (router solo, LR=1e-3) → Stage2 (router+cabezas, LR=1e-4) → Stage3 (global, LR router=1e-4 / expertos=1e-6). L_total = L_task + 0.01·L_aux + 0.1·L_error. ✅ Implementado — dry-run verificado | `src/pipeline/fase5/fase5_finetune_global.py` |
| Fase 6 | Evaluación + Despliegue web | **Paso 9:** Evaluación end-to-end del MoE sobre test sets reales (inferencia sin expert_id). ✅ Implementado — dry-run verificado. **Paso 10:** Verificación funcional completa (9 categorías, detección de penalizaciones). ✅ Implementado — dry-run verificado. **Paso 11:** Dashboard Interactivo MoE — 8 funcionalidades, Gradio, --dry-run verificado. ✅ Implementado. **Paso 12:** Dashboard de reporte final — 6 tabs + 5 figuras obligatorias §12.2 (matplotlib Agg, 150 dpi), --dry-run verificado. ✅ Implementado | `src/pipeline/fase6/fase9_test_real.py` (Paso 9) / `src/pipeline/fase6/paso10_verificacion.py` (Paso 10) / `src/pipeline/fase6/paso11_webapp.py` (Paso 11) / `src/pipeline/fase6/webapp_helpers.py` (Paso 11 helpers) / `src/pipeline/fase6/paso12_dashboard.py` (Paso 12) / `src/pipeline/fase6/dashboard_figures.py` (Paso 12 figures) |

---

## 3. Expertos de dominio (1–5)

### 3.1 Experto 0 — Chest (NIH ChestXray14)

| Campo | Valor |
| --- | --- |
| **Dataset** | NIH ChestXray14 |
| **Modalidad** | Radiografía de tórax 2D |
| **Tarea** | Clasificación multilabel (14 patologías simultáneas) |
| **Arquitectura** | ConvNeXt-Tiny |
| **Loss** | `BCEWithLogitsLoss` con `pos_weight` (multilabel) |
| **Métricas** | AUC-ROC por clase + F1 Macro + AUPRC |
| **Volumen** | ~112K imágenes (train: 88,999 / val: 11,349 / test: 11,772) |
| **Preprocesado** | Resize 224×224, normalización ImageNet |
| **Split** | Por Patient ID (`train_val_list.txt` / `test_list.txt` oficiales) |

**Estado del dataset (verificado 2026-04-04):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Completo |
| **Imágenes** | 112,120 `.png` (`images_001/` a `images_012/`) |
| **all_images/** | 112,120 symlinks (namespace plano) |
| **Splits** | train 88,999 / val 11,349 / test 11,772 = 112,120 ✅ |
| **Método de split** | Patient ID (listas oficiales `train_val_list.txt` / `test_list.txt`) |
| **Metadata** | `Data_Entry_2017.csv` (112,120 filas), `BBox_List_2017.csv` |
| **data.zip** | ~~45 GB eliminado el 2026-04-04~~ |
| **Anomalías** | Ninguna |
| **Fuente** | Kaggle `nih-chest-xrays` (organización oficial NIH) — Riesgo 🟢 Bajo |

**Patologías (14):** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.

**Estado del script de entrenamiento (Paso 5.1, 2026-04-05):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Script listo — dry-run verificado, datos pendientes |
| **Modelo** | `Expert1ConvNeXtTiny` — ConvNeXt-Tiny, ~27.8M params, `weights=None` |
| **Archivos** | `expert1_config.py`, `models/expert1_convnext.py`, `dataloader_expert1.py`, `train_expert1.py` |
| **Checkpoint** | `checkpoints/expert_00_convnext_tiny/expert1_best.pt` |
| **Pipeline** | `CLAHETransform → Resize(224×224) → TVF → ToTensor → Normalize(ImageNet)` |

**Notas críticas:**
- Etiquetas generadas por NLP — benchmark DenseNet-121 ≈ 0.81 AUC macro.
- AUC > 0.85 debe investigarse por posible confounding.
- **Nunca** usar Accuracy como métrica (~54% prediciendo siempre "No Finding").
- BBox disponible para 8/14 clases (~1,000 imágenes) — útil para validar heatmaps en dashboard.
- Filtro `View Position = PA` recomendado para estudios controlados.

### 3.2 Experto 1 — ISIC 2019 (Lesiones cutáneas)

| Campo | Valor |
| --- | --- |
| **Dataset** | ISIC 2019 (HAM10000 + BCN_20000 + MSK) |
| **Modalidad** | Dermatoscopía 2D |
| **Tarea** | Clasificación multiclase (8 clases en train, UNK solo en test) |
| **Arquitectura** | EfficientNet-B3 |
| **Loss** | `CrossEntropyLoss` con `class_weights` |
| **Métricas** | BMCA (oficial) + AUC-ROC por clase |
| **Volumen** | ~25K imágenes |
| **Preprocesado** | Resize 224×224, ColorJitter agresivo |
| **Split** | Por `lesion_id` (función `build_lesion_split`) |

**Estado del dataset (verificado 2026-04-04):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Completo |
| **Imágenes** | 25,331 `.jpg` en `datasets/isic_2019/ISIC_2019_Training_Input/` |
| **Directorio `isic_images/`** | Vacío — directorio huérfano. **No usar.** |
| **Splits** | train 20,409 / val 2,474 / test 2,448 = 25,331 ✅ |
| **Método de split** | `build_lesion_split` por `lesion_id` |
| **Sufijo `_downsampled`** | Manejado por exclusión en `dataset_builder.py` — no es un bug activo |
| **Bug corregido** | `fase0_pipeline.py`, `fase1_pipeline.py`, `pre_modelo.py` — rutas actualizadas a nivel único (fuente oficial) |
| **isic-2019.zip** | ~~9.3 GB eliminado el 2026-04-04~~ |
| **Fuente** | ISIC Archive oficial (S3: `isic-archive.s3.amazonaws.com`). Re-descargado 2026-04-05 (9,771,618,190 bytes, 25,331 imágenes verificadas). Riesgo 🟢 Bajo. |

**Notas críticas:**
- 3 fuentes con bias de dominio — augmentación agresiva obligatoria.
- 9 neuronas en la capa de salida (incluye slot UNK para inferencia).
- Sin `metadata_csv` → riesgo de leakage entre fuentes.
- UNK en inferencia → H(g) alta → Experto 5 OOD la captura.

**Estado del script de entrenamiento (Paso 5.2, 2026-04-05):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Script listo — dry-run verificado, datos pendientes |
| **Modelo** | `Expert2EfficientNetB3` — EfficientNet-B3, ~10.7M params, `weights=None` |
| **Archivos** | `expert2_config.py`, `models/expert2_efficientnet.py`, `dataloader_expert2.py`, `train_expert2.py` |
| **Checkpoint** | `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` |
| **Pipeline** | `BCNCrop (por ISICDataset) → Resize(224×224) → ToTensor → Normalize(ImageNet)` |
| **Augmentation** | `ColorJitter(0.3, 0.3, 0.3, 0.1)` + flips + affine (agresivo, obligatorio por 3 fuentes) |
| **Nota BCNCrop** | Aplicado internamente por `ISICDataset`, no en el objeto `transform` — evita doble crop |

### 3.3 Experto 2 — OA Knee (Osteoartritis de rodilla)

| Campo | Valor |
| --- | --- |
| **Dataset** | Osteoarthritis Knee X-ray |
| **Modalidad** | Radiografía de rodilla 2D |
| **Tarea** | Clasificación ordinal (3 clases: Normal KL0 / Leve KL1-2 / Severo KL3-4) |
| **Arquitectura** | VGG16-BN con clasificador modificado (BN en capas FC, Dropout=0.5) |
| **Expert ID** | 2 (`EXPERT_IDS["oa"]`) |
| **Parámetros** | ~134M (~11.7M features + ~119M classifier) |
| **Loss** | `CrossEntropyLoss` con `class_weights` (opción pragmática) u `OrdinalLoss(n_classes=3)` |
| **Métricas** | QWK (Quadratic Weighted Kappa) + F1-macro (secundaria) — **no** Accuracy |
| **Volumen** | 4,766 imágenes KL-graded (de 9,339 en disco). Remapeo: KL0→Cls0 / KL1+2→Cls1 / KL3+4→Cls2 (`pre_modelo.py:569`) |
| **Preprocesado** | CLAHE (interno en OAKneeDataset) → resize 224×224 BICUBIC (CLAHE **siempre** antes del resize) |
| **Split** | Directorios `oa_splits/{train,val,test}/` — train:3,814 / val:480 / test:472. Copias físicas, ratio 80/10/10 |

**Notas críticas:**
- Remapeo KL0→0 / KL1+2→1 / KL3+4→2 documentado — ver `paso_01_descarga_datos.md` §3.3.
- Sin Patient ID — split por grupo de similitud visual (heurística, no equivale a patient_id). Documentar como limitación.
- KL1 contamina la frontera Clase 0 ↔︎ Clase 1 (la más difícil).
- **Nunca** usar `RandomVerticalFlip` (anatomía simétrica solo horizontalmente).
- Matriz de costos para penalización ordinal disponible:
`[[0, 1, 4],    [1, 0, 1],    [4, 1, 0]]`
- **Nomenclatura de archivos:** usan prefijo `expert_oa_` (no `expert3_`) para evitar confusión entre el expert_id (2) y el paso del pipeline (5.3).

**Estado del script de entrenamiento (Paso 5.3, 2026-04-05):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Script listo — dry-run verificado, pendiente de entrenamiento completo |
| **Modelo** | VGG16-BN con clasificador modificado (BN en FC layers), ~134M params, `weights=None` |
| **Archivos** | `expert_oa_config.py`, `models/expert_oa_vgg16bn.py`, `dataloader_expert_oa.py`, `train_expert_oa.py` |
| **Checkpoint** | `checkpoints/expert_02_vgg16_bn/expert_oa_best.pt` |
| **Pipeline** | `CLAHETransform (interno en dataset) → Resize(224×224, BICUBIC) → ToTensor → Normalize(ImageNet)` |
| **Augmentation** | `RandomHorizontalFlip(0.5)` + `RandomRotation(10)` + `ColorJitter(0.2, 0.15)`. **Prohibido:** `RandomVerticalFlip` |
| **Config** | LR=1e-4, WD=0.05, batch=32, accum=4 (efectivo=128), FP16, patience=10, max_epochs=100 |

### 3.4 Experto 3 — LUNA16 (Cáncer de pulmón)

| Campo | Valor |
| --- | --- |
| **Dataset** | LUNA16 / LIDC-IDRI |
| **Modalidad** | CT pulmonar 3D (parches, no volúmenes completos) |
| **Tarea** | Clasificación binaria de parches (nódulo sí/no) |
| **Arquitectura** | MC3-18 (Mixed Convolutions 3D) — **cambio justificado respecto a la spec original (ViViT-Tiny)** |
| **Expert ID** | 3 |
| **Parámetros** | ~11.4M |
| **Loss** | `FocalLoss(gamma=2, alpha=0.85)` — obligatoria. **Corrección:** alpha=0.85 (no 0.25 como indicaba la doc anterior) |
| **Métricas** | AUC-ROC + F1 + sensitivity + specificity |
| **Tensor** | `[B, 1, 64, 64, 64]` — parche centrado en candidato (x, y, z) |
| **Preprocesado** | HU clip [-1000, 400] → normalización [0, 1] |

**Cambios de arquitectura respecto a la documentación original:**

| Aspecto | Spec original | Implementación | Razón |
| --- | --- | --- | --- |
| Modelo | ViViT-Tiny (~25M params) | MC3-18 (~11.4M params) | Ratio params/datos ViViT: ~1,700:1 (alto riesgo overfitting) vs. MC3-18: ~761:1 (manejable) |
| FocalLoss alpha | 0.25 | 0.85 | alpha=0.85 pondera correctamente la clase positiva minoritaria con ratio ~10.7:1 |

**Notas críticas:**
- Desbalance ~10.7:1 (post-fix leakage) — BCELoss produce modelo trivial (siempre clase 0).
- **Siempre** usar `candidates_V2.csv` (V1 obsoleto, 24 nódulos menos).
- Conversión world→vóxel con `LUNA16PatchExtractor.world_to_voxel()`.
- FP16 obligatorio. Gradient checkpointing recomendado (batch=4, 12GB VRAM).
- Techo teórico de sensibilidad: 94.4%.
- Spacing variable entre volúmenes — verificado en `__init__`.
- **NO usar** `datasets/luna_lung_cancer/patches/_LEAKED_DO_NOT_USE/` (data leakage).

**Estado del script de entrenamiento (Paso 5.4, 2026-04-05):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Script corregido — dry-run verificado (exit_code=0, 0 errores) |
| **Modelo** | MC3-18, ~11.4M params, `weights=None` |
| **Archivos** | `train_expert3.py` (corregido BUG-1: paths de checkpoint/log) |
| **Checkpoint** | `checkpoints/expert_03_vivit_tiny/expert3_best.pt` |
| **BUG-1 corregido** | Paths de checkpoint y log apuntaban a raíz del proyecto → corregidos a `expert_03_vivit_tiny/` |
| **Config** | LR=3e-4, WD=0.03, batch=4, accum=8 (efectivo=32), FP16, patience=20, max_epochs=100 |

### 3.5 Experto 4 — Páncreas (PANORAMA / Zenodo)

| Campo | Valor |
| --- | --- |
| **Dataset** | PANORAMA / Zenodo CT abdominal |
| **Modalidad** | CT abdominal 3D |
| **Tarea** | Clasificación binaria de volumen (PDAC+ / PDAC−) |
| **Arquitectura** | Swin3D-Tiny |
| **Loss** | `FocalLoss(alpha=0.75, gamma=2)` |
| **Métricas** | AUC-ROC > 0.85 (bueno); baseline nnU-Net ≈ 0.88 |
| **Volumen** | ~281 volúmenes — k-fold CV (k=5) obligatorio |
| **Preprocesado** | HU clip [-100, 400] (**no** [-1000, 400]), z-score por volumen |

**Notas críticas:**
- Etiquetas en repositorio GitHub separado — fijar hash del commit.
- Páncreas ocupa ~1% del volumen → resize naïve destruye la señal diagnóstica.
- HU clip diferente al de LUNA16: `HU_ABDOMEN_CLIP = (-100, 400)`.
- z-score por volumen para compensar bias multicéntrico (Radboudumc/MSD/NIH).
- batch_size=1–2 + FP16 + gradient checkpointing (más restrictivo que LUNA16).
- **Estado actual:** ✅ Listo — 557 archivos `.nii.gz` (1,850 pacientes únicos), ~93 GB en `datasets/zenodo_13715870/`. Todos los bugs corregidos: OOM en `paso4_pancreas_labels()` (loop secuencial + streaming), leakage en `split_pancreas()` (`GroupKFold` por `patient_id`). `pancreas_labels_binary.csv`: 1,864 filas (1,756 PDAC+, 108 PDAC−, 0 errores). `pancreas_splits.csv`: 1,864 casos, 186 test (0 leakage), ~1,342/~336 train/val por fold. `cae_splits.csv` regenerado. A19 resuelto: `_build_pairs()` omite silenciosamente los 1,307 casos sin CT en disco (guard `if candidates:`), `log.warning` añadido.

**Estado del script de entrenamiento (Paso 5.5, 2026-04-05):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Script implementado — dry-run verificado (exit_code=0, train=1,491 / val=373 fold 0) |
| **Modelo** | SwinTransformer3d (Tiny), ~6.94M params, `weights=None`, `Conv3d(1, 48, kernel=4, stride=4)` para entrada monocanal |
| **Archivos** | `expert4_config.py`, `models/expert4_swin3d.py`, `dataloader_expert4.py`, `train_expert4.py`, `datasets/pancreas.py` |
| **Checkpoint** | `checkpoints/expert_04_swin3d_tiny/expert4_best.pt` |
| **Pipeline** | `HU clip [-100, 400] → z-score → clip [-3, 3] → rescale [0, 1] → resize 64³ (trilinear)` |
| **Config** | LR=5e-5, WD=0.05, batch=2, accum=8 (efectivo=16), FP16, patience=15, max_epochs=100, FocalLoss(α=0.75, γ=2.0) |
| **Scheduler** | `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` |
| **k-fold** | 5 folds obligatorio (`--fold 0..4`), gradient checkpointing habilitado |

### 3.6 Orden de entrenamiento del sistema completo

Basado en las notas directas del profesor:

1. **Fase 2:** Entrenar los 5 expertos de dominio por separado hasta que converjan bien.
2. **Congelar gradientes:** Una vez convergidos, se aplica `freeze` a todos los expertos (sus pesos no se actualizan más).
3. **Fase 3:** Entrenar el Experto 6 (CAE) con los 5 datasets combinados.
4. **Fase 4:** Con todos los expertos congelados, conectar y entrenar el **router**.
   - El router recibe las métricas de los expertos como señal de pérdida combinada.
   - Aprende a enrutar según si le fue bien o mal con cada experto.
5. **Fase 5:** Fine-tuning global (descongelar cabezas → fine-tuning progresivo).

> **Analogía del profesor:** el router actúa como un "celador/guachimán" — solo aprende a dirigir el tráfico porque ya sabe qué experto es bueno para cada tipo de imagen.

---

## 4. Experto 6 — CAE (Experto de Incertidumbre)

### 4.1 Concepto

> ⚠️ **ACLARACIÓN FUNDAMENTAL: El Experto 6 NO clasifica patologías.** Su función exclusiva es actuar como filtro de calidad de imagen durante la inferencia. Solo recibe una imagen cuando el router no tiene suficiente confianza para delegarla a ninguno de los 5 expertos de dominio.

El sexto experto es un **Convolutional AutoEncoder (CAE)** entrenado sobre los 5 datasets de dominio combinados. Su función no es clasificar patologías — es detectar y gestionar las imágenes que el router no sabe a dónde enviar.

### 4.2 Condición de activación

El CAE se activa cuando la distribución de routing `g = softmax(W·z + b)` tiene **entropía alta**, lo que indica que el router está indeciso sobre qué experto de dominio debe procesar la imagen.

- **Entropía baja** en `g` → alta confianza → delegar al `argmax(g)` (experto 0–4)
- **Entropía alta** en `g` → incertidumbre → delegar al experto 5 (CAE)
- Umbral de entropía calibrado sobre el set de validación (percentil 95 de `H(g)`)

### 4.3 Pipeline de reconstrucción y decisión

El CAE no clasifica la imagen directamente. Intenta **reconstruir** la imagen y toma decisiones basadas en la calidad de la reconstrucción (error de reconstrucción).

### Flujo completo:

```
Imagen con H(g) alta
        │
        ▼
    CAE: Encoder → Latent → Decoder
        │
        ▼
    Calcular error de reconstrucción (MSE / L1)
        │
        ├── Error MUY ALTO ──────────────► Clasificar como "BASURA"
        │                                  (no es imagen médica: gato, ruido, etc.)
        │
        └── Error MODERADO ──────────────► Imagen médica DEFECTUOSA
                    │                       (borrosa, corrupta, artefactos)
                    │
                    ▼
              CAE reconstruye la imagen (denoising)
                    │
                    ▼
              Re-enviar imagen limpia al ROUTER
                    │
                    ├── Router ahora tiene H(g) baja ──► Clasificación normal
                    │
                    └── Router SIGUE con H(g) alta ────► CAE recibe de nuevo
                                │
                                ▼
                          Deconstruir a imagen original
                                │
                                ▼
                          Etiquetar: "NECESITA REVISIÓN DE UN PROFESIONAL"
```

### 4.4 Diagrama Mermaid del árbol de decisión

```mermaid
flowchart TD
    A[Imagen recibida por el Router] --> B{Entropía H de g}
    B -->|H baja: alta confianza| C[Delegar al Experto argmax]
    B -->|H alta: incertidumbre| D[Experto 6 — CAE]

    D --> E[Intentar reconstruir la imagen]
    E --> F{Error de reconstrucción}

    F -->|Error MUY ALTO| G["🗑️ BASURA<br/>(imagen no médica / OOD)"]
    F -->|Error MODERADO| H[Imagen médica DEFECTUOSA]

    H --> I[CAE reconstruye: denoising]
    I --> J[Re-enviar imagen limpia al Router]

    J --> K{Entropía H de g<br/>segunda evaluación}
    K -->|H baja ahora| L[Clasificación normal<br/>por experto de dominio]
    K -->|H sigue alta| M[Deconstruir a imagen original]

    M --> N["⚠️ NECESITA REVISIÓN<br/>DE UN PROFESIONAL"]

    style G fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style N fill:#ffd43b,stroke:#e67700,color:#000
    style L fill:#69db7c,stroke:#2b8a3e,color:#000
    style C fill:#69db7c,stroke:#2b8a3e,color:#000
```

### 4.5 Clases de salida del Experto 6

El CAE produce exactamente dos etiquetas de salida:

| Etiqueta | Significado | Criterio |
| --- | --- | --- |
| **"basura"** | La imagen es completamente ajena a imagenología médica (foto de un gato, ruido aleatorio, captura de pantalla, etc.) | Error de reconstrucción muy alto — el CAE no puede reconstruir algo que nunca vio |
| **"necesita revisión"** | La imagen es legítimamente médica pero está defectuosa (borrosa, corrupta, con artefactos graves) y tras limpieza sigue siendo ambigua para el router | Error moderado + re-routing fallido |

### 4.6 Entrenamiento del CAE

- **Datos de entrenamiento:** los 5 datasets de dominio combinados (no un dataset aparte).
- **Objetivo:** minimizar el error de reconstrucción sobre imágenes médicas legítimas.
- **Resultado:** el CAE aprende la distribución de "imágenes médicas válidas". Cualquier imagen fuera de esta distribución produce un error de reconstrucción alto.
- **Fase del pipeline:** Fase 3 (después de entrenar los 5 expertos de dominio).

**Notas adicionales sobre el entrenamiento:**
- El CAE se entrena con los 5 datasets combinados para aprender la distribución de imágenes médicas válidas.
- **Durante el entrenamiento, el CAE no recibe imágenes por el router.** Solo en inferencia recibe imágenes cuando el router tiene entropía alta.
- **Riesgo del "router perezoso":** si el CAE aprende demasiado bien de todo tipo de imagen, el router se volvería perezoso y le enviaría todo al Experto 6 porque siempre daría métricas positivas. Esto colapsa la lógica MoE. La `L_error` en la función de pérdida total existe precisamente para penalizar este comportamiento.

**Estado del script de entrenamiento (Paso 5.6, 2026-04-05):**

| Campo | Detalle |
| --- | --- |
| **Estado** | ✅ Script implementado — dry-run verificado (exit_code=0, train=130,002 / val=15,959) |
| **Modelo** | `ConvAutoEncoder` 2D, ~206M params (206,464,771). Encoder: 3× Conv2d(stride=2)+BN+ReLU → Linear(200704, 512). Decoder: Linear(512, 200704) → 3× ConvTranspose2d → Sigmoid |
| **Archivos** | `expert5_cae_config.py`, `models/expert5_cae.py`, `dataloader_cae.py`, `train_cae.py`, `datasets/cae.py` |
| **Checkpoint** | `checkpoints/expert_05_cae/cae_best.pt` |
| **Dataset** | `cae_splits.csv` (162,611 filas). `__getitem__` → `(tensor [3,224,224], path_str)` — sin label. 3D dispatch: LUNA→axial central, Páncreas→HU clip+central slice |
| **Config** | LR=1e-3, WD=1e-5, batch=32, **FP32 obligatorio** (NO FP16), patience=15, max_epochs=100 |
| **Loss** | `MSE(recon, x) + 0.1 · L1(recon, x)` |
| **Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=5)` |
| **Nota** | expert_id=5, excluido de load balancing, activado por entropía H(g)>P95, extensión verbal (no en proyecto_moe.md) |

### 4.7 Feedback loop con el router

Durante el entrenamiento del router (Fase 1 y Fase 5), el sistema incluye un término de pérdida `L_error` que penaliza al router cuando delega una imagen médica válida al Experto 6. Este feedback entrena al router a **no enviar** imágenes legítimas al CAE.

```
L_total = L_task + α·L_aux + β·L_error
```

- `L_task`: pérdida de clasificación del experto seleccionado
- `L_aux`: pérdida auxiliar de balance de carga (Switch Transformer)
- `L_error`: penalización por delegar imágenes válidas al Experto 6
- `β` se calibra para que el Experto 6 solo reciba OOD real

---

## 5. Diseño del router

### 5.1 Entrada y salida

- **Entrada:** vector de embedding `z ∈ ℝ^d_model` extraído por el backbone compartido (CLS token)
- **Salida:** distribución de probabilidad sobre los expertos: `g = softmax(W·z + b) ∈ ℝ^6`
- **Sin metadatos:** el router opera exclusivamente sobre la representación aprendida de los píxeles

### 5.2 Lógica de entropía

La entropía de Shannon de la distribución de routing determina la acción:

```
H(g) = -Σ g_i · log(g_i)
```

| Condición | Interpretación | Acción |
| --- | --- | --- |
| `H(g)` baja | El router está seguro de un experto | Delegar al `argmax(g)` (experto 0–4) |
| `H(g)` alta | El router no distingue entre expertos | Delegar al experto 5 (CAE) |

**Calibración del umbral:** percentil 95 de `H(g)` sobre el set de validación (`ENTROPY_PERCENTILE = 95` en `fase2_config.py`). El 5% de las muestras más confusas se tratan como OOD en inferencia.

### 5.3 Ablation study: 4 métodos de routing

> **Estado:** ✅ Implementado y dry-run verificado (2026-04-05). Script: `src/pipeline/fase2/fase2_pipeline.py`. Documentación completa: `docs/documentacion_pasos/paso_06_ablation_study.md`.

El núcleo científico del proyecto compara 4 routers sobre los mismos embeddings:

| Router | Tipo | Parámetros clave | Naturaleza |
| --- | --- | --- | --- |
| **Linear + Softmax** | Paramétrico (gradiente) | LR=1e-3, 50 épocas, batch=512, α_L_aux=0.01 | Deep Learning |
| **GMM** | Paramétrico (EM) | 5 componentes, covarianza `full` (fallback `diag`), max_iter=200 | Estadístico |
| **Naive Bayes** | MLE analítico | GaussianNB — solución cerrada, sin iteraciones | Estadístico |
| **k-NN (FAISS)** | No paramétrico | k=5, distancia coseno (IndexFlatIP + L2 norm), suavizado Laplace ε=0.01 | Estadístico |

**Input:** embeddings pre-computados de Fase 1 (`Z_train.npy`, `Z_val.npy`), d_model=192 (ViT-Tiny), N_EXPERTS_DOMAIN=5.

**Winner selection (dos pasos):**
1. Filtrar: descartar routers con `load_balance > LOAD_BALANCE_THRESHOLD` (1.30) — riesgo de expert collapse
2. Seleccionar: `max(accuracy_val)` entre los que pasan el filtro
3. Si todos fallan → seleccionar menor ratio load_balance (con warning)

**Output:** `ablation_results.json` (comparación completa + ganador + metadata) + modelo ganador serializado (`best_router_{type}.{ext}`).

**Modo `--dry-run`:** ejecuta el ablation completo con embeddings sintéticos (200 train, 40 val, d=192) y parámetros reducidos (Linear: 2 épocas, GMM: 5 iter). No guarda artefactos a disco. Verificado con exit code 0.

**Protocolo del ablation:**
1. Fase 1 extrae embeddings con el backbone congelado → `Z_train`, `Z_val`, `Z_test` en disco
2. Cada router entrena/ajusta sobre `Z_train`
3. Se evalúa routing accuracy + balance de carga + latencia sobre `Z_val`
4. El ganador se selecciona por routing accuracy **solo si** cumple la restricción de balance de carga
5. Resultados se guardan en `ablation_results.json`
6. Calibración de α: barrer α ∈ {0.01, 0.02, 0.05, 0.10} sobre Z_val monitoreando el cociente max(f_i)/min(f_i). Registrar el α óptimo en ablation_results.json.

> **Importante sobre el balance de carga en el ablation:** La restricción `max(f_i)/min(f_i) ≤ 1.30` y la penalización de −40% se aplican **exclusivamente al router ViT+Linear en el sistema final**. Los routers estadísticos (GMM, Naive Bayes, k-NN) **no tienen L_aux** y no están sujetos a esta penalización durante el ablation study. En el ablation se monitorea el balance como métrica comparativa, pero no descalifica a un router estadístico.

### 5.4 Restricción de balance de carga

```
max(f_i) / min(f_i) ≤ 1.30
```

Donde `f_i` es la fracción de muestras asignadas al experto `i` (sobre los expertos 0–4 de dominio).

- En el ablation study, un router estadístico que viole este umbral **se penaliza en el ranking** pero no es automáticamente descalificado (no tienen L_aux). El router ViT+Linear **sí está sujeto a descalificación** si viola el umbral en el sistema final desplegado.
- Violación en el sistema final: **−40% en la nota del proyecto**.
- La Auxiliary Loss del Switch Transformer (`L_aux = α · N · Σ f_i · P_i`) se usa en el router Linear para forzar el balance durante el entrenamiento.

### 5.5 Backbones disponibles

| Backbone | `d_model` | VRAM | Uso previsto |
| --- | --- | --- | --- |
| `vit_tiny_patch16_224` (default) | 192 | ~2GB | Primera corrida, iteración rápida |
| `cvt_13` | 384 | ~3GB | Balance intermedio |
| `swin_tiny_patch4_window7_224` | 768 | ~4GB | Ablation study final |
| `densenet121_custom` | 1024 | ~3GB | Imágenes 2D — recomendado por el profesor para imagenología médica |

### 5.6 Fases de entrenamiento del router

| Fase | Qué se entrena | Qué está congelado | LR | Épocas |
| --- | --- | --- | --- | --- |
| Fase 1 (Router solo) | Router (W, b) | Backbone + 5 expertos | 1e-3 | ~50 |
| Fase 2 (Cabezas) | Router + cabezas clasificadoras | Backbone + capas convolucionales | 1e-4 | ~30 |
| Fase 3 (Global) | Todo (Backbone + Router + Expertos) | Nada | 1e-6 | 7–10 |

**Regla crítica en Fase 3:** NO reiniciar el optimizer — se pierden los momentos acumulados de fases anteriores.

> **Nota de diseño (divergencia consciente con la guía):** La guía oficial propone LR=1e-5 en la Fase 3 global. Este proyecto usa LR=1e-6 de forma deliberada: dado que el sistema integra 6 expertos heterogéneos (2D y 3D), un learning rate más bajo en la fase global minimiza el riesgo de desestabilizar los expertos ya convergidos, especialmente los modelos 3D con gradientes más inestables. Se acepta convergencia más lenta a cambio de mayor estabilidad.

**Estado de implementación (2026-04-05):** ✅ Implementado — dry-run verificado
- Script: `src/pipeline/fase5/fase5_finetune_global.py`
- Freeze utilities: `src/pipeline/fase5/freeze_utils.py`
- Config: `src/pipeline/fase5/fase5_config.py`
- Divergencia LR consciente documentada en config: `STAGE3_LR_EXPERTS=1e-6` (guía: 1e-5)
- Documentación completa: `docs/documentacion_pasos/paso_08_finetune_por_etapas.md`

### 5.7 Paso 9 — Prueba con datos reales (Fase 6)

**Estado:** Implementado ✅ — dry-run OK
**Script orquestador:** `src/pipeline/fase6/fase9_test_real.py`
**Fase pipeline:** fase6

**Componentes:**
- `fase6_config.py` — constantes, umbrales de métricas, rutas
- `inference_engine.py` — forward de inferencia (sin expert_id)
- `test_evaluator.py` — evaluador batch por experto
- `gradcam_heatmap.py` — generación Grad-CAM/Attention Rollout
- `ood_detector.py` — calibración umbral entropía + AUROC OOD
- `fase9_test_real.py` — orquestador principal

**Forward de inferencia (brecha crítica cerrada):**
La diferencia entre entrenamiento e inferencia:
- Entrenamiento: `forward(x, expert_id)` — expert_id conocido (supervisado)
- Inferencia: backbone(x) → router(z) → H = -Σg·log(g+ε) → if H>θ: CAE else experts[argmax(g)]

**Prerequisitos bloqueantes para ejecución real:**
- Paso 4.2: embeddings generados
- Pasos 5.1–5.6: checkpoints de todos los expertos
- Paso 8: `checkpoints/fase5/moe_final.pt`

**Documentación completa:** `docs/documentacion_pasos/paso_09_prueba_datos_reales.md`

---

## 6. Pipeline de preprocesamiento de imagen

### 6.1 Formato de entrada: `.mha`

El formato preferido para imágenes médicas es `.mha` (MetaImage), porque:
- Preserva el **volumen 3D interno** completo.
- Soporta el rango de **Unidades Hounsfield (HU): −1000 a +3000**, muy superior al rango 0–255 de JPG/RGB.
- A diferencia de JPG o PNG (3 dimensiones máx.), `.mha` permite trabajar con la profundidad real del volumen sin destruirla.

> ⚠️ **Advertencia crítica:** pasar la 3ª dimensión del vóxel directamente al canal RGB destruye el volumen médico. Esta estrategia está **prohibida** en este proyecto.

---

### 6.2 Pipelines de preprocesamiento per-dataset

Cada dataset utiliza un pipeline de preprocesamiento específico, adaptado a la modalidad de imagen y a las características diagnósticas relevantes. Un pipeline uniforme (aplicar CLAHE + TVF + Gamma a todos los datasets) sería clínicamente incorrecto: cada modalidad tiene señales diagnósticas distintas que requieren tratamiento diferenciado.

#### 6.2.1 NIH ChestXray14 — Radiografía de tórax (escala de grises)

| Paso | Técnica | Justificación clínica |
|---|---|---|
| 1 | **CLAHETransform** | Realza contraste en campos pulmonares; mejora visibilidad de infiltrados y nódulos en escala de grises. Recomendación directa del profesor. |
| 2 | **Resize(224×224)** | Estandarización espacial sin interpolación agresiva. |
| 3 | **TotalVariationFilter (TVF)** | Elimina ruido del sensor de rayos X sin borrar estructuras diagnósticas (bordes pulmonares, silueta cardíaca). |
| 4 | **ToTensor** | Conversión a tensor PyTorch. |
| 5 | **Normalize** (ImageNet stats) | Normalización de rango para el backbone. |

**Pipeline:** `CLAHETransform → Resize → TVF → ToTensor → Normalize`

> **Nota (INC-04):** El paso de `GammaCorrection(γ=1.0)` que existía en la implementación original es un **no-op** (γ=1.0 devuelve la imagen sin modificar). Se omite aquí porque no aporta transformación alguna. Si se desea activar, debe calibrarse γ ≠ 1.0 con validación empírica.

#### 6.2.2 ISIC 2019 — Dermatoscopía (imágenes color)

| Paso | Técnica | Justificación clínica |
|---|---|---|
| 1 | **BCNCrop** | Crop específico del protocolo BCN que elimina los bordes negros del dispositivo dermatoscópico, los cuales no contienen información diagnóstica. |
| 2 | **Resize(224×224)** | Estandarización espacial. |
| 3 | **ToTensor** | Conversión a tensor PyTorch. |
| 4 | **Normalize** (ImageNet stats) | Normalización de rango para el backbone. |

**Pipeline:** `BCNCrop → Resize → ToTensor → Normalize`

**¿Por qué NO se aplican CLAHE ni TVF en dermatoscopía?**
- **CLAHE ❌:** En dermatoscopía, la información diagnóstica principal es el **color** (regla ABCD: Asymmetry, Border, Color, Diameter). CLAHE opera sobre luminancia y distorsiona los gradientes cromáticos que son señal diagnóstica — un melanoma se distingue por heterogeneidad de color, no por contraste de grises.
- **TVF ❌:** El filtro de variación total suavizaría los gradientes de color heterogéneo que son precisamente la señal diagnóstica en lesiones pigmentadas. Aplicarlo eliminaría las transiciones de color que diferencian melanoma de nevus benigno.

#### 6.2.3 OsteoArthritis KneeXray — Radiografía de rodilla (escala de grises)

| Paso | Técnica | Justificación clínica |
|---|---|---|
| 1 | **CLAHETransform** | Realza estructura ósea y espacio articular; mejora la diferenciación visual para gradación Kellgren-Lawrence (KL). |
| 2 | **Resize(224×224, BICUBIC)** | Interpolación bicúbica preserva mejor los detalles óseos finos (osteofitos, esclerosis subcondral) que la interpolación bilineal por defecto. |
| 3 | **ToTensor** | Conversión a tensor PyTorch. |
| 4 | **Normalize** (ImageNet stats) | Normalización de rango para el backbone. |

**Pipeline:** `CLAHETransform → Resize(BICUBIC) → ToTensor → Normalize`

#### 6.2.4 Resumen comparativo y justificación

| Técnica | NIH Chest | ISIC Derm | OA Knee | Razón de la diferencia |
|---|---|---|---|---|
| CLAHE | ✅ | ❌ | ✅ | CLAHE beneficia imágenes en escala de grises (rayos X) pero daña la señal de color en dermatoscopía |
| TVF | ✅ | ❌ | ❌ | TVF elimina ruido de sensor; en OA Knee no se aplica porque la señal ósea no tiene ruido de alta frecuencia comparable |
| BCNCrop | ❌ | ✅ | ❌ | Solo aplica a imágenes de dermatoscopio con bordes de dispositivo |
| Resize BICUBIC | ❌ | ❌ | ✅ | Los detalles óseos finos requieren interpolación de mayor calidad |
| GammaCorrection | ⚠️ no-op | ❌ | ❌ | γ=1.0 es identidad; requiere calibración (ver INC-04) |

> **Conclusión:** Los pipelines per-dataset son **superiores** al pipeline uniforme porque cada modalidad de imagen tiene características diagnósticas distintas que requieren preprocesamiento específico. Un pipeline uniforme aplicaría CLAHE/TVF a imágenes de dermatoscopía, lo cual dañaría la señal diagnóstica de color.

#### 6.2.5 Reglas transversales (aplican a todos los pipelines)

| Paso | Técnica | Justificación |
|---|---|---|
| — | **Guardar el `transform`** antes de normalizar | Permite recuperar el factor de redimensionalidad para generar mapas de calor (Grad-CAM). Requerido por Funcionalidad 4 del dashboard (§8.1). |
| — | **Generar tensor 5D normalizado** | Soporta imágenes 2D y 3D simultáneamente. La profundidad varía entre sistemas pero la dimensionalidad se mantiene consistente. |
| — | **Pasar por cada backbone de forma independiente** | Cada modelo (ViT, CvT, Swin, DenseNet) recibe su propio tensor redimensionado. |

> **Regla del profesor:** "Los filtros redimensionan la imagen → guardar el `transform` antes de normalizar embeddings. Un método que normalice Y que a la salida permita recuperar el factor de redimensionalidad para generar mapas de calor."

---

### 6.3 Tensor 5D y detección de dimensionalidad

La guía oficial establece que la detección 2D/3D es automática por el **rank del tensor de entrada**:
- `rank=4` → imagen 2D: `[B, C, H, W]`
- `rank=5` → volumen 3D: `[B, C, D, H, W]`

Esta detección opera en la **interfaz de entrada del preprocesador** (antes de cualquier transformación). Internamente, una vez detectada la dimensionalidad, el sistema normaliza la representación a un **tensor 5D unificado** para el procesamiento:

```
Entrada 2D (rank=4):  [B, C, H, W]      →  expande D=1  →  [B, C, 1, H, W]
Entrada 3D (rank=5):  [B, C, D, H, W]   →  se mantiene  →  [B, C, D, H, W]
```

- La detección sigue el criterio de la guía (rank del tensor de entrada).
- El procesamiento interno usa siempre estructura 5D para código unificado entre 2D y 3D.
- Al extraer embeddings, cada backbone recibe el tensor en el formato que le corresponde.

---

### 6.4 Tabla de input por backbone

| Modelo | Origen | Input recomendado | Tipo de embeddings |
|---|---|---|---|
| **ViT** | Desde cero (arq. predefinida) | Parches 2D de slices del volumen | Transformer sobre secuencia de patches |
| **CvT** | Desde cero (arq. predefinida) | Convolutions + Transformer híbrido | Captura local + global |
| **Swin** | Desde cero (arq. predefinida) | Ventanas deslizantes sobre slices | Eficiente para volúmenes grandes |
| **DenseNet** | Arquitectura **propia/custom** | Imagen 2D por canal/slice | Embeddings ricos con conexiones densas |

> **Nota del profesor:** "Hacer embeddings ricos es bueno." No depender solo de ImageNet. DenseNet custom + TVF + Gamma = mejor resultado según PMC9340712.

---

### 6.5 Data Leak Prevention

Estrategia en dos niveles para evitar filtraciones entre splits:

1. **Primer nivel — Metadata DICOM (`.dcm`):**
   - El formato DICOM contiene metadata + imagen en un solo archivo.
   - Comparar primero por metadata (rápido, barato computacionalmente).
   - Si la metadata falla o es idéntica, pasar al siguiente nivel.

2. **Segundo nivel — PHash (Perceptual Hashing):**
   - Reduce dimensiones con coseno discreto.
   - Extrae frecuencias bajas/bordes del contenido.
   - Genera una huella hexadecimal corta para comparación eficiente.
   - Alternativas: histograma de píxeles o comparación píxel a píxel (más exacto pero más costoso).

---

### 6.6 Aumento de datos sintéticos (técnica del profesor)

El profesor propone una técnica propia para generación de imágenes sintéticas de alta precisión:

1. Entrenar un **encoder-decoder** completo.
2. Descartar el encoder entrenado.
3. Conectar **Smooth** (genera vectores tabulares a partir de distribuciones) al decoder.
4. Los vectores tabulares de Smooth pasan al decoder para **recrear imágenes sintéticas nuevas** muy precisas.

Esta técnica produce imágenes médicas sintéticas superiores a augmentación estándar (flip, crop, rotate) para datos de baja disponibilidad (ej: Páncreas con ~281 volúmenes).

---

## 7. Pipeline de datos (Fase 0)

### 7.1 Pasos de Fase 0

| Paso | Descripción | Estado | Tiempo |
| --- | --- | --- | --- |
| 0 | Prerrequisitos (SimpleITK, etc.) | ✅ | 0.0s |
| 1 | Descargar datasets | ✅ | 8,773.7s (~2.4h) |
| 2 | Extraer archivos | ✅ | 0.4s |
| 3 | Post-procesado NIH | ✅ | 6.1s |
| 4 | Etiquetas páncreas | ✅ | ~4,205s (70 min) — re-ejecutado con fix OOM |
| 5 | Splits 80/10/10 | ⚠️ parcial | 3.0s |
| 6 | Datos 3D (parches) | ⚠️ parcial | 162.2s |
| 7 | Parche CvT-13 | ⚠️ | 7.3s |
| 8 | Reporte | — | — |

### 7.2 Estado actual de los datos en disco (2026-04-05)

| Dataset | Descargado | Extraído | Splits | Procesado |
| --- | --- | --- | --- | --- |
| NIH ChestXray14 | ✅ | ✅ | ✅ `splits/nih_*.txt` | N/A |
| ISIC 2019 | ✅ | ✅ | ✅ `splits/isic_*.csv` | N/A |
| OA Knee | ✅ | ✅ | ✅ `oa_splits/{train,val,test}/` | ✅ Listo — 4,766 imgs KL-graded (9,339 total en disco), remapeo KL0→0/KL1+2→1/KL3+4→2 documentado |
| LUNA16 | ✅ (meta+CT) | ✅ | ✅ `luna_splits.json` | ✅ Listo — ZIPs eliminados (~62 GB), no bugs, `_LEAKED_DO_NOT_USE/` (ex-`train_stale_backup/`, 1,839 parches con leakage). `annotations.csv`: 1,186 ✅. `candidates_V2.csv`: 754,975 ✅. `seg-lungs-LUNA16/`: 441 MB presente (no usado por pipeline). Licencia: CC BY 4.0. |
| Pancreas | ✅ | ✅ | ✅ `pancreas_splits.csv` | ✅ Listo — todos los bugs corregidos (ruta + OOM + leakage GroupKFold). `pancreas_labels_binary.csv`: 1,864 filas (1,756 PDAC+, 108 PDAC−). `pancreas_splits.csv`: 1,864 casos, 186 test, 0 leakage. `cae_splits.csv` regenerado (162,611 filas). A19 resuelto: `_build_pairs()` omite casos sin CT silenciosamente. |
| PANORAMA labels | ✅ (clonado) | N/A | N/A (etiquetas auxiliares) | N/A — 2,238 label files (1,756 auto + 482 manual) |

### 7.3 Bloqueantes

**Páncreas:** ✅ Todos los bloqueantes resueltos. Datos descargados (557 `.nii.gz`), labels regenerados (1,864 filas, 0 errores), splits con `GroupKFold` sin leakage, `cae_splits.csv` regenerado. A19 resuelto: `_build_pairs()` omite silenciosamente los 1,307 casos sin CT (guard `if candidates:`). **Paso 1 — 19/19 items resolved ✅**.

**LUNA16 test:** ✅ Resuelto — el directorio `patches/test/` existe con 1,914 parches.

### 7.4 Splits generados

| Dataset | Train | Val | Test | Método de split |
| --- | --- | --- | --- | --- |
| NIH | 88,999 | 11,349 | 11,772 | Patient ID (listas oficiales) |
| ISIC | — | — | — | `lesion_id` (`build_lesion_split`) |
| OA | 3,814 | 480 | 472 | Grupo de similitud visual 80/10/10 (proxy de patient_id) |
| LUNA | 14,728 | 1,143 | 1,914 | `seriesuid` (leakage fix aplicado 2026-04-02) |
| Pancreas | ~1,342 | ~336 | 186 | k-fold CV (k=5) por `patient_id` (`GroupKFold`). A19 resuelto: `_build_pairs()` omite los 1,307/1,864 casos sin CT silenciosamente. |
| CAE | ✅ (skipped) | — | — | Combinación de los 5 datasets |

### 7.5 Artefactos producidos por Fase 0 → consumidos por Fase 1

Fase 1 es consumidora pura de los artefactos de Fase 0:
- Splits (archivos `.txt`, `.csv`, directorios)
- Parches 3D preprocesados (`.npy`)
- Etiquetas cruzadas (páncreas: `pancreas_splits.csv`)
- Manifiesto CAE: `cae_splits.csv` — 162,611 filas (NIH:112,120 | ISIC:25,331 | OA:4,766 | LUNA:17,785 | Pancreas:2,609). Consumido por Fase 3 (pendiente de implementación).
- Reporte de estado (`fase0_report.md`)
- Comando de ejecución para Fase 1 (generado en el paso 8)

---

## 8. Dashboard interactivo (Fase 6)

### 8.1 Funcionalidades obligatorias (requeridas por la guía)

La guía oficial establece explícitamente las siguientes 8 funcionalidades:

| # | Funcionalidad | Descripción | Estado |
|---|---|---|---|
| 1 | **Carga de imagen** | Soporta PNG, JPEG, NIfTI. Detecta automáticamente 2D/3D por rank del tensor. | ✅ Implementado |
| 2 | **Preprocesado transparente** | Muestra dimensiones originales y adaptadas tras el preprocesador. | ✅ Implementado |
| 3 | **Inferencia en tiempo real** | Devuelve etiqueta, confianza (%) y tiempo de respuesta en milisegundos. | ✅ Implementado |
| 4 | **Attention Heatmap del Router ViT** | Mapa de calor sobre la imagen original. Usa Grad-CAM con el `transform` guardado en el paso 5 del preprocesamiento. | ✅ Implementado |
| 5 | **Panel del experto activado** | Muestra nombre del experto, arquitectura, dataset de entrenamiento y gating score asignado. | ✅ Implementado |
| 6 | **Panel del ablation study** | Tabla comparativa de Routing Accuracy de los 4 métodos sobre el set de validación. | ✅ Implementado |
| 7 | **Gráfica de Load Balance** | Barras con `f_i` acumulado desde que arrancó el sistema, en tiempo real. | ✅ Implementado |
| 8 | **Alerta OOD** | Alerta visual cuando la entropía H(g) > umbral calibrado (percentil 95). Indica si la imagen fue al Experto 6 (CAE). | ✅ Implementado |

### 8.2 Conexión con el pipeline

- **Funcionalidad 4 (Heatmap):** requiere que el `transform` del preprocesamiento esté guardado por cada inferencia (paso 5 de §6.2). Sin este artefacto, Grad-CAM no puede mapear la activación al espacio original de la imagen.
- **Funcionalidad 6 (Ablation):** se alimenta de `ablation_results.json` generado en Fase 4.
- **Funcionalidad 7 (Load Balance):** se alimenta de un contador en memoria actualizado en cada inferencia. Se resetea al reiniciar el servicio.
- **Funcionalidad 8 (OOD):** se alimenta del umbral calibrado en `fase4_ablation.py` (percentil 95 de H(g)).

### 8.3 Artefactos previos necesarios para el dashboard

| Artefacto | Generado en | Consumido por |
|---|---|---|
| `ablation_results.json` | Fase 4 | Funcionalidad 6 |
| `entropy_threshold.pkl` | Fase 4 | Funcionalidad 8 |
| `preprocessing_transform.pkl` (por inferencia) | Preprocesador (§6.2) | Funcionalidad 4 |
| Pesos de los 6 expertos + router | Fases 2, 3, 5 | Funcionalidades 3, 5 |

---

## 9. Optimización de hardware

### 9.1 Configuración de hardware de trabajo

| Característica | Valor |
|---|---|
| **GPU de trabajo** | 1 × 20 GB VRAM (máquina personal) |
| **GPU de referencia (guía)** | 2 × 12 GB VRAM (máquina del profesor) |
| **Estrategia** | Desarrollar y validar en 1×20 GB. Si se migra a 2×12 GB, activar `nn.DataParallel`. |

> **Decisión:** Se trabaja primero (y posiblemente de forma exclusiva) con la máquina personal de 1×20 GB. La guía fue redactada para 2×12 GB, pero 20 GB unificados ofrecen mayor flexibilidad para modelos 3D sin necesidad de DataParallel.

### 9.2 Técnicas obligatorias de optimización de memoria

| Técnica | Obligatoria | Configuración | Aplica a |
|---|---|---|---|
| **FP16 Mixed Precision** | Sí | `torch.cuda.amp.autocast()` | Todos los modelos |
| **Gradient Accumulation** | Sí | `ACCUMULATION_STEPS=4`, batch efectivo=32 | Todos los entrenamientos |
| **Gradient Checkpointing** | Sí (3D) | `model.gradient_checkpointing_enable()` | Expertos 3D (LUNA16, Páncreas) |
| **DataParallel** | Condicional | `nn.DataParallel(model)` | Solo si se migra a 2×12 GB |

> **Gradient Accumulation:** con batch real de 8 imágenes y `ACCUMULATION_STEPS=4`, el sistema simula un batch efectivo de 32 sin cargar 32 imágenes simultáneamente en GPU. Crítico para modelos 3D con tensores grandes.

### 9.3 Presupuesto VRAM estimado por fase

| Componente | VRAM estimada | Técnica de control |
|---|---|---|
| Backbone ViT-Tiny (d=192) | ~2 GB | — |
| Backbone Swin-Tiny (d=768) | ~4 GB | — |
| Experto 3D LUNA16 (MC3-18, batch=4) | ~8–10 GB | FP16 + grad. checkpoint |
| Experto 3D Páncreas (Swin3D, batch=1–2) | ~10–12 GB | FP16 + grad. checkpoint |
| Sistema MoE completo (Fase 5) | ~15–18 GB | FP16 + grad. accum. + checkpoint |

> El límite objetivo es **< 19 GB** para dejar margen de seguridad en la GPU de 20 GB. La guía menciona < 11.5 GB por tarjeta para 2×12 GB — en nuestra configuración el límite efectivo es ~19 GB total.

---

## 10. Métricas de evaluación y umbrales mínimos

### 10.1 Umbrales obligatorios de la rúbrica

Los siguientes umbrales aparecen en la rúbrica de evaluación del proyecto (§13 de `proyecto_moe.md`) y son **criterios de calificación**, no valores de referencia:

| Métrica | Umbral "Full marks" | Umbral "Aceptable" | Aplica a |
|---|---|---|---|
| **F1 Macro — Expertos 2D** | > 0.72 | > 0.65 | Chest, ISIC, OA-Knee |
| **F1 Macro — Expertos 3D** | > 0.65 | > 0.58 | LUNA16, Páncreas |
| **Routing Accuracy** | > 0.80 (mejor router) | — | Ablation study (Fase 4) |
| **Load Balance max/min** | < 1.30 | — | Router ViT+Linear en producción (**penalización −40%** si supera) |

### 10.2 Métrica OOD — umbral a validar empíricamente

La guía fija un umbral orientativo de **OOD AUROC > 0.80** pero no lo incluye en la rúbrica de calificación como criterio de penalización. Sin embargo, dado que el Experto 6 (CAE) es una extensión aprobada verbalmente, se establece este umbral como objetivo interno:

> **Compromiso:** el AUROC del Experto 6 como detector OOD debe justificarse empíricamente sobre un conjunto de imágenes no médicas (OOD sintéticas) vs. imágenes médicas del set de validación. El valor de 0.80 es el umbral mínimo aceptable internamente; si no se alcanza, se revisará la arquitectura del CAE.

### 10.3 Restricción de VRAM

La guía menciona `< 11.5 GB por tarjeta` para la configuración 2×12 GB. En nuestra configuración de **1×20 GB**, el objetivo equivalente es:

> **< 19 GB total de VRAM** durante el entrenamiento del sistema completo (Fase 5). Superarlo implica ajustar batch size, reducir `d_model` del backbone, o activar gradient accumulation más agresivo.

### 10.4 Resultados orientativos del ablation (referencia, no umbrales)

La guía incluye una tabla de resultados **aproximados esperados** bajo el título *"Tabla Comparativa Esperada (Guía de Resultados)"*. Son rangos orientativos, **no umbrales evaluables**:

| Router | Routing Acc. esperada | Latencia esperada | VRAM esperada |
|---|---|---|---|
| Linear + Softmax (ViT) | ~85–92% | ~8 ms | ~1.5 GB |
| GMM | ~78–87% | ~3 ms | ~50 MB |
| Naive Bayes | ~72–82% | ~1 ms | ~5 MB |
| k-NN (FAISS) | ~75–85% | ~5 ms | ~85 MB |

> La guía establece explícitamente: *"No hay un resultado correcto predefinido. Lo que se evalúa es la calidad del experimento, la honestidad del análisis y la profundidad de la discusión."*

### 10.5 Preparación del reporte técnico por fases

Para facilitar la redacción del reporte técnico final (≤ 7 páginas), cada fase del pipeline debe generar un **mini-reporte de resultados** con:

| Fase | Artefacto de reporte | Contenido mínimo |
|---|---|---|
| Fase 0 | `fase0_report.md` | Estado de datos, splits, bloqueantes |
| Fase 1 | `fase1_report.md` | Dimensión de embeddings, tiempo extracción, distribución por clase |
| Fase 2 | `fase2_report.md` | Métricas por experto (F1, AUC), curvas de aprendizaje, comparación con umbrales §10.1 |
| Fase 3 | `fase3_report.md` | Error de reconstrucción del CAE, distribución OOD vs. in-distribution |
| Fase 4 | `fase4_report.md` | Tabla comparativa 4 routers, Routing Accuracy, balance de carga, α óptimo |
| Fase 5 | `fase5_report.md` | Métricas end-to-end, curvas fine-tuning, análisis de load balance final |

> Estos mini-reportes son la materia prima del reporte técnico formal. Generarlos durante el desarrollo evita tener que reconstruir resultados al final.

---

## 11. Restricciones de penalización

El proyecto tiene dos restricciones de evaluación con penalización explícita sobre la nota.

### 11.1 Restricción 1 — Solo píxeles (−20%)

| Campo | Detalle |
| --- | --- |
| **Restricción** | El sistema recibe **únicamente** la imagen o volumen como entrada |
| **Prohibido** | Metadatos, etiquetas de modalidad, nombres de archivo, texto, información de dataset |
| **Detección 2D/3D** | Automática por rango del tensor (`rank=4` → 2D, `rank=5` → 3D) |
| **Penalización** | −20% sobre la nota final del proyecto |
| **Alcance** | Todo el pipeline: preprocesado, backbone, router, expertos, inferencia, dashboard |

**Implicación práctica:** ningún componente del sistema puede recibir una variable que indique "esta imagen viene del dataset X" o "esta imagen es una radiografía de tórax". El router debe descubrirlo solo a partir del embedding del backbone.

### 11.2 Restricción 2 — Balance de carga (−40%)

| Campo | Detalle |
| --- | --- |
| **Restricción** | `max(f_i) / min(f_i) ≤ 1.30` sobre los expertos de dominio (0–4) |
| **Definición** | `f_i` = fracción de muestras asignadas al experto `i` |
| **Penalización** | −40% sobre la nota final del proyecto |
| **Mecanismo de control** | Auxiliary Loss del Switch Transformer: `L_aux = α · N · Σ f_i · P_i` |
| **Parámetro** | `LOAD_BALANCE_THRESHOLD = 1.30` en `fase2_config.py` |
| **Alcance** | Solo el router ViT+Linear en producción. Los routers estadísticos del ablation no tienen L_aux. |

**Implicación práctica:** si el router envía el 40% de las imágenes al experto Chest y solo el 10% al experto LUNA16, la ratio sería 4.0, violando masivamente el umbral. La L_aux penaliza activamente este desequilibrio durante el entrenamiento del router Linear.

**Nota:** el experto 6 (CAE/OOD) no cuenta para el cálculo de balance de carga. Solo los expertos 0–4 de dominio participan en la ratio.

---

*Documento generado por MNEMON — agente de memoria del proyecto MoE Médico.Fuentes: `config.py`, `fase2_config.py`, `fase0_report.md`, `arquitectura.md`, `flujo_arquitectura.md`, `proyecto_moe.md`, especificación del arquitecto.*
