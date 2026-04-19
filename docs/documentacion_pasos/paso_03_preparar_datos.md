# Paso 3 — Preparar Datos: Auditoría Completa

| Campo | Valor |
|---|---|
| **Fecha de auditoría** | 2026-04-05 |
| **Auditor** | Multi-agente: ARGOS (alineación guía↔código), SIGMA (verificación de transforms), EXPLORE (inspección de source code) |
| **Alcance** | Splits, transforms, y augmentation pipelines de los 5 datasets de dominio + manifiesto CAE global |
| **Estado general** | ⚠️ **Paso 3 PARCIALMENTE COMPLETO — Splits ✅, Transforms ✅ inconsistencias resueltas (INC-01 a INC-05, 2026-04-05), Augmentations ✅ BUG-C1/C2 resueltos (2026-04-05), Embeddings ⏳ stale eliminados — regeneración pendiente.** |
| **Commit del proyecto** | `948cd78b6de16a53d7220b92ce8863ba3d910edc` |

---

## 1. Resumen ejecutivo

Los splits de los cinco datasets están correctamente generados y verificados (sin leakage, ratios correctos, conteos consistentes). Los tres bugs críticos han sido resueltos (2026-04-05): **BUG-C1** y **BUG-C2** — que aplicaban augmentaciones de entrenamiento a datos de validación y test en ISIC 2019 y OA Knee — fueron corregidos añadiendo guard `self.split == "train"` en `__getitem__` de ambos datasets. **BUG-C3** — embeddings generados con parches LUNA contaminados — fue resuelto eliminando los 4 sets de embeddings stale (1,463 MB); la regeneración con `fase1_pipeline.py --force` está pendiente. Las **cinco incongruencias (INC-01 a INC-05)** han sido resueltas (2026-04-05): la documentación de transforms en `arquitectura_documentacion.md` §6.2 fue actualizada para reflejar los pipelines per-dataset (INC-01); NIH recibió augmentaciones de entrenamiento vía `build_2d_aug_transform()` (INC-02); Pancreas recibió augmentaciones 3D vía `_augment_3d()` para clase minoritaria (INC-03); el no-op `GammaCorrection(gamma=1.0)` se omite cuando γ=1.0 (INC-04); y OA ahora importa CLAHE de `transform_domain` como fuente canónica (INC-05).

---

## 2. Descripción del paso

| Campo | Valor |
|---|---|
| **Objetivo** | Generar splits train/val/test, definir pipelines de transformación por dataset, y configurar augmentaciones de datos para entrenamiento |
| **Archivos involucrados** | `pre_modelo.py` (splits), `chest.py`, `isic.py`, `osteoarthritis.py`, `luna.py`, `pancreas.py` (datasets), `transform_2d.py`, `transform_domain.py`, `transform_3d.py` (transforms) |
| **Dependencia directa** | Paso 2 (Extracción) — requiere datos extraídos en disco |
| **Fase del pipeline** | Fase 0 – Pasos 3–8 (splits y preparación pre-modelo) |
| **Definido en** | `arquitectura_documentacion.md` §6.2 (transforms), §7 (pipeline) |

---

## 3. Splits — Estado completo

### 3.1 Resumen de splits

| # | Dataset | Train | Val | Test | Total | Método | Leakage |
|---|---------|-------|-----|------|-------|--------|---------|
| 1 | NIH ChestXray14 | 88,999 | 11,349 | 11,772 | 112,120 | Patient ID (listas oficiales) | ✅ Ninguno |
| 2 | ISIC 2019 | 20,409 | 2,474 | 2,448 | 25,331 | `lesion_id` (`build_lesion_split`) | ✅ Ninguno |
| 3 | OA Knee | 3,814 | 480 | 472 | 4,766 | Grupo de similitud (fingerprint proxy) | ✅ Ninguno (heurístico) |
| 4 | LUNA16 (UIDs) | 712 | 88 | 88 | 888 | `seriesuid` (post leakage fix) | ✅ Corregido |
| 4b | LUNA16 (parches) | 14,728 | 1,143 | 1,914 | 17,785 | Derivado de UIDs | ✅ Corregido |
| 5 | Pancreas | 5-fold CV (~1,342/~336) | — | 186 | 1,864 | `GroupKFold` por `patient_id` | ✅ Corregido |

**Verificaciones aritméticas:**
- NIH: 88,999 + 11,349 + 11,772 = **112,120** ✅
- ISIC: 20,409 + 2,474 + 2,448 = **25,331** ✅
- OA: 3,814 + 480 + 472 = **4,766** ✅
- LUNA parches: 14,728 + 1,143 + 1,914 = **17,785** ✅
- Pancreas: 186 test + 1,678 pool CV = **1,864** ✅ (557 con CT en disco)

### 3.2 Manifiesto CAE global (`cae_splits.csv`)

| Campo | Valor |
|---|---|
| **Archivo** | `datasets/cae_splits.csv` |
| **Generado por** | `pre_modelo.py:947` → `build_cae_splits()` |
| **Total filas** | **162,611** |
| **Columnas** | `ruta_imagen`, `dataset_origen`, `split`, `expert_id`, `tipo_dato` |

| Dataset | Filas |
|---|---|
| NIH ChestXray14 | 112,120 |
| ISIC 2019 | 25,331 |
| OA Knee | 4,766 |
| LUNA16 | 17,785 |
| Pancreas | 2,609 |
| **Total** | **162,611** |

> Consumido por Fase 3 (CAE — aún no implementada). Regenerado el 2026-04-05 tras corrección de bugs OOM y leakage.

---

## 4. Transforms por dataset — Código real

### 4.1 NIH ChestXray14

| Campo | Valor |
|---|---|
| **Archivo** | `src/pipeline/datasets/chest.py` + `src/pipeline/fase1/transform_2d.py` |
| **Función** | `build_2d_transform()` en `transform_2d.py` |
| **Tipo** | 2D — imágenes .png de rayos X de tórax |

**Pipeline de transformación (todas las fases):**

```
convert("RGB")
→ CLAHE(clip_limit=2.0, tile_grid_size=8×8)
→ Resize(224)
→ TotalVariationFilter(weight=10, n_iter=30)
→ GammaCorrection(gamma=1.0)          ← ⚠️ IDENTIDAD (no-op)
→ ToTensor
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ← ImageNet
```

**Augmentaciones de entrenamiento:** ✅ **Implementadas** (INC-02 resuelto)

> Pipeline de augmentación vía `build_2d_aug_transform()`: CLAHE → Resize → RandomHorizontalFlip(0.5) → RandomRotation(10°) → ColorJitter(br=0.2, ct=0.2) → TVF → ToTensor → Normalize. Aplicado solo cuando `split=="train"` y `aug_transform is not None`. Sin transformaciones de oclusión (prohibidas por guía).

> **No-op eliminado (INC-04 resuelto):** `GammaCorrection(gamma=1.0)` era operación identidad. `build_2d_transform()` ahora omite el paso cuando `gamma=1.0`.

---

### 4.2 ISIC 2019

| Campo | Valor |
|---|---|
| **Archivo** | `src/pipeline/datasets/isic.py` |
| **Función** | `build_isic_transforms()` en `isic.py` |
| **Tipo** | 2D — imágenes .jpg dermatoscópicas |

**Pipeline base (todas las fases):**

```
convert("RGB")
→ BCNCrop                              ← crop circular (dermoscopio)
→ Resize(224)
→ ToTensor
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ← ImageNet
```

> **Sin CLAHE, sin TVF, sin GammaCorrection** — pipeline completamente distinto a NIH.

**Tres pipelines de augmentación definidos:**

| Pipeline | Componentes | Uso previsto |
|---|---|---|
| `"embedding"` | Sin augmentación (solo base) | Generación de embeddings |
| `"standard"` | HFlip(0.5) + RandomRotation(±30°) + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) | Clases mayoritarias |
| `"minority"` | Todo de `standard` + VFlip(0.5) + RandomRotation(±90°, ±180°) + RandomAffine(±15°, translate=0.1, scale=[0.85,1.15]) + RandomErasing(p=0.3) | Clases minoritarias (oversampling) |

**✅ BUG-C1 resuelto:** Ver §5.1 — guard `self.split == "train"` aplicado en `isic.py`.

---

### 4.3 Osteoarthritis Knee (OA)

| Campo | Valor |
|---|---|
| **Archivo** | `src/pipeline/datasets/osteoarthritis.py` |
| **Función** | Transforms definidos inline en `__init__` |
| **Tipo** | 2D — radiografías de rodilla |

**Pipeline base (todas las fases):**

```
convert("RGB")
→ CLAHE(clip_limit=2.0)               ← importación diferida de transform_domain.py (INC-05 resuelto)
→ Resize(224, interpolation=BICUBIC)
→ ToTensor
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ← ImageNet
```

> **CLAHE sí, pero sin TVF ni GammaCorrection** — distinto a NIH y a ISIC.

**Augmentaciones de entrenamiento:**

| Augmentación | Parámetros |
|---|---|
| HorizontalFlip | p=0.5 |
| RandomRotation | ±10° |
| ColorJitter | brightness=0.2, contrast=0.15 |

**✅ BUG-C2 resuelto:** Ver §5.2 — guard `self.split == "train"` aplicado en `osteoarthritis.py`.

---

### 4.4 LUNA16 (Nódulos Pulmonares)

| Campo | Valor |
|---|---|
| **Archivo** | `src/pipeline/datasets/luna.py` |
| **Tipo** | 3D — parches volumétricos 64³ de CT pulmonar |

**Pipeline de preprocesamiento:**

```
HU clip(min=-1000, max=400)
→ normalize [0, 1]
→ extract patch 64³
```

**Pipeline de embeddings (3D → 2D para ViT):**

```
Select 3 central slices from patch 64³
→ Resize to [3, 224, 224]                ← 3 slices = 3 channels RGB
```

**Augmentaciones de entrenamiento (5 transforms):**

| Augmentación | Parámetros | Aplicación |
|---|---|---|
| Random flip | axes aleatorios | Solo train |
| Random rotation | ±15° | Solo train |
| HU variation | ±20 HU | Solo train |
| Gaussian noise | p=0.3 | Solo train |
| Spatial shift | ±4 voxels | Solo train |

**✅ Guard correcto** (`luna.py:390`):

```python
if mode == "expert" and split == "train" and augment_3d:
    # aplica augmentaciones
```

> LUNA16 es el **único dataset que implementa correctamente el guard de split** para augmentaciones.

---

### 4.5 Pancreas

| Campo | Valor |
|---|---|
| **Archivo** | `src/pipeline/datasets/pancreas.py` |
| **Tipo** | 3D — volúmenes CT abdominales (.nii.gz) |

**Pipeline de preprocesamiento:**

```
HU clip(min=-100, max=400)
→ z-score per volume                    ← modo por defecto
→ clip [-3, 3]
→ rescale [0, 1]
```

> Dos modos de normalización disponibles: `z-score` (default) y `linear`. La guía no especifica cuál usar.

**Pipeline de embeddings (3D → 2D para ViT):**

```
Select 3 central slices from volume
→ Resize to [3, 224, 224]                ← mismo pipeline que LUNA
```

**Augmentaciones de entrenamiento:** ✅ **Implementadas** (INC-03 resuelto)

> Nuevo método `_augment_3d()` con: flips en 3 ejes (p=0.5), rotación ±15° en plano axial, offset HU ±0.03 (~±15 HU normalizado), ruido gaussiano (p=0.3, σ∈[0, 0.02]). Sin spatial shift (volúmenes son ROIs Z-cropped). Augmentación aplicada ÚNICAMENTE a clase minoritaria (PDAC+, label=1) durante training en modo expert.

---

## 5. Bugs críticos

### 5.1 ~~BUG-C1~~ — RESUELTO (2026-04-05): ISIC guard de split

| Campo | Valor |
|---|---|
| **Archivos modificados** | `src/pipeline/datasets/isic.py` |
| **Fix aplicado** | Añadido parámetro `split="train"` a `__init__` (línea 121), almacenado como `self.split` (línea 135) y `self.transform_base = tfs["embedding"]` (línea 146). En `__getitem__`, augmentaciones solo cuando `self.split == "train"` (línea 440); val/test usan `self.transform_base` (Resize→ToTensor→Normalize, determinístico). Fix backward-compatible: default `split="train"` preserva comportamiento previo en callers existentes (todos usan `mode="embedding"`). |
| **Estado** | ✅ **RESUELTO** |

---

### 5.2 ~~BUG-C2~~ — RESUELTO (2026-04-05): OA Knee guard de split

| Campo | Valor |
|---|---|
| **Archivos modificados** | `src/pipeline/datasets/osteoarthritis.py` |
| **Fix aplicado** | Añadido `self.split = split` en `__init__` (línea 117, junto a los demás `self.*`). En `__getitem__`, la condición para `aug_transform` extendida a `self.mode == "expert" and not self._aug_offline_detected and self.split == "train"` (línea 359). Val/test usan `self.base_transform` (CLAHE→Resize→ToTensor→Normalize, determinístico). |
| **Estado** | ✅ **RESUELTO** |

---

### 5.3 ~~BUG-C3~~ — EMBEDDINGS ELIMINADOS (2026-04-05): regeneración pendiente

| Campo | Valor |
|---|---|
| **Directorios eliminados** | `embeddings/cvt_13/`, `embeddings/densenet121_custom/`, `embeddings/swin_tiny_patch4_window7_224/`, `embeddings/vit_tiny_patch16_224/` |
| **Espacio liberado** | 1,463 MB (239 + 629 + 473 + 122 MB) |
| **Motivo** | Los 4 sets contenían 1,839 parches LUNA con leakage (generados 2026-04-01, antes del fix de leakage 2026-04-02 que movió esos parches a `_LEAKED_DO_NOT_USE/`). |
| **Pendiente** | Regenerar con `python src/pipeline/fase1/fase1_pipeline.py --backbone <backbone> --force` para los 4 backbones: `cvt_13`, `densenet121_custom`, `swin_tiny_patch4_window7_224`, `vit_tiny_patch16_224`. |
| **Estado** | ⏳ **EMBEDDINGS ELIMINADOS — regeneración pendiente antes de Fase 2** |

---

## 6. Incongruencias detectadas

### 6.1 Tabla resumen

| ID | Severidad | Descripción corta | Impacto | Estado |
|---|---|---|---|---|
| INC-01 | 🔴 ALTA | `arquitectura_documentacion.md` §6.2 documenta pipeline uniforme CLAHE→TVF→Gamma para todos los 2D; código real difiere por dataset | Documentación engañosa | ✅ Resuelto (2026-04-05) |
| INC-02 | 🟡 MEDIA | NIH sin augmentaciones de entrenamiento vs. guía "agresivo" | Rendimiento subóptimo | ✅ Resuelto (2026-04-05) |
| INC-03 | 🟡 MEDIA | Pancreas sin augmentaciones 3D (LUNA tiene 5) | Rendimiento subóptimo | ✅ Resuelto (2026-04-05) |
| INC-04 | 🔵 BAJA | `GammaCorrection(gamma=1.0)` es identidad — desperdicio de CPU | Performance | ✅ Resuelto (2026-04-05) |
| INC-05 | 🔵 BAJA | OA importa CLAHE de `preprocessing.py`, no de `transform_domain.py` | Duplicación de código | ✅ Resuelto (2026-04-05) |

### 6.2 Detalle de incongruencias

#### ~~INC-01~~ — 🔴 ALTA: Pipeline documentado vs. código real — RESUELTO (2026-04-05)

| Campo | Valor |
|---|---|
| **Fuente afectada** | `arquitectura_documentacion.md` §6.2 |
| **Descripción** | La documentación describe un pipeline uniforme para todos los datasets 2D: `CLAHE → Resize → TVF → GammaCorrection → ToTensor → Normalize(ImageNet)`. En la realidad, cada dataset tiene su propio pipeline con componentes distintos. |

**Comparación documentación vs. código:**

| Componente | Documentación §6.2 | NIH (código) | ISIC (código) | OA (código) |
|---|---|---|---|---|
| CLAHE | ✅ Todos | ✅ clip=2.0, grid=8×8 | ❌ No tiene | ✅ clip=2.0 |
| BCNCrop | No documentado | ❌ | ✅ Sí | ❌ |
| Resize(224) | ✅ Todos | ✅ | ✅ | ✅ BICUBIC |
| TVF | ✅ Todos | ✅ weight=10, iter=30 | ❌ No tiene | ❌ No tiene |
| GammaCorrection | ✅ Todos | ✅ gamma=1.0 (no-op) | ❌ No tiene | ❌ No tiene |
| Normalize(ImageNet) | ✅ Todos | ✅ | ✅ | ✅ |

**Resolución:**
- **Acción:** Documentación actualizada. Los pipelines per-dataset en el código son CORRECTOS y superiores al pipeline uniforme.
- **Justificación clínica:** NIH (radiografía, escala de grises) requiere CLAHE+TVF; ISIC (dermatoscopía, color) NO usa CLAHE/TVF porque la señal diagnóstica es el color (regla ABCD); OA (radiografía rodilla) usa CLAHE+BICUBIC.
- `arquitectura_documentacion.md` §6.2 actualizado con pipelines per-dataset y justificación clínica.

**Estado:** ✅ **RESUELTO** (2026-04-05)

---

#### ~~INC-02~~ — 🟡 MEDIA: NIH sin augmentaciones de entrenamiento — RESUELTO (2026-04-05)

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `proyecto_moe.md` vs. `chest.py` + `transform_2d.py` |
| **Descripción** | La guía oficial indica augmentación "agresiva" para NIH ChestXray14. El código no implementaba ninguna augmentación de entrenamiento — `build_2d_transform()` producía un pipeline idéntico para train, val y test. |

**Resolución:**
- **Acción:** Añadida función `build_2d_aug_transform()` en `transform_2d.py` con: CLAHE → Resize → RandomHorizontalFlip(0.5) → RandomRotation(10°) → ColorJitter(br=0.2, ct=0.2) → TVF → ToTensor → Normalize.
- `ChestXray14Dataset` actualizado en `chest.py`: nuevo parámetro `split="train"` y `aug_transform=None`; en `__getitem__` se aplica `aug_transform` si `split=="train"` y `aug_transform is not None`.
- Sin transformaciones de oclusión (prohibidas por guía).

**Estado:** ✅ **RESUELTO** (2026-04-05)

---

#### ~~INC-03~~ — 🟡 MEDIA: Pancreas sin augmentaciones 3D — RESUELTO (2026-04-05)

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `pancreas.py` vs. `luna.py` |
| **Descripción** | LUNA16 implementa 5 augmentaciones 3D con guard correcto por split. Pancreas no implementaba ninguna augmentación 3D. |

**Resolución:**
- **Acción:** `PancreasDataset` en `pancreas.py` actualizado con `split="train"` y `augment_3d=True`. Nuevo método `_augment_3d()` con: flips en 3 ejes (p=0.5), rotación ±15° en plano axial, offset HU ±0.03 (~±15 HU normalizado), ruido gaussiano (p=0.3, σ∈[0, 0.02]).
- Sin spatial shift (los volúmenes ya son ROIs Z-cropped).
- Augmentación aplicada ÚNICAMENTE a clase minoritaria (PDAC+, label=1) durante training en modo expert.

**Estado:** ✅ **RESUELTO** (2026-04-05)

---

#### ~~INC-04~~ — 🔵 BAJA: `GammaCorrection(gamma=1.0)` es operación identidad — RESUELTO (2026-04-05)

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `transform_2d.py`, `fase1_config.py` |
| **Descripción** | `GammaCorrection(gamma=1.0)` aplicaba `pixel^(1/1.0) = pixel`, sin modificar la imagen. Consumía CPU/GPU sin efecto alguno. |

**Resolución:**
- **Acción:** `build_2d_transform()` en `transform_2d.py` actualizado para añadir `GammaCorrection` SOLO cuando `gamma != 1.0`. Con `DEFAULT_GAMMA=1.0`, el paso se omite por completo — sin LUT, sin iteración sobre imágenes.

**Estado:** ✅ **RESUELTO** (2026-04-05)

---

#### ~~INC-05~~ — 🔵 BAJA: OA importa CLAHE de módulo distinto — RESUELTO (2026-04-05)

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `osteoarthritis.py` vs. `transform_2d.py` |
| **Descripción** | Tres implementaciones de CLAHE en el codebase (en `transform_2d.py`, `preprocessing.py`, y `transform_domain.py`). OA importaba de `preprocessing.py` en lugar de la fuente canónica. |

**Resolución:**
- **Acción:** `osteoarthritis.py` actualizado para usar importación diferida desde `transform_domain` (fuente canónica) en lugar de `preprocessing` (implementación duplicada). Eliminado `from preprocessing import apply_clahe` a nivel de módulo; añadida importación diferida en el método `__getitem__`: `from transform_domain import apply_clahe`.

**Estado:** ✅ **RESUELTO** (2026-04-05)

---

## 7. Resumen de estado por componente

| Componente | Estado | Detalle |
|---|---|---|
| **Splits** | ✅ Correcto | 5/5 datasets con splits verificados, sumas consistentes, leakage corregido |
| **Transforms** | ✅ Correcto | Pipelines funcionales, inconsistencias documentales resueltas (INC-01 a INC-05, 2026-04-05) |
| **Augmentaciones** | ✅ Guards aplicados | BUG-C1 y BUG-C2 resueltos (2026-04-05) — ISIC y OA ahora solo augmentan en train |
| **Augmentaciones (gaps)** | ✅ Resuelto | NIH augmentaciones añadidas (`build_2d_aug_transform()`), Pancreas augmentaciones 3D añadidas (`_augment_3d()`) |
| **Embeddings** | ⏳ Regeneración pendiente | Stale eliminados (1,463 MB) — regenerar con `fase1_pipeline.py --force` antes de Fase 2 |
| **Manifiesto CAE** | ✅ Correcto | `cae_splits.csv` regenerado (162,611 filas), consistente con splits actuales |

---

## 8. Archivos de transforms — Inventario

### 8.1 `transform_2d.py` (372 líneas)

| Clase/Función | Descripción | Usado por |
|---|---|---|
| `CLAHETransform` | CLAHE con OpenCV (clip_limit, tile_grid_size) | NIH (vía `build_2d_transform()`) |
| `TotalVariationFilter` | Filtro de variación total (weight, n_iter) | NIH (vía `build_2d_transform()`) |
| `GammaCorrection` | Corrección gamma (gamma) | NIH (vía `build_2d_transform()`) — ⚠️ gamma=1.0 = no-op |
| `build_2d_transform()` | Construye pipeline CLAHE→Resize→TVF→Gamma→ToTensor→Normalize | NIH |

### 8.2 `transform_domain.py` (55 líneas)

| Función | Descripción | Usado por |
|---|---|---|
| `apply_clahe()` | CLAHE para OA | ✅ OA (importación diferida en `__getitem__`) |
| `apply_circular_crop()` | BCN crop circular para dermoscopio | ISIC |

### 8.3 `transform_3d.py` (84 líneas)

| Función | Descripción | Usado por |
|---|---|---|
| `normalize_hu()` | Normalización Hounsfield Units | LUNA, Pancreas |
| `resize_volume_3d()` | Resize volumétrico | LUNA, Pancreas |
| `volume_to_vit_input()` | 3 slices centrales → [3, 224, 224] | LUNA, Pancreas (embeddings) |
| `full_3d_pipeline()` | Pipeline completo 3D | LUNA, Pancreas |

---

## 9. Ítems de acción

| # | Prioridad | Descripción | Bloqueante | Estado |
|---|---|---|---|---|
| P3-A1 | 🔴 CRÍTICO | **Corregir BUG-C1:** Almacenar `self.split` en `isic.py:__init__` y añadir guard `self.split == "train"` en `__getitem__` (línea 434) | Sí — contamina evaluación Experto 2 | ✅ Completado (2026-04-05) |
| P3-A2 | 🔴 CRÍTICO | **Corregir BUG-C2:** Almacenar `self.split` en `osteoarthritis.py:__init__` y añadir guard `self.split == "train"` en `__getitem__` (línea 355) | Sí — contamina evaluación Experto 3 | ✅ Completado (2026-04-05) |
| P3-A3 | 🔴 CRÍTICO | **Resolver BUG-C3:** Eliminar embeddings pre-computados de LUNA16 y regenerarlos con parches post-fix leakage | Sí — leakage persiste en embeddings | ⏳ Eliminados (2026-04-05) — regeneración pendiente |
| P3-A4 | 🟡 MEDIA | **Actualizar `arquitectura_documentacion.md` §6.2:** Reemplazar pipeline uniforme documentado con los pipelines reales por dataset (INC-01) | No | ✅ Completado (2026-04-05) |
| P3-A5 | 🟡 MEDIA | **Implementar augmentaciones NIH:** Añadida `build_2d_aug_transform()` en `transform_2d.py` con guard por split (INC-02) | No | ✅ Completado (2026-04-05) |
| P3-A6 | 🟡 MEDIA | **Implementar augmentaciones 3D Pancreas:** Nuevo método `_augment_3d()` en `pancreas.py` con flips/rot/HU offset/noise (INC-03) | No | ✅ Completado (2026-04-05) |
| P3-A7 | 🔵 BAJA | **Corregir `DEFAULT_GAMMA`:** `build_2d_transform()` ahora omite `GammaCorrection` cuando gamma=1.0 (INC-04) | No | ✅ Completado (2026-04-05) |
| P3-A8 | 🔵 BAJA | **Consolidar CLAHE:** OA ahora importa de `transform_domain` (fuente canónica) vía importación diferida (INC-05) | No | ✅ Completado (2026-04-05) |

---

## 10. Comparación de pipelines — Vista consolidada

### 10.1 Datasets 2D

| Paso | NIH | ISIC | OA |
|---|---|---|---|
| Input format | .png → RGB | .jpg → RGB | .jpg/.png → RGB |
| Domain-specific crop | — | BCNCrop (circular) | — |
| CLAHE | ✅ clip=2.0, grid=8×8 | ❌ | ✅ clip=2.0 |
| Resize | 224 (default) | 224 | 224 (BICUBIC) |
| TVF | ✅ w=10, iter=30 | ❌ | ❌ |
| GammaCorrection | ✅ γ=1.0 (no-op) | ❌ | ❌ |
| ToTensor + Normalize | ImageNet | ImageNet | ImageNet |
| Train augmentation | ✅ HFlip/Rot±10°/CJ (vía `build_2d_aug_transform()`) | ✅ HFlip/Rot/CJ + minority extras | ✅ HFlip/Rot±10°/CJ |
| Val/Test augmentation guard | ✅ Resuelto (split guard) | ✅ Resuelto (línea 440) | ✅ Resuelto (línea 359) |

### 10.2 Datasets 3D

| Paso | LUNA16 | Pancreas |
|---|---|---|
| Input format | .mhd+.raw → parches 64³ | .nii.gz → volúmenes completos |
| HU clipping | [-1000, 400] | [-100, 400] |
| Normalization | Linear [0, 1] | z-score → clip [-3,3] → rescale [0,1] |
| Embedding pipeline | 3 central slices → [3,224,224] | 3 central slices → [3,224,224] |
| Train augmentation | ✅ 5 transforms (flip/rot/HU/noise/shift) | ✅ 4 transforms (flip/rot±15°/HU offset/noise) — clase minoritaria only |
| Val/Test augmentation guard | ✅ Correcto (línea 390) | ✅ Correcto (split guard + label guard) |

---

*Documento generado el 2026-04-05 por auditoría multi-agente (ARGOS + SIGMA + EXPLORE). Fuentes: `src/pipeline/datasets/chest.py`, `isic.py`, `osteoarthritis.py`, `luna.py`, `pancreas.py`, `src/pipeline/fase1/transform_2d.py`, `transform_domain.py`, `transform_3d.py`, `pre_modelo.py`, `arquitectura_documentacion.md` §6.2, `proyecto_moe.md`. Formato replicado de `docs/documentacion_pasos/paso_01_descarga_datos.md` y `paso_02_extraccion_archivos.md`.*
