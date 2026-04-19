# Fase 0 LUNA16 — Bug Fixes & Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Corregir 6 bugs confirmados en el pipeline de Fase 0 LUNA16 (pre_embeddings.py, create_augmented_train.py, luna.py) sin alterar la lógica correcta existente.

**Architecture:** Dos grupos de cambios independientes y paralelos: (A) pre_embeddings.py — idempotencia de zero-centering, UID fallback, refactor redundancia paso 7; (B) create_augmented_train.py + luna.py — clip correcto sobre datos zero-centered, dead-code, clave corrupt_files.

**Tech Stack:** Python 3.11+, NumPy, SciPy, SimpleITK, multiprocessing

---

## Grupo A — `src/pipeline/fase0/pre_embeddings.py`

### Fix A1: Problem 6C — UID fallback a "train"

**Archivo:** `src/pipeline/fase0/pre_embeddings.py`

**Problema:** `uid_to_split()` línea 584 hace `return "train"` para UIDs no encontrados en luna_splits.json. Esto contamina el split de entrenamiento con datos sin split asignado.

**Fix:**
```python
# ANTES (línea 577-584):
def uid_to_split(uid):
    if uid in train_uids:
        return "train"
    if uid in val_uids:
        return "val"
    if uid in test_uids:
        return "test"
    return "train"  # fallback — INCORRECTO

df_sub["split"] = df_sub["seriesuid"].apply(uid_to_split)

# DESPUÉS:
def uid_to_split(uid):
    if uid in train_uids:
        return "train"
    if uid in val_uids:
        return "val"
    if uid in test_uids:
        return "test"
    return None  # excluir UIDs desconocidos

df_sub["split"] = df_sub["seriesuid"].apply(uid_to_split)
n_excluded = df_sub["split"].isna().sum()
if n_excluded > 0:
    log.warning("[LUNA] %d candidatos excluidos (UIDs no en luna_splits.json)", n_excluded)
df_sub = df_sub[df_sub["split"].notna()].copy()
```

---

### Fix A2: Step 6 — Idempotencia del zero-centering

**Archivo:** `src/pipeline/fase0/pre_embeddings.py` (sección Step 6, líneas 727-771)

**Problema:** Si `run_luna_patches()` se ejecuta dos veces, el zero-centering se aplica dos veces. Los parches ya centrados (mean ≈ 0) se vuelven a centrar erróneamente.

**Estrategia de idempotencia:**
1. Después de calcular `global_mean` dinámicamente (mantener cálculo dinámico), guardar a `global_mean.npy` (ya existe).
2. Antes de aplicar el centering in-place, tomar muestra de 50 patches de train/ y calcular su media actual.
3. Si `|current_mean| < 0.01` → patches ya están centrados → saltar aplicación (log warning + skip).
4. Si `current_mean >= 0.01` → aplicar normalmente.

```python
# Después de calcular global_mean y guardarlo:

# ── Idempotency check: verify patches are not already zero-centered ──────
IDEMPOTENCY_SAMPLE = min(50, len(train_patches))
sample_patches = random.sample(train_patches, IDEMPOTENCY_SAMPLE) if train_patches else []
current_mean = 0.0
if sample_patches:
    vals = [np.load(pp).mean() for pp in sample_patches]
    current_mean = float(np.mean(vals))

if abs(current_mean) < 0.01:
    log.info(
        "[LUNA] Zero-centering SKIPPED — patches already centered "
        "(sample mean=%.4f, threshold=0.01). Idempotency guard active.",
        current_mean,
    )
else:
    log.info(
        "[LUNA] Applying zero-centering (mean=%.6f → subtracting global_mean=%.6f)...",
        current_mean, global_mean,
    )
    for sp in ["train", "val", "test"]:
        ...  # mismo código existente
```

---

### Fix A3: Step 7 — Refactor redundancia CT preprocessing

**Archivo:** `src/pipeline/fase0/pre_embeddings.py`

**Problema:** Los pasos 1-5 del pipeline (load CT, resamplear, máscara, clip, normalizar) están duplicados entre `extract_patch()` (líneas 118-230) y `_worker()` (líneas 236-374). Hay ~100 líneas duplicadas.

**Fix:** Extraer helper `_preprocess_ct(mhd_path, seg_dir)` que ejecuta pasos 1-5 y retorna `(array, origin, spacing, direc)`. Ambas funciones llaman al helper.

```python
def _preprocess_ct(mhd_path, seg_dir, clip_hu=HU_LUNG_CLIP):
    """Pasos 1-5 del pipeline: load, resample, mask, clip, normalize.
    
    Returns:
        (array, origin, spacing, direc) donde array es float32 en [0,1]
        normalizado a 1mm isotropic.
    """
    import SimpleITK as sitk
    from scipy.ndimage import zoom as scipy_zoom

    # Step 1: Load + HU conversion
    image = sitk.ReadImage(str(mhd_path))
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direc = np.array(image.GetDirection())
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array[array < -1000] = -1000.0

    # Step 2: Isotropic resampling
    zoom_factors = (spacing[2], spacing[1], spacing[0])
    array = scipy_zoom(array, zoom_factors, order=1)

    # Step 3: Lung segmentation mask
    uid = Path(mhd_path).stem
    seg_path = Path(seg_dir) / (uid + ".mhd")
    if seg_path.exists():
        mask_img = sitk.ReadImage(str(seg_path))
        mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
        mask_arr = scipy_zoom(mask_arr, zoom_factors, order=0)
        mask_arr = (mask_arr > 0.5).astype(np.uint8)
        min_z = min(array.shape[0], mask_arr.shape[0])
        min_y = min(array.shape[1], mask_arr.shape[1])
        min_x = min(array.shape[2], mask_arr.shape[2])
        array[:min_z, :min_y, :min_x][mask_arr[:min_z, :min_y, :min_x] == 0] = -1000.0
        if min_z < array.shape[0]: array[min_z:, :, :] = -1000.0
        if min_y < array.shape[1]: array[:, min_y:, :] = -1000.0
        if min_x < array.shape[2]: array[:, :, min_x:] = -1000.0

    # Step 4: Clip HU
    array = np.clip(array, clip_hu[0], clip_hu[1])

    # Step 5: Min-max normalization
    array = (array - clip_hu[0]) / (clip_hu[1] - clip_hu[0])

    return array, origin, spacing, direc
```

Luego `extract_patch()` y `_worker()` llaman a `_preprocess_ct()` en vez de duplicar los pasos.

---

### Fix A4: Bug B.4 — `corrupt_files` como lista en el reporte de `_paso6b_fix_zerocentering()`

**Archivo:** `src/pipeline/fase0/pre_embeddings.py` (función `_paso6b_fix_zerocentering`)

**Problema:** El reporte JSON guarda `corrupt` como un entero (count), pero `create_augmented_train.py` espera leer `corrupt_files` como una lista de nombres de archivo.

**Fix:** Agregar recolección de nombres de archivos corruptos:
```python
# En _paso6b_fix_zerocentering(), counters dict:
counters = {"total": total, "fixed": 0, "ok": 0, "corrupt": 0, "corrupt_files": []}

# En el loop de resultados:
else:  # CORRUPT
    counters["corrupt"] += 1
    counters["corrupt_files"].append(Path(path_str).name)

# En all_splits_stats:
all_splits_stats[split] = {
    "fixed": counters["fixed"],
    "ok": counters["ok"],
    "corrupt": counters["corrupt"],
    "corrupt_files": counters["corrupt_files"],  # lista de nombres, no solo count
}
```

---

## Grupo B — `create_augmented_train.py` + `luna.py`

### Fix B2+B3: Clip correcto sobre patches zero-centered + usar GLOBAL_MEAN

**Archivo:** `src/pipeline/fase0/create_augmented_train.py`

**Problema:** `np.clip(volume, 0.0, 1.0)` en línea 143 trunca valores negativos legítimos de patches zero-centered (rango real: ~[-0.099, 0.901]). GLOBAL_MEAN está definido pero sin usar (dead code).

**Fix en `augment_patch()`** (línea 143):
```python
# ANTES:
return np.clip(np.ascontiguousarray(volume, dtype=np.float32), 0.0, 1.0)

# DESPUÉS:
return np.clip(
    np.ascontiguousarray(volume, dtype=np.float32),
    -GLOBAL_MEAN,
    1.0 - GLOBAL_MEAN,
)
```

Esto usa `GLOBAL_MEAN` (fix B3 — ya no es dead code) y preserva el rango correcto `[-0.09921..., 0.90078...]`.

---

### Fix C (online): Clip correcto en `luna.py::_augment_3d()`

**Archivo:** `src/pipeline/datasets/luna.py`

**Problema:** Mismo que B2 — `np.clip(volume, 0.0, 1.0)` en línea 820 sobre datos potencialmente zero-centered.

**Fix:**
1. Añadir constante cerca del inicio del módulo o de la clase:
```python
# Zero-centering global mean (computed on LUNA16 train split)
_LUNA_GLOBAL_MEAN: float = 0.09921630471944809
```

2. Cambiar línea 820 en `_augment_3d()`:
```python
# ANTES:
volume = np.clip(volume, 0.0, 1.0)

# DESPUÉS:
volume = np.clip(volume, -_LUNA_GLOBAL_MEAN, 1.0 - _LUNA_GLOBAL_MEAN)
```

3. Actualizar el docstring de `_augment_3d()` que dice "rango [0,1]" para reflejar que la entrada/salida es zero-centered.

---

## Verificación

Después de aplicar todos los fixes:

1. `python -c "import ast; ast.parse(open('src/pipeline/fase0/pre_embeddings.py').read()); print('OK')"` — sin errores de sintaxis
2. `python -c "import ast; ast.parse(open('src/pipeline/fase0/create_augmented_train.py').read()); print('OK')"` — sin errores
3. `python -c "import ast; ast.parse(open('src/pipeline/datasets/luna.py').read()); print('OK')"` — sin errores
4. Verificar que `GLOBAL_MEAN` en `create_augmented_train.py` se usa en la expresión de clip.
5. Verificar que `uid_to_split()` ya no tiene `return "train"` como fallback.
6. Verificar que `_paso6b_fix_zerocentering()` incluye `corrupt_files` como lista en `all_splits_stats`.
7. Verificar que el idempotency check existe en Step 6 de `run_luna_patches()`.
8. Verificar que `_preprocess_ct()` helper existe y es llamado por `extract_patch()` y `_worker()`.

**Estado:** ✅ Todos los fixes A1–A4, B2+B3, C implementados y verificados con `ast.parse()` — 2026-04-13

---

## Verificación de Fase 0 completa — Discrepancias vs. `arquitectura_documentacion.md`

> Ejecutada el 2026-04-13. Cubre todos los scripts de Fase 0 aún no auditados.

### Archivos sin discrepancias

| Archivo | Resultado |
|---------|-----------|
| `fase0_pipeline.py` | ✅ Sin discrepancias funcionales. Pasos 0–8 orquestados en orden correcto. |
| `pre_chestxray14.py` | ✅ Sin discrepancias. Responsabilidad limitada a estructura de directorios/symlinks. |
| `descargar.py` | ✅ INC-03 implementado: soporte multi-batch para Páncreas (4 batches, loop, URLs, MD5). |
| `extraer.py` | ✅ INC-03 implementado: extracción multi-batch con idempotencia per-batch. |
| `fix_valtest_corrupt.py` | ✅ Funcional y completo para su alcance (val/test, corrupción de memoria no inicializada). |
| `reextract_corrupt.py` | ✅ Funcional y completo para su alcance (train/val/test, archivos sin header .npy válido). |

### Discrepancias identificadas

#### Riesgo MEDIO

**DM-1 — Duplicación Pancreas en `cae_splits.csv`**
- **Archivo:** `pre_modelo.py`, `build_cae_splits()` líneas 1038–1064
- **Arquitectura:** `cae_splits.csv` con Pancreas: 2,609 filas
- **Código:** Itera sobre CADA fila de `pancreas_splits.csv`, que tiene múltiples entradas por case_id (una por fold en k-fold). Un mismo volumen `.nii.gz` puede aparecer como `train` en una fila y `val` en otra → mismo volumen Pancreas en train Y val del CAE simultáneamente → **data leakage potencial dentro del dataset CAE**.
- **Recomendación:** Seleccionar un único fold (o deduplicar por `ruta_imagen` tomando un split canónico por volumen) antes de construir el CAE CSV.

**DM-2 — `audit_dataset.py` solo cubre LUNA16 (4 de 5 datasets sin auditoría)**
- **Archivo:** `audit_dataset.py`
- **Arquitectura §7.2:** Documenta estado de datos para los 5 datasets
- **Código:** Script exclusivamente de auditoría de parches 3D LUNA16. No hay auditoría formal de integridad para NIH, ISIC, OA ni Pancreas.
- **Impacto:** Fase 1 intentará generar embeddings para 5 datasets sin verificación previa de integridad en 4 de ellos.

**DM-3 — `fase0_report.md` incompleto y con estados contradictorios**
- **Archivo:** `fase0_report.md`
- **Arquitectura §7.1:** Paso 5 "⚠️ parcial", Paso 6 "⚠️ parcial", Paso 7 "⚠️"
- **Reporte:** Paso 5 y 6 sin indicador de estado ("—"). Paso 7 marcado ✅ (contradice arquitectura). Solo NIH tiene conteos de split; ISIC, OA, LUNA, Pancreas y CAE dicen "skipped" sin estadísticas.

#### Riesgo BAJO

**DB-1 — `fase0_pipeline.py` Paso 0 no verifica SimpleITK**
- El paso 0 verifica `7z`, `wget`, `git`, `kaggle` pero no `SimpleITK`, que es crítico para el Paso 6. Un fallo ocurriría tardíamente sin mensaje diagnóstico útil.

**DB-2 — `pre_modelo.py`: nombre `split_isic()` vs. `build_lesion_split` en la arquitectura**
- La arquitectura §7.4 referencia `build_lesion_split` como método de split de ISIC. La función en código se llama `split_isic()`. Inconsistencia de nomenclatura, no funcional.

**DB-3 — `pre_modelo.py`: split LUNA sin estratificación**
- LUNA tiene desbalance 10.7:1 (positivos/negativos). `split_luna()` baraja aleatoriamente sin estratificar. Los demás datasets sí estratifican. No hay exigencia explícita en la arquitectura, pero es una asimetría de diseño.

**DB-4 — MD5 de Páncreas no verificado en `descargar.py`**
- `ZENODO_PANCREAS_BATCHES` define MD5 por batch pero la función no los verifica. Solo valida por tamaño mínimo. Un archivo corrupto del tamaño correcto pasaría.

**DB-5 — Idempotencia agresiva puede enmascarar correcciones pasadas**
- Todas las funciones `split_*` en `pre_modelo.py` saltan si los artefactos ya existen ("skipped"). Si un split anterior fue generado con un bug ya corregido, el archivo viejo persiste. Requiere borrado manual para forzar regeneración.

### Veredicto de readiness para Fase 1

**Fase 0 NO está lista para avanzar a Fase 1 con confianza plena.**

Bloqueantes concretos:
1. **DM-1** — Leakage potencial en `cae_splits.csv` para Pancreas. Requiere verificación o fix antes de que Fase 1 consuma ese manifiesto.
2. **DM-2** — Sin auditoría de integridad para NIH, ISIC, OA, Pancreas. Riesgo de fallo silencioso en Fase 1.
3. **DM-3** — `fase0_report.md` incompleto. La arquitectura lo lista como artefacto de transición Fase 0→Fase 1. En su estado actual no cumple ese rol.

Acciones mínimas antes de avanzar:
| Acción | Prioridad |
|--------|-----------|
| Investigar `build_cae_splits` Pancreas: ¿hay duplicados reales en disco? | Alta |
| Ampliar `audit_dataset.py` o crear auditorías equivalentes para los otros 4 datasets | Alta |
| Regenerar `fase0_report.md` con conteos reales de todos los splits | Alta |
| Verificar que `fix_valtest_corrupt.py` + `reextract_corrupt.py` ya se ejecutaron | Media |
