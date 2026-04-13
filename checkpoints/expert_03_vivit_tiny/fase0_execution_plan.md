# Fase 0 — Plan de Ejecución: ¿Qué pasaría si ejecutas `fase0_pipeline.py`?

> **Comando:** `python3 src/pipeline/fase0/fase0_pipeline.py`
> **Fecha de análisis:** 2026-04-10
> **Última actualización:** 2026-04-13

> **⚠️ Nota sobre el modelo entrenado:** El notebook `luna_training_kaggle.ipynb` entrena
> un **3D Faster R-CNN (MC3-18 backbone)**, no el ViViT-Tiny originalmente planificado.
> Además, el notebook usa extracción de parches **on-the-fly desde .mhd** en lugar de los
> parches `.npy` pre-extraídos por este pipeline. Los datos de Fase 0 siguen siendo válidos
> para re-entrenamiento con el pipeline offline, pero el modelo actual no los utiliza
> directamente.

## Resumen ejecutivo

Al ejecutar `fase0_pipeline.py` sin flags especiales, **todos los 7 datasets están activos**
(`nih`, `isic`, `oa`, `luna_meta`, `luna_ct`, `pancreas`, `panorama`). La mayoría de datos
ya existen en disco, por lo que los pasos 1–5 y 6c se saltarán por idempotencia en pocos
segundos. **El cuello de botella crítico es el sub-paso 6a**: aunque todos los parches de
LUNA16 ya existen, el pipeline cargará los 888 CTs completos en memoria antes de decidir
saltarlos, consumiendo ~30–90 minutos innecesariamente. Los sub-pasos 6b (zero-centering
scan) y 6d (audit) sí ejecutarán trabajo real (~5–8 min en total).

---

## Paso 0 — Verificar Prerequisites

**Estado esperado:** ✅ PASA (con 1 advertencia de espacio)

| Herramienta | Estado |
|-------------|--------|
| 7z | ✅ `/usr/bin/7z` — disponible |
| wget | ✅ `/usr/bin/wget` — disponible |
| git | ✅ `/usr/bin/git` — disponible |
| kaggle CLI | ✅ `/home/mrsasayo_mesa/.pyenv/shims/kaggle` — disponible |
| kaggle.json | ✅ `~/.kaggle/kaggle.json` — existe |
| Espacio libre | ⚠️ **100 GB** libres (932 GB total, 90% uso) — pipeline advertirá `< 150 GB` |

**Acción:** No hay bloqueantes. Emitirá 1 warning de espacio pero **no abortará**.

---

## Paso 1 — Descargar datasets

**Estado esperado:** ✅ Todo ya descargado — comportamiento dominado por `descargar.py`

| Dataset | Estado en disco | Acción esperada |
|---------|----------------|-----------------|
| NIH ChestXray14 | ✅ `nih_chest_xrays/all_images/` con 112,120 imgs | Saltará |
| ISIC 2019 | ✅ `isic_2019/ISIC_2019_Training_Input/` con 25,333 imgs | Saltará |
| Osteoarthritis | ✅ `osteoarthritis/KLGrade/` + `discarded/` | Saltará |
| LUNA16 metadata | ✅ `luna_lung_cancer/candidates_V2/`, `annotations.csv` | Saltará |
| LUNA16 CT volumes | ✅ subset0–subset9 con 888 archivos `.mhd` | Saltará |
| Páncreas (Zenodo) | ✅ `zenodo_13715870/` con volúmenes `.nii.gz` | Saltará |
| Panorama labels | ✅ `panorama_labels/automatic_labels/` | Saltará |

**Tiempo estimado:** < 30 s — solo verificaciones de existencia de archivos.

---

## Paso 2 — Extraer archivos

**Estado esperado:** ✅ Todo ya extraído

| Dataset | Estado | Acción esperada |
|---------|--------|-----------------|
| NIH | `all_images/` con 112,120 archivos | Saltará |
| ISIC | `ISIC_2019_Training_Input/` con 25,333 archivos | Saltará |
| OA | `KLGrade/` ya existe | Saltará |
| LUNA16 CT | `subset0/`–`subset9/` con `.mhd/.raw` | Saltará |
| Páncreas | `.nii.gz` no requieren extracción adicional | Saltará |

**Tiempo estimado:** < 30 s.

---

## Paso 3 — Post-procesado NIH ChestXray14

**Estado esperado:** ✅ Probablemente saltará por idempotencia

Los splits NIH ya existen en `datasets/nih_chest_xrays/splits/`:
- `nih_train_list.txt` — 88,999 líneas
- `nih_val_list.txt` — 11,349 líneas
- `nih_test_list.txt` — 11,772 líneas

**Acción:** `pre_chestxray14.run_pre_chestxray14()` detectará que los splits existen y
retornará `{"status": "✅"}` inmediatamente.

**Tiempo estimado:** < 1 min.

---

## Paso 4 — Etiquetas páncreas

**Estado esperado:** ✅ SALTARÁ por idempotencia

`datasets/pancreas_labels_binary.csv` **existe en disco**.

**Acción:** El código en la línea 382 verifica `if out_csv.exists() and out_csv.stat().st_size > 100`
→ leerá el CSV, contará filas válidas y retornará `{"status": "✅", "skipped": True}` de inmediato.

**Tiempo estimado:** < 1 s.

---

## Paso 5 — Splits 80/10/10

**Estado esperado:** ✅ Saltará para la mayoría de datasets

| Dataset | Splits en disco | Acción esperada |
|---------|----------------|-----------------|
| NIH | `nih_train_list.txt` etc. | Saltará |
| ISIC | `isic_train.csv`, `isic_val.csv`, `isic_test.csv` | Saltará |
| LUNA16 | `luna_splits.json` (712/88/88 UIDs) | Saltará |
| Páncreas | `pancreas_splits.csv` | Saltará |
| OA | Parcialmente según `pre_modelo.py` | Posiblemente saltará |

**Tiempo estimado:** < 30 s.

---

## Paso 6 — Datos 3D LUNA16 (4 sub-pasos)

**Estado esperado:** Mixto — 6a y 6c saltan (con advertencia), 6b y 6d ejecutan.

---

### Sub-paso 6a — Extracción de parches LUNA16

**Parches actuales en disco:**

| Split | Parches |
|-------|---------|
| train | 13,880 |
| val | 1,156 |
| test | 2,014 |
| train_aug | 17,669 |

- **luna_splits.json:** `train_uids=712`, `val_uids=88`, `test_uids=88`
- **Idempotencia:** Cada parche se verifica con `if out_path.exists()` → retorna `SKIPPED`.

**⚠️ ADVERTENCIA CRÍTICA — cuello de botella:**
Aunque los parches se saltan individualmente, el código **primero carga cada CT completo en
memoria** (SimpleITK read + scipy resample a 1 mm isotrópico) antes de verificar los parches.
Esto consume ~30–90 minutos procesando 888 CTs innecesariamente.

**Segundo riesgo — ~~RESUELTO (2026-04-13)~~:** ~~Después de la extracción, `run_luna_patches()`
recalculaba `global_mean` y aplicaba zero-centering a TODOS los parches, restando `0.09922`
otra vez a parches ya centrados.~~ **Corregido:** el código ahora incluye un idempotency guard
(líneas ~750–770 de `pre_embeddings.py`) que muestrea 50 parches de train; si `abs(sample_mean) < 0.01`,
detecta que ya están centrados y **salta la aplicación**. El sub-paso 6b ya no es necesario
como corrector de 6a.

**Tiempo estimado:** ~30–90 min (dominado por carga de 888 CTs aunque no haya trabajo real).

**Recomendación:** Ejecutar con `--solo_pasos 6 7 8` junto con `--skip_zerocentering` NO
evita el problema de 6a. La solución real es añadir un guard al inicio de `run_luna_patches()`
que detecte parches existentes antes de cargar CTs.

---

### Sub-paso 6b — Zero-centering fix

- **Estado actual:** ✅ Todos los parches están correctamente centrados (auditado y confirmado)
- **global_mean:** `0.09921630471944809`
- **Idempotencia:** Escanea TODOS los parches en los 4 splits. Para cada parche verifica
  `mean < -0.09922`. Parches correctos retornan `OK`.
- **Acción:** Si 6a corrompió los parches (restó global_mean de nuevo), 6b los corregirá.
  Si 6a se saltó correctamente, 6b solo escaneará y encontrará todo `OK`.
- Usa pool de 8 workers con `mp.get_context("fork")`.

**Tiempo estimado:** ~3–5 min para escanear ~34,600 parches.

---

### Sub-paso 6c — Crear train_aug

- `train_aug/` existe con **17,669 parches** ✅
- `train_aug_manifest.csv` existe ✅
- **Idempotencia:** Verifica `n_rows >= 15,000` → **SALTARÁ inmediatamente**.

**Tiempo estimado:** < 1 s.

---

### Sub-paso 6d — Auditoría del dataset

- **Acción:** Ejecutará `audit_dataset.py` como subproceso.
- Muestreará 200 parches por split y verificará: shape, dtype, NaN/Inf, zero-centering,
  balance pos/neg, variabilidad de augmentaciones, duplicados exactos.
- **Resultado esperado (estado actual del disco):** `overall_pass = ✅ TRUE`
  - Todos los checks pasan (parches corruptos de val/test ya fueron re-extraídos)

**Tiempo estimado:** ~2–3 min.

---

## Paso 7 — CvT-13

**Estado esperado:** ✅

**Acción:**
1. Verificará que `transformers` y `einops` están instalados → probablemente ya lo están
2. Verificará `scripts/cvt13_backbone.py` (puede o no existir)
3. ✅ Encontrará `src/pipeline/fase1/backbone_cvt13.py` — **existe en disco**
4. Retornará `{"status": "✅", "native": True}`

**Tiempo estimado:** < 30 s.

---

## Paso 8 — Reporte final

**Acción:** Genera `src/pipeline/fase0/fase0_report.md` con:
- Tabla de estado por paso (0–8 con tiempos)
- Lista de datasets activos
- Splits generados
- Comando completo para Fase 1 (con todos los argumentos)
- Copia con timestamp en `logs/fase0_report_YYYYMMDD_HHMMSS.md`

**Tiempo estimado:** < 1 s.

---

## ⏱️ Tiempo total estimado

| Paso | Tiempo estimado | Motivo |
|------|----------------|--------|
| 0 | < 5 s | Solo verificaciones del sistema |
| 1 | < 30 s | Todos los datasets ya descargados |
| 2 | < 30 s | Todos los archivos ya extraídos |
| 3 | < 1 min | NIH splits ya existen |
| 4 | < 1 s | `pancreas_labels_binary.csv` ya existe — skip inmediato |
| 5 | < 30 s | Splits ya existen para todos los datasets |
| **6a** | **~30–90 min** | **🔴 CUELLO DE BOTELLA: carga 888 CTs aunque todos los parches se saltan** |
| 6b | ~3–5 min | Escanea ~34,719 parches con 8 workers |
| 6c | < 1 s | Idempotente — train_aug ya existe con ≥15,000 filas |
| 6d | ~2–3 min | Audit sobre muestra de 200 por split |
| 7 | < 30 s | backbone_cvt13.py encontrado |
| 8 | < 1 s | Generación de reporte markdown |
| **Total** | **~35–100 min** | **Dominado por 6a (carga innecesaria de CTs)** |

---

## ✅ Estado real de los datos LUNA16 (listos para entrenar)

Los datos de LUNA16 están **completamente listos para entrenamiento**.

### Parches por split

| Split | Total parches | Positivos | Negativos | Ratio neg:pos |
|-------|--------------|-----------|-----------|---------------|
| train | 13,880 | 1,263 | 12,617 | 9.99:1 |
| val | 1,156 | 105 | 1,051 | 10.0:1 |
| test | 2,014 | 183 | 1,831 | 10.0:1 |
| train_aug | 17,669 | 5,052 | 12,617 | 2.50:1 |

### Especificaciones técnicas
- **Shape:** `(64, 64, 64)` float32 — cubo de 64 mm³ a 1 mm isotrópico
- **Pipeline:** HU load → resample 1 mm³ → lung mask → clip [-1000, 400] HU → min-max [0,1] → zero-centering
- **Global mean:** `0.09921630471944809` (guardado en `patches/global_mean.npy`)
- **Splits por paciente:** 712 train / 88 val / 88 test UIDs (sin data leakage entre pacientes)

### Resultados de auditoría (audit_report.json — actualizado 2026-04-10)

| Check | Resultado |
|-------|-----------|
| `overall_pass` | ✅ **TRUE** |
| `shape_dtype_ok` | ✅ TRUE — todos los parches muestreados tienen shape y dtype correctos |
| `zero_centering_ok` | ✅ TRUE — 0% de fallo en todos los splits |
| `balance_train_ok` | ✅ TRUE — ratio 9.99:1 (rango esperado 8–12) |
| `balance_train_aug_ok` | ✅ TRUE — ratio 2.48:1 (rango esperado 2.0–3.5) |
| `global_mean_ok` | ✅ TRUE — valor exacto 0.09921630471944809 |
| `augmentation_variability_ok` | ✅ TRUE — 100% de pares aug/original con std_diff > 0.01 (mean=0.128) |
| `no_exact_duplicates_ok` | ✅ TRUE — 0 duplicados exactos en 20 pares muestreados |

### Estado de otros datasets del sistema MoE

| Dataset | Descargado | Extraído | Splits | ¿Listo? |
|---------|-----------|----------|--------|---------|
| NIH ChestXray14 | ✅ 112,120 imgs | ✅ | ✅ 88,999/11,349/11,772 | ✅ |
| ISIC 2019 | ✅ 25,333 imgs | ✅ | ✅ ~20,409/2,474/2,448 | ✅ |
| Osteoarthritis | ✅ | ✅ KLGrade/ | Parcial | ~⚠️ |
| LUNA16 | ✅ 888 CTs | ✅ 34,719 parches | ✅ 712/88/88 UIDs | ✅ |
| Páncreas (Zenodo) | ✅ NIfTI volumes | ⏳ 152/2,238 `.npy` preprocesados | ✅ `pancreas_splits.csv` | ⏳ En progreso |
| Panorama labels | ✅ masks | N/A | N/A | ✅ |

---

## ⚠️ Advertencias y consideraciones antes de ejecutar

1. **🔴 Paso 6a carga 888 CTs innecesariamente (~30–90 min de tiempo muerto).**
   Aunque todos los parches ya existen y se saltan individualmente, el worker abre cada CT
   completo (SimpleITK + scipy resample a 1 mm) antes de chequear si el parche existe.
   **Solución sin modificar código:** ejecutar con `--solo_pasos 0 1 2 3 4 5 7 8`
   para saltarte el paso 6 completamente (los datos de LUNA16 ya están listos).

2. **~~🔴 Paso 6a re-aplica zero-centering a parches ya centrados.~~ ✅ RESUELTO (2026-04-13)**
   `run_luna_patches()` ahora incluye un idempotency guard que muestrea 50 parches de train
   y verifica `abs(sample_mean) < 0.01`. Si los parches ya están centrados, salta la aplicación
   de zero-centering. Ya no genera trabajo circular ni corrupción de parches.

3. **⚠️ Espacio en disco: 100 GB libres (de 932 GB, 90% usado).**
   No hay descargas pendientes pero es ajustado. El pipeline emitirá advertencia de espacio
   en el paso 0 pero no abortará.

4. **⚠️ El audit (6d) siempre se re-ejecuta (no es idempotente).**
   Genera un reporte fresco cada vez. Tarda ~2–3 min. Con `--skip_audit` se puede omitir.

5. **ℹ️ `backbone_cvt13.py` existe en `src/pipeline/fase1/`.**
   El paso 7 lo encontrará y retornará `{"status": "✅", "native": True}`.

6. **ℹ️ `luna_splits.json` usa claves `train_uids`/`val_uids`/`test_uids`** (no `train`/`val`/`test`).
   El código de `pre_embeddings.py` lee las claves correctas. No es un problema.

7. **Comando recomendado si solo quieres verificar el estado sin riesgo:**
   ```bash
   python3 src/pipeline/fase0/fase0_pipeline.py --solo_pasos 0 7 8 --dry_run
   ```

8. **Comando para re-ejecutar solo los sub-pasos de LUNA16 de forma segura:**
   ```bash
   python3 src/pipeline/fase0/fase0_pipeline.py \
       --solo luna \
       --solo_pasos 6 \
       --skip_zerocentering \
       --skip_augmentation
   ```
   Esto ejecutará 6a (idempotente sobre parches), 6c (skip), y 6d (audit fresco).
   Pero seguirá cargando los 888 CTs en 6a.
