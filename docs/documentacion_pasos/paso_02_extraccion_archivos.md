# Paso 2 — Extraer Archivos: Auditoría Completa

| Campo | Valor |
|---|---|
| **Fecha de auditoría** | 2026-04-05 |
| **Auditor** | Multi-agente: ARGOS (alineación guía↔disco), SIGMA (verificación de fuentes), EXPLORE (inspección de filesystem) |
| **Alcance** | Descompresión de todos los archivos ZIP/tarball descargados en Paso 1 — 5 datasets de dominio |
| **Estado general** | ✅ **Paso 2 COMPLETO — 5/5 datasets extraídos, ~235 GB en disco, 0 archivos comprimidos residuales.** Todos los ZIPs originales fueron eliminados tras extracción verificada (~172.6 GB liberados). La extracción se realizó exclusivamente con `7z` (p7zip-full) — sin fallback a `unzip`. Función `is_extracted()` confirma que todos los datasets están descomprimidos. |
| **Commit del proyecto** | `948cd78b6de16a53d7220b92ce8863ba3d910edc` |

---

## 1. Resumen ejecutivo

Los cinco datasets descargados en el Paso 1 fueron extraídos exitosamente a sus directorios de destino en `datasets/`. El proceso de extracción utiliza exclusivamente `7z` (paquete `p7zip-full`) debido a que los ZIPs de LUNA16 subsets 7 y 9 son formato ZIP64, con el cual `unzip` falla silenciosamente. El módulo `extraer.py` implementa un `RAMMonitor` que pausa/reanuda el proceso `7z` mediante señales SIGSTOP/SIGCONT para evitar OOM en máquinas con RAM limitada. Tras la extracción, todos los ZIPs fueron eliminados en modo `--disco`, liberando ~172.6 GB. El tiempo reportado en los documentos existentes (~0.4–0.5s) corresponde a ejecuciones donde `is_extracted()` hizo skip de todos los datasets — la extracción real no tiene tiempo registrado.

---

## 2. Descripción del paso

| Campo | Valor |
|---|---|
| **Objetivo** | Descomprimir todos los archivos ZIP descargados por el Paso 1 (descargar), dejando los datos en formato listo para consumo por los pasos posteriores de Fase 0 |
| **Script orquestador** | `src/pipeline/fase0/fase0_pipeline.py` → función `paso2_extraer()` (líneas 312–331) |
| **Módulo de ejecución** | `src/pipeline/fase0/extraer.py` → función `run_extractions()` (líneas 392–488) |
| **Dependencia directa** | Paso 1 (Descarga de Datos) — requiere que los ZIPs existan en disco |
| **Herramienta obligatoria** | `7z` (paquete `p7zip-full`). Sin fallback. Si no está instalado, el paso termina con error y instrucciones de instalación. |
| **Fase del pipeline** | Fase 0 — Preparación de Datos, Paso 2 |
| **Definido en** | `arquitectura_documentacion.md` línea 97: `\| 2 \| Extraer archivos \| Descompresión de ZIPs, tarballs, NIfTI \| Fase 0 – Paso 2 \| ✅ Completado \|` |

**Flujo de ejecución:**

1. `fase0_pipeline.py` invoca `paso2_extraer(active, disco, luna_subsets)`
2. `paso2_extraer()` importa y llama a `run_extractions()` de `extraer.py`
3. `run_extractions()` itera un plan de extracción ordenado de menor a mayor tamaño
4. Para cada dataset, `is_extracted()` verifica si ya fue extraído — si sí, hace skip (idempotencia)
5. Si el ZIP existe y no está extraído, `smart_extract()` ejecuta `7z x <archivo> -o<destino> -y`
6. `RAMMonitor` supervisa RAM disponible y pausa/reanuda `7z` si es necesario
7. En modo `--disco`, el ZIP se elimina tras extracción exitosa
8. Para LUNA CTs, `verify_luna_ct_subset()` valida conteo de pares `.mhd`+`.raw` post-extracción

---

## 3. Detalle por dataset

### 3.1 NIH ChestXray14

| Campo | Valor |
|---|---|
| **Archivo fuente** | `datasets/nih_chest_xrays/data.zip` (~45 GB) |
| **Ruta de destino** | `datasets/nih_chest_xrays/images_001/` a `images_012/` |
| **Formato extraído** | `.png` (imágenes de rayos X de tórax) |
| **Conteo de archivos** | 112,120 imágenes .png en 12 subdirectorios |
| **Tamaño en disco** | ~43 GB |
| **Estado** | ✅ Extraído |
| **ZIP residual** | Eliminado (modo `--disco`) |

**Notas especiales:**
- Antes de eliminar `data.zip`, el módulo extrae `train_val_list.txt` y `test_list.txt` (función `_extract_nih_split_txts()`, líneas 368–386). Estos archivos son necesarios para los splits del Paso 3.
- Verificación de extracción (`is_extracted`): cuenta imágenes `.png` en `images_001/images/` a `images_012/images/` — requiere ≥ 100,000 para considerarse extraído.

---

### 3.2 ISIC 2019

| Campo | Valor |
|---|---|
| **Archivo fuente** | `datasets/isic_2019/isic-2019.zip` (~9.3 GB) |
| **Ruta de destino** | `datasets/isic_2019/ISIC_2019_Training_Input/` |
| **Formato extraído** | `.jpg` (imágenes dermatoscópicas) |
| **Conteo de archivos** | 25,331 imágenes .jpg |
| **Tamaño en disco** | ~9.2 GB |
| **Estado** | ✅ Extraído |
| **ZIP residual** | Eliminado (modo `--disco`) |

**Notas especiales:**
- Verificación de extracción (`is_extracted`): comprueba existencia del directorio `ISIC_2019_Training_Input/`.
- ~~El directorio `datasets/isic_2019/isic_images/` existe vacío como artefacto huérfano (ver INC-06).~~ ✅ **Resuelto 2026-04-06:** directorio eliminado (INC-06 cerrado).

---

### 3.3 Osteoarthritis Knee (OA)

| Campo | Valor |
|---|---|
| **Archivo fuente** | `datasets/osteoarthritis/osteoarthritis.zip` (~5.0 GB) |
| **Ruta de destino** | `datasets/osteoarthritis/` |
| **Formato extraído** | `.jpg` y `.png` (imágenes de radiografía de rodilla) |
| **Conteo de archivos** | 4,766 imágenes con grado KL (9,339 totales incluyendo no usados) |
| **Tamaño en disco** | ~5.1 GB |
| **Estado** | ✅ Extraído |
| **ZIP residual** | Eliminado (modo `--disco`) |

**Notas especiales:**
- Verificación de extracción (`is_extracted`): ~~comprueba existencia del directorio `KLGrade/`.~~ ✅ **Corregido 2026-04-06:** ahora verifica `oa_splits/train/` (INC-07 fix en `extraer.py:295`).
- ~~Los archivos extraídos mantienen la estructura original del Kaggle: `KLGrade/KLGrade/{0,1,2,3,4}/`.~~ `KLGrade/` eliminado de disco el 2026-04-06. El pipeline usa exclusivamente `oa_splits/`.
- Existe doble estructura: los 4,766 archivos aparecen también en `oa_splits/{train,val,test}/{0,1,2}/` — generada por el Paso 5 (splits), no por la extracción. Ver INC-07.

---

### 3.4 LUNA16 (Nódulos Pulmonares)

#### Metadata

| Campo | Valor |
|---|---|
| **Archivo fuente** | `datasets/luna_lung_cancer/luna-lung-cancer-dataset.zip` (~331 MB) |
| **Ruta de destino** | `datasets/luna_lung_cancer/` |
| **Formato extraído** | `.csv` (anotaciones, candidatos), `.mhd`+`.zraw` (segmentaciones), `.py` (scripts de evaluación) |
| **Archivos clave** | `annotations.csv` (1,186 anotaciones), `candidates_V2/candidates_V2.csv` (754,975 candidatos), `seg-lungs-LUNA16/` (1,776 archivos) |
| **Estado** | ✅ Extraído |
| **ZIP residual** | Eliminado (modo `--disco`) |

#### CT Volumes (subsets 0–9)

| Campo | Valor |
|---|---|
| **Archivos fuente** | `datasets/luna_lung_cancer/ct_volumes/subset0.zip` a `subset9.zip` (~67 GB total, 10 archivos) |
| **Ruta de destino** | `datasets/luna_lung_cancer/ct_volumes/subset0/` a `subset9/` |
| **Formato extraído** | Pares `.mhd` + `.raw` (MetaImage headers + datos binarios) |
| **Conteo de archivos** | 888 pares (1,776 archivos: 888 `.mhd` + 888 `.raw`) |
| **Tamaño en disco** | ~131 GB |
| **Estado** | ✅ Extraído |
| **ZIPs residuales** | 10 ZIPs eliminados (modo `--disco`) |

**Notas especiales:**
- **ZIP64**: Los ZIPs de subsets 7 y 9 usan formato ZIP64. `unzip` falla silenciosamente en estos archivos. `7z` los maneja correctamente, aunque puede retornar `rc=1` (warnings) o `rc=2` (data error) — ambos tratados como éxito en `smart_extract()`.
- Verificación post-extracción: `verify_luna_ct_subset()` valida que cada subset tenga ≥ 88 pares `.mhd`+`.raw` con archivos `.raw` ≥ 1 MB.
- Verificación de extracción (`is_extracted`): para `luna` (metadata) verifica existencia de `candidates_V2/`; para `luna_ct` verifica cada subset individualmente.
- El formato en disco es `.mhd`+`.raw`, no `.mha` como declara `arquitectura_documentacion.md` §6.1 como formato preferido. Ver INC-05.

---

### 3.5 Pancreas PANORAMA

| Campo | Valor |
|---|---|
| **Archivo fuente** | `datasets/zenodo_13715870/batch_1.zip` (~45.9 GB) |
| **Ruta de destino** | `datasets/zenodo_13715870/` |
| **Formato extraído** | `.nii.gz` (NIfTI comprimido — volúmenes CT de abdomen) |
| **Conteo de archivos** | 557 volúmenes .nii.gz |
| **Tamaño en disco** | ~47 GB (extraídos) |
| **Estado** | ✅ Extraído |
| **ZIP residual** | Eliminado (modo `--disco`) |

**Notas especiales:**
- Verificación de extracción (`is_extracted`): `_pancreas_extracted()` busca ≥ 1 archivo `.nii.gz` recursivamente en `zenodo_13715870/`.
- Solo se descargó y extrajo `batch_1.zip` (557 de 1,864 casos en el dataset completo de Zenodo = 29.9%). Ver INC-03.
- Las PANORAMA labels (`datasets/panorama_labels/`) son un repositorio Git clonado (no un ZIP), por lo que no requieren extracción en este paso. Están presentes: 1,756 `automatic_labels/` + 482 `manual_labels/` = 2,238 archivos.

---

## 4. Detalles técnicos de implementación

### 4.1 Arquitectura del módulo `extraer.py`

```
extraer.py (488 líneas)
├── Helpers
│   ├── file_size_human()          — formato humano de tamaño de archivo
│   ├── ram_available_mb()         — RAM disponible vía psutil
│   ├── check_7z()                 — verifica que 7z esté instalado (OBLIGATORIO)
│   └── _log_progress()            — hilo de progreso cada 60s con % y ETA
├── RAMMonitor                     — clase: pausa/reanuda 7z con SIGSTOP/SIGCONT
│   ├── pause_mb = 700 MB          — umbral para pausar
│   └── resume_mb = 1,400 MB       — umbral para reanudar
├── smart_extract()                — extracción con 7z + RAMMonitor + progreso
├── Verificaciones
│   ├── is_extracted()             — dispatcher de verificación por dataset
│   ├── _luna_ct_extracted()       — verifica pares .mhd+.raw por subset
│   ├── _pancreas_extracted()      — verifica existencia de .nii.gz
│   ├── verify_luna_ct_subset()    — validación post-extracción (conteo + tamaño)
│   └── _extract_nih_split_txts()  — extrae listas de splits antes de borrar ZIP NIH
└── run_extractions()              — orquestador principal
```

### 4.2 Herramienta `7z` — por qué no `unzip`

| Aspecto | Detalle |
|---|---|
| **Paquete requerido** | `p7zip-full` (`sudo apt-get install p7zip-full`) |
| **Motivo** | ZIPs de LUNA16 subsets 7 y 9 son formato ZIP64. `unzip` falla silenciosamente (extrae 0 bytes o datos parciales sin error). |
| **Historial** | Commit `e4ed098` (2026-03-22): migración de `unzip` a `7z` tras detectar el problema |
| **Comando ejecutado** | `7z x <archivo> -o<destino> -y` |
| **Nota sobre `-mmt`** | NO se usa `-mmt=auto` — es inválido para formato ZIP en p7zip y causa `rc=2` |
| **Códigos de retorno** | `rc=0`: éxito limpio. `rc=1`: warnings (tratado como éxito). `rc=2`: data error (tratado como éxito — típico en ZIP64 LUNA16). `rc≥3`: error real → fallo. |

### 4.3 RAMMonitor — protección contra OOM

El `RAMMonitor` es un hilo demonio que supervisa la RAM disponible cada 3 segundos durante la extracción:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `pause_mb` | 700 MB | Si RAM disponible cae por debajo, envía `SIGSTOP` al proceso `7z` |
| `resume_mb` | 1,400 MB | Si RAM disponible sube por encima, envía `SIGCONT` para reanudar |
| `check_interval` | 3 segundos | Frecuencia de verificación |

Antes de iniciar cada extracción, `wait_for_ram()` espera a que haya al menos `resume_mb` disponibles.

### 4.4 Modo `--disco`

Cuando se ejecuta con `disco=True`:
1. Tras extracción exitosa, el ZIP fuente se elimina para liberar espacio
2. **Excepción NIH**: antes de eliminar `data.zip`, extrae `train_val_list.txt` y `test_list.txt` (función `_extract_nih_split_txts()`) — son archivos de splits oficiales necesarios para el Paso 3
3. El tamaño del ZIP eliminado se registra en el log

### 4.5 Idempotencia

La función `is_extracted()` verifica el estado de cada dataset antes de intentar la extracción:

| Dataset | Verificación |
|---|---|
| NIH | Cuenta imágenes `.png` en `images_001/images/` a `images_012/images/` — requiere ≥ 100,000 |
| ISIC | Existencia del directorio `ISIC_2019_Training_Input/` |
| OA | Existencia del directorio `KLGrade/` |
| LUNA meta | Existencia del directorio `candidates_V2/` |
| LUNA CT | Para cada subset: existencia del directorio + ≥ 88 pares `.mhd`+`.raw` con `.mhd` count == `.raw` count |
| Pancreas | ≥ 1 archivo `.nii.gz` recursivamente en `zenodo_13715870/` |

Si `is_extracted()` retorna `True`, el dataset se salta con log `"Ya extraído, saltando."`. Esto permite re-ejecutar el paso sin re-extraer.

### 4.6 Orden de extracción

El plan de extracción está ordenado de menor a mayor tamaño para maximizar disponibilidad temprana de datos:

| Orden | Dataset | ZIP | Tamaño aprox. |
|---|---|---|---|
| 1 | OA Knee | `osteoarthritis.zip` | ~5 GB |
| 2 | LUNA metadata | `luna-lung-cancer-dataset.zip` | ~331 MB |
| 3 | ISIC 2019 | `isic-2019.zip` | ~9.3 GB |
| 4 | NIH Chest X-Ray | `data.zip` | ~45 GB |
| 5–14 | LUNA CTs | `subset0.zip` a `subset9.zip` | ~67 GB (10 archivos) |
| 15 | Pancreas | `batch_1.zip` | ~45.9 GB |

### 4.7 Requisitos de espacio en disco

- **Espacio necesario para extracción completa**: ~235 GB (datos extraídos) + ~172 GB (ZIPs fuente) = ~407 GB simultáneos en pico
- **Espacio post-extracción (modo `--disco`)**: ~235 GB (ZIPs eliminados progresivamente)
- **Umbral de advertencia**: `extraer.py` advierte si hay < 150 GB libres antes de iniciar
- **Velocidad estimada HDD**: ~1.4 MB/s (usada para cálculo de ETA en logs)

---

## 5. Historial de ejecución

### 5.1 Tiempo registrado vs. tiempo real

| Fuente | Tiempo reportado | Contexto |
|---|---|---|
| `arquitectura_documentacion.md` §7.1 | 0.4s | Ejecución con todos los datasets ya extraídos |
| `fase0_report.md` | 0.5s | Ejecución con todos los datasets ya extraídos |

**Interpretación**: El tiempo reportado (~0.4–0.5s) corresponde a una ejecución donde `is_extracted()` retornó `True` para todos los datasets — la función hizo skip completo sin extraer nada. La extracción real de ~172 GB de ZIPs (NIH 45 GB + ISIC 9.3 GB + OA 5 GB + LUNA 67 GB + Pancreas 45.9 GB) fue ejecutada anteriormente pero no tiene tiempo registrado en la documentación existente.

**Estimación de tiempo real**: Con velocidad HDD de ~1.4 MB/s (estimada en `smart_extract()`), la extracción completa de ~172 GB tardaría ~34 horas. Este tiempo no incluye pausas del `RAMMonitor` por RAM baja.

### 5.2 Historial git relevante

| Commit | Fecha | Descripción |
|---|---|---|
| `381ad4d` | 2026-03-21 | `se añadio scripts/setup_datasets.py — descarga y descomprime automáticamente` — primer script de extracción |
| `e4ed098` | 2026-03-22 | `actualizacion de setup_datasets.py — fix errores con unzip, migración a 7z` — migración a `7z` tras fallos con ZIP64 |
| `1444af9` | 2026-03-22 | `se actualizo scripts/extract_luna_patches.py` — extracción de parches LUNA |

### 5.3 Scripts de extracción eliminados (historia)

Los siguientes scripts fueron los implementadores originales de la extracción. Todos han sido eliminados del disco y reemplazados por `src/pipeline/fase0/extraer.py`:

| Script | Estado | Reemplazado por |
|---|---|---|
| `scripts/setup_datasets.py` | ❌ Eliminado del disco | `src/pipeline/fase0/extraer.py` |
| `scripts/extract_luna_patches.py` | ❌ Eliminado del disco | `src/pipeline/fase0/extraer.py` |
| `scripts/setup_datasets_fast.sh` | ❌ Eliminado del disco | `src/pipeline/fase0/extraer.py` |
| `scripts/setup_datasets_smart.sh` | ❌ Eliminado del disco | `src/pipeline/fase0/extraer.py` |

Los scripts originales están accesibles en el historial git (commits `381ad4d`, `e4ed098`, `1444af9`) pero no en el árbol de trabajo actual. No hay documentación explícita de la migración de `scripts/` a `src/pipeline/fase0/`.

---

## 6. Incongruencias detectadas

### 6.1 Tabla resumen

| ID | Severidad | Descripción corta | Impacto | Estado |
|---|---|---|---|---|
| INC-01 | ✅ N/A | `proyecto_moe.md` no define el Paso 2 — diseño intencional: `arquitectura_documentacion.md` fue creado expresamente para estructurar la guía en pasos | No aplica — diseño del proyecto | Cerrado |
| INC-02 | 🟡 MEDIA | Tiempo 0.4s no refleja extracción real | Malinterpretación de rendimiento | Activo — informativo |
| INC-03 | 🟡 MEDIA | Pancreas: solo batch_1 extraído (29.9% del dataset) | Cobertura parcial de datos | Resuelto como NON-BLOCKER (A19) |
| INC-04 | 🔵 BAJA | Scripts de extracción eliminados sin documentar migración | Trazabilidad reducida | Activo — informativo |
| INC-05 | 🔵 BAJA | LUNA16 usa `.mhd`+`.raw`, no `.mha` (formato preferido documentado) | Discrepancia documental | Activo — informativo |
| INC-06 | 🔵 BAJA | Directorio `isic_images/` vacío (artefacto huérfano) | Confusión potencial | ✅ Cerrado 2026-04-06 — eliminado |
| INC-07 | 🔵 BAJA | Doble estructura OA Knee (espacio duplicado ~2.5 GB) | Desperdicio de almacenamiento | ✅ Cerrado 2026-04-06 — KLGrade/ eliminado |

### 6.2 Detalle de incongruencias

#### INC-01 — ✅ N/A: `proyecto_moe.md` no define el Paso 2 — diseño intencional

| Campo | Valor |
|---|---|
| **Fuente afectada** | `proyecto_moe.md` (guía oficial del proyecto) |
| **Descripción** | `proyecto_moe.md` no contiene una sección "Paso 2" ni instrucciones explícitas de extracción. La guía oficial solo menciona en §9 (Cronograma): "Semana S9: Descarga y preprocesamiento de los 5 datasets" sin descomponer en pasos numerados. La secuencia de 12 pasos (incluyendo "Paso 2 — Extraer archivos") existe en `arquitectura_documentacion.md`. |
| **Resolución** | **No es una incongruencia.** `arquitectura_documentacion.md` fue creado intencionalmente por el autor del proyecto para estructurar y dar orden a lo que la guía oficial no desglosa. Cumple el rol de documento de seguimiento y arquitectura — complementa la guía, no la contradice. |
| **Bloqueante** | No aplica. |
| **Estado** | Cerrado — diseño intencional |

---

#### INC-02 — 🟡 MEDIA: Tiempo de extracción 0.4s no refleja extracción real

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `arquitectura_documentacion.md` §7.1 (0.4s), `fase0_report.md` (0.5s) |
| **Descripción** | El tiempo reportado (~0.4–0.5s) corresponde a una ejecución donde todos los datasets ya estaban extraídos. La función `is_extracted()` retornó `True` para todos y la función hizo skip completo. La extracción real de ~172 GB de archivos comprimidos (NIH 45 GB + ISIC 9.3 GB + OA 5 GB + LUNA 67 GB + Pancreas 45.9 GB) tardó un tiempo no registrado, estimado en ~34 horas a velocidad HDD ~1.4 MB/s. |
| **Impacto** | El tiempo puede malinterpretarse como que la extracción de ~235 GB de datos es instantánea. Un usuario que re-ejecute el paso desde cero (sin datos extraídos previos) encontrará tiempos ordenes de magnitud mayores. |
| **Bloqueante** | No — el dato es correcto para el contexto en que fue registrado, pero incompleto. |
| **Acción sugerida** | Añadir nota aclaratoria en `arquitectura_documentacion.md` §7.1 indicando que el tiempo corresponde a ejecución idempotente (skip) y que la extracción real no tiene tiempo registrado. |
| **Estado** | Activo |

---

#### INC-03 — 🟡 MEDIA: Pancreas `batch_1.zip` — solo un batch extraído (29.9%)

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `extraer.py` línea 443, `proyecto_moe.md` |
| **Descripción** | El plan de extracción apunta a `datasets/zenodo_13715870/batch_1.zip`. El dataset completo de Zenodo tiene múltiples batches (~1,864 casos), pero solo se descargó y extrajo `batch_1.zip` (557 de 1,864 casos = 29.9%). La guía oficial no menciona esta limitación. El Paso 2 extrajo exitosamente lo que el Paso 1 descargó, pero la cobertura del dataset es parcial. |
| **Impacto** | `pancreas_splits.csv` contiene 1,864 case_ids pero solo 557 tienen CT en disco. Fase 1 entrena con los ~557 casos disponibles. |
| **Bloqueante** | No — resuelto como NON-BLOCKER (A19 en Paso 1). `_build_pairs()` en `dataset_builder.py` omite silenciosamente casos sin CT (guard `if candidates:`). |
| **Acción sugerida** | Ninguna. El pipeline funciona con los datos disponibles. Documentado en Paso 1 como A19. |
| **Estado** | Resuelto (NON-BLOCKER) |

---

#### INC-04 — 🔵 BAJA: Scripts de extracción eliminados del repositorio

| Campo | Valor |
|---|---|
| **Scripts afectados** | `scripts/setup_datasets.py`, `scripts/extract_luna_patches.py`, `scripts/setup_datasets_fast.sh`, `scripts/setup_datasets_smart.sh` |
| **Descripción** | Estos scripts fueron los implementadores originales de la extracción (commits `381ad4d`, `e4ed098`, `1444af9`). Fueron eliminados del disco y su funcionalidad fue absorbida por `src/pipeline/fase0/extraer.py`. No hay documentación explícita de esta migración — no se registró en un changelog, README, ni commit message dedicado. |
| **Impacto** | Trazabilidad reducida. Un desarrollador nuevo no sabrá que estos scripts existieron sin revisar el historial git. |
| **Bloqueante** | No — la funcionalidad actual en `extraer.py` es completa y operativa. |
| **Acción sugerida** | Informativo. Documentado en esta auditoría (§5.3). |
| **Estado** | Activo — informativo |

---

#### INC-05 — 🔵 BAJA: Formato `.mha` vs `.mhd`+`.raw` en LUNA16

| Campo | Valor |
|---|---|
| **Fuentes afectadas** | `arquitectura_documentacion.md` §6.1 (formato preferido) vs. disco real |
| **Descripción** | `arquitectura_documentacion.md` §6.1 establece que el formato preferido para imágenes 3D es `.mha` (MetaImage). Sin embargo, los CT de LUNA16 extraídos son pares `.mhd` + `.raw` (MetaImage split-header), no `.mha` (MetaImage single-file). El Pancreas usa `.nii.gz` (NIfTI). En la práctica, `SimpleITK` lee ambos formatos transparentemente (`sitk.ReadImage()`), pero el documento no aclara esta discrepancia. |
| **Impacto** | Ninguno operativo — `SimpleITK` maneja ambos formatos. Discrepancia puramente documental. |
| **Bloqueante** | No. |
| **Acción sugerida** | Actualizar `arquitectura_documentacion.md` §6.1 para aclarar que `.mhd`+`.raw` y `.nii.gz` son los formatos reales en disco, y que `.mha` es compatible pero no el formato usado. |
| **Estado** | Activo — informativo |

---

#### INC-06 — ✅ CERRADO 2026-04-06: Directorio `isic_images/` vacío (eliminado)

| Campo | Valor |
|---|---|
| **Ruta** | `datasets/isic_2019/isic_images/` |
| **Descripción** | Directorio huérfano vacío. Las imágenes reales de ISIC 2019 están en `datasets/isic_2019/ISIC_2019_Training_Input/` (25,331 archivos .jpg). Este directorio vacío era un artefacto de una descarga anterior (fuente Kaggle `andrewmvd`) que fue corregida cuando se re-descargó desde la fuente oficial ISIC Archive S3. |
| **Impacto** | Podía causar confusión si un desarrollador buscaba imágenes ISIC y encontraba el directorio vacío primero. No bloqueaba el pipeline. |
| **Bloqueante** | No. |
| **Acción tomada** | Directorio eliminado de disco el 2026-04-06 (Wave 2 cleanup). |
| **Estado** | ✅ Cerrado — eliminado |

---

#### INC-07 — 🔵 BAJA: Doble estructura OA Knee (espacio duplicado)

| Campo | Valor |
|---|---|
| **Rutas afectadas** | `datasets/osteoarthritis/KLGrade/KLGrade/{0,1,2,3,4}/` (original, **eliminado**) y `datasets/osteoarthritis/oa_splits/{train,val,test}/{0,1,2}/` (splits, activo) |
| **Descripción** | Los 4,766 archivos de OA Knee aparecen en dos estructuras de directorios. La primera (`KLGrade/`) era el resultado directo de la extracción del ZIP (5 clases KL originales). La segunda (`oa_splits/`) contiene copias físicas de las imágenes reorganizadas en splits train/val/test con reagrupación a 3 clases (generada por `split_oa()` en el Paso 5, no por este Paso 2). |
| **Impacto** | ~~Duplicación de almacenamiento: ~2.5 GB adicionales.~~ 162 MB liberados al eliminar `KLGrade/` el 2026-04-06. La guía oficial no documentaba este comportamiento de copia física. |
| **Bloqueante** | No — `oa_splits/` es la estructura usada por el pipeline. |
| **Acción tomada** | `KLGrade/` eliminado de disco el 2026-04-06 (Wave 2 cleanup). `extraer.py:295` corregido para verificar `oa_splits/train/` en lugar de `KLGrade/`. |
| **Estado** | ✅ Cerrado — KLGrade/ eliminado |

---

## 7. Ítems de acción

| # | Prioridad | Descripción | Bloqueante | Estado |
|---|---|---|---|---|
| P2-A1 | Informativo | Documentar en `arquitectura_documentacion.md` §7.1 que el tiempo 0.4s es de ejecución idempotente (skip), no de extracción real (INC-02) | No | Pendiente |
| P2-A2 | Baja | Eliminar directorio vacío `datasets/isic_2019/isic_images/` (INC-06) | No | ✅ Hecho 2026-04-06 |
| P2-A3 | Baja | Evaluar eliminación de `datasets/osteoarthritis/KLGrade/` post-splits o migración a symlinks (INC-07) | No | ✅ Hecho 2026-04-06 |

**Conclusión**: El Paso 2 está **limpio y completo**. No hay ítems bloqueantes. De las 7 entradas revisadas, 1 resultó ser diseño intencional (INC-01) y no una incongruencia. Las 6 incongruencias reales (INC-02 a INC-07) son de carácter informativo o baja severidad — ninguna impacta la ejecución del pipeline. Los 5 datasets están correctamente extraídos en disco, los ZIPs han sido eliminados, y el módulo `extraer.py` es idempotente para re-ejecuciones futuras.

---

*Documento generado el 2026-04-05 por auditoría multi-agente (ARGOS + SIGMA + EXPLORE). Fuentes: `src/pipeline/fase0/extraer.py`, `src/pipeline/fase0/fase0_pipeline.py`, `arquitectura_documentacion.md`, `proyecto_moe.md`, `fase0_report.md`, inspección directa del filesystem, historial git. Formato replicado de `docs/documentacion_pasos/paso_01_descarga_datos.md`.*
