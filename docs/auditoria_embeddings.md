# Auditoría de Embeddings — Fase 1 MoE Imágenes Médicas

**Fecha:** 2026-04-01  
**Auditor:** ARGOS Data Audit Agent  
**Veredicto Global:** ⚠️ **WARN** (condicionalmente apto para Fase 2)

---

## 1. Resumen Ejecutivo

| Backbone | Archivos | Shapes | NaN/Inf | Labels | Names | Inter-BB | E1 Collapse | Leakage | Magnitudes |
|---|---|---|---|---|---|---|---|---|---|
| `vit_tiny_patch16_224` | ✅ 10/10 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ E1 colapsado | ⚠️ 180 LUNA16 | ✅ |
| `densenet121_custom` | ✅ 10/10 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ E1 colapsado | ⚠️ 180 LUNA16 | ⚠️ magnitudes ~10⁷ |
| `swin_tiny_patch4_window7_224` | ✅ 10/10 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ E1 colapsado | ⚠️ 180 LUNA16 | ✅ |
| `cvt_13` | ✅ 10/10 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ E1 colapsado | ⚠️ 180 LUNA16 | ✅ |

**Leyenda:** ✅ = OK | ⚠️ = Warning (proceder con precaución) | ❌ = Fallo crítico

---

## 2. Detalle de Hallazgos

### 2.1 Estructura de Archivos (PASS ✅)

Todos los backbones contienen los 10 archivos requeridos. Los tamaños de archivo coinciden exactamente con lo esperado (`N × d_model × 4 bytes + header`).

| Backbone | Z_train | Z_val | Z_test | y_train | y_val | y_test |
|---|---|---|---|---|---|---|
| `vit_tiny_patch16_224` | 98.5 MB | 11.7 MB | 12.6 MB | 1003 KB | 120 KB | 129 KB |
| `densenet121_custom` | 525.6 MB | 62.6 MB | 67.2 MB | 1003 KB | 120 KB | 129 KB |
| `swin_tiny_patch4_window7_224` | 394.2 MB | 47.0 MB | 50.4 MB | 1003 KB | 120 KB | 129 KB |
| `cvt_13` | 197.1 MB | 23.5 MB | 25.2 MB | 1003 KB | 120 KB | 129 KB |

`backbone_meta.json` parseable y correcto en todos los casos. Campos `d_model`, `n_train`, `n_val`, `n_test` coinciden con los valores esperados.

### 2.2 Shapes y Dimensiones (PASS ✅)

Todos los shapes son exactos:

| Backbone | d_model | Z_train | Z_val | Z_test | dtype |
|---|---|---|---|---|---|
| `vit_tiny_patch16_224` | 192 | (128320, 192) ✅ | (15292, 192) ✅ | (16397, 192) ✅ | float32 |
| `densenet121_custom` | 1024 | (128320, 1024) ✅ | (15292, 1024) ✅ | (16397, 1024) ✅ | float32 |
| `swin_tiny_patch4_window7_224` | 768 | (128320, 768) ✅ | (15292, 768) ✅ | (16397, 768) ✅ | float32 |
| `cvt_13` | 384 | (128320, 384) ✅ | (15292, 384) ✅ | (16397, 384) ✅ | float32 |

Labels: `y_{split}.shape = (N,)`, dtype=`int64` en todos los casos.
Names: `names_{split}.txt` tiene exactamente N líneas en todos los casos.

**Conteos verificados:**
- train: N = 128,320 ✅
- val: N = 15,292 ✅
- test: N = 16,397 ✅

### 2.3 Valores Numéricos en Z (PASS con WARNING ⚠️)

#### Zero NaN/Inf: ✅ Correcto en todos los backbones y splits

#### Estadísticas Globales Z_train

| Backbone | Mean | Std | Min | Max | p1 | p50 | p99 |
|---|---|---|---|---|---|---|---|
| `vit_tiny` | 0.000000 | 1.000 | -3.768 | 4.344 | -2.241 | -0.056 | 2.491 |
| `densenet121` | -123,288 | 3,888,074 | -29,262,482 | 32,689,988 | -10,053,220 | -0.006 | 9,688,424 |
| `swin_tiny` | -0.000000 | 0.710 | -3.238 | 3.547 | -1.676 | -0.008 | 1.778 |
| `cvt_13` | -0.036 | 0.531 | -1.760 | 1.661 | -1.307 | -0.031 | 1.154 |

#### Columnas Degeneradas (std=0 o std<1e-6)

| Backbone | Columnas con std=0 (global) | Columnas con std<1e-6 (global) |
|---|---|---|
| `vit_tiny` | 0/192 | 0/192 |
| `densenet121` | 0/1024 | 0/1024 |
| `swin_tiny` | 0/768 | 0/768 |
| `cvt_13` | 0/384 | 0/384 |

> **Nota:** A nivel global no hay columnas degeneradas, pero a nivel de E1 (ISIC) sí hay colapso total. Ver Sección 2.7.

#### Norma L2

| Backbone | L2 Mean | L2 Std | L2 Min | L2 Max | L2/√d |
|---|---|---|---|---|---|
| `vit_tiny` | 13.856 | 0.000013 | 13.856 | 13.856 | 1.0000 |
| `densenet121` | 112,082,736 | 54,156,780 | 0.589 | 306,166,944 | — |
| `swin_tiny` | 19.098 | 4.721 | 10.296 | 27.713 | 0.689 |
| `cvt_13` | 10.429 | 0.048 | 10.248 | 10.723 | 0.532 |

**Hallazgos:**

- **⚠️ WARN-01: `densenet121_custom` tiene magnitudes extremas (~10⁷).** El backbone DenseNet121 no incluye normalización de salida (BatchNorm/LayerNorm en la capa final). Los valores de embedding abarcan de -29M a +33M. Esto es consecuencia de la arquitectura custom con pesos aleatorios. **No es un error per se** — los embeddings son numéricamente válidos (sin NaN/Inf) — pero requiere normalización previa al entrenamiento MoE.

- **⚠️ WARN-02: `vit_tiny_patch16_224` tiene L2 prácticamente constante** (std=0.0000134). Todos los vectores tienen L2 ≈ √192 = 13.8564, con diferencia max-min de 0.000058. Esto es efecto del LayerNorm final del ViT. **No es un error** — es comportamiento esperado de la arquitectura.

### 2.4 Labels (PASS ✅)

#### Valores únicos: `{0, 1, 2, 3, 4}` en todos los backbones y splits ✅
#### dtype: `int64` en todos los casos ✅

#### Conteos exactos verificados:

| Experto | Esperado Train | Actual Train | Esperado Val | Actual Val | Esperado Test | Actual Test |
|---|---|---|---|---|---|---|
| E0 (NIH) | 88,999 | 88,999 ✅ | 11,349 | 11,349 ✅ | 11,772 | 11,772 ✅ |
| E1 (ISIC) | 18,767 | 18,767 ✅ | 2,276 | 2,276 ✅ | 2,214 | 2,214 ✅ |
| E2 (OA Rodilla) | 3,814 | 3,814 ✅ | 480 | 480 ✅ | 472 | 472 ✅ |
| E3 (LUNA16) | 16,567 | 16,567 ✅ | 1,143 | 1,143 ✅ | 1,914 | 1,914 ✅ |
| E4 (Páncreas) | 173 | 173 ✅ | 44 | 44 ✅ | 25 | 25 ✅ |

Conteos byte-a-byte idénticos entre los 4 backbones.

### 2.5 Names (PASS con WARNING ⚠️)

- Sin líneas vacías ✅
- Sin duplicados dentro de cada split ✅
- Formatos de nombre coherentes por experto ✅:

| Experto | Formato | Ejemplo |
|---|---|---|
| E0 (NIH ChestXray14) | `XXXXXXXX_XXX.png` | `00000001_000.png` |
| E1 (ISIC 2019) | `ISIC_XXXXXXX` | `ISIC_0000000` |
| E2 (OA Rodilla) | `X_XXX_png.rf.HASH.jpg` | `0_110_png.rf.640ac861f605968e142166b62b883aa2.jpg` |
| E3 (LUNA16) | `candidate_XXXXXX` | `candidate_000080` |
| E4 (Páncreas) | `XXXXXX_XXXXX_XXXX` | `100074_00001_0000` |

#### ⚠️ WARN-03: Data Leakage entre Train y Val/Test (LUNA16 — E3)

| Overlap | Cantidad | % de E3 en split destino |
|---|---|---|
| train ∩ val | **50 nombres** | 4.37% del E3 val |
| train ∩ test | **130 nombres** | 6.79% del E3 test |
| val ∩ test | 0 | — |
| **Total filtraciones** | **180** | 1.09% del E3 train |

**Todos los 180 nombres filtrados pertenecen exclusivamente al Experto 3 (LUNA16)**, con formato `candidate_XXXXXX`. Las labels coinciden (train_label=3, val/test_label=3). Los embeddings son **byte-a-byte idénticos** en train y val/test para las muestras filtradas (L2_diff=0.0 en todos los casos verificados).

**Causa probable:** El dataset LUNA16 tiene candidatos duplicados entre subsets, y el split train/val/test del pipeline de Fase 1 no realizó deduplicación cruzada suficiente a nivel de nombre de candidato.

**Impacto:** 0.14% del total del dataset. Bajo impacto en la evaluación global, pero inflará métricas del Experto 3 en val/test entre 4-7%.

### 2.6 Consistencia Inter-backbone (PASS ✅)

| Archivo | md5 idéntico entre 4 backbones |
|---|---|
| `y_train.npy` | ✅ `0861ecdfa23879479e64f41b9e1a613a` |
| `y_val.npy` | ✅ `c139a2752b7ae36dd95507924ae62b5e` |
| `y_test.npy` | ✅ `1ca367103e4db72b0477cc87bce0e657` |
| `names_train.txt` | ✅ `fbe3b1f5fbed7a0342b3ca919d8605e9` |
| `names_val.txt` | ✅ `f4db08d92d0fb3d1b539c841e6b4a223` |
| `names_test.txt` | ✅ `f320d83ce04460f02b2d7f7c97c5f34d` |

Los archivos de labels y nombres son **byte-a-byte idénticos** entre los 4 backbones. Esto confirma que el mismo pipeline de datos generó todos los embeddings con el mismo orden de muestras.

### 2.7 ⚠️ WARN-04 (CRÍTICA): Experto 1 (ISIC 2019) — Embeddings Completamente Colapsados

**Todos los 23,257 embeddings del Experto 1 (ISIC 2019) son idénticos** en los 4 backbones, en los 3 splits.

| Backbone | Split | n_E1 | Todas las filas idénticas | L2 std |
|---|---|---|---|---|
| `vit_tiny` | train | 18,767 | **SÍ** | 0.000001 |
| `vit_tiny` | val | 2,276 | **SÍ** | 0.000000 |
| `vit_tiny` | test | 2,214 | **SÍ** | 0.000000 |
| `densenet121` | train | 18,767 | **SÍ** | 0.000000 |
| `densenet121` | val | 2,276 | **SÍ** | 0.000000 |
| `densenet121` | test | 2,214 | **SÍ** | 0.000000 |
| `swin_tiny` | train | 18,767 | **SÍ** | 0.000002 |
| `swin_tiny` | val | 2,276 | **SÍ** | 0.000000 |
| `swin_tiny` | test | 2,214 | **SÍ** | 0.000000 |
| `cvt_13` | train | 18,767 | **SÍ** | 0.000001 |
| `cvt_13` | val | 2,276 | **SÍ** | 0.000002 |
| `cvt_13` | test | 2,214 | **SÍ** | 0.000001 |

Adicionalmente, en `densenet121_custom` se encontró **1 muestra de LUNA16 (E3)** con embedding idéntico al vector ISIC constante:
- `candidate_494241` (idx=127601, L2=0.5889 = igual al vector ISIC de DenseNet)

**Causa probable:** El pipeline de preprocesamiento del Experto 1 (ISIC 2019) genera una imagen de entrada idéntica para todas las muestras (posiblemente una imagen negra/blanca/constante), o el loader siempre devuelve el mismo tensor. Como los backbones usan pesos aleatorios congelados, una entrada constante produce una salida constante.

**Impacto:**
- El Experto 1 no será aprendible por el gating network, ya que todos sus embeddings son un único punto en el espacio.
- Representa el 14.6% del dataset total (18,767 muestras de train).
- El gating network solo podrá distinguir E1 del resto si el vector constante es suficientemente diferente a las otras distribuciones (lo es en norma L2 pero no en dirección discriminante).

**Verificación a realizar:** Inspeccionar el pipeline de carga de imágenes ISIC para determinar si la causa es una imagen constante, un error en la ruta, o un bug en la transformación.

### 2.8 Proporciones por Experto

| Experto | Train % | Val % | Test % | Ratio Train:Val:Test |
|---|---|---|---|---|
| E0 (NIH) | 69.36% | 74.22% | 71.79% | 79.4 : 10.1 : 10.5 |
| E1 (ISIC) | 14.63% | 14.88% | 13.50% | 80.7 : 9.8 : 9.5 |
| E2 (OA Rodilla) | 2.97% | 3.14% | 2.88% | 80.0 : 10.1 : 9.9 |
| E3 (LUNA16) | 12.91% | 7.47% | 11.67% | 84.4 : 5.8 : 9.8 |
| E4 (Páncreas) | 0.13% | 0.29% | 0.15% | 71.5 : 18.2 : 10.3 |

**Observaciones:**
- E3 (LUNA16) tiene un ratio de validación bajo (5.8%) comparado con los demás (~10%). Esto puede dificultar la evaluación del gate para este experto.
- E4 (Páncreas) tiene un ratio de validación alto (18.2%) debido al bajo conteo absoluto (44 muestras), y solo 25 en test. Las métricas de E4 tendrán alta varianza.

### 2.9 Fisher Ratio — Separabilidad

| Backbone | Var Inter-clase | Var Intra-clase | Fisher Ratio |
|---|---|---|---|
| `vit_tiny` | 67.07 | 37.33 | **1.797** |
| `densenet121` | 2.34×10¹⁵ | 6.02×10¹⁴ | **3.877** |
| `swin_tiny` | 105.57 | 47.71 | **2.213** |
| `cvt_13` | 1.12 | 0.85 | **1.331** |

**Interpretación:** Con pesos aleatorios, se espera un Fisher ratio cercano a 1.0 (sin separabilidad). Los ratios observados (1.3–3.9) son moderados y se explican por:
1. El **colapso de E1 (ISIC)** genera un cluster puntual muy diferenciado que aumenta artificialmente la varianza inter-clase.
2. Las **diferencias de escala entre expertos** (ej. DenseNet tiene E1 en L2~0.6 vs E0 en L2~10⁸) contribuyen al ratio elevado.
3. El ratio de DenseNet (3.88) es el más alto, consistente con las magnitudes extremas y el colapso.

**Conclusión:** Los ratios no indican separabilidad real sino artefactos del colapso E1 y las diferencias de escala. Tras eliminar el colapso E1, se espera ratio ~1.0.

---

## 3. Inventario de Inconsistencias

| # | Severidad | Backbone | Hallazgo | Impacto |
|---|---|---|---|---|
| W-01 | ⚠️ MEDIA | densenet121_custom | Magnitudes de embedding ~10⁷ (sin normalización final) | Requiere normalización antes de Fase 2 |
| W-02 | ℹ️ BAJA | vit_tiny_patch16_224 | L2 constante (efecto LayerNorm, L2 ≈ √192) | Comportamiento esperado, no requiere acción |
| W-03 | ⚠️ MEDIA | TODOS | 180 nombres LUNA16 (E3) filtrados entre train↔val/test | 4-7% de inflación en métricas E3 |
| **W-04** | **🔴 ALTA** | **TODOS** | **23,257 embeddings E1 (ISIC) colapsados (todas las filas idénticas)** | **E1 no será aprendible por gating** |
| W-05 | ℹ️ BAJA | densenet121_custom | 1 muestra LUNA16 (`candidate_494241`) tiene embedding idéntico al vector ISIC constante | Anomalía puntual |
| W-06 | ℹ️ BAJA | TODOS | E3 (LUNA16) tiene ratio val bajo (5.8%) y E4 ratio val alto (18.2%) | Evaluación con alta varianza en E4 |

---

## 4. Verificaciones Superadas (sin hallazgos)

- [x] Todos los 40 archivos presentes (10 × 4 backbones)
- [x] Tamaños de archivo correctos
- [x] `backbone_meta.json` parseable con `d_model` correcto
- [x] Shapes exactos: `(N, d_model)` para Z, `(N,)` para y, N líneas para names
- [x] Conteos N exactos: train=128,320 / val=15,292 / test=16,397
- [x] Zero NaN en todos los embeddings
- [x] Zero Inf en todos los embeddings
- [x] Zero columnas con std=0 a nivel global
- [x] Labels: valores exactamente `{0, 1, 2, 3, 4}`
- [x] Labels: conteos exactos por experto y split
- [x] Labels: dtype `int64` (compatible con int)
- [x] Names: sin líneas vacías
- [x] Names: sin duplicados intra-split
- [x] val ∩ test = 0 (sin leakage val↔test)
- [x] `y_{split}.npy` byte-a-byte idéntico entre 4 backbones
- [x] `names_{split}.txt` byte-a-byte idéntico entre 4 backbones
- [x] Formatos de nombres coherentes por experto

---

## 5. Conclusiones y Recomendaciones

### ¿Son los embeddings válidos para continuar a Fase 2?

**Sí, condicionalmente.** Los embeddings son estructuralmente correctos y numéricamente válidos. Sin embargo, hay dos hallazgos que requieren atención:

#### Acción OBLIGATORIA antes de Fase 2:

1. **Investigar y corregir el colapso de E1 (ISIC 2019).** Todas las 23,257 muestras ISIC producen el mismo embedding en los 4 backbones. Esto indica un bug en el pipeline de carga/preprocesamiento de imágenes ISIC — no en los backbones. Se recomienda:
   - Verificar que el dataloader ISIC no devuelve siempre la misma imagen.
   - Inspeccionar las transformaciones aplicadas (¿se está aplicando un resize/crop que colapsa a un valor constante?).
   - Regenerar los embeddings de E1 tras corregir el bug.
   - Si no se puede corregir: proceder sin E1, pero el gating network no podrá rutear correctamente las muestras de dermoscopía.

#### Acciones RECOMENDADAS:

2. **Normalización de DenseNet121.** Aplicar normalización z-score por columna o LayerNorm a los embeddings de `densenet121_custom` antes de alimentar al gating network. Las magnitudes ~10⁷ pueden causar inestabilidad numérica en el entrenamiento.

3. **Documentar el leakage LUNA16.** Los 180 candidatos duplicados representan 0.14% del dataset. Si bien el impacto global es bajo, las métricas de E3 en val/test estarán ligeramente infladas (4-7%). Dos opciones:
   - (Ideal) Eliminar los 180 nombres duplicados de val/test y regenerar.
   - (Aceptable) Documentar la inflación y reportar métricas ajustadas.

4. **Evaluar el desbalance de E4 (Páncreas).** Con solo 173 muestras de train, 44 de val y 25 de test, el Experto 4 es extremadamente vulnerable a la varianza de evaluación. Considerar oversampling o métricas robustas (bootstrapped CI) para E4.

---

## 6. Reproducibilidad

Todas las verificaciones fueron ejecutadas con el siguiente entorno:
```
Python: /home/mrsasayo_mesa/venv_global/bin/python
NumPy: disponible en el entorno
Base: /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/embeddings/
```

Código de auditoría completo disponible en scripts temporales. Resultados numéricos reproducibles dado que los archivos .npy son determinísticos.

---

*Fin del reporte de auditoría.*
