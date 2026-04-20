# Análisis Estadístico 1 — Distribución y Correlaciones de Complejidad

**Fecha:** 2026-04-19
**Alcance:** 97 archivos Python en `src/pipeline/`
**Método:** Análisis estático (LOC, CC, type hints, docstrings, health score)

---

## 1. Estadísticas Descriptivas

### 1.1 Lines of Code (LOC — líneas no vacías, sin comentarios puros)

| Métrica | Valor |
|---------|-------|
| N (archivos) | 97 |
| Media | 342.0 |
| Mediana | 245.0 |
| Desv. Estándar | 300.1 |
| Mínimo | 42 |
| Máximo | 1,714 |
| Q1 (percentil 25) | 135 |
| Q3 (percentil 75) | 480 |
| IQR | 345 |

### 1.2 Complejidad Ciclomática Máxima por Archivo (Max CC)

| Métrica | Valor |
|---------|-------|
| Media | 14.8 |
| Mediana | 11.0 |
| Desv. Estándar | 13.4 |
| Mínimo | 1 |
| Máximo | 56 |
| Q1 | 5 |
| Q3 | 21 |
| IQR | 16 |

### 1.3 Complejidad Ciclomática Promedio por Archivo (Avg CC)

| Métrica | Valor |
|---------|-------|
| Media | 5.3 |
| Mediana | 4.8 |
| Desv. Estándar | 4.1 |
| Mínimo | 0.0 |
| Máximo | 19.0 |
| Q1 | 2.4 |
| Q3 | 7.4 |
| IQR | 5.0 |

### 1.4 Health Score (0–100)

| Métrica | Valor |
|---------|-------|
| Media | 85.5 |
| Mediana | 90.0 |
| Desv. Estándar | 16.2 |
| Mínimo | 20.8 |
| Máximo | 100.0 |
| Q1 | 80.0 |
| Q3 | 100.0 |
| IQR | 20.0 |

### 1.5 Type Hints Coverage (%)

| Métrica | Valor |
|---------|-------|
| Media | 57.3% |
| Mediana | 75.0% |
| Desv. Estándar | 42.4% |
| Q1 | 0.0% |
| Q3 | 100.0% |

### 1.6 Docstring Coverage (%)

| Métrica | Valor |
|---------|-------|
| Media | 62.9% |
| Mediana | 75.0% |
| Desv. Estándar | 36.0% |
| Q1 | 44.4% |
| Q3 | 91.7% |

---

## 2. Distribuciones

### 2.1 Histograma: LOC por Archivo

```
LOC Range        │ Count │ ████████████████████████████████████████
─────────────────┼───────┼──────────────────────────────────────────
    0 –  100     │  18   │ ██████████████████
  101 –  200     │  21   │ █████████████████████
  201 –  300     │  12   │ ████████████
  301 –  400     │   8   │ ████████
  401 –  500     │   7   │ ███████
  501 –  700     │  13   │ █████████████
  701 –  900     │  10   │ ██████████
  901 – 1100     │   4   │ ████
 1100 – 1500     │   3   │ ███
 1500+           │   1   │ █
```

**Observación:** Distribución sesgada a la derecha (right-skewed). El 60% de archivos tienen <300 LOC. Cola larga con 8 archivos >700 LOC.

### 2.2 Histograma: CC Máxima por Archivo

```
Max CC Range     │ Count │ ████████████████████████████████████████
─────────────────┼───────┼──────────────────────────────────────────
   1 –   5       │  30   │ ██████████████████████████████
   6 –  10       │  17   │ █████████████████
  11 –  15       │  11   │ ███████████
  16 –  20       │  11   │ ███████████
  21 –  25       │  10   │ ██████████
  26 –  35       │   9   │ █████████
  36 –  45       │   5   │ █████
  46 –  56       │   4   │ ████
```

**Observación:** 49% de archivos con CC ≤ 10 (saludable). 18 archivos (19%) con CC > 20 concentran el riesgo.

### 2.3 Distribución: Type Hints % por Archivo

```
Type Hints %     │ Count │
─────────────────┼───────┼──────────────────────────────────────────
     0%          │  30   │ ██████████████████████████████
   1% –  50%     │   5   │ █████
  51% –  75%     │  10   │ ██████████
  76% –  99%     │  16   │ ████████████████
   100%          │  36   │ ████████████████████████████████████
```

**Observación:** Distribución bimodal. 37% de archivos tienen 100% type hints, pero 31% tienen 0%. Los archivos sin type hints son predominantemente de `fase0/` (preprocesamiento) y archivos de configuración.

### 2.4 Distribución: Docstring Coverage % por Archivo

```
Docstring %      │ Count │
─────────────────┼───────┼──────────────────────────────────────────
     0%          │  18   │ ██████████████████
   1% –  50%     │  10   │ ██████████
  51% –  75%     │  18   │ ██████████████████
  76% –  99%     │  20   │ ████████████████████
   100%          │  31   │ ███████████████████████████████
```

**Observación:** Mejor cobertura que type hints. Solo 18 archivos sin docstrings (mayoría configs).

---

## 3. Correlaciones (Pearson)

### 3.1 Matriz de Correlación

```
                │  LOC   │ Max CC │ Health │ Hints% │ Docs%  │ Avg CC
────────────────┼────────┼────────┼────────┼────────┼────────┼───────
LOC             │  1.000 │  0.791 │ -0.712 │  0.184 │  0.115 │  0.620
Max CC          │  0.791 │  1.000 │ -0.685 │  0.191 │  0.082 │  0.556
Health          │ -0.712 │ -0.685 │  1.000 │  0.155 │  0.090 │ -0.488
Type Hints %    │  0.184 │  0.191 │  0.155 │  1.000 │  0.581 │  0.094
Docstring %     │  0.115 │  0.082 │  0.090 │  0.581 │  1.000 │  0.042
Avg CC          │  0.620 │  0.556 │ -0.488 │  0.094 │  0.042 │  1.000
```

### 3.2 Heatmap: Correlation Matrix

```
              LOC    MaxCC  Health  Hints  Docs   AvgCC
         ┌────────────────────────────────────────────┐
LOC      │ ■■■■■  ■■■■   ░░░░   ·      ·      ■■■  │
MaxCC    │ ■■■■   ■■■■■  ░░░░   ·      ·      ■■■  │
Health   │ ░░░░   ░░░░   ■■■■■  ·      ·      ░░   │
Hints    │ ·      ·      ·      ■■■■■  ■■■    ·    │
Docs     │ ·      ·      ·      ■■■    ■■■■■  ·    │
AvgCC    │ ■■■    ■■■    ░░     ·      ·      ■■■■■│
         └────────────────────────────────────────────┘
■ = correlación positiva fuerte   ░ = correlación negativa fuerte   · = débil/nula
```

### 3.3 Interpretación de Correlaciones

| Correlación | r | Interpretación |
|------------|-------|----------------|
| **LOC vs Max CC** | **0.791** | **Fuerte positiva.** Archivos más grandes tienen funciones más complejas. |
| **LOC vs Health** | **-0.712** | **Fuerte negativa.** Archivos grandes tienen peor health score. |
| **Max CC vs Health** | **-0.685** | **Fuerte negativa.** Alta complejidad ciclomática degrada salud. |
| **Type Hints vs Docstrings** | **0.581** | **Moderada positiva.** Archivos bien documentados tienden a tener type hints. |
| **LOC vs Type Hints** | 0.184 | Débil. El tamaño no predice cobertura de type hints. |
| **LOC vs Docstrings** | 0.115 | Débil. Igual para docstrings. |

---

## 4. Scatter Plots

### 4.1 Scatter: LOC vs Health Score

```
Health
100 │ ·····*··*·*··*·**··*···*··*·*····*·*····*·
    │  ··*···*····*··· ·*·*··
 80 │ ····*···*·*···*····
    │ ··*·····
 60 │          ·*·  ·*·  **··
    │          ···  · ··
 40 │                     ·  ·
    │
 20 │                              ·
    └──────────────────────────────────────────── LOC
    0    200   400   600   800  1000  1200  1400  1714

Tendencia: lineal descendente (r = -0.712)
```

### 4.2 Scatter: Max CC vs Health Score

```
Health
100 │ ****·**····*··*····*·*·*····
    │  ·····*···*··
 80 │ ·*·*···· ····
    │        ··  ·
 60 │            ··  · ·· · ·· ·
    │             ·       ·
 40 │                  ·      ·
    │
 20 │                         ·
    └──────────────────────────────── Max CC
    1    10    20    30    40    50    56

Tendencia: descendente (r = -0.685)
```

### 4.3 Box Plot: Health Score Distribution

```
     ├──────────┬─────────────────────────────────────────┤
     │          │                                         │
  ───┤    ╔═════╪══════════════════╗                      │
     │    ║     │   MEDIAN=90      ║                      │
  ───┤    ╚═════╪══════════════════╝                      │
     │          │                                         │
     ├──────────┴─────────────────────────────────────────┤
   20.8        80    90           100
              Q1          Q3

   Whisker inferior: 20.8 (min)
   Q1: 80.0
   Mediana: 90.0
   Q3: 100.0
   IQR: 20.0
   Outlier inferior: pre_embeddings.py (20.8)
```

---

## 5. Outliers y Anomalías

### 5.1 Archivos Anormalmente Grandes (LOC > 998, umbral IQR×1.5)

| Archivo | LOC | Max CC | Health | Riesgo |
|---------|-----|--------|--------|--------|
| `fase0/pre_embeddings.py` | 1,714 | 45 | 20.8 | **CRÍTICO** |
| `fase6/paso10_verificacion.py` | 1,254 | 31 | 39.9 | **ALTO** |

### 5.2 Archivos con CC Anormalmente Alta (Max CC > 45, umbral IQR×1.5)

| Archivo | Max CC | Función más compleja | LOC | Health |
|---------|--------|----------------------|-----|--------|
| `fase3/train_expert5_ddp.py` | 56 | `train()` | 789 | 64.2 |
| `fase2/train_expert3_ddp.py` | 55 | `train()` | 801 | 64.0 |
| `fase2/train_expert4_ddp.py` | 52 | `train()` | 737 | 65.3 |
| `fase2/train_expert1_ddp.py` | 49 | `train()` | 922 | 61.6 |

**Patrón:** Todas las funciones `train()` de los scripts DDP tienen CC extrema. Son candidatas a refactorización extrayendo subfunciones.

### 5.3 Archivos sin Type Hints (30 archivos, LOC > 100)

| Archivo | LOC | Impacto |
|---------|-----|---------|
| `fase0/pre_embeddings.py` | 1,714 | **Crítico** — archivo más grande sin type hints |
| `fase0/pre_modelo.py` | 939 | Alto |
| `fase1/fase1_pipeline.py` | 909 | Alto |
| `fase0/fase0_pipeline.py` | 765 | Alto |
| `fase0/descargar.py` | 513 | Medio |
| `fase0/extraer.py` | 505 | Medio |
| `fase1/dataset_builder.py` | 333 | Medio |
| `fase1/backbone_loader.py` | 175 | Bajo |
| `fase2/routers/gmm.py` | 161 | Bajo |
| `datasets/osteoarthritis.py` | 145 | Bajo |

**Patrón:** `fase0/` (preprocesamiento) tiene la peor cobertura de type hints. Los archivos de configuración (0 funciones) están excluidos del análisis.

### 5.4 Archivos sin Docstrings (LOC > 100)

| Archivo | LOC |
|---------|-----|
| `fase0/pre_modelo.py` | 939 |
| `fase0/descargar.py` | 513 |
| `fase0/extraer.py` | 505 |
| `datasets/osteoarthritis.py` | 145 |
| `fase2/losses.py` | 117 |

### 5.5 Archivos de Mayor Riesgo Combinado (Health < 50)

| Archivo | LOC | Max CC | Type Hints | Docstrings | Health |
|---------|-----|--------|------------|------------|--------|
| `fase0/pre_embeddings.py` | 1,714 | 45 | 0% | 29% | **20.8** |
| `fase6/paso10_verificacion.py` | 1,254 | 31 | 100% | 88% | **39.9** |
| `fase0/pre_modelo.py` | 939 | 35 | 0% | 0% | **41.2** |
| `fase0/fase0_pipeline.py` | 765 | 29 | 0% | 7% | **46.1** |
| `fase0/extraer.py` | 505 | 39 | 0% | 0% | **49.9** |

---

## 6. Tabla Completa de Datos (Top 30 por LOC)

| Archivo | LOC | Total | Funcs | Classes | Max CC | Avg CC | Hints% | Docs% | Health |
|---------|-----|-------|-------|---------|--------|--------|--------|-------|--------|
| fase0/pre_embeddings.py | 1714 | 2165 | 24 | 0 | 45 | 13.1 | 0% | 29% | 20.8 |
| fase6/paso10_verificacion.py | 1254 | 1449 | 16 | 0 | 31 | 12.9 | 100% | 88% | 39.9 |
| fase0/pre_modelo.py | 939 | 1234 | 12 | 0 | 35 | 15.1 | 0% | 0% | 41.2 |
| fase2/train_expert1_ddp.py | 922 | 1163 | 14 | 2 | 49 | 9.6 | 100% | 86% | 61.6 |
| fase1/fase1_pipeline.py | 909 | 1128 | 12 | 0 | 41 | 9.8 | 0% | 92% | 51.8 |
| fase2/train_expert2_ddp.py | 894 | 1108 | 10 | 1 | 26 | 9.2 | 100% | 90% | 62.1 |
| fase6/dashboard_figures.py | 847 | 1086 | 10 | 0 | 18 | 7.5 | 100% | 100% | 87.1 |
| fase6/paso11_webapp.py | 818 | 993 | 10 | 0 | 26 | 7.2 | 30% | 80% | 59.6 |
| fase2/train_expert3_ddp.py | 801 | 1046 | 12 | 1 | 55 | 8.6 | 83% | 75% | 64.0 |
| fase3/train_expert5_ddp.py | 789 | 1022 | 16 | 2 | 56 | 6.3 | 100% | 75% | 64.2 |
| fase0/fase0_pipeline.py | 765 | 993 | 14 | 0 | 29 | 8.4 | 0% | 7% | 46.1 |
| fase6/paso12_dashboard.py | 744 | 921 | 25 | 1 | 22 | 4.5 | 92% | 92% | 81.1 |
| fase2/train_expert4_ddp.py | 737 | 934 | 12 | 1 | 52 | 9.1 | 83% | 75% | 65.3 |
| datasets/luna.py | 709 | 860 | 14 | 3 | 34 | 7.4 | 57% | 64% | 65.8 |
| fase2/train_expert1.py | 658 | 826 | 10 | 1 | 25 | 6.7 | 100% | 90% | 76.8 |
| fase2/train_expert2.py | 654 | 806 | 8 | 1 | 14 | 6.1 | 88% | 88% | 96.9 |
| fase3/train_expert5.py | 653 | 851 | 16 | 2 | 27 | 3.8 | 100% | 75% | 66.9 |
| fase2/train_expert_oa_ddp.py | 622 | 795 | 9 | 1 | 45 | 8.8 | 100% | 89% | 67.6 |
| datasets/pancreas.py | 620 | 733 | 15 | 4 | 23 | 8.1 | 40% | 47% | 78.9 |
| datasets/isic.py | 605 | 742 | 18 | 4 | 21 | 4.6 | 61% | 50% | 85.9 |
| fase0/descargar.py | 513 | 631 | 14 | 0 | 16 | 7.4 | 0% | 0% | 77.7 |
| fase3/models/expert6_resunet.py | 512 | 633 | 19 | 9 | 4 | 1.5 | 100% | 37% | 97.1 |
| fase0/extraer.py | 505 | 624 | 16 | 1 | 39 | 7.2 | 0% | 0% | 49.9 |
| fase2/train_expert4.py | 485 | 621 | 10 | 1 | 23 | 5.3 | 70% | 70% | 84.0 |
| fase2/train_expert3.py | 480 | 623 | 10 | 1 | 21 | 5.2 | 70% | 70% | 88.0 |
| fase1/fase1_train_pipeline.py | 450 | 568 | 8 | 0 | 13 | 4.8 | 25% | 50% | 95.0 |
| fase0/pre_isic.py | 439 | 554 | 7 | 0 | 15 | 6.3 | 100% | 100% | 100.0 |
| fase2/dataloader_expert1.py | 424 | 536 | 8 | 0 | 6 | 3.0 | 100% | 100% | 100.0 |
| fase2/train_expert_oa.py | 422 | 540 | 7 | 1 | 20 | 5.9 | 86% | 86% | 90.0 |
| fase3/train_cae.py | 393 | 499 | 7 | 1 | 18 | 5.3 | 86% | 86% | 94.0 |

---

## 7. Conclusiones: Mayores Riesgos de Complejidad

### 7.1 Hallazgos Principales

1. **La función `train()` es el hotspot universal de complejidad.** En los 5 scripts DDP, `train()` concentra CC de 45–56. Cada una de estas funciones maneja setup, training loop, validation, early stopping, checkpointing y cleanup en un solo bloque monolítico.

2. **`fase0/` es la zona más desatendida del codebase.** Los 6 archivos de preprocesamiento acumulan 4,436 LOC con 0% type hints y docstrings parciales o nulas. `pre_embeddings.py` (1,714 LOC, CC=45, health=20.8) es el archivo de mayor riesgo de todo el proyecto.

3. **Correlación LOC↔CC fuerte (r=0.791):** Los archivos no crecen por tener más funciones simples, sino por tener funciones más complejas. Esto indica que el crecimiento del código es orgánico y no modular.

4. **La mediana de health score (90.0) es buena,** pero la distribución tiene cola izquierda con 5 archivos bajo 50. El proyecto está saludable en general, con puntos de dolor concentrados.

5. **Type hints tienen distribución bimodal:** O un archivo tiene 100% o tiene 0%. No hay adopción gradual: los archivos nuevos/refactorizados los tienen, los legacy no.

### 7.2 Prioridades de Refactorización (por impacto)

| Prioridad | Acción | Archivos | Impacto estimado |
|-----------|--------|----------|------------------|
| **P0** | Dividir `pre_embeddings.py` en módulos por dataset | 1 archivo → 4-5 | Health 20.8 → ~80+ |
| **P1** | Extraer subfunciones de `train()` en scripts DDP | 5 archivos | CC 45-56 → <20 |
| **P2** | Agregar type hints a `fase0/` | 6 archivos, ~4400 LOC | Cobertura global 57% → ~75% |
| **P3** | Agregar docstrings a archivos >100 LOC sin docs | 5 archivos | Cobertura global 63% → ~75% |
| **P4** | Refactorizar `pre_modelo.py` (CC=35, 0% hints, 0% docs) | 1 archivo | Health 41.2 → ~85+ |

### 7.3 Resumen de Riesgo por Módulo

| Módulo | Archivos | LOC Total | Health Promedio | Riesgo |
|--------|----------|-----------|-----------------|--------|
| `fase0/` | 9 | 5,418 | 57.0 | **ALTO** |
| `fase2/` (training DDP) | 5 | 3,976 | 63.4 | **MEDIO-ALTO** |
| `fase6/` | 9 | 4,855 | 74.3 | MEDIO |
| `fase1/` | 12 | 3,481 | 87.7 | BAJO |
| `fase2/` (otros) | 20 | 4,126 | 93.2 | BAJO |
| `datasets/` | 5 | 2,288 | 82.3 | BAJO |
| `fase3/` | 6 | 2,835 | 82.7 | BAJO |
| `fase5/` | 5 | 924 | 95.2 | MUY BAJO |
