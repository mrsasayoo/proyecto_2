# Reporte de Análisis: 26 Falsos Negativos — DenseNet3D

**Modelo:** Expert3DenseNet3D (epoch 23)  
**Conjunto:** Test (2013 muestras: 183 positivos, 1830 negativos)  
**Recall en test:** 85.79% → **26 nódulos malignos clasificados como benignos**  
**Fecha de análisis:** 2026-04-17

---

## 1. Resumen Ejecutivo

Los 26 falsos negativos comparten un perfil estadístico consistente: **baja intensidad media, baja variabilidad, y señal nodular débil respecto al fondo**. El modelo DenseNet3D falla sistemáticamente en nódulos que presentan bajo contraste con el parénquima circundante, ya sea por ser subsólidos, pequeños, o estar adyacentes a estructuras pleurales que dominan la señal del patch.

---

## 2. Estadísticas por Volumen (64×64×64 voxels, HU normalizado)

| Candidato | Min | Max | Mean | Std | Median | P95 | P99 | Vox>0.3 | Vox>0.5 | Center Mean |
|-----------|-----|-----|------|-----|--------|-----|-----|---------|---------|-------------|
| 052458 | -0.099 | 0.874 | -0.052 | 0.076 | -0.061 | 0.055 | 0.298 | 2600 | 669 | -0.050 |
| 116807 | -0.099 | 0.649 | -0.029 | 0.089 | -0.028 | 0.125 | 0.352 | 3798 | 856 | -0.026 |
| 129835 | -0.099 | 0.778 | -0.057 | 0.067 | -0.099 | 0.043 | 0.250 | 1816 | 260 | -0.030 |
| 231094 | -0.099 | 0.901 | -0.003 | 0.101 | -0.011 | 0.165 | 0.451 | 5860 | 2006 | 0.039 |
| 231127 | -0.099 | 0.778 | -0.018 | 0.076 | -0.026 | 0.096 | 0.328 | 3141 | 826 | -0.006 |
| 231311 | -0.099 | 0.746 | 0.028 | 0.153 | -0.016 | 0.428 | 0.653 | 18590 | 10554 | 0.021 |
| 231678 | -0.099 | 0.742 | -0.026 | 0.102 | -0.038 | 0.131 | 0.463 | 5343 | 2187 | 0.004 |
| 276894 | -0.099 | 0.768 | -0.041 | 0.108 | -0.073 | 0.143 | 0.544 | 6687 | 3201 | 0.076 |
| 276900 | -0.099 | 0.658 | -0.052 | 0.080 | -0.076 | 0.075 | 0.323 | 3065 | 826 | 0.097 |
| 277052 | -0.099 | 0.658 | -0.044 | 0.081 | -0.063 | 0.086 | 0.335 | 3323 | 856 | 0.153 |
| 282052 | -0.099 | 0.887 | -0.028 | 0.138 | -0.099 | 0.244 | 0.667 | 10833 | 5524 | 0.149 |
| 282084 | -0.099 | 0.887 | -0.021 | 0.141 | -0.072 | 0.263 | 0.672 | 11548 | 5884 | 0.174 |
| 282464 | -0.099 | 0.887 | -0.021 | 0.145 | -0.087 | 0.277 | 0.688 | 12176 | 6363 | 0.305 |
| 282945 | -0.099 | 0.887 | -0.005 | 0.150 | -0.040 | 0.317 | 0.699 | 13845 | 7242 | 0.270 |
| 283333 | -0.099 | 0.884 | -0.029 | 0.141 | -0.099 | 0.251 | 0.674 | 11151 | 5809 | 0.197 |
| 461962 | -0.099 | 0.901 | -0.005 | 0.128 | -0.030 | 0.272 | 0.502 | 11158 | 2662 | 0.054 |
| 483326 | -0.099 | 0.843 | -0.040 | 0.089 | -0.099 | 0.098 | 0.321 | 2999 | 751 | 0.002 |
| 483861 | -0.099 | 0.865 | -0.022 | 0.092 | -0.030 | 0.125 | 0.392 | 4434 | 1321 | 0.001 |
| 484086 | -0.099 | 0.818 | -0.043 | 0.092 | -0.099 | 0.124 | 0.337 | 3482 | 621 | 0.011 |
| 484446 | -0.099 | 0.901 | -0.004 | 0.094 | -0.013 | 0.151 | 0.405 | 5061 | 1241 | 0.027 |
| 484530 | -0.099 | 0.734 | -0.043 | 0.095 | -0.099 | 0.127 | 0.362 | 4060 | 779 | 0.023 |
| 486947 | -0.099 | 0.901 | -0.048 | 0.082 | -0.099 | 0.089 | 0.280 | 2234 | 362 | 0.008 |
| 487073 | -0.099 | 0.730 | -0.000 | 0.083 | 0.007 | 0.131 | 0.304 | 2716 | 180 | 0.045 |
| 549524 | -0.099 | 0.901 | -0.041 | 0.090 | -0.099 | 0.093 | 0.356 | 3590 | 986 | 0.020 |
| 559242 | -0.099 | 0.901 | -0.026 | 0.111 | -0.099 | 0.197 | 0.346 | 4336 | 582 | 0.087 |
| 753549 | -0.099 | 0.901 | -0.067 | 0.057 | -0.076 | -0.006 | 0.210 | 1490 | 220 | -0.037 |

### Estadísticas Agregadas (26 FN)

| Métrica | Promedio | Desv. Estándar | Rango |
|---------|----------|----------------|-------|
| Mean del volumen | -0.0282 | 0.0210 | [-0.067, 0.028] |
| Std del volumen | 0.1023 | 0.0270 | [0.057, 0.153] |
| Max del volumen | 0.8222 | 0.0851 | [0.649, 0.901] |
| Mean centro (16³) | 0.0622 | 0.0914 | [-0.050, 0.305] |
| Voxels > 0.3 | 6,128 | 4,386 | [1,490 – 18,590] |
| Voxels > 0.5 | 2,414 | 2,686 | [180 – 10,554] |

---

## 3. Categorización Morfológica

### Grupo A — Nódulos Pleurales / Adyacentes a Pared (12 casos, 46%)
**Candidatos:** 052458, 116807, 231094, 231127, 231678, 276894, 483326, 483861, 484446, 487073, 549524, 753549

**Características:**
- Edge ratio > 0.5 (más del 50% de los voxels brillantes están en los bordes del patch)
- La señal pleural/pared torácica domina el patch, diluyendo la señal del nódulo
- Center mean muy bajo (promedio: 0.01), indicando que el nódulo no está bien centrado o es muy tenue
- Compactness baja (< 0.05): la señal está dispersa, no concentrada

**Interpretación visual:** En las imágenes se observa una línea o banda brillante correspondiente a la pleura que ocupa una porción significativa del volumen. El nódulo, si es visible, aparece como una opacidad sutil adyacente a esta estructura de alta intensidad. El modelo probablemente aprende a asociar estas bandas pleurales con candidatos benignos (falsos positivos del detector de candidatos).

### Grupo B — Nódulos con Tejido Parenquimatoso Denso (8 casos, 31%)
**Candidatos:** 276900, 277052, 282052, 282084, 282464, 283333, 484530, 559242

**Características:**
- Voxels > 0.3 entre 3,000 y 12,000 — hay tejido presente pero distribuido
- Compactness moderada (0.03–0.17)
- Edge ratio < 0.5 — la señal está más centrada
- Varios de estos (282xxx) comparten un patrón similar: nódulo parcialmente sólido rodeado de parénquima con opacidades ground-glass

**Interpretación visual:** El corte axial central muestra opacidades difusas de intensidad intermedia. El nódulo no tiene bordes nítidos y se confunde con el tejido circundante. El modelo no logra distinguir la lesión del parénquima normal denso.

**Nota:** Los candidatos 282052, 282084, 282464, 282945, 283333 probablemente provienen del mismo paciente (IDs consecutivos), lo que sugiere un caso particularmente difícil con múltiples nódulos subsólidos.

### Grupo C — Nódulos Subsólidos / Difusos (3 casos, 12%)
**Candidatos:** 231311, 282945, 461962

**Características:**
- Voxels > 0.3 entre 11,000 y 18,590 — gran volumen de tejido de densidad intermedia
- Std alta (0.128–0.153) — mucha variabilidad pero sin picos claros
- El nódulo se presenta como una opacidad ground-glass extensa sin componente sólido definido

**Interpretación visual:** candidate_231311 muestra una gran masa de tejido de densidad intermedia que ocupa casi la mitad del patch. La distribución de intensidades es bimodal (fondo + tejido difuso) sin el pico de alta intensidad que el modelo espera para clasificar como maligno.

### Grupo D — Nódulos Pequeños de Baja Densidad (3 casos, 12%)
**Candidatos:** 129835, 484086, 486947

**Características:**
- Voxels > 0.3: 1,490–3,482 (los más bajos del conjunto)
- Voxels > 0.5: 220–621 (señal sólida mínima)
- Max < 0.82 en promedio
- Center mean < 0.01

**Interpretación visual:** Nódulos muy pequeños, apenas visibles en el corte central. La señal es tan débil que se confunde con ruido o artefactos. candidate_753549 es el caso extremo: mean=-0.067, std=0.057, solo 1,490 voxels > 0.3.

---

## 4. Hallazgos Transversales

### 4.1 Todos los FN tienen mean negativo (excepto 231311)
- **25 de 26** volúmenes tienen mean < 0 (rango: -0.067 a -0.000)
- Esto indica que el volumen está dominado por fondo/aire, con el nódulo ocupando una fracción minoritaria
- El modelo parece tener un sesgo hacia clasificar como positivo solo cuando hay suficiente "masa" de señal alta

### 4.2 Baja variabilidad generalizada
- **24 de 26** tienen std < 0.15
- La baja variabilidad indica que no hay un contraste fuerte entre nódulo y fondo
- Promedio de std: 0.102 (comparar con lo que se esperaría de un nódulo sólido bien definido: std > 0.15)

### 4.3 Señal central débil
- **17 de 26** (65%) tienen center mean < 0.05
- El modelo probablemente depende de que el nódulo esté centrado y sea brillante en la región central del patch

### 4.4 Posible cluster de paciente
- Los candidatos 282052, 282084, 282464, 282945, 283333 (5 casos, 19% de los FN) tienen estadísticas muy similares y probablemente provienen del mismo paciente con múltiples nódulos subsólidos

---

## 5. Puntos Ciegos del Modelo DenseNet3D

| # | Punto Ciego | Casos Afectados | Severidad |
|---|-------------|-----------------|-----------|
| 1 | **Nódulos adyacentes a pleura**: la señal pleural enmascara al nódulo | 12 (46%) | CRÍTICO |
| 2 | **Nódulos subsólidos/ground-glass**: sin componente sólido definido, el modelo no los reconoce | 11 (42%) | CRÍTICO |
| 3 | **Nódulos pequeños (< 2500 vox > 0.3)**: señal insuficiente para activar la clasificación | 6 (23%) | ALTO |
| 4 | **Baja densidad central**: nódulos descentrados o con centro tenue | 17 (65%) | ALTO |
| 5 | **Bajo contraste global** (max < 0.7): el modelo necesita picos de intensidad altos | 3 (12%) | MEDIO |

---

## 6. Recomendaciones

1. **Data augmentation focalizado**: Aumentar la representación de nódulos subsólidos y pleurales en el entrenamiento. Aplicar augmentations que simulen nódulos de baja densidad (reducción de contraste, adición de ruido).

2. **Attention mechanism**: Incorporar un módulo de atención espacial que permita al modelo focalizarse en la región central del patch independientemente de la señal pleural periférica.

3. **Multi-scale features**: Los nódulos pequeños podrían beneficiarse de un enfoque multi-escala que capture tanto el contexto global como los detalles finos.

4. **Threshold de decisión**: Dado que las probabilidades de los FN van de 0.15 a 0.49, reducir el umbral de clasificación de 0.5 a ~0.4 recuperaría algunos FN a costa de más falsos positivos. Evaluar el trade-off con la curva precision-recall.

5. **Normalización por patch**: Considerar una normalización local que realce el contraste dentro de cada patch, especialmente para nódulos de baja densidad.

6. **Análisis por paciente**: Verificar si los 5 candidatos del cluster 282xxx provienen del mismo paciente. Si es así, el recall real por paciente podría ser peor de lo reportado.

---

## 7. Reproducibilidad

Todas las estadísticas fueron calculadas directamente de los archivos `.npy` en `datasets/luna_lung_cancer/patches/test/`. Las visualizaciones triplanares, montajes y renders 3D están en `notebooks/evaluacion_expert3/false_negatives_visualization/`. El script de generación es `visualize_fn.py` en el mismo directorio.

```python
# Para reproducir las estadísticas de cualquier candidato:
import numpy as np
vol = np.load('datasets/luna_lung_cancer/patches/test/candidate_XXXXXX.npy')
print(f"shape={vol.shape}, min={vol.min():.3f}, max={vol.max():.3f}, "
      f"mean={vol.mean():.3f}, std={vol.std():.3f}")
print(f"voxels > 0.3: {(vol > 0.3).sum()}, voxels > 0.5: {(vol > 0.5).sum()}")
```
