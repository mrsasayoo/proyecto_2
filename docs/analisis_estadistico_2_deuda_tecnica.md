# Análisis Estadístico 2 — Antipatrones, Deuda Técnica y Oportunidades de Mejora

**Fecha:** 2026-04-19  
**Alcance:** `scripts/scripts_amigos/train_exp1v21.py` (1502 LOC) vs `scripts/scripts_amigos/moe/` (1356 LOC, 13 archivos)  
**Fuentes:** Análisis de código AST, CODE_REVIEW.md, inspección directa

---

## 1. Inventario de Antipatrones

| # | Antipatrón | Severidad | Archivos | LOC Afectadas | Esfuerzo (h) |
|---|-----------|-----------|----------|---------------|---------------|
| 1 | **God File** — Todo en un solo archivo (10 clases, 52 funciones, 1502 LOC) | Crítico | `train_exp1v21.py` | 1502 | 16 |
| 2 | **Missing Type Hints** — 50/52 funciones sin return type hint | Alto | `train_exp1v21.py` | ~600 | 4 |
| 3 | **Missing Docstrings** — 50/52 funciones sin docstring | Alto | `train_exp1v21.py` | ~600 | 3 |
| 4 | **Magic Numbers** — ~52 literales numéricos sin nombre (learning rates, thresholds, tamaños) | Alto | `train_exp1v21.py` | ~80 | 3 |
| 5 | **Duplicate Imports** — `os`, `sys`, `time` importados múltiples veces | Bajo | `train_exp1v21.py` | 6 | 0.25 |
| 6 | **Runtime pip install** — Instala dependencias condicionalmente en import-time | Medio | `train_exp1v21.py` | 8 | 1 |
| 7 | **Zero Test Coverage** — Sin tests unitarios para ninguno de los dos módulos | Crítico | ambos | 0 | 24 |
| 8 | **Missing Docstrings (moe/)** — 48/59 funciones sin docstring, 14/18 clases sin docstring | Medio | `moe/` | ~400 | 3 |
| 9 | **No `__all__` exports** — Los `__init__.py` vacíos (routing, experts, preprocessing) | Bajo | `moe/` | 3 archivos | 0.5 |
| 10 | **Tight Coupling** — Script monolítico mezcla data loading, model, training, eval, upload | Alto | `train_exp1v21.py` | 1502 | 12 |

**Total estimado de refactoring:** ~66.75 horas

---

## 2. Índice de Deuda Técnica

### Fórmula

```
TDI = LOC_sin_docstrings + LOC_sin_type_hints + LOC_duplicado
TDI_ratio = TDI / (LOC_total × 3)  →  0.0 (sin deuda) a 1.0 (deuda máxima)
```

### Por archivo — `train_exp1v21.py`

| Métrica | Valor |
|---------|-------|
| LOC total | 1502 |
| LOC código (sin blancos/comentarios) | 1185 |
| Funciones totales | 52 |
| Funciones sin docstring | 50 (96%) |
| Funciones sin return type hint | 50 (96%) |
| LOC en funciones sin docstring | ~1100 |
| LOC en funciones sin type hints | ~1100 |
| LOC duplicado (imports) | ~6 |
| **TDI** | **2206** |
| **TDI ratio** | **0.49** |

### Por archivo — `moe/` (paquete completo)

| Métrica | Valor |
|---------|-------|
| LOC total | 1356 |
| LOC código (sin vacíos/\_\_init\_\_.py) | ~1050 |
| Funciones totales | 59 |
| Funciones sin docstring | 48 (81%) |
| Funciones sin return type hint | 2 (3%) |
| LOC en funciones sin docstring | ~700 |
| LOC en funciones sin type hints | ~30 |
| LOC duplicado | ~0 |
| **TDI** | **730** |
| **TDI ratio** | **0.18** |

### Detalle por archivo `moe/`

| Archivo | LOC | Funcs | Sin Docstring | Sin Hints | TDI |
|---------|-----|-------|---------------|-----------|-----|
| `moe_system.py` | 245 | 8 | 5 | 0 | 120 |
| `experts/loaders.py` | 271 | 9 | 2 | 1 | 50 |
| `experts/wrappers.py` | 269 | 8 | 7 | 1 | 180 |
| `routing/linear_router.py` | 82 | 8 | 8 | 0 | 65 |
| `routing/aux_loss.py` | 74 | 5 | 5 | 0 | 60 |
| `preprocessing/adaptive.py` | 119 | 7 | 7 | 0 | 95 |
| `experts/archs/densenet3d_luna.py` | 172 | 10 | 10 | 0 | 130 |
| `experts/archs/cxr_v21_wrapper.py` | 68 | 4 | 4 | 0 | 55 |

### Comparativa

| Métrica | `train_exp1v21.py` | `moe/` |
|---------|-------------------|--------|
| TDI ratio | **0.49** (Alto) | **0.18** (Bajo) |
| Tiempo estimado resolver | ~26 h | ~6 h |

---

## 3. Oportunidades de Mejora — Quick Wins

| # | Tarea | LOC/Items | Dificultad | Horas | Reducción TDI |
|---|-------|-----------|------------|-------|---------------|
| 1 | Agregar 50 docstrings a `train_exp1v21.py` (1 línea promedio por función) | 50 funcs | Fácil | 2.5 | -550 |
| 2 | Agregar 50 return type hints a `train_exp1v21.py` | 50 funcs | Fácil | 2.0 | -550 |
| 3 | Agregar 48 docstrings a `moe/` | 48 funcs | Fácil | 2.0 | -500 |
| 4 | Eliminar imports duplicados (`os`, `sys`, `time`) | 3 líneas | Trivial | 0.1 | -6 |
| 5 | Extraer magic numbers a constantes con nombre | ~52 valores | Medio | 3.0 | -80 |
| 6 | Agregar `__all__` a `__init__.py` vacíos de `moe/` | 3 archivos | Trivial | 0.25 | -3 |
| 7 | Mover runtime pip install a `requirements.txt` + docs | 8 LOC | Fácil | 0.5 | -8 |

**Total Quick Wins: ~10.35 horas → reduce TDI en ~1697 puntos (58% de la deuda total)**

---

## 4. Comparativa: `moe/` (90/100) vs `train_exp1v21.py` (35/100)

| Dimensión | `train_exp1v21.py` | `moe/` | Ganador |
|-----------|-------------------|--------|---------|
| **Modularización** | 1 archivo, 10 clases, 52 funcs | 13 archivos, 18 clases, 59 funcs | moe/ |
| **Separación de responsabilidades** | Data + Model + Train + Eval + Upload mezclados | routing/ experts/ preprocessing/ separados | moe/ |
| **Type hints (return)** | 4% (2/52) | 97% (57/59) | moe/ |
| **Docstrings (funciones)** | 4% (2/52) | 19% (11/59) | moe/ (parcial) |
| **Docstrings (clases)** | 50% (5/10) | 22% (4/18) | Empate bajo |
| **Tests** | 0% | 0% | Ninguno |
| **Magic numbers** | ~52 | ~5 | moe/ |
| **Imports duplicados** | 3 | 0 | moe/ |
| **LOC/archivo promedio** | 1502 | 104 | moe/ |
| **Reusabilidad** | Baja (script monolítico) | Alta (paquete importable) | moe/ |
| **Acoplamiento** | Alto (todo interdependiente) | Bajo (interfaces claras entre módulos) | moe/ |

### ¿Por qué `moe/` es mejor?

1. **Arquitectura de paquete**: Separación clara en subdirectorios funcionales
2. **Type hints casi completos** (97%): Facilita IDE, refactoring, detección de bugs
3. **Archivos pequeños**: Promedio 104 LOC/archivo vs 1502 LOC monolítico
4. **Bajo acoplamiento**: Cada módulo tiene una responsabilidad definida

### ¿Qué le falta a `moe/`?

1. Docstrings (especialmente en funciones de routing y preprocessing)
2. Tests unitarios
3. `__init__.py` con exports explícitos

---

## 5. Recomendaciones Priorizadas

### Rank 1 — Crítico (bloquea producción, ≥1 día)

| # | Recomendación | Esfuerzo | Impacto |
|---|---------------|----------|---------|
| 1.1 | **Escribir tests unitarios** para `moe/` (al menos routing + experts) | 16h | Detectar regresiones, habilitar CI |
| 1.2 | **Descomponer `train_exp1v21.py`** en módulos: `data/`, `models/`, `training/`, `eval/`, `utils/` | 16h | Mantenibilidad, reusabilidad |

### Rank 2 — Alto (degrada mantenibilidad, ≥4 horas)

| # | Recomendación | Esfuerzo | Impacto |
|---|---------------|----------|---------|
| 2.1 | **Agregar type hints** a todas las funciones de `train_exp1v21.py` | 4h | IDE support, detección de bugs |
| 2.2 | **Extraer magic numbers** a un módulo de configuración | 3h | Legibilidad, configurabilidad |
| 2.3 | **Agregar docstrings** a `train_exp1v21.py` (50 funciones) | 2.5h | Onboarding, mantenibilidad |

### Rank 3 — Medio (mejora salud general, ≥1 hora)

| # | Recomendación | Esfuerzo | Impacto |
|---|---------------|----------|---------|
| 3.1 | **Agregar docstrings** a `moe/` (48 funciones) | 2h | Documentación API |
| 3.2 | **Eliminar runtime pip install** — documentar deps en `requirements.txt` | 0.5h | Reproducibilidad |
| 3.3 | **Agregar `__all__`** a `__init__.py` de `moe/` | 0.25h | API pública explícita |

### Rank 4 — Bajo (nice-to-have, <1 hora)

| # | Recomendación | Esfuerzo | Impacto |
|---|---------------|----------|---------|
| 4.1 | **Limpiar imports duplicados** | 0.1h | Limpieza |
| 4.2 | **Renombrar `probs` → `logits`** donde corresponda (ref: CODE_REVIEW) | 0.25h | Claridad |
| 4.3 | **Cambiar `assert` → `ValueError`** en validación de inputs | 0.25h | Seguridad en producción |

### Orden de ejecución sugerido

```
Semana 1: 4.1 → 4.2 → 4.3 → 3.2 → 3.3          (~1.5h, limpieza rápida)
Semana 2: 2.3 → 2.1 → 3.1                         (~8.5h, docstrings + hints)
Semana 3: 2.2                                       (~3h, configuración)
Semana 4-5: 1.2                                     (~16h, descomposición)
Semana 6-7: 1.1                                     (~16h, tests)
                                              Total: ~45h en 7 semanas
```

---

## 6. Gráfica: Cost of Fixing vs Time to Fix

```
Impacto (reducción deuda)
    ▲
    │
100%│                                          ★ 1.2 Descomponer script
    │                                    ★ 1.1 Tests
 80%│
    │
 60%│
    │          ★ 2.1 Type hints
 50%│     ★ 2.3 Docstrings (train)
    │
 40%│
    │
 30%│                    ★ 2.2 Magic numbers
    │  ★ 3.1 Docstrings (moe)
 20%│
    │
 10%│★ 3.2 Eliminar pip install
    │★ 4.1-4.3 Limpieza menor
  0%├──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──► Horas
    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16

    Leyenda:
    ★  = tarea (posición X = horas, posición Y = impacto)
    
    QUICK WINS (abajo-izquierda): 4.1-4.3, 3.2, 3.3
    BEST ROI (centro-izquierda):  2.3, 2.1, 3.1
    HIGH EFFORT (derecha):        1.1, 1.2, 2.2
```

### Interpretación

| Zona | Tareas | Estrategia |
|------|--------|------------|
| **Quick Wins** (bajo esfuerzo, bajo-medio impacto) | 4.1-4.3, 3.2, 3.3 | Hacer inmediatamente |
| **Best ROI** (bajo-medio esfuerzo, alto impacto) | 2.1, 2.3, 3.1 | Priorizar en sprint actual |
| **Strategic** (alto esfuerzo, muy alto impacto) | 1.1, 1.2 | Planificar en roadmap |

---

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Antipatrones encontrados | 10 |
| Deuda técnica total (TDI) | 2936 puntos |
| TDI ratio promedio | 0.34 |
| Quick wins disponibles | 7 tareas (~10h) |
| Reducción TDI con quick wins | 58% |
| Tiempo total para resolver toda la deuda | ~45 horas (7 semanas) |
| Archivo más problemático | `train_exp1v21.py` (TDI=0.49) |
| Archivo más sano | `moe/experts/loaders.py` (TDI≈0.05) |
