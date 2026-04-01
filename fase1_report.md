# Fase 1 — Reporte Final: Extracción de Embeddings

**Fecha:** 2026-04-01  
**Estado:** ✅ COMPLETADA

---

## Resumen Ejecutivo

Se extrajeron embeddings (CLS tokens) para los **4 backbones** del sistema MoE sobre los **5 datasets de dominio** (Expertos 0–4). Todos los artefactos fueron verificados: shapes correctos, cero NaN/Inf, y los 5 expertos presentes en cada split.

---

## Backbones procesados

| Backbone | d_model | Params | Tiempo TRAIN | img/s | Tamaño Z_train |
|---|---|---|---|---|---|
| vit_tiny_patch16_224 | 192 | 5.52M | ~3h 8min | ~11 | 94 MB |
| densenet121_custom | 1024 | 7.98M | ~5h 45min | ~7 | 502 MB |
| swin_tiny_patch4_window7_224 | 768 | 28M | ~7h 15min | ~5 | 376 MB |
| cvt_13 | 384 | 20M | ~7h 9min | ~5 | 188 MB |

> Todos ejecutados en CPU (AMD Athlon 3000G, 4 threads, sin GPU).  
> Pesos: **aleatorios** (sin preentrenamiento), backbones **congelados** (Fase 1 solo extrae features).

---

## Datasets / Splits

| Split | N total | E0 Chest | E1 ISIC | E2 OA | E3 LUNA | E4 Páncreas |
|---|---|---|---|---|---|---|
| train | 128,320 | 88,999 | 18,767 | 3,814 | 16,567 | 173 |
| val   | 15,292  | 11,349 | 2,276  | 480   | 1,143  | 44  |
| test  | 16,397  | 11,772 | 2,214  | 472   | 1,914  | 25  |

> Experto 5 (OOD/CAE) sin dataset en Fase 1 — se inicializa en Fase 2/3.

---

## Artefactos generados

```
embeddings/
├── vit_tiny_patch16_224/       (163 MB)
│   ├── Z_train.npy  (128320×192)
│   ├── Z_val.npy    (15292×192)
│   ├── Z_test.npy   (16397×192)
│   ├── y_train/val/test.npy
│   ├── names_train/val/test.txt
│   ├── backbone_meta.json
│   └── fase1_report.md
├── densenet121_custom/          (671 MB)
│   └── (misma estructura, 128320×1024)
├── swin_tiny_patch4_window7_224/ (515 MB)
│   └── (misma estructura, 128320×768)
└── cvt_13/                      (280 MB)
    └── (misma estructura, 128320×384)
```

**Total en disco:** ~1.63 GB

---

## Validación (verificar_embeddings.py)

```
✅ vit_tiny_patch16_224   — OK  (0 NaN, 0 Inf, 5 expertos presentes)
✅ densenet121_custom      — OK  (0 NaN, 0 Inf, 5 expertos presentes)
✅ swin_tiny_patch4_window7_224 — OK (0 NaN, 0 Inf, 5 expertos presentes)
✅ cvt_13                  — OK  (0 NaN, 0 Inf, 5 expertos presentes)
```

---

## Riesgos y notas

| # | Riesgo | Severidad | Estado |
|---|---|---|---|
| R1 | Balance extremo train (E0:E4 = 514:1) | ⚠️ Media | Documentado — se gestiona con weighted sampler en Fase 2 |
| R2 | Embeddings son random projections (backbone sin preentrenamiento) | ⚠️ Media | Diseño explícito §6.4 arquitectura_moe.md — OK para routing |
| R3 | Ambigüedad "desde cero": ¿frozen-random vs preentrenado médico? | 🔴 Pendiente | Clarificar con profesor antes de Fase 5 |
| R4 | LUNA extraction_report.json desactualizado (dice 14937, real 16567) | ✅ Resuelto | Dataset correcto en disco — report es artefacto obsoleto |

---

## Próxima fase

**Fase 2 — Entrenamiento de Expertos de Dominio (Expertos 0–4)**

Entradas: `embeddings/{backbone}/Z_train.npy`, `y_train.npy`  
Salidas: 5 clasificadores entrenados por backbone  
Herramienta: `sklearn.GridSearchCV(n_jobs=-1)` (sin Ray, CPU local)  
Backbone recomendado para desarrollo inicial: `vit_tiny_patch16_224` (más ligero)
