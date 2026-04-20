---
titulo: "MoE — estado operativo y siguientes pasos"
fecha: 2026-04-18
owner: mitgar14
estado: vivo (actualizar tras cada avance)
---

# 1. Estado operativo al 2026-04-18

## 1.1) Lo ejecutable hoy

- `MoESystem` (`moe/moe_system.py`) arranca en CPU/GPU y responde `forward(x_raw)` end-to-end:
  AdaptivePreprocessor → ViT-Tiny (CLS 192) → LinearRouter → Mahalanobis → hard gating
  argmax → ExpertWrapper nativo por experto.
- Los 5 expertos cargan con `strict_load=True` (validado con `scripts/moe/test_moe_system.py`):

| Label | Arch real (reconstruida de state_dict) | Fuente | Val del checkpoint |
|---|---|---|---|
| 0 CXR14 v21 | `CXRExpertSingleHead` (ConvNeXt-V2 Base `features_only` + `LSEPool(r=10)` + `Linear(1024, 6)`) | HF `mitgar14/moe-medical-experts/exp1v21/` | AUROC=0,8064 · F1_opt=0,4979 |
| 1 ISIC | `torchvision.efficientnet_b3` + `classifier(Dropout, Linear(1536, 8))` | `models/expert3_isic_best.pth.zip` | F1_macro=0,7414 |
| 2 Osteo | `torchvision.efficientnet_b0` + `classifier(Dropout, Linear(1280, 5))` | `models/expert2_osteo_best.pth.zip` | F1_macro=0,7987 |
| 3 LUNA | DenseNet-3D custom (`block=(4,8,16,12)`, `growth=32`, `bn_size=4`, `compression=0.5`, `init=64`) | `models/LUNA-LIDCIDRI_best.pt.zip` | F1_macro=0,9438 · AUC=0,9911 |
| 4 Pancreas | `torchvision.r3d_18` + `fc(512, 2)`, input 3ch replicado | `models/exp5_best.pth.zip` | best_F1=0,6558 |

## 1.2) Artefactos listos

| Artefacto | Ruta | Funcion |
|---|---|---|
| Ablation v2 (4 routers) | `outputs/ablation_v2/results.json` | Linear gana: RA=1,0000, p50=3,8 μs |
| Fase 2 Router | `outputs/moe_phase2/router_linear_best.pt` + `metrics.json` + `histories.json` | Natural RA=0,99995 · Balanceada RA=1,0000 |
| Benchmark OOD brain MRI | `outputs/moe_phase2/ood_benchmark_brats.{json,png}` | AUROC=0,9997 · TPR=99,49% @ FPR=0,46% |
| Eval scaffold | `scripts/moe/run_moe_eval.py` + `outputs/moe_exec/eval_report.json` | Listo para val sets reales del companero |
| Codigo MoE | `moe/` (preprocessing, routing, experts, moe_system.py) | API publica `MoESystem`, `ExpertWrapper`, `EXPERT_SPECS` |

## 1.3) Rubrica del PDF (p. 17) — progreso

| Item | Peso | Estado |
|---|---|---|
| Routing Accuracy > 0,80 mejor router | 10% | **Cumple** (Linear 1,0000 balanceada) |
| OOD AUROC > 0,80 | 20% | **Cumple** (0,9997 brain MRI) |
| F1 Macro 2D full > 0,72 / acceptable > 0,58 | 20% | Pendiente val real (depende del companero) |
| F1 3D full > 0,65 | 20% | Pancreas 0,656 en limite · LUNA 0,944 cumple |
| Ablation study 4 routers | 15% | **Cumple** (Linear/GMM/NB/kNN documentados) |
| Dashboard 8 items §11.1 | 15% | No iniciado (diferido por decision) |
| Reporte tecnico 5-7 pag | 15% | No iniciado (diferido por decision) |
| Repositorio README + seeds + cluster | 5% | Codigo listo, README MoE pendiente |
| **Penalizacion load_balance > 1,30** | -40% | Natural 44,50 (estructural) · Balanceada 1,00 — reportar ambas |
| **Penalizacion metadatos externos** | -20% | NO aplica (router solo ve la imagen) |

# 2. Siguientes pasos (priorizados)

## 2.1) No dependen de terceros — procedibles ya

### A. Demo end-to-end con 1 imagen real publica por experto (prioridad alta)
- Descargar 1 muestra publica por dataset: CXR14 Kaggle sample · ISIC 2019 sample · Osteo KL · LUNA NIfTI de HF · Pancreas sample.
- Ejecutar `MoESystem(sample)` y verificar: (a) rutea al experto correcto, (b) no marca OOD, (c) prediccion razonable con la etiqueta esperada.
- Objetivo: detectar bugs sutiles (escalas HU, normalizacion, orientacion NIfTI) que `torch.randn` no expone.
- Entregable: `scripts/moe/smoke_test_real_samples.py` + `outputs/moe_exec/smoke_report.json`.

### B. Aplicar `tau_logit_adjustment=0.5` en inferencia de exp1 (prioridad alta)
- El training de v21 aplica logit adjustment (`tau=0.5`) con prior por clase antes del sigmoid. El `ExpertWrapper` actual NO lo aplica en inferencia, por lo que las probs sigmoid no son directamente comparables con los thresholds calibrados por clase (`[0.48, 0.54, 0.58, 0.63, 0.66, 0.69]`).
- Accion: cargar `prevalences` del summary v21 o del set de training, calcular `logit_offset = tau * log(prevalence)` y aplicarlo en `ExpertWrapper.forward` cuando `extras.get("tau_logit_adjustment")` esta presente.
- Entregable: ajuste en `moe/experts/wrappers.py` + test que verifique consistencia de probs con las del training.

### C. Actualizar `agent-docs/moe-backbone/experts_manifest.yaml` a v3 (prioridad media)
- Reflejar estado real al 2026-04-18: `loaded=True` para los 5 expertos con sus rutas definitivas.
- Correcciones de specs: exp3 Osteo 5 clases KL (no 3), exp5 Pancreas 3 canales replicados, exp1 v21 oficial con thresholds por clase, exp4 LUNA arq reconstruida.
- Entregable: manifest v3 commiteado.

### D. Profile + stress test (prioridad media)
- Medir latencia end-to-end CPU vs GPU para batch sizes 1/4/16.
- Memoria pico por stage (preproc / ViT / router / experto).
- Throughput agregado bajo carga mixta 2D+3D.
- Entregable: `scripts/moe/bench_latency.py` + `outputs/moe_exec/latency_report.json` con tabla por stage.

### E. `moe/__init__.py` con exports publicos (prioridad baja, cosmetica)
- Facilitar `from moe import MoESystem, ExpertWrapper, EXPERT_SPECS, AdaptivePreprocessor, LinearRouter`.
- Util para el companero que enchufara el val.

### F. README del paquete MoE (prioridad media)
- `moe/README.md` con: vista arquitectura · como correr inferencia · como enchufar val sets · rubrica cumplida.
- Breve y autocontenido.

## 2.2) Dependen de otras personas

### G. Val real por experto → companero
- Contrato explicito en `scripts/moe/run_moe_eval.py`:
  - `--val-expN <dir>` con estructura `<dir>/images/` + `<dir>/labels.csv`.
  - Encoding de `labels.csv` documentado por task (multilabel / multiclass / binary).
- Cuando llegue, ejecutar `python scripts/moe/run_moe_eval.py --val-exp0 ... --val-exp4 ...` y anexar output al reporte.

### H. Dashboard 8 items §11.1
- Diferido hasta nueva definicion del stack y layout. Especificacion existente en
  `investigaciones/2026-04-19-moe-profundizacion.md` seccion 11.
- Cuando se active, partir de los artefactos actuales (`outputs/ablation_v2/results.json`,
  `outputs/moe_phase2/*`) para items 20 y 22.

### I. Reporte tecnico 5-7 paginas
- Diferido hasta definicion final de estructura y alcance.
- Datos ya listos para rellenar §3 (ablation), §4 (resultados F2), §4 OOD, §5 load balance,
  §6 limitaciones.

# 3. Decisiones abiertas para el usuario

1. Priorizar A+B+C (endurece pipeline, corrige gap en exp1) vs priorizar la espera del val
   del companero.
2. Si se ejecuta A (demo con imagen real), confirmar fuentes preferidas (Kaggle samples
   publicos vs HF Hub samples del proyecto).
3. Decidir cuando retomar dashboard e informe, y bajo que especificacion ajustada.

# 4. Invariantes a preservar en cualquier siguiente paso

- **Determinismo**: seeds fijos (42, 137, 256) y `set_seed` previo a eval.
- **No metadata externa al modelo**: penalizacion rubrica -20%.
- **Load balance honesto**: reportar natural + balanceada (nunca inflar val para cumplir 1,30).
- **strict_load=True** para todos los expertos: si falla, reconstruir arq, nunca aceptar missing/unexpected keys silenciosamente.
- **OOD score combinado** con stats del set ID (z-score 0,5·H + 0,5·mahal), no normalizar por batch.
