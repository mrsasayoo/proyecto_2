# Paquete `moe`

Sistema Mixture of Experts médico con 5 expertos heterogéneos, router Linear sobre CLS tokens de ViT-Tiny y detección OOD combinada (entropía + Mahalanobis).

## Vista general

```
 x_raw (2D BCHW uint8 o 3D BCDHW float [0,1])
   |
   v
 AdaptivePreprocessor
   |  2D -> (B, 3, 224, 224)
   |  3D -> (B*16, 3, 224, 224) con 16 slices equidistantes
   v
 ViT-Tiny (frozen, augreg_in21k_ft_in1k) -> CLS (192)
   |  (si 3D: mean-pool sobre 16 slices -> (B, 192))
   v
 LinearRouter -> probs (B, 5)  +  MahalanobisOOD -> d^2
   |
   v
 argmax -> expert_idx (B,)
   |
   v
 Por muestra: vista nativa del experto (resize + normalize)
   v
 ExpertWrapper (con logit_offset opcional para exp1 v21)
   v
 MoEResponse { selected_expert, probs, expert_outputs, is_ood, timing_ms }
```

## Estado actual (2026-04-19)

| # | Dataset | Modalidad | Arquitectura | Task | Clases | F1 val | AUROC |
|---|---|---|---|---|---|---|---|
| 0 | NIH ChestX-ray14 (v21) | 2D xray | ConvNeXt-V2 Base 384 features_only + LSEPool2d(r=10) + Linear(1024, 6) | multilabel | 6 | 0,4979 (F1@opt) | 0,8064 |
| 1 | ISIC 2019 | 2D dermoscopy | efficientnet_b3 (torchvision) + Linear(1536, 8) | multiclass | 8 | 0,7414 | — |
| 2 | Osteoarthritis Knee | 2D xray | efficientnet_b0 (torchvision) + Linear(1280, 5) | multiclass | 5 (KL) | 0,7987 | — |
| 3 | LUNA16 + LIDC-IDRI | 3D CT | DenseNet-3D custom reconstruida (4,8,16,12) | binary | 2 | 0,9438 | 0,9911 |
| 4 | PANORAMA Pancreas | 3D CT | r3d_18 (torchvision.video) + Linear(512, 2) | binary | 2 | 0,6558 | 0,7903 |

Los 5 expertos se cargan con `strict_load=True` sin missing/unexpected keys. Ver `agent-docs/moe-backbone/experts_manifest.yaml` (version 3) para el manifest completo.

## Inferencia end-to-end

```python
import torch
from moe import MoESystem

device = "cuda" if torch.cuda.is_available() else "cpu"
system = MoESystem(device=device)

# 2D: radiografia de torax 1024x1024 uint8
x_cxr = torch.randint(0, 255, (1, 3, 1024, 1024), dtype=torch.uint8).float()
resp = system(x_cxr)

print("experto seleccionado:", int(resp.selected_expert[0]))
print("probs routing:", resp.routing_probs[0].tolist())
print("probs expert:", resp.expert_outputs[0].probs[0].tolist())
print("is_ood:", bool(resp.is_ood[0]))
print("latencia total (ms):", resp.timing_ms["total_ms"])
```

Para 3D (LUNA o Pancreas), pasar tensor `(B, 1, D, H, W)` con valores ya normalizados a `[0, 1]` (equivalente a `post-offline`):

```python
x_3d = torch.rand((1, 1, 96, 96, 96))  # o volumen real normalizado
resp = system(x_3d)
```

## Logit adjustment (exp1 v21)

El training de exp1 v21 aplica `logits - tau * log(prevalence_train)` antes del sigmoid para compensar desbalance de clases (Menon 2020). En inferencia, `ExpertWrapper` aplica el mismo offset via `self.logit_offset` precomputado como tensor `(1, 6)`, de manera que los probs sean consistentes con los thresholds calibrados por clase `[0,48, 0,54, 0,58, 0,63, 0,66, 0,69]`.

Fuente: `exp1v21/oof_val_fold0_meta.json` (prevalence) + `exp1v21_summary.json` (tau=0,5).

## Evaluar contra val sets reales

El script `scripts/moe/run_moe_eval.py` acepta directorios de val por experto con formato:

```
<val_root_L>/
  images/
    000001.png         # .jpg, .nii, .nii.gz
    ...
  labels.csv           # columna `filename` + etiquetas segun task
```

Encoding de `labels.csv`:

- **multilabel** (exp0): `filename,Infiltration,Effusion,Atelectasis,Nodule,Mass,Pneumothorax`
- **multiclass** (exp1, exp2): `filename,label` con label entero
- **binary** (exp3, exp4): `filename,label` con label 0/1

Uso:

```bash
python scripts/moe/run_moe_eval.py \
    --val-exp0 data/val_cxr14 \
    --val-exp1 data/val_isic \
    --val-exp2 data/val_osteo \
    --val-exp3 data/val_luna \
    --val-exp4 data/val_pancreas
```

Output: `outputs/moe_exec/eval_report.json` con F1/AUROC nativos, Routing Accuracy, Load Balance (f_i y max/min), OOD rate, confusion 5x5.

## Rúbrica

| Criterio | Umbral | Estado |
|---|---|---|
| F1 Macro 2D | ≥ 0,72 (full) / ≥ 0,65 (aceptable) | exp1 0,4979 (no cumple), exp2 0,7414 (full), exp3 0,7987 (full) |
| F1 Macro 3D | ≥ 0,65 (full) / ≥ 0,58 (aceptable) | exp4 0,9438 (full, +0,29), exp5 0,6558 (full) |
| Routing Accuracy | ≥ 0,80 | 1,0000 (Linear, cls_tokens_v2 val, ablation v2) |
| Load Balance (max/min) | ≤ 1,30 | 1,00 balanced / 44,5 natural (violación estructural por prevalencia CXR14 56% vs Pancreas 0,9%) |
| OOD AUROC | ≥ 0,80 | 0,9997 (BraTS MRI cerebro, TPR=99,49% @ FPR=0,46%) |

Smoke test end-to-end con muestras sintéticas fisiológicas más HF datasets públicos (CXR14, ISIC): 4/5 expertos routean correctamente. Pancreas requiere volumen real PANORAMA para validación.

## Artefactos del paquete

- `moe/moe_system.py`: orquestador `MoESystem` + dataclass `MoEResponse`
- `moe/preprocessing/adaptive.py`: `AdaptivePreprocessor` con modos `generic` y `domain_aware`
- `moe/routing/linear_router.py`: `LinearRouter` + `MahalanobisOOD`
- `moe/routing/aux_loss.py`: `SwitchTransformerAuxLoss` (solo training)
- `moe/experts/wrappers.py`: `ExpertWrapper`, `ExpertSpec`, `EXPERT_SPECS` (specs de los 5 expertos)
- `moe/experts/loaders.py`: builders por experto que cargan checkpoints reales con `strict_load=True`
- `moe/experts/archs/cxr_v21_wrapper.py`: arquitectura exp1 v21 (features_only + LSEPool)
- `moe/experts/archs/densenet3d_luna.py`: DenseNet-3D custom para exp4 (reconstruida desde shapes)

## Scripts

- `scripts/moe/run_moe_eval.py`: evaluación con val sets reales por experto
- `scripts/moe/smoke_test_real_samples.py`: smoke test con muestras sintéticas + HF datasets
- `scripts/moe/bench_latency.py`: benchmark p50/p95/p99 por stage (preproc, ViT, router, expert)
- `scripts/moe/benchmark_ood_brats.py`: OOD benchmark vs BraTS MRI
- `scripts/moe/train_phase2_router.py`: entrenamiento Fase 2 del router con aux loss Switch

## Latencia (CPU, sin GPU disponible localmente)

| Shape / batch | Preprocess | ViT | Router | Expert | Total p50 |
|---|---|---|---|---|---|
| 2D 1024x1024, b=1 | 2,9 ms | 23,3 ms | 0,4 ms | 0,1 ms | **27 ms** |
| 2D 1024x1024, b=4 | 7,7 ms | 49,9 ms | 0,4 ms | 0,2 ms | **58 ms** |
| 2D 1024x1024, b=16 | 28,2 ms | 138,9 ms | 0,5 ms | 0,5 ms | **170 ms** |
| 3D 96³, b=1 | 3,5 ms | 139,2 ms | 0,4 ms | 58,7 ms | **201 ms** |
| 3D 96³, b=4 | 13,5 ms | 617,9 ms | 0,6 ms | 224,9 ms | **857 ms** |

El ViT sobre 16 slices domina la latencia 3D. Con GPU se espera reducción de 10-30x.

## Pendientes (S10-S12)

- Validación contra val sets reales de los 5 expertos: en curso por compañeras (2026-04-19).
- Dashboard Gradio 5 con 8 items §11.1 del PDF: completado.
- Reporte técnico: en proceso.
