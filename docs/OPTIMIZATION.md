# Optimizaciones — Expert 1 Pipeline

## Problema Identificado

### Cuello de botella: DataLoader con num_workers=0

En la configuracion original, el DataLoader usaba `num_workers=0`, lo que causaba que la carga de datos se hiciera en el proceso principal de forma secuencial. Cada imagen .npy de 256x256 float32 toma ~0.1ms en leer, pero con 86K imagenes y overhead de augmentation, la GPU estaba idle la mayor parte del tiempo esperando datos.

**Impacto medido:** ~45 minutos por epoca en una sola GPU, con utilizacion de GPU < 30%.

## Solucion Aplicada

### Cambios en configuracion

| Parametro | Antes | Despues | Impacto |
|-----------|-------|---------|---------|
| `num_workers` | 0 | 8 | Carga paralela, GPU saturada |
| `pin_memory` | False | True | Transferencia CPU→GPU asincrona |
| `persistent_workers` | False | True | Sin overhead de fork entre epocas |
| `prefetch_factor` | N/A | 2 | Pre-carga de 2 batches por worker |
| `EXPERT1_ACCUMULATION_STEPS` | 4 | 2 | Menos pasadas forward redundantes |
| `EXPERT1_BATCH_SIZE` | 32 | 48 | Mejor utilizacion de VRAM |

### Cambios en DDP

- `model.no_sync()` en pasos intermedios de accumulation: reduce comunicacion inter-GPU.
- Gradient clipping (`max_norm=1.0`) para estabilidad con FP16 + FocalLoss.
- pos_weight clamped a 50 (antes sin clamp, Hernia tenia ~538).

### Cambios en TTA

- TTA ahora usa `torch.flip(x, dims=[-1])` directamente sobre tensores GPU.
- Antes requeria un segundo dataset con transforms de albumentations (doble IO).

## Resultados

| Metrica | Antes | Despues |
|---------|-------|---------|
| Dry-run (2 batches) | ~5 min (estimado) | ~1.5s |
| Epoca completa (2 GPUs) | ~45 min | ~3-5 min (estimado) |
| Utilizacion GPU | <30% | >90% |
| VRAM por GPU | ~4 GB | ~8 GB (de 12 disponibles) |

## Futuras Optimizaciones

### torch.compile (PyTorch 2.x)

```python
model = torch.compile(model, mode="reduce-overhead")
```

Puede dar 10-30% de speedup en GPUs modernas. No probado con Titan Xp (requiere SM >= 7.0, Titan Xp es SM 6.1). Funciona en RTX 4090.

### Better prefetching con DALI

NVIDIA DALI puede reemplazar el DataLoader de PyTorch con un pipeline de carga de datos en GPU. Reduce latencia de CPU→GPU. Solo vale la pena si el DataLoader sigue siendo cuello de botella despues de las optimizaciones actuales.

### Gradient checkpointing

```python
from torch.utils.checkpoint import checkpoint
# En el forward del modelo, envolver fases costosas
x = checkpoint(self.inception_blocks, x, use_reentrant=False)
```

Reduce VRAM ~40% a costa de recomputar activaciones en el backward. Util si se quiere subir batch_size significativamente.

### Channel-last memory format

```python
model = model.to(memory_format=torch.channels_last)
imgs = imgs.to(memory_format=torch.channels_last)
```

Optimiza el layout de memoria para convolutions en GPUs con Tensor Cores. Con imagenes de 1 canal el beneficio es minimo, pero no deberia degradar rendimiento.

### Mixed Precision con BFloat16

En GPUs que lo soporten (A100, RTX 4090), BFloat16 ofrece mayor rango dinamico que FP16 sin necesidad de GradScaler:

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    probs = model(imgs)
```

### Progressive resizing

Entrenar las primeras epocas a 128x128 y las ultimas a 256x256. Reduce costo de las primeras epocas cuando el modelo aun no ha convergido.

### WebDataset / Mosaic StreamingDataset

Para datasets mas grandes (>100K imagenes), empaquetar los .npy en shards TAR para lectura secuencial optimizada. Reduce overhead de filesystem con muchos archivos pequenos.
