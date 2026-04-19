# Metricas — Expert 1 (ChestXray14)

## Definiciones

### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

Mide la capacidad del modelo de separar positivos de negativos, independiente del umbral. Rango [0, 1]:
- **1.0:** Separacion perfecta
- **0.5:** Clasificador aleatorio
- **<0.5:** Peor que aleatorio (modelo invertido)

Se calcula por clase (14 curvas) y como **macro AUC** (promedio simple de las 14). Se usa `np.nanmean` para excluir clases sin positivos o sin negativos en el batch evaluado.

AUC es la metrica principal porque es invariante al umbral de decision, lo que importa en screening medico donde el umbral se ajusta segun el trade-off sensibilidad/especificidad.

### F1 Score (Macro)

Media armonica de precision y recall, calculada por clase y promediada (macro):

```
F1_clase = 2 * (Precision * Recall) / (Precision + Recall)
F1_macro = mean(F1_clase para todas las clases)
```

Usa umbral fijo de 0.5 sobre las salidas del modelo. Es informativo pero no se usa para early stopping ni seleccion de modelo.

### Focal Loss

```
FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
```

- **pt:** probabilidad predicha para la clase correcta
- **gamma=2.0:** reduce peso de ejemplos faciles (pt alto). Cuando pt → 1, el factor (1-pt)^2 → 0.
- **alpha:** pos_weight por clase (n_neg/n_pos, clamped a 50). Compensa desbalance.

Relacion con BCE: cuando gamma=0, FocalLoss = alpha-weighted BCE.

## Metricas durante el Entrenamiento

Cada epoca genera un log con:

```json
{
  "epoch": 5,
  "train_loss": 0.1234,
  "val_loss": 0.1456,
  "val_macro_auc": 0.7823,
  "val_macro_f1": 0.3210,
  "val_auc_per_class": [0.82, 0.91, 0.85, ...],
  "lr": 2.8e-04,
  "epoch_time_s": 180.5,
  "is_best": true,
  "world_size": 2
}
```

### Como interpretar

| Metrica | Buen signo | Mal signo |
|---------|------------|-----------|
| `train_loss` | Desciende gradualmente | Sube, oscila, NaN |
| `val_loss` | Desciende o se estabiliza | Sube mientras train_loss baja (overfitting) |
| `val_macro_auc` | Sube, se acerca a 0.80+ | Estancado en <0.60 despues de 10 epocas |
| `val_macro_f1` | Sube con el AUC | Puede ser bajo por el umbral fijo, no preocuparse |
| `lr` | Decae segun cosine schedule | Constante (scheduler no funciona) |
| `epoch_time_s` | Estable entre epocas | Crece (memory leak o swap) |

### Valores de referencia para ChestXray14

Literature benchmarks (Wang et al., CheXNet, etc.):
- Macro AUC en test: 0.75 - 0.85 (con pretrained ImageNet)
- Training desde cero: esperar 0.70 - 0.78

## Red Flags

### NaN en loss
**Causa probable:** Learning rate demasiado alto, pos_weight sin clamp, o FP16 overflow.
**Solucion:** Verificar que `pos_weight.clamp(max=50)` esta activo. Reducir LR. Verificar que GradScaler esta habilitado.

### Divergencia (loss sube continuamente)
**Causa probable:** LR muy alto para el batch efectivo actual.
**Solucion:** Reducir LR a la mitad. Verificar que gradient clipping (`max_norm=1.0`) esta activo.

### GPU memory spikes / OOM
**Causa probable:** Batch demasiado grande, memory leak en el DataLoader, o acumulacion de tensores fuera de `torch.no_grad()`.
**Solucion:** Reducir batch_size. Verificar que validacion usa `@torch.no_grad()`. Revisar que no se acumulen tensores en listas sin `.cpu()`.

### AUC = NaN para alguna clase
**Esperado en dry-run:** Con 64 muestras, clases raras (Hernia ~1%) pueden no tener positivos.
**Preocupante en training completo:** Si una clase tiene NaN con >10K muestras, algo esta mal en las etiquetas o el preprocesamiento.

### val_macro_auc estancado
- Despues de 5 epocas sin mejora → normal, patience es 20.
- Despues de 20 epocas sin mejora → early stopping activara.
- Si se estanca en <0.60 → verificar que las augmentaciones no son demasiado agresivas, o que las etiquetas son correctas.

## TTA (Test-Time Augmentation)

En la evaluacion final, se promedian las probabilidades de:
1. Input original
2. Input con flip horizontal

```
probs_final = (probs_original + probs_flipped) / 2
```

El flip horizontal es la unica augmentation razonable para chest X-rays (la anatomia es aproximadamente simetrica). Rotaciones o escalados distorsionarian las proporciones anatomicas.

TTA tipicamente mejora el AUC en 0.5-1.5 puntos porcentuales.
