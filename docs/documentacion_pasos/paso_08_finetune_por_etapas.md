# Paso 8 — Fine-tuning por etapas del sistema MoE completo

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-05 |
| **Alcance** | Paso 8 — Fine-tuning por etapas del sistema MoE completo. Descongelamiento progresivo en 3 stages: router solo → router + cabezas → fine-tuning global. Implementado en `src/pipeline/fase5/`. Genera checkpoints por stage consumidos por Fase 6 (dashboard) |
| **Estado general** | ✅ **Implementado — dry-run verificado** |

---

## 1. Objetivo

Fine-tuning progresivo del sistema MoE completo en **3 etapas**: primero el router solo sobre embeddings congelados, luego router + cabezas clasificadoras, y finalmente fine-tuning global de todo el sistema. El objetivo es **estabilizar el routing antes de descongelar gradualmente los expertos**, evitando desestabilizar los modelos ya convergidos de las fases anteriores (Pasos 5.1–5.6).

La estrategia sigue el Método B de `proyecto_moe.md` (por partes), adaptado con 3 stages explícitos y LRs diferenciados por componente. Los expertos ya convergidos individualmente en Fases 2–3 se incorporan al sistema MoE con sus checkpoints, y el fine-tuning global ajusta las interfaces entre componentes sin destruir los pesos aprendidos.

---

## 2. Prerrequisitos

Los siguientes 7 artefactos deben existir antes de ejecutar Fase 5:

| # | Artefacto | Generado en | Ruta |
|---|---|---|---|
| 1 | Expert 0 checkpoint (ConvNeXt-Tiny, Chest) | Paso 5.1 | `checkpoints/expert_00_convnext_tiny/expert1_best.pt` |
| 2 | Expert 1 checkpoint (EfficientNet-B3, ISIC) | Paso 5.2 | `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` |
| 3 | Expert 2 checkpoint (VGG16-BN, OA) | Paso 5.3 | `checkpoints/expert_02_vgg16_bn/expert_oa_best.pt` |
| 4 | Expert 3 checkpoint (MC3-18, LUNA16) | Paso 5.4 | `checkpoints/expert_03_vivit_tiny/expert3_best.pt` |
| 5 | Expert 4 checkpoint (Swin3D-Tiny, Páncreas) | Paso 5.5 | `checkpoints/expert_04_swin3d_tiny/expert4_best.pt` |
| 6 | Expert 5 checkpoint (CAE, OOD) | Paso 5.6 | `checkpoints/expert_05_cae/cae_best.pt` |
| 7 | Router ganador del ablation study | Paso 6/7 | `best_router_{type}.{pt,joblib,faiss}` (en directorio de embeddings) |

En modo `--dry-run` estos artefactos no son necesarios: el sistema se instancia con pesos aleatorios (`weights=None`).

---

## 3. Etapas de fine-tuning

| Etapa | Qué se entrena | Qué está congelado | LR | Épocas | Paciencia |
|---|---|---|---|---|---|
| **Stage 1** | Router solo | Backbone + todos los expertos (0–5) | Router: `1e-3` | 50 | 10 |
| **Stage 2** | Router + cabezas clasificadoras | Backbone + capas conv/feature de expertos | Router: `1e-4`, Cabezas: `1e-4` | 30 | 8 |
| **Stage 3** | Todo (fine-tuning global) | Nada | Router: `1e-4`, Expertos: `1e-6`, Backbone: `1e-6` | 10 (max) | 5 |

Valores extraídos de `src/pipeline/fase5/fase5_config.py` (fuente de verdad):

| Constante | Valor | Stage |
|---|---|---|
| `STAGE1_LR_ROUTER` | 1e-3 | 1 |
| `STAGE1_EPOCHS` | 50 | 1 |
| `STAGE1_PATIENCE` | 10 | 1 |
| `STAGE1_BATCH_SIZE` | 64 | 1 |
| `STAGE2_LR_ROUTER` | 1e-4 | 2 |
| `STAGE2_LR_HEADS` | 1e-4 | 2 |
| `STAGE2_EPOCHS` | 30 | 2 |
| `STAGE2_PATIENCE` | 8 | 2 |
| `STAGE3_LR_ROUTER` | 1e-4 | 3 |
| `STAGE3_LR_EXPERTS` | 1e-6 | 3 |
| `STAGE3_LR_BACKBONE` | 1e-6 | 3 |
| `STAGE3_EPOCHS_MAX` | 10 | 3 |
| `STAGE3_PATIENCE` | 5 | 3 |

---

## 4. Función de pérdida total

```
L_total = L_task + α·L_aux + β·L_error
```

### 4.1 L_task — Pérdida del experto activo

La pérdida de la tarea depende del experto que procesa el batch:

| Experto | Loss | Detalle |
|---|---|---|
| Expert 0 (Chest) | `BCEWithLogitsLoss` | Multilabel, 14 patologías |
| Expert 1 (ISIC) | `CrossEntropyLoss` | Multiclase, 9 clases |
| Expert 2 (OA) | `CrossEntropyLoss` | Multiclase, 3 clases |
| Expert 3 (LUNA) | `CrossEntropyLoss` | Binaria, 2 clases |
| Expert 4 (Páncreas) | `CrossEntropyLoss` | Binaria, 2 clases |
| Expert 5 (CAE) | `MSE(recon, x) + 0.1·L1(recon, x)` | Reconstrucción, sin clasificación |

### 4.2 L_aux — Balance de carga (Switch Transformer)

```
L_aux = α · N · Σ f_i · P_i
```

- `α = ALPHA_L_AUX = 0.01` — coeficiente de penalización
- `N = N_EXPERTS_DOMAIN = 5` — número de expertos de dominio
- `f_i` = fracción real de muestras del batch asignadas al experto `i` (decisión dura, `argmax`)
- `P_i` = probabilidad media que el router asigna al experto `i` (softmax medio sobre el batch)

**Solo expertos 0–4 participan en L_aux.** El Expert 5 CAE está excluido del cálculo de balance de carga — se activa por entropía, no por routing directo.

L_aux penaliza la concentración de carga: si toda la carga va a un experto, `L_aux ≈ α·N`. Si la carga está perfectamente balanceada, `L_aux = α`.

### 4.3 L_error — Penalización por enviar imágenes válidas al CAE

```
L_error = β · MSE(recon, x)
```

- `β = BETA_L_ERROR = 0.1` — coeficiente de penalización

L_error penaliza al router cuando delega imágenes médicas válidas (in-distribution) al Expert 5 CAE. Previene el problema del "router perezoso" que enviaría todo al CAE porque siempre reconstruye bien.

En el dry-run con router de 5 salidas (sin slot OOD expandido), `L_error = 0` (placeholder para producción).

---

## 5. Estrategia de congelamiento por etapa

| Componente | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| **Backbone** | ❄️ Congelado | ❄️ Congelado | 🔥 Entrenable |
| **Expert 0** (ConvNeXt) | ❄️ Congelado | ❄️ Conv congelado / 🔥 Cabeza (`model.classifier`) entrenable | 🔥 Entrenable |
| **Expert 1** (EfficientNet) | ❄️ Congelado | ❄️ Conv congelado / 🔥 Cabeza (`model.classifier`) entrenable | 🔥 Entrenable |
| **Expert 2** (VGG16-BN) | ❄️ Congelado | ❄️ Conv congelado / 🔥 Cabeza (`classifier`) entrenable | 🔥 Entrenable |
| **Expert 3** (MC3-18) | ❄️ Congelado | ❄️ Conv congelado / 🔥 Cabeza (`classifier`) entrenable | 🔥 Entrenable |
| **Expert 4** (Swin3D) | ❄️ Congelado | ❄️ Conv congelado / 🔥 Cabeza (`backbone.head`) entrenable | 🔥 Entrenable |
| **Expert 5** (CAE) | ❄️ Congelado | ❄️ Encoder congelado / 🔥 Decoder (`decoder_fc`, `decoder_conv`) entrenable | 🔥 Entrenable |
| **Router** | 🔥 Entrenable | 🔥 Entrenable | 🔥 Entrenable |

Los prefijos de cabeza por experto están definidos en `EXPERT_HEAD_PREFIXES` de `fase5_config.py`. La función `freeze_except_head()` en `freeze_utils.py` congela todos los parámetros excepto aquellos cuyo nombre comienza con los prefijos especificados.

---

## 6. Nota de divergencia con la guía oficial

| Aspecto | Guía oficial (`proyecto_moe.md`) | Implementación (`fase5_config.py`) | Razón |
|---|---|---|---|
| **LR expertos Stage 3** | `1e-5` (Método B, FASE 3) | `1e-6` (`STAGE3_LR_EXPERTS`) | Mayor estabilidad con expertos 3D heterogéneos |

La guía oficial (`proyecto_moe.md` §7, Método B) propone `LR=1e-5` para los expertos en la fase de fine-tuning global. Este proyecto usa `LR=1e-6` de forma **deliberada**: dado que el sistema integra 6 expertos heterogéneos (2D y 3D), un learning rate más bajo en la fase global minimiza el riesgo de desestabilizar los expertos ya convergidos, especialmente los modelos 3D con gradientes más inestables.

Se acepta convergencia más lenta a cambio de mayor estabilidad. Esta decisión está documentada en `arquitectura_documentacion.md` §5.6 y en el docstring de `STAGE3_LR_EXPERTS` en `fase5_config.py`.

---

## 7. Técnicas de entrenamiento

| Técnica | Configuración | Detalle |
|---|---|---|
| **FP16 Mixed Precision** | `FP16_ENABLED = True` | Expertos 0–4 y router. **Expert 5 CAE: FP32 obligatorio** (no autocast) |
| **Gradient Accumulation** | `ACCUMULATION_STEPS = 4` | Batch efectivo mínimo del proyecto |
| **Gradient Checkpointing** | `GRAD_CHECKPOINT_3D = True` | Obligatorio para Expert 3 (MC3-18) y Expert 4 (Swin3D). Reduce VRAM a cambio de velocidad |
| **No reset optimizer** | `RESET_OPTIMIZER_BETWEEN_STAGES = False` | **REGLA CRÍTICA (§5.6):** NO reiniciar el optimizer entre stages. El momentum acumulado en Stage 1–2 ayuda a la convergencia de Stage 3 |
| **Param groups diferenciados** | Por stage | Cada stage construye param groups con LRs diferentes para router vs. expertos vs. backbone |
| **Mixed batch DataLoader** | `MIXED_BATCH_SIZE = 32` | DataLoader mixto proporcional para Stage 2–3, con pesos proporcionales al tamaño de cada dataset |

### 7.1 Excepción FP32 para Expert 5 CAE

El Expert 5 (ConvAutoEncoder) **no puede usar FP16**. La reconstrucción requiere precisión completa para evitar artefactos numéricos en la decodificación. Durante el forward pass del MoE, el autocast se desactiva selectivamente para Expert 5.

---

## 8. Archivos clave

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase5/fase5_finetune_global.py` | Orquestador principal: CLI, flujo de ejecución, dry-run por stage, forward/backward sintético |
| `src/pipeline/fase5/fase5_config.py` | Constantes de hiperparámetros (LRs, épocas, paciencia, loss coefficients, checkpoints). Fuente de verdad |
| `src/pipeline/fase5/freeze_utils.py` | Funciones de congelamiento: `freeze_module`, `unfreeze_module`, `freeze_except_head`, `apply_stage{1,2,3}_freeze`, `log_freeze_state` |
| `src/pipeline/fase5/moe_model.py` | `MoESystem` wrapper: ensambla 6 expertos + router en un `nn.Module`. `build_moe_system_dry_run()` para instanciación sintética |
| `src/pipeline/fase5/dataloader_mixed.py` | `SyntheticMixedDataset` para dry-run, `get_mixed_dataloader()`, `_mixed_collate_fn` para shapes mixtos (2D/3D) |
| `src/pipeline/fase2/routers/linear.py` | Router `LinearGatingHead(d_model, n_experts)` — el router ganador del ablation study |
| `src/pipeline/fase2/models/expert1_convnext.py` | Expert 0 — `Expert1ConvNeXtTiny` |
| `src/pipeline/fase2/models/expert2_efficientnet.py` | Expert 1 — `Expert2EfficientNetB3` |
| `src/pipeline/fase2/models/expert_oa_vgg16bn.py` | Expert 2 — `ExpertOAVGG16BN` |
| `src/pipeline/fase2/models/expert3_r3d18.py` | Expert 3 — `Expert3MC318` |
| `src/pipeline/fase2/models/expert4_swin3d.py` | Expert 4 — `ExpertPancreasSwin3D` |
| `src/pipeline/fase3/models/expert5_cae.py` | Expert 5 — `ConvAutoEncoder` |

---

## 9. Output artifacts

| Artefacto | Stage | Descripción |
|---|---|---|
| `checkpoints/fase5/moe_stage1.pt` | Stage 1 | Checkpoint del sistema MoE tras entrenar solo el router |
| `checkpoints/fase5/moe_stage2.pt` | Stage 2 | Checkpoint tras entrenar router + cabezas clasificadoras |
| `checkpoints/fase5/moe_final.pt` | Stage 3 | Checkpoint final del fine-tuning global — artefacto de producción |

En modo `--dry-run` **no se guardan checkpoints a disco**.

---

## 10. Cómo ejecutar

### 10.1 Dry-run completo (verifica todos los stages, sin entrenar)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py --dry-run
```

### 10.2 Dry-run de un stage específico

```bash
# Solo Stage 1
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py --dry-run --stage 1

# Solo Stage 2
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py --dry-run --stage 2

# Solo Stage 3
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py --dry-run --stage 3
```

### 10.3 Ejecución real Stage 1 (requiere embeddings de Fase 1)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py \
    --stage 1 --embeddings ./embeddings/vit_tiny_patch16_224
```

### 10.4 Ejecución real Stage 2 (requiere checkpoint Stage 1)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py \
    --stage 2 --resume checkpoints/fase5/moe_stage1.pt
```

### 10.5 Ejecución real Stage 3 (fine-tuning global)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py \
    --stage 3 --resume checkpoints/fase5/moe_stage2.pt
```

### 10.6 Ejecución completa (stages 1→2→3 secuencial)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase5/fase5_finetune_global.py --stage all
```

### 10.7 Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--stage` | `all` | Etapa a ejecutar: `1`, `2`, `3`, o `all` |
| `--dry-run` | `False` | Verifica el pipeline sin entrenar (datos sintéticos) |
| `--resume` | `None` | Ruta al checkpoint de una etapa anterior para continuar |
| `--batch-size` | 32 (`MIXED_BATCH_SIZE`) | Batch size del DataLoader mixto |
| `--embeddings` | `./embeddings/vit_tiny_patch16_224` | Directorio de embeddings para Stage 1 |

> **⚠️ Restricción activa:** sin `--dry-run`, el script se rehúsa a ejecutar (`sys.exit(1)` con mensaje de error). El entrenamiento real está bloqueado hasta completar los prerrequisitos.

---

## 11. Dry-run verificado

| Verificación | Resultado |
|---|---|
| Importaciones de todos los módulos | ✅ OK |
| Sistema MoE instanciado (6 expertos + 1 router) | ✅ OK — `build_moe_system_dry_run(d_model=192)` |
| **Stage 1** — Freeze aplicado (solo Router entrenable) | ✅ OK |
| **Stage 1** — Forward+backward sintético (6 expertos) | ✅ OK — L_total verificado |
| **Stage 2** — Freeze aplicado (Router + cabezas entrenables) | ✅ OK |
| **Stage 2** — Forward+backward sintético (6 expertos) | ✅ OK — L_total verificado |
| **Stage 3** — Freeze aplicado (TODO descongelado) | ✅ OK |
| **Stage 3** — Forward+backward sintético (6 expertos) | ✅ OK — L_total verificado |
| L_total = L_task + α·L_aux + β·L_error verificado | ✅ OK — con valores sintéticos |
| No se guardaron checkpoints a disco | ✅ Confirmado (dry-run) |
| Dry-run exit code | ✅ 0 |
| Entrenamiento real | ⏳ Pendiente (requiere checkpoints de Pasos 5.1-5.6 y embeddings de Fase 1) |

---

## 12. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin pesos preentrenados** | Todos los modelos instanciados con `weights=None` — entrenamiento from scratch obligatorio |
| **Sin augmentaciones de imagen** | Este script de fine-tuning no aplica image augmentation directamente — usa los transforms definidos en los dataloaders de cada experto |
| **PROHIBIDO ENTRENAR** | Solo `--dry-run` para verificación. Sin `--dry-run`, el script hace `sys.exit(1)` |
| **Expert 5 CAE: FP32 obligatorio** | No autocast — precisión completa para reconstrucción |
| **No reiniciar optimizer** | `RESET_OPTIMIZER_BETWEEN_STAGES = False` — el momentum acumulado en Stage 1–2 se preserva |
| **Balance de carga solo expertos 0–4** | Expert 5 CAE excluido de L_aux y del cálculo de `max(f_i)/min(f_i)` |
| **Gradient checkpointing 3D** | Obligatorio para Expert 3 (MC3-18) y Expert 4 (Swin3D) |

---

## 13. Incongruencias detectadas y resolución

Se encontraron 5 incongruencias entre `proyecto_moe.md` (guía oficial) y `arquitectura_documentacion.md` (documento de diseño del proyecto). La resolución adopta `arquitectura_documentacion.md` §5.6 como fuente de verdad, reflejada en `fase5_config.py`.

| # | Aspecto | `proyecto_moe.md` | `arquitectura_documentacion.md` | Resolución adoptada |
|---|---|---|---|---|
| 1 | **LR expertos Stage 3** | `1e-5` (§7 Método B, FASE 3) | `1e-6` (§5.6) | `1e-6` — mayor estabilidad con expertos 3D heterogéneos |
| 2 | **Épocas Stage 1** | 10 (Método A: "épocas 1–10") | ~50 (§5.6) | 50 — Stage 1 requiere más épocas al entrenar solo el router sobre embeddings |
| 3 | **Épocas Stage 2** | 15 (Método A: "épocas 11–25") | ~30 (§5.6) | 30 — más épocas para estabilizar cabezas clasificadoras con 6 expertos heterogéneos |
| 4 | **Stages naming** | "Método B" usa FASE 1/2/3 del entrenamiento MoE | Stage 1/2/3 del fine-tuning | Stage 1/2/3 — alineado con terminología de `fase5_config.py` |
| 5 | **Expert 5 CAE** | No contemplado en la guía (5 expertos) | Integrado como expert_id=5 con L_error | Incluido — extensión aprobada verbalmente por el profesor |

---

## 14. Notas

- **Los expertos deben haberse entrenado** (Pasos 5.1–5.6) antes de ejecutar este paso en modo real. En dry-run se instancian con pesos aleatorios.
- **En Stage 1, el router opera sobre embeddings pre-computados** (vectores `.npy` de Fase 1), no sobre imágenes crudas. El batch size de Stage 1 es 64 (embeddings son vectores pequeños, no imágenes).
- **En Stages 2–3, el DataLoader mixto proporcional** (`dataloader_mixed.py`) garantiza exposición a todos los dominios con pesos proporcionales al tamaño de cada dataset (`MIXED_SAMPLES_PER_DATASET` en `fase5_config.py`).
- **Expert 5 CAE excluido del cálculo de balance de carga:** solo expertos 0–4 participan en L_aux y en la restricción `max(f_i)/min(f_i) ≤ 1.30`.
- **Presupuesto VRAM estimado para Stage 3:** ~15–18 GB (objetivo < 19 GB en GPU de 20 GB). Los expertos 3D con gradient checkpointing y FP16 caben dentro del presupuesto.
- **Collate mixto:** el DataLoader mixto usa `_mixed_collate_fn` que maneja batches con shapes heterogéneos (2D: `[3, 224, 224]` vs. 3D: `[1, 64, 64, 64]`). Si todas las muestras del batch tienen la misma shape, se apilan en un tensor; si no, se retornan como lista.
- **Forward pass del MoE en dry-run:** el router recibe un embedding sintético (`torch.randn(B, d_model)`) en lugar del embedding real del backbone, ya que el backbone no existe como módulo separado en esta implementación.

---

*Documento generado el 2026-04-05. Fuentes: `src/pipeline/fase5/fase5_config.py`, `fase5_finetune_global.py`, `freeze_utils.py`, `moe_model.py`, `dataloader_mixed.py`, `proyecto_moe.md` §7, `arquitectura_documentacion.md` §5.6.*
