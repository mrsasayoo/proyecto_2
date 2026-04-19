# Refactorización: Eliminar Warnings y Optimizar Carga de Datos

> **Para Claude:** Plan de refactorización estructurado para Expert 1 (NIH ChestX-ray14). Ejecutar tareas en orden secuencial con revisión entre cada una.

**Objetivo:** Eliminar deprecations, limpiar parámetros legacy, optimizar carga de datasets y corregir bugs de AUC/validación sin afectar la estabilidad del entrenamiento DDP.

**Arquitectura:** 5 tareas independientes pero que se integran en el flujo de entrenamiento: (1) variable NCCL, (2) dataloader Albumentations, (3) dataset legacy params, (4) train script redundancia, (5) validación AUC.

**Tech Stack:** PyTorch, DDP, Albumentations 1.3+, NumPy.

---

## Tarea 1: Corregir NCCL_BLOCKING_WAIT → TORCH_NCCL_BLOCKING_WAIT

**Archivos:**
- Modificar: `run_expert.sh` (línea donde se define `NCCL_BLOCKING_WAIT`)

**Contexto:**
El log muestra 4 warnings deprecation sobre `NCCL_BLOCKING_WAIT`. PyTorch ≥2.0 cambió a `TORCH_NCCL_BLOCKING_WAIT`. Es una variable de entorno, no código Python.

**Pasos:**

1. **Localizar la línea en run_expert.sh**
   ```bash
   grep -n "NCCL_BLOCKING_WAIT" /mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/run_expert.sh
   ```
   Espera ver algo como: `export NCCL_BLOCKING_WAIT=1`

2. **Reemplazar la variable**
   - Buscar: `NCCL_BLOCKING_WAIT=`
   - Reemplazar por: `TORCH_NCCL_BLOCKING_WAIT=`
   - Mantener el valor (1)

3. **Verificar que no hay otras referencias antiguas**
   ```bash
   grep -r "NCCL_BLOCKING_WAIT" /mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/src/ 2>/dev/null || echo "OK"
   ```

4. **Confirmar**
   - Ejecutar dry-run nuevamente y verificar que desaparecen los 4 warnings NCCL.

**Impacto:** Cosmético. Sin cambios en funcionamiento, solo desaparece warning.

---

## Tarea 2: Actualizar ShiftScaleRotate → Affine en Dataloader

**Archivos:**
- Modificar: `src/pipeline/fase2/dataloader_expert1.py` (líneas ~130-150)

**Contexto:**
Albumentations 1.3+ deprecó `ShiftScaleRotate` a favor de `Affine`. El log muestra warnings. `fill_value` no es válido en ambas transformaciones.

**Pasos:**

1. **Revisar la configuración actual en dataloader_expert1.py**
   - Buscar `ShiftScaleRotate` (línea ~137)
   - Buscar `Rotate` (línea ~144)
   - Nota qué parámetros usan

2. **Reemplazar ShiftScaleRotate por Affine**
   Antes:
   ```python
   A.ShiftScaleRotate(
       shift_limit=0.06,
       scale_limit=(-0.15, 0.10),
       rotate_limit=0,
       border_mode=cv2.BORDER_CONSTANT,
       value=0,  # ← ELIMINAR (no válido)
       p=0.5
   ),
   ```
   
   Después:
   ```python
   A.Affine(
       translate_percent=(-0.06, 0.06),  # shift_limit → translate_percent
       scale=(0.85, 1.10),  # scale_limit invertido: (1-0.15, 1+0.10)
       rotate=0,
       mode=cv2.BORDER_CONSTANT,
       cval=0,  # ← CORRECTO (fill_value → cval en Affine)
       p=0.5
   ),
   ```

3. **Corregir Rotate**
   Antes:
   ```python
   A.Rotate(
       limit=10,
       border_mode=cv2.BORDER_CONSTANT,
       value=0,  # ← ELIMINAR
       p=0.5
   ),
   ```
   
   Después:
   ```python
   A.Rotate(
       limit=10,
       border_mode=cv2.BORDER_CONSTANT,
       cval=0,  # ← CORRECTO
       p=0.5
   ),
   ```

4. **Verificar imports**
   - Confirmar que `import cv2` y `from albumentations import ...` están presentes.

5. **Dry-run sintáctico**
   ```bash
   python3 -c "import ast; ast.parse(open('src/pipeline/fase2/dataloader_expert1.py').read()); print('OK')"
   ```

6. **Confirmar**
   - Ejecutar dry-run nuevamente.
   - Warnings de Albumentations deben desaparecer.

**Impacto:** Cosmético + compatibilidad futura. Sin cambios en distribución de augmentaciones.

---

## Tarea 3: Limpiar Parámetros Legacy en Dataset

**Archivos:**
- Modificar: `src/pipeline/datasets/chest.py` (constructor líneas ~90-120)
- Revisar: `src/pipeline/fase2/train_expert1_ddp.py` (líneas donde se llama ChestXray14Dataset)

**Contexto:**
El dataset acepta `csv_path`, `img_dir`, `file_list` (legacy, para compatibilidad con código antiguo) pero los ignora. El log muestra warnings repetidos. Limpiar esto requiere:
1. Hacer `preprocessed_dir` obligatorio en ChestXray14Dataset.
2. Actualizar todas las llamadas en train_expert1_ddp.py para pasar `preprocessed_dir`.

**Pasos:**

1. **En chest.py — Revisar constructor actual**
   - Buscar la sección de parámetros legacy
   - Nota que hay un bloque que loguea warnings si se pasan estos parámetros

2. **Opción A: Simplificar el constructor (RECOMENDADO)**
   - Eliminar `csv_path`, `img_dir`, `file_list` del constructor
   - Hacer `preprocessed_dir` obligatorio (sin default relativo)
   - Ejemplo:
     ```python
     def __init__(
         self,
         preprocessed_dir: str,  # ← obligatorio
         split: str = "train",
         mode: str = "expert",
         transform=None,
         use_cache: bool = True
     ):
         self.preprocessed_dir = Path(preprocessed_dir)
         # ... resto igual
     ```

3. **En train_expert1_ddp.py — Actualizar todas las llamadas**
   - Buscar todas las líneas donde se instancia `ChestXray14Dataset`
   - Eliminar los parámetros legacy (`csv_path`, `img_dir`, `file_list`)
   - Asegurar que `preprocessed_dir` se pasa explícitamente
   - Ejemplo (ya implementado en fixes anteriores):
     ```python
     train_ds = ChestXray14Dataset(
         preprocessed_dir=str(paths["images_dir"].parent / "preprocessed"),
         split="train",
         mode="expert",
         transform=train_tfm,
     )
     ```

4. **Dry-run sintáctico**
   ```bash
   python3 -c "import ast; ast.parse(open('src/pipeline/datasets/chest.py').read()); ast.parse(open('src/pipeline/fase2/train_expert1_ddp.py').read()); print('OK')"
   ```

5. **Confirmar**
   - Ejecutar dry-run nuevamente.
   - Los 6 warnings de parámetros legacy deben desaparecer.

**Impacto:** Código más limpio. Sin cambios en comportamiento.

---

## Tarea 4: Investigar y Optimizar Carga Redundante de Test Dataset

**Archivos:**
- Revisar: `src/pipeline/fase2/train_expert1_ddp.py` (función `_build_datasets()`, líneas ~430-500)
- Revisar: `src/pipeline/fase2/train_expert1_ddp.py` (loop de validación y TTA)

**Contexto:**
El log muestra que el dataset de test se carga **dos veces** (líneas 78-97 y 99-117 del dry_run.log). Esto es necesario para TTA (Test Time Augmentation), pero la estructura actual carga la metadata dos veces, causando overhead de I/O y memoria.

**Análisis:**

1. **Revisar `_build_datasets()`**
   - Buscar dónde se instancian: `train_ds`, `val_ds`, `test_ds`, `test_flip_ds`
   - Confirmación: hay 4 instancias de ChestXray14Dataset.

2. **Entender la intención de TTA**
   - `test_ds`: dataset normal (sin flip)
   - `test_flip_ds`: dataset con HorizontalFlip en transformación (para TTA)
   - Ambos leen desde `preprocessed_dir`, no hay diferencia en cuáles archivos cargan, solo en qué transformaciones se aplican.

3. **Optimización propuesta**
   - **Opción A (Recomendado):** Crear una sola instancia de dataset y aplicar transformaciones diferentes en la evaluación (TTA en el loop de inferencia, no en el dataloader).
   - **Opción B (Menos cambios):** Mantener dos instancias pero lazy-load metadata (cachear la metadata cargada).
   
   **Usaremos Opción A por claridad y eficiencia.**

**Pasos:**

1. **En train_expert1_ddp.py — Refactorizar TTA**
   - Eliminar la creación de `test_flip_ds`
   - Crear solo una instancia `test_ds`
   - En la evaluación final (función `eval_tta` o dónde se hace), aplicar HorizontalFlip manualmente a los logits/probabilidades en PyTorch (no en albumentations)

2. **Implementar TTA en la evaluación**
   Pseudo-código:
   ```python
   def eval_with_tta(model, test_loader, device):
       all_probs = []
       for batch in test_loader:
           imgs = batch['img'].to(device)  # [B, 1, 256, 256]
           
           # Original prediction
           probs1 = model(imgs)
           
           # Flipped prediction
           imgs_flipped = torch.flip(imgs, dims=[-1])  # horizontal flip
           probs2 = model(imgs_flipped)
           
           # Average the probabilities
           probs = (probs1 + probs2) / 2.0
           all_probs.append(probs.cpu())
       
       all_probs = torch.cat(all_probs, dim=0)
       return all_probs
   ```

3. **Eliminar la instancia `test_flip_ds` de `_build_datasets()`**

4. **Actualizar referencias a `test_flip_ds` en el código principal**
   - Buscar dónde se usa `test_flip_ds` en el loop de entrenamiento
   - Reemplazar con la nueva función `eval_with_tta`

5. **Dry-run sintáctico**
   ```bash
   python3 -c "import ast; ast.parse(open('src/pipeline/fase2/train_expert1_ddp.py').read()); print('OK')"
   ```

6. **Confirmar**
   - Ejecutar dry-run nuevamente.
   - En el log, el dataset de test debería cargarse UNA sola vez (no dos).
   - TTA debería seguir funcionando, produciendo los mismos resultados (o muy similares).

**Impacto:** Mejora memoria y I/O. Resultado TTA igual o ligeramente mejor (más suave averaging de predicciones).

---

## Tarea 5: Investigar AUC en 5/14 Clases y Optimizar Validación

**Archivos:**
- Revisar: `src/pipeline/fase2/train_expert1_ddp.py` (función de validación, líneas ~1200-1400)
- Revisar: métricas y cálculo de AUC

**Contexto:**
El log muestra: `AUC calculado sobre 5/14 clases (9 sin positivos o sin negativos)`. Esto es *esperado* en el dry-run (solo 64 muestras), pero debería verificarse que desaparece en entrenamiento real.

**Análisis:**

1. **Entender dónde se calcula el AUC**
   - Buscar función que calcula `val_macro_auc`
   - Buscar dónde se genera el warning "AUC calculado sobre X/14 clases"

2. **Causa raíz**
   - Con 64 muestras en validación y distribución desbalanceada (Hernia 0.2%), muchas clases no tendrán positivos
   - `sklearn.metrics.roc_auc_score` requiere ambas clases (0, 1) presentes en y_true
   - Si falta una clase, devuelve NaN y se filtra

3. **Verificación de corrección**
   - El comportamiento actual es correcto: reportar AUC solo para clases donde se puede calcular
   - En entrenamiento real (88K samples en train, 11K en val), todas las 14 clases deberían tener positivos

4. **Mejora opcional (no crítica)**
   - Considerar usar `roc_auc_score(..., multi_class='ovr', zero_division=0)` para clases con un solo valor
   - O implementar una métrica custom que interpole AUC=0.5 para clases con n_samples < 10

**Pasos:**

1. **En train_expert1_ddp.py — Revisar cálculo de AUC**
   - Buscar la función que calcula el macro_auc
   - Confirmar que no hay código muerto ni lógica duplicada

2. **Considerar si la métrica es suficiente**
   - ¿Es suficiente reportar AUC solo para clases representadas?
   - ¿O debería rellenar AUC=0.5 para clases no representadas?
   
   **Recomendación:** Mantener como está (es estándar), pero documentar en un comentario.

3. **Dry-run sintáctico**
   ```bash
   python3 -c "import ast; ast.parse(open('src/pipeline/fase2/train_expert1_ddp.py').read()); print('OK')"
   ```

4. **Confirmar**
   - Ejecutar dry-run nuevamente.
   - Si sigue mostrando warning en 64 muestras: correcto, esperado.
   - En entrenamiento real, el warning debería desaparecer.

**Impacto:** Ninguno en código, solo confirmación de corrección.

---

## Plan de Ejecución

**Orden recomendado:** 1 → 2 → 3 → 4 → 5 (Tareas 1, 2, 3 son independientes; 4 y 5 son confirmaciones).

**Criterio de éxito:** 
- Dry-run ejecutado sin warnings (excepto los de PyTorch/CUDA, que son normales).
- Código pasa PEP8 (usa `black` o `flake8` si disponible).
- Entrenamiento real puede lanzarse sin cambios posteriores.

**Commits recomendados:**
```
commit 1: refactor: update NCCL_BLOCKING_WAIT to TORCH_NCCL_BLOCKING_WAIT
commit 2: refactor: replace ShiftScaleRotate with Affine, fix fill_value
commit 3: refactor: remove legacy params from ChestXray14Dataset
commit 4: refactor: optimize test dataset loading, implement TTA in evaluation loop
commit 5: docs: add comment about AUC metric behavior with imbalanced data
```

---

## Notas Técnicas

- **DDP Compatibility:** Todas las modificaciones mantienen compatibilidad con DDP (no hay colectivas nuevas ni operaciones asincrónicas).
- **Backward Compatibility:** Las tareas 1-3 son cambios de código limpio (sin cambios de comportamiento). Tarea 4 requiere validación de que TTA produce resultados equivalentes.
- **Testing:** Ejecutar dry-run después de cada tarea principal (1-2, 2, 3-4) para verificar que no hay regresiones.

