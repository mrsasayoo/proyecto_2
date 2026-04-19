# Experto 5 — CAE (Convolutional AutoEncoder) — Model Card

## Resumen

| Campo                | Valor                                                                    |
|----------------------|--------------------------------------------------------------------------|
| **Experto ID**       | 5                                                                        |
| **Arquitectura**     | CAE (Convolutional AutoEncoder) — Encoder-Latent-Decoder                 |
| **Dataset**          | Todos los 5 datasets combinados (Chest + ISIC + OA + LUNA16 + Páncreas) |
| **Modalidad**        | Todas (2D y 3D)                                                          |
| **Tarea**            | Detección OOD — NO clasifica patologías                                  |
| **Activación**       | Solo cuando `H(g) > umbral` (percentil 95 de entropía en val set)        |
| **Loss**             | MSE (reconstrucción pixel-a-pixel) + L1 opcional                         |
| **Fase entrenamiento** | Fase 3 (`fase3_train_cae.py`) — DESPUÉS de los 5 expertos de dominio  |
| **Checkpoint dir**   | `checkpoints/expert_05_cae/`                                             |
| **Pesos esperados**  | `weights_cae.pt`                                                         |

## Teoría del modelo

### Autoencoders para detección Out-of-Distribution (OOD)

Un autoencoder es una red neuronal que aprende la función identidad bajo restricción:
comprimir la entrada en una representación latente de menor dimensionalidad y luego
reconstruirla. La clave teórica es que el autoencoder solo puede reconstruir bien
datos que pertenecen a la distribución de entrenamiento.

```
            IN-DISTRIBUTION                    OUT-OF-DISTRIBUTION
    ┌──────────────────────────┐       ┌──────────────────────────┐
    │  Radiografía de tórax    │       │  Foto de un gato         │
    │         ↓                │       │         ↓                │
    │  Encoder → Latente       │       │  Encoder → Latente       │
    │         ↓                │       │         ↓                │
    │  Decoder → Reconstrucción│       │  Decoder → Reconstrucción│
    │         ↓                │       │         ↓                │
    │  Error BAJO (≈ original) │       │  Error ALTO (≠ original) │
    └──────────────────────────┘       └──────────────────────────┘
```

**¿Por qué CAE y no un clasificador binario (médico/no-médico)?**

Un clasificador binario requiere ejemplos negativos (imágenes no médicas) durante
entrenamiento. Pero el espacio de "cosas que no son imágenes médicas" es infinito:
fotos, texto, ruido, capturas de pantalla, imágenes médicas de modalidades no vistas...
Un clasificador nunca puede cubrir todo ese espacio. El CAE, en cambio, aprende solo la
distribución positiva (imágenes médicas válidas) y rechaza todo lo que no puede reconstruir,
sin necesitar definir explícitamente qué es "no médico".

### ¿Por qué convolucional?

Las CNNs tienen un sesgo inductivo de localidad y invarianza traslacional que es perfecto
para imágenes médicas: texturas de tejido, bordes anatómicos, y patrones de contraste son
features locales que se repiten en diferentes posiciones. Un autoencoder fully-connected
perdería esta estructura espacial y necesitaría órdenes de magnitud más parámetros.

### Rol del CAE en el sistema MoE

El CAE NO es un experto de clasificación como los expertos 0-4. Es un **filtro de
seguridad** que actúa como última línea de defensa cuando el router no tiene confianza
suficiente para asignar una imagen a ningún experto especializado. Su activación depende
exclusivamente de la entropía del vector de gates del router.

## Diagrama de arquitectura (ASCII)

```
                      ENTRADA: [B, 3, 224, 224]
                      (imagen médica 2D — cualquier modalidad)
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      ENCODER                            │
    │                                                         │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Conv2d(3, 64, kernel=3, stride=2, padding=1)     │  │
    │  │  BatchNorm2d(64)                                   │  │
    │  │  ReLU                                              │  │
    │  │  [B, 3, 224, 224] → [B, 64, 112, 112]             │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Conv2d(64, 128, kernel=3, stride=2, padding=1)   │  │
    │  │  BatchNorm2d(128)                                  │  │
    │  │  ReLU                                              │  │
    │  │  [B, 64, 112, 112] → [B, 128, 56, 56]             │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Conv2d(128, 256, kernel=3, stride=2, padding=1)  │  │
    │  │  BatchNorm2d(256)                                  │  │
    │  │  ReLU                                              │  │
    │  │  [B, 128, 56, 56] → [B, 256, 28, 28]              │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Flatten                                           │  │
    │  │  [B, 256, 28, 28] → [B, 200704]                   │  │
    │  │  (256 × 28 × 28 = 200,704)                        │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Linear(200704, 512)                               │  │
    │  │  [B, 200704] → [B, 512]                            │  │
    │  └───────────────────────────────────────────────────┘  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   ESPACIO LATENTE                        │
    │              z ∈ ℝ^512  (bottleneck)                    │
    │  Compresión: 224×224×3 = 150,528 → 512 (294:1)         │
    └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      DECODER                            │
    │                                                         │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Linear(512, 200704)                               │  │
    │  │  [B, 512] → [B, 200704]                            │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Unflatten → [B, 256, 28, 28]                      │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  ConvTranspose2d(256, 128, kernel=3, stride=2,    │  │
    │  │                  padding=1, output_padding=1)      │  │
    │  │  BatchNorm2d(128)                                  │  │
    │  │  ReLU                                              │  │
    │  │  [B, 256, 28, 28] → [B, 128, 56, 56]              │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  ConvTranspose2d(128, 64, kernel=3, stride=2,     │  │
    │  │                  padding=1, output_padding=1)      │  │
    │  │  BatchNorm2d(64)                                   │  │
    │  │  ReLU                                              │  │
    │  │  [B, 128, 56, 56] → [B, 64, 112, 112]             │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                         ▼                               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  ConvTranspose2d(64, 3, kernel=3, stride=2,       │  │
    │  │                  padding=1, output_padding=1)      │  │
    │  │  Sigmoid                                           │  │
    │  │  [B, 64, 112, 112] → [B, 3, 224, 224]             │  │
    │  └───────────────────────────────────────────────────┘  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      SALIDA: [B, 3, 224, 224]
                      (reconstrucción de la imagen)

              Loss = MSE(entrada, salida) + λ·L1(entrada, salida)
```

## Diagrama de integración MoE (ASCII)

Este es el diagrama más importante del CAE. Muestra el **árbol de decisión completo**
con los 3 caminos posibles desde que el router detecta alta entropía.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │                        PIPELINE MoE                              │
    │                                                                  │
    │  Imagen / Volumen de entrada                                     │
    │       │                                                          │
    │       ▼                                                          │
    │  Backbone compartido → CLS token z                               │
    │       │                                                          │
    │       ▼                                                          │
    │  Router: g = softmax(W·z + b) ∈ ℝ^6                             │
    │       │                                                          │
    │       ▼                                                          │
    │  Calcular entropía: H(g) = -Σ gᵢ·log(gᵢ)                       │
    │       │                                                          │
    │       ├────── H(g) ≤ umbral ──────────────────────────────────┐  │
    │       │       (entropía BAJA)                                 │  │
    │       │       Router confiado                                 │  │
    │       │              │                                        │  │
    │       │              ▼                                        │  │
    │       │       Experto argmax(g)                               │  │
    │       │       (Exp 0, 1, 2, 3, o 4)                          │  │
    │       │              │                                        │  │
    │       │              ▼                                        │  │
    │       │       Diagnóstico normal                              │  │
    │       │                                                       │  │
    │       └────── H(g) > umbral ──────────────┐                  │  │
    │               (entropía ALTA)              │                  │  │
    │               Router NO confiado           │                  │  │
    │               umbral = percentil 95        │                  │  │
    │               del H(g) en val set          │                  │  │
    │                                            │                  │  │
    └────────────────────────────────────────────│──────────────────┘  │
                                                 │                     │
                                                 ▼                     │
    ┌────────────────────────────────────────────────────────────────┐ │
    │                    EXPERTO 5: CAE                              │ │
    │                                                                │ │
    │  Recibe la imagen original (NO el CLS token)                   │ │
    │       │                                                        │ │
    │       ▼                                                        │ │
    │  Encoder → z_latente ∈ ℝ^512                                  │ │
    │       │                                                        │ │
    │       ▼                                                        │ │
    │  Decoder → Reconstrucción x̂                                   │ │
    │       │                                                        │ │
    │       ▼                                                        │ │
    │  Error = MSE(x, x̂)                                            │ │
    │       │                                                        │ │
    │       ▼                                                        │ │
    │  ┌─────────────────────────────────────────────────────────┐   │ │
    │  │                                                         │   │ │
    │  │   CAMINO 1: Error MUY ALTO (> umbral_basura)            │   │ │
    │  │   ─────────────────────────────────────────              │   │ │
    │  │   La imagen no pertenece a ninguna distribución          │   │ │
    │  │   médica conocida.                                      │   │ │
    │  │                                                         │   │ │
    │  │   Ejemplos: foto de gato, captura de pantalla,          │   │ │
    │  │   ruido aleatorio, documento escaneado                  │   │ │
    │  │                                                         │   │ │
    │  │              ──→  RESULTADO: "BASURA"                   │   │ │
    │  │                   Imagen rechazada                      │   │ │
    │  │                   No se procesa más                     │   │ │
    │  │                                                         │   │ │
    │  ├─────────────────────────────────────────────────────────┤   │ │
    │  │                                                         │   │ │
    │  │   CAMINO 2: Error MODERADO (≤ umbral_basura)            │   │ │
    │  │   ──────────────────────────────────────────             │   │ │
    │  │   La imagen ES médica pero estaba ruidosa/degradada.    │   │ │
    │  │   El CAE la reconstruye (efecto denoising).             │   │ │
    │  │                                                         │   │ │
    │  │              ──→  Re-routing al Router                  │   │ │
    │  │                   con imagen reconstruida x̂             │   │ │
    │  │                          │                              │   │ │
    │  │                          ▼                              │   │ │
    │  │                   Backbone → z' → Router → g'           │   │ │
    │  │                          │                              │   │ │
    │  │                   ┌──────┴──────┐                       │   │ │
    │  │                   │             │                       │   │ │
    │  │                   ▼             ▼                       │   │ │
    │  │            H(g') BAJA     H(g') SIGUE ALTA             │   │ │
    │  │               │                │                       │   │ │
    │  │               ▼                ▼                        │   │ │
    │  │         Clasificación    CAMINO 3 (abajo)              │   │ │
    │  │         normal por                                     │   │ │
    │  │         Exp 0–4                                        │   │ │
    │  │                                                         │   │ │
    │  ├─────────────────────────────────────────────────────────┤   │ │
    │  │                                                         │   │ │
    │  │   CAMINO 3: Re-routing fallido                          │   │ │
    │  │   ────────────────────────────                          │   │ │
    │  │   Después del denoising, el router SIGUE sin tener      │   │ │
    │  │   confianza. La imagen es médica legítima pero no       │   │ │
    │  │   encaja en ningún dominio conocido (modalidad no       │   │ │
    │  │   entrenada, patología atípica, caso límite).           │   │ │
    │  │                                                         │   │ │
    │  │              ──→  RESULTADO:                            │   │ │
    │  │                   "NECESITA REVISIÓN DE PROFESIONAL"    │   │ │
    │  │                   Flag para revisión humana             │   │ │
    │  │                                                         │   │ │
    │  └─────────────────────────────────────────────────────────┘   │ │
    │                                                                │ │
    └────────────────────────────────────────────────────────────────┘ │
                                                                      │
    ──────────────────────────────────────────────────────────────────-┘

    RESUMEN DE LOS 3 CAMINOS:
    ┌──────────────┬────────────────────────┬──────────────────────────────┐
    │ Camino       │ Condición              │ Acción                       │
    ├──────────────┼────────────────────────┼──────────────────────────────┤
    │ BASURA       │ Error reconstrucción   │ Rechazar imagen.             │
    │              │ MUY ALTO               │ No procesar.                 │
    ├──────────────┼────────────────────────┼──────────────────────────────┤
    │ Re-routing   │ Error MODERADO +       │ Clasificación por Exp 0–4    │
    │ exitoso      │ H(g') baja tras        │ con imagen reconstruida.     │
    │              │ reconstrucción         │                              │
    ├──────────────┼────────────────────────┼──────────────────────────────┤
    │ REVISIÓN     │ Error MODERADO +       │ Flag para revisión humana.   │
    │ PROFESIONAL  │ H(g') SIGUE alta tras  │ El sistema no puede decidir. │
    │              │ reconstrucción         │                              │
    └──────────────┴────────────────────────┴──────────────────────────────┘
```

## Hiperparámetros de entrenamiento

| Parámetro                | Valor                                                     |
|--------------------------|-----------------------------------------------------------|
| **Batch size**           | 32 (imágenes 2D)                                          |
| **Learning rate**        | 1e-3                                                      |
| **Optimizer**            | Adam (weight_decay=1e-5)                                  |
| **Scheduler**            | ReduceLROnPlateau (factor=0.5, patience=5)                |
| **Epochs**               | 100 (early stopping patience=15 sobre val MSE)            |
| **Precisión**            | FP32 (reconstrucción requiere precisión numérica)         |
| **Gradient checkpointing** | No necesario (modelo ligero)                           |
| **VRAM estimada**        | ~4 GB                                                     |
| **GPU mínima**           | Cualquier NVIDIA con ≥4 GB                                |
| **Loss primaria**        | MSE (reconstrucción pixel-a-pixel)                        |
| **Loss secundaria**      | L1 opcional (promueve reconstrucción más nítida)           |
| **Fase de entrenamiento** | Fase 3 — después de los 5 expertos de dominio            |
| **Datos**                | Muestreo balanceado de los 5 datasets                     |

## Consideraciones especiales

### Riesgo del router perezoso y L_error feedback

El problema más sutil del sistema MoE con filtro OOD es el **router perezoso**. Si el
router aprende que delegar imágenes al CAE nunca produce pérdida (porque el CAE siempre
"procesa" algo, ya sea rechazar o reconstruir), puede empezar a enviar cada vez más
imágenes válidas al CAE en lugar de clasificarlas directamente.

**Mecanismo de prevención: L_error feedback loop**

Durante la Fase 5 (fine-tuning global), la loss total incluye un término de penalización:

```
L_total = L_clasificación + β · L_error
```

Donde `L_error` mide cuántas imágenes médicas **válidas** fueron delegadas al CAE:

```
L_error = Σ  gate_CAE(xᵢ) · (1 - MSE(xᵢ, CAE(xᵢ)))
          i∈batch
```

- `gate_CAE(xᵢ)`: peso que el router asigna al CAE para la imagen i
- `1 - MSE(...)`: si la reconstrucción es buena (MSE bajo), este término es alto,
  penalizando al router por enviar una imagen que el CAE sabe reconstruir bien
  (porque es médica válida, no OOD)

Esto fuerza al router a reservar el CAE solo para imágenes genuinamente fuera de
distribución.

### Umbral de entropía: percentil 95

El umbral `H(g) > τ` se calibra como el **percentil 95 de H(g)** calculado sobre el
validation set completo. Esto significa que solo el 5% de las imágenes de validación
con mayor incertidumbre serían derivadas al CAE.

```python
# Calibración del umbral
import numpy as np

entropias_val = []
for x in val_loader:
    z = backbone(x)
    g = router(z)  # softmax gates
    H = -(g * torch.log(g + 1e-8)).sum(dim=-1)
    entropias_val.extend(H.cpu().numpy())

umbral = np.percentile(entropias_val, 95)
print(f"Umbral de entropía OOD: {umbral:.4f}")
```

¿Por qué percentil 95 y no otro valor?

- **Percentil 90:** demasiado agresivo — muchas imágenes válidas pero difíciles
  serían filtradas, reduciendo coverage del sistema.
- **Percentil 99:** demasiado permisivo — imágenes OOD moderadas pasarían al
  sistema de clasificación, produciendo predicciones erróneas con alta confianza.
- **Percentil 95:** compromiso empírico entre seguridad (filtrar OOD) y utilidad
  (no filtrar demasiadas imágenes legítimas).

### El CAE NO recibe routing directo

A diferencia de los expertos 0-4, el CAE **nunca** recibe asignación directa del
argmax del router. Su activación depende exclusivamente de la entropía:

```
Expertos 0-4: activados por argmax(g) cuando H(g) ≤ umbral
Experto 5:    activado por H(g) > umbral (independiente de argmax)
```

Esto es una distinción arquitectónica fundamental. El router tiene 6 salidas
(g ∈ ℝ^6), pero el índice 5 del softmax NO activa al CAE. Solo la entropía
del vector completo determina si el CAE interviene.

### FP32 para reconstrucción

A diferencia de los expertos 3D que usan FP16, el CAE usa FP32. La razón es que
la métrica de decisión (MSE de reconstrucción) requiere precisión numérica: una
diferencia de 0.001 en MSE puede ser la diferencia entre clasificar una imagen
como "BASURA" o "médica válida". Los errores de cuantización de FP16 pueden
introducir falsos positivos/negativos en el filtro OOD.

### Entrenamiento sobre los 5 datasets

El CAE debe aprender la distribución de TODAS las imágenes médicas del sistema,
no solo de un dominio. El muestreo durante entrenamiento es balanceado entre los
5 datasets para evitar que el CAE sobre-aprenda una modalidad:

| Dataset    | Modalidad          | Proporción en batch |
|------------|--------------------|---------------------|
| Chest      | Radiografía tórax  | 20%                 |
| ISIC       | Dermatoscopia      | 20%                 |
| OA         | Radiografía rodilla| 20%                 |
| LUNA16     | CT pulmonar 3D*    | 20%                 |
| Páncreas   | CT abdominal 3D*   | 20%                 |

\* Para datos 3D, se extraen slices 2D representativos o se usa una versión
3D del CAE con Conv3d / ConvTranspose3d equivalentes.

## Checkpoint

### Estructura del directorio

```
checkpoints/expert_05_cae/
├── model_card.md          ← este archivo
└── weights_cae.pt         ← pesos del modelo (pendiente de entrenamiento)
```

### Cargar el modelo

```python
import torch
import torch.nn as nn

# Asumiendo que CAE está definido en el proyecto
from models.expert_05 import ConvAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvAutoEncoder(
    in_channels=3,
    latent_dim=512,
    img_size=224,
)

checkpoint_path = "checkpoints/expert_05_cae/weights_cae.pt"
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Evaluar si una imagen es OOD
dummy_input = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    reconstruction = model(dummy_input)          # [1, 3, 224, 224]
    mse = ((dummy_input - reconstruction) ** 2).mean().item()

# Decisión
umbral_basura = 0.15   # calibrado empíricamente
umbral_ood    = 0.05   # percentil 95 del MSE en val set

if mse > umbral_basura:
    print("BASURA — imagen rechazada")
elif mse > umbral_ood:
    print("Imagen ruidosa — intentar re-routing con reconstrucción")
else:
    print("Imagen médica válida — error: no debería haber llegado al CAE")
```
