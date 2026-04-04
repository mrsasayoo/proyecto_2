```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │       Imagen o Volumen (PNG / JPEG / NIfTI) — sin metadatos              │
 └─────────────────────────────────┬────────────────────────────────────────┘
                                   │
          ┌────────────────────────▼──────────────────────────────────┐
          │         PREPROCESADOR ADAPTATIVO                          │
          │  Detección: rank=4 → 2D  │  rank=5 → 3D                   │
          │                                                           │
          │  Cadena interna:                                          │
          │   Resize → TVF → Gamma Correction → CLAHE                 │
          │     → Guardar transform → Tensor 5D normalizado           │
          │                                                           │
          │  2D: [B, 3, 224, 224]  (Norm ImageNet)                    │
          │  3D: [B, 1, 64, 64, 64] (Norm HU CT)                      │
          └────────────────────────┬──────────────────────────────────┘
                                   │ tensor 5D
          ┌────────────────────────▼──────────────────────────────────┐
          │         BACKBONES DISPONIBLES (compartidos)               │
          │                                                           │
          │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
          │  │  ViT-Tiny   │ │   CvT-13    │ │  Swin-Tiny  │          │
          │  │ desde cero  │ │ desde cero  │ │ desde cero  │          │
          │  │arq. predef. │ │arq. predef. │ │arq. predef. │          │
          │  └─────────────┘ └─────────────┘ └─────────────┘          │
          │  ┌──────────────────┐                                     │
          │  │  DenseNet custom │  ← arquitectura propia del proyecto │
          │  │   desde cero     │                                     │
          │  └──────────────────┘                                     │
          │                                                           │
          │  Patch Embedding → tokens [B, N, d_model]                 │
          │  Transformer Blocks + Self-Attention                      │
          │  Salida:  z = CLS token / embedding  ∈ R^d_model          │
          └───────────────────────┬───────────────────────────────────┘
                                  │ z

 ═════════════════════════════════════════════════════════════════════════════
  FASE 0 ── PREPARACIÓN DE DATOS
  Descarga → Extracción → Splits 80/10/10 → Parches 3D → Reporte
  Script: fase0_pipeline.py
 ═════════════════════════════════════════════════════════════════════════════

          Artefactos: splits (.txt, .csv, dirs) + parches .npy
                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FASE 1 ── EXTRACCIÓN DE EMBEDDINGS
  Backbone congelado ❄ → CLS tokens → Z_train, Z_val, Z_test (.npy en disco)
  Script: fase1_pipeline.py
 ═════════════════════════════════════════════════════════════════════════════

          Artefactos: Z_train.npy, Z_val.npy, Z_test.npy
                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FASE 2 ── ENTRENAMIENTO INDIVIDUAL DE LOS 5 EXPERTOS
  (Construidos desde cero — NO pesos preentrenados de HuggingFace/timm)
  Cada experto entrena por separado hasta convergencia.
  Script: fase2_train_experts.py
 ═════════════════════════════════════════════════════════════════════════════
                                  │
    ┌──────────┬──────────────────┼───────────────┬────────────┐
    ▼          ▼                  ▼               ▼            ▼
 ┌──────┐  ┌──────┐           ┌──────┐       ┌──────┐    ┌──────┐
 │ Exp0 │  │ Exp1 │           │ Exp2 │       │ Exp3 │    │ Exp4 │
 │Conv  │  │Effic │           │VGG16 │       │ViViT │    │Swin3D│
 │Next-T│  │Net-B3│           │  BN  │       │ Tiny │    │ Tiny │
 │ 2D   │  │ 2D   │           │ 2D   │       │ 3D   │    │ 3D   │
 │Chest │  │ ISIC │           │Knee  │       │LUNA16│    │Pancr.│
 │NIH   │  │ 2019 │           │  OA  │       │      │    │      │
 │14cls │  │ 9cls │           │ 3cls │       │ 2cls │    │ 2cls │
 │BCE   │  │ CE   │           │ CE   │       │Focal │    │Focal │
 └──┬───┘  └──┬───┘           └──┬───┘       └──┬───┘    └──┬───┘
    │         │                   │              │ ckpt ✓    │ ckpt ✓
    │         │                   │              │ FP16 ✓    │ FP16 ✓
    └─────────┴───────────────────┴──────────────┴───────────┘
                                  │
                      Freeze ALL expertos ❄
                                  │
          Guardar: weights_expN.pt  (×5)

                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FASE 3 ── ENTRENAMIENTO DEL EXPERTO 6 (CAE)
  Convolutional AutoEncoder — aprende distribución de imágenes médicas válidas
  Script: fase3_train_cae.py
 ═════════════════════════════════════════════════════════════════════════════
                                  │
          ┌───────────────────────▼───────────────────────────────────┐
          │         EXPERTO 6: CAE (Convolutional AutoEncoder)        │
          │                                                           │
          │  Datos: 5 datasets combinados (Chest+ISIC+OA+LUNA+Panc)   │
          │  Objetivo: aprender distribución de imgs médicas válidas  │
          │  Loss: MSE / L1 (error de reconstrucción)                 │
          │  NO recibe imágenes del router durante entrenamiento      │
          │                                                           │
          │  Encoder → Latent → Decoder                               │
          └───────────────────────┬───────────────────────────────────┘
                                  │
          Guardar: weights_cae.pt
                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FASE 4 ── ABLATION STUDY — Selección del Router
  Expertos 0–5 congelados ❄ — 4 routers compiten sobre Z_train / Z_val
  Script: fase4_ablation.py
 ═════════════════════════════════════════════════════════════════════════════
                                  │ z → Z_train, Z_val en disco
         ┌────────────────────────┼───────────────────┬─────────────────┐
         ▼                        ▼                   ▼                 ▼
  ┌────────────┐       ┌──────────────┐       ┌──────────────┐   ┌──────────────┐
  │  Linear    │       │     GMM      │       │  Naive Bayes │   │    k-NN      │
  │  Softmax   │       │  5 comp. EM  │       │  MLE analít. │   │  FAISS coseno│
  └─────┬──────┘       └──────┬───────┘       └──────┬───────┘   └──────┬───────┘
        └─────────────────────┴──────────────────────┴──────────────────┘
                                  │
                   Calibración: α ∈ [0.01, 0.1]
                   Selección por: Routing Accuracy (umbral >0.80)
                   Balance: max(f_i)/min(f_i) ≤ 1.30  (solo ViT+Linear)
                                  │
          Guardar: ablation_results.json
                   entropy_threshold.pkl
                   α_optimo
                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FASE 5 ── FINE-TUNING GLOBAL  (3 sub-fases)
  Cargar pesos desde fases anteriores
  L_total = L_task + α·L_aux + β·L_error
  max(f_i)/min(f_i) ≤ 1.30 (Exp0–Exp4) ← penalización −40% si se viola
  Script: fase5_finetune_global.py
 ═════════════════════════════════════════════════════════════════════════════


 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  Sub-fase A ── Entrenar SOLO Router  │  Exp0–5 congelados ❄
  LR = 1e-3  │  ~50 épocas
 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

          ┌──────────────────────────────────────────────┐
          │     ROUTER  ViT + Linear + Softmax  🔥       │ ← gradientes ON
          │     g = softmax(W·z + b)  ∈ R^6              │   LR = 1e-3
          │     L = L_task + α·L_aux + β·L_error         │
          └──────────────┬───────────────────────────────┘
       ┌────────┬────────┼────────┬────────┬──────────────────────┐
       ▼        ▼        ▼        ▼        ▼                      ▼
  [Exp0 ❄]  [Exp1 ❄] [Exp2 ❄] [Exp3 ❄] [Exp4 ❄]       ┌──────────────┐
  predict   predict  predict  predict  predict        │   Exp5 (CAE) │
  only      only     only     only     only           │  AutoEncoder │
                                                      │  ❄ congelado │
                                                      └──────┬───────┘
                                                             │ L_error
                                                  ┌──────────┘
                                                  │  penaliza al router si
                                                  │  delega entrada válida
                                                  ▼  a Exp5 (feedback loop)
                                              ROUTER ← ajusta pesos
                                              aprende a NO enviar aquí
                                              imágenes médicas válidas
                                  │
                                  ▼

 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  Sub-fase B ── Router + Cabezas clasificadoras  │  Capas conv ❄
  LR = 1e-4  │  ~30 épocas   (cargar Sub-fase A)
 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

          ┌──────────────────────────────────────────────┐
          │     ROUTER  ViT + Linear + Softmax  🔥       │ LR = 1e-4
          └──────────────┬───────────────────────────────┘
       ┌────────┬────────┼────────┬────────┐
       ▼        ▼        ▼        ▼        ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Exp0    │ │ Exp1    │ │ Exp2    │ │ Exp3    │ │ Exp4    │
  │ conv ❄  │ │ conv ❄  │ │ conv ❄  │ │ conv ❄  │ │ conv ❄  │
  │ head 🔥 │ │ head 🔥 │ │ head 🔥 │ │ head 🔥 │ │ head 🔥 │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
  (capas convolucionales congeladas, solo clasificadores entrenan)
                                  │
                                  ▼

 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  Sub-fase C ── Fine-tuning TODO 🔥  (cargar Sub-fase B)
  LR = 1e-6  │  7–10 épocas  │  ⚠ NO reiniciar optimizer
 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

          TODO ENTRENADO 🔥  (Backbone + Router + Exp0–5)
          L_total = L_task + α·L_aux + β·L_error
          • α ∈ [0.01, 0.1]  (balance de carga)
          • β calibrado para que Exp5 solo reciba OOD real
          • max(f_i)/min(f_i) ≤ 1.30  (Exp0–Exp4 balanceados)
          • Penalización: −40% si se viola

                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FLUJO DEL CAE EN INFERENCIA  (Experto 5 — activado por entropía)
 ═════════════════════════════════════════════════════════════════════════════

          Router produce g = softmax(W·z + b)
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              H(g) BAJA                   H(g) ALTA
            (alta confianza)          (> umbral, percentil 95)
                     │                         │
                     ▼                         ▼
            Experto argmax(g)         ┌─────────────────┐
            (Exp 0–4)                 │   Exp5: CAE     │
            Clasificación             │ Encoder→Latent  │
            normal                    │    →Decoder     │
                                      └────────┬────────┘
                                               │
                                      Error de reconstrucción
                                               │
                              ┌────────────────┴────────────────┐
                              ▼                                 ▼
                    Error MUY ALTO                      Error MODERADO
                              │                                 │
                              ▼                                 ▼
                    ┌─────────────────┐              CAE Denoising
                    │    "BASURA"     │              (reconstruye imagen)
                    │  (no es imagen  │                         │
                    │  médica: gato,  │                         ▼
                    │  ruido, etc.)   │              Re-routing al Router
                    └─────────────────┘                         │
                                               ┌────────────────┴──────────┐
                                               ▼                           ▼
                                        H(g) ahora BAJA            H(g) SIGUE ALTA
                                               │                           │
                                               ▼                           ▼
                                      Clasificación              ┌──────────────────┐
                                      normal                     │   "NECESITA      │
                                                                 │    REVISIÓN DE   │
                                                                 │  UN PROFESIONAL" │
                                                                 └──────────────────┘

          L_error feedback: penaliza al router si delega
          imágenes válidas al CAE (previene router perezoso)

                                  │
                                  ▼

 ═════════════════════════════════════════════════════════════════════════════
  FASE 6 ── DESPLIEGUE WEB + DASHBOARD
  Script: fase6_webapp.py
 ═════════════════════════════════════════════════════════════════════════════

          ┌───────────────────────────────────────────────────────────┐
          │              INFERENCIA  +  Dashboard                     │
          │                                                           │
          │  • Carga PNG / JPEG / NIfTI — detección 2D/3D automática  │
          │  • Preprocesado transparente (dims originales → adaptadas)│
          │  • Backbone genera z → Router delega al experto           │
          │                                                           │
          │  ┌─────────────────────────────────────────────────────┐  │
          │  │  Attention Heatmap Router ViT                       │  │
          │  │  Grad-CAM con transform guardado del preprocesador  │  │
          │  └─────────────────────────────────────────────────────┘  │
          │                                                           │
          │  ┌─────────────────────────────────────────────────────┐  │
          │  │  Panel Experto Activado                             │  │
          │  │  Nombre │ Arquitectura │ Dataset │ Gating Score     │  │
          │  └─────────────────────────────────────────────────────┘  │
          │                                                           │
          │  ┌─────────────────────────────────────────────────────┐  │
          │  │  Panel Ablation Study                               │  │
          │  │  Tabla 4 métodos: Linear │ GMM │ NB │ k-NN          │  │
          │  │  Routing Accuracy comparativa sobre val set         │  │
          │  └─────────────────────────────────────────────────────┘  │
          │                                                           │
          │  ┌─────────────────────────────────────────────────────┐  │
          │  │  Load Balance f_i en tiempo real                    │  │
          │  │  Barras por experto │ max/min ratio │ acumulado     │  │
          │  └─────────────────────────────────────────────────────┘  │
          │                                                           │
          │  ┌─────────────────────────────────────────────────────┐  │
          │  │  OOD Alert                                          │  │
          │  │  H(g) > umbral → CAE activo → "BASURA" o "REVISIÓN" │  │
          │  └─────────────────────────────────────────────────────┘  │
          │                                                           │
          │  Artefactos consumidos:                                   │
          │    ablation_results.json │ entropy_threshold.pkl          │
          │    weights_exp0–4.pt │ weights_cae.pt │ weights_router.pt │
          │    α_optimo │ preprocessing_transform.pkl                 │
          └───────────────────────────────────────────────────────────┘
```
