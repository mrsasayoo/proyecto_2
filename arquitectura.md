```
 ┌──────────────────────────────────────────────────────────────────────┐
 │      Imagen o Volumen (PNG / JPEG / NIfTI) — sin metadatos           │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────────────┐
          │        PREPROCESADOR ADAPTATIVO                    │
          │   • rank=4 → 2D  [B, 3, 224, 224]  (Norm ImageNet) │
          │   • rank=5 → 3D  [B, 1, 64, 64, 64] (Norm HU CT)   │
          └─────────────────────┬──────────────────────────────┘
                                │ patches
          ┌─────────────────────▼──────────────────────────────┐
          │        BACKBONE ViT-Tiny  (compartido)             │
          │   Patch Embedding → tokens [B, N, d_model]         │
          │   Transformer Blocks + Self-Attention              │
          │   CLS token:  z ∈ R^d_model                        │
          └──────────────────┬─────────────────────────────────┘
                             │ z (mismo embedding para todo)
 ═══════════════════════════════════════════════════════════════════════
  FASE 0 ── PRE-ENTRENAMIENTO INDIVIDUAL DE EXPERTOS
  (Pesos descargados de HuggingFace/timm, fine-tuned por separado
   con sus respectivos datasets antes de cualquier MoE)
 ═══════════════════════════════════════════════════════════════════════
                             │
    ┌──────────┬─────────────┼────────────┬────────────┐
    ▼          ▼             ▼            ▼            ▼
 ┌──────┐  ┌──────┐      ┌──────┐    ┌──────┐    ┌──────┐
 │ Exp1 │  │ Exp2 │      │ Exp3 │    │ Exp4 │    │ Exp5 │
 │Conv  │  │Effic │      │VGG16 │    │ViViT │    │Swin3D│
 │Next-T│  │Net-B3│      │  BN  │    │ Tiny │    │ Tiny │
 │ 2D   │  │ 2D   │      │ 2D   │    │ 3D   │    │ 3D   │
 │Chest │  │ ISIC │      │Knee  │    │LUNA16│    │Pancr.│
 │14cls │  │ 9cls │      │ 3cls │    │ 2cls │    │ 2cls │
 │BCE   │  │ CE   │      │ CE   │    │Focal │    │Focal │
 └──┬───┘  └──┬───┘      └──┬───┘    └──┬───┘    └──┬───┘
    │         │              │          │ ckpt ✓    │ ckpt ✓
    └─────────┴──────────────┴──────────┴───────────┘
    Guardar: [weights_expN.pt + optimizer_expN.pt]  (×5)

 ═══════════════════════════════════════════════════════════════════════
  ABLATION STUDY ── Selección del Router
  (ViT extrae CLS tokens → 4 cabezas compiten sobre Z_train / Z_val)
 ═══════════════════════════════════════════════════════════════════════
                             │ z → guardar Z_train, Z_val en disco
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                     ▼                ▼
 ┌────────────┐    ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
 │  Linear    │    │     GMM      │    │  Naive Bayes │   │    k-NN      │
 │  Softmax   │    │  5 comp. EM  │    │  MLE analít. │   │  FAISS coseno│
 │ ~88% acc   │    │  ~83% acc    │    │  ~78% acc    │   │  ~82% acc    │
 └─────┬──────┘    └──────┬───────┘    └──────┬───────┘   └──────┬───────┘
       └──────────────────┴───────────────────┴──────────────────┘
                                     │
                          GANADOR → Linear (~88%)

 ═══════════════════════════════════════════════════════════════════════
  FASE 1 ── Entrenar SOLO el Router  (Expertos CONGELADOS ❄)
  Guardar: [weights_router_f1.pt + optimizer_f1.pt]   ~50 épocas
 ═══════════════════════════════════════════════════════════════════════

          ┌──────────────────────────────────────────┐
          │    ROUTER  ViT + Linear + Softmax  🔥    │ ← gradientes ON
          │    g = softmax(W·z + b)  ∈ R^6           │   LR = 1e-3
          │    L = L_task + α·L_aux + β·L_error      │
          └──────────────┬───────────────────────────┘
       ┌────────┬────────┼────────┬────────┬───────────────────┐
       ▼        ▼        ▼        ▼        ▼                   ▼
  [Exp1 ❄]  [Exp2 ❄] [Exp3 ❄] [Exp4 ❄] [Exp5 ❄]     ┌──────────────┐
  predict   predict  predict  predict  predict      │   Exp6       │
  only      only     only     only     only         │  ERROR /OOD  │
                                                    │  MLP simple  │
                                                    │  sin pesos   │
                                                    │  de dominio  │
                                                    └──────┬───────┘
                                                           │ L_error
                                              ┌────────────┘
                                              │  penaliza al router si
                                              │  delega entrada válida
                                              ▼  a Exp6 (feedback loop)
                                          ROUTER ← ajusta pesos
                                          aprende a NO enviar aquí
                                          imágenes médicas válidas

 ═══════════════════════════════════════════════════════════════════════
  FASE 2 ── Descongelar cabezas clasificadoras  (cargar FASE 1)
  Guardar: [weights_all_f2.pt + optimizer_f2.pt]   ~30 épocas
 ═══════════════════════════════════════════════════════════════════════

          ┌──────────────────────────────────────────┐
          │    ROUTER  ViT + Linear + Softmax  🔥    │ LR = 1e-4
          └──────────────┬───────────────────────────┘
       ┌────────┬────────┼────────┬────────┐
       ▼        ▼        ▼        ▼        ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Exp1    │ │ Exp2    │ │ Exp3    │ │ Exp4    │ │ Exp5    │
  │ conv ❄  │ │ conv ❄  │ │ conv ❄  │ │ conv ❄  │ │ conv ❄  │
  │ head 🔥 │ │ head 🔥 │ │ head 🔥 │ │ head 🔥 │ │ head 🔥 │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
  (capas convolucionales congeladas, solo clasificadores entrenan)

 ═══════════════════════════════════════════════════════════════════════
  FASE 3 ── Fine-tuning Global  (cargar FASE 2 weights + optimizer)
  7-10 épocas · LR = 1e-6 · estabiliza pesos router ↔ conv ↔ head
  ⚠ NO reiniciar optimizer — se pierden los pasos anteriores
 ═══════════════════════════════════════════════════════════════════════

          TODO ENTRENADO 🔥  (ViT backbone + Router + Exp1-5)
          L_total = L_task + α·L_aux + β·L_error
          • α ∈ [0.01, 0.1]  (balance de carga)
          • β calibrado para que Exp6 solo reciba OOD real
          • max(f_i)/min(f_i) < 1.30  (Exp1–Exp5 balanceados)

                             │
          ┌──────────────────▼────────────────────────────────┐
          │          INFERENCIA  +  Dashboard                 │
          │  • Preprocesador 2D/3D automático                 │
          │  • ViT genera z → Router delega al experto        │
          │  • Exp6 activa → OOD alert + L_error registrada   │
          │  • Attention Heatmap sobre imagen original        │
          │  • Load Balance f_i por experto en tiempo real    │
          │  • Gating score + confianza por clase             │
          └───────────────────────────────────────────────────┘
```