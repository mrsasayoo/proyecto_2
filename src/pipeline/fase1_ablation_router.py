"""
FASE 1 — Ablation Study del Router
====================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II
 
Lee Z_train.npy / Z_val.npy generados por FASE 0 y compara los 4 mecanismos
de routing sobre el mismo espacio de embeddings.
 
Arquitectura MoE (6 expertos — diseño propio):
  Expertos de dominio (ID 0–4) — targets del ablation study:
    0 → NIH ChestXray14   4 → Pancreas PANORAMA
    1 → ISIC 2019         5 → OOD/Error (ver nota abajo)
    2 → OA Rodilla
    3 → LUNA16
 
  Experto 5 (OOD/Error) — NO es un target de routing del ablation study.
    El router aprende a enviar imágenes a los expertos 0–4. El Experto 5 se
    activa en inferencia por umbral de entropía: H(g) ≥ ENTROPY_THRESHOLD.
    Este script calibra ese umbral analizando H(g) en el set de validación.
 
  LinearGatingHead en el ablation: salida = N_EXPERTS_DOMAIN = 5 logits.
  LinearGatingHead en FASE 1 real: salida = N_EXPERTS_TOTAL  = 6 logits
    (el slot 5 recibe L_error durante el entrenamiento del router).
 
Mecanismos comparados (sección 4.2 del proyecto):
  A) ViT + Linear + Softmax  — paramétrico, gradiente
  B) ViT + GMM               — paramétrico, EM
  C) ViT + Naive Bayes       — paramétrico, MLE analítico
  D) ViT + k-NN FAISS        — no paramétrico, distancia coseno
 
Uso:
  python fase1_ablation_router.py --embeddings ./embeddings --epochs 50
"""
 
import argparse
import json
import logging
import os
import time
from pathlib import Path
 
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
 
 
# ──────────────────────────────────────────────────────────
# 0. LOGGING
# ──────────────────────────────────────────────────────────
 
def setup_logging(output_dir: str) -> logging.Logger:
    """
    Escribe simultáneamente a consola (INFO) y a fase1_ablation.log (DEBUG).
    Niveles usados:
      DEBUG   → detalles internos (distribución por época, etc.)
      INFO    → progreso normal del ablation
      WARNING → anomalía no fatal (GMM diverge, entropía inesperada, etc.)
      ERROR   → fallo grave que invalida resultados
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / "fase1_ablation.log"
 
    fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
 
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    logging.getLogger().handlers[1].setLevel(logging.INFO)
 
    log = logging.getLogger("fase1")
    log.info(f"Log iniciado → {log_path}")
    return log
 
 
log = logging.getLogger("fase1")


# ──────────────────────────────────────────────────────────
# 1. CONSTANTES DE ARQUITECTURA
# ──────────────────────────────────────────────────────────
 
# Número de expertos de dominio (los que tienen dataset → targets del router)
N_EXPERTS_DOMAIN = 5   # Chest, ISIC, OA, LUNA, Pancreas
 
# Número total de expertos en la arquitectura final (incluye OOD)
N_EXPERTS_TOTAL  = 6   # + Experto 5 OOD/Error
 
# Nombres de los expertos de dominio para logs y tabla
EXPERT_NAMES = {
    0: "Chest",
    1: "ISIC",
    2: "OA-Knee",
    3: "LUNA16",
    4: "Pancreas",
    # 5: "OOD" ← activado por entropía, no por routing directo
}
 
# Umbral de entropía para OOD detection (se calibra automáticamente en este script)
# H(g) = -sum(g_i * log(g_i + eps))
# Si H(g) >= ENTROPY_THRESHOLD → imagen va al Experto 5 (OOD)
# Valor inicial conservador; se ajusta en la sección de calibración
ENTROPY_THRESHOLD_DEFAULT = 1.4   # ln(5) ≈ 1.609 = entropía máxima uniforme en 5 clases
 
# ── Notas por experto para el JSON de resultados ────────────
# Estas notas se insertan en ablation_results.json para guiar FASE 2.
EXPERT_NOTES = {
    0: ("NIH ChestXray14 — MULTI-LABEL (14 patologías). "
        "Loss: BCEWithLogitsLoss con pos_weight por clase. "
        "NO usar CrossEntropyLoss. "
        "Hallazgo 3: etiquetas por NLP, benchmark DenseNet-121 ≈ 0.81 AUC macro. "
        "AUC > 0.85 significativo → revisar por confounding."),
    1: "ISIC 2019 — MULTICLASE (8 clases en train, UNK solo en test). Loss: CrossEntropyLoss con pesos.",
    2: "OA Knee — ORDINAL (3 clases KL). Loss: CrossEntropyLoss + QWK como métrica principal.",
    3: "LUNA16 — BINARIO parches 3D. Loss: FocalLoss(gamma=2). Métrica: CPM + curva FROC.",
    4: "Pancreas PANORAMA — BINARIO volumen 3D. Loss: FocalLoss(alpha=0.75, gamma=2). Métrica: AUC-ROC.",
}


# ──────────────────────────────────────────────────────────
# 2. UTILIDADES COMPARTIDAS
# ──────────────────────────────────────────────────────────


# Esta función calcula la Entropía de Shannon de las probabilidades de salida del router. Es el medidor de 
# "incertidumbre" del sistema: mide qué tan repartida está la decisión del router. Una entropía alta significa 
# que el router está confundido y no sabe a qué experto enviar la imagen, lo cual dispara la alerta de 
# detección de OOD (Out-of-Distribution).
def compute_entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Entropía de Shannon por muestra: H(g) = -sum(g_i * log(g_i + eps))
    Rango: [0, ln(N_EXPERTS_DOMAIN)]
      H=0   → router completamente seguro (toda la masa en un experto)
      H=max → router completamente inseguro (distribución uniforme)

    Umbral OOD: si H(g) >= ENTROPY_THRESHOLD → Experto 5 (OOD)
    """
    return -(probs * np.log(probs + eps)).sum(axis=1)


# Esta función calcula la precisión (Accuracy) de enrutamiento por experto. Su función es desglosar el 
# rendimiento global del router en métricas individuales para cada dataset (Chest, ISIC, etc.), permitiéndote 
# detectar si el router está ignorando sistemáticamente a un experto o si tiene dificultades para distinguir 
# una modalidad específica, lo cual es crítico para cumplir con el balance de carga exigido por el proyecto.
def per_expert_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Accuracy de routing por experto de dominio.
    Útil para detectar si el router ignora sistemáticamente un experto.
    Objetivo del proyecto: max
    (f_i)/min(f_i) < 1.30 en FASE 1 real.
    """
    result = {}
    for exp_id, exp_name in EXPERT_NAMES.items():
        mask = y_true == exp_id
        if mask.sum() == 0:
            result[exp_name] = None
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        result[exp_name] = acc
    return result
# Consejo para tu reporte:
# Puedes generar una tabla en tu dashboard o reporte que diga:
# | Experto | Routing Accuracy |
# | :--- | :--- |
# | ChestXray14 | 0.92 |
# | ISIC 2019 | 0.88 |
# | ... | ... |


# Esta función actúa como un visualizador de rendimiento en consola. Toma los resultados de la métrica de 
# precisión calculada para cada experto y los despliega mediante un gráfico de barras simple (ASCII), 
# permitiéndote identificar de un vistazo qué expertos están rindiendo bien y cuáles están siendo ignorados 
# por el router.
def log_per_expert(tag: str, acc_dict: dict) -> None:
    """Imprime la tabla de accuracy por experto en el log."""
    log.info(f"  [{tag}] Accuracy por experto de dominio:")
    for name, acc in acc_dict.items():
        if acc is None:
            log.warning(f"    {name:<12}: sin muestras en val")
        else:
            bar = "█" * int(acc * 20)
            log.info(f"    {name:<12}: {acc:.4f}  {bar}")


# Esta función calcula el Cociente de Desbalanceo de Carga. Su rol es verificar matemáticamente si el Router 
# está distribuyendo las imágenes equitativamente entre los 5 expertos de dominio. Si la diferencia entre el 
# experto más usado y el menos usado supera el 30% (ratio > 1.30), la función emite una advertencia para que p
# uedas ajustar los hiperparámetros de entrenamiento.
def check_load_balance(y_pred: np.ndarray, tag: str) -> float:
    """
    Verifica que el router no ignore expertos.
    Objetivo del proyecto: max(f_i)/min(f_i) < 1.30
    Retorna el ratio para incluirlo en la tabla comparativa.
    """
    counts = np.array([(y_pred == i).sum() for i in range(N_EXPERTS_DOMAIN)], dtype=float)
    counts = np.maximum(counts, 1)   # evitar división por cero
    ratio  = counts.max() / counts.min()
    if ratio > 1.30:
        log.warning(f"  [{tag}] Balance de carga: max/min = {ratio:.2f}x — supera el objetivo 1.30x. "
                    f"Distribución: {counts.astype(int).tolist()}")
    else:
        log.info(f"  [{tag}] Balance de carga: max/min = {ratio:.2f}x ✓  "
                 f"Distribución: {counts.astype(int).tolist()}")
    return float(ratio)
# Consejo para el reporte:
# En tu reporte técnico (Sección 5: Balance de Carga), puedes incluir un gráfico o una tabla que muestre cómo 
# el ratio max/min descendió a medida que ajustaste la Auxiliary Loss durante tus pruebas. Esto demostrará que 
# no solo implementaste el MoE, sino que lo controlaste científicamente.


# Esta función es el "Calibrador de Incertidumbre". Su rol es determinar estadísticamente el nivel de confusión 
# (entropía) a partir del cual el Router debe dejar de confiar en su propia predicción y declarar una imagen 
# como "Fuera de Dominio" (OOD). Utiliza una estrategia de percentil 95 sobre los datos de validación para 
# establecer un umbral de seguridad.
def calibrate_entropy_threshold(probs: np.ndarray, y_true: np.ndarray, tag: str) -> float:
    """
    Calibra el umbral de entropía para el Experto 5 (OOD) analizando la
    distribución de H(g) en el set de validación.

    Estrategia: el umbral es el percentil 95 de H(g) sobre muestras bien
    clasificadas. Esto significa que el 5% de las muestras más confusas del
    router (aunque sean dominio conocido) se tratarán como OOD en inferencia.

    El valor se guarda en ablation_results.json para usarlo en FASE 1 real.
    """
    entropies = compute_entropy(probs)
    correct_mask = (probs.argmax(axis=1) == y_true)

    h_correct   = entropies[correct_mask]
    h_incorrect = entropies[~correct_mask]

    threshold = float(np.percentile(entropies, 95))

    log.info(f"  [{tag}] Calibración entropía OOD (Experto 5):")
    log.info(f"    H(g) media (correctas)  : {h_correct.mean():.4f}")
    log.info(f"    H(g) media (incorrectas): {h_incorrect.mean():.4f}")
    log.info(f"    H(g) máxima posible     : {np.log(N_EXPERTS_DOMAIN):.4f}  (uniforme)")
    log.info(f"    ENTROPY_THRESHOLD (p95) : {threshold:.4f}")
    log.info(f"    → Muestras que irían a Experto 5 OOD: "
             f"{(entropies >= threshold).sum()}/{len(entropies)} "
             f"({100*(entropies >= threshold).mean():.1f}%)")

    if threshold < 0.3:
        log.warning(f"  [{tag}] Umbral muy bajo ({threshold:.4f}). El router es muy seguro "
                    f"o los embeddings están colapsados. Verifica con otra semilla.")
    return threshold


# ──────────────────────────────────────────────────────────
# 3A. LINEAR + SOFTMAX (baseline deep learning)
# ──────────────────────────────────────────────────────────


# En el ablation study: salida = N_EXPERTS_DOMAIN = 5 logits
# En FASE 1 real (entrenamiento MoE completo): salida = N_EXPERTS_TOTAL = 6
#   El slot 5 (OOD) aprende mediante L_error — no tiene labels propios.
#   Se pasa n_experts=N_EXPERTS_TOTAL al instanciar en FASE 1 real.


# Esta clase define la "Cabeza de Decisión" (Gating Head) del Router. Su función es recibir un vector de 
# características (embedding) del backbone y transformarlo mediante una capa lineal en un vector de 
# probabilidades que indica qué experto debe procesar la entrada.
class LinearGatingHead(nn.Module):
    """
    Cabeza de gating lineal.
    n_experts=5  → ablation study (solo dominios clínicos)
    n_experts=6  → FASE 1 real   (incluye slot OOD entrenado via L_error)
    """
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(z), dim=-1)   # [B, n_experts]


# Esta función implementa el bucle de entrenamiento (training loop) para el router neuronal. Utiliza descenso 
# de gradiente (vía Adam) y CrossEntropyLoss para aprender a clasificar los embeddings del backbone en los 5 
# expertos clínicos, guardando automáticamente el mejor modelo alcanzado durante la validación para asegurar 
# que el resultado del estudio de ablación sea óptimo.
def train_linear_router(Z_train, y_train, Z_val, y_val, d_model,
                        epochs=50, lr=1e-3, batch_size=512):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    # Ablation: solo 5 expertos de dominio
    model   = LinearGatingHead(d_model, N_EXPERTS_DOMAIN).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Z_t = torch.from_numpy(Z_train).float().to(device)
    y_t = torch.from_numpy(y_train).long().to(device)

    best_acc     = 0.0
    best_weights = None
    no_improve   = 0   # contador de épocas sin mejora (early stopping suave)

    log.info(f"  [Linear] Entrenando en {device} | epochs={epochs} lr={lr} batch={batch_size}")
    log.info(f"  [Linear] n_experts={N_EXPERTS_DOMAIN} (ablation) | "
             f"nota: FASE 1 real usará n_experts={N_EXPERTS_TOTAL} (+OOD slot)")

    for epoch in range(epochs):
        model.train()
        idx        = torch.randperm(len(Z_t))
        epoch_loss = 0.0

        for i in range(0, len(Z_t), batch_size):
            batch_idx = idx[i:i + batch_size]
            z_b, y_b  = Z_t[batch_idx], y_t[batch_idx]
            logits    = model.gate(z_b)
            loss      = loss_fn(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                Z_v   = torch.from_numpy(Z_val).float().to(device)
                probs = model(Z_v).cpu().numpy()
                preds = probs.argmax(axis=1)
                acc   = accuracy_score(y_val, preds)

            improved = acc > best_acc
            if improved:
                best_acc     = acc
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            log.info(f"  [Linear] época {epoch+1:3d}/{epochs} | "
                     f"loss {epoch_loss:.4f} | val acc {acc:.4f}"
                     + (" ✓ mejor" if improved else ""))

            # Detectar colapso: loss cae a 0 rápido o acc estancada desde el inicio
            if epoch_loss < 1e-6:
                log.warning(f"  [Linear] Loss ≈ 0 en época {epoch+1} — posible overfitting "
                            f"o etiquetas con fuga de información.")

    if best_weights is None:
        log.error("[Linear] best_weights es None — ninguna época de validación mejoró. "
                  "Revisa que y_train tenga las 5 clases representadas.")
        best_weights = model.state_dict()

    model.load_state_dict(best_weights)

    # Métricas finales
    model.eval()
    with torch.no_grad():
        Z_v   = torch.from_numpy(Z_val).float().to(device)
        probs = model(Z_v).cpu().numpy()
        preds = probs.argmax(axis=1)

    per_exp   = per_expert_accuracy(y_val, preds)
    log_per_expert("Linear", per_exp)
    balance   = check_load_balance(preds, "Linear")
    threshold = calibrate_entropy_threshold(probs, y_val, "Linear")

    return model, best_acc, probs, balance, threshold
# Un consejo pro para el ablation study:
# Como estás guardando los resultados en best_weights, asegúrate de incluir en tu reporte un gráfico de "Pérdida 
# vs. Épocas" (Loss curves). Si el router Lineal converge muy rápido (ej. en 20 épocas), puedes mencionarlo como 
# evidencia de que el espacio latente del backbone (ViT/Swin) ya es linealmente separable, lo cual es un 
# indicador de que el backbone ha extraído características de muy alta calidad.


# ──────────────────────────────────────────────────────────
# 3B. GMM (paramétrico EM)
# ──────────────────────────────────────────────────────────


# El GMM ajusta N_EXPERTS_DOMAIN=5 componentes gaussianas sin supervisión.
# Luego mapea cada componente al experto de dominio por voto mayoritario.
# El Experto 5 (OOD) no es un componente GMM — es entropía en inferencia.


# Esta función implementa un modelo de mezclas gaussianas (GMM) para aprender la distribución de los embeddings 
# en el espacio latente. A diferencia de una red neuronal que "aprende" etiquetas, el GMM "agrupa" los datos de 
# forma no supervisada en 5 nubes (clusters) y luego los asigna estadísticamente al experto de dominio 
# correspondiente mediante votación mayoritaria.
def train_gmm_router(Z_train, y_train, Z_val, y_val):
    log.info(f"  [GMM] Ajustando GaussianMixture "
             f"({N_EXPERTS_DOMAIN} comp., full covariance) ...")

    cov_type = "full"
    try:
        gmm = GaussianMixture(
            n_components=N_EXPERTS_DOMAIN,
            covariance_type="full",
            max_iter=200,
            random_state=42,
            verbose=0
        )
        gmm.fit(Z_train)
        if not gmm.converged_:
            log.warning("  [GMM] EM no convergió en 200 iteraciones con full covariance. "
                        "Considera aumentar max_iter o cambiar a 'diag'.")
    except Exception as e:
        log.warning(f"  [GMM] full covariance falló ({e}) → reintentando con 'diag'")
        cov_type = "diag"
        gmm = GaussianMixture(
            n_components=N_EXPERTS_DOMAIN,
            covariance_type="diag",
            max_iter=200,
            random_state=42
        )
        gmm.fit(Z_train)
        if not gmm.converged_:
            log.warning("  [GMM] EM tampoco convergió con 'diag'. Resultados poco confiables.")

    log.debug(f"  [GMM] covariance_type usado: '{cov_type}'")

    # Mapeo componente → experto de dominio por voto mayoritario
    train_comp     = gmm.predict(Z_train)
    comp_to_expert = {}
    for comp in range(N_EXPERTS_DOMAIN):
        mask = train_comp == comp
        if mask.sum() == 0:
            log.warning(f"  [GMM] Componente {comp} vacía en train — mapeada a experto 0.")
            comp_to_expert[comp] = 0
            continue
        labels = y_train[mask]
        winner = int(np.bincount(labels, minlength=N_EXPERTS_DOMAIN).argmax())
        comp_to_expert[comp] = winner

    # Verificar que todos los expertos de dominio tienen al menos una componente asignada
    assigned_experts = set(comp_to_expert.values())
    unassigned = set(range(N_EXPERTS_DOMAIN)) - assigned_experts
    if unassigned:
        log.warning(f"  [GMM] Expertos sin componente asignada: "
                    f"{[EXPERT_NAMES[e] for e in unassigned]}. "
                    f"El GMM nunca rutará a estos expertos.")

    log.info(f"  [GMM] Mapeo componente→experto: "
             + " | ".join(f"C{c}→{EXPERT_NAMES[e]}" for c, e in comp_to_expert.items()))

    val_comp  = gmm.predict(Z_val)
    val_preds = np.array([comp_to_expert[c] for c in val_comp])
    acc       = accuracy_score(y_val, val_preds)
    log.info(f"  [GMM] Routing Accuracy val: {acc:.4f}")

    # Aproximar probabilidades de pertenencia para análisis de entropía
    val_probs    = gmm.predict_proba(Z_val)   # [N_val, N_EXPERTS_DOMAIN]
    per_exp      = per_expert_accuracy(y_val, val_preds)
    log_per_expert("GMM", per_exp)
    balance      = check_load_balance(val_preds, "GMM")
    threshold    = calibrate_entropy_threshold(val_probs, y_val, "GMM")

    return gmm, comp_to_expert, acc, val_probs, balance, threshold
# Sugerencia de ingeniería:
# El random_state=42 es una decisión excelente. En un proyecto académico, esto garantiza que cada vez que 
# ejecutes el ablation study, el GMM encontrará los mismos clústeres, haciendo que tus resultados sean 
# reproducibles. ¡Bien hecho!


# ──────────────────────────────────────────────────────────
# 3C. NAIVE BAYES (paramétrico MLE analítico)
# ──────────────────────────────────────────────────────────


# Esta función implementa un clasificador probabilístico basado en el Teorema de Bayes. Su función es estimar 
# la probabilidad de que una imagen pertenezca a un experto asumiendo que las características del embedding 
# (sus 192 o 768 dimensiones) son independientes entre sí, calculando medias y varianzas de forma analítica 
# para cada experto.
def train_nb_router(Z_train, y_train, Z_val, y_val):
    log.info("  [NB] Ajustando GaussianNB (MLE analítico)...")

    # Verificar que todas las clases de dominio estén representadas en train
    classes_present = np.unique(y_train)
    missing = set(range(N_EXPERTS_DOMAIN)) - set(classes_present.tolist())
    if missing:
        log.warning(f"  [NB] Clases ausentes en train: "
                    f"{[EXPERT_NAMES[m] for m in missing]}. "
                    f"GaussianNB no podrá aprender estos expertos.")

    nb = GaussianNB()
    nb.fit(Z_train, y_train)

    val_preds = nb.predict(Z_val)
    val_probs = nb.predict_proba(Z_val)   # [N_val, N_EXPERTS_DOMAIN]
    acc       = accuracy_score(y_val, val_preds)
    log.info(f"  [NB] Routing Accuracy val: {acc:.4f}")

    per_exp   = per_expert_accuracy(y_val, val_preds)
    log_per_expert("NB", per_exp)
    balance   = check_load_balance(val_preds, "NB")
    threshold = calibrate_entropy_threshold(val_probs, y_val, "NB")

    return nb, acc, val_probs, balance, threshold
# Consejo: 
# Si el Naive Bayes supera en precisión a tu red neuronal Lineal, es una señal clara de que tu red neuronal 
# necesita más épocas de entrenamiento o que el espacio de embeddings está muy bien estructurado de forma 
# lineal. ¡Es una comparación excelente para tu tesis!


# ──────────────────────────────────────────────────────────
# 3D. k-NN con FAISS (no paramétrico, distancia coseno)
# ──────────────────────────────────────────────────────────


# Esta función implementa un clasificador basado en vecinos cercanos utilizando la librería FAISS 
# (Facebook AI Similarity Search). Su rol es buscar en el conjunto de entrenamiento las k imágenes cuyos 
# embeddings son más similares (usando similitud coseno) a la nueva imagen, y dejar que estos "vecinos" voten 
# por mayoría para decidir qué experto debe procesar la entrada.
def train_knn_router(Z_train, y_train, Z_val, y_val, k=5):
    log.info(f"  [kNN] Construyendo índice FAISS coseno (k={k})...")
    d = Z_train.shape[1]

    # Normalizar para usar Inner Product como coseno
    Z_t_norm = Z_train.copy().astype(np.float32)
    faiss.normalize_L2(Z_t_norm)
    Z_v_norm = Z_val.copy().astype(np.float32)
    faiss.normalize_L2(Z_v_norm)

    # Verificar que los vectores no sean cero (causaría NaN tras normalize_L2)
    zero_train = (np.linalg.norm(Z_train, axis=1) < 1e-9).sum()
    zero_val   = (np.linalg.norm(Z_val,   axis=1) < 1e-9).sum()
    if zero_train or zero_val:
        log.error(f"  [kNN] Vectores cero detectados: "
                  f"{zero_train} en train, {zero_val} en val. "
                  f"normalize_L2 producirá NaN. Verifica la extracción de embeddings.")

    index = faiss.IndexFlatIP(d)
    index.add(Z_t_norm)
    log.debug(f"  [kNN] Índice construido con {index.ntotal:,} vectores")

    # Buscar k vecinos y hacer voto mayoritario
    distances, I = index.search(Z_v_norm, k)   # [N_val, k]

    # Detectar búsquedas con similitud muy baja (posibles OOD naturales)
    min_sim = distances.max(axis=1)   # similitud con el vecino más cercano
    low_sim_count = (min_sim < 0.5).sum()
    if low_sim_count > 0:
        log.info(f"  [kNN] {low_sim_count} muestras de val con similitud coseno < 0.5 "
                 f"con su vecino más cercano — candidatas naturales a OOD.")

    neighbor_labels = y_train[I]   # [N_val, k]
    val_preds = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=N_EXPERTS_DOMAIN).argmax(),
        axis=1, arr=neighbor_labels
    )

    # Construir probabilidades suaves (fracción de votos por experto) para análisis entropía
    val_probs = np.zeros((len(Z_val), N_EXPERTS_DOMAIN), dtype=np.float32)
    for i, row in enumerate(neighbor_labels):
        counts = np.bincount(row, minlength=N_EXPERTS_DOMAIN)
        val_probs[i] = counts / counts.sum()

    acc = accuracy_score(y_val, val_preds)
    log.info(f"  [kNN] Routing Accuracy val: {acc:.4f}")

    per_exp   = per_expert_accuracy(y_val, val_preds)
    log_per_expert("kNN", per_exp)
    balance   = check_load_balance(val_preds, "kNN")
    threshold = calibrate_entropy_threshold(val_probs, y_val, "kNN")

    return index, y_train, acc, val_probs, balance, threshold


# ──────────────────────────────────────────────────────────
# 4. LATENCIA DE INFERENCIA
# ──────────────────────────────────────────────────────────


# Esta función implementa un test de benchmarking de alto rendimiento. Su propósito es medir con precisión de 
# microsegundos cuánto tiempo tarda el Router en tomar una decisión, descartando el ruido inicial 
# (calentamiento del hardware) para obtener una medida fiable y comparable de la latencia de inferencia entre 
# diferentes métodos.
def measure_latency(fn, n_runs: int = 10) -> float:
    """
    Mide la latencia promedio de inferencia en ms sobre un batch de 32 muestras.
    Descarta la primera corrida (JIT/warm-up).
    """
    times = []
    fn()   # warm-up
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000   # ms


# ──────────────────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────


# Este script es el Orquestador Maestro del Estudio de Ablación. Su rol es leer los embeddings pre-calculados, 
# entrenar y evaluar los cuatro métodos de enrutamiento (Lineal, GMM, Naive Bayes y k-NN) bajo las mismas 
# condiciones, generar la tabla comparativa requerida por la guía y exportar automáticamente la configuración 
# ganadora (incluyendo el umbral de entropía para el experto OOD) para que la FASE 1 real sepa exactamente cómo 
# configurarse.
def main(args):
    global log
    log = setup_logging(args.embeddings)

    log.info("=" * 65)
    log.info("FASE 1 — Ablation Study del Router")
    log.info(f"Arquitectura: {N_EXPERTS_DOMAIN} expertos de dominio + "
             f"1 OOD (total={N_EXPERTS_TOTAL})")
    log.info("=" * 65)

    emb_dir = Path(args.embeddings)

    # ── Cargar backbone_meta.json (generado por FASE 0) ──────────────────
    meta_path = emb_dir / "backbone_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            backbone_meta = json.load(f)
        log.info(f"[Setup] backbone_meta.json cargado: {backbone_meta}")
    else:
        backbone_meta = {}
        log.warning("[Setup] backbone_meta.json no encontrado. "
                    "d_model se inferirá de Z_train.shape[1]. "
                    "Ejecuta FASE 0 con la versión actualizada para generarlo.")

    # ── Cargar embeddings ─────────────────────────────────────────────────
    for fname in ["Z_train.npy", "y_train.npy", "Z_val.npy", "y_val.npy"]:
        if not (emb_dir / fname).exists():
            log.error(f"[Setup] Archivo faltante: {emb_dir / fname}. "
                      f"Ejecuta FASE 0 primero.")
            raise FileNotFoundError(emb_dir / fname)

    Z_train = np.load(emb_dir / "Z_train.npy")
    y_train = np.load(emb_dir / "y_train.npy")
    Z_val   = np.load(emb_dir / "Z_val.npy")
    y_val   = np.load(emb_dir / "y_val.npy")
    d_model = Z_train.shape[1]

    log.info(f"[Setup] Z_train: {Z_train.shape}  |  Z_val: {Z_val.shape}")
    log.info(f"[Setup] d_model: {d_model}")

    # Verificar coherencia con backbone_meta
    if backbone_meta.get("d_model") and backbone_meta["d_model"] != d_model:
        log.error(f"[Setup] d_model del archivo ({d_model}) ≠ backbone_meta "
                  f"({backbone_meta['d_model']}). Los embeddings pueden estar mezclados.")

    # ── Distribución de expertos ─────────────────────────────────────────
    log.info("[Setup] Distribución de expertos en train:")
    for exp_id, exp_name in EXPERT_NAMES.items():
        count = (y_train == exp_id).sum()
        pct   = 100 * count / len(y_train)
        log.info(f"  Experto {exp_id} ({exp_name:<12}): {count:>6,}  ({pct:.1f}%)")
    log.info(f"  Experto 5 (OOD        ):      0  (0.0%)  "
             f"← sin dataset, activado por entropía en inferencia")

    # Verificar que las 5 clases de dominio estén presentes
    classes_in_train = set(np.unique(y_train).tolist())
    expected = set(range(N_EXPERTS_DOMAIN))
    if classes_in_train != expected:
        log.error(f"[Setup] Clases en y_train: {classes_in_train} — "
                  f"se esperaban {expected}. El ablation study no será válido.")

    # Verificar NaN/Inf en embeddings antes de entrenar
    if np.isnan(Z_train).any() or np.isnan(Z_val).any():
        log.error("[Setup] NaN detectado en embeddings. Regenera con FASE 0.")
    if np.isinf(Z_train).any() or np.isinf(Z_val).any():
        log.error("[Setup] Inf detectado en embeddings. Regenera con FASE 0.")

    results = {}

    # ── A) LINEAR ────────────────────────────────────────────────────────
    log.info("\n[A] Entrenando Linear + Softmax (baseline DL)...")
    t0 = time.time()
    linear_model, linear_acc, linear_probs, linear_balance, linear_thresh = \
        train_linear_router(Z_train, y_train, Z_val, y_val, d_model,
                            epochs=args.epochs, lr=1e-3)
    linear_time = time.time() - t0

    # Latencia: batch de 32 muestras
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = torch.from_numpy(Z_val[:32]).float().to(device)
    linear_latency = measure_latency(lambda: linear_model(sample))

    results["Linear"] = {
        "acc":            linear_acc,
        "train_time_s":   linear_time,
        "latency_ms":     linear_latency,
        "params":         d_model * N_EXPERTS_DOMAIN + N_EXPERTS_DOMAIN,
        "needs_gpu":      True,
        "load_balance":   linear_balance,
        "entropy_thresh": linear_thresh,
        "model":          linear_model,
    }

    # ── B) GMM ───────────────────────────────────────────────────────────
    log.info("\n[B] Entrenando GMM (paramétrico EM)...")
    t0 = time.time()
    gmm_model, gmm_map, gmm_acc, gmm_probs, gmm_balance, gmm_thresh = \
        train_gmm_router(Z_train, y_train, Z_val, y_val)
    gmm_time = time.time() - t0

    sample_np = Z_val[:32].astype(np.float32)
    gmm_latency = measure_latency(lambda: gmm_model.predict_proba(sample_np))

    results["GMM"] = {
        "acc":            gmm_acc,
        "train_time_s":   gmm_time,
        "latency_ms":     gmm_latency,
        "params":         N_EXPERTS_DOMAIN * (d_model + d_model * d_model),
        "needs_gpu":      False,
        "load_balance":   gmm_balance,
        "entropy_thresh": gmm_thresh,
        "model":          (gmm_model, gmm_map),
    }

    # ── C) NAIVE BAYES ───────────────────────────────────────────────────
    log.info("\n[C] Entrenando Naive Bayes (MLE analítico)...")
    t0 = time.time()
    nb_model, nb_acc, nb_probs, nb_balance, nb_thresh = \
        train_nb_router(Z_train, y_train, Z_val, y_val)
    nb_time = time.time() - t0

    nb_latency = measure_latency(lambda: nb_model.predict_proba(sample_np))

    results["NaiveBayes"] = {
        "acc":            nb_acc,
        "train_time_s":   nb_time,
        "latency_ms":     nb_latency,
        "params":         N_EXPERTS_DOMAIN * 2 * d_model,
        "needs_gpu":      False,
        "load_balance":   nb_balance,
        "entropy_thresh": nb_thresh,
        "model":          nb_model,
    }

    # ── D) kNN FAISS ─────────────────────────────────────────────────────
    log.info("\n[D] Construyendo índice kNN-FAISS (no paramétrico)...")
    t0 = time.time()
    knn_index, knn_labels, knn_acc, knn_probs, knn_balance, knn_thresh = \
        train_knn_router(Z_train, y_train, Z_val, y_val, k=args.knn_k)
    knn_time = time.time() - t0

    sample_norm = sample_np.copy(); faiss.normalize_L2(sample_norm)
    knn_latency = measure_latency(lambda: knn_index.search(sample_norm, args.knn_k))

    results["kNN-FAISS"] = {
        "acc":            knn_acc,
        "train_time_s":   knn_time,
        "latency_ms":     knn_latency,
        "params":         len(Z_train) * d_model,
        "needs_gpu":      False,
        "load_balance":   knn_balance,
        "entropy_thresh": knn_thresh,
        "model":          (knn_index, knn_labels),
    }

    # ── TABLA COMPARATIVA ────────────────────────────────────────────────
    col = 78
    log.info("\n" + "=" * col)
    log.info(f"{'ABLATION STUDY — Tabla comparativa (sección 4.3 del proyecto)':^{col}}")
    log.info("=" * col)
    log.info(f"{'Router':<14} {'Tipo':<24} {'Acc':>6} {'Lat(ms)':>8} "
             f"{'Train(s)':>9} {'Bal':>5} {'H_thr':>6} {'GPU':>4}")
    log.info("-" * col)

    ROUTER_TYPE = {
        "Linear":    "Paramétrico (gradiente)",
        "GMM":       "Paramétrico (EM)",
        "NaiveBayes":"Paramétrico (MLE)",
        "kNN-FAISS": "No paramétrico",
    }
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        log.info(
            f"{name:<14} {ROUTER_TYPE[name]:<24} "
            f"{r['acc']:>6.4f} "
            f"{r['latency_ms']:>8.2f} "
            f"{r['train_time_s']:>9.1f} "
            f"{r['load_balance']:>5.2f} "
            f"{r['entropy_thresh']:>6.3f} "
            f"{'Sí' if r['needs_gpu'] else 'No':>4}"
        )
    log.info("=" * col)
    log.info("  Bal = max(f_i)/min(f_i) objetivo < 1.30 | "
             "H_thr = umbral entropía OOD (p95)")

    # ── GANADOR ──────────────────────────────────────────────────────────
    winner_name, winner = max(results.items(), key=lambda x: x[1]["acc"])
    log.info(f"\nROUTER GANADOR: {winner_name}  "
             f"(Routing Accuracy = {winner['acc']:.4f})")
    log.info(f"  Latencia de inferencia : {winner['latency_ms']:.2f} ms (batch=32)")
    log.info(f"  Balance de carga       : {winner['load_balance']:.2f}x")
    log.info(f"  Entropy threshold OOD  : {winner['entropy_thresh']:.4f}")
    log.info(f"\n  → Usar este router en FASE 1 del entrenamiento MoE")
    log.info(f"  → En FASE 1 real: LinearGatingHead(d_model={d_model}, "
             f"n_experts={N_EXPERTS_TOTAL})  ← añade slot OOD")
    log.info(f"  → El slot OOD (ID=5) se entrena con L_error, "
             f"no con labels de dominio")
    log.info(f"  → Registrar tabla completa en Reporte Técnico (sección 3)")

    # ── GUARDAR RESULTADOS ───────────────────────────────────────────────
    report = {}
    for k, v in results.items():
        report[k] = {
            "acc":            float(v["acc"]),
            "train_time_s":   float(v["train_time_s"]),
            "latency_ms":     float(v["latency_ms"]),
            "params":         int(v["params"]),
            "needs_gpu":      v["needs_gpu"],
            "load_balance":   float(v["load_balance"]),
            "entropy_thresh": float(v["entropy_thresh"]),
        }

    output = {
        "results":          report,
        "winner":           winner_name,
        "n_experts_domain": N_EXPERTS_DOMAIN,
        "n_experts_total":  N_EXPERTS_TOTAL,
        "d_model":          d_model,
        "backbone":         backbone_meta.get("backbone", "unknown"),
        "entropy_threshold_winner": float(winner["entropy_thresh"]),
        "note_ood": (
            f"Experto 5 (OOD) se activa en inferencia cuando H(g) >= "
            f"{winner['entropy_thresh']:.4f}. "
            f"En FASE 1 real usar LinearGatingHead(n_experts={N_EXPERTS_TOTAL})."
        ),
    }

    out_path = emb_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResultados guardados en {out_path}")
    log.info(f"Siguiente paso: entrenar FASE 1 MoE con router='{winner_name}' "
             f"y n_experts={N_EXPERTS_TOTAL}")

# Este bloque es el "Panel de Control de Línea de Comandos". Su función es recibir y validar los parámetros de 
# configuración necesarios para ejecutar el estudio de ablación, permitiendo al usuario indicar dónde están los 
# datos (embeddings), qué tan intensivo debe ser el entrenamiento del router neuronal (epochs) y qué 
# tan sensible debe ser el router estadístico (knn_k) de manera flexible y profesional.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FASE 1 — Ablation Study del Router (arquitectura 6 expertos)"
    )
    parser.add_argument(
        "--embeddings", default="./embeddings",
        help="Carpeta con Z_train.npy, Z_val.npy y backbone_meta.json (output de FASE 0)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Épocas para entrenar el Linear router (ablation). "
             "FASE 1 real usa ~50 épocas con LR=1e-3."
    )
    parser.add_argument(
        "--knn_k", type=int, default=5,
        help="Número de vecinos para kNN-FAISS (default=5)"
    )
    args = parser.parse_args()
    main(args)