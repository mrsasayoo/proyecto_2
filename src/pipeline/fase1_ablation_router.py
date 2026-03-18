"""
FASE 1 — Ablation Study del Router
====================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II

Lee Z_train.npy y Z_val.npy generados por FASE 0 y compara
los 4 mecanismos de routing sobre el mismo espacio de embeddings.

Mecanismos (sección 4.2 del proyecto):
  A) ViT + Linear + Softmax  (baseline DL)
  B) ViT + GMM               (paramétrico EM)
  C) ViT + Naive Bayes       (paramétrico MLE analítico)
  D) ViT + k-NN con FAISS    (no paramétrico)

Uso:
  python fase1_ablation_router.py --embeddings ./embeddings
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import faiss


N_EXPERTS = 5   # Expertos 0-4 (Chest, ISIC, OA, LUNA, Pancreas)


# ──────────────────────────────────────────────────────────
# A) LINEAR + SOFTMAX (baseline deep learning)
# ──────────────────────────────────────────────────────────

class LinearGatingHead(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, z):
        return torch.softmax(self.gate(z), dim=-1)   # [B, n_experts]


def train_linear_router(Z_train, y_train, Z_val, y_val, d_model,
                        epochs=50, lr=1e-3, batch_size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LinearGatingHead(d_model, N_EXPERTS).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Z_t = torch.from_numpy(Z_train).float().to(device)
    y_t = torch.from_numpy(y_train).long().to(device)

    best_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(Z_t))
        epoch_loss = 0.0
        for i in range(0, len(Z_t), batch_size):
            batch_idx = idx[i:i + batch_size]
            z_b, y_b  = Z_t[batch_idx], y_t[batch_idx]
            logits = model.gate(z_b)
            loss   = loss_fn(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Validación cada 10 épocas
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                Z_v   = torch.from_numpy(Z_val).float().to(device)
                probs = model(Z_v).cpu().numpy()
                preds = probs.argmax(axis=1)
                acc   = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc     = acc
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  [Linear] época {epoch+1:3d} | loss {epoch_loss:.4f} | val acc {acc:.4f}")

    model.load_state_dict(best_weights)
    return model, best_acc


# ──────────────────────────────────────────────────────────
# B) GMM (paramétrico EM)
# ──────────────────────────────────────────────────────────

def train_gmm_router(Z_train, y_train, Z_val, y_val):
    print("  [GMM] Ajustando GaussianMixture (5 comp., full covariance)...")

    # Intentar full; si hay problemas de rank → diag
    try:
        gmm = GaussianMixture(
            n_components=N_EXPERTS,
            covariance_type="full",
            max_iter=200,
            random_state=42,
            verbose=0
        )
        gmm.fit(Z_train)
    except Exception:
        print("  [GMM] full covarianza falló → usando diag")
        gmm = GaussianMixture(
            n_components=N_EXPERTS,
            covariance_type="diag",
            max_iter=200,
            random_state=42
        )
        gmm.fit(Z_train)

    # El GMM agrupa sin saber las etiquetas originales.
    # Hacemos el mapeo componente → experto por voto mayoritario.
    train_comp = gmm.predict(Z_train)
    comp_to_expert = {}
    for comp in range(N_EXPERTS):
        mask   = train_comp == comp
        if mask.sum() == 0:
            comp_to_expert[comp] = 0
            continue
        labels = y_train[mask]
        comp_to_expert[comp] = int(np.bincount(labels).argmax())

    val_comp  = gmm.predict(Z_val)
    val_preds = np.array([comp_to_expert[c] for c in val_comp])
    acc       = accuracy_score(y_val, val_preds)
    print(f"  [GMM] Mapeo componente→experto: {comp_to_expert}")
    print(f"  [GMM] Routing Accuracy val: {acc:.4f}")
    return gmm, comp_to_expert, acc


# ──────────────────────────────────────────────────────────
# C) NAIVE BAYES (paramétrico MLE analítico)
# ──────────────────────────────────────────────────────────

def train_nb_router(Z_train, y_train, Z_val, y_val):
    print("  [NB] Ajustando GaussianNB (MLE analítico)...")
    nb = GaussianNB()
    nb.fit(Z_train, y_train)

    val_preds = nb.predict(Z_val)
    acc       = accuracy_score(y_val, val_preds)
    print(f"  [NB] Routing Accuracy val: {acc:.4f}")
    return nb, acc


# ──────────────────────────────────────────────────────────
# D) k-NN con FAISS (no paramétrico, distancia coseno)
# ──────────────────────────────────────────────────────────

def train_knn_router(Z_train, y_train, Z_val, y_val, k=5):
    print(f"  [kNN] Construyendo índice FAISS (k={k}, coseno)...")
    d = Z_train.shape[1]

    # Normalizar para usar Inner Product como coseno
    Z_t_norm = Z_train.copy().astype(np.float32)
    faiss.normalize_L2(Z_t_norm)
    Z_v_norm = Z_val.copy().astype(np.float32)
    faiss.normalize_L2(Z_v_norm)

    index = faiss.IndexFlatIP(d)
    index.add(Z_t_norm)

    # Buscar k vecinos
    _, I = index.search(Z_v_norm, k)                 # [N_val, k]

    # Voto mayoritario
    neighbor_labels = y_train[I]                     # [N_val, k]
    val_preds = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=N_EXPERTS).argmax(),
        axis=1, arr=neighbor_labels
    )
    acc = accuracy_score(y_val, val_preds)
    print(f"  [kNN] Routing Accuracy val: {acc:.4f}")
    return index, y_train, acc


# ──────────────────────────────────────────────────────────
# COMPARACIÓN FINAL
# ──────────────────────────────────────────────────────────

def measure_latency(fn, n_runs=5):
    """Mide latencia promedio de inferencia sobre un batch pequeño."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000   # ms


def main(args):
    print("=" * 60)
    print("FASE 1 — Ablation Study del Router")
    print("=" * 60)

    # Cargar embeddings guardados en FASE 0
    Z_train = np.load(f"{args.embeddings}/Z_train.npy")
    y_train = np.load(f"{args.embeddings}/y_train.npy")
    Z_val   = np.load(f"{args.embeddings}/Z_val.npy")
    y_val   = np.load(f"{args.embeddings}/y_val.npy")
    d_model = Z_train.shape[1]

    print(f"\nEmbeddings cargados:")
    print(f"  Z_train: {Z_train.shape}  |  Z_val: {Z_val.shape}")
    print(f"  d_model: {d_model}")
    print(f"  Clases en train: {np.unique(y_train, return_counts=True)}")

    results = {}

    # ── A) LINEAR ──
    print("\n[A] Entrenando Linear + Softmax...")
    t0 = time.time()
    linear_model, linear_acc = train_linear_router(
        Z_train, y_train, Z_val, y_val, d_model,
        epochs=args.epochs, lr=1e-3
    )
    linear_time = time.time() - t0
    results["Linear"] = {
        "acc": linear_acc,
        "train_time_s": linear_time,
        "params": d_model * N_EXPERTS + N_EXPERTS,
        "needs_gpu": True,
        "model": linear_model
    }

    # ── B) GMM ──
    print("\n[B] Entrenando GMM...")
    t0 = time.time()
    gmm_model, gmm_map, gmm_acc = train_gmm_router(Z_train, y_train, Z_val, y_val)
    gmm_time = time.time() - t0
    results["GMM"] = {
        "acc": gmm_acc,
        "train_time_s": gmm_time,
        "params": N_EXPERTS * (d_model + d_model * d_model),   # media + cov full
        "needs_gpu": False,
        "model": (gmm_model, gmm_map)
    }

    # ── C) NAIVE BAYES ──
    print("\n[C] Entrenando Naive Bayes...")
    t0 = time.time()
    nb_model, nb_acc = train_nb_router(Z_train, y_train, Z_val, y_val)
    nb_time = time.time() - t0
    results["NaiveBayes"] = {
        "acc": nb_acc,
        "train_time_s": nb_time,
        "params": N_EXPERTS * 2 * d_model,    # media + varianza por clase
        "needs_gpu": False,
        "model": nb_model
    }

    # ── D) kNN FAISS ──
    print("\n[D] Construyendo índice kNN-FAISS...")
    t0 = time.time()
    knn_index, knn_labels, knn_acc = train_knn_router(Z_train, y_train, Z_val, y_val, k=5)
    knn_time = time.time() - t0
    results["kNN-FAISS"] = {
        "acc": knn_acc,
        "train_time_s": knn_time,
        "params": len(Z_train) * d_model,     # todos los vectores de train
        "needs_gpu": False,
        "model": (knn_index, knn_labels)
    }

    # ── TABLA COMPARATIVA ──
    print("\n")
    print("=" * 72)
    print(f"{'ABLATION STUDY — Tabla comparativa (sección 4.3 del proyecto)':^72}")
    print("=" * 72)
    print(f"{'Router':<14} {'Tipo':<22} {'Acc. val':>10} {'Train(s)':>10} {'Paráms':>12} {'GPU':>6}")
    print("-" * 72)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        tipo = {
            "Linear":    "Paramétrico (gradiente)",
            "GMM":       "Paramétrico (EM)",
            "NaiveBayes":"Paramétrico (MLE)",
            "kNN-FAISS": "No paramétrico"
        }[name]
        print(
            f"{name:<14} {tipo:<22} "
            f"{r['acc']:>10.4f} "
            f"{r['train_time_s']:>10.1f} "
            f"{r['params']:>12,} "
            f"{'Sí' if r['needs_gpu'] else 'No':>6}"
        )
    print("=" * 72)

    # ── GANADOR ──
    winner = max(results.items(), key=lambda x: x[1]["acc"])
    print(f"\nROUTER GANADOR: {winner[0]}  (Routing Accuracy = {winner[1]['acc']:.4f})")
    print(f"→ Usar este router en FASE 1 del entrenamiento MoE")
    print(f"→ Registrar tabla completa en el Reporte Técnico (sección 3)")

    # Guardar resultados para el reporte
    import json
    report = {k: {"acc": float(v["acc"]), "train_time_s": float(v["train_time_s"]),
                  "params": int(v["params"]), "needs_gpu": v["needs_gpu"]}
              for k, v in results.items()}
    with open(f"{args.embeddings}/ablation_results.json", "w") as f:
        json.dump({"results": report, "winner": winner[0]}, f, indent=2)
    print(f"\nResultados guardados en {args.embeddings}/ablation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FASE 1 — Ablation Study del Router")
    parser.add_argument("--embeddings", default="./embeddings",
                        help="Carpeta con Z_train.npy y Z_val.npy (output de FASE 0)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Épocas para entrenar el Linear router")
    args = parser.parse_args()
    main(args)
