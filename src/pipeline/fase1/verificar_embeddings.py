"""
verificar_embeddings.py
Verifica la integridad de todos los embeddings generados en Fase 1.
Comprueba: shapes, NaN/Inf, distribución de labels (0-4), consistencia entre splits.
"""

import numpy as np
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJ / "embeddings"

BACKBONES = {
    "vit_tiny_patch16_224": {"d_model": 192},
    "densenet121_custom": {"d_model": 1024},
    "swin_tiny_patch4_window7_224": {"d_model": 768},
    "cvt_13": {"d_model": 384},
}

SPLITS_ESPERADOS = {
    "train": {"min": 128000, "max": 130000},
    "val": {"min": 15000, "max": 16000},
    "test": {"min": 16000, "max": 17000},
}

EXPERTOS_ESPERADOS = list(range(5))  # 0=Chest 1=ISIC 2=OA 3=LUNA 4=Pancreas


def verificar_backbone(nombre: str, d_model_esperado: int) -> dict:
    """Verifica un directorio de embeddings completo. Retorna dict con estado."""
    ruta = EMBEDDINGS_DIR / nombre
    resultado = {
        "backbone": nombre,
        "ok": True,
        "errores": [],
        "advertencias": [],
        "stats": {},
    }

    if not ruta.exists():
        resultado["ok"] = False
        resultado["errores"].append(f"Directorio no existe: {ruta}")
        return resultado

    # 1. Verificar backbone_meta.json
    meta_path = ruta / "backbone_meta.json"
    if not meta_path.exists():
        resultado["ok"] = False
        resultado["errores"].append("backbone_meta.json no encontrado")
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        d_model_real = meta.get("d_model", -1)
        if d_model_real != d_model_esperado:
            resultado["ok"] = False
            resultado["errores"].append(
                f"d_model incorrecto: {d_model_real} (esperado {d_model_esperado})"
            )
        else:
            resultado["stats"]["d_model"] = d_model_real

    # 2. Verificar cada split
    for split, limites in SPLITS_ESPERADOS.items():
        Z_path = ruta / f"Z_{split}.npy"
        y_path = ruta / f"y_{split}.npy"
        names_path = ruta / f"names_{split}.txt"

        # Existencia
        for p in [Z_path, y_path, names_path]:
            if not p.exists():
                resultado["ok"] = False
                resultado["errores"].append(f"Falta archivo: {p.name}")

        if not Z_path.exists() or not y_path.exists():
            continue

        # Cargar
        Z = np.load(Z_path)
        y = np.load(y_path)

        # Shape
        n_muestras, dim = Z.shape
        if dim != d_model_esperado:
            resultado["ok"] = False
            resultado["errores"].append(
                f"[{split}] d_model real={dim} ≠ esperado={d_model_esperado}"
            )

        if not (limites["min"] <= n_muestras <= limites["max"]):
            resultado["ok"] = False
            resultado["errores"].append(
                f"[{split}] n_muestras={n_muestras} fuera de rango [{limites['min']}, {limites['max']}]"
            )

        # NaN / Inf
        n_nan = int(np.isnan(Z).sum())
        n_inf = int(np.isinf(Z).sum())
        if n_nan > 0:
            resultado["ok"] = False
            resultado["errores"].append(f"[{split}] Z contiene {n_nan} NaN")
        if n_inf > 0:
            resultado["ok"] = False
            resultado["errores"].append(f"[{split}] Z contiene {n_inf} Inf")

        # Consistencia y ↔ Z
        if len(y) != n_muestras:
            resultado["ok"] = False
            resultado["errores"].append(
                f"[{split}] len(y)={len(y)} ≠ Z.shape[0]={n_muestras}"
            )

        # Distribución de labels (expertos 0-4)
        labels_unicos = sorted(np.unique(y).tolist())
        labels_faltantes = [e for e in EXPERTOS_ESPERADOS if e not in labels_unicos]
        if labels_faltantes:
            resultado["advertencias"].append(
                f"[{split}] Labels faltantes (expertos sin muestras): {labels_faltantes}"
            )

        conteos = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

        # Consistencia names ↔ Z
        if names_path.exists():
            n_names = sum(1 for _ in open(names_path))
            if n_names != n_muestras:
                resultado["ok"] = False
                resultado["errores"].append(
                    f"[{split}] n_names={n_names} ≠ Z.shape[0]={n_muestras}"
                )

        resultado["stats"][split] = {
            "n": n_muestras,
            "dim": dim,
            "nan": n_nan,
            "inf": n_inf,
            "labels": conteos,
        }

    return resultado


def main():
    print("=" * 65)
    print("VERIFICACIÓN DE EMBEDDINGS — FASE 1")
    print("=" * 65)

    todos_ok = True
    for nombre, cfg in BACKBONES.items():
        res = verificar_backbone(nombre, cfg["d_model"])
        estado = "✓ OK" if res["ok"] else "✗ ERROR"
        print(f"\n[{estado}] {nombre}")

        if res["errores"]:
            todos_ok = False
            for e in res["errores"]:
                print(f"  ❌ {e}")

        if res["advertencias"]:
            for w in res["advertencias"]:
                print(f"  ⚠  {w}")

        if res["stats"]:
            if "d_model" in res["stats"]:
                print(f"  d_model: {res['stats']['d_model']}")
            for split in ["train", "val", "test"]:
                if split in res["stats"]:
                    s = res["stats"][split]
                    labels_str = " | ".join(
                        f"E{k}:{v}" for k, v in sorted(s["labels"].items())
                    )
                    print(
                        f"  {split:5s}: {s['n']:7,d} muestras × {s['dim']}d  [{labels_str}]"
                    )

    print("\n" + "=" * 65)
    if todos_ok:
        print("✅ TODOS LOS BACKBONES VERIFICADOS CORRECTAMENTE")
    else:
        print("❌ HAY ERRORES — revisar mensajes anteriores")
    print("=" * 65)

    return 0 if todos_ok else 1


if __name__ == "__main__":
    sys.exit(main())
