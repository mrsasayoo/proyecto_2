"""
fix_luna_leakage.py — Mueve parches de train/ que pertenecen a val/test UIDs.

Problema (W-03): 1,839 archivos .npy en patches/train/ corresponden a
seriesuid que fueron reasignados a val o test en luna_splits.json.
Estos son archivos residuales de una extracción anterior.

Solución: mover (no eliminar) los archivos filtrados a patches/train_stale_backup/.
"""

import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    base = Path(__file__).resolve().parents[3]  # proyecto_2/
    luna_dir = base / "datasets" / "luna_lung_cancer"

    # 1. Leer splits
    splits_path = luna_dir / "luna_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"No se encontró luna_splits.json en {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)

    val_uids = set(splits["val_uids"])
    test_uids = set(splits["test_uids"])
    print(f"val UIDs: {len(val_uids)}, test UIDs: {len(test_uids)}")

    # 2. Leer candidates CSV
    csv_path = luna_dir / "candidates_V2" / "candidates_V2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró candidates CSV en {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"candidates_V2.csv: {len(df):,} filas")

    # 3. Escanear patches/train/
    train_dir = luna_dir / "patches" / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"No se encontró directorio train en {train_dir}")

    backup_dir = luna_dir / "patches" / "train_stale_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    kept = 0
    errors = 0

    for npy_file in sorted(train_dir.glob("candidate_*.npy")):
        # Parsear índice del nombre: candidate_001234.npy → 1234
        try:
            idx = int(npy_file.stem.split("_")[1])
        except (IndexError, ValueError) as e:
            print(f"  WARN: no se pudo parsear índice de '{npy_file.name}': {e}")
            errors += 1
            continue

        if idx >= len(df):
            print(f"  WARN: índice {idx} fuera de rango (CSV tiene {len(df)} filas)")
            errors += 1
            continue

        uid = df.iloc[idx]["seriesuid"]

        if uid in val_uids or uid in test_uids:
            dest = backup_dir / npy_file.name
            shutil.move(str(npy_file), str(dest))
            moved += 1
        else:
            kept += 1

    print(f"\n=== Resumen ===")
    print(f"  Archivos movidos a train_stale_backup/: {moved}")
    print(f"  Archivos que permanecen en train/:      {kept}")
    print(f"  Errores de parseo:                      {errors}")
    print(f"  Total procesados:                       {moved + kept + errors}")


if __name__ == "__main__":
    main()
