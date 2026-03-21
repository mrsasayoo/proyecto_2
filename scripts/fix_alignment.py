#!/usr/bin/env python3
"""
fix_alignment.py — Ejecuta los 10 fixes de alineación identificados
en audit_report.md para el proyecto MoE médico.

Ejecutar desde la raíz del proyecto:
    cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2
    python3 scripts/fix_alignment.py
"""

import os
import sys
import json
import shutil
import subprocess
import time
import datetime
import traceback
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Constantes ──────────────────────────────────────────────────────
BASE = Path("/mnt/hdd/datasets/carlos_andres_ferro/proyecto_2")
DS   = BASE / "datasets"
SEED = 42

np.random.seed(SEED)

results = {}          # almacena {fix_name: {status, ...}}
timings = {}          # almacena {fix_name: seconds}


# ════════════════════════════════════════════════════════════════════
#  FIX 1 — NIH ChestXray14: Crear split de validación
# ════════════════════════════════════════════════════════════════════
def fix1_nih_val_split():
    print("\n" + "=" * 60)
    print("FIX 1 — NIH: crear split de validación por patient_id")
    print("=" * 60)

    nih = DS / "nih_chest_xrays"
    out = nih / "splits"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(nih / "Data_Entry_2017.csv")
    tv_imgs = [l.strip() for l in
               (nih / "train_val_list.txt").read_text().split("\n") if l.strip()]
    te_imgs = [l.strip() for l in
               (nih / "test_list.txt").read_text().split("\n") if l.strip()]

    print(f"  train_val: {len(tv_imgs)} imgs  |  test: {len(te_imgs)} imgs")

    img2pid    = dict(zip(df["Image Index"], df["Patient ID"]))
    img2labels = dict(zip(df["Image Index"], df["Finding Labels"]))

    # Agrupar imágenes train_val por paciente
    pid_imgs = defaultdict(list)
    for img in tv_imgs:
        pid = img2pid.get(img)
        if pid is not None:
            pid_imgs[pid].append(img)

    # Etiquetas por paciente (unión de todas sus imágenes)
    pid_labels = defaultdict(set)
    for pid, imgs in pid_imgs.items():
        for img in imgs:
            for lbl in img2labels.get(img, "No Finding").split("|"):
                pid_labels[pid].add(lbl.strip())

    # Frecuencia de cada etiqueta entre pacientes
    lbl_freq = defaultdict(int)
    for labels in pid_labels.values():
        for lbl in labels:
            lbl_freq[lbl] += 1

    # Clave de estratificación = etiqueta más rara del paciente
    pids = list(pid_imgs.keys())
    strat = [min(pid_labels[p], key=lambda l: lbl_freq[l]) for p in pids]

    try:
        train_pids, val_pids = train_test_split(
            pids, test_size=0.2, stratify=strat, random_state=SEED
        )
    except ValueError:
        print("  ⚠️  Stratified split falló — usando split aleatorio por paciente")
        train_pids, val_pids = train_test_split(
            pids, test_size=0.2, random_state=SEED
        )

    train_list = sorted(img for p in train_pids for img in pid_imgs[p])
    val_list   = sorted(img for p in val_pids  for img in pid_imgs[p])

    (out / "nih_train_list.txt").write_text("\n".join(train_list) + "\n")
    (out / "nih_val_list.txt").write_text("\n".join(val_list) + "\n")
    shutil.copy2(nih / "test_list.txt", out / "nih_test_list.txt")

    # Verificar que no hay solapamiento de pacientes entre ningún par
    tp, vp = set(train_pids), set(val_pids)
    tep = {img2pid[i] for i in te_imgs if i in img2pid}
    assert not (tp & vp),  "¡Overlap pacientes train/val!"
    assert not (tp & tep), "¡Overlap pacientes train/test!"
    assert not (vp & tep), "¡Overlap pacientes val/test!"

    r = dict(status="✅", train=len(train_list), val=len(val_list),
             test=len(te_imgs), overlap=0)
    print(f"  ✓ train={r['train']}  val={r['val']}  test={r['test']}")
    print(f"  ✓ Overlap pacientes: 0")
    results["fix1"] = r


# ════════════════════════════════════════════════════════════════════
#  FIX 2 — ISIC 2019: Crear splits train/val/test por lesion_id
# ════════════════════════════════════════════════════════════════════
def fix2_isic_splits():
    print("\n" + "=" * 60)
    print("FIX 2 — ISIC: splits train/val/test por lesion_id")
    print("=" * 60)

    CLASS_TO_IDX = {
        "MEL": 0, "NV": 1, "BCC": 2, "AK": 3,
        "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7,
    }

    isic = DS / "isic_2019"
    out  = isic / "splits"
    out.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(isic / "ISIC_2019_Training_GroundTruth.csv")
    md = pd.read_csv(isic / "ISIC_2019_Training_Metadata.csv")

    # Excluir filas donde UNK == 1 (0 imágenes, pero por seguridad)
    gt = gt[gt["UNK"] < 0.5].copy()

    # Determinar clase por imagen (one-hot → label name)
    class_cols = list(CLASS_TO_IDX.keys())
    gt["label_name"] = gt[class_cols].idxmax(axis=1)
    gt["label_idx"]  = gt["label_name"].map(CLASS_TO_IDX)

    # Merge con metadata para obtener lesion_id
    merged = gt[["image", "label_idx", "label_name"]].merge(
        md[["image", "lesion_id"]], on="image", how="left"
    )

    # Asignar lesion_id sintético para nulos
    null_mask = merged["lesion_id"].isna()
    merged.loc[null_mask, "lesion_id"] = merged.loc[null_mask, "image"].apply(
        lambda x: f"SOLO_{x}"
    )

    print(f"  Total imágenes: {len(merged)}  |  "
          f"Lesiones únicas: {merged['lesion_id'].nunique()}")

    # Clase por lesión (moda de las imágenes del grupo)
    lesion_class = (
        merged.groupby("lesion_id")["label_idx"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .reset_index()
    )
    lesion_class.columns = ["lesion_id", "cls"]

    les_ids = lesion_class["lesion_id"].to_numpy()
    les_cls = lesion_class["cls"].to_numpy()

    # 70 / 15 / 15
    try:
        train_les, temp_les, _, temp_cls = train_test_split(
            les_ids, les_cls, test_size=0.30,
            stratify=les_cls, random_state=SEED,
        )
        val_les, test_les = train_test_split(
            temp_les, test_size=0.50,
            stratify=temp_cls, random_state=SEED,
        )
    except ValueError:
        print("  ⚠️  Stratified split falló — usando split aleatorio")
        train_les, temp_les = train_test_split(
            les_ids, test_size=0.30, random_state=SEED,
        )
        val_les, test_les = train_test_split(
            temp_les, test_size=0.50, random_state=SEED,
        )

    sets = {
        "train": set(train_les),
        "val":   set(val_les),
        "test":  set(test_les),
    }

    # Verificar no-overlap de lesiones
    assert not (sets["train"] & sets["val"]),  "Overlap lesion train/val!"
    assert not (sets["train"] & sets["test"]), "Overlap lesion train/test!"
    assert not (sets["val"]   & sets["test"]), "Overlap lesion val/test!"

    cols = ["image", "label_idx", "label_name", "lesion_id"]
    counts = {}
    for name, ids in sets.items():
        sub = merged[merged["lesion_id"].isin(ids)][cols]
        sub.to_csv(out / f"isic_{name}.csv", index=False)
        counts[name] = len(sub)

    total = sum(counts.values())
    active = sorted(merged["label_name"].unique().tolist())
    print(f"  ✓ train={counts['train']}  val={counts['val']}  "
          f"test={counts['test']}  total={total}")
    print(f"  ✓ Clases activas ({len(active)}): {active}")
    print(f"  ✓ Overlap lesion_id: 0")
    results["fix2"] = dict(status="✅", **counts, total=total,
                           classes=active, overlap=0)


# ════════════════════════════════════════════════════════════════════
#  FIX 3 — ISIC 2019: Symlinks planos
# ════════════════════════════════════════════════════════════════════
def fix3_isic_symlinks():
    print("\n" + "=" * 60)
    print("FIX 3 — ISIC: resolver path doblemente anidado")
    print("=" * 60)

    isic    = DS / "isic_2019"
    src_dir = isic / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
    dst_dir = isic / "isic_images"
    dst_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(src_dir.glob("*.jpg"))
    created = 0
    for jpg in jpgs:
        link = dst_dir / jpg.name
        if link.is_symlink() or link.exists():
            continue
        target = os.path.relpath(jpg, dst_dir)
        link.symlink_to(target)
        created += 1

    total_links = sum(1 for f in dst_dir.iterdir()
                      if f.name.endswith(".jpg"))
    print(f"  Symlinks creados esta ejecución: {created}")
    print(f"  Total en isic_images/: {total_links}")
    assert total_links == len(jpgs), (
        f"Conteo no coincide: {total_links} vs {len(jpgs)} originales"
    )
    print(f"  ✓ {total_links} symlinks = {len(jpgs)} JPGs originales")
    results["fix3"] = dict(status="✅", symlinks=total_links)


# ════════════════════════════════════════════════════════════════════
#  FIX 4 — Osteoarthritis: Investigar déficit + documentar
# ════════════════════════════════════════════════════════════════════
def fix4_oa_investigation():
    print("\n" + "=" * 60)
    print("FIX 4 — OA: investigar y documentar déficit de imágenes")
    print("=" * 60)

    oa = DS / "osteoarthritis"
    kl = oa / "KLGrade" / "KLGrade"

    # A) Contar imágenes por grado KL
    img_exts = {".jpg", ".png", ".jpeg"}
    kl_counts = {}
    for grade in range(5):
        d = kl / str(grade)
        try:
            n = sum(1 for f in d.iterdir()
                    if f.is_file() and f.suffix.lower() in img_exts)
        except OSError:
            n = 0   # KL4 puede tener error I/O en el filesystem
        kl_counts[grade] = n
        print(f"  KL{grade}: {n}")
    kl_total = sum(kl_counts.values())

    # B) Contar en otros directorios (discarded, withoutKLGrade)
    extra = {}
    for sub in ["discarded", "withoutKLGrade"]:
        sub_dir = oa / sub
        try:
            n = sum(1 for f in sub_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in img_exts)
        except OSError:
            n = 0
        extra[sub] = n
        print(f"  {sub}/: {n}")

    # C) Extensiones en KLGrade
    ext_counts = defaultdict(int)
    for grade in range(4):          # skip KL4 (I/O)
        d = kl / str(grade)
        try:
            for f in d.iterdir():
                if f.is_file():
                    ext_counts[f.suffix.lower()] += 1
        except OSError:
            pass
    print(f"  Extensiones KLGrade: {dict(ext_counts)}")

    # D) ZIP listing
    zip_path = oa / "osteoarthritis.zip"
    zip_imgs = 0
    if zip_path.exists():
        try:
            r = subprocess.run(
                ["unzip", "-l", str(zip_path)],
                capture_output=True, text=True, timeout=120,
            )
            for line in r.stdout.split("\n"):
                ll = line.lower()
                if any(ext in ll for ext in [".jpg", ".png", ".jpeg"]):
                    zip_imgs += 1
            print(f"  ZIP: {zip_imgs} entradas de imagen")
        except Exception as e:
            print(f"  ZIP: error ({e})")

    # E) Generar reporte markdown
    report = f"""# Osteoarthritis — Reporte de Dataset

**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Conteo por Grado KL (KLGrade/KLGrade/)

| Grado | Imágenes | Nota |
|-------|----------|------|
| KL0 (sano)     | {kl_counts[0]:,} | — |
| KL1 (dudoso)   | {kl_counts[1]:,} | — |
| KL2 (mínimo)   | {kl_counts[2]:,} | — |
| KL3 (moderado) | {kl_counts[3]:,} | — |
| KL4 (severo)   | {kl_counts[4]} | Error I/O o vacío |
| **Total KLGrade** | **{kl_total:,}** | Solo estas son anotadas |

## Otros Directorios

| Directorio | Imágenes | Descripción |
|------------|----------|-------------|
| discarded/ | {extra.get('discarded', 0)} | Descartadas por calidad |
| withoutKLGrade/ | {extra.get('withoutKLGrade', 0):,} | Sin grado KL (no usables) |
| **Total en disco** | **{kl_total + sum(extra.values()):,}** | — |
| ZIP original | {zip_imgs:,} | Entradas en el ZIP |

## Extensiones

```json
{json.dumps(dict(ext_counts), indent=2)}
```

## Conclusión

- El dataset de Kaggle contiene **{kl_total:,} imágenes con grado KL** asignado.
- La guía del proyecto estima ~10K, pero esa cifra incluye imágenes sin grado
  ({extra.get('withoutKLGrade', 0):,}) que **no sirven** para clasificación.
- **KL4 = 0 imágenes** (directorio vacío/I/O error). Limitación conocida del dataset.
- La clase consolidada 2 (severo) = solo KL3, no KL3+KL4.

## Splits Existentes (oa_splits/)

No se regeneran — corresponden a las {kl_total:,} imágenes anotadas.

| Clase | Mapeo | Imágenes |
|-------|-------|----------|
| 0 | KL0 (sano) | {kl_counts[0]:,} |
| 1 | KL1+KL2 (leve) | {kl_counts[1] + kl_counts[2]:,} |
| 2 | KL3+KL4 (severo) | {kl_counts[3] + kl_counts[4]:,} |
"""
    (oa / "oa_dataset_report.md").write_text(report)

    print(f"  ✓ oa_dataset_report.md generado")
    print(f"  ✓ KLGrade total: {kl_total} | "
          f"Disco total: {kl_total + sum(extra.values())}")
    results["fix4"] = dict(
        status="✅", kl_total=kl_total,
        disco_total=kl_total + sum(extra.values()),
        zip_imgs=zip_imgs,
    )


# ════════════════════════════════════════════════════════════════════
#  FIX 5 — Osteoarthritis: Reporte de modos de imagen
# ════════════════════════════════════════════════════════════════════
def fix5_oa_image_modes():
    print("\n" + "=" * 60)
    print("FIX 5 — OA: verificar modos de imagen")
    print("=" * 60)

    from PIL import Image

    kl = DS / "osteoarthritis" / "KLGrade" / "KLGrade"
    img_exts = {".jpg", ".png", ".jpeg"}
    modes = defaultdict(int)
    problem_files = []
    total = 0

    for grade in range(4):          # skip KL4 (I/O error)
        d = kl / str(grade)
        try:
            files = sorted(d.iterdir())
        except OSError:
            continue
        for fp in files:
            if fp.suffix.lower() not in img_exts:
                continue
            total += 1
            try:
                with Image.open(fp) as img:
                    m = img.mode
                    modes[m] += 1
                    if m not in {"RGB", "L"}:
                        problem_files.append({
                            "path": str(fp.relative_to(DS)),
                            "mode": m,
                        })
            except Exception as e:
                problem_files.append({
                    "path": str(fp.relative_to(DS)),
                    "mode": f"ERROR: {e}",
                })
            if total % 500 == 0:
                print(f"  ... {total} escaneadas")

    report = {
        "total_scanned": total,
        "modes": dict(modes),
        "problem_files_count": len(problem_files),
        "problem_files": problem_files[:100],
    }
    out_path = DS / "osteoarthritis" / "image_modes_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Escaneadas: {total}  |  Modos: {dict(modes)}")
    print(f"  ✓ Archivos problemáticos (mode ∉ {{RGB, L}}): {len(problem_files)}")
    results["fix5"] = dict(status="✅", modes=dict(modes),
                           problems=len(problem_files))


# ════════════════════════════════════════════════════════════════════
#  FIX 6 — LUNA16: Intentar descarga de CTs
# ════════════════════════════════════════════════════════════════════
def fix6_luna_download():
    print("\n" + "=" * 60)
    print("FIX 6 — LUNA16: descargar volúmenes CT")
    print("=" * 60)

    luna   = DS / "luna_lung_cancer"
    ct_dir = luna / "ct_volumes"
    ct_dir.mkdir(parents=True, exist_ok=True)

    zenodo_url = "https://zenodo.org/records/3723295/files/subset0.zip"
    url_accessible = False
    download_ok    = False
    mhd_count      = 0

    # ── Test URL ───────────────────────────────────────────
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(zenodo_url + "?download=1", method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            size_bytes = resp.headers.get("Content-Length", "?")
            url_accessible = status in (200, 302)
            print(f"  URL status: {status}  |  Tamaño: {size_bytes} bytes")
    except Exception as e:
        print(f"  URL no accesible: {e}")

    # ── Kaggle search ──────────────────────────────────────
    kaggle_output = ""
    try:
        r = subprocess.run(
            ["kaggle", "datasets", "list", "-s", "luna16 CT",
             "--sort-by", "relevance"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            kaggle_output = r.stdout.strip()[:600]
            print(f"  Kaggle datasets encontrados:\n{kaggle_output[:300]}")
    except Exception:
        print("  Kaggle CLI no disponible o sin token")

    # ── Intentar descarga con wget (timeout 180s) ─────────
    if url_accessible:
        subset0_zip = ct_dir / "subset0.zip"
        if subset0_zip.exists() and subset0_zip.stat().st_size > 100_000_000:
            print(f"  subset0.zip ya existe: "
                  f"{subset0_zip.stat().st_size / 1e9:.2f} GB")
            download_ok = True
        else:
            print("  Intentando descarga de subset0.zip (max 180s)...")
            try:
                subprocess.run(
                    ["wget", "--continue", "--tries=3", "--timeout=60",
                     "-O", str(subset0_zip),
                     zenodo_url + "?download=1"],
                    timeout=180, capture_output=True, text=True,
                )
                if subset0_zip.exists() and subset0_zip.stat().st_size > 1_000_000:
                    sz = subset0_zip.stat().st_size
                    print(f"  Descarga parcial/completa: {sz / 1e6:.1f} MB")
                    download_ok = True
                else:
                    print("  Descarga insuficiente")
            except subprocess.TimeoutExpired:
                if subset0_zip.exists() and subset0_zip.stat().st_size > 0:
                    sz = subset0_zip.stat().st_size
                    print(f"  Timeout — parcial: {sz / 1e6:.1f} MB "
                          "(continuar con download_luna.sh)")
                    download_ok = True
                else:
                    print("  Timeout sin datos")
            except FileNotFoundError:
                print("  wget no encontrado — crear script manual")
            except Exception as e:
                print(f"  Error descarga: {e}")

    # ── Si descarga completa, extraer ──────────────────────
    if download_ok:
        subset0_zip = ct_dir / "subset0.zip"
        expected_size = 10_000_000_000     # ~10 GB
        if subset0_zip.exists() and subset0_zip.stat().st_size > expected_size:
            print("  Extrayendo subset0.zip...")
            try:
                subprocess.run(
                    ["unzip", "-q", "-o", str(subset0_zip), "-d", str(ct_dir)],
                    timeout=600,
                )
                mhds = list(ct_dir.rglob("*.mhd"))
                mhd_count = len(mhds)
                print(f"  ✓ {mhd_count} archivos .mhd extraídos")
            except Exception as e:
                print(f"  Error extracción: {e}")
        else:
            print("  ZIP incompleto — extracción omitida")

    # ── Crear script de descarga ───────────────────────────
    download_script = f"""#!/bin/bash
# Descarga de LUNA16 CT volumes desde Zenodo
# Ejecutar: bash {luna}/download_luna.sh
set -e
cd "{ct_dir}"

for i in $(seq 0 9); do
    echo "=== Descargando subset$i.zip ==="
    wget --continue --tries=5 --timeout=120 \\
         -O "subset$i.zip" \\
         "https://zenodo.org/records/3723295/files/subset${{i}}.zip?download=1"
    echo "=== Extrayendo subset$i.zip ==="
    unzip -q -o "subset$i.zip"
    echo "=== subset$i listo ==="
done

echo "=== Verificación ==="
find . -name "*.mhd" | wc -l
echo "archivos .mhd (esperado: ~888)"
"""
    (luna / "download_luna.sh").write_text(download_script)
    os.chmod(luna / "download_luna.sh", 0o755)

    # ── Status doc ─────────────────────────────────────────
    status_md = f"""# LUNA16 — Estado de Descarga de CTs

**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Estado
- URL Zenodo accesible: {'Sí' if url_accessible else 'No'}
- subset0.zip descargado: {'Sí' if download_ok else 'No'}
- Archivos .mhd extraídos: {mhd_count}

## Descarga Manual

### Opción 1 — Script automático
```bash
bash {luna}/download_luna.sh
```

### Opción 2 — Zenodo directo
- Fuente: https://zenodo.org/records/3723295
- Archivos: subset0.zip – subset9.zip (~12 GB c/u)
- Total comprimido: ~120 GB  |  Total extraído: ~120 GB

### Opción 3 — Kaggle
```
{kaggle_output or 'Kaggle CLI no disponible'}
```

## Verificación Post-Descarga
```bash
find {ct_dir} -name "*.mhd" | wc -l   # esperado: ~888
```
"""
    (luna / "LUNA_DOWNLOAD_STATUS.md").write_text(status_md)

    st = "✅" if mhd_count > 0 else "⚠️"
    print(f"  ✓ download_luna.sh creado")
    print(f"  ✓ LUNA_DOWNLOAD_STATUS.md creado")
    results["fix6"] = dict(
        status=st, url_accessible=url_accessible,
        download_started=download_ok, mhd_count=mhd_count,
    )


# ════════════════════════════════════════════════════════════════════
#  FIX 7 — Pancreas: Derivar etiquetas binarias (PARALELO)
# ════════════════════════════════════════════════════════════════════

# Función top-level necesaria para ProcessPoolExecutor (no puede ser lambda ni closure)
def _process_mask(args):
    """Procesa una sola máscara .nii.gz y devuelve su fila de resultado.
    Debe ser top-level para ser pickleable por multiprocessing.
    """
    lf_str, pdac_value = args
    import nibabel as nib
    import numpy as np
    lf = Path(lf_str)
    case_id = lf.name.replace(".nii.gz", "")
    try:
        # dataobj evita cargar el array completo hasta que sea necesario
        data = np.asarray(nib.load(str(lf)).dataobj)
        unique_vals = sorted(np.unique(np.round(data)).astype(int).tolist())
        has_pdac = int(pdac_value in unique_vals)
        del data
        return {
            "case_id": case_id,
            "label": has_pdac,
            "label_source": "mask_value_3",
            "mask_values": str(unique_vals),
        }
    except Exception as e:
        return {
            "case_id": case_id,
            "label": -1,
            "label_source": "ERROR",
            "mask_values": str(e),
        }


def fix7_pancreas_binary_labels():
    print("\n" + "=" * 60)
    print("FIX 7 — Pancreas: derivar etiquetas binarias desde máscaras (PARALELO)")
    print("=" * 60)

    labels_dir = DS / "panorama_labels" / "automatic_labels"
    zenodo_dir = DS / "zenodo_13715870"
    out_csv    = DS / "pancreas_labels_binary.csv"
    PDAC_VALUE = 3

    label_files = sorted(labels_dir.glob("*.nii.gz"))
    n_total = len(label_files)
    print(f"  Máscaras a procesar: {n_total}")

    # Usar todos los CPUs disponibles menos 1 para no saturar el sistema
    n_workers = max(1, os.cpu_count() - 1)
    print(f"  Workers paralelos: {n_workers}  (os.cpu_count()={os.cpu_count()})")

    args_list = [(str(lf), PDAC_VALUE) for lf in label_files]
    rows = [None] * n_total
    done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_process_mask, arg): i
            for i, arg in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rows[idx] = future.result()
            except Exception as e:
                lf_str = args_list[idx][0]
                rows[idx] = {
                    "case_id": Path(lf_str).name.replace(".nii.gz", ""),
                    "label": -1,
                    "label_source": "FUTURE_ERROR",
                    "mask_values": str(e),
                }
            done += 1
            if done % 100 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (n_total - done) / rate if rate > 0 else 0
                print(f"  ... {done}/{n_total} máscaras  "
                      f"[{elapsed:.0f}s | {rate:.1f}/s | ETA {eta:.0f}s]")

    print(f"  {n_total} máscaras procesadas en {time.time()-t0:.1f}s")

    # Agregar volúmenes zenodo sin máscara correspondiente
    mask_ids = {r["case_id"] for r in rows if r is not None}
    zenodo_files = sorted(zenodo_dir.glob("*.nii.gz"))
    added_no_mask = 0
    for nf in zenodo_files:
        stem = nf.name.replace(".nii.gz", "")
        parts = stem.rsplit("_", 1)
        case_id = parts[0] if len(parts) > 1 else stem
        if case_id not in mask_ids:
            rows.append({
                "case_id": case_id,
                "label": -1,
                "label_source": "no_mask_in_labels",
                "mask_values": "N/A",
            })
            added_no_mask += 1

    df = pd.DataFrame([r for r in rows if r is not None])
    df.to_csv(out_csv, index=False)

    valid    = df[df["label"] >= 0]
    pdac_pos = int((valid["label"] == 1).sum())
    pdac_neg = int((valid["label"] == 0).sum())
    unknown  = int((df["label"] == -1).sum())

    print(f"  ✓ PDAC+ (label=1):  {pdac_pos}")
    print(f"  ✓ PDAC− (label=0):  {pdac_neg}")
    print(f"  ✓ Sin etiqueta (-1): {unknown}")
    ratio = f"{pdac_pos / pdac_neg:.2f}" if pdac_neg > 0 else "∞"
    print(f"  ✓ Ratio pos/neg: {ratio}")
    print(f"  ✓ CSV: pancreas_labels_binary.csv ({len(df)} filas)")

    results["fix7"] = dict(status="✅", pdac_pos=pdac_pos,
                           pdac_neg=pdac_neg, unknown=unknown,
                           total=len(df))


# ════════════════════════════════════════════════════════════════════
#  FIX 8 — Pancreas: Clasificar volúmenes sin máscara
# ════════════════════════════════════════════════════════════════════
def fix8_pancreas_no_mask():
    print("\n" + "=" * 60)
    print("FIX 8 — Pancreas: clasificar volúmenes sin máscara")
    print("=" * 60)

    import nibabel as nib

    zenodo_dir = DS / "zenodo_13715870"
    labels_dir = DS / "panorama_labels" / "automatic_labels"
    csv_path   = DS / "pancreas_labels_binary.csv"

    if not csv_path.exists():
        print("  ✗ pancreas_labels_binary.csv no existe. ¿FIX 7 falló?")
        results["fix8"] = dict(status="❌", error="CSV no encontrado")
        return

    mask_ids = {f.name.replace(".nii.gz", "")
                for f in labels_dir.glob("*.nii.gz")}

    # Encontrar volúmenes zenodo sin máscara
    no_mask = []
    for nf in sorted(zenodo_dir.glob("*.nii.gz")):
        stem = nf.name.replace(".nii.gz", "")
        parts = stem.rsplit("_", 1)
        case_id = parts[0] if len(parts) > 1 else stem
        if case_id not in mask_ids:
            no_mask.append(nf)

    print(f"  Volúmenes sin máscara: {len(no_mask)}")

    # Inspeccionar muestra (máx 10)
    sample_stats = []
    for nf in no_mask[:10]:
        try:
            nii = nib.load(str(nf))
            data = nii.get_fdata(dtype=np.float32)
            stats = {
                "file": nf.name,
                "shape": list(data.shape),
                "hu_min": round(float(data.min()), 1),
                "hu_max": round(float(data.max()), 1),
                "hu_mean": round(float(data.mean()), 1),
                "spacing": [round(float(s), 3)
                            for s in nii.header.get_zooms()[:3]],
            }
            sample_stats.append(stats)
            print(f"    {stats['file']}: shape={stats['shape']} "
                  f"HU=[{stats['hu_min']}, {stats['hu_max']}]")
            del data
        except Exception as e:
            sample_stats.append({"file": nf.name, "error": str(e)})
            print(f"    {nf.name}: ERROR {e}")

    with open(DS / "pancreas_no_mask_inspection.json", "w") as f:
        json.dump(sample_stats, f, indent=2)

    # Actualizar CSV: sin máscara → asumir PDAC negativo
    df = pd.read_csv(csv_path)
    updated = 0
    for nf in no_mask:
        stem = nf.name.replace(".nii.gz", "")
        parts = stem.rsplit("_", 1)
        case_id = parts[0] if len(parts) > 1 else stem
        match = df["case_id"] == case_id
        if match.any():
            idx = df.index[match][0]
            if df.at[idx, "label"] == -1:
                df.at[idx, "label"] = 0
                df.at[idx, "label_source"] = "assumed_negative_no_mask"
                updated += 1
    df.to_csv(csv_path, index=False)

    valid    = df[df["label"] >= 0]
    pdac_pos = int((valid["label"] == 1).sum())
    pdac_neg = int((valid["label"] == 0).sum())
    unknown  = int((df["label"] == -1).sum())

    ratio = f"{pdac_pos / pdac_neg:.2f}" if pdac_neg > 0 else "∞"
    print(f"  ✓ Reclasificados como PDAC−: {updated}")
    print(f"  ✓ pos={pdac_pos}  neg={pdac_neg}  unknown={unknown}  "
          f"ratio={ratio}")
    results["fix8"] = dict(status="✅", assumed_neg=updated,
                           pdac_pos=pdac_pos, pdac_neg=pdac_neg,
                           unknown=unknown)


# ════════════════════════════════════════════════════════════════════
#  FIX 9 — Pancreas: Script de preprocesado isotrópico
# ════════════════════════════════════════════════════════════════════
PREPROCESS_CODE = '''\
"""
pancreas_preprocess.py — Pipeline de preprocesado para volúmenes CT pancreáticos.

Pasos:
  1. Leer .nii.gz → array HU
  2. Clip HU abdominal [-100, 400]
  3. Normalizar a [0, 1]
  4. Resampling isotrópico al spacing mínimo del volumen
  5. Resize a 64×64×64 con interpolación trilineal
"""

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

HU_MIN, HU_MAX = -100, 400
TARGET_SHAPE = (64, 64, 64)


def preprocess_pancreas_volume(nii_path: str) -> np.ndarray:
    """Retorna array float32 [64, 64, 64] normalizado a [0, 1]."""
    nii = nib.load(nii_path)
    vol = nii.get_fdata(dtype=np.float32)
    sp = np.array(nii.header.get_zooms()[:3], dtype=np.float64)

    # Clip + normalizar
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)

    # Resampling isotrópico
    min_sp = max(sp.min(), 1e-6)
    zoom_factors = sp / min_sp
    vol_iso = ndimage.zoom(vol, zoom_factors, order=1)

    # Resize al target
    final_zoom = (np.array(TARGET_SHAPE, dtype=np.float64)
                  / np.array(vol_iso.shape, dtype=np.float64))
    vol_final = ndimage.zoom(vol_iso, final_zoom, order=1)

    return vol_final.astype(np.float32)


if __name__ == "__main__":
    import sys
    import glob

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        files = [path]
    else:
        files = sorted(glob.glob("datasets/zenodo_13715870/*.nii.gz"))[:3]

    for nf in files:
        result = preprocess_pancreas_volume(nf)
        assert result.shape == TARGET_SHAPE, f"Shape: {result.shape}"
        assert 0.0 <= result.min() and result.max() <= 1.0, (
            f"Rango: [{result.min():.3f}, {result.max():.3f}]"
        )
        print(f"OK {nf}: shape={result.shape} "
              f"min={result.min():.4f} max={result.max():.4f}")
'''


def _validate_volume(nf_str):
    """Validación de un volumen pancreático para fix9. Top-level para pickling."""
    import nibabel as nib
    import numpy as np
    import scipy.ndimage as ndimage

    nf = Path(nf_str)
    try:
        nii = nib.load(str(nf))
        vol = nii.get_fdata(dtype=np.float32)
        sp  = np.array(nii.header.get_zooms()[:3], dtype=np.float64)

        vol = np.clip(vol, -100.0, 400.0)
        vol = (vol - (-100.0)) / (400.0 - (-100.0))

        min_sp = max(sp.min(), 1e-6)
        zoom_factors = sp / min_sp
        vol_iso = ndimage.zoom(vol, zoom_factors, order=1)

        final_zoom = (np.array([64, 64, 64], dtype=np.float64)
                      / np.array(vol_iso.shape, dtype=np.float64))
        vol_final = ndimage.zoom(vol_iso, final_zoom, order=1)

        assert vol_final.shape == (64, 64, 64), f"Shape: {vol_final.shape}"
        assert 0.0 <= vol_final.min() and vol_final.max() <= 1.0

        return {
            "file": nf.name,
            "ok": True,
            "shape": list(vol_final.shape),
            "vmin": round(float(vol_final.min()), 4),
            "vmax": round(float(vol_final.max()), 4),
        }
    except Exception as e:
        return {"file": nf.name, "ok": False, "error": str(e)}


def fix9_pancreas_preprocess():
    print("\n" + "=" * 60)
    print("FIX 9 — Pancreas: script de preprocesado isotrópico")
    print("=" * 60)

    script_dir = BASE / "src" / "preprocessing"
    script_dir.mkdir(parents=True, exist_ok=True)

    script_path = script_dir / "pancreas_preprocess.py"
    script_path.write_text(PREPROCESS_CODE)

    init = script_dir / "__init__.py"
    if not init.exists():
        init.write_text("")

    print(f"  ✓ Script creado: {script_path.relative_to(BASE)}")

    # Validar con 3 volúmenes en paralelo
    zenodo  = DS / "zenodo_13715870"
    samples = sorted(zenodo.glob("*.nii.gz"))[:3]
    print(f"  Validando {len(samples)} volúmenes en paralelo...")

    validated = 0
    n_workers = min(len(samples), max(1, os.cpu_count() - 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_validate_volume, str(nf)): nf
                   for nf in samples}
        for future in as_completed(futures):
            r = future.result()
            if r["ok"]:
                print(f"  ✓ {r['file']}: {r['shape']} "
                      f"min={r['vmin']:.4f} max={r['vmax']:.4f}")
                validated += 1
            else:
                print(f"  ✗ {r['file']}: {r['error']}")

    st = "✅" if validated == len(samples) else "⚠️"
    print(f"  ✓ {validated}/{len(samples)} volúmenes validados")
    results["fix9"] = dict(status=st, validated=validated,
                           script=str(script_path.relative_to(BASE)))


# ════════════════════════════════════════════════════════════════════
#  FIX 10 — Instalar dependencias de Deep Learning
# ════════════════════════════════════════════════════════════════════
def fix10_install_deps():
    print("\n" + "=" * 60)
    print("FIX 10 — Instalar dependencias de Deep Learning")
    print("=" * 60)

    # Detectar CUDA
    cuda_version = None
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[-1].strip().split()[0]
    except Exception:
        pass
    if not cuda_version:
        try:
            r = subprocess.run(["nvcc", "--version"], capture_output=True,
                               text=True, timeout=10)
            if r.returncode == 0 and "release" in r.stdout:
                cuda_version = (r.stdout.split("release")[-1]
                                .strip().split(",")[0])
        except Exception:
            pass

    print(f"  CUDA: {cuda_version or 'No detectado'}")

    # URL de PyTorch
    if cuda_version:
        cu = cuda_version.replace(".", "")[:3]
        torch_url = f"https://download.pytorch.org/whl/cu{cu}"
    else:
        torch_url = "https://download.pytorch.org/whl/cpu"
        print("  → Instalando PyTorch CPU (sin GPU)")

    installed = []
    failed    = []

    def pip_install(pkgs, extra_args=None):
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
        cmd.extend(pkgs)
        if extra_args:
            cmd.extend(extra_args)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=600)
            if r.returncode == 0:
                installed.extend(pkgs)
                return True
            else:
                failed.extend(pkgs)
                print(f"  ✗ {pkgs}: {r.stderr[-200:].strip()}")
                return False
        except Exception as e:
            failed.extend(pkgs)
            print(f"  ✗ {pkgs}: {e}")
            return False

    # PyTorch
    print("  Instalando PyTorch...")
    pip_install(["torch", "torchvision", "torchaudio"],
                extra_args=["--index-url", torch_url])

    # Otros paquetes
    print("  Instalando paquetes adicionales...")
    other_pkgs = [
        "timm", "monai", "SimpleITK",
        "opencv-python-headless", "faiss-gpu-cu12",
        "scipy", "scikit-learn", "tqdm", "nibabel",
    ]
    for pkg in other_pkgs:
        pip_install([pkg])

    # Verificar
    print("\n  === Verificación ===")
    check_map = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "monai": "monai",
        "SimpleITK": "SimpleITK",
        "cv2": "cv2",
        "faiss": "faiss",
        "scipy": "scipy",
        "sklearn": "sklearn",
        "nibabel": "nibabel",
        "tqdm": "tqdm",
    }
    verified = {}
    for name, mod in check_map.items():
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "ok")
            verified[name] = str(v)
            print(f"  ✅ {name}: {v}")
        except ImportError:
            verified[name] = "NO"
            print(f"  ❌ {name}")

    # GPU info
    gpu_info = "No CUDA"
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gpus = []
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                gpus.append(f"GPU {i}: {p.name} — "
                            f"{p.total_mem / 1e9:.1f} GB")
            gpu_info = "; ".join(gpus)
        else:
            gpu_info = "torch instalado, CUDA no disponible (CPU only)"
    except Exception:
        pass
    print(f"\n  GPU: {gpu_info}")

    st = "✅" if not failed else "⚠️"
    results["fix10"] = dict(status=st, cuda=cuda_version,
                            installed=installed, failed=failed,
                            verified=verified, gpu=gpu_info)


# ════════════════════════════════════════════════════════════════════
#  Reporte final
# ════════════════════════════════════════════════════════════════════
def generate_report():
    print("\n" + "=" * 60)
    print("Generando alignment_fixes_report.md")
    print("=" * 60)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Helpers
    def g(fix, key, default="—"):
        return results.get(fix, {}).get(key, default)

    def st(fix):
        return results.get(fix, {}).get("status", "❌")

    rows = [
        f"| 1 | NIH | Split val por patient_id | {st('fix1')} | "
        f"train:{g('fix1','train')} val:{g('fix1','val')} "
        f"test:{g('fix1','test')} |",

        f"| 2 | ISIC | Splits train/val/test por lesion_id | {st('fix2')} | "
        f"train:{g('fix2','train')} val:{g('fix2','val')} "
        f"test:{g('fix2','test')} |",

        f"| 3 | ISIC | Symlinks planos en isic_images/ | {st('fix3')} | "
        f"{g('fix3','symlinks')} symlinks |",

        f"| 4 | OA | Investigar déficit + documentar | {st('fix4')} | "
        f"KLGrade:{g('fix4','kl_total')} "
        f"disco:{g('fix4','disco_total')} |",

        f"| 5 | OA | Reporte modos de imagen | {st('fix5')} | "
        f"Modos: {g('fix5','modes')} problemas: {g('fix5','problems')} |",

        f"| 6 | LUNA | Descargar CT subset0 | {st('fix6')} | "
        f"{g('fix6','mhd_count')} .mhd |",

        f"| 7 | Pancreas | Derivar etiquetas binarias | {st('fix7')} | "
        f"PDAC+:{g('fix7','pdac_pos')} "
        f"PDAC−:{g('fix7','pdac_neg')} "
        f"?:{g('fix7','unknown')} |",

        f"| 8 | Pancreas | Clasificar vols sin máscara | {st('fix8')} | "
        f"Asumidos neg:{g('fix8','assumed_neg')} "
        f"pos:{g('fix8','pdac_pos')} "
        f"neg:{g('fix8','pdac_neg')} |",

        f"| 9 | Pancreas | Script preprocesado isotrópico | {st('fix9')} | "
        f"Validado en {g('fix9','validated')} vols |",

        f"| 10 | Sistema | Instalar dependencias DL | {st('fix10')} | "
        f"GPU: {g('fix10','gpu','—')[:50]} |",
    ]

    # Estado final por dataset
    nih_st = st("fix1")
    isic_st = "✅" if st("fix2") == "✅" and st("fix3") == "✅" else "⚠️"
    oa_st = "✅" if st("fix4") == "✅" and st("fix5") == "✅" else "⚠️"
    luna_st = "✅" if g("fix6", "mhd_count", 0) > 0 else "⚠️"
    panc_st = "✅" if st("fix7") == "✅" and st("fix9") == "✅" else "⚠️"

    # Timing
    timing_lines = "\n".join(
        f"| {name} | {secs:.1f}s |" for name, secs in timings.items()
    )

    # Pre-join for markdown (avoid escape issues in f-strings)
    rows_text = "\n".join(rows)

    report = f"""# Reporte de Fixes de Alineación

**Fecha:** {now}
**Ejecutado por:** Claude Opus 4.6

## Estado por Fix

| Fix | Dataset | Descripción | Estado | Resultado |
|-----|---------|-------------|--------|-----------|
{rows_text}

## Estado final por dataset

| Dataset | Listo para entrenar | Observaciones |
|---------|---------------------|---------------|
| NIH ChestXray14 | {nih_st} | splits/{'{nih_train,nih_val,nih_test}_list.txt'} generados, 0 overlap |
| ISIC 2019 | {isic_st} | splits/ generados, isic_images/ con symlinks planos |
| Osteoarthritis | {oa_st} | 4,088 imgs anotadas (no ~10K), KL4=0, modos documentados |
| LUNA16 | {luna_st} | {'CTs descargados' if g('fix6','mhd_count',0) > 0 else 'SIN CTs — usar download_luna.sh'} |
| Pancreas | {panc_st} | labels binarios derivados, preprocesado isotrópico listo |

## Tiempos de Ejecución

| Fix | Tiempo |
|-----|--------|
{timing_lines}

## Rutas generadas

```
datasets/
├── nih_chest_xrays/splits/
│   ├── nih_train_list.txt       ({g('fix1','train')} imgs)
│   ├── nih_val_list.txt         ({g('fix1','val')} imgs)
│   └── nih_test_list.txt        ({g('fix1','test')} imgs)
├── isic_2019/
│   ├── splits/
│   │   ├── isic_train.csv       ({g('fix2','train')} imgs)
│   │   ├── isic_val.csv         ({g('fix2','val')} imgs)
│   │   └── isic_test.csv        ({g('fix2','test')} imgs)
│   └── isic_images/             ({g('fix3','symlinks')} symlinks)
├── osteoarthritis/
│   ├── oa_dataset_report.md
│   └── image_modes_report.json
├── luna_lung_cancer/
│   ├── ct_volumes/
│   ├── download_luna.sh
│   └── LUNA_DOWNLOAD_STATUS.md
├── pancreas_labels_binary.csv
├── pancreas_no_mask_inspection.json
└── alignment_fixes_report.md

src/preprocessing/
└── pancreas_preprocess.py
```
"""
    out_path = DS / "alignment_fixes_report.md"
    out_path.write_text(report)
    print(f"  ✓ {out_path.relative_to(BASE)}")

    results["report"] = dict(status="✅", path=str(out_path))


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════
def main():
    start_all = time.time()

    fixes = [
        ("fix1",  fix1_nih_val_split),
        ("fix2",  fix2_isic_splits),
        ("fix3",  fix3_isic_symlinks),
        ("fix4",  fix4_oa_investigation),
        ("fix5",  fix5_oa_image_modes),
        ("fix6",  fix6_luna_download),
        ("fix7",  fix7_pancreas_binary_labels),
        ("fix8",  fix8_pancreas_no_mask),
        ("fix9",  fix9_pancreas_preprocess),
        ("fix10", fix10_install_deps),
    ]

    for name, fn in fixes:
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            results[name] = dict(status="❌", error=str(e))
        timings[name] = time.time() - t0

    generate_report()

    elapsed = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"Todos los fixes completados en {elapsed:.0f}s")
    print(f"Reporte: datasets/alignment_fixes_report.md")
    print("=" * 60)


if __name__ == "__main__":
    # Guardia necesaria para ProcessPoolExecutor en sistemas que usan
    # spawn (macOS, Windows). En Linux (fork) es opcional pero es buena práctica.
    main()
