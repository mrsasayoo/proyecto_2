#!/usr/bin/env python3
"""
run_pipeline.py — Orquestador maestro del pipeline MoE médico
=============================================================

Ejecuta las 6 fases del sistema en orden secuencial con:
  - Idempotencia: cada fase detecta si ya completó y la omite
  - Manejo de errores: fallo en una fase detiene el pipeline (o continúa con --skip-errors)
  - Logging centralizado: un log por fase + resumen final

Fases:
  0 — Preparación de datos     [fase0/fase0_pipeline.py]
  1 — Extracción de embeddings [fase1/fase1_pipeline.py]
  2 — Ablation router          [fase2/fase2_pipeline.py]
  3 — Entrenamiento CAE (OOD)  [fase3/train_cae.py]
  5 — Fine-tuning global MoE   [fase5/fase5_finetune_global.py]
  6 — Inferencia y evaluación  [fase6/inference_engine.py]

Uso:
  # Pipeline completo
  python src/pipeline/run_pipeline.py

  # Dry-run (verifica dependencias, no ejecuta)
  python src/pipeline/run_pipeline.py --dry-run

  # Ejecutar solo fases específicas
  python src/pipeline/run_pipeline.py --only 0 1 2

  # Saltar una fase (ya completada manualmente)
  python src/pipeline/run_pipeline.py --skip 0

  # Continuar desde una fase específica
  python src/pipeline/run_pipeline.py --from-fase 2

  # No detener el pipeline si una fase falla
  python src/pipeline/run_pipeline.py --skip-errors
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent  # src/pipeline/
_PROJECT_ROOT = _THIS_DIR.parent.parent  # proyecto_2/
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_log_path = _LOGS_DIR / f"run_pipeline_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger("run_pipeline")

# ── Definición de fases ────────────────────────────────────────────────────
# Cada fase tiene:
#   id        : identificador numérico (entero)
#   name      : nombre legible
#   script    : ruta al script de entrada (relativa a src/pipeline/)
#   done_file : archivo cuya existencia indica que la fase ya completó (opcional)
#   extra_args: argumentos adicionales pasados al script

PHASES: list[dict] = [
    {
        "id": 0,
        "name": "Preparación de datos",
        "script": "fase0/fase0_pipeline.py",
        "done_file": None,  # fase0_pipeline.py tiene su propia idempotencia interna
        "extra_args": [],
    },
    {
        "id": 1,
        "name": "Extracción de embeddings",
        "script": "fase1/fase1_pipeline.py",
        "done_file": None,  # fase1_pipeline.py usa --force para re-extracción
        "extra_args": [],
    },
    {
        "id": 2,
        "name": "Ablation router",
        "script": "fase2/fase2_pipeline.py",
        "done_file": None,
        "extra_args": [],
    },
    {
        "id": 3,
        "name": "Entrenamiento CAE (OOD)",
        "script": "fase3/train_cae.py",
        "done_file": None,
        "extra_args": [],
    },
    {
        "id": 5,
        "name": "Fine-tuning global MoE",
        "script": "fase5/fase5_finetune_global.py",
        "done_file": None,
        "extra_args": [],
    },
    {
        "id": 6,
        "name": "Inferencia y evaluación",
        "script": "fase6/inference_engine.py",
        "done_file": None,
        "extra_args": [],
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_phase(phase: dict, dry_run: bool, skip_errors: bool) -> dict:
    """Ejecuta una fase como subproceso y retorna su resultado."""
    phase_id = phase["id"]
    phase_name = phase["name"]
    script = _THIS_DIR / phase["script"]
    done_file = phase.get("done_file")

    result = {
        "id": phase_id,
        "name": phase_name,
        "status": "unknown",
        "duration_s": 0.0,
        "returncode": None,
        "skipped_reason": None,
    }

    # Idempotencia: omitir si el archivo de completado ya existe
    if done_file and Path(done_file).exists():
        log.info(
            "[Fase %d] ── %s — ya completada (found: %s)",
            phase_id,
            phase_name,
            done_file,
        )
        result["status"] = "skipped_done"
        result["skipped_reason"] = f"done_file exists: {done_file}"
        return result

    if not script.exists():
        msg = f"Script no encontrado: {script}"
        log.error("[Fase %d] %s", phase_id, msg)
        result["status"] = "error"
        result["skipped_reason"] = msg
        if not skip_errors:
            raise FileNotFoundError(msg)
        return result

    cmd = [sys.executable, str(script)] + phase.get("extra_args", [])
    if dry_run:
        cmd.append("--dry-run")

    log.info("[Fase %d] ── %s — iniciando", phase_id, phase_name)
    log.info("[Fase %d]    cmd: %s", phase_id, " ".join(cmd))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=False,  # permite ver output en tiempo real
            timeout=7200,  # 2 horas máximo por fase
        )
        duration = time.monotonic() - t0
        result["duration_s"] = round(duration, 1)
        result["returncode"] = proc.returncode

        if proc.returncode == 0:
            log.info(
                "[Fase %d] ── %s — COMPLETADA en %.1fs",
                phase_id,
                phase_name,
                duration,
            )
            result["status"] = "ok"
        else:
            msg = f"Fase {phase_id} terminó con returncode={proc.returncode}"
            log.error("[Fase %d] %s", phase_id, msg)
            result["status"] = "failed"
            if not skip_errors:
                raise RuntimeError(msg)

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        result["duration_s"] = round(duration, 1)
        msg = f"Fase {phase_id} superó el timeout de 7200s"
        log.error("[Fase %d] %s", phase_id, msg)
        result["status"] = "timeout"
        if not skip_errors:
            raise TimeoutError(msg)

    except Exception as exc:
        duration = time.monotonic() - t0
        result["duration_s"] = round(duration, 1)
        log.error("[Fase %d] Excepción: %s", phase_id, exc)
        result["status"] = "error"
        result["skipped_reason"] = str(exc)
        if not skip_errors:
            raise

    return result


def _print_summary(results: list[dict], total_s: float) -> None:
    """Imprime tabla de resumen al final del pipeline."""
    log.info("")
    log.info("══════════════════════════════════════════════════════")
    log.info("  RESUMEN DEL PIPELINE")
    log.info("══════════════════════════════════════════════════════")
    log.info("  %-5s %-35s %-12s %s", "FASE", "NOMBRE", "ESTADO", "DURACIÓN")
    log.info("  %s", "─" * 65)
    for r in results:
        status_icon = {
            "ok": "✅",
            "skipped": "⏭️ ",
            "skipped_done": "✅",
            "failed": "❌",
            "timeout": "⏱️ ",
            "error": "🔥",
        }.get(r["status"], "❓")
        log.info(
            "  %-5s %-35s %-12s %.1fs",
            f"[{r['id']}]",
            r["name"][:35],
            f"{status_icon} {r['status']}",
            r["duration_s"],
        )
    log.info("  %s", "─" * 65)
    log.info("  Total: %.1fs (%.1f min)", total_s, total_s / 60)
    log.info("══════════════════════════════════════════════════════")
    log.info("  Log guardado en: %s", _log_path)
    log.info("")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orquestador maestro del pipeline MoE médico.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pasa --dry-run a cada fase; no ejecuta entrenamiento real.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        metavar="N",
        help="Ejecutar solo las fases indicadas (ej. --only 0 1 2).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        type=int,
        metavar="N",
        help="Saltar las fases indicadas (ej. --skip 0).",
    )
    parser.add_argument(
        "--from-fase",
        type=int,
        metavar="N",
        help="Empezar desde la fase N (omitir todas las anteriores).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continuar el pipeline aunque una fase falle.",
    )
    args = parser.parse_args()

    # Filtrar fases según flags
    phases = list(PHASES)
    all_ids = [p["id"] for p in phases]

    if args.only:
        invalid = [i for i in args.only if i not in all_ids]
        if invalid:
            log.error(
                "Fases inválidas en --only: %s. Disponibles: %s", invalid, all_ids
            )
            return 1
        phases = [p for p in phases if p["id"] in args.only]

    if args.skip:
        phases = [p for p in phases if p["id"] not in args.skip]

    if args.from_fase is not None:
        phases = [p for p in phases if p["id"] >= args.from_fase]

    if not phases:
        log.error("No hay fases a ejecutar con los filtros proporcionados.")
        return 1

    mode = "DRY-RUN" if args.dry_run else "REAL"
    log.info("════════════════════════════════════════════════════════")
    log.info("  PIPELINE MOE MÉDICO — %s", mode)
    log.info("  Fases a ejecutar: %s", [p["id"] for p in phases])
    log.info("  Proyecto: %s", _PROJECT_ROOT)
    log.info("  Inicio: %s", datetime.now(timezone.utc).isoformat())
    log.info("════════════════════════════════════════════════════════")

    t_global = time.monotonic()
    results: list[dict] = []
    exit_code = 0

    for phase in phases:
        try:
            result = _run_phase(
                phase, dry_run=args.dry_run, skip_errors=args.skip_errors
            )
            results.append(result)
            if result["status"] in ("failed", "timeout", "error"):
                exit_code = 1
        except Exception as exc:
            log.critical("Pipeline abortado en Fase %d: %s", phase["id"], exc)
            results.append(
                {
                    "id": phase["id"],
                    "name": phase["name"],
                    "status": "aborted",
                    "duration_s": 0.0,
                    "returncode": None,
                    "skipped_reason": str(exc),
                }
            )
            exit_code = 1
            break

    total_s = time.monotonic() - t_global
    _print_summary(results, total_s)

    # Guardar reporte JSON
    report_path = _LOGS_DIR / f"pipeline_report_{_ts}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "total_duration_s": round(total_s, 1),
                "phases": results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    log.info("Reporte JSON guardado en: %s", report_path)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
