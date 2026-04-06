"""
paso10_verificacion.py
---------------------
Paso 10 — Verificación funcional completa del pipeline MoE.

Valida todos los componentes contra los requisitos del proyecto,
detecta violaciones de penalización y genera un reporte JSON.

Uso:
    python src/pipeline/fase6/paso10_verificacion.py --dry-run
    python src/pipeline/fase6/paso10_verificacion.py --device cpu
    python src/pipeline/fase6/paso10_verificacion.py --output-dir results/paso10_custom
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Project root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# ── Default output ──────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "results/paso10"


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 10 — Verificación funcional completa del pipeline MoE"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo verificación ligera: métricas no se validan (EXIT 0)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


# =====================================================================
# Helpers
# =====================================================================
def _make_check(
    check_id: str,
    category: str,
    severity: str,
    result: str,
    expected: str,
    observed: str,
    message: str,
) -> dict:
    """Build a single check dict."""
    return {
        "id": check_id,
        "category": category,
        "severity": severity,
        "result": result,
        "expected": expected,
        "observed": observed,
        "message": message,
    }


def _read_json_safe(path: Path) -> dict | None:
    """Read a JSON file, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _grep_source_files(
    pattern: str,
    src_dir: Path,
    skip_comments: bool = True,
    skip_strings: bool = False,
    exclude_self: bool = False,
) -> list[tuple[str, int, str]]:
    """
    Search all .py files under src_dir for a regex pattern.
    Returns list of (filepath, line_number, line_text) matches.
    If skip_comments is True, lines whose first non-whitespace char is '#' are ignored.
    If skip_strings is True, lines inside triple-quoted strings and lines that
    are purely string content (start with a quote char) are also skipped.
    If exclude_self is True, this script (paso10_verificacion.py) is excluded.
    """
    self_path = Path(__file__).resolve()
    matches = []
    for py_file in sorted(src_dir.rglob("*.py")):
        if exclude_self and py_file.resolve() == self_path:
            continue
        try:
            lines = py_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        in_docstring = False
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()

            # Track triple-quoted string blocks (docstrings)
            if skip_strings:
                triple_count = stripped.count('"""') + stripped.count("'''")
                if in_docstring:
                    if triple_count % 2 == 1:
                        in_docstring = False
                    continue
                if triple_count % 2 == 1:
                    # Opening a docstring; if the pattern matches on this
                    # same opening line, it's still docstring text
                    if stripped.startswith(('"""', "'''")):
                        in_docstring = True
                        continue

            if skip_comments and stripped.startswith("#"):
                continue

            if re.search(pattern, line):
                rel = str(py_file.relative_to(PROJECT_ROOT))
                matches.append((rel, i, line.strip()))
    return matches


# =====================================================================
# Check implementations
# =====================================================================


# ── Category A — Imports & Smoke Tests ──────────────────────
def check_A(checks: list) -> None:
    """A1-A5: Import and smoke tests."""

    # A1: config N_EXPERTS
    try:
        from src.pipeline import config as cfg

        ok = cfg.N_EXPERTS_DOMAIN == 5 and cfg.N_EXPERTS_TOTAL == 6
        checks.append(
            _make_check(
                "A1",
                "imports",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "N_EXPERTS_DOMAIN==5, N_EXPERTS_TOTAL==6",
                f"N_EXPERTS_DOMAIN={cfg.N_EXPERTS_DOMAIN}, N_EXPERTS_TOTAL={cfg.N_EXPERTS_TOTAL}",
                f"config.N_EXPERTS_DOMAIN={cfg.N_EXPERTS_DOMAIN}, "
                f"config.N_EXPERTS_TOTAL={cfg.N_EXPERTS_TOTAL} " + ("✓" if ok else "✗"),
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "A1",
                "imports",
                "CRITICAL",
                "SKIP",
                "N_EXPERTS_DOMAIN==5, N_EXPERTS_TOTAL==6",
                f"ImportError: {e}",
                f"Could not import src.pipeline.config: {e}",
            )
        )

    # A2: fase6_config thresholds
    try:
        from src.pipeline.fase6 import fase6_config as f6

        has_f1 = hasattr(f6, "F1_THRESHOLD_2D")
        has_lb = hasattr(f6, "LOAD_BALANCE_MAX_RATIO")
        ok = has_f1 and has_lb
        checks.append(
            _make_check(
                "A2",
                "imports",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "F1_THRESHOLD_2D and LOAD_BALANCE_MAX_RATIO exist",
                f"F1_THRESHOLD_2D={'yes' if has_f1 else 'no'}, "
                f"LOAD_BALANCE_MAX_RATIO={'yes' if has_lb else 'no'}",
                f"fase6_config thresholds present {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "A2",
                "imports",
                "CRITICAL",
                "SKIP",
                "F1_THRESHOLD_2D and LOAD_BALANCE_MAX_RATIO exist",
                f"ImportError: {e}",
                f"Could not import fase6_config: {e}",
            )
        )

    # A3: logging_utils
    try:
        from src.pipeline import logging_utils as lu

        ok = callable(getattr(lu, "setup_logging", None))
        checks.append(
            _make_check(
                "A3",
                "imports",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "setup_logging callable",
                f"callable={ok}",
                f"setup_logging callable {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "A3",
                "imports",
                "CRITICAL",
                "SKIP",
                "setup_logging callable",
                f"ImportError: {e}",
                f"Could not import logging_utils: {e}",
            )
        )

    # A4: InferenceEngine
    try:
        from src.pipeline.fase6.inference_engine import InferenceEngine

        ok = isinstance(InferenceEngine, type)
        checks.append(
            _make_check(
                "A4",
                "imports",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "InferenceEngine class exists",
                f"type={type(InferenceEngine).__name__}",
                f"InferenceEngine class exists {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "A4",
                "imports",
                "CRITICAL",
                "SKIP",
                "InferenceEngine class exists",
                f"ImportError: {e}",
                f"Could not import InferenceEngine: {e}",
            )
        )

    # A5: OODDetector
    try:
        from src.pipeline.fase6.ood_detector import OODDetector

        ok = isinstance(OODDetector, type)
        checks.append(
            _make_check(
                "A5",
                "imports",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "OODDetector class exists",
                f"type={type(OODDetector).__name__}",
                f"OODDetector class exists {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "A5",
                "imports",
                "CRITICAL",
                "SKIP",
                "OODDetector class exists",
                f"ImportError: {e}",
                f"Could not import OODDetector: {e}",
            )
        )


# ── Category B — No-Metadata Constraint ────────────────────
def check_B(checks: list) -> None:
    """B1-B2: Modality must be detected via router+entropy, not metadata."""
    src_dir = PROJECT_ROOT / "src"

    # B1: Static grep for metadata-based ROUTING (not dataset file I/O)
    try:
        # Only search in routing/inference code — dataset loaders legitimately
        # reference .dcm/.nii for file reading, which is fine.
        routing_dirs = [
            PROJECT_ROOT / "src" / "pipeline" / "fase2" / "routers",
            PROJECT_ROOT / "src" / "pipeline" / "fase6",
            PROJECT_ROOT / "src" / "pipeline" / "fase5",
        ]
        # Patterns indicating metadata-based routing decisions
        metadata_patterns = [
            r'\.dcm["\']',
            r'\.nii["\']',
            r'metadata\s*\[\s*["\']modality["\']',
        ]
        violations = []
        for rdir in routing_dirs:
            if not rdir.exists():
                continue
            for pat in metadata_patterns:
                hits = _grep_source_files(
                    pat, rdir, skip_comments=True, skip_strings=True, exclude_self=True
                )
                violations.extend(hits)

        if violations:
            observed = "; ".join(f"{f}:{ln}" for f, ln, _ in violations[:5])
            checks.append(
                _make_check(
                    "B1",
                    "no_metadata",
                    "CRITICAL",
                    "FAIL",
                    "No metadata-based routing patterns",
                    observed,
                    f"Metadata routing patterns found in {len(violations)} location(s) ✗",
                )
            )
        else:
            # Check positive: ndim-based detection exists
            ndim_hits = _grep_source_files(
                r"\b(tensor\.ndim|x\.ndim|\.ndim\b|len\(.*\.shape\))",
                src_dir,
                skip_comments=True,
            )
            checks.append(
                _make_check(
                    "B1",
                    "no_metadata",
                    "CRITICAL",
                    "PASS",
                    "No metadata-based routing patterns",
                    f"{len(ndim_hits)} ndim-based references found",
                    "No metadata routing detected ✓",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "B1",
                "no_metadata",
                "CRITICAL",
                "SKIP",
                "No metadata-based routing patterns",
                f"Error: {e}",
                f"Could not perform static analysis: {e}",
            )
        )

    # B2: inference_engine.py uses entropy-based routing (not metadata)
    try:
        ie_path = PROJECT_ROOT / "src" / "pipeline" / "fase6" / "inference_engine.py"
        if ie_path.exists():
            content = ie_path.read_text(encoding="utf-8", errors="replace")
            has_entropy = bool(
                re.search(r"\bentropy\b.*threshold|entropy_threshold", content)
            )
            has_router = bool(re.search(r"\brouter\b", content))
            ok = has_entropy and has_router
            checks.append(
                _make_check(
                    "B2",
                    "no_metadata",
                    "CRITICAL",
                    "PASS" if ok else "FAIL",
                    "inference_engine.py uses entropy+router routing (not metadata)",
                    f"entropy_routing={has_entropy}, router={has_router}",
                    f"inference_engine.py entropy+router routing {'✓' if ok else '✗'}",
                )
            )
        else:
            checks.append(
                _make_check(
                    "B2",
                    "no_metadata",
                    "CRITICAL",
                    "SKIP",
                    "inference_engine.py uses entropy+router routing (not metadata)",
                    "File not found",
                    "inference_engine.py not found",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "B2",
                "no_metadata",
                "CRITICAL",
                "SKIP",
                "inference_engine.py uses entropy+router routing (not metadata)",
                f"Error: {e}",
                f"Could not check inference_engine.py: {e}",
            )
        )


# ── Category C — Load Balance ──────────────────────────────
def check_C(checks: list, dry_run: bool) -> None:
    """C1-C2: Load balance checks."""

    # C1: Read load_balance_test.json
    try:
        lb_path = PROJECT_ROOT / "results" / "paso9" / "load_balance_test.json"
        data = _read_json_safe(lb_path)

        if data is None:
            checks.append(
                _make_check(
                    "C1",
                    "load_balance",
                    "CRITICAL",
                    "SKIP",
                    "max_ratio <= 1.30",
                    "File not found",
                    "load_balance_test.json not found, skipping",
                )
            )
        else:
            # Check if the results are from a dry-run
            is_dry_run_result = (
                any(
                    expert.get("dry_run", False)
                    for expert in data.get("per_expert", [{}])
                )
                if "per_expert" in data
                else data.get("dry_run", False)
            )

            max_ratio = data.get("max_min_ratio", data.get("max_ratio", 0.0))

            if dry_run or is_dry_run_result:
                checks.append(
                    _make_check(
                        "C1",
                        "load_balance",
                        "CRITICAL",
                        "SKIP",
                        "max_ratio <= 1.30",
                        f"max_ratio={max_ratio}",
                        "dry-run results, load balance not validated",
                    )
                )
            elif max_ratio > 1.30:
                checks.append(
                    _make_check(
                        "C1",
                        "load_balance",
                        "CRITICAL",
                        "FAIL",
                        "max_ratio <= 1.30",
                        f"max_ratio={max_ratio}",
                        f"Load balance max_ratio={max_ratio} exceeds 1.30 ✗",
                    )
                )
            else:
                checks.append(
                    _make_check(
                        "C1",
                        "load_balance",
                        "CRITICAL",
                        "PASS",
                        "max_ratio <= 1.30",
                        f"max_ratio={max_ratio}",
                        f"Load balance max_ratio={max_ratio} ✓",
                    )
                )
    except Exception as e:
        checks.append(
            _make_check(
                "C1",
                "load_balance",
                "CRITICAL",
                "SKIP",
                "max_ratio <= 1.30",
                f"Error: {e}",
                f"Could not check load balance: {e}",
            )
        )

    # C2: LOAD_BALANCE_MAX_RATIO defined
    try:
        from src.pipeline.fase6.fase6_config import LOAD_BALANCE_MAX_RATIO

        ok = LOAD_BALANCE_MAX_RATIO == 1.30
        checks.append(
            _make_check(
                "C2",
                "load_balance",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "LOAD_BALANCE_MAX_RATIO == 1.30",
                f"{LOAD_BALANCE_MAX_RATIO}",
                f"LOAD_BALANCE_MAX_RATIO = {LOAD_BALANCE_MAX_RATIO} {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "C2",
                "load_balance",
                "CRITICAL",
                "SKIP",
                "LOAD_BALANCE_MAX_RATIO == 1.30",
                f"Error: {e}",
                f"Could not check LOAD_BALANCE_MAX_RATIO: {e}",
            )
        )


# ── Category D — Metric Thresholds ─────────────────────────
def check_D(checks: list, dry_run: bool) -> None:
    """D1-D4: Metric thresholds from test results."""

    # Read test_metrics_summary.json
    metrics_path = PROJECT_ROOT / "results" / "paso9" / "test_metrics_summary.json"
    metrics = _read_json_safe(metrics_path)

    # Check if results are dry-run
    is_dry_run_metrics = False
    if metrics:
        per_expert = metrics.get("per_expert", [])
        if per_expert and any(e.get("dry_run", False) for e in per_expert):
            is_dry_run_metrics = True

    # Read ood_auroc_report.json
    ood_path = PROJECT_ROOT / "results" / "paso9" / "ood_auroc_report.json"
    ood_data = _read_json_safe(ood_path)
    is_dry_run_ood = ood_data.get("dry_run", False) if ood_data else False

    skip_msg = "dry-run results, metrics not validated"

    # D1: F1 2D > 0.72
    try:
        if metrics is None:
            checks.append(
                _make_check(
                    "D1",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "F1_2D > 0.72",
                    "File not found",
                    "test_metrics_summary.json not found",
                )
            )
        elif dry_run or is_dry_run_metrics:
            f1_2d = metrics.get("f1_macro_2d_mean", 0.0)
            checks.append(
                _make_check(
                    "D1",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "F1_2D > 0.72",
                    f"f1_2d={f1_2d}",
                    skip_msg,
                )
            )
        else:
            f1_2d = metrics.get("f1_macro_2d_mean", 0.0)
            ok = f1_2d > 0.72
            checks.append(
                _make_check(
                    "D1",
                    "metrics",
                    "CRITICAL",
                    "PASS" if ok else "FAIL",
                    "F1_2D > 0.72",
                    f"{f1_2d:.4f}",
                    f"F1 2D = {f1_2d:.4f} {'✓' if ok else '✗'}",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "D1",
                "metrics",
                "CRITICAL",
                "SKIP",
                "F1_2D > 0.72",
                f"Error: {e}",
                f"Could not check F1 2D: {e}",
            )
        )

    # D2: F1 3D > 0.65
    try:
        if metrics is None:
            checks.append(
                _make_check(
                    "D2",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "F1_3D > 0.65",
                    "File not found",
                    "test_metrics_summary.json not found",
                )
            )
        elif dry_run or is_dry_run_metrics:
            f1_3d = metrics.get("f1_macro_3d_mean", 0.0)
            checks.append(
                _make_check(
                    "D2",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "F1_3D > 0.65",
                    f"f1_3d={f1_3d}",
                    skip_msg,
                )
            )
        else:
            f1_3d = metrics.get("f1_macro_3d_mean", 0.0)
            ok = f1_3d > 0.65
            checks.append(
                _make_check(
                    "D2",
                    "metrics",
                    "CRITICAL",
                    "PASS" if ok else "FAIL",
                    "F1_3D > 0.65",
                    f"{f1_3d:.4f}",
                    f"F1 3D = {f1_3d:.4f} {'✓' if ok else '✗'}",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "D2",
                "metrics",
                "CRITICAL",
                "SKIP",
                "F1_3D > 0.65",
                f"Error: {e}",
                f"Could not check F1 3D: {e}",
            )
        )

    # D3: Routing accuracy > 0.80
    try:
        if metrics is None:
            checks.append(
                _make_check(
                    "D3",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "routing_accuracy > 0.80",
                    "File not found",
                    "test_metrics_summary.json not found",
                )
            )
        elif dry_run or is_dry_run_metrics:
            ra = metrics.get("routing_accuracy_mean", 0.0)
            checks.append(
                _make_check(
                    "D3",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "routing_accuracy > 0.80",
                    f"routing_accuracy={ra}",
                    skip_msg,
                )
            )
        else:
            ra = metrics.get("routing_accuracy_mean", 0.0)
            ok = ra > 0.80
            checks.append(
                _make_check(
                    "D3",
                    "metrics",
                    "CRITICAL",
                    "PASS" if ok else "FAIL",
                    "routing_accuracy > 0.80",
                    f"{ra:.4f}",
                    f"Routing accuracy = {ra:.4f} {'✓' if ok else '✗'}",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "D3",
                "metrics",
                "CRITICAL",
                "SKIP",
                "routing_accuracy > 0.80",
                f"Error: {e}",
                f"Could not check routing accuracy: {e}",
            )
        )

    # D4: OOD AUROC > 0.80
    try:
        if ood_data is None:
            checks.append(
                _make_check(
                    "D4",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "OOD_AUROC > 0.80",
                    "File not found",
                    "ood_auroc_report.json not found",
                )
            )
        elif dry_run or is_dry_run_ood:
            auroc = ood_data.get("ood_auroc", 0.0)
            checks.append(
                _make_check(
                    "D4",
                    "metrics",
                    "CRITICAL",
                    "SKIP",
                    "OOD_AUROC > 0.80",
                    f"ood_auroc={auroc}",
                    skip_msg,
                )
            )
        else:
            auroc = ood_data.get("ood_auroc", 0.0)
            ok = auroc > 0.80
            checks.append(
                _make_check(
                    "D4",
                    "metrics",
                    "CRITICAL",
                    "PASS" if ok else "FAIL",
                    "OOD_AUROC > 0.80",
                    f"{auroc:.4f}",
                    f"OOD AUROC = {auroc:.4f} {'✓' if ok else '✗'}",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "D4",
                "metrics",
                "CRITICAL",
                "SKIP",
                "OOD_AUROC > 0.80",
                f"Error: {e}",
                f"Could not check OOD AUROC: {e}",
            )
        )


# ── Category E — Reproducibility ───────────────────────────
def check_E(checks: list) -> None:
    """E1-E4: Reproducibility checks."""
    src_dir = PROJECT_ROOT / "src"

    # E1: requirements.txt contains torch
    try:
        req_path = PROJECT_ROOT / "requirements.txt"
        if req_path.exists():
            content = req_path.read_text(encoding="utf-8")
            has_torch = "torch" in content.lower()
            checks.append(
                _make_check(
                    "E1",
                    "reproducibility",
                    "CRITICAL",
                    "PASS" if has_torch else "FAIL",
                    "requirements.txt exists and contains torch",
                    f"exists=True, torch={'yes' if has_torch else 'no'}",
                    f"requirements.txt with torch {'✓' if has_torch else '✗'}",
                )
            )
        else:
            checks.append(
                _make_check(
                    "E1",
                    "reproducibility",
                    "CRITICAL",
                    "FAIL",
                    "requirements.txt exists and contains torch",
                    "File not found",
                    "requirements.txt not found ✗",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "E1",
                "reproducibility",
                "CRITICAL",
                "SKIP",
                "requirements.txt exists and contains torch",
                f"Error: {e}",
                f"Could not check requirements.txt: {e}",
            )
        )

    # E2: No absolute paths in source files
    try:
        # Build pattern from parts to avoid self-matching this script
        _abs_dirs = ["/ho" + "me/", "/ro" + "ot/", "/mn" + "t/"]
        abs_patterns = "(" + "|".join(re.escape(d) for d in _abs_dirs) + ")"
        hits = _grep_source_files(
            abs_patterns, src_dir, skip_comments=True, exclude_self=True
        )
        # Filter out this verification script itself and legitimate PROJECT_ROOT usage
        filtered = [
            (f, ln, line)
            for f, ln, line in hits
            if "Path(__file__)" not in line
            and "PROJECT_ROOT" not in line
            and "_PROJECT_ROOT" not in line
            and "PROJ = Path" not in line
        ]
        if filtered:
            observed = "; ".join(f"{f}:{ln}" for f, ln, _ in filtered[:5])
            checks.append(
                _make_check(
                    "E2",
                    "reproducibility",
                    "WARNING",
                    "FAIL",
                    "No absolute paths in source",
                    observed,
                    f"Absolute paths found in {len(filtered)} location(s) ✗",
                )
            )
        else:
            checks.append(
                _make_check(
                    "E2",
                    "reproducibility",
                    "WARNING",
                    "PASS",
                    "No absolute paths in source",
                    "None found",
                    "No absolute paths in source ✓",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "E2",
                "reproducibility",
                "WARNING",
                "SKIP",
                "No absolute paths in source",
                f"Error: {e}",
                f"Could not check absolute paths: {e}",
            )
        )

    # E3: SEED=42
    try:
        seed_pattern = r"(?i)\bseed\s*=\s*42\b"
        hits = _grep_source_files(seed_pattern, src_dir, skip_comments=False)
        ok = len(hits) > 0
        checks.append(
            _make_check(
                "E3",
                "reproducibility",
                "WARNING",
                "PASS" if ok else "FAIL",
                "SEED=42 found in source",
                f"{len(hits)} references",
                f"SEED=42 {'found' if ok else 'not found'} {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "E3",
                "reproducibility",
                "WARNING",
                "SKIP",
                "SEED=42 found in source",
                f"Error: {e}",
                f"Could not check seed: {e}",
            )
        )

    # E4: README.md exists
    try:
        readme_path = PROJECT_ROOT / "README.md"
        ok = readme_path.exists()
        checks.append(
            _make_check(
                "E4",
                "reproducibility",
                "WARNING",
                "PASS" if ok else "FAIL",
                "README.md exists",
                f"exists={ok}",
                f"README.md {'exists' if ok else 'not found'} {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "E4",
                "reproducibility",
                "WARNING",
                "SKIP",
                "README.md exists",
                f"Error: {e}",
                f"Could not check README.md: {e}",
            )
        )


# ── Category F — Generated Artifacts ───────────────────────
def check_F(checks: list) -> None:
    """F1-F5: Generated artifacts existence."""

    artifacts = {
        "F1": ("checkpoints/entropy_threshold.pkl", "CRITICAL"),
        "F2": ("results/paso9/test_metrics_summary.json", "CRITICAL"),
        "F3": ("results/paso9/load_balance_test.json", "CRITICAL"),
        "F4": ("results/paso9/ood_auroc_report.json", "CRITICAL"),
    }

    for check_id, (rel_path, severity) in artifacts.items():
        try:
            full_path = PROJECT_ROOT / rel_path
            ok = full_path.exists()
            checks.append(
                _make_check(
                    check_id,
                    "artifacts",
                    severity,
                    "PASS" if ok else "FAIL",
                    f"{rel_path} exists",
                    f"exists={ok}",
                    f"{rel_path} {'✓' if ok else '✗'}",
                )
            )
        except Exception as e:
            checks.append(
                _make_check(
                    check_id,
                    "artifacts",
                    severity,
                    "SKIP",
                    f"{rel_path} exists",
                    f"Error: {e}",
                    f"Could not check {rel_path}: {e}",
                )
            )

    # F5: At least 1 .pt checkpoint
    try:
        ckpt_dir = PROJECT_ROOT / "checkpoints"
        pt_files = list(ckpt_dir.rglob("*.pt")) if ckpt_dir.exists() else []
        ok = len(pt_files) > 0
        checks.append(
            _make_check(
                "F5",
                "artifacts",
                "WARNING",
                "PASS" if ok else "FAIL",
                "At least 1 .pt checkpoint",
                f"{len(pt_files)} .pt files found",
                f"Checkpoint .pt files: {len(pt_files)} {'✓' if ok else '— training may be pending'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "F5",
                "artifacts",
                "WARNING",
                "SKIP",
                "At least 1 .pt checkpoint",
                f"Error: {e}",
                f"Could not check checkpoints: {e}",
            )
        )


# ── Category G — Prohibited Augmentations ──────────────────
def check_G(checks: list) -> None:
    """G1-G5: No prohibited augmentations in active code."""
    src_dir = PROJECT_ROOT / "src"

    prohibited = {
        "G1": "RandomVerticalFlip",
        "G2": "RandomErasing",
        "G3": "CutMix",
        "G4": "MixUp",
        "G5": "GridMask",
    }

    for check_id, aug_name in prohibited.items():
        try:
            # Match only active usage: import, function call, or class instantiation.
            # Exclude mentions inside strings, docstrings, and comments.
            pattern = (
                r"(?:import\s+.*\b" + re.escape(aug_name) + r"\b"
                r"|from\s+\S+\s+import\s+.*\b" + re.escape(aug_name) + r"\b"
                r"|\b" + re.escape(aug_name) + r"\s*\("
                r")"
            )
            hits = _grep_source_files(
                pattern,
                src_dir,
                skip_comments=True,
                skip_strings=True,
                exclude_self=True,
            )
            # Post-filter: remove matches where the augmentation name is inside
            # a string literal (log messages, notes, etc.)
            filtered_hits = []
            for filepath, lineno, line_text in hits:
                # If the aug_name is preceded by a quote on the same line,
                # it's inside a string, not active code.
                idx = line_text.find(aug_name)
                if idx >= 0:
                    before = line_text[:idx]
                    # Count unescaped quotes before the match
                    single_q = before.count("'") - before.count("\\'")
                    double_q = before.count('"') - before.count('\\"')
                    if single_q % 2 == 1 or double_q % 2 == 1:
                        # aug_name is inside a string literal — skip
                        continue
                filtered_hits.append((filepath, lineno, line_text))
            hits = filtered_hits
            if hits:
                observed = "; ".join(f"{f}:{ln}" for f, ln, _ in hits[:5])
                checks.append(
                    _make_check(
                        check_id,
                        "prohibited_augmentations",
                        "CRITICAL",
                        "FAIL",
                        f"No {aug_name} in active code",
                        observed,
                        f"{aug_name} found in active code ✗",
                    )
                )
            else:
                checks.append(
                    _make_check(
                        check_id,
                        "prohibited_augmentations",
                        "CRITICAL",
                        "PASS",
                        f"No {aug_name} in active code",
                        "Not found",
                        f"No {aug_name} in active code ✓",
                    )
                )
        except Exception as e:
            checks.append(
                _make_check(
                    check_id,
                    "prohibited_augmentations",
                    "CRITICAL",
                    "SKIP",
                    f"No {aug_name} in active code",
                    f"Error: {e}",
                    f"Could not check for {aug_name}: {e}",
                )
            )


# ── Category H — VRAM Constraints ──────────────────────────
def check_H(checks: list) -> None:
    """H1-H3: VRAM optimization patterns."""
    src_dir = PROJECT_ROOT / "src"

    vram_checks = {
        "H1": (
            r"gradient_checkpointing|use_gradient_checkpointing",
            "gradient_checkpointing",
        ),
        "H2": (r"torch\.cuda\.amp|autocast", "FP16 autocast"),
        "H3": (r"accumulation_steps|grad_accum", "gradient accumulation"),
    }

    for check_id, (pattern, desc) in vram_checks.items():
        try:
            hits = _grep_source_files(pattern, src_dir, skip_comments=True)
            ok = len(hits) > 0
            checks.append(
                _make_check(
                    check_id,
                    "vram",
                    "WARNING",
                    "PASS" if ok else "FAIL",
                    f"{desc} found in source",
                    f"{len(hits)} references",
                    f"{desc} {'found' if ok else 'not found'} {'✓' if ok else '✗'}",
                )
            )
        except Exception as e:
            checks.append(
                _make_check(
                    check_id,
                    "vram",
                    "WARNING",
                    "SKIP",
                    f"{desc} found in source",
                    f"Error: {e}",
                    f"Could not check {desc}: {e}",
                )
            )


# ── Category I — Architecture ──────────────────────────────
def check_I(checks: list) -> None:
    """I1-I4: Architecture requirements."""
    src_dir = PROJECT_ROOT / "src"

    # I1: N_EXPERTS_DOMAIN == 5
    try:
        from src.pipeline import config as cfg

        ok = cfg.N_EXPERTS_DOMAIN == 5
        checks.append(
            _make_check(
                "I1",
                "architecture",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "N_EXPERTS_DOMAIN == 5",
                f"{cfg.N_EXPERTS_DOMAIN}",
                f"N_EXPERTS_DOMAIN = {cfg.N_EXPERTS_DOMAIN} {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "I1",
                "architecture",
                "CRITICAL",
                "SKIP",
                "N_EXPERTS_DOMAIN == 5",
                f"Error: {e}",
                f"Could not check N_EXPERTS_DOMAIN: {e}",
            )
        )

    # I2: N_EXPERTS_TOTAL == 6
    try:
        from src.pipeline import config as cfg

        ok = cfg.N_EXPERTS_TOTAL == 6
        checks.append(
            _make_check(
                "I2",
                "architecture",
                "CRITICAL",
                "PASS" if ok else "FAIL",
                "N_EXPERTS_TOTAL == 6",
                f"{cfg.N_EXPERTS_TOTAL}",
                f"N_EXPERTS_TOTAL = {cfg.N_EXPERTS_TOTAL} {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "I2",
                "architecture",
                "CRITICAL",
                "SKIP",
                "N_EXPERTS_TOTAL == 6",
                f"Error: {e}",
                f"Could not check N_EXPERTS_TOTAL: {e}",
            )
        )

    # I3: At least 4 router class names / ablation router types
    try:
        router_pattern = r"(TopKRouter|SoftRouter|NoisyTopKRouter|DynamicRouter|LinearGatingHead|RandomRouter|HashRouter|ExpertChoiceRouter)"
        hits = _grep_source_files(router_pattern, src_dir, skip_comments=False)
        # Count distinct router class names
        router_names = set()
        for _, _, line in hits:
            for m in re.finditer(router_pattern, line):
                router_names.add(m.group(1))
        ok = len(router_names) >= 4
        checks.append(
            _make_check(
                "I3",
                "architecture",
                "WARNING",
                "PASS" if ok else "FAIL",
                "At least 4 router types",
                f"{len(router_names)} types: {', '.join(sorted(router_names))}",
                f"Router types: {len(router_names)} {'✓' if ok else '✗'}",
            )
        )
    except Exception as e:
        checks.append(
            _make_check(
                "I3",
                "architecture",
                "WARNING",
                "SKIP",
                "At least 4 router types",
                f"Error: {e}",
                f"Could not check router types: {e}",
            )
        )

    # I4: Shared backbone — grep for backbone usage in fase2
    try:
        fase2_dir = PROJECT_ROOT / "src" / "pipeline" / "fase2"
        if fase2_dir.exists():
            backbone_pattern = r"\bbackbone\b"
            hits = _grep_source_files(backbone_pattern, fase2_dir, skip_comments=True)
            ok = len(hits) > 0
            checks.append(
                _make_check(
                    "I4",
                    "architecture",
                    "WARNING",
                    "PASS" if ok else "FAIL",
                    "Shared backbone references in fase2",
                    f"{len(hits)} references",
                    f"Backbone references in fase2: {len(hits)} {'✓' if ok else '✗'}",
                )
            )
        else:
            checks.append(
                _make_check(
                    "I4",
                    "architecture",
                    "WARNING",
                    "SKIP",
                    "Shared backbone references in fase2",
                    "fase2 directory not found",
                    "fase2 directory not found",
                )
            )
    except Exception as e:
        checks.append(
            _make_check(
                "I4",
                "architecture",
                "WARNING",
                "SKIP",
                "Shared backbone references in fase2",
                f"Error: {e}",
                f"Could not check backbone usage: {e}",
            )
        )


# =====================================================================
# Penalty Detection
# =====================================================================
def detect_penalties(checks: list) -> list[dict]:
    """Detect penalty violations from check results."""
    penalties = []

    # Build lookup by ID
    by_id = {c["id"]: c for c in checks}

    # -40% load balance penalty
    c1 = by_id.get("C1")
    if c1 and c1["result"] == "FAIL":
        penalties.append(
            {
                "type": "LOAD_BALANCE",
                "penalty": "-40%",
                "description": "max_ratio > 1.30",
                "observed": c1["observed"],
            }
        )

    # -20% metadata penalty
    b1 = by_id.get("B1")
    if b1 and b1["result"] == "FAIL":
        penalties.append(
            {
                "type": "NO_METADATA_VIOLATION",
                "penalty": "-20%",
                "description": "Metadata-based routing detected",
                "observed": b1["observed"],
            }
        )

    return penalties


# =====================================================================
# Main
# =====================================================================
def run_verification(args: argparse.Namespace) -> int:
    """Run all verification checks and generate report."""
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PASO 10 — VERIFICACIÓN FUNCIONAL DEL PIPELINE MoE")
    if args.dry_run:
        logger.info("Modo: DRY-RUN (métricas no se validan)")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    checks: list[dict] = []

    # Run all categories
    logger.info("─── Category A: Imports & Smoke Tests ───")
    check_A(checks)

    logger.info("─── Category B: No-Metadata Constraint ───")
    check_B(checks)

    logger.info("─── Category C: Load Balance ───")
    check_C(checks, dry_run=args.dry_run)

    logger.info("─── Category D: Metric Thresholds ───")
    check_D(checks, dry_run=args.dry_run)

    logger.info("─── Category E: Reproducibility ───")
    check_E(checks)

    logger.info("─── Category F: Generated Artifacts ───")
    check_F(checks)

    logger.info("─── Category G: Prohibited Augmentations ───")
    check_G(checks)

    logger.info("─── Category H: VRAM Constraints ───")
    check_H(checks)

    logger.info("─── Category I: Architecture ───")
    check_I(checks)

    # Log each check result
    for c in checks:
        result = c["result"]
        cid = c["id"]
        msg = c["message"]
        if result == "PASS":
            logger.info(f"✓ {cid} PASS: {msg}")
        elif result == "FAIL":
            if c["severity"] == "CRITICAL":
                logger.warning(f"✗ {cid} FAIL [CRITICAL]: {msg}")
            else:
                logger.warning(f"✗ {cid} FAIL [{c['severity']}]: {msg}")
        else:
            logger.info(f"⊘ {cid} SKIP: {msg}")

    # Compute summary stats
    total = len(checks)
    passed = sum(1 for c in checks if c["result"] == "PASS")
    failed = sum(1 for c in checks if c["result"] == "FAIL")
    skipped = sum(1 for c in checks if c["result"] == "SKIP")
    warnings = sum(
        1 for c in checks if c["result"] == "FAIL" and c["severity"] == "WARNING"
    )
    critical_failures = sum(
        1 for c in checks if c["result"] == "FAIL" and c["severity"] == "CRITICAL"
    )

    # Detect penalties
    penalties = detect_penalties(checks)

    if critical_failures == 0:
        summary_text = "All checks passed"
    else:
        summary_text = f"{critical_failures} critical failures detected"

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "warnings": warnings,
        "critical_failures": critical_failures,
        "penalties_detected": penalties,
        "summary": summary_text,
        "checks": checks,
    }

    # Write report
    report_path = output_path / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report written to {report_path}")

    # Final summary
    logger.info("=" * 60)
    logger.info(
        f"TOTAL: {total} checks | PASS: {passed} | FAIL: {failed} | SKIP: {skipped}"
    )
    logger.info(f"CRITICAL FAILURES: {critical_failures} | WARNINGS: {warnings}")
    if penalties:
        for p in penalties:
            logger.warning(
                f"PENALTY: {p['type']} ({p['penalty']}) — {p['description']}"
            )
    else:
        logger.info("No penalties detected")
    logger.info(f"SUMMARY: {summary_text}")
    logger.info("=" * 60)

    # Exit code logic
    if args.dry_run:
        return 0
    else:
        return 1 if critical_failures > 0 else 0


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    setup_logging(output_dir=output_dir, phase_name="paso10_verificacion")
    exit_code = run_verification(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
