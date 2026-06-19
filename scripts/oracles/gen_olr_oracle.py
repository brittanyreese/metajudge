"""Provenance harness for the DIF oracle constants pinned in tests/test_dif.py.

SPEC R19 makes DIF numerical correctness non-compressible, and CONTRIBUTING.md
requires every reported number to be regenerable from a clean clone. The pinned
``_OR_*`` (external-conditioner) and ``_RS_*`` (leave-one-rater-out rest-score)
constants are produced by R ``MASS::polr`` -- the canonical proportional-odds
fit. This script makes that derivation reproducible and falsifiable:

  1. The conditioner fixture ``_QUALITY`` is regenerated from its seed
     (``numpy.random.default_rng(20260622).normal(size=24)``) and checked
     against the literal pinned in tests/test_dif.py.
  2. The response matrix ``_SCORES`` is the frozen "scored once" fixture; its
     canonical home is tests/test_dif.py. This script reads it from there
     (single source of truth, no duplicated literals) rather than regenerating
     it -- the original stochastic scoring is intentionally frozen so the oracle
     constants stay stable.
  3. The external-conditioner and rest-score designs are rebuilt exactly as
     ``metajudge.dif.logistic_dif`` builds them (z-scored conditioner, ddof=0),
     handed to ``fit_polr_dif.R``, and the polr output is compared to the pinned
     constants at the same tolerances tests/test_dif.py asserts.

If a regenerated constant ever disagrees with a pinned literal, the reference
(polr) wins and the literal must be corrected (numerical-reference convention,
AGENTS.md) -- this script exits non-zero so the disagreement cannot pass silently.

Requirements:
    R with the MASS package. Install R, then ``install.packages("MASS")``.

Usage:
    uv run python scripts/oracles/gen_olr_oracle.py

Exit codes:
    0  all reproduced (or only the Python-side checks ran and agreed)
    1  a regenerated constant disagrees with its pinned literal
    2  R / MASS unavailable -- polr leg skipped (Python-side checks still ran)
"""

from __future__ import annotations

import ast
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_DIF = _REPO_ROOT / "tests" / "test_dif.py"
_FIT_R = Path(__file__).resolve().parent / "fit_polr_dif.R"
_SEED = 20260622

# Tolerances mirror the assertions in tests/test_dif.py exactly.
_TOL = {
    "chi2_total": 1e-3,
    "chi2_uniform": 1e-3,
    "chi2_nonuniform": 1e-3,
    "p_total": 1e-5,
    "p_uniform": 1e-5,
    "p_nonuniform": 1e-4,
    "nagelkerke_r2_delta": 1e-3,
}


def _load_test_constants() -> dict[str, object]:
    """Parse the pinned fixture and oracle constants out of tests/test_dif.py.

    Reading the literals (rather than importing the test module, which is not a
    package) keeps tests/test_dif.py the single source of truth -- this harness
    has no constants of its own to drift from it.
    """
    tree = ast.parse(_TEST_DIF.read_text())
    wanted = {
        "_QUALITY",
        "_SCORES",
        "_N_ITEMS",
        "_N_RATERS",
        "_OR_CHI2_TOTAL",
        "_OR_CHI2_UNIFORM",
        "_OR_CHI2_NONUNIFORM",
        "_OR_P_TOTAL",
        "_OR_P_UNIFORM",
        "_OR_P_NONUNIFORM",
        "_OR_NAGELKERKE_R2_DELTA",
        "_RS_CHI2_UNIFORM",
        "_RS_CHI2_NONUNIFORM",
        "_RS_NAGELKERKE_R2_DELTA",
    }
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"could not parse from {_TEST_DIF}: {sorted(missing)}")
    return found


def _verify_quality_seed(quality: list[float]) -> None:
    """The conditioner fixture must be regenerable from its documented seed."""
    regenerated = np.random.default_rng(_SEED).normal(size=len(quality))
    if not np.allclose(regenerated, quality, atol=1e-6):
        raise SystemExit(
            "FAIL: _QUALITY is not reproduced by default_rng(20260622).normal"
            f"(size={len(quality)}); the fixture seed in tests/test_dif.py is stale."
        )
    print(f"OK   _QUALITY reproduced from default_rng({_SEED}).normal(size={len(quality)})")


def _build_designs(
    scores: list[int], quality: list[float], n_items: int, n_raters: int
) -> dict[str, list[tuple[int, float, float]]]:
    """Rebuild the engine's two designs: (score, standardized conditioner, group).

    Mirrors metajudge.dif.logistic_dif: items 0..n/2-1 are reference, the rest
    focal; the conditioner is z-scored over all rows with ddof=0. The external
    path uses the per-item quality; the rest-score path uses the leave-one-
    rater-out mean of the other raters' scores.
    """
    mat = np.asarray(scores, dtype=float).reshape(n_items, n_raters)
    ext_score: list[int] = []
    ext_cond: list[float] = []
    ext_group: list[float] = []
    rest_score: list[int] = []
    rest_cond: list[float] = []
    rest_group: list[float] = []
    for i in range(n_items):
        group = 1.0 if i >= n_items // 2 else 0.0
        row = mat[i]
        total = float(row.sum())
        count = len(row)
        for r in range(n_raters):
            value = int(row[r])
            ext_score.append(value)
            ext_cond.append(float(quality[i]))
            ext_group.append(group)
            rest_score.append(value)
            rest_cond.append((total - float(row[r])) / (count - 1))
            rest_group.append(group)

    def _zscore(values: list[float]) -> list[float]:
        arr = np.asarray(values, dtype=float)
        return ((arr - arr.mean()) / arr.std(ddof=0)).tolist()

    return {
        "external": list(zip(ext_score, _zscore(ext_cond), ext_group, strict=True)),
        "rest_score": list(zip(rest_score, _zscore(rest_cond), rest_group, strict=True)),
    }


def _run_polr(rows: list[tuple[int, float, float]], workdir: Path) -> dict[str, float]:
    """Write the design to CSV, fit it with fit_polr_dif.R, parse key=value output."""
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "design.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["score", "cond", "group"])
        writer.writerows(rows)
    completed = subprocess.run(
        ["Rscript", str(_FIT_R), str(csv_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key.strip()] = float(value)
    return parsed


def _compare(label: str, expected: dict[str, float], got: dict[str, float]) -> bool:
    """Compare polr output to the pinned constants; print one line per constant."""
    ok = True
    for key, exp in expected.items():
        tol = _TOL[key]
        actual = got[key]
        passed = abs(actual - exp) <= tol
        ok = ok and passed
        flag = "OK  " if passed else "FAIL"
        print(f"{flag} {label}.{key}: polr={actual:.6f} pinned={exp:.6f} (tol={tol:g})")
    return ok


def main() -> int:
    constants = _load_test_constants()
    quality = list(constants["_QUALITY"])  # type: ignore[call-overload]
    scores = list(constants["_SCORES"])  # type: ignore[call-overload]
    n_items = int(constants["_N_ITEMS"])  # type: ignore[call-overload]
    n_raters = int(constants["_N_RATERS"])  # type: ignore[call-overload]

    _verify_quality_seed(quality)

    designs = _build_designs(scores, quality, n_items, n_raters)

    if shutil.which("Rscript") is None:
        print("SKIP Rscript not found on PATH -- polr leg skipped.")
        print("     Install R + MASS to reproduce the pinned DIF constants:")
        print('       install.packages("MASS"); see scripts/oracles/fit_polr_dif.R')
        return 2

    external_expected = {
        "chi2_total": float(constants["_OR_CHI2_TOTAL"]),  # type: ignore[arg-type]
        "chi2_uniform": float(constants["_OR_CHI2_UNIFORM"]),  # type: ignore[arg-type]
        "chi2_nonuniform": float(constants["_OR_CHI2_NONUNIFORM"]),  # type: ignore[arg-type]
        "p_total": float(constants["_OR_P_TOTAL"]),  # type: ignore[arg-type]
        "p_uniform": float(constants["_OR_P_UNIFORM"]),  # type: ignore[arg-type]
        "p_nonuniform": float(constants["_OR_P_NONUNIFORM"]),  # type: ignore[arg-type]
        "nagelkerke_r2_delta": float(constants["_OR_NAGELKERKE_R2_DELTA"]),  # type: ignore[arg-type]
    }
    rest_expected = {
        "chi2_uniform": float(constants["_RS_CHI2_UNIFORM"]),  # type: ignore[arg-type]
        "chi2_nonuniform": float(constants["_RS_CHI2_NONUNIFORM"]),  # type: ignore[arg-type]
        "nagelkerke_r2_delta": float(constants["_RS_NAGELKERKE_R2_DELTA"]),  # type: ignore[arg-type]
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            external_got = _run_polr(designs["external"], workdir / "ext")
            rest_got = _run_polr(designs["rest_score"], workdir / "rest")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if "there is no package called" in stderr or "MASS" in stderr:
            print("SKIP R is present but the MASS package is missing.")
            print('     Install it: install.packages("MASS")')
            return 2
        print(f"FAIL Rscript errored:\n{stderr}")
        return 1

    ext_ok = _compare("external", external_expected, external_got)
    rest_ok = _compare("rest_score", rest_expected, rest_got)

    if ext_ok and rest_ok:
        print("\nALL DIF oracle constants reproduced from R MASS::polr.")
        return 0
    print("\nMISMATCH: a pinned constant disagrees with polr. The reference wins;")
    print("correct the literal in tests/test_dif.py (never loosen the tolerance).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
