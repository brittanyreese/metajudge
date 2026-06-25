# scripts/oracles/gen_dgp_recovery.py
"""Provenance gate for the DGP: R MASS::polr must recover the planted b1/b2.

Regenerates a DGP sample (rater_sd = 0, one rating per item, so the data is a clean
fixed-effects cumulative logit), writes it to a temp CSV, fits MASS::polr, and checks the
recovered theta/group coefficients against the planted values. Exit 0 pass, 1 fail, 2 if R
or MASS is unavailable. Mirrors scripts/oracles/gen_olr_oracle.py.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.dgp import DgpParams, simulate

_R_FITTER = Path(__file__).resolve().parents[2] / "sim" / "oracles" / "dgp_recovery.R"
_PLANTED_TRAIT = 1.0
_PLANTED_GROUP = 0.8
_TOL = 0.12  # logits; Monte-Carlo recovery error at this n


def _write_csv(path: Path) -> None:
    params = DgpParams(
        n_items_per_group=4000,
        n_raters=1,
        trait_slope=_PLANTED_TRAIT,
        rater_sd=0.0,
        mu_focal=0.0,
        dif_uniform=_PLANTED_GROUP,
    )
    sample = simulate(params, seed=20260624)
    long = sample.ratings._long  # pyright: ignore[reportPrivateUsage]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["score", "theta", "group"])
        for _, row in long.iterrows():
            item = row["item"]
            group = 1 if row["stratum"] == "focal" else 0
            writer.writerow([int(row["score"]), sample.theta[item], group])


def main() -> int:
    if shutil.which("Rscript") is None:
        print("Rscript not found")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "dgp_recovery.csv"
        _write_csv(csv_path)
        proc = subprocess.run(
            ["Rscript", str(_R_FITTER), str(csv_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode == 2:
        print("R package MASS unavailable")
        return 2
    if proc.returncode != 0:
        print(f"Rscript failed:\n{proc.stdout}\n{proc.stderr}")
        return 1
    recovered: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        name, _, value = line.partition(" ")
        if name in ("theta", "group"):
            recovered[name] = float(value)
    planted = {"theta": _PLANTED_TRAIT, "group": _PLANTED_GROUP}
    bad = {
        k: recovered.get(k)
        for k in planted
        if abs(recovered.get(k, np.inf) - planted[k]) > _TOL
    }
    if bad:
        print(f"recovery outside tolerance {_TOL}: planted={planted} recovered={recovered}")
        return 1
    print(f"recovery OK: planted={planted} recovered={recovered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
