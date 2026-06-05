"""Profile where a training run actually spends CPU time, by package.

Purpose: keep an honest, regression-protected answer to "would integrating C++
speed SurroForge up?". A run's heavy lifting is delegated to compiled libraries
(SMT/scipy/numpy/scikit-learn → LAPACK/BLAS); SurroForge's own Python is thin
orchestration. This script measures that on the *real* ``run_training`` path for
every available method — including the expensive O(n³) Kriging/RBF when ``smt`` is
installed (it is in CI) — and asserts that our own code stays a small fraction of
CPU time. If a compute-bound hotspot ever lands in Core, this job fails and points
at it; a C++ rewrite of Core would only ever pay off if that fraction were large.

Run locally: ``python scripts/profile_training.py`` (profiles polynomial; adds
Kriging/RBF when the ``smt`` extra is present). CI runs it with ``[all,dev]``.

Metric: cProfile ``tottime`` (time IN a function, excluding sub-calls). Summed over
all functions it equals total runtime with no double counting, so bucketing it by
package answers "which code burns the cycles?" directly.
"""

from __future__ import annotations

import cProfile
import importlib.util
import os
import pstats
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import surroforge
from surroforge.core.config import parse_config
from surroforge.core.project import Project
from surroforge.engine import run_training

# Our own code is expected to be a sliver of total CPU time. The ceiling is set
# far above the observed share (~1% on the cheapest method, less on Kriging/RBF):
# it is a guard against a genuine compute-bound regression in Core, not a tight
# performance budget (CI timing is noisy; this fraction is robust because the
# compiled libraries dominate by a wide margin).
MAX_OWN_CODE_SHARE = float(os.environ.get("SF_PROFILE_MAX_SHARE", "0.10"))

_SF_PKG_DIR = os.path.dirname(os.path.abspath(surroforge.__file__)).replace("\\", "/")
_ONNX_PKGS = {"skl2onnx", "onnx", "onnxconverter_common", "onnxruntime"}


def have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def bucket_for(filename: str) -> str:
    """Map a profiled function's source file to a coarse package bucket."""
    if filename.startswith("<") or filename.startswith("~"):
        return "stdlib/builtins"
    f = filename.replace("\\", "/")
    if f.startswith(_SF_PKG_DIR) or ("/surroforge/" in f and "site-packages" not in f):
        return "surroforge (our code)"
    if "site-packages/" in f:
        top = f.split("site-packages/", 1)[1].split("/", 1)[0]
        if top in _ONNX_PKGS:
            return "onnx export"
        if top in {"yaml", "_yaml"}:
            return "pyyaml"
        if top in {"importlib_metadata"}:
            return "packaging metadata"
        return top  # smt, scipy, numpy, sklearn, pandas, joblib, pydantic, ...
    if "importlib/metadata" in f or "/email/" in f:
        return "packaging metadata"
    if "/lib/python3" in f or "/lib64/python3" in f or "/python3." in f:
        return "stdlib/builtins"
    return "other"


def make_dataset(n: int, d: int, seed: int, path: Path) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = (
        np.sin(np.pi * x[:, 0])
        + 0.5 * x[:, 1] ** 2
        + 0.25 * x[:, 0] * x[:, 2]
        + 0.02 * rng.standard_normal(n)
    )
    cols = {f"x{i + 1}": x[:, i] for i in range(d)}
    cols["y"] = y
    pd.DataFrame(cols).to_csv(path, index=False)


def config(method: str, csv: Path, d: int, folds: int) -> object:
    model = {"method": method, "degree": 2} if method == "polynomial" else {"method": method}
    return parse_config(
        {
            "experiment": f"profile-{method}",
            "data": {
                "source": str(csv),
                "inputs": [f"x{i + 1}" for i in range(d)],
                "outputs": ["y"],
                "split": {"test_fraction": 0.2, "seed": 42},
            },
            "model": model,
            "validation": {"cross_validation": folds, "metrics": ["rmse", "r2", "mae"]},
        }
    )


def profile_method(method: str, n: int, d: int, folds: int, seed: int, root: Path) -> dict:
    csv = root / f"{method}.csv"
    make_dataset(n, d, seed=seed, path=csv)
    cfg = config(method, csv, d, folds)

    # Warm up imports/JITs once (untimed) so the profile excludes one-off import cost.
    warm = Project.init(root / f"{method}_warm", name="w")
    run_training(warm, cfg)

    proj = Project.init(root / f"{method}_run", name="p")
    pr = cProfile.Profile()
    pr.enable()
    run_training(proj, cfg)
    pr.disable()

    buckets: dict[str, float] = {}
    for (filename, _lineno, _func), values in pstats.Stats(pr).stats.items():
        name = bucket_for(filename)
        buckets[name] = buckets.get(name, 0.0) + values[2]
    total = sum(buckets.values()) or 1e-12
    own = buckets.get("surroforge (our code)", 0.0)
    return {"method": method, "n": n, "buckets": buckets, "total": total, "own_share": own / total}


def main() -> int:
    methods = [("polynomial", 1500, 5, 5)]
    if have("smt"):
        methods += [("rbf", 400, 5, 3), ("kriging", 200, 5, 3)]
    else:
        print("NOTE: 'smt' not installed — profiling polynomial only "
              "(Kriging/RBF need the 'smt' extra; CI installs it).")

    root = Path(tempfile.mkdtemp(prefix="sf-profile-"))
    results = []
    try:
        for seed, (method, n, d, folds) in enumerate(methods):
            results.append(profile_method(method, n, d, folds, seed, root))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 76)
    print("Where a training run spends CPU time (cProfile tottime, by package)")
    print("=" * 76)
    for r in results:
        print(f"\n  method={r['method']:<11} n={r['n']:<5} total-CPU={r['total'] * 1e3:8.1f} ms")
        for name, t in sorted(r["buckets"].items(), key=lambda kv: -kv[1]):
            pct = 100.0 * t / r["total"]
            if pct < 0.5:
                continue
            print(f"      {name:<26} {t * 1e3:8.1f} ms  {pct:5.1f}%  {'#' * int(pct / 2)}")

    worst = max(results, key=lambda r: r["own_share"])
    share = worst["own_share"]
    print("\n" + "-" * 76)
    print(
        f"surroforge's own code peaks at {share * 100:.2f}% of CPU time "
        f"(method={worst['method']}); the rest is compiled libraries "
        f"(SMT/scipy/numpy/sklearn → LAPACK/BLAS)."
    )
    print(
        "Conclusion: the cost lives in compiled libraries, not in Core's Python — "
        "a C++ rewrite of Core would address none of it."
    )
    if share > MAX_OWN_CODE_SHARE:
        print(
            f"\nFAIL: own-code share {share * 100:.2f}% exceeds ceiling "
            f"{MAX_OWN_CODE_SHARE * 100:.0f}% — a compute-bound hotspot has landed in "
            f"Core (method={worst['method']}). Investigate before assuming it is glue."
        )
        return 1
    print(f"\nPASS: own-code share {share * 100:.2f}% is within the "
          f"{MAX_OWN_CODE_SHARE * 100:.0f}% ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
