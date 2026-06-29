# DoE Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Professional design-of-experiments generator — `surroforge doe run spec.yaml` writes a `samples.csv` (+ manifest) of parameter points for any external solver.

**Architecture:** A new generic Core command seam (`surroforge.commands` entry-point group) lets plugins mount Typer sub-apps into the `surroforge` CLI. The DoE capability lives entirely in the existing `surroforge-professional` plugin (reusing its license gate): a Pydantic spec, a sampling core (full-factorial / LHS / Sobol / random), a CSV+manifest writer, and a `doe` Typer app. Core gains only the generic seam; all DoE logic and the `DoeError` (a `SurroForgeError` subclass) live in the plugin.

**Tech Stack:** Python 3.10+, Typer, Pydantic v2, pandas, numpy, scipy (optional, `scipy.stats.qmc`), PyYAML (via `surroforge`), pytest.

## Global Constraints

- Python `>=3.10`; ruff `line-length = 100`.
- Core stays dependency-light: **no scipy in Core**. scipy is an optional plugin extra `doe = ["scipy>=1.10"]`, imported **lazily** inside sampling functions only.
- The DoE command is **license-gated**: `require_license()` runs before any spec file is read (reuse `surroforge_professional/license.py`).
- All domain errors subclass `surroforge.errors.SurroForgeError` so the CLI renders them as `Error: …` (exit 1). DoE uses `DoeError(SurroForgeError)`.
- CLI mirrors the Python API one-to-one (project convention).
- The plugin distribution license field stays `Proprietary`; the new Core seam code is Apache-2.0 Core.
- Output `samples.csv` format is fixed: a `case` column (zero-padded, min width 3) followed by one column per factor — exactly what `surroforge data add-openfoam --samples` / `read_openfoam_sweep` consume.
- Run all commands from the repo root. Python is `.venv/bin/python`.

---

## File Structure

**Core (Apache-2.0):**
- Modify `src/surroforge/plugins/registry.py` — add `COMMAND_GROUP`, `available_command_names()`, `available_commands()`.
- Modify `src/surroforge/plugins/__init__.py` — export the two new functions.
- Modify `src/surroforge/cli/main.py` — add `_mount_plugin_commands(target, commands)` and call it at import.
- Test `tests/test_command_seam.py` — discovery + mounting + guard + zero-plugins.

**Plugin (`plugins/surroforge-professional/`):**
- Create `src/surroforge_professional/doe/__init__.py` — package marker + public re-exports.
- Create `src/surroforge_professional/doe/spec.py` — `DoeError`, the Pydantic models, `load_spec`, `validate_spec`.
- Create `src/surroforge_professional/doe/sampling.py` — `generate(spec) -> DataFrame`.
- Create `src/surroforge_professional/doe/output.py` — `write_design(spec, out_dir, force) -> (csv, manifest)`.
- Create `src/surroforge_professional/commands/__init__.py` — package marker.
- Create `src/surroforge_professional/commands/doe.py` — `doe_app` Typer app + license gate.
- Modify `pyproject.toml` — `surroforge.commands` entry point + `doe` extra.
- Modify `README.md` — document `surroforge doe`.
- Test `tests/test_doe_spec.py`, `tests/test_doe_sampling.py`, `tests/test_doe_output.py`, `tests/test_doe_command.py`.

---

## Task 1: Core Command Seam — Discovery

**Files:**
- Modify: `src/surroforge/plugins/registry.py`
- Modify: `src/surroforge/plugins/__init__.py`
- Test: `tests/test_command_seam.py`

**Interfaces:**
- Produces: `COMMAND_GROUP = "surroforge.commands"`; `available_command_names() -> list[str]`; `available_commands() -> list[tuple[str, Any]]` (name → loaded object, sorted by name; raises `PluginError` on an ambiguous name registered by two plugins).

- [ ] **Step 1: Write the failing discovery test**

Create `tests/test_command_seam.py`:

```python
from __future__ import annotations

import pytest

from surroforge.errors import PluginError
from surroforge.plugins import available_command_names, available_commands
from surroforge.plugins import registry


class _FakeEP:
    def __init__(self, name, value, obj):
        self.name = name
        self.value = value
        self._obj = obj

    def load(self):
        return self._obj


def test_available_commands_discovers_and_loads(monkeypatch):
    sentinel = object()
    grouped = {"doe": [_FakeEP("doe", "pkg.mod:app", sentinel)]}
    monkeypatch.setattr(
        registry, "_entry_points_grouped", lambda group: grouped if group == registry.COMMAND_GROUP else {}
    )

    assert available_command_names() == ["doe"]
    assert available_commands() == [("doe", sentinel)]


def test_available_commands_rejects_ambiguous(monkeypatch):
    grouped = {
        "doe": [_FakeEP("doe", "a:app", object()), _FakeEP("doe", "b:app", object())]
    }
    monkeypatch.setattr(
        registry, "_entry_points_grouped", lambda group: grouped if group == registry.COMMAND_GROUP else {}
    )

    with pytest.raises(PluginError, match="Ambiguous command 'doe'"):
        available_commands()


def test_available_commands_empty_by_default(monkeypatch):
    monkeypatch.setattr(registry, "_entry_points_grouped", lambda group: {})
    assert available_command_names() == []
    assert available_commands() == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_command_seam.py -q`
Expected: FAIL — `ImportError: cannot import name 'available_command_names'`.

- [ ] **Step 3: Implement discovery in the registry**

In `src/surroforge/plugins/registry.py`, after the existing `REPORTER_GROUP = "surroforge.reporters"` line (around line 27), add:

```python
COMMAND_GROUP = "surroforge.commands"
```

Then at the end of the file (after `available_reporters`), add:

```python
def available_command_names() -> list[str]:
    return sorted(_entry_points_grouped(COMMAND_GROUP))


def available_commands() -> list[tuple[str, Any]]:
    """Plugin-contributed CLI sub-apps registered under ``surroforge.commands``.

    Each entry point is expected to load a ``typer.Typer`` instance. Discovery is
    kept separate from :func:`_resolve` (which is trainer/exporter/reporter-shaped
    and consults the ``_BUILTIN`` table); Core ships no built-in plugin commands.
    """
    loaded: list[tuple[str, Any]] = []
    for name, candidates in sorted(_entry_points_grouped(COMMAND_GROUP).items()):
        if len(candidates) > 1:
            raise PluginError(
                f"Ambiguous command '{name}': it is registered by multiple plugins "
                f"({', '.join(sorted(ep.value for ep in candidates))}). Uninstall or "
                "rename one — SurroForge will not pick a provider for you."
            )
        loaded.append((name, candidates[0].load()))
    return loaded
```

- [ ] **Step 4: Export the new functions**

In `src/surroforge/plugins/__init__.py`, add `available_command_names` and `available_commands` to both the `from surroforge.plugins.registry import (...)` block and `__all__`:

```python
from surroforge.plugins.registry import (
    available_command_names,
    available_commands,
    available_exporters,
    available_reporters,
    available_trainers,
    get_exporter,
    get_reporter,
    get_trainer,
)
```

and in `__all__` add `"available_command_names",` and `"available_commands",` (keep alphabetical with the rest).

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_command_seam.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/plugins/registry.py src/surroforge/plugins/__init__.py tests/test_command_seam.py
git commit -m "feat(core): add surroforge.commands plugin discovery seam"
```

---

## Task 2: Core Command Seam — Mount Into CLI

**Files:**
- Modify: `src/surroforge/cli/main.py`
- Test: `tests/test_command_seam.py`

**Interfaces:**
- Consumes: `available_commands()` (Task 1); `_guard` (existing, `src/surroforge/cli/main.py:86`).
- Produces: `_mount_plugin_commands(target: typer.Typer, commands: list[tuple[str, typer.Typer]]) -> None` — wraps every command callback of each sub-app with `_guard`, then `target.add_typer(sub_app, name=name)`.

- [ ] **Step 1: Write the failing mount test**

Append to `tests/test_command_seam.py`:

```python
import typer
from typer.testing import CliRunner

from surroforge.cli.main import _mount_plugin_commands
from surroforge.errors import DataError


def _build_plugin_app() -> typer.Typer:
    sub = typer.Typer(no_args_is_help=True)

    @sub.command("ok")
    def ok(name: str = typer.Argument(...)) -> None:
        typer.echo(f"hello {name}")

    @sub.command("boom")
    def boom() -> None:
        raise DataError("kaboom")

    return sub


def test_mount_adds_subcommand_and_passes_args():
    root = typer.Typer()
    _mount_plugin_commands(root, [("demo", _build_plugin_app())])

    result = CliRunner().invoke(root, ["demo", "ok", "World"])
    assert result.exit_code == 0
    assert "hello World" in result.stdout


def test_mount_wraps_callbacks_with_guard():
    root = typer.Typer()
    _mount_plugin_commands(root, [("demo", _build_plugin_app())])

    result = CliRunner().invoke(root, ["demo", "boom"])
    assert result.exit_code == 1
    assert "Error: kaboom" in result.stderr


def test_mount_handles_no_commands():
    root = typer.Typer()
    _mount_plugin_commands(root, [])  # must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_command_seam.py -q`
Expected: FAIL — `ImportError: cannot import name '_mount_plugin_commands'`.

- [ ] **Step 3: Implement the mount helper and call it at import**

In `src/surroforge/cli/main.py`, add the import near the existing plugin import (line 26 `from surroforge.plugins import get_trainer`):

```python
from surroforge.plugins import available_commands, get_trainer
```

Then add this function just before `def main()` (around line 550), and a call to it immediately after the function definition:

```python
def _mount_plugin_commands(target, commands) -> None:
    """Mount plugin-contributed Typer sub-apps under the root CLI.

    Each sub-app's command callbacks are wrapped with :func:`_guard` so a plugin
    raising a :class:`SurroForgeError` renders as a clean ``Error: ...`` line even
    under Typer's test runner (which bypasses :func:`main`'s top-level handler).
    """
    for name, sub_app in commands:
        for command in getattr(sub_app, "registered_commands", []):
            if command.callback is not None:
                command.callback = _guard(command.callback)
        target.add_typer(sub_app, name=name)


_mount_plugin_commands(app, available_commands())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_command_seam.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Verify the existing CLI still builds**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: PASS (no regressions from the import-time mount call).

- [ ] **Step 6: Commit**

```bash
git add src/surroforge/cli/main.py tests/test_command_seam.py
git commit -m "feat(core): mount plugin commands into the CLI seam"
```

---

## Task 3: DoE Spec Schema And Loader

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/doe/__init__.py`
- Create: `plugins/surroforge-professional/src/surroforge_professional/doe/spec.py`
- Test: `plugins/surroforge-professional/tests/test_doe_spec.py`

**Interfaces:**
- Produces:
  - `DoeError(SurroForgeError)`.
  - `ContinuousFactor(name: str, type: "continuous", bounds: tuple[float, float], num_levels: int | None)`.
  - `DiscreteFactor(name: str, type: "discrete", levels: list[float])`.
  - `CategoricalFactor(name: str, type: "categorical", categories: list[str])`.
  - `DoeSpec(design, samples, seed, case_prefix, factors)`.
  - `load_spec(path: str | Path) -> DoeSpec` — parse YAML, build, semantically validate; raises `DoeError`.
  - `validate_spec(spec: DoeSpec) -> None` — semantic checks; raises `DoeError`.

- [ ] **Step 1: Write the failing spec tests**

Create `plugins/surroforge-professional/tests/test_doe_spec.py`:

```python
from __future__ import annotations

import pytest

from surroforge_professional.doe.spec import (
    CategoricalFactor,
    ContinuousFactor,
    DiscreteFactor,
    DoeError,
    DoeSpec,
    load_spec,
)


def _write(tmp_path, text):
    path = tmp_path / "doe.yaml"
    path.write_text(text)
    return path


def test_load_valid_lhs_spec(tmp_path):
    spec = load_spec(
        _write(
            tmp_path,
            """
design: lhs
samples: 10
seed: 1
factors:
  - {name: aoa, type: continuous, bounds: [-5, 16]}
  - {name: speed, type: discrete, levels: [20, 40, 60]}
  - {name: turb, type: categorical, categories: [SA, kOmega]}
""",
        )
    )
    assert spec.design == "lhs"
    assert spec.samples == 10
    assert spec.seed == 1
    assert spec.case_prefix == "case"
    assert isinstance(spec.factors[0], ContinuousFactor)
    assert isinstance(spec.factors[1], DiscreteFactor)
    assert isinstance(spec.factors[2], CategoricalFactor)


def test_missing_file_raises_doe_error(tmp_path):
    with pytest.raises(DoeError, match="not found"):
        load_spec(tmp_path / "nope.yaml")


def test_unknown_design_raises_doe_error(tmp_path):
    with pytest.raises(DoeError):
        load_spec(_write(tmp_path, "design: grid\nsamples: 4\nfactors: [{name: a, type: discrete, levels: [1,2]}]\n"))


def test_samples_forbidden_for_full_factorial(tmp_path):
    with pytest.raises(DoeError, match="full-factorial does not take 'samples'"):
        load_spec(
            _write(
                tmp_path,
                "design: full-factorial\nsamples: 4\nfactors: [{name: a, type: discrete, levels: [1,2]}]\n",
            )
        )


def test_full_factorial_continuous_needs_num_levels(tmp_path):
    with pytest.raises(DoeError, match="num_levels"):
        load_spec(
            _write(
                tmp_path,
                "design: full-factorial\nfactors: [{name: a, type: continuous, bounds: [0, 1]}]\n",
            )
        )


def test_sampling_design_requires_samples(tmp_path):
    with pytest.raises(DoeError, match="requires 'samples'"):
        load_spec(_write(tmp_path, "design: lhs\nfactors: [{name: a, type: discrete, levels: [1,2]}]\n"))


def test_duplicate_factor_names_rejected(tmp_path):
    with pytest.raises(DoeError, match="duplicate factor name"):
        load_spec(
            _write(
                tmp_path,
                "design: lhs\nsamples: 4\nfactors: [{name: a, type: discrete, levels: [1]}, "
                "{name: a, type: discrete, levels: [2]}]\n",
            )
        )


def test_bounds_require_lo_below_hi(tmp_path):
    with pytest.raises(DoeError, match="lo < hi"):
        load_spec(
            _write(
                tmp_path,
                "design: lhs\nsamples: 4\nfactors: [{name: a, type: continuous, bounds: [3, 1]}]\n",
            )
        )


def test_empty_levels_rejected(tmp_path):
    with pytest.raises(DoeError, match="non-empty"):
        load_spec(_write(tmp_path, "design: lhs\nsamples: 4\nfactors: [{name: a, type: discrete, levels: []}]\n"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge_professional.doe'`.

- [ ] **Step 3: Create the doe package marker**

Create `plugins/surroforge-professional/src/surroforge_professional/doe/__init__.py`:

```python
"""Design-of-experiments generation for SurroForge Professional."""

from surroforge_professional.doe.spec import (
    CategoricalFactor,
    ContinuousFactor,
    DiscreteFactor,
    DoeError,
    DoeSpec,
    load_spec,
    validate_spec,
)

__all__ = [
    "CategoricalFactor",
    "ContinuousFactor",
    "DiscreteFactor",
    "DoeError",
    "DoeSpec",
    "load_spec",
    "validate_spec",
]
```

- [ ] **Step 4: Implement the spec module**

Create `plugins/surroforge-professional/src/surroforge_professional/doe/spec.py`:

```python
"""DoE specification schema, loader and semantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from surroforge.errors import SurroForgeError


class DoeError(SurroForgeError):
    """Raised for an invalid DoE specification or generation request."""


class _FactorBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class ContinuousFactor(_FactorBase):
    type: Literal["continuous"]
    bounds: tuple[float, float]
    num_levels: int | None = None


class DiscreteFactor(_FactorBase):
    type: Literal["discrete"]
    levels: list[float]


class CategoricalFactor(_FactorBase):
    type: Literal["categorical"]
    categories: list[str]


Factor = Annotated[
    Union[ContinuousFactor, DiscreteFactor, CategoricalFactor],
    Field(discriminator="type"),
]


class DoeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    design: Literal["full-factorial", "lhs", "sobol", "random"]
    samples: int | None = None
    seed: int = 0
    case_prefix: str = "case"
    factors: list[Factor]


def validate_spec(spec: DoeSpec) -> None:
    """Semantic validation beyond field shapes; raises :class:`DoeError`."""
    if not spec.factors:
        raise DoeError("a DoE spec must declare at least one factor.")

    names = [f.name for f in spec.factors]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise DoeError(f"duplicate factor name(s): {', '.join(dupes)}.")

    for factor in spec.factors:
        if isinstance(factor, ContinuousFactor):
            lo, hi = factor.bounds
            if not lo < hi:
                raise DoeError(f"factor '{factor.name}': bounds require lo < hi, got [{lo}, {hi}].")
            if factor.num_levels is not None and factor.num_levels < 2:
                raise DoeError(
                    f"factor '{factor.name}': num_levels must be >= 2, got {factor.num_levels}."
                )
        elif isinstance(factor, DiscreteFactor):
            if not factor.levels:
                raise DoeError(f"factor '{factor.name}': levels must be non-empty.")
        else:
            if not factor.categories:
                raise DoeError(f"factor '{factor.name}': categories must be non-empty.")

    if spec.design == "full-factorial":
        if spec.samples is not None:
            raise DoeError(
                "full-factorial does not take 'samples'; the count is the product of factor levels."
            )
        for factor in spec.factors:
            if isinstance(factor, ContinuousFactor) and factor.num_levels is None:
                raise DoeError(
                    f"factor '{factor.name}': continuous factors need 'num_levels' under full-factorial."
                )
    else:
        if spec.samples is None or spec.samples < 1:
            raise DoeError(f"design '{spec.design}' requires 'samples' >= 1.")


def load_spec(path: str | Path) -> DoeSpec:
    """Read a DoE spec YAML file, parse it, and validate it."""
    path = Path(path)
    if not path.is_file():
        raise DoeError(f"DoE spec file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise DoeError(f"{path}: DoE spec must be a YAML mapping.")
    try:
        spec = DoeSpec(**raw)
    except ValidationError as exc:
        raise DoeError(f"{path}: invalid DoE spec:\n{exc}") from exc
    validate_spec(spec)
    return spec
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_spec.py -q`
Expected: PASS (9 passed).

- [ ] **Step 6: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/doe/__init__.py plugins/surroforge-professional/src/surroforge_professional/doe/spec.py plugins/surroforge-professional/tests/test_doe_spec.py
git commit -m "feat(professional): add DoE spec schema and loader"
```

---

## Task 4: Sampling Core

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/doe/sampling.py`
- Test: `plugins/surroforge-professional/tests/test_doe_sampling.py`

**Interfaces:**
- Consumes: `DoeSpec`, the factor classes, `DoeError` (Task 3).
- Produces: `generate(spec: DoeSpec) -> pandas.DataFrame` — one column per factor (original names; floats for numeric, strings for categorical); rows ordered by `itertools.product` (declared factor order) for full-factorial, else `samples` rows.

- [ ] **Step 1: Write the failing sampling tests**

Create `plugins/surroforge-professional/tests/test_doe_sampling.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from surroforge_professional.doe.sampling import generate
from surroforge_professional.doe.spec import DoeSpec


def _spec(**kw):
    return DoeSpec(**kw)


def test_full_factorial_is_cartesian_product():
    spec = _spec(
        design="full-factorial",
        factors=[
            {"name": "a", "type": "discrete", "levels": [1, 2, 3]},
            {"name": "b", "type": "categorical", "categories": ["x", "y"]},
        ],
    )
    frame = generate(spec)
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 6  # 3 x 2
    assert set(frame["b"]) == {"x", "y"}
    assert set(frame["a"]) == {1.0, 2.0, 3.0}


def test_full_factorial_continuous_uses_linspace():
    spec = _spec(
        design="full-factorial",
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1], "num_levels": 3}],
    )
    frame = generate(spec)
    assert list(frame["a"]) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("design", ["lhs", "sobol", "random"])
def test_sampling_designs_respect_count_and_bounds(design):
    spec = _spec(
        design=design,
        samples=8,
        seed=1,
        factors=[
            {"name": "aoa", "type": "continuous", "bounds": [-5, 16]},
            {"name": "speed", "type": "discrete", "levels": [20, 40, 60]},
            {"name": "turb", "type": "categorical", "categories": ["SA", "kOmega"]},
        ],
    )
    frame = generate(spec)
    assert len(frame) == 8
    assert frame["aoa"].between(-5, 16).all()
    assert set(frame["speed"]).issubset({20.0, 40.0, 60.0})
    assert set(frame["turb"]).issubset({"SA", "kOmega"})


@pytest.mark.parametrize("design", ["lhs", "sobol", "random"])
def test_sampling_is_deterministic_per_seed(design):
    kw = dict(
        design=design,
        samples=8,
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1]}],
    )
    same = generate(_spec(seed=1, **kw)).equals(generate(_spec(seed=1, **kw)))
    differ = generate(_spec(seed=1, **kw)).equals(generate(_spec(seed=2, **kw)))
    assert same is True
    assert differ is False


def test_sobol_non_power_of_two_warns():
    spec = _spec(
        design="sobol",
        samples=6,
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1]}],
    )
    with pytest.warns(UserWarning, match="power of two"):
        generate(spec)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_sampling.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge_professional.doe.sampling'`.

- [ ] **Step 3: Implement the sampling core**

Create `plugins/surroforge-professional/src/surroforge_professional/doe/sampling.py`:

```python
"""Sampling strategies that turn a DoeSpec into a table of points."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from surroforge_professional.doe.spec import (
    CategoricalFactor,
    ContinuousFactor,
    DiscreteFactor,
    DoeError,
    DoeSpec,
)


def _require_qmc():
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise DoeError(
            "LHS and Sobol designs require scipy. Install it with "
            "`pip install 'surroforge-professional[doe]'`."
        ) from exc
    return qmc


def _factorial_levels(factor) -> list:
    if isinstance(factor, ContinuousFactor):
        lo, hi = factor.bounds
        return [float(v) for v in np.linspace(lo, hi, factor.num_levels)]
    if isinstance(factor, DiscreteFactor):
        return [float(v) for v in factor.levels]
    return list(factor.categories)


def _discrete_choices(factor) -> list | None:
    if isinstance(factor, DiscreteFactor):
        return [float(v) for v in factor.levels]
    if isinstance(factor, CategoricalFactor):
        return list(factor.categories)
    return None


def _map_unit(factor, column: np.ndarray) -> list:
    if isinstance(factor, ContinuousFactor):
        lo, hi = factor.bounds
        return [float(v) for v in (lo + column * (hi - lo))]
    choices = _discrete_choices(factor)
    k = len(choices)
    index = np.minimum((column * k).astype(int), k - 1)
    return [choices[i] for i in index]


def _unit_hypercube(spec: DoeSpec, dimensions: int) -> np.ndarray:
    if spec.design == "lhs":
        qmc = _require_qmc()
        return qmc.LatinHypercube(d=dimensions, seed=spec.seed).random(spec.samples)
    if spec.design == "sobol":
        qmc = _require_qmc()
        n = spec.samples
        if n & (n - 1):  # not a power of two
            warnings.warn(
                f"Sobol sampling is most uniform at a power of two; got samples={n}.",
                UserWarning,
                stacklevel=2,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence scipy's own balance warning
            return qmc.Sobol(d=dimensions, seed=spec.seed).random(n)
    rng = np.random.default_rng(spec.seed)
    return rng.random((spec.samples, dimensions))


def generate(spec: DoeSpec) -> pd.DataFrame:
    """Return a DataFrame of sample points, one column per factor."""
    if spec.design == "full-factorial":
        columns = [_factorial_levels(f) for f in spec.factors]
        rows = list(itertools.product(*columns))
        data = {f.name: [row[i] for row in rows] for i, f in enumerate(spec.factors)}
        return pd.DataFrame(data)

    unit = _unit_hypercube(spec, len(spec.factors))
    data = {f.name: _map_unit(f, unit[:, i]) for i, f in enumerate(spec.factors)}
    return pd.DataFrame(data)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_sampling.py -q`
Expected: PASS (all parametrized cases pass).

- [ ] **Step 5: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/doe/sampling.py plugins/surroforge-professional/tests/test_doe_sampling.py
git commit -m "feat(professional): add DoE sampling strategies"
```

---

## Task 5: Output Writer

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/doe/output.py`
- Test: `plugins/surroforge-professional/tests/test_doe_output.py`

**Interfaces:**
- Consumes: `DoeSpec`, `DoeError` (Task 3); `generate` (Task 4); `surroforge_professional.__version__`.
- Produces: `write_design(spec: DoeSpec, out_dir: str | Path, force: bool = False) -> tuple[Path, Path]` — writes `samples.csv` (a `case` column zero-padded to min width 3, then factor columns) and `doe.manifest.json`; returns `(csv_path, manifest_path)`. Raises `DoeError` if a target file exists and `force` is False.

- [ ] **Step 1: Write the failing output tests**

Create `plugins/surroforge-professional/tests/test_doe_output.py`:

```python
from __future__ import annotations

import json

import pandas as pd
import pytest

from surroforge_professional.doe.output import write_design
from surroforge_professional.doe.spec import DoeError, DoeSpec


def _spec():
    return DoeSpec(
        design="full-factorial",
        factors=[
            {"name": "aoa", "type": "discrete", "levels": [0, 4, 8]},
            {"name": "speed", "type": "discrete", "levels": [20, 40]},
        ],
    )


def test_writes_samples_csv_with_case_column(tmp_path):
    csv_path, manifest_path = write_design(_spec(), tmp_path)

    assert csv_path == tmp_path / "samples.csv"
    assert manifest_path == tmp_path / "doe.manifest.json"
    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["case", "aoa", "speed"]
    assert len(frame) == 6
    assert frame["case"].iloc[0] == "case_000"
    assert frame["case"].iloc[-1] == "case_005"


def test_manifest_records_design_seed_and_hash(tmp_path):
    _, manifest_path = write_design(_spec(), tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["design"] == "full-factorial"
    assert manifest["samples"] == 6
    assert manifest["seed"] == 0
    assert len(manifest["factors"]) == 2
    assert isinstance(manifest["table_sha256"], str) and len(manifest["table_sha256"]) == 64
    assert "plugin_version" in manifest


def test_case_padding_widens_for_large_designs(tmp_path):
    spec = DoeSpec(
        design="lhs",
        samples=1000,
        factors=[{"name": "a", "type": "continuous", "bounds": [0, 1]}],
    )
    csv_path, _ = write_design(spec, tmp_path)
    frame = pd.read_csv(csv_path)
    assert frame["case"].iloc[0] == "case_0000"
    assert frame["case"].iloc[-1] == "case_0999"


def test_refuses_to_overwrite_without_force(tmp_path):
    write_design(_spec(), tmp_path)
    with pytest.raises(DoeError, match="already exists"):
        write_design(_spec(), tmp_path)


def test_force_overwrites(tmp_path):
    write_design(_spec(), tmp_path)
    csv_path, _ = write_design(_spec(), tmp_path, force=True)
    assert csv_path.is_file()


def test_output_columns_match_openfoam_ingest_contract(tmp_path):
    # `read_openfoam_sweep` / `_openfoam_params` expect a 'case' column plus
    # one numeric parameter column per factor. Confirm the contract shape.
    csv_path, _ = write_design(_spec(), tmp_path)
    frame = pd.read_csv(csv_path)
    assert frame.columns[0] == "case"
    assert {"aoa", "speed"}.issubset(set(frame.columns))
    assert frame["case"].is_unique
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_output.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge_professional.doe.output'`.

- [ ] **Step 3: Implement the output writer**

Create `plugins/surroforge-professional/src/surroforge_professional/doe/output.py`:

```python
"""Write a generated design to samples.csv plus a reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from surroforge_professional import __version__
from surroforge_professional.doe.sampling import generate
from surroforge_professional.doe.spec import DoeError, DoeSpec

CSV_NAME = "samples.csv"
MANIFEST_NAME = "doe.manifest.json"


def _case_ids(count: int, prefix: str) -> list[str]:
    width = max(3, len(str(count - 1))) if count > 0 else 3
    return [f"{prefix}_{i:0{width}d}" for i in range(count)]


def write_design(spec: DoeSpec, out_dir: str | Path, force: bool = False) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    csv_path = out_dir / CSV_NAME
    manifest_path = out_dir / MANIFEST_NAME

    if not force:
        for path in (csv_path, manifest_path):
            if path.exists():
                raise DoeError(f"{path} already exists; pass force=True (or --force) to overwrite.")

    frame = generate(spec)
    frame.insert(0, "case", _case_ids(len(frame), spec.case_prefix))

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    table_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "design": spec.design,
        "samples": int(len(frame)),
        "seed": spec.seed,
        "case_prefix": spec.case_prefix,
        "factors": [factor.model_dump() for factor in spec.factors],
        "table_sha256": table_hash,
        "plugin_version": __version__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return csv_path, manifest_path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_output.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/doe/output.py plugins/surroforge-professional/tests/test_doe_output.py
git commit -m "feat(professional): write DoE samples.csv and manifest"
```

---

## Task 6: DoE CLI Command, Entry Point, License Gate

**Files:**
- Create: `plugins/surroforge-professional/src/surroforge_professional/commands/__init__.py`
- Create: `plugins/surroforge-professional/src/surroforge_professional/commands/doe.py`
- Modify: `plugins/surroforge-professional/pyproject.toml`
- Modify: `plugins/surroforge-professional/README.md`
- Test: `plugins/surroforge-professional/tests/test_doe_command.py`

**Interfaces:**
- Consumes: `load_spec` (Task 3), `write_design` (Task 5), `require_license` (existing `surroforge_professional/license.py`), the `surroforge.commands` seam (Tasks 1–2).
- Produces: `doe_app` (a `typer.Typer`) with a `run` command; registered as `doe` under `surroforge.commands`.

- [ ] **Step 1: Write the failing command app**

Create `plugins/surroforge-professional/tests/test_doe_command.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from surroforge.errors import PluginError
from surroforge_professional.commands.doe import doe_app


def _spec_file(tmp_path):
    path = tmp_path / "doe.yaml"
    path.write_text(
        "design: full-factorial\n"
        "factors:\n"
        "  - {name: aoa, type: discrete, levels: [0, 4, 8]}\n"
        "  - {name: speed, type: discrete, levels: [20, 40]}\n"
    )
    return path


def test_run_writes_outputs(licensed, tmp_path):
    spec = _spec_file(tmp_path)
    out = tmp_path / "design"
    result = CliRunner().invoke(doe_app, ["run", str(spec), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    frame = pd.read_csv(out / "samples.csv")
    assert list(frame.columns) == ["case", "aoa", "speed"]
    assert (out / "doe.manifest.json").is_file()


def test_run_requires_license_before_reading_spec(monkeypatch, tmp_path):
    monkeypatch.delenv("SURROFORGE_PRO_LICENSE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    missing = tmp_path / "does-not-exist.yaml"

    # License is checked first: a missing-license error, not a missing-file error.
    with pytest.raises(PluginError, match="Professional license required"):
        from surroforge_professional.commands.doe import run as run_cmd

        run_cmd(spec=missing, out=tmp_path, force=False)
```

> Note: the `licensed` fixture (sets `SURROFORGE_PRO_LICENSE` and `HOME`) is defined in `plugins/surroforge-professional/tests/conftest.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_command.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'surroforge_professional.commands'`.

- [ ] **Step 3: Create the commands package and DoE app**

Create `plugins/surroforge-professional/src/surroforge_professional/commands/__init__.py`:

```python
"""CLI commands contributed by SurroForge Professional."""
```

Create `plugins/surroforge-professional/src/surroforge_professional/commands/doe.py`:

```python
"""`surroforge doe` — generate design-of-experiments sample tables."""

from __future__ import annotations

from pathlib import Path

import typer

from surroforge_professional.doe.output import write_design
from surroforge_professional.doe.spec import load_spec
from surroforge_professional.license import require_license

doe_app = typer.Typer(
    help="Generate design-of-experiments sample tables (Professional).",
    no_args_is_help=True,
)


@doe_app.command("run")
def run(
    spec: Path = typer.Argument(..., help="Path to a DoE spec YAML file."),
    out: Path = typer.Option(Path("."), "--out", help="Output directory for the design."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output files."),
) -> None:
    """Generate samples.csv and doe.manifest.json from a DoE spec."""
    require_license()
    parsed = load_spec(spec)
    csv_path, manifest_path = write_design(parsed, out, force=force)
    typer.echo(f"Wrote {csv_path}")
    typer.echo(f"Wrote {manifest_path}")
```

- [ ] **Step 4: Run the command tests to verify they pass**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_command.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Register the entry point and scipy extra**

In `plugins/surroforge-professional/pyproject.toml`, after the existing `[project.entry-points."surroforge.reporters"]` block, add:

```toml
[project.entry-points."surroforge.commands"]
doe = "surroforge_professional.commands.doe:doe_app"
```

And in `[project.optional-dependencies]`, add a `doe` extra alongside the existing `dev` entry:

```toml
doe = ["scipy>=1.10"]
```

- [ ] **Step 6: Reinstall the plugin so the entry point is published**

Run: `.venv/bin/python -m pip install -e plugins/surroforge-professional`
Expected: reinstalls `surroforge-professional` with the new `surroforge.commands` entry point.

- [ ] **Step 7: Write the registration + end-to-end CLI test**

Append to `plugins/surroforge-professional/tests/test_doe_command.py`:

```python
def test_doe_registered_as_command():
    from surroforge.plugins import available_command_names

    assert "doe" in available_command_names()


def test_surroforge_doe_run_end_to_end(licensed, tmp_path):
    # The plugin must be installed (editable) for the entry point to resolve.
    import importlib

    from surroforge.cli import main as cli_main

    importlib.reload(cli_main)  # re-mount commands now that the entry point exists
    spec = _spec_file(tmp_path)
    out = tmp_path / "design"
    result = CliRunner().invoke(cli_main.app, ["doe", "run", str(spec), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    assert (out / "samples.csv").is_file()
```

- [ ] **Step 8: Run the full command test file**

Run: `.venv/bin/python -m pytest plugins/surroforge-professional/tests/test_doe_command.py -q`
Expected: PASS (4 passed).

- [ ] **Step 9: Document the command in the plugin README**

In `plugins/surroforge-professional/README.md`, add a section after the existing CLI usage:

````markdown
## DoE Generation

Generate a design-of-experiments sample table from a YAML spec:

```yaml
# doe.yaml
design: lhs            # full-factorial | lhs | sobol | random
samples: 50
seed: 1
factors:
  - {name: aoa, type: continuous, bounds: [-5, 16]}
  - {name: speed, type: discrete, levels: [20, 40, 60, 80, 100]}
```

```bash
surroforge doe run doe.yaml --out ./design
```

This writes `design/samples.csv` (a `case` column plus one column per factor —
exactly what `surroforge data add-openfoam --samples` consumes) and
`design/doe.manifest.json` (a reproducibility record). LHS/Sobol need the extra:
`pip install 'surroforge-professional[doe]'`.
````

- [ ] **Step 10: Commit**

```bash
git add plugins/surroforge-professional/src/surroforge_professional/commands plugins/surroforge-professional/pyproject.toml plugins/surroforge-professional/README.md plugins/surroforge-professional/tests/test_doe_command.py
git commit -m "feat(professional): add surroforge doe command"
```

---

## Task 7: Final Verification, Changelog

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run the whole DoE + seam suite**

Run: `.venv/bin/python -m pytest tests/test_command_seam.py plugins/surroforge-professional/tests -q`
Expected: PASS.

- [ ] **Step 2: Run Core regression slices**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_plugins.py -q`
Expected: PASS (the new import-time mount does not regress existing CLI/plugin behavior).

- [ ] **Step 3: Lint the new code**

Run: `.venv/bin/python -m ruff check src plugins/surroforge-professional`
Expected: PASS (line-length 100).

- [ ] **Step 4: Add the changelog entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add a bullet to `### Added — Plugins` (after the `surroforge-autoencoder` entry):

```markdown
- **DoE generator (`surroforge doe`)** — the `surroforge-professional` plugin adds a
  design-of-experiments generator on a new generic Core command seam (entry-point group
  `surroforge.commands`, the first plugin-contributed CLI command). `surroforge doe run
  spec.yaml` reads a Pydantic-validated YAML spec (numeric continuous/discrete and
  categorical factors), samples it by **full-factorial**, **Latin Hypercube**, **Sobol**,
  or **random** (LHS/Sobol via the optional `scipy` extra `surroforge-professional[doe]`,
  imported lazily), and writes `samples.csv` (a `case` column plus one column per factor —
  exactly what `surroforge data add-openfoam --samples` consumes) plus a reproducibility
  `doe.manifest.json`. License-gated like the HTML reporter; deterministic per `seed`.
```

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): document the DoE generator"
```

---

## Self-Review

**Spec coverage:**
- Pure plan generator, no solver calls → Tasks 3–6 (no solver code anywhere). ✓
- Professional plugin, reuse license gate → Task 6 (`require_license`). ✓
- New Core command seam `surroforge.commands` → Tasks 1–2. ✓
- YAML spec, numeric + categorical factors → Task 3. ✓
- Four strategies + unit-interval mapping + determinism + Sobol warning → Task 4. ✓
- `samples.csv` (case + factor cols) + `doe.manifest.json` + hash → Task 5. ✓
- CLI `surroforge doe run --out/--force` + mirrored API (`load_spec`/`generate`/`write_design`) → Tasks 3–6. ✓
- License-first, `DoeError`, scipy-missing message, unified rendering → Tasks 3, 4, 6 (+ `_guard` in Task 2). ✓
- Tests: plugin-local + Core seam; column contract vs OpenFOAM ingest → Tasks 1–6. ✓
- Out-of-scope items (adaptive sampling, RSM, project registration) → not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `DoeSpec`, `DoeError`, `ContinuousFactor`/`DiscreteFactor`/`CategoricalFactor`, `load_spec`, `validate_spec`, `generate`, `write_design`, `doe_app`/`run`, `available_command_names`/`available_commands`, `_mount_plugin_commands`, `COMMAND_GROUP` are named identically across all tasks. ✓
