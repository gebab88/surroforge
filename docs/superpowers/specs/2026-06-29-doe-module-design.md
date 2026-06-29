# DoE Module Design

## Goal

Add a design-of-experiments (DoE) generator to SurroForge: given a parameter
space, write a sample table of points to evaluate. It is a **pure plan
generator** — it produces a `samples.csv` for any external solver and never calls
a solver itself (PRD §8: "keine direkten FEM/CFD-Aufrufe"). The output is exactly
the table `surroforge data add-openfoam --samples` already consumes, so the loop
closes: `doe` generates the plan → the user runs their solver → `data add-openfoam`
ingests the results.

The capability is a **Professional feature**. It ships inside the existing
`surroforge-professional` plugin (reusing its license gate), surfaced as
`surroforge doe ...` through a new, generic Core command seam.

## Architecture

### Core command seam (Apache-2.0, generic)

Core gains a new entry-point group **`surroforge.commands`**, alongside the
existing `surroforge.trainers|exporters|reporters`.

- `src/surroforge/plugins/` gains `available_commands()`, which loads each
  registered entry point as a `typer.Typer` sub-app.
- `src/surroforge/cli/main.py` mounts them once at app construction via
  `app.add_typer(sub_app, name=ep.name)`. No builtin-fallback table is needed
  (Core ships no plugin commands of its own); discovery is robust when zero
  plugins are present.
- Mounted commands are wrapped by the same `_guard` rendering as Core commands so
  errors render as `Error: …` with exit code 1.

The seam is reusable for future plugin commands; it carries no DoE-specific logic.

### DoE in `surroforge-professional`

New modules in the existing plugin:

- `…/doe/spec.py` — the Pydantic `DoeSpec` model and `load_spec(path)`.
- `…/doe/sampling.py` — `generate(spec) -> pandas.DataFrame` and the four
  strategies. Heavy imports (`scipy.stats.qmc`) are lazy inside the functions so
  `surroforge --help` stays fast.
- `…/doe/output.py` — `write_design(spec, out_dir, force=False) -> (csv_path, manifest_path)`.
- `…/commands/doe.py` — the `doe` Typer sub-app and the license gate.
- `…/commands/__init__.py` — package marker.

The plugin's `pyproject.toml` registers `doe` under `surroforge.commands` and adds
an optional extra `doe = ["scipy>=1.10"]` (numpy/pandas come via `surroforge`).

## User-Facing Configuration

One YAML file fully specifies a design, Pydantic-validated like the rest of
SurroForge:

```yaml
# doe.yaml
design: lhs            # full-factorial | lhs | sobol | random
samples: 50            # total point count (lhs/sobol/random; forbidden for full-factorial)
seed: 1                # deterministic
case_prefix: case      # -> case_000, case_001, ...

factors:
  - name: aoa
    type: continuous   # continuous | discrete | categorical
    bounds: [-5, 16]

  - name: speed
    type: discrete     # numeric levels
    levels: [20, 40, 60, 80, 100]

  - name: turbulence
    type: categorical
    categories: [SA, kOmega]
```

### Schema rules

- `design` ∈ {`full-factorial`, `lhs`, `sobol`, `random`}.
- `samples`: required and `> 0` for `lhs`/`sobol`/`random`; **forbidden** for
  `full-factorial` (its count is the product of the factor levels).
- `seed`: integer, default `0`.
- `case_prefix`: string, default `case`.
- Factor types:
  - `continuous` → `bounds: [lo, hi]` with `lo < hi`. For `full-factorial`,
    additionally `num_levels: k` (`k ≥ 2`) → `linspace(lo, hi, k)`.
  - `discrete` → `levels: [...]`, numeric, non-empty.
  - `categorical` → `categories: [...]`, non-empty.
- Factor names are unique.
- For `full-factorial`, every factor must be discretizable: `discrete`,
  `categorical`, or `continuous` with `num_levels`.

## Sampling Core

`generate(spec)` returns a DataFrame with one column per factor (original names and
values; numeric for numeric factors, strings for categorical).

- **full-factorial** — cartesian product (`itertools.product`) of each factor's
  levels: continuous → `linspace(bounds, num_levels)`, discrete → `levels`,
  categorical → `categories`.
- **lhs** — `scipy.stats.qmc.LatinHypercube(d, seed=seed).random(samples)`.
- **sobol** — `scipy.stats.qmc.Sobol(d, seed=seed).random(samples)`. A non
  power-of-two `samples` emits a warning (Sobol balance is best at powers of two)
  but is not an error.
- **random** — `numpy.random.default_rng(seed).random((samples, d))`.

For `lhs`/`sobol`/`random`, each unit-interval value `u ∈ [0, 1)` maps per factor:

- continuous → `lo + u·(hi − lo)`
- discrete / categorical → index `min(int(u·k), k − 1)` → level / category.

This preserves the space-filling structure across continuous dimensions while
sampling discrete factors roughly uniformly.

**Determinism:** every strategy takes `seed`; identical spec + seed → an identical
table.

## Output

`write_design` writes two files to the output directory:

1. **`samples.csv`** — a `case` column (`case_000`, … zero-padded to the case
   count, minimum width 3) followed by one column per factor with original
   names/values. This is exactly the format `surroforge data add-openfoam
   --samples` and `read_openfoam_sweep` consume.
2. **`doe.manifest.json`** — a reproducibility record: design method, `samples`,
   `seed`, the factor spec, a content hash of the table, and the plugin version.
   No project registration (this is the pure generator); the manifest only keeps a
   plan auditable/reproducible.

## CLI / API

CLI (mounted through the Core seam):

```bash
surroforge doe run doe.yaml [--out DIR] [--force]
```

- `run <spec.yaml>` → writes `samples.csv` + `doe.manifest.json` to `--out DIR`
  (default: current directory). `--force` overwrites existing files.
- No spec-overriding flags; the YAML is the sole source of truth. `--out`/`--force`
  affect output only.

Python API (mirrors the CLI, per the project convention):

- `load_spec(path) -> DoeSpec`
- `generate(spec) -> pandas.DataFrame`
- `write_design(spec, out_dir, force=False) -> (csv_path, manifest_path)`

## Error Handling

Consistent with Core (`SurroForgeError` + `_guard` → `Error: …`, exit 1):

- **License gate first** — a missing Professional license raises `PluginError`
  with an actionable message *before* the spec is read (reusing the plugin's
  existing `license.py`).
- A new `DoeError(SurroForgeError)` covers: missing spec file, unknown `design`
  method, `samples` missing/invalid for the method, `full-factorial` with a
  non-discretizable factor, `lo ≥ hi`, empty `levels`/`categories`.
- `scipy` not installed → a clear message to install `surroforge-professional[doe]`
  (lazy-import guard).

## Scope

v1 includes the four strategies above, numeric (continuous + discrete) and
categorical factors, the YAML spec, the `samples.csv` + manifest output, the
`surroforge doe run` command, and the mirrored Python API.

Out of scope for v1 (explicitly deferred): adaptive/active-learning sampling
(PRD Phase 3), response-surface designs (central-composite, Box-Behnken),
project-dataset registration of the generated plan, and any solver invocation.

## Validation / Testing

- **Plugin-local** (`plugins/surroforge-professional/tests/`): spec validation
  (valid plus each failure mode); per strategy shape/bounds/determinism (same seed
  identical, different seed different); `full-factorial` count = product of levels;
  discrete/categorical values ∈ declared sets; output format + case zero-padding +
  manifest hash; the column contract against `read_openfoam_sweep` /
  `_openfoam_params`; the license gate firing before any file read; the CLI
  end-to-end on a temp spec.
- **Core** (`tests/`): the new `surroforge.commands` seam — `available_commands()`
  discovery, the command mounted into the app, and graceful behavior with zero
  registered command plugins.
