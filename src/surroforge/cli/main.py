"""SurroForge command-line interface (PRD 4.2).

Commands mirror the Python API one-to-one. Domain errors (:class:`SurroForgeError`)
are caught by the :func:`_guard` decorator on each command and rendered as a clean
``Error: ...`` line instead of a traceback.
"""

from __future__ import annotations

import functools
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import typer

from surroforge import __version__, services
from surroforge.core.config import load_config
from surroforge.core.project import Project
from surroforge.core.run import FieldCoverageWarning, ProvenanceWarning, Run
from surroforge.data import SUPPORTED_SUFFIXES, load_table, read_openfoam_sweep
from surroforge.engine import run_training
from surroforge.errors import DataError, SurroForgeError
from surroforge.plugins import available_commands, get_trainer

app = typer.Typer(
    help="SurroForge — surrogate models for FEM/CFD simulation data.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Register and inspect datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")
rom_app = typer.Typer(help="Reduced-order model (ROM) inspection commands.", no_args_is_help=True)
app.add_typer(rom_app, name="rom")


@dataclass
class _State:
    verbose: bool = False
    quiet: bool = False
    config_dir: Path | None = None


_STATE = _State()


# --- Global options ----------------------------------------------------------

@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output."),
    config_dir: Path = typer.Option(
        None, "--config-dir", help="Project directory (default: discovered from the cwd)."
    ),
) -> None:
    _STATE.verbose = verbose
    _STATE.quiet = quiet
    _STATE.config_dir = config_dir


# --- Output helpers ----------------------------------------------------------

def _info(message: str) -> None:
    if not _STATE.quiet:
        typer.echo(message)


def _success(message: str) -> None:
    if not _STATE.quiet:
        typer.secho(message, fg=typer.colors.GREEN)


def _echo_cb():
    return (lambda m: typer.echo(f"  · {m}")) if _STATE.verbose else None


def _project() -> Project:
    if _STATE.config_dir is not None:
        return Project.open(_STATE.config_dir)
    return Project.discover()


def _guard(func):
    """Render :class:`SurroForgeError` as a clean ``Error:`` line and exit 1.

    Applied to every command so the same friendly behaviour holds whether the
    CLI is run for real (via :func:`main`) or invoked through Typer's test runner.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SurroForgeError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


def _metrics_summary(run: Run) -> None:
    metrics = run.metrics
    names = run.config.validation.metrics
    cv = metrics["cross_validation"]
    _info(f"  experiment : {run.config.experiment}")
    _info(f"  method     : {run.method}")
    header = f"  {'metric':<8}{'train':>12}{'test':>12}{'cv_mean':>12}{'cv_std':>12}"
    _info(header)
    for name in names:
        _info(
            f"  {name:<8}"
            f"{metrics['train'][name]:>12.5g}"
            f"{metrics['test'][name]:>12.5g}"
            f"{cv[name]['mean']:>12.5g}"
            f"{cv[name]['std']:>12.5g}"
        )


def _save_frame(frame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix in {".h5", ".hdf5"}:
        # Raw h5py column group, matching surroforge.data.loader's HDF5 reader.
        try:
            import h5py
        except ImportError as exc:
            raise DataError(
                "Writing HDF5 requires the 'h5py' backend. "
                "Install it with: pip install 'surroforge[formats]'"
            ) from exc
        names = [str(col) for col in frame.columns]
        if len(set(names)) != len(names):
            raise DataError(
                "Cannot write HDF5: column names collide after string conversion "
                "(duplicate or str-identical labels). Rename the input columns "
                "and retry."
            )
        invalid = [name for name in names if "/" in name or name in {"", ".", ".."}]
        if invalid:
            raise DataError(
                "Cannot write HDF5: these column names are not valid h5py dataset "
                f"names (empty, '.', '..', or containing '/'): {invalid}. "
                "Rename the input columns."
            )
        arrays = {
            name: frame[col].to_numpy()
            for name, col in zip(names, frame.columns, strict=True)
        }
        non_numeric = [
            name for name, data in arrays.items() if data.dtype.kind not in "biufc"
        ]
        if non_numeric:
            raise DataError(
                f"Cannot write HDF5: columns {non_numeric} are not numeric and "
                "have no native HDF5 representation. Drop them from the input "
                "or write .csv/.parquet instead."
            )
        # Atomic like the persistence writers: h5py.File(path, "w") truncates the
        # destination immediately, so a failure mid-write must never destroy an
        # existing output file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
        os.close(fd)
        try:
            # track_order: preserve the frame's column order on read-back —
            # HDF5 otherwise iterates names lexicographically (f10 before f2).
            with h5py.File(tmp, "w", track_order=True) as handle:
                for name, data in arrays.items():
                    handle.create_dataset(name, data=data)
            # mkstemp creates 0600; a user-requested output file should honor
            # the umask like the csv/parquet branches do.
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    else:
        raise DataError(
            f"Cannot write predictions to '{suffix}'. "
            "Use .csv, .parquet/.pq, or .h5/.hdf5."
        )


# --- Commands ----------------------------------------------------------------

@app.command()
@_guard
def init(
    name: str = typer.Argument(..., help="Project name / directory to create."),
) -> None:
    """Create a new project directory (PRD 4.2)."""
    root = Path(name)
    if not root.is_absolute():
        root = Path.cwd() / root
    project = Project.init(root, name=root.name)
    _success(f"Initialized SurroForge project '{project.name}' at {project.root}")
    _info("  created: data/  configs/  runs/  reports/  surroforge.yaml")


@data_app.command("add")
@_guard
def data_add(
    file: Path = typer.Argument(..., exists=True, help="Tabular data file (CSV/Parquet/HDF5)."),
    name: str = typer.Option(None, "--name", "-n", help="Dataset name (default: file stem)."),
) -> None:
    """Register a dataset: validate schema and record a content hash (PRD 4.2)."""
    project = _project()
    dataset = project.add_dataset(file, name=name)
    _success(f"Registered dataset '{dataset.name}' ({dataset.n_rows} rows, {dataset.n_cols} cols)")
    _info(f"  path: {dataset.path}")
    _info(f"  hash: {dataset.hash}")


def _has_pyarrow() -> bool:
    import importlib.util

    return importlib.util.find_spec("pyarrow") is not None


def _openfoam_params(samples: Path) -> dict[str, dict[str, float]]:
    """Parse a DoE table (a ``case`` column + parameter columns) into the mapping
    ``read_openfoam_sweep`` expects: ``{case_name: {param: value}}``."""
    table = load_table(samples)
    if "case" not in table.columns:
        raise DataError(
            f"Samples table {samples} must have a 'case' column naming each case directory; "
            f"got columns {list(table.columns)}."
        )
    if table.empty:
        raise DataError(f"Samples table {samples} has no rows.")
    param_cols = [c for c in table.columns if c != "case"]
    if not param_cols:
        raise DataError(
            f"Samples table {samples} has only a 'case' column and no parameter columns."
        )
    # Build keys from the column directly (astype(str)), not row iteration: iterrows() upcasts
    # an integer 'case' column to float, turning case '0' into '0.0' and never matching the dir.
    cases = table["case"].astype(str).tolist()
    mapping: dict[str, dict[str, float]] = {}
    for position, case in enumerate(cases):
        if case in mapping:
            raise DataError(f"Duplicate case '{case}' in samples table {samples}.")
        case_params: dict[str, float] = {}
        for col in param_cols:
            value = table[col].iloc[position]
            try:
                case_params[col] = float(value)
            except (TypeError, ValueError) as exc:
                raise DataError(
                    f"Non-numeric parameter '{col}' in samples table {samples} for case "
                    f"'{case}': {value!r} ({exc})"
                ) from exc
        mapping[case] = case_params
    return mapping


@data_app.command("add-openfoam")
@_guard
def data_add_openfoam(
    case_root: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="OpenFOAM case-root (one sub-dir per DOE point)."
    ),
    field: str = typer.Option(..., "--field", "-f", help="Field to ingest (e.g. 'p', 'U')."),
    name: str = typer.Option(..., "--name", "-n", help="Name for the registered dataset."),
    samples: Path = typer.Option(
        None, "--samples", exists=True, help="DoE table CSV: a 'case' column + parameter columns."
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Snapshot file to write (default: data/<name>.parquet|csv)."
    ),
    time: str = typer.Option(None, "--time", help="Pin a time directory (default: the latest)."),
) -> None:
    """Ingest an OpenFOAM case sweep and register it as a dataset (PRD-ROM 4.2)."""
    project = _project()
    # Fail fast on a name collision BEFORE writing — otherwise the default output path
    # (data/<name>.<ext>) would clobber the existing dataset's backing file, then abort.
    if any(dataset.name == name for dataset in project.datasets()):
        raise DataError(f"A dataset named '{name}' is already registered.")
    if output is None:
        # Prefer a lossless default where available; CSV is fine but lossy at ULP precision.
        output = project.data_dir / (f"{name}.parquet" if _has_pyarrow() else f"{name}.csv")
    if output.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DataError(
            f"Unsupported --output format '{output.suffix}'. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}."
        )
    params = _openfoam_params(samples) if samples is not None else None
    frame = read_openfoam_sweep(case_root, field=field, params=params, time=time)
    try:
        _save_frame(frame, output)
    except OSError as exc:
        raise DataError(f"Could not write snapshot to {output}: {exc}") from exc
    dataset = project.add_dataset(output, name=name)
    _success(f"Registered dataset '{dataset.name}' ({dataset.n_rows} rows, {dataset.n_cols} cols)")
    _info(f"  field: {field}")
    _info(f"  path: {dataset.path}")
    _info(f"  hash: {dataset.hash}")


@data_app.command("list")
@_guard
def data_list() -> None:
    """List registered datasets (PRD 4.2)."""
    project = _project()
    datasets = project.datasets()
    if not datasets:
        _info("No datasets registered. Add one with 'surroforge data add <file>'.")
        return
    for dataset in datasets:
        _info(
            f"{dataset.name:<20} {dataset.n_rows:>8} rows  "
            f"{dataset.n_cols:>3} cols  {dataset.path}"
        )


@app.command()
@_guard
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="YAML training config."),
    onnx: bool = typer.Option(
        False,
        "--onnx/--no-onnx",
        help="Also export ONNX at train time (default: off — export on demand with "
        "'surroforge export <run> --format onnx').",
    ),
) -> None:
    """Train a surrogate model from a YAML configuration (PRD 4.2)."""
    project = _project()
    cfg = load_config(config)
    _info(f"Training '{cfg.experiment}' (method: {cfg.model.method}) ...")
    run = run_training(project, cfg, echo=_echo_cb(), export_onnx=onnx)
    _success(f"Run {run.run_id} complete.")
    _metrics_summary(run)
    # Discoverability: ONNX is now opt-in, so point users at the on-demand path
    # when the trained method actually supports it (e.g. polynomial).
    if not onnx and "onnx" in getattr(get_trainer(cfg.model.method), "export_formats", frozenset()):
        _info(f"  ONNX export available — run 'surroforge export {run.run_id} --format onnx'.")


@app.command()
@_guard
def validate(run_id: str = typer.Argument(..., help="Run id to validate.")) -> None:
    """Show a run's validation metrics (PRD 4.2)."""
    run = _project().run(run_id)
    _success(f"Validation metrics for {run_id}:")
    _metrics_summary(run)


@app.command()
@_guard
def compare(
    runs: str = typer.Option(..., "--runs", "-r", help="Comma-separated run ids."),
    metric: str = typer.Option("rmse", "--metric", "-m", help="Metric used to pick the best run."),
) -> None:
    """Compare several training runs side by side (PRD 4.2)."""
    project = _project()
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    result = services.compare_runs(project, run_ids, metric=metric)
    header = f"  {'run':<12}{'method':<14}" + "".join(f"{m:>12}" for m in _metric_cols(result))
    _info(header)
    for row in result["rows"]:
        line = f"  {row['run_id']:<12}{row['method']:<14}"
        line += "".join(f"{_fmt(row['test'].get(m)):>12}" for m in _metric_cols(result))
        _info(line)
    if result["best"]:
        _success(f"Best by test {result['metric']}: {result['best']}")


def _metric_cols(result) -> list[str]:
    return list(result["rows"][0]["test"].keys()) if result["rows"] else []


def _fmt(value) -> str:
    return f"{value:.5g}" if isinstance(value, (int, float)) else "-"


@app.command()
@_guard
def predict(
    run_id: str = typer.Argument(..., help="Run id whose model to use."),
    input: Path = typer.Option(..., "--input", "-i", exists=True, help="Input data file."),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Write predictions here (.csv, .parquet/.pq, .h5/.hdf5).",
    ),
    field: bool = typer.Option(
        False,
        "--field",
        help="Reconstruct and write the full field "
        "(default: compact modal coefficients for reduced runs).",
    ),
) -> None:
    """Predict with a trained model (PRD 4.2, 4.7)."""
    run = _project().run(run_id)
    frame = services.predict_from_file(run, input, field=field)
    if output is not None:
        _save_frame(frame, output)
        _success(f"Wrote {len(frame)} predictions to {output}")
    else:
        _info(frame.to_string(index=False))


@app.command()
@_guard
def export(
    run_id: str = typer.Argument(..., help="Run id to export."),
    format: str = typer.Option("onnx", "--format", "-f", help="Export format: onnx or pickle."),
    output: Path = typer.Option(None, "--output", "-o", help="Destination path."),
) -> None:
    """Export a run's model (PRD 4.2)."""
    run = _project().run(run_id)
    path = services.export_model(run, format, output)
    _success(f"Exported {run_id} model ({format}) to {path}")


@app.command()
@_guard
def report(
    run_id: str = typer.Argument(..., help="Run id to report on."),
    format: str = typer.Option("markdown", "--format", "-f", help="Report format."),
    output: Path = typer.Option(None, "--output", "-o", help="Destination path."),
) -> None:
    """Generate a report for a run (PRD 4.2)."""
    project = _project()
    run = project.run(run_id)
    path, _ = services.generate_report(project, run, format, output)
    _success(f"Wrote {format} report to {path}")


@app.command("list")
@_guard
def list_runs() -> None:
    """List all runs in the project (PRD 4.2)."""
    project = _project()
    runs = project.runs()
    if not runs:
        _info("No runs yet. Train one with 'surroforge train --config <yaml>'.")
        return
    _info(f"  {'run':<12}{'method':<14}{'created':<22}{'test rmse':>12}")
    for run in runs:
        rmse = run.metrics["test"].get("rmse")
        _info(f"  {run.run_id:<12}{run.method:<14}{run.created_at:<22}{_fmt(rmse):>12}")


@app.command()
@_guard
def reproduce(run_id: str = typer.Argument(..., help="Run id to reproduce.")) -> None:
    """Re-run a stored run and verify reproducibility (PRD 4.2, 4.5)."""
    project = _project()
    result = services.reproduce(project, run_id, echo=_echo_cb())
    new_id = result.reproduction.run_id
    # A reproduced field run whose basis bytes re-gauged is still reproduced — the basis
    # hash is diagnostic, never a gate (Story R2.3). Annotate, do not fail.
    regauged = " (basis re-gauged; field reproduced)" if result.basis_regauged else ""
    if result.verdict == "reproduced":
        _success(f"Reproduced {run_id} as {new_id} — bit-identical model.{regauged}")
    elif result.verdict == "reproduced_within_tolerance":
        if result.field_relative_diff is not None:
            measure = f"field-energy diff {result.field_relative_diff:.2e}"
        else:
            measure = f"max diff {result.max_abs_diff:.2e}"
        _success(
            f"Reproduced {run_id} as {new_id} — numerically identical ({measure}).{regauged}"
        )
    else:
        # data_changed or diverged → emit the verdict as a hard failure (exit 1).
        services.assert_reproducible(result)


@app.command()
@_guard
def info(run_id: str = typer.Argument(..., help="Run id to inspect.")) -> None:
    """Show detailed information about a run (PRD 4.2)."""
    run = _project().run(run_id)
    lineage = run.lineage
    _info(f"Run {run.run_id}")
    for key in (
        "experiment", "method", "created_at", "data_source",
        "config_hash", "data_hash", "seed", "prediction_hash",
        "python_version", "surroforge_version", "parent_run",
    ):
        _info(f"  {key:<18}: {lineage.get(key, '-')}")
    _info(f"  {'onnx_artifact':<18}: {'yes' if run.has_onnx() else 'no'}")
    _info(f"  {'dependencies':<18}: {', '.join(lineage.get('dependencies', []))}")
    _info("")
    _metrics_summary(run)


@rom_app.command("inspect")
@_guard
def rom_inspect(run_id: str = typer.Argument(..., help="Run id to inspect.")) -> None:
    """Show the energy spectrum, rank, and probe coverage for a reduced-field run (PRD 4.8 #30)."""
    run = _project().run(run_id)
    result = services.inspect_rom_run(run)
    if not result.is_reduced:
        _info(f"Run {run_id} is not a reduced-field run.")
        return
    _info(f"ROM inspection: {run_id}")
    _info("")
    _info(f"  method         : {run.method}")
    _info(f"  basis          : {result.basis}")
    _info(f"  rank           : {result.rank} / {result.total_modes} modes retained")
    _info(f"  energy_captured: {result.energy_captured * 100:.2f}%")
    _info(f"  probe_coverage : {result.probe_coverage_energy * 100:.2f}%")
    _info("")
    _info(f"  energy spectrum ({result.total_modes} modes total):")
    _info(f"  {'mode':>5}  {'energy_frac':>12}  {'cumulative':>12}  status")
    for i, (e, c) in enumerate(
        zip(result.per_mode_energy, result.cumulative_energy, strict=True)
    ):
        status = "[retained]" if i < result.rank else ""
        _info(f"  {i:>5}  {e * 100:>11.2f}%  {c * 100:>11.2f}%  {status}")


@app.command()
def version() -> None:
    """Show the SurroForge version."""
    _info(f"surroforge {__version__}")


def _render_warnings() -> None:
    """Render SurroForge warnings as sober ``Warning:`` lines.

    Without this the core's ``warnings.warn`` reaches the terminal in raw Python
    form (``.../run.py:NN: ProvenanceWarning: ...`` plus the echoed source line),
    which leaks internals and clashes with the CLI's output tone (NFR9). We only
    reformat our own warning types (``ProvenanceWarning`` security signal,
    ``FieldCoverageWarning`` reproduce-coverage advisory); every other warning falls
    through to Python's default handler untouched.
    """
    default_showwarning = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, (ProvenanceWarning, FieldCoverageWarning)):
            typer.secho(f"Warning: {message}", fg=typer.colors.YELLOW, err=True)
        else:
            default_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = show


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


def main() -> None:
    """Console-script entry point with friendly error rendering."""
    _render_warnings()
    try:
        app()
    except SurroForgeError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
