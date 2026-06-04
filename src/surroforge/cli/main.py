"""SurroForge command-line interface (PRD 4.2).

Commands mirror the Python API one-to-one. Domain errors (:class:`SurroForgeError`)
are caught by the :func:`_guard` decorator on each command and rendered as a clean
``Error: ...`` line instead of a traceback.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from pathlib import Path

import typer

from surroforge import __version__, services
from surroforge.core.config import load_config
from surroforge.core.project import Project
from surroforge.core.run import ProvenanceWarning, Run
from surroforge.engine import run_training
from surroforge.errors import DataError, SurroForgeError

app = typer.Typer(
    help="SurroForge — surrogate models for FEM/CFD simulation data.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Register and inspect datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")


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
    else:
        raise DataError(f"Cannot write predictions to '{suffix}'. Use .csv or .parquet.")


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
) -> None:
    """Train a surrogate model from a YAML configuration (PRD 4.2)."""
    project = _project()
    cfg = load_config(config)
    _info(f"Training '{cfg.experiment}' (method: {cfg.model.method}) ...")
    run = run_training(project, cfg, echo=_echo_cb())
    _success(f"Run {run.run_id} complete.")
    _metrics_summary(run)


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
        None, "--output", "-o", help="Write predictions here (.csv/.parquet)."
    ),
) -> None:
    """Predict with a trained model (PRD 4.2)."""
    run = _project().run(run_id)
    frame = services.predict_from_file(run, input)
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
    if result.verdict == "reproduced":
        _success(f"Reproduced {run_id} as {new_id} — bit-identical model.")
    elif result.verdict == "reproduced_within_tolerance":
        _success(
            f"Reproduced {run_id} as {new_id} — numerically identical "
            f"(max diff {result.max_abs_diff:.2e})."
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


@app.command()
def version() -> None:
    """Show the SurroForge version."""
    _info(f"surroforge {__version__}")


def _render_warnings() -> None:
    """Render SurroForge security warnings as sober ``Warning:`` lines.

    Without this the core's ``warnings.warn`` reaches the terminal in raw Python
    form (``.../run.py:NN: ProvenanceWarning: ...`` plus the echoed source line),
    which leaks internals and clashes with the CLI's output tone (NFR9). We only
    reformat our own ``ProvenanceWarning``; every other warning falls through to
    Python's default handler untouched.
    """
    default_showwarning = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, ProvenanceWarning):
            typer.secho(f"Warning: {message}", fg=typer.colors.YELLOW, err=True)
        else:
            default_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = show


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
