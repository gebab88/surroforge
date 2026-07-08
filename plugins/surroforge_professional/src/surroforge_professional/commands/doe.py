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


@doe_app.callback()
def _doe_callback() -> None:
    """Generate design-of-experiments sample tables (Professional)."""


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
