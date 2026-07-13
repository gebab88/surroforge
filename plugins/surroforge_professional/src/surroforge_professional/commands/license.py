"""`surroforge license` — inspect and install the local license (M3).

Deliberately NOT license-gated: it must work before a valid license exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from surroforge_professional.license import (
    LICENSE_ENV,
    LICENSE_ERROR,
    LicenseStatus,
    license_file,
    license_status,
    validate_token,
)

license_app = typer.Typer(
    help="Inspect and install the SurroForge license.",
    no_args_is_help=True,
)


@license_app.callback()
def _license_callback() -> None:
    """Inspect and install the SurroForge license."""


def _verdict(status: LicenseStatus) -> str:
    if status.error is None:
        return "valid"
    if status.error.startswith("expired "):
        return "expired"
    return f"invalid ({status.error})"


@license_app.command("status")
def status() -> None:
    """Show the installed license and its validity (exit 0 only when valid)."""
    st = license_status()
    if st.error == "missing":
        typer.echo("No license installed.")
        typer.echo(LICENSE_ERROR)
        raise typer.Exit(1)
    typer.echo(f"Source:  {st.source}")
    typer.echo(f"Holder:  {st.holder or '—'}")
    typer.echo(f"Tier:    {st.tier or '—'}")
    typer.echo(f"ID:      {st.license_id or '—'}")
    typer.echo(f"Issued:  {st.issued or '—'}")
    typer.echo(f"Expires: {st.expires.isoformat() if st.expires else 'never'}")
    typer.echo(f"Status:  {_verdict(st)}")
    if st.error is not None:
        raise typer.Exit(1)


@license_app.command("install")
def install(
    source: str = typer.Argument(
        ..., help="License token, or path to a file containing it."
    ),
) -> None:
    """Validate a license token and install it to ~/.surroforge/professional.lic."""
    candidate = Path(source).expanduser()
    try:
        is_file = candidate.is_file()
    except OSError:  # e.g. ENAMETOOLONG: a long token passed directly, not a path
        is_file = False
    if is_file:
        try:
            token = candidate.read_text().strip()
        except (OSError, UnicodeDecodeError) as exc:
            typer.echo(f"Error: license not installed — cannot read {candidate}: {exc}.")
            raise typer.Exit(1) from exc
    else:
        token = source.strip()
    st = validate_token(token)
    if st.error is not None:
        typer.echo(f"Error: license not installed — {st.error}.")
        raise typer.Exit(1)
    target = license_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(token + "\n")
    expires = st.expires.isoformat() if st.expires else "never"
    typer.echo(f"Installed {st.tier} license for {st.holder} (expires: {expires}) -> {target}")
    if os.environ.get(LICENSE_ENV, "").strip():
        typer.echo(f"Note: {LICENSE_ENV} is set and takes precedence over the installed file.")
