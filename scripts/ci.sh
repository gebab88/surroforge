#!/usr/bin/env bash
#
# Local CI for SurroForge.
#
# GitHub Actions was removed (2026-06); this script is the single source of truth
# for the project's quality gates and runs them on the developer's machine. It
# mirrors the jobs that used to live in .github/workflows/ci.yml as closely as one
# machine allows. The only thing it cannot reproduce is the cross-OS / multi-Python
# *matrix*: it runs against whatever interpreter is active (or ./.venv if present).
#
#   Usage:  scripts/ci.sh
#   Setup:  pip install -e ".[all,dev,docs]"   # one-time; brings every gate's tooling
#
# Required gates (ruff, mypy, pyright, import-linter, pytest) fail the run if their
# tool is missing. The release-time gates (docs, profile, licenses) are skipped with
# a hint when their optional tooling is absent, so the everyday lint+type+test loop
# still runs from a minimal install. Every gate runs; a summary is printed at the end.

set -uo pipefail  # deliberately NOT -e: run all gates, then report, don't fail-fast.

cd "$(dirname "$0")/.."

# Work without activating the venv: put a local ./.venv at the front of PATH if present.
if [[ -x ".venv/bin/python" ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
results=()
fail=0
skipped=0

have() { command -v "$1" >/dev/null 2>&1; }

run() {  # run <name> <cmd...>
  local name="$1"
  shift
  printf '\n%s━━━ %s ━━━%s\n' "$bold" "$name" "$reset"
  if "$@"; then
    results+=("PASS  $name")
  else
    results+=("FAIL  $name")
    fail=1
  fi
}

require() {  # require <name> <tool> <cmd...>
  local name="$1" tool="$2"
  shift 2
  if have "$tool"; then
    run "$name" "$@"
  else
    printf '\n%s━━━ %s ━━━%s\n' "$bold" "$name" "$reset"
    echo "MISSING required tool '$tool' — run: pip install -e \".[all,dev,docs]\""
    results+=("FAIL  $name (tool '$tool' missing)")
    fail=1
  fi
}

optional() {  # optional <name> <tool> <hint> <cmd...>
  local name="$1" tool="$2" hint="$3"
  shift 3
  if have "$tool"; then
    run "$name" "$@"
  else
    printf '\n%s━━━ %s ━━━%s\n' "$bold" "$name" "$reset"
    echo "SKIPPED — $hint"
    results+=("SKIP  $name")
    skipped=1
  fi
}

# Pinned BLAS/OpenMP threads for the reproducibility + profile gates, so float
# reductions don't reorder run-to-run (Spike 3). Scoped to those gates via `env`.
PIN=(env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)

# 1. lint + types + import boundaries -----------------------------------------
require "ruff"          ruff         ruff check src tests
require "mypy"          mypy         mypy src
require "pyright"       pyright      pyright
require "import-linter" lint-imports lint-imports

# 2. tests --------------------------------------------------------------------
require "pytest"        pytest       pytest -q

# 3. reproducibility (pinned threads) -----------------------------------------
require "reproducibility" pytest "${PIN[@]}" pytest -q \
  tests/test_reproduce.py tests/test_canonicalize.py tests/test_reproduce_field.py \
  tests/test_reproduce_scalar_identity.py tests/test_probe_coverage.py

# 4. docs build (mkdocs --strict) — release-time gate -------------------------
optional "docs" mkdocs "mkdocs not found — pip install -e '.[docs]'" \
  mkdocs build --strict

# 5. profiling (SurroForge code stays a small fraction of CPU time) -----------
require "profile" python "${PIN[@]}" python scripts/profile_training.py

# 6. license hygiene (no GPL/LGPL in Core — NFR8) — release-time gate ---------
optional "licenses" pip-licenses "pip-licenses not found — pip install pip-licenses (in .[dev])" \
  bash -c 'pip-licenses --format=json | python scripts/check_licenses.py'

# Summary ---------------------------------------------------------------------
printf '\n%s━━━ summary ━━━%s\n' "$bold" "$reset"
for r in "${results[@]}"; do echo "  $r"; done
echo
if [[ $fail -ne 0 ]]; then
  echo "${bold}CI FAILED${reset} — see the gate(s) marked FAIL above."
  exit 1
elif [[ $skipped -ne 0 ]]; then
  echo "${bold}CI passed${reset}, but some release-time gates were skipped (see SKIP above)."
  echo "Install full tooling for complete parity: pip install -e \".[all,dev,docs]\""
  exit 0
else
  echo "${bold}CI PASSED${reset} — all gates green."
  exit 0
fi
