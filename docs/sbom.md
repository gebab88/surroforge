# Software Bill of Materials (SBOM)

SurroForge ships a machine-readable inventory of every third-party component in
its dependency environment, together with a license classification. This lets
downstream users run vulnerability scans, check license compatibility, and
satisfy supply-chain requirements without inspecting the source tree by hand.

!!! tip "Interactive overview"
    A filterable dashboard of all packages — license traffic-light, copyleft
    analysis, and a searchable inventory — is available at
    [**sbom.html**](sbom.html).

## Formats and generation

The SBOM is produced with [Syft](https://github.com/anchore/syft) and emitted in
two industry-standard formats:

| File | Format | Spec |
|------|--------|------|
| `cdx.json` | CycloneDX | 1.7 |
| `spdx.json` | SPDX | 2.3 |

Both are generated from the same scan (Syft 1.48.0). The Windows launcher stubs
that Syft otherwise catalogs from inside `pip` and `setuptools`
(`t64.exe`, `cli-64.exe`, …) are excluded, because they are packaging artifacts
rather than dependencies:

```bash
syft dir:. --exclude './.venv/**/*.exe' -o cyclonedx-json=cdx.json
syft dir:. --exclude './.venv/**/*.exe' -o spdx-json=spdx.json
```

!!! note "Generated, not committed"
    `cdx.json` and `spdx.json` are build outputs and are git-ignored. Regenerate
    them on demand with the commands above.

## Scope

The scan is an **environment inventory** of the project's virtual environment,
not a pure runtime dependency list.

| Metric | Value |
|--------|------:|
| Components (total) | 453 |
| Libraries (PyPI) | 130 |
| Files | 323 |
| Ecosystem | PyPI · 100% |

Because it scans the whole `.venv`, the inventory also contains build, test,
publish, and docs tooling (`pytest`, `ruff`, `mypy`, `twine`, `mkdocs-material`,
`pip`, `setuptools`, `wheel`). Five packages appear twice at different versions
(`jaraco-context`, `jaraco-functools`, `more-itertools`, `packaging`,
`platformdirs`) — these come from the vendored copies inside `setuptools`/`pip`.

!!! note "Producing a lean release SBOM"
    For a distribution SBOM that lists only what a user installs with the
    `surroforge` wheel, generate from the resolved dependency graph instead of
    the environment:

    ```bash
    uv pip compile pyproject.toml -o requirements.txt
    syft file:requirements.txt -o cyclonedx-json=cdx.json
    ```

## License overview

Every library is classified into a three-level traffic light. The declared
runtime dependencies (Core: `numpy`, `scipy`, `pandas`, `scikit-learn`,
`pyyaml`, `pydantic`, `typer`, `jinja2`; plus the optional extras) are **all
permissive**.

| Status | Count | Meaning |
|--------|------:|---------|
| **Permissive** | 112 | MIT · BSD · Apache-2.0 · ISC · PSF — unrestricted |
| **Review** | 14 | Copyleft, ambiguous, or license not detected |
| **Clarify** | 4 | Proprietary or unknown |

See the [dashboard](sbom.html) for the full per-package breakdown.

## Copyleft

Five packages carry copyleft licenses. Two of them
(`pathspec`, `docutils`) were **missed by Syft** — the SBOM recorded no license
— and were confirmed against the packages' `METADATA` in the environment.

| Package | Version | License | Copyleft | Role |
|---------|---------|---------|----------|------|
| `certifi` | 2026.5.20 | MPL-2.0 | weak (file-level) | publish tooling |
| `orjson` | 3.11.9 | MPL-2.0 (+ Apache/MIT) | weak (file-level) | transitive |
| `pathspec` | 1.1.1 | MPL-2.0 *(Syft-corrected)* | weak (file-level) | build/docs tooling |
| `autocommand` | 2.2.2 | LGPL-3.0 | weak (library-level) | setuptools vendoring |
| `docutils` | 0.23 | Public Domain + BSD + **GPL** (mixed) | partly strong | publish tooling |

!!! success "Low practical risk"
    - **No pure GPL/AGPL package.** The only GPL appearance is `docutils`, which
      is multi-licensed (mostly Public Domain/BSD; GPL applies only to a few
      helper files).
    - **MPL-2.0 and LGPL are file-/library-level copyleft.** Used as unmodified
      dependencies, they impose no obligation to open-source SurroForge; the
      copyleft terms trigger only if you modify and redistribute *those specific
      files*.
    - **None of the copyleft packages is a runtime dependency.** They enter only
      via dev/build/publish/docs tooling and are **not part of the shipped
      `surroforge` wheel**.

## Items to address for a public release SBOM

!!! warning "First-party proprietary packages"
    `surroforge-enterprise` (0.1.0) and `surroforge-professional` (0.2.0) appear
    as `Proprietary` because they were installed in the local environment.
    They are first-party commercial tiers — for a public distribution SBOM they
    should be excluded or explicitly labeled as first-party, so they do not read
    as a foreign proprietary dependency.

!!! warning "Local test artifact"
    `my-test-package` (1.0) is a local test artifact and should not appear in a
    release SBOM. `mergedeep` reports `UNKNOWN` but is in fact MIT — Syft simply
    did not resolve it.

!!! note "License-detection quirks"
    - **Detection failed (SHA-256):** `matplotlib`, `pandas`, `kiwisolver` — Syft
      emitted the hash of the `LICENSE` file instead of an SPDX ID. The real
      licenses are PSF/BSD-style (matplotlib) and BSD-3-Clause (pandas,
      kiwisolver).
    - **No license detected:** `annotated-types`, `colorama`, `id`, `jinja2`,
      `markdown-it-py` — all permissive in reality (MIT/BSD/Apache); verify
      before a release.
    - **Inconsistent naming:** the same license appears under several spellings
      (`Apache 2.0` / `Apache Software License` / `Apache License v2.0`).
      Normalize before feeding the SBOM to compliance tooling.

## Regenerating

```bash
# both formats, launcher stubs excluded
syft dir:. --exclude './.venv/**/*.exe' -o cyclonedx-json=cdx.json
syft dir:. --exclude './.venv/**/*.exe' -o spdx-json=spdx.json
```

The local CI license gate (`scripts/ci.sh`, backed by `pip-licenses`) is the
authoritative check for the Core edition's permissive-only policy.
