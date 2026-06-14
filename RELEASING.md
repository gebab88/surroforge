# Releasing SurroForge

This is the checklist to cut a release. SurroForge follows
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`. The single source of truth for
the version is `version` in `pyproject.toml`.

> **Status:** the release **artifacts** are prepared in the repo (version, changelog, metadata,
> typing marker). Cutting the tag and publishing to PyPI are deliberate, outward-facing steps and
> are left to a maintainer — follow the steps below.

## 1. Pre-flight (on a clean `main`)

```bash
git switch main && git pull && git status   # clean working tree, up to date

pip install -e ".[all,dev]"

# Gates — all must pass
ruff check src tests
mypy src
lint-imports
python -m pytest -q
mkdocs build --strict
```

CI must also be green on the full matrix (Ubuntu/macOS/Windows × Python 3.10–3.12) plus the
reproducibility and license-hygiene jobs. Do not release on a red matrix.

## 2. Version & changelog

- [ ] Confirm/bump `version` in `pyproject.toml` (e.g. `0.1.0`). Bump `Development Status`
      classifier if the maturity changed (`4 - Beta` → `5 - Production/Stable`).
- [ ] In `CHANGELOG.md`: move the `[Unreleased]` items into a new `## [X.Y.Z] — YYYY-MM-DD`
      section, reset `[Unreleased]` to "_Nothing yet._", and update the compare/tag links at the
      bottom.
- [ ] Commit: `git commit -m "Release vX.Y.Z"`.

## 3. Build & verify the distribution

```bash
rm -rf dist/
python -m build                 # builds the sdist (.tar.gz) and wheel (.whl)
twine check dist/*              # validates the rendered metadata/README

# Sanity-check the wheel ships what it should
python -m zipfile -l dist/*.whl | grep -E "py\.typed|METADATA"
```

Verify in `dist/*.whl` METADATA: correct version, the `rom`/`all`/`smt`/`onnx`/`formats` extras,
the Python 3.10–3.12 classifiers, `Typing :: Typed`, and the Apache-2.0 license.

Optional install smoke-test in a throwaway env:

```bash
python -m venv /tmp/sf && /tmp/sf/bin/pip install "dist/"*.whl
/tmp/sf/bin/surroforge version && /tmp/sf/bin/surroforge --help
```

## 4. Tag

```bash
git tag -a vX.Y.Z -m "SurroForge vX.Y.Z"
git push origin vX.Y.Z
```

(Tags are `v`-prefixed, e.g. `v0.1.0`. The `CHANGELOG.md` links assume this convention.)

## 5. Publish to PyPI

**Recommended: PyPI Trusted Publishing (OIDC, no API token in the repo).**

1. On PyPI, register the `surroforge` project and add a *trusted publisher* for this GitHub repo
   and a release workflow (`.github/workflows/release.yml`, environment e.g. `pypi`).
2. Add the release workflow (not included in this prep — opt-in): on `push: tags: ['v*']`, build
   and `pypa/gh-action-pypi-publish`. The job needs `permissions: id-token: write`.
3. Pushing the tag then publishes automatically.

**Manual fallback (API token):**

```bash
twine upload dist/*            # uses ~/.pypirc or TWINE_USERNAME=__token__ / TWINE_PASSWORD=pypi-…
```

Test first against TestPyPI if unsure: `twine upload --repository testpypi dist/*`.

## 6. GitHub Release

- [ ] Create a GitHub Release for the `vX.Y.Z` tag; paste the `CHANGELOG.md` section as the notes.
- [ ] Attach the `dist/*` artifacts (optional — PyPI is the canonical source).

## 7. Post-release

- [ ] `pip install surroforge==X.Y.Z` from PyPI in a clean env to confirm it resolves and runs.
- [ ] Open a new `[Unreleased]` cycle in `CHANGELOG.md` (already reset in step 2).

## Notes

- **No GPL/LGPL in Core (NFR8).** The `licenses` CI job enforces this; keep new dependencies
  permissive (Apache/MIT/BSD).
- **`surroforge[rom]` is empty in v1** by design (POD/SVD runs on numpy/scipy, already Core deps).
  It is the documented install target and the Phase-2 home for mesh readers — see `pyproject.toml`.
- Build/release tooling (`build`, `twine`) ships in the `dev` extra.
</content>
