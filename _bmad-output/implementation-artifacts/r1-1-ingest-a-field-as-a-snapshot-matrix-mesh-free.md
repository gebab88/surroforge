# Story R1.1: Ingest a field as a snapshot matrix (mesh-free)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a methods engineer,
I want to register a wide `(n_samples × n_dof)` table and reference its field columns as **one output field via a glob/prefix** (e.g. `data.outputs: [temp_*]`),
so that I can build a field surrogate from a snapshot matrix **without exporting a mesh** — naming a thousand DOF columns one by one is not viable.

## Acceptance Criteria

1. **Given** a registered CSV/Parquet/HDF5 dataset whose output columns are an ordered field group (e.g. `temp_0001 … temp_NNNN`) **when** a config references them with a glob pattern in `data.outputs` (e.g. `[temp_*]`) **then** the pattern **expands deterministically** into the concrete `n_dof` field-column vector, in **dataset column order** (the snapshot matrix's native DOF order), and training proceeds exactly as if the columns had been listed explicitly. The dataset is hashed and recorded as today (full-frame hash at registration; unaffected by how a config later references it).
2. **Given** the expansion path **then** **no coordinates or connectivity are read or required** — the field is "columns that move together." Columns present in the dataset but matched by neither `inputs` nor `outputs` (e.g. `x`, `y`, `z` coordinate columns) are simply ignored.
3. **Given** a misaligned field group **when** a pattern is resolved **then** a clear `DataError` **naming the offending columns** is raised for: a pattern that matches **zero** columns; a column matched by **two patterns** or by a pattern **and** also listed as an input (duplicate/overlapping DOF). The existing `select_columns` guards (column exists, numeric, no NaN, no input/output overlap) continue to hold on the **expanded** column set.
4. **Given** a config using literal output column names (no glob metacharacters) **then** behaviour is **byte-for-byte unchanged** (full back-compat — every existing config and test still resolves identically).
5. **Given** a field run trained from a glob config **when** it is reproduced **then** the config snapshot stores the **pattern verbatim**, the pattern re-expands against the same dataset to the **same ordered columns**, and the run reproduces (the data hash already captures column identity + order, so a changed column set is caught as `data_changed`).

> **Scope boundary (do not exceed).**
> - **Globs apply to `data.outputs` only** (the field). `data.inputs` stay literal (they are a handful of named parameters). Do not add glob support to inputs in v1.
> - **Do NOT implement the `model.reduction` block or `rank: adaptive`** — that is R1.2. R1.1 is purely the data/ingestion + config-reference layer.
> - **Do NOT change `predict`/report field-aware naming or reconstruction** — that is R4.1. R1.1 must keep them *working* (a reduced-field predict already returns `(n, n_dof)` and names columns `output_0…` generically when the output list isn't 1:1 — that graceful path is acceptable here; do not break it, do not "fix" the naming).
> - **Do NOT read or require coordinates/connectivity** (mesh-free is the invariant).
> - **No new dependency** — `fnmatch` is stdlib. Keep `data/loader.py` a leaf (it imports only `errors`).
> - **Do NOT change the dataset registration/hashing** (`add_dataset`, `hash_dataframe`) — globs are a *config-reference* concern, resolved at training time against the already-registered frame.

## Tasks / Subtasks

- [x] **Task 1 — Field-pattern expansion helper in `data/loader.py`** (AC: 1, 2, 3)
  - [x] Add `expand_field_columns(frame: pd.DataFrame, outputs: list[str]) -> list[str]`. For each entry in `outputs`: if it contains a glob metacharacter (`*`, `?`, or `[`), treat it as a **pattern**; otherwise it is a **literal** column name kept as-is. Use `fnmatch.fnmatchcase` (case-sensitive, deterministic) to match patterns against `frame.columns` — **not** `fnmatch.fnmatch`, which lowercases via `os.path.normcase` on case-insensitive platforms and would silently mismatch e.g. `Temp_*`.
  - [x] **Order: dataset column order.** A pattern expands to its matching columns in the order they appear in `frame.columns` (the snapshot matrix's native DOF order — the user authors DOF order via their file layout). Iterate `frame.columns` once and, for each column, include it if it is a literal in `outputs` or matches any pattern — this yields a single deterministic pass that preserves file order and naturally interleaves literals + patterns by position. (Do **not** sort by a parsed numeric suffix — that would assume `<stem><int>` naming and silently reorder non-numeric fields like `stress_xx/_yy/_zz`.)
  - [x] **Guards (each a `DataError` naming the offenders):** a pattern that matches **zero** columns → `DataError("output pattern 'temp_*' matched no columns. Available: …")`; the same column reached by more than one entry (two patterns, or a pattern + a literal) → `DataError` naming the duplicated columns. Return the de-duplicated, ordered concrete list.
  - [x] Docstring: state that this is mesh-free (only column names are read; no coordinates/connectivity), the order contract (dataset column order), and the glob-metacharacter rule. Note the edge: a literal column name containing `*?[` cannot be expressed (treat as out of scope — scientific field columns do not use glob metacharacters).

- [x] **Task 2 — Wire expansion into `select_columns` (the single choke point)** (AC: 1, 3, 4)
  - [x] At the top of `select_columns`, expand `outputs` via `expand_field_columns(frame, outputs)` to a concrete ordered list **before** the existing checks. Keep `inputs` literal. Then run the existing guards on the expanded set: missing-columns, input/output overlap (`set(inputs) & set(expanded_outputs)`), numeric, NaN. This makes the **engine unchanged** — `run_training` already calls `select_columns` and reads `y.shape[1]` for `n_outputs`, so the resolved `n_dof` count flows through automatically.
  - [x] A **literal** entry is kept as-is (no match-test in `expand_field_columns`); if that literal does not exist in the frame, the **existing missing-columns guard** (loader.py:71) catches it after expansion. Do **not** add a redundant existence check for literals inside the helper — only patterns get the zero-match guard.
  - [x] Verify literal-only `outputs` (no metacharacters) produce an **identical** result to today (back-compat): no pattern branch taken, same order, same errors.

- [x] **Task 3 — Confirm the end-to-end + reproduce paths (no engine change expected)** (AC: 1, 5)
  - [x] Confirm `engine.run_training` needs **no change**: it passes `config.data.outputs` (the patterns) to `select_columns`, which now resolves them; `n_outputs`, the holdout fingerprint, and metrics all derive from the resolved `y`. The config snapshot (`config.model_dump`) stores the patterns verbatim, so `reproduce()` re-expands them deterministically against the same dataset.
  - [x] Confirm `predict_with_run` and the markdown reporter do not crash on a glob config: predict returns `(n, n_dof)` and falls to generic `output_i` names when `len(config.data.outputs) != n_dof` (acceptable for R1.1; field-aware naming is R4.1). The reporter shows the pattern string in "outputs" and the real count from `metrics["n_outputs"]`. Do **not** modify these — just verify they degrade gracefully.

- [x] **Task 4 — Tests: new `tests/test_field_ingest.py`** (AC: 1–5)
  - [x] `test_glob_expands_in_dataset_column_order`: a frame with `a, b, u0…u5` → `expand_field_columns(frame, ["u*"])` returns `["u0", …, "u5"]` in column order; mixing a literal + pattern preserves position.
  - [x] `test_literal_outputs_unchanged`: `expand_field_columns(frame, ["u0", "u3"])` returns exactly `["u0", "u3"]` (no metacharacters → identity); a literal list through `select_columns` is byte-identical to pre-R1.1 (mirror an existing `test_data.py` case).
  - [x] `test_zero_match_pattern_raises_named`: `["temp_*"]` on a frame with no `temp_` columns → `DataError` whose message names the pattern (and lists available columns).
  - [x] `test_overlapping_patterns_raise_named`: `["u*", "u3"]` (or two overlapping patterns) → `DataError` naming the duplicated column(s). Also: a pattern matching a column also in `inputs` → the existing input/output overlap `DataError`.
  - [x] `test_coordinates_ignored_mesh_free`: a frame with `a, b, x, y, z, temp_0…temp_5` referenced as inputs `[a, b]`, outputs `[temp_*]` → `select_columns` returns `y` of width 6, the coordinate columns untouched (no error, not read into y).
  - [x] `test_numeric_and_nan_guards_apply_to_expanded_set`: a `temp_*` group with a non-numeric or NaN-containing column → the existing numeric/NaN `DataError` fires on the expanded set.
  - [x] **End-to-end** `test_reduced_field_trains_from_glob_outputs`: register a field dataset, train a `reduced-field` run with `outputs: ["u*"]` (mirror `tests/test_reduced_field.py`'s fixture/config but with a glob) → run succeeds, `run.metrics["n_outputs"] == n_dof`. (This is the AC1 integration proof.)
  - [x] **Reproduce determinism** `test_glob_run_reproduces`: a glob-outputs `reduced-field` run reproduces — `reproduce()` verdict is a reproduce and the re-expanded field is identical (deterministic order). (Use polynomial inner; field-energy verdict from R2.3 applies.)

- [x] **Task 5 — Gates green**
  - [x] `ruff check src tests`, `mypy src` (a `list[str]` helper, fully typed), `lint-imports` (the 3 enforced contracts stay green; the `fnmatch` import is stdlib ⇒ no new internal edge). Full `pytest` (0 failed; expected smt/onnx/BLAS skips locally). Run `pytest -q tests/test_field_ingest.py` and the data/config/reduced-field set.

### Review Findings

_Code review 2026-06-10 (Blind Hunter · Edge Case Hunter · Acceptance Auditor). Auditor verdict: **ACCEPT** — all 5 ACs satisfied, every scope boundary honored, tests/gates independently re-run (single choke point; engine/config/services/reporter/trainer unchanged; `fnmatchcase` case-sensitivity confirmed; diff = `data/loader.py` + test only). Blind + Edge converged on one real correctness bug (fixed) plus latent/out-of-scope items._

- [x] [Review][Patch] A literal column name with balanced brackets (`disp[0]`, `stress[xx]` — a near-universal convention for indexed tensor/vector DOF) was misclassified as a glob and raised a spurious "matched no columns" `DataError` [src/surroforge/data/loader.py] — **Fixed 2026-06-10:** an entry that exactly equals an existing column name is now always a literal (short-circuit before glob detection), so `fnmatch` never parses `[xx]` as a character class. New test `test_bracketed_index_column_is_literal_not_glob` pins it (bracketed names resolve as literals; a bare `disp*` glob still expands the group). The story's assumption that "scientific field columns do not use `[`" was wrong — they frequently do. (blind+edge, High; edge reproduced end-to-end.)
- [x] [Review][Defer] `predict_with_run` names a glob run's field columns generically (`output_0…`) because the stored pattern isn't re-expanded [src/surroforge/services.py:346-349] — deferred to **R4.1** (field-aware predict naming / reconstruction), which the R1.1 scope boundary explicitly carves out ("keep working; generic names acceptable"). The field still predicts correctly at `(n, n_dof)`; only the result column labels are generic. Tracked by epic Story R4.1.
- [x] [Review][Defer] `expand_field_columns` returns `str(col)` for pattern matches, so a frame with **non-string** column labels (e.g. a headerless read) would mis-resolve against the original-dtype `frame.columns` [src/surroforge/data/loader.py] — deferred: **latent**, no SurroForge reader produces non-string labels (`load_table` uses `pd.read_csv` *with* header / parquet / hdf → string columns; the schema is `{str(col): …}`; Edge could find no headerless path). Harden (return original labels) if a headerless/numeric-label reader is ever added. Recorded in `deferred-work.md`.

_Dismissed (4): empty-string outputs entry (Edge-confirmed safe — caught downstream by the missing-columns guard; `DataConfig` requires `min_length=1` on the list); duplicate-guard message wording for a plain literal-literal dup (cosmetic; `DataConfig._no_duplicates` catches raw dups earlier in the normal flow); a frame with duplicate column *labels* (pathological — such a dataset is a deeper problem than this guard); the Blind layer's "test assertions unverifiable" (refuted — Edge + Auditor both re-ran the suite, 13/13 pass, and confirmed every `match=` regex against the real `DataError` strings)._

## Dev Notes

### The central realization (read this first — it scopes the story)
**R1.1 is a small, leaf-level change at one choke point.** `select_columns` (data/loader.py:64) is the *only* place training turns `config.data.outputs` into a concrete `y` matrix, and it already has the frame. Expanding glob patterns there makes the field-as-column-group work **with zero engine/services/config-model changes** — `run_training` reads `y.shape[1]` for `n_outputs`, the fingerprint derives from `y`, and the config snapshot stores the patterns verbatim so `reproduce()` re-expands deterministically. Expect the production diff to be **`data/loader.py` only**, plus tests.

### Design decisions (pin these — they are the story's substance)
- **Order = dataset column order**, not a parsed numeric suffix. A snapshot matrix's column order *is* its DOF order; the user authors it. Numeric-suffix sorting would assume `<stem><int>` naming and silently reorder fields like `stress_xx/_yy/_zz` or `mode_re/mode_im`. File-order is deterministic, assumption-free, and matches "snapshot matrix" semantics. **(Worth a one-line note to Georg — see the question at the end.)**
- **Glob detection:** an entry is a pattern iff it contains `*`, `?`, or `[`. Plain names stay literal ⇒ full back-compat (AC4). Use `fnmatch.fnmatchcase` (case-sensitive, stdlib, no dep).
- **"Ragged/misaligned" (epic wording) → the well-formedness guards that actually apply to a wide table:** zero-match, duplicate/overlapping DOF, and the existing numeric/NaN/overlap checks — all naming offenders. Gap-detection in a numeric sequence is **not** done in v1 (file-order is the contract; there is no canonical "expected" index set without assuming numeric suffixes).
- **Reproduce determinism (AC5):** the snapshot stores `outputs: [temp_*]`; re-expansion against the same dataset gives the same ordered columns. The dataset hash (`hash_dataframe`, column-name+order-sensitive) already catches a changed column set as `data_changed` — so a silently reordered/renamed field cannot pass as "reproduced." Do not add a separate field-column record for v1 (n_outputs already captures the resolved count; recording the resolved column list is an `rom inspect` enhancement, R4.2).
- **Graceful degradation, not new behaviour, for predict/report:** a glob run's `predict` returns `(n, n_dof)` and names columns `output_0…` (`predict_with_run`, services.py:346-349) because `len(outputs)=1 ≠ n_dof`; the reporter shows the pattern string + the real `n_outputs` from metrics (markdown.py:90,92). Both are acceptable for R1.1 — field-aware naming/reconstruction is R4.1. Verify, don't change.

### Source-tree components to touch
- **UPDATE `src/surroforge/data/loader.py`** — add `expand_field_columns` (new) and call it at the top of `select_columns` (loader.py:64-87). Preserve the existing missing/overlap/numeric/NaN guards (run them on the expanded set). Keep the module a leaf (imports: `numpy`, `pandas`, `surroforge.errors` only).
- **NEW `tests/test_field_ingest.py`** — flat `tests/`, one file per concern.
- **NO change expected** to `engine.py`, `services.py`, `core/config.py`, `core/project.py`, `reporters/markdown.py`, `trainers/reduced_field.py`. If a genuine break surfaces (e.g. an unexpected predict/report crash), fix it minimally and document it; otherwise leave them.

### Current behaviour to preserve (read before editing)
- `select_columns(frame, inputs, outputs)` (loader.py:64-87): checks missing columns → input/output overlap → numeric → NaN, then returns `x (n, n_inputs)`, `y (n, n_outputs)`. The expansion must slot in **before** these checks and feed them the concrete columns. Literal lists must take an identity path.
- `DataConfig` (config.py:32-47): `outputs` is `list[str]`, `min_length=1`, `_no_duplicates` on the raw entries. Patterns are valid strings; the raw-duplicate check stays (it catches a literally-repeated pattern). The *expanded* duplicate check lives in the loader (it needs the frame). **No config-model change.**
- `hash_dataframe` (data/hashing.py:26): column-name + order + value sensitive — already the guarantee AC5 leans on. Unchanged.
- Reduced-field e2e pattern: `tests/test_reduced_field.py` registers a field dataset and trains with explicit `outputs: [f"f{j}" for j in range(6)]` and asserts `run.metrics["n_outputs"] == 6` — mirror this with a glob for the R1.1 e2e test.

### Import-linter / architecture constraints (hard)
- `data/loader.py` keeps importing only `surroforge.errors` (+ numpy/pandas) — a leaf **by convention** (not itself one of the named contracts). `fnmatch` is stdlib, so the import adds **no new internal edge**; the 3 enforced import-linter contracts stay green. Re-run `lint-imports` as a gate.

### Testing standards
- pytest, flat `tests/`, deterministic data (build small DataFrames inline or via `make_field_samples`-style helpers). **[MUST] error-path tests with teeth** (project-context): the zero-match and overlapping-DOF `DataError`s must be asserted via `pytest.raises(DataError, match=…)` naming the offenders — these are the guard's teeth. No BLAS concern (column selection is exact; the e2e/reproduce tests use the polynomial inner, byte-stable).

### Previous-story intelligence (R1.3, Epic R2)
- **R1.3** (done) shipped `ReducedFieldTrainer` as a pure plugin and an e2e test that uses an explicit output column list — it already treats `outputs` as the field column-group. R1.1 makes that group expressible as a glob; the trainer is **unchanged** (it receives the resolved `(n, n_dof)` `y`).
- **Epic R2** (done) made `reproduce()` field-primary and byte-identical on the scalar path; nothing in R1.1 touches the verdict. The glob run reproduces via the field-energy verdict (R2.3) + deterministic re-expansion (this story).
- **Process (Epic R2 retro):** commit this story at `done` before starting the next; keep the spike-first/validate/adversarial-review pipeline.

### Project Structure Notes
- A single leaf-module helper + one choke-point call + tests — the smallest possible surface for "field = a glob of columns," consistent with the spine theorem (#36): ingestion is a data-layer concern, invisible to the engine/trainer contract. The `(planned)` tags in `_bmad-output/project-context.md` for CI/conformance are stale (shipped); trust the source.

### References
- [Source: _bmad-output/planning-artifacts/epics-field-rom-v1.md#Story R1.1] — acceptance criteria (FR-R1, AR-R1, UX-DR7; #19, #53).
- [Source: _bmad-output/planning-artifacts/prd-field-rom-v1.md §1.5] — "Snapshot-Matrix-Ingestion (#19) — `(n_samples × n_dof)` aus Parquet/HDF5/CSV, mesh-frei (#53)."
- [Source: src/surroforge/data/loader.py#select_columns] — the choke point to extend (loader.py:64-87); existing missing/overlap/numeric/NaN guards to preserve.
- [Source: src/surroforge/core/config.py#DataConfig] — `outputs: list[str]` (config.py:32-47); no model change needed (patterns are valid strings).
- [Source: src/surroforge/data/hashing.py#hash_dataframe] — column-order-sensitive dataset hash backing AC5.
- [Source: src/surroforge/engine.py#run_training] — calls `select_columns` (engine.py:85) and records `n_outputs = y.shape[1]`; unchanged.
- [Source: src/surroforge/services.py#predict_with_run] — generic `output_i` naming when outputs aren't 1:1 (graceful for a glob; R4.1 owns field-aware naming).
- [Source: tests/test_reduced_field.py] — the field-dataset + reduced-field e2e pattern to mirror with a glob.
- [Source: tests/test_data.py] — existing `select_columns`/loader test patterns to extend.

## Dev Agent Record

### Agent Model Used

claude-opus-4-8[1m] (Opus 4.8, 1M context)

### Debug Log References

- Targeted: `pytest -q tests/test_field_ingest.py` → **13 passed** (12 + 1 code-review test for bracketed-index literals).
- Full suite: `pytest` → **241 passed, 2 skipped, 0 failed** (post-review; the 1 reported warning is the pre-existing foreign-warning passthrough test).
- Gates: `ruff check src tests` clean · `mypy src` clean (41 files) · `lint-imports` **3/3 kept** (the `import fnmatch` is stdlib ⇒ no new internal edge; `data/loader.py` stays a leaf).
- Diff scope as predicted: `src/surroforge/data/loader.py` (+58/-1) + new `tests/test_field_ingest.py`. **Zero engine/config/services/reporter/trainer change.**

### Completion Notes List

- **Single-choke-point design held: zero engine change.** Added `expand_field_columns(frame, outputs)` to `data/loader.py` and call it at the top of `select_columns`; `run_training` reads `y.shape[1]` for `n_outputs` unchanged, so the resolved `n_dof` flows through automatically.
- **Ordering:** literals keep the user's written order (back-compat — `["u3","u0"]` → `["u3","u0"]`); a pattern expands to its matches **in frame column order** (the snapshot matrix's native DOF order). Confirmed via `test_glob_expands_in_dataset_column_order` + `test_mixed_literal_and_pattern_interleaves`. No numeric-suffix sorting (avoids reordering non-numeric fields).
- **Glob detection** via `_GLOB_CHARS = "*?["`; matching via `fnmatch.fnmatchcase` (case-sensitive, deterministic — NOT `fnmatch.fnmatch`, which would case-fold).
- **Guards name offenders (AC3):** zero-match pattern → `DataError` naming the pattern + available columns; a column reached by >1 entry (two patterns, or pattern + literal) → `DataError` listing the duplicated DOF. The existing missing-columns / input-output-overlap / numeric / NaN guards now run on the **expanded** set (a non-existent literal is left in place and caught by the missing-columns guard — no redundant literal existence check added).
- **Mesh-free (AC2):** `test_coordinates_ignored_mesh_free` — `x/y/z` columns present but referenced by neither inputs nor outputs are never read into `y`.
- **Back-compat (AC4):** literal lists (no metacharacters) take the identity path; the full suite (incl. `test_reduced_field.py`'s explicit `[f"f{j}"…]` and all `test_data.py` literal cases) passes unchanged.
- **Reproduce determinism (AC5):** `test_glob_run_reproduces` — a `reduced-field` run with `outputs: ["f*"]` reproduces (field-energy verdict); the config snapshot stores the pattern verbatim and re-expansion against the same dataset is deterministic.
- **Scope held:** no `model.reduction` block / `rank: adaptive` (R1.2); no field-aware predict naming (R4.1 — verified `predict_with_run` falls to generic `output_i` names without crashing); no coordinate/mesh reading; globs on `outputs` only; no new dependency (`fnmatch` stdlib); no engine/config edit.

### File List

- `src/surroforge/data/loader.py` (modified — `expand_field_columns` helper + `_GLOB_CHARS`; `select_columns` resolves output patterns first)
- `tests/test_field_ingest.py` (new — 13 tests: ordering/back-compat, named-offender guards, mesh-free, numeric/overlap/missing guards on the expanded set, glob e2e train, glob reproduce determinism, + bracketed-index-literal added in code review)

### Change Log

- 2026-06-10: Story created (create-story) — field-as-glob ingestion (`data.outputs: [temp_*]`), expanded deterministically in dataset column order at the `select_columns` choke point. Mesh-free; back-compat for literal lists; `DataError` naming offenders for zero-match / overlapping DOF; reproduce re-expands deterministically. Scoped to `data/loader.py` + tests (zero engine/config change); `model.reduction` block (R1.2) and field-aware predict naming (R4.1) explicitly deferred. Status → ready-for-dev.
- 2026-06-10: Validated (validate-create-story, independent fresh-context) → ACCEPT-WITH-FIXES. Verified the core thesis (single choke point `select_columns`; engine/services/config/reporter genuinely unchanged; back-compat; reproduce round-trip; `data/loader.py` leaf). Applied fixes: corrected the predict-naming citation (`services.py:346-349` in `predict_with_run`, not `313-320` which is `compare_runs`); added the `fnmatchcase`-not-`fnmatch` case-folding trap; clarified that a non-existent literal is caught by the existing missing-columns guard (no redundant literal existence check); clarified that loader-is-leaf is a convention, not one of the 3 enforced contracts.
- 2026-06-10: Implemented R1.1 (dev-story) — `expand_field_columns` in `data/loader.py` resolves output glob patterns (deterministic, frame-column order) at the `select_columns` choke point; literal back-compat preserved; `DataError` names offenders for zero-match / overlapping DOF; mesh-free. New `tests/test_field_ingest.py` (12 tests, incl. glob e2e train + reproduce determinism). **Zero engine/config change** (diff: `data/loader.py` + tests). Gates: 240 passed / 2 skipped; ruff + mypy clean; import-linter 3/3. Status → review.
- 2026-06-10: Code review (Blind · Edge · Auditor) — Auditor **ACCEPT** (5/5 ACs; single choke point; engine/config/services/reporter/trainer unchanged; `fnmatchcase` confirmed). Applied 1 [Patch]: a bracketed-index literal column (`disp[0]`/`stress[xx]`) was misparsed as a `[…]` glob → spurious "matched no columns"; now an exact column name is always a literal (+ `test_bracketed_index_column_is_literal_not_glob`). 2 [Defer]: glob-run predict field-naming → R4.1; non-string column labels → latent hardening (both in `deferred-work.md`). 4 dismissed. Post-review gates: 241 passed / 2 skipped / 0 failed; ruff + mypy clean; import-linter 3/3. Status → done.
