---
stepsCompleted: [1, 2, 3, 4]
inputDocuments: []
session_active: false
workflow_completed: true
session_topic: 'Reduced Order Models (ROM) with modal reduction for FEM models, as a SurroForge feature'
session_goals: 'A broad option map — every way ROM/modal reduction could fit into SurroForge (architecture, data, math, validation, UX, packaging)'
selected_approach: 'ai-recommended'
techniques_used: ['Morphological Analysis', 'SCAMPER Method', 'What If Scenarios', 'Cross-Pollination']
ideas_generated: 101
context_file: ''
---

# Brainstorming Session Results

**Facilitator:** Georg
**Date:** 2026-06-04

## Session Overview

**Topic:** Reduced Order Models (ROM) with modal reduction for FEM models, as a SurroForge feature.

**Goals:** A broad option map — every way ROM/modal reduction could fit into SurroForge, spanning architecture, data ingestion, the math/method layer, validation & reproducibility, UX/CLI, and packaging. Maximum divergence first (target 100+ ideas, deliberate domain shifts every ~10 ideas), organizing only afterward.

### Session Setup

SurroForge today maps parameters → scalar/vector outputs via `BaseTrainer` plugins (Kriging, RBF, polynomial), with versioned runs, an out-of-sample SHA-256 reproducibility fingerprint, and pickle/ONNX export. ROM/modal reduction introduces **field-valued outputs** over a mesh (displacement, stress, temperature, eigenfrequencies), reduced onto a small modal basis (eigenmodes or POD/snapshot modes); a surrogate then maps parameters → modal coefficients, with full-field reconstruction on demand. The session explores how this could fit SurroForge without breaking the existing scalar path.

## Technique Selection

**Approach:** AI-Recommended Techniques
**Analysis Context:** ROM + modal reduction in SurroForge, aiming for a broad cross-domain option map (max divergence, 100+ ideas).

**Recommended Techniques:**

- **Morphological Analysis (Phase 1 — foundation):** Names the design-space axes and enumerates options per axis — the skeleton of the option map.
- **SCAMPER Method (Phase 2 — generation):** Systematically transforms the existing SurroForge architecture (BaseTrainer, runs, fingerprint, export) to fit field/ROM outputs.
- **What If Scenarios (Phase 3 — divergence):** Breaks constraints and assumptions to reach the novel tail of ideas.
- **Cross-Pollination (Phase 3 — divergence):** Imports solutions from adjacent domains (PCA/image codecs, recommender latent factors, control-theory balanced truncation) to combat semantic clustering.

**AI Rationale:** Goal is breadth, not convergence. Morphological Analysis gives the map structure; SCAMPER grounds it in the real codebase; What-If + Cross-Pollination push past the obvious into ideas 50–100.

## Technique Execution — Idea Log

_Ideas captured live, in [Category #X] format. Anti-bias: domain pivots every ~10 ideas._

### Technique 1 — Morphological Analysis

Design-space axes: A1 basis source · A2 reduction method · A3 surrogate target · A4 field ingestion · A5 reconstruction/query · A6 field validation metrics · A7 reproducibility fingerprint · A8 export/artifact · A9 CLI/API/config · A10 packaging/deps.

**A1 — Source of the modal basis**
- **[A1 #1] Eigenmode Borrow** — use the solver's modal-analysis eigenvectors (.op2/.rst/.frd) as the basis (classic modal superposition). *Novelty:* ride the FEM solver's physically-meaningful modes; no extra math.
- **[A1 #2] POD-from-Snapshots** — SVD/POD over full-field DOE results, modes ranked by captured energy. *Novelty:* basis tuned to the parameter variation you care about → fewer modes.
- **[A1 #3] Learned Modes (Autoencoder)** — latent space of a (nonlinear) autoencoder is the reduced basis. *Novelty:* breaks POD's linearity; handles contact/large-deformation manifolds.
- **[A1 #4] Greedy Reduced-Basis** — greedily add worst-error parameter points via an a-posteriori estimator. *Novelty:* certified error bound, minimal snapshots, no full pre-run DOE.
- **[A1 #5] Krylov / Moment-Matching** — basis from Krylov subspaces matching transfer-function moments. *Novelty:* preserves input→output dynamics (FRF) for vibration/acoustics.
- **[A1 #6] Hybrid Eigen + POD-Residual** — eigenmodes plus POD of the residual snapshots. *Novelty:* physical interpretability + data-driven correction.

**A2 — Reduction / projection method**
- **[A2 #7] Energy-Threshold SVD** — keep modes to cumulative energy ≥ 99.x%; rank auto-selected, YAML knob. *Novelty:* data picks the rank.
- **[A2 #8] DEIM / Gappy-POD** — reconstruct full field from a few sampled nodes; nonlinear/sparse-sensor capable. *Novelty:* "full field from a handful of gauges" → sensor-placement product angle.
- **[A2 #9] HOSVD / Tensor Reduction** — decompose (nodes × params × time) tensor via Tucker/HOSVD. *Novelty:* one model spanning transient + parametric variation.
- **[A2 #10] LSPG Projection** — least-squares Petrov–Galerkin residual minimization. *Novelty:* stable for advection-dominated CFD where Galerkin ROMs blow up.

**A3 — Surrogate target (what the regression predicts)**
- **[A3 #11] One-Surrogate-Per-Mode** — independent existing trainer per modal coefficient. *Novelty:* reuses the entire current `BaseTrainer` stack unchanged; ROM is a pre/post wrapper.
- **[A3 #12] Multi-Output Coefficient Vector** — one trainer predicts the whole coefficient vector, shared hyperparameters. *Novelty:* captures inter-mode correlation; one artifact.
- **[A3 #13] Grassmann-Manifold Interpolation** — interpolate basis subspaces across parameters on the Grassmann manifold. *Novelty:* handles dominant modes that *rotate* with parameters.
- **[A3 #14] Parametric / Local Bases** — cluster parameter space, switch bases per regime via a classifier. *Novelty:* survives topology/contact changes one global basis can't.

**A7 — Reproducibility fingerprint**
- **[A7 #15] Probe-Node Field Fingerprint** — SHA-256 the reconstructed field at fixed out-of-sample probe nodes. *Novelty:* preserves SurroForge's exact out-of-sample reproducibility contract on the user-observable field.
- **[A7 #16] Sign/Order-Canonicalized Mode Hash** — fix mode sign (max-abs component) and order (energy) before hashing. *Novelty:* bit-reproducible despite SVD sign-flip/ordering ambiguity.

**A4 — Field ingestion format**
- **[A4 #17] Solver-Native Readers** — optional backend reads Nastran .op2, Ansys .rst, Abaqus .odb, CalculiX .frd directly. *Novelty:* meets engineers where their data lives; no manual export.
- **[A4 #18] Open-Standard Mesh+Field** — ingest VTK/VTU, Exodus, XDMF/HDF5 via meshio. *Novelty:* solver-agnostic, plays with ParaView ecosystem.
- **[A4 #19] Snapshot-Matrix Mode** — accept a plain (n_nodes × n_samples) Parquet/HDF5 matrix, mesh optional. *Novelty:* reduction works even without topology; bring fields from anywhere.
- **[A4 #20] Mesh as Versioned Artifact** — register & hash mesh/topology as a first-class dataset separate from values. *Novelty:* extends data-hashing/lineage to geometry; detects silent mesh changes.

**A5 — Reconstruction & query**
- **[A5 #21] Field Write-Back** — `predict` reconstructs and writes a full VTK/Exodus field for new parameters. *Novelty:* closes the loop back into ParaView/post-processors.
- **[A5 #22] Lazy Coefficient Storage** — store coefficients + basis, reconstruct on demand. *Novelty:* artifact stays KB-sized even for million-node meshes.
- **[A5 #23] Probe-Point Query** — field at specific coords/nodes without rebuilding the whole field. *Novelty:* instant "stress at this fillet?" queries.
- **[A5 #24] Superposition Preview** — reconstruct with only the first k modes interactively. *Novelty:* user trades speed/fidelity at query time, not train time.

**A6 — Validation metrics for fields**
- **[A6 #25] Two-Source Error Split** — report truncation error vs. surrogate error separately. *Novelty:* tells user whether to add modes or add data — actionable.
- **[A6 #26] Relative L2 / Energy-Norm** — mesh-normalized, optionally mass-weighted field error. *Novelty:* physically meaningful global accuracy number vs. scalar RMSE.
- **[A6 #27] Worst-Region Heatmap** — localize & export where the surrogate is least accurate as a field. *Novelty:* spatial trust map, not one scalar.
- **[A6 #28] QoI Validation** — validate engineering QoIs (peak stress, max displacement, eigenfreq) from the reconstructed field. *Novelty:* validates what engineers actually sign off on.

**A9 — CLI / API / config surface**
- **[A9 #29] `model.reduction:` YAML Block** — config section (method, rank or energy target, basis source) beside `model.method`. *Novelty:* ROM configured declaratively, same versionable source-of-truth.
- **[A9 #30] `surroforge rom inspect`** — show energy spectrum, chosen rank, per-mode contribution. *Novelty:* makes the reduction auditable, fitting the audit-output buyer persona.

**A10 — Packaging / dependencies**
- **[A10 #31] `surroforge[rom]` Extra** — optional dep group (meshio/pyvista/scipy), lazy imports, mirrors `[smt]`/`[onnx]`. *Novelty:* ROM ships without bloating base install.

### Technique 2 — SCAMPER (against the existing SurroForge architecture)

_Fanned out 7 lenses in parallel (42 raw ideas), curated and de-duplicated below. Numbering continues from the morphological map._

**Substitute**
- **[#32] Basis-Plus-Coeffs Artifact** — split `model.pkl` into a content-addressed `basis.npz` (deduplicated across runs by hash) + a tiny coeffs mapper. *50 runs over one POD basis cost one copy of the modes; the basis is auditable on its own.* (touch: PickleExporter, core/run.py, persistence)
- **[#33] Mesh-Aware Data-Hash** — split `data_hash` into `{mesh-topology, values}`; a cosmetic node-renumber stays "reproduced", a genuine remesh is flagged. *Inverts today's byte-order hash, which would do the opposite.* (touch: data/hashing.py, lineage, reproduce())
- **[#34] Rank-As-Resolved** — `reduction.rank: adaptive`; engine picks rank at fit time and writes the **resolved** rank into the config *snapshot*, which becomes the reproducible spec. *Inverts config immutability — the snapshot, not the input YAML, is canonical.* (touch: core/config, CONFIG_SNAPSHOT, reproduce())
- **[#35] Reconstruct-On-Predict** — `predict` returns compact coefficients by default; full field only on `--field`. *KB vs GB per query on a million-node mesh; decouples prediction shape from training-target shape.* (touch: cli predict, services.predict)

**Combine**
- **[#36] Basis-Inside-Trainer** ⭐ — a `ReducedFieldTrainer(BaseTrainer)` smuggles basis+inner trainer behind the ordinary `predict()` contract, so export/compare/conformance/ONNX see a normal `(n, n_field_nodes)` trainer. *ROM needs **zero** new downstream touchpoints; a closed-source reducer ships as an ordinary entry point.* (touch: registry, BaseTrainer, engine)
- **[#37] Co-Trained QoI Anchor** — train the field-coefficient surrogate AND a scalar-QoI head fed by the *predicted coefficients*; ship as one artifact. *The QoI head becomes a physics-aware cross-check catching reconstruction error the field metrics miss — one fit, not two pipelines.* (touch: engine, metrics.json, model.pkl)
- **[#38] Multi-Physics One Run** — several fields share parameter inputs; per-field bases but ONE joint coefficient surrogate. *Thermal modes inform structural coefficients (thermoelastic coupling) for free; lineage/fingerprint over the joint output.* (touch: config outputs grouping, engine, compare_runs)
- **[#39] Compare-Across-Bases** — `compare` two runs with different basis sources (eigenmode vs POD) in **reconstructed physical space** at shared probe nodes; pick best by energy-norm + "modes used"/"energy captured" columns. *Makes basis-source an A/B-testable hyperparameter; coefficient RMSE is meaningless across incommensurable bases.* (touch: compare_runs, _select_best, markdown reporter)

**Adapt**
- **[#40] Honest-Basis CV** — re-derive the modal basis *inside each CV fold* from only that fold's training snapshots, reusing the existing per-fold fresh-factory contract. *Standard k-fold reuses a global basis → leaks test snapshots into the basis and inflates R²; this surfaces basis variance as a first-class number.* (touch: cross_validation factory)
- **[#41] Metric-as-Dict-Key** — register `energy_l2`, `rmse@mode_k` as ordinary `AVAILABLE_METRICS` callables. *Everything downstream iterates `config.validation.metrics` generically, so a per-mode/energy metric flows through CV, best-run selection, and the report with zero new plumbing.* (touch: validators/metrics, cross_validation, compare_runs)
- **[#42] Two-Pass Report Diff** — render the Markdown report twice (coefficients vs reconstructed field) into one table. *Visibly separates the projection floor (error the basis can never beat) from surrogate-added error — exposes a "great R² from a tiny basis doing nothing".* (touch: reporters/markdown context build)

**Modify / Magnify**
- **[#43] Reduced-Space Fingerprint** ⭐ — fingerprint the sign/order-canonicalized out-of-sample **coefficients**, not the reconstructed field. *Cheaper AND more sensitive — reduction removes the 512-row cap (coefficients are tiny), isolates the learned map from the deterministic basis, and is basis-relative so a sign-flipped-equivalent basis still reproduces.* (merges 5 fingerprint variants) (touch: engine._prediction_fingerprint, reproduce/_verdict)
- **[#44] Transient Fingerprint Spine** — for transient fields, hash the per-timestep coefficient *trajectory*. *Catches late-time integration drift (model reproduces t=0, diverges late) that any single-instant hash misses.* (touch: engine fingerprint cap → timesteps, reproduce)
- **[#45] Living Field Report** ⭐ — emit a self-contained HTML/JS report embedding the basis + holdout coefficients; reader scrubs a parameter slider and the field reconstructs client-side (no solver, no server). *The report file becomes the lightest possible deployment of the ROM — blurs report and exporter.* (touch: new html reporter entry point)

**Put To Other Uses**
- **[#46] Modal Residual Sentry** — basis as a one-class anomaly detector at `data add`: flag any imported field whose out-of-basis residual energy exceeds the worst seen in the snapshot set. *Free calibrated outlier score from the reconstruction error; catches cracked-mesh/regime-change/divergent solves before they poison a surrogate.* (touch: new validator on `data add`)
- **[#47] Modal Artifact Compressor** — store run snapshot fields as coefficients + shared basis instead of full mesh fields. *The run dir becomes **smaller than the raw data** it was trained on; reconstruction tolerance recorded as an auditable lineage field.* (touch: persistence/store, lineage)
- **[#48] Solver-Regression Catcher** — `reproduce()` across solver versions: project parent and freshly re-solved fields onto the same stored basis, diff per-mode coefficients. *Localizes WHICH spatial mode regressed (mode 3 drifted 4%), not just an opaque max-abs-diff.* (touch: services.reproduce, ReproduceResult)
- **[#49] Optimal Probe Picker** — pivoted QR / Q-DEIM over the basis to recommend the minimal node set that best reconstructs all modes → feeds the field-fingerprint probe nodes & sensor placement. *Same DEIM math repurposed to answer "which N nodes matter most"; the ROM becomes an experiment-design advisor.* (touch: `rom inspect` subcommand, engine fingerprint)
- **[#50] Modal Gap-Filler** — gappy-POD as in-painting: solve coefficients over observed nodes only, fill missing/masked nodes from the reconstruction. *Lets SurroForge ingest partial fields (failed elements, sensor dropout) it would otherwise reject — reduction math reused as data recovery.* (touch: data/loader ingestion, provenance)

**Eliminate**
- **[#51] Reduction-Implies-Method** — when a `reduction` block is present, `model.method` becomes optional/inferred (kriging for few modes, polynomial otherwise). *Inverts "method is king, reduction is preprocessing" → reduction is a first-class method that auto-selects its inner surrogate.* (touch: config validator, engine.get_trainer)
- **[#52] Append-A-Mode** — exploit per-mode independence: re-open a run in "augment" mode, fit ONLY the new mode's surrogate, reuse cached basis/surrogates by hash via `parent_run` lineage. *Makes "add mode k+1" an O(1) incremental run vs. rebuilding from scratch.* (touch: engine incremental branch, Run lineage)
- **[#53] Meshless Snapshot-ROM** — never ingest a mesh; a field is just "output columns that move together"; the conformance suite *enforces* the reducer touches no coordinates/connectivity. *Makes "meshes need not match" a tested invariant instead of a feature; the wide-output CSV path is the whole ingestion story.* (touch: data/loader, conformance no-mesh check)

**Reverse**
- **[#54] Inverse-Run Flag** ⭐ — `model.direction: inverse` swaps x/y at the engine boundary: fit observed field/coefficients → the parameters that produced them. *Inverse identification ships as a one-line config polarity switch reusing the entire run/lineage/fingerprint stack — no new subsystem.* (touch: config direction, engine x/y assignment)
- **[#55] Sensitivity-By-Reproduce** — re-run a stored run with one input perturbed by ε; reuse the holdout-fingerprint `max_abs_diff` as the raw field-sensitivity signal. *Repurposes the divergence machinery (built to prove sameness) as a finite-difference derivative engine.* (touch: services.reproduce, new `sensitivity` verb)
- **[#56] Goal-Field Optimizer** — outer loop over the cheap surrogate forward model: search parameter space for the design whose reconstructed field best matches a target field/QoI. *The ROM becomes the objective function for inverse design — prediction tool → design-search tool, no trainer changes.* (touch: new services.optimize_to_target, `design` verb)
- **[#57] Reconstruction Conformance Probe** — extend the shippable conformance suite with a coeffs→field→coeffs round-trip check within `DEFAULT_TOLERANCE`. *A commercial ROM plugin must prove its basis is a faithful projector (no sign/normalization bug) before claiming conformance — reuses the same `record()`/tolerance plumbing as `fingerprint_participation`.* (touch: conformance/trainer.py)

### Technique 3 — What-If Scenarios + Cross-Pollination (divergence)

_Fanned out 5 What-If provocations + cross-domain raids in parallel (35 recovered); the 4 failed domains (codecs/recommender/control/foundation-models) were mined manually to crest 100. Numbering continues._

**What-If: Fully-Autonomous ROM**
- **[#58] Rank-by-Reproduce** — pick rank as the smallest that still reproduces within 1e-9 across two independent re-imports (rank = where the verdict flips, minus one). *Reproducibility becomes the objective function selecting a structural hyperparameter, not a post-hoc audit.*
- **[#59] Field Auto-Triage** — with no `outputs:` declared, compute each field's intrinsic modal dimensionality and reduce only the reducible ones, **declining** the rest in the report. *A CAE tool that refuses to model an un-reducible field — honest, uncomfortable.*
- **[#60] History Meta-Prior** — mine the project's `runs/*/lineage.json` as meta-training data (schema-signature → winning basis/rank/method) to seed a new run's auto-config. *Turns the audit trail into a meta-learner; data SurroForge already has but never exploits.*
- **[#61] Propose-the-Experiment** — `surroforge propose` emits an annotated draft `config.yaml` + prose rationale and stops, instead of training. *A falsifiable, reviewable proposal with a paper trail of WHY — fits the sober engineer who must defend choices.*
- **[#62] Budget-Driven ROM** — user states only a deployment envelope ("predict <2 ms", "artifact <500 KB", "must export ONNX") and the engine solves backwards for the richest model that fits. *Constraint-first, not accuracy-first — engineer states the envelope, not the math.*

**What-If: Mesh Changes Every Run**
- **[#63] Coordinate-Channel Trainer** ⭐ — promote `(x,y,z)` to input columns: a row becomes `(params, point) → value`, so an existing Kriging/RBF learns a continuous, mesh-free field (neural-field-like). *Dissolves modal reduction entirely, reusing the existing X/y `select_columns` contract — no new machinery.*
- **[#64] Quadrature Fingerprint** — evaluate the predicted field at config-pinned probe coordinates, integrate against fixed quadrature weights, hash those scalars. *Rescues the crown-jewel verdict in coordinate space when node order/array shape are no longer comparable across re-meshed runs.*
- **[#65] Mesh-As-Salt Lineage** — hash each snapshot's mesh separately; `data_hash` becomes `(values-hash, set-of-mesh-hashes)`, enabling a "same physics, silently re-meshed" verdict distinct from "data changed". *(extends #33)*
- **[#66] Snapshot Self-Basis** — no persistent basis; each prediction regenerates a basis deterministically from a config-pinned reference point cloud. *The artifact holds the recipe to rebuild the modes, not the modes.*
- **[#67] Optimal-Transport Conformance** — conformance check: fit on mesh A, predict onto a different mesh B, assert agreement under a Wasserstein/OT field distance. *Adds a "mesh-invariance" contract clause, trading byte-exactness for a geometric distance.*

**What-If: Real-Time / Embedded Digital Twin**
- **[#68] Fixed-Point Foundry** ⭐ — `export --format embed-q15` co-quantizes basis + surrogate to int16, searching per-mode integer scales so on-device reconstruction stays within a field-error budget. *Quantization judged by reconstructed-field error, spending bits on energetic modes — borrowed from codec quant tables.*
- **[#69] Latency Conformance** — conformance checks for worst-case predict+reconstruct wall-time, peak working-set bytes, allocation-free-after-warmup. *Treats a hard real-time/memory ceiling as a contract — a numerically-perfect model that blows the PLC budget fails.*
- **[#70] Drift Watermark** — burn the holdout fingerprint into the embedded artifact as a golden `(param → expected coeffs)` vector; the on-device twin self-tests on a heartbeat and flags drift. *Ships the crown-jewel SHA-256 into firmware as continuous edge self-verification.*
- **[#71] Tiered Reconstruct** — export the basis in importance-ordered append-only chunks; the twin reconstructs as many modes as its current latency slack allows (4 under load, 40 idle). *Anytime/progressive decoding (JPEG) lands naturally because modes are already energy-ranked; fidelity becomes a function of instantaneous load.*

**What-If: Cross-Solver / Cross-Hardware Reproducibility**
- **[#72] Per-Mode Tolerance Bands** ⭐ — replace the flat `REPRODUCTION_TOLERANCE=1e-9` with a per-mode tolerance vector: tight for high-energy modes, loose for float-noise tail modes. *Reproducibility tolerance becomes heterogeneous and physics-derived; a 1e-12-energy tail mode shouldn't gate at 1e-9.*
- **[#73] Two-Verdict Split** — `reproduce()` returns two orthogonal verdicts: BITWISE ("same bytes, same machine") and PHYSICS ("same field within energy-norm, bytes may differ"). *Admits cross-solver reproduction is a different KIND of true, not a weaker rung.*
- **[#74] Solver-Neutral Coeff Crate** — define sign-canonicalized modal coefficients in fixed dtype/order as a canonical cross-solver interchange artifact with its own hash. *Inverts the pickle-provenance warning: the one artifact deliberately designed to be shared and compared across machines.*
- **[#75] Precision-Sweep Conformance** — fit float64 vs float32, assert agreement within an energy-band, report per-mode precision sensitivity. *The existing `_deterministic` check (same precision, 1e-9) can never catch a method whose tail explodes under float32.*
- **[#76] Divergence Bisector** — on a `diverged` verdict, walk the recorded environment delta (python/platform/each dep pin) reverting one at a time to attribute divergence to the single change that closes the band. *Turns `capture_environment`'s passive forensic record into an active git-bisect over the environment.*

**What-If: Streaming / Online Modes**
- **[#77] Drift Ledger** ⭐ — replace `lineage.json + parent_run` with an append-only hash-chained ledger of incremental-SVD updates; the model is the ledger head; `reproduce()` = deterministic event-sourced replay from genesis. *Reproducibility as git-style commit verification of an unbounded stream — order-of-arrival is part of the fingerprint.*
- **[#78] Subspace Odometer** — fingerprint the Grassmann principal-angle trajectory between consecutive bases, not a frozen prediction. *Two ROMs are "the same" because they LEARNED identically, not because they predict identically today.*
- **[#79] Forgetting Curve** — exponential-forgetting weight in the incremental SVD (`model.reduction.forgetting_halflife`) so old snapshots decay and modes track a drifting regime. *Deliberately non-stationary — a Kalman-filter-on-a-moving-plant view, the uncomfortable inverse of "more data is always better".*
- **[#80] Rank Tribunal** — each streamed batch votes; if residual energy exceeds threshold, auto-promote rank+1 and record a "mode-birth" event with timestamp + triggering data-hash. *Rank becomes an emergent, data-triggered event with a birth certificate, not a config knob.*
- **[#81] Tombstone Replay** — a discovered-bad sim (NaN/non-converged) becomes a retraction event undone by a rank-1 SVD downdate; `reproduce()` replays insert-then-retract and still lands identically. *Event-sourced, replayable deletion inside a numerics library — surgically remove bad data without breaking the chain.*

**Cross-Pollination: Computer Graphics / Animation**
- **[#82] Corrective Pose-Space ROM** ⭐ — primary surrogate maps params → coefficients linearly; a SECOND trainer learns RBF "corrective shapes" in field-residual space keyed on the parameter vector. *Pose-Space Deformation from rigging: the linear part is deliberately wrong, a sparse parameter-activated correction carries the nonlinearity.*
- **[#83] Skinning Rotation Guard** — detect the linear-blend-skinning "candy-wrapper" collapse: large rotation/advection across the sweep (rotating wake, moving shock) makes linear POD collapse volume. *Reframes a known POD weakness as a named, detectable failure with a borrowed CG diagnostic — a sentinel, not a new basis.*
- **[#84] Quadric Probe Decimation** — use Garland-Heckbert Quadric Error Metrics (mesh-decimation engine) to pick the fingerprint/validation probe nodes by reconstruction-error cost. *Replaces the arbitrary first-512-rows fingerprint slice with an error-aware, geometry-aware decimation. (cf. #49)*
- **[#85] Residual Normal-Map Bake** — train on a cheap coarse mesh, bake the systematic fine-vs-coarse residual into a stored "displacement-map" field artifact, reapply on `--hires`. *Decouples surrogate cost (coarse) from output fidelity (fine) — the game-LOD normal-map baking trick.*
- **[#86] Keyframe Curve Decimation** — treat each mode's coefficient-vs-time as an animation curve; decimate to error-bounded keyframes with spline tangents. *Makes the temporal axis a tolerance-driven compressible channel — animation-curve compression for transient ROMs.*

**Cross-Pollination: Weather / Data Assimilation**
- **[#87] Kalman-Gain Predict** ⭐ — surrogate emits an ensemble of coefficient vectors; at predict time, ingest a few live sensor readings and apply the EnKF Kalman gain to pull predictions toward reality. *Makes `predict()` a two-input operation (parameters AND sparse measurements) that self-corrects against observations.*
- **[#88] Field Nudging** — `predict --nudge sensors.csv` adds a Newtonian-relaxation term pulling the reconstructed field gently toward measured points without refitting. *The meteorologist's nudging trick — turns the surrogate from an oracle into a steerable field, cheaper/more conservative than full DA.*
- **[#89] Innovation Whiteness** — a validator that runs a whiteness test (lag-1 autocorrelation / Durbin-Watson) on the residual stream; autocorrelated innovations = systematic bias. *DA insists a healthy filter has WHITE innovations — correlation in the residual stream is itself the diagnosis, beyond scalar RMSE.*
- **[#90] Mode Localization** — apply Gaspari-Cohn covariance localization to the modal-coefficient covariance, tapered by mode-index/eigenvalue distance, to kill spurious small-sample cross-mode coupling. *Transplants small-ensemble DA's key fix into spectral space where "distance" is mode separation, not geometry.*
- **[#91] Run Reanalysis** — `surroforge reanalyse <ref-run>` freezes ONE reference surrogate and replays the whole archive of datasets through it for a single internally-consistent error record across time. *The ERA5 reanalysis idea — the opposite of "every run is a fresh model"; reveals genuine improvement vs. drift.*

**Cross-Pollination: Image / Video Codecs** _(manually mined)_
- **[#92] Inter-Run Delta Coding** — store a child run as a P-frame-style delta against its parent run's basis + coefficients (only the changed modes/residual), reconstructable by chaining. *Inter-frame coding over `parent_run` lineage — a re-train after adding a few snapshots costs a delta, not a full artifact. (cf. #47, #52)*
- **[#93] Block-Local Bases** — partition the mesh into spatial patches (JPEG 8×8-block analogue) and reduce each patch independently with its own small basis. *Localizes basis error to regions and lets a high-detail region get more modes than a quiet one — spatial, vs. the parameter-space regimes of #14.*

**Cross-Pollination: Recommender Systems / Matrix Completion** _(manually mined)_
- **[#94] DOE Matrix Completion** — treat the (DOE-point × mode-coefficient) table as a sparsely-observed matrix and complete the unrun design points via regularized matrix factorization, before/instead of a separate surrogate. *Collaborative-filtering reframing: "fill in the DOE you didn't run" rather than "fit a regressor".*
- **[#95] Cold-Start Basis Borrow** — a new field/case with too few snapshots borrows a related field's basis as a prior, blending it out as its own snapshots accrue. *The recommender cold-start problem — graceful behaviour at low data instead of refusing to model. (cf. #60)*

**Cross-Pollination: Control Theory** _(manually mined)_
- **[#96] Balanced-Truncation Ranking** — rank modes by Hankel singular values (controllability×observability), not energy, keeping input→output-important modes. *Energy-dominant ≠ response-important; the systems-theory ordering keeps modes that actually move the QoI.*
- **[#97] Certified Error Bound** — emit a control-theory a-posteriori error bound as a guaranteed accuracy **certificate** in lineage/report, not just an empirical CV number. *A provable upper bound the sober-engineer persona can put in a sign-off document. (cf. #4)*
- **[#98] Stability-Preserving Verdict** — check the reduced operator preserves stability/passivity; flag a ROM that is accurate on the holdout yet dynamically unstable. *For transient CFD a ROM can fit every snapshot and still blow up when integrated — a failure no accuracy metric catches.*

**Cross-Pollination: Foundation Models / Embeddings** _(manually mined)_
- **[#99] Pretrained Basis Hub** — ship/community-share pretrained modal bases per problem class (a "basis zoo") via the existing plugin/registry mechanism; users fine-tune on their DOE. *Transfer learning + model-hub culture for ROMs; a natural commercial-edition surface.*
- **[#100] Basis Fine-Tuning** — start from a pretrained/parent basis and adapt only the trailing modes to new data. *Pretrain→fine-tune transfer learning: most physics is shared, only the case-specific tail needs new snapshots.*
- **[#101] Cross-Field Embedding Transfer** — learn a shared latent embedding across multiple physics fields so a data-rich field's structure informs a data-poor one. *Shared embedding spaces from multimodal ML — borrow displacement's well-sampled modes to bootstrap a sparse stress field. (cf. #38)*

## Deep Dive (Advanced Elicitation) — ADR-ROM-001: Fingerprint design for field/ROM runs

_Method: Architecture Decision Records. Pressure-tested the fingerprint-rethink cluster (#43, #64, #72, #73, #74, #76, #16, #91)._

**Context.** Today's fingerprint = SHA-256 over the first ≤512 rows of float64 out-of-sample holdout predictions, compared within a flat `1e-9` by `reproduce()`. For a field/ROM run this inherits four problems: million-node fields (the 512-row cap samples an arbitrary spatial slice), node ordering shifts across re-meshed runs, SVD bases carry sign/order gauge ambiguity (false *diverged*), and a single linear verdict can't express "same physics, different solver".

**Options & trade-offs** (H/M/L per force):

| Force | A: Reconstructed-field @ probes | B: Coefficient-space | C: Two-verdict + per-mode bands | D: Status quo |
|---|---|---|---|---|
| Fidelity | H | M (basis-relative) | H | M (512-slice) |
| No false-diverge | H | M (needs #16) | H | L |
| Cost on huge meshes | M | H (no cap) | M | L |
| Continuity w/ scalar path | H | H (identity basis ⇒ today's bytes) | M | H |
| Cross-solver "physics" verdict | M | L alone (needs #74) | H | L |
| Auditability (which mode?) | L | H | H | L |
| Conformance-suite fit | H | H | M | H |

**Decision — layered, not a single winner:**
1. **Foundation (non-negotiable): sign/order canonicalization (#16)** — every option false-diverges without it. Ship first.
2. **Primary fingerprint: coefficient-space (B)** — small, exact, uncapped, per-mode auditable. A scalar run's basis is the identity, so coefficients *are* the predictions ⇒ byte-compare is literally today's behaviour ⇒ scalar path untouched by construction.
3. **Semantics: two-verdict split (C), phased** — `reproduce()` returns a **bitwise** verdict (B's byte-compare, now) and later a **physics** verdict (reconstruct at Q-DEIM probe nodes — option A — within per-mode energy-weighted tolerance bands #72). Physics verdict collapses to bitwise when no reduction is active.
4. **Orthogonal tooling on top:** coeff crate (#74), divergence bisector (#76), reanalysis (#91) consume whichever verdict exists — additive.

**Rationale.** B+C maximize fidelity/sensitivity/auditability while continuity is *automatic* via identity-basis equivalence; phasing keeps the blast radius small (canonicalization + coefficient hash are low-risk and immediately useful; the physics verdict — the new contract surface — lands only when cross-solver matters).

**⚠️ Key guard (the trap this ADR exists to catch).** Coefficient-space (B) can falsely read "reproduced" for *identical coefficients against a different basis*. Mandatory mitigation: **bind a canonicalized basis hash into the verdict** (store beside `holdout_fingerprint` in `lineage.json`); reproduction must match coefficients **and** basis identity. Without that guard, B is a correctness regression.

### First-Principles amendment to ADR-ROM-001

_Method: First Principles Analysis. Stripped the design to fundamentals and it inverted one decision._

**Truths.** (T1) The witness must be *what the user depends on* — the reconstructed physical field / QoIs at engineer-meaningful locations, not modal coefficients (a private, basis-relative implementation detail). (T2) Reproducibility is an equivalence relation on *observable behaviour*; byte-equality is a cheap sufficient proxy, never the definition. (T3) A correct fingerprint is invariant to *exactly* the incidental (node order, basis gauge, dtype, energy-negligible tail modes, machine) and sensitive to *exactly* the field. (T4) A fingerprint is a deliberately-chosen lossy projection to a comparable token — the scalar design's "first 512 raw bytes" was invariant to nothing and sensitive to everything.

**What it overturns.** The ADR made coefficient-space (B) the *primary* fingerprint. First principles **flips the authority**: coefficients are incidental (T1/T3) and cannot *define* reproducibility — they are at best a fast proxy, valid only under the basis-hash guard. Principled ordering:
- **Definition / ground-truth verdict → observable-field equivalence** (reconstructed field/QoIs at pinned, gauge-aware locations; invariant to node-order, sign/order, dtype).
- **Accelerator + diagnostic → canonicalized coefficients + bound basis hash** (provably equivalent under a pinned basis; used for speed + per-mode "which mode moved" attribution, NOT as source of truth).

**Further consequences.** (a) Tolerance isn't fundamental — "indistinguishable to the engineer's decision" is; express tolerance in decision-relevant physical units (field-energy-relative / per-QoI), of which per-mode bands (#72) are one mechanism. (b) The unit of reproducibility need not be the run — "is the model I rely on *now* still the one signed off?" legitimizes a continuous witness (#70 watermark, #91 reanalysis) as a direct consequence of the trust principle.

**Net effect:** keep the layering, **flip the primacy** — observable-field equivalence is the definition; coefficient-space is the fast/audit lens. This also closes the "key guard" trap by construction: you cannot false-"reproduce" on a different basis if the *field* is the ground truth.

### Pre-mortem on ADR-ROM-001 (failure → cause → prevention)

_Method: Pre-mortem Analysis. "It's 2027; a model marked 'reproduced' no longer matches its sign-off." Two findings change the design._

1. **Degenerate/repeated modes break canonicalization** ⚠️ *(design-changing).* Symmetric FEM parts have repeated singular values → modes 3/4 span an arbitrary degenerate subspace; per-mode sign/order canonicalization picks different representatives run-to-run → false *diverged* on the most common (symmetric) parts. **Cause:** #16 assumes well-separated modes. **Prevention:** detect near-equal singular-value clusters and compare the *subspace* via principal angles, or fall back to the gauge-invariant field verdict. *Confirms the field-primary definition is load-bearing, not just cleaner.*
2. **The basis-hash guard becomes a false-diverge engine** ⚠️ *(design-changing).* A routine LAPACK/SciPy upgrade changes basis bytes → basis hash changes → every reproduction reads *diverged* despite identical fields; users learn to ignore the verdict. **Cause:** binding basis *bytes* when only basis *equivalence* matters. **Prevention:** with field-primary, **don't gate on the basis hash** — demote it to diagnostic metadata ("basis changed, field reproduced ⇒ benign re-gauge"). *Retires the ADR's own guard-as-gate.*
3. **Per-mode tolerance becomes a fudge factor.** Someone widens tail bands to silence spurious diverges; later the loosened bands swallow a real regression. **Prevention:** **hash the tolerance vector into the run spec** (loosening = a different run, not a looser compare); cap auto-bands; require logged overrides with a reason.
4. **Probe blind spots drift.** Probes chosen once from the original basis; new regimes move energy into unprobed regions → "reproduced" at probes while the field diverges elsewhere. **Prevention:** record probe coverage energy in lineage; warn on out-of-probe energy; prefer a global energy-weighted field norm as the *definition*, probes only as the fast-path.
5. **"Reproduced" means different things to different readers.** `physics: reproduced / bitwise: diverged` → engineer signs off, auditor escalates, CI gates the wrong axis. **Prevention:** explicit precedence policy — deployment/sign-off keys on **physics**, bit-audit/forensics keys on **bitwise**; CI config names its axis.

**Design deltas:** (1) field verdict is *required*, not optional; (2) basis hash → diagnostic, not gate; (3) tolerance folded into the hashed spec; (4) record probe coverage + prefer global energy-norm; (5) ship a verdict-precedence policy.

## Idea Organization and Prioritization

**Thematic organization (101 ideas → 6 themes):**
1. **Modal Math & Basis** — the reduction engine: basis sources #1–6, reductions #7–10, Grassmann #13, local/regime bases #14, balanced-truncation ranking #96, corrective pose-space #82, skinning-collapse guard #83, block-local #93, pretrained-hub/fine-tune/transfer #99–101, mesh-free coordinate-channel #63.
2. **Trainer Contract & Artifacts** — how ROM bolts onto `BaseTrainer`: per-mode #11 / multi-output #12, **basis-inside-trainer #36**, rank-as-resolved #34, reduction-implies-method #51, split artifact #32, modal compressor #47, inter-run delta #92, reconstruct-on-predict #35, append-a-mode #52, reconstruction conformance #57.
3. **Reproducibility & Fingerprint** — the crown-jewel rethink (most-developed; see ADR deep-dive): canonicalization #16, coefficient-space #43, quadrature/field-primary #64, per-mode bands #72, two-verdict #73, coeff crate #74, precision-sweep #75, divergence bisector #76, probe pickers #28/#49, transient spine #44, solver-regression catcher #48, drift ledger #77, subspace odometer #78, tombstone replay #81, stability/certified verdicts #97/#98, mesh-aware hash #33/#65.
4. **Data In / Mesh / Field Validation** — readers & formats #17–20, meshless #53, field metrics #25–28, honest-basis CV #40, metric-as-key #41, two-pass report #42, innovation whiteness #89, mode localization #90, residual sentry #46, DOE matrix completion #94, mode-stratified folds #80.
5. **Query, Reconstruction & Output UX** — field write-back #21, lazy storage #22, probe query #23, superposition/tiered preview #24/#71, **living field report #45**, displacement-map bake #85, keyframe decimation #86, multi-physics #38, co-trained QoI #37, compare-across-bases #39, config/inspect #29–30, packaging #31.
6. **Frontier: Autonomy, Streaming, Inverse & Deployment** — auto-ROM #58–62, streaming/online #77–81, embedded twin #68–70, Kalman-gain predict #87, field nudging #88, reanalysis #91, inverse-run #54, sensitivity #55, goal-field optimizer #56, cold-start #95, slew limiter (streaming).

**The organizing insight:** **#36 (basis hidden inside the trainer) + a field-primary fingerprint (#43→#64)** form a spine where *most of the other 100 ideas become optional enhancements on an otherwise-unchanged Core* — which is what makes a thin MVP feasible.

**Prioritization:**

| Tier | Ideas |
|---|---|
| 🟢 **Quick wins / MVP foundation** | #36 basis-inside-trainer · #11 per-mode · #16 canonicalization · #19 snapshot-matrix ingest · #25 two-source error · #29 `model.reduction:` block · #31 `[rom]` extra · #41 metric-as-key · #57 conformance probe |
| 🟡 **Core architecture (decide early)** | Fingerprint design (ADR-ROM-001) · #34 rank-as-resolved · #32 split artifact · #21 field write-back · #40 honest-basis CV |
| 🔵 **Differentiators (Phase 2)** | #45 living report · #87 Kalman predict · #82 pose-space · #99 basis hub · #54 inverse-run · #68/#70 embedded twin · #91 reanalysis |
| 🔴 **Needs research / risky** | #63 coordinate-channel · #66 self-basis · #13 Grassmann · #96 balanced truncation · degeneracy handling · #67 OT conformance |

## Action Plans

### Plan A — "v1 field ROM" (the thin end-to-end slice)
**Why this matters:** the smallest path that delivers field surrogates while keeping Core's scalar path and reproducibility guarantee intact.
**Next steps:**
1. Spike a `ReducedFieldTrainer(BaseTrainer)` (#36) wrapping a POD energy-threshold basis (#2/#7) + a per-mode inner trainer (#11); assert it passes the **existing** conformance suite unmodified.
2. Snapshot-matrix ingestion (#19): accept `(n_samples × n_dof)` Parquet/HDF5 — no mesh required for v1.
3. `model.reduction:` config block (#29): `method`, `rank` or `energy` target.
4. Field-primary fingerprint with canonicalization (#16 + #64): reconstruct at pinned probes, canonicalize, hash; **field verdict is ground truth**, coefficient-space the fast/audit path.
5. Two-source error split (#25) + field write-back (#21).
6. `surroforge[rom]` extra (#31) + reconstruction conformance probe (#57).
**Resources:** numpy/scipy SVD (no new heavy dep for v1); meshio/pyvista deferred. **Risks:** degenerate modes → rely on field verdict; SVD determinism across BLAS threads (already a Core concern). **Success:** a greenfield field-ROM run trains → validates → reports → `reproduce()` returns a field verdict, with all three built-in inner trainers passing conformance.

### Plan B — Fingerprint spike (de-risk the crown jewel before v1 freezes)
**Why:** highest-leverage and highest-risk change; must be specced before the v1 contract sets.
**Next steps:**
1. Prototype the field-primary verdict + coefficient-space fast-path; prove equivalence under a pinned basis.
2. Build the degeneracy test (symmetric part → repeated singular values); confirm subspace/principal-angle comparison avoids false-diverge.
3. Decide verdict-precedence policy (physics vs. bitwise) and a defensible decision-relevant tolerance derivation; **fold tolerance into the hashed spec**; **demote basis hash to diagnostic**.
**Output:** an updated ADR + a `reproduce()` contract proposal ready to become a PRD/story.

### Follow-ups
- Draft a **PRD for "Field ROM v1"** from Plan A (BMAD `create-prd`).
- Optional parallel **research spike** on the mesh-free coordinate-channel track (#63) as an alternative to modal reduction.

## Session Summary and Insights

**Key achievements:**
- 101 ideas across 3 techniques, deliberately domain-pivoted for divergence; a curated, de-duplicated map organized into 6 themes and 4 priority tiers.
- A worked architectural deep-dive (ADR-ROM-001) on the fingerprint, refined by First-Principles (field-observable is the *definition*; coefficients are the fast/audit lens) and a pre-mortem (field verdict required; basis hash → diagnostic; tolerance in the hashed spec).
- A concrete, thin **v1 slice** and a **fingerprint spike** ready to action.

**Creative breakthroughs:**
- The **minimal-surface-area thesis** (#36 + field-primary fingerprint) — ROM can ship as essentially one new trainer + a fingerprint evolution.
- The **fingerprint is the real problem**, not the math — every field assumption (node order, size, gauge, tolerance, single verdict, same-environment) breaks the current design, making it the central engineering decision.
- Cross-domain transfers that each unlock a capability: pose-space corrective shapes (#82), Kalman-gain prediction (#87), innovation-whiteness validation (#89), balanced-truncation ranking (#96).

**Session reflections:** AI-recommended divergence (Morphological → SCAMPER → What-If/Cross-Pollination) produced strong breadth; parallel fan-out per lens/domain gave genuine independence (and exposed where adjacent domains converge — the fingerprint cluster). Convergence via ADR + First-Principles + pre-mortem turned the richest cluster into a decision-ready design.

