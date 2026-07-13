# surroforge-professional

Professional report formats for SurroForge.

## Install For Development

From the repository root:

```bash
.venv/bin/python -m pip install -e plugins/surroforge-professional
```

## License Gate

Professional features need a signed license token (M3: content validation,
offline against the vendor key). Obtain a token, then either

```bash
surroforge license install <token-or-file>   # writes ~/.surroforge/professional.lic
surroforge license status                    # verify: holder, tier, expiry
```

or set it for the session (takes precedence over the installed file):

```bash
export SURROFORGE_PRO_LICENSE='SFL1.…'
```

For development, issue yourself a license: `python tools/licensing/issue.py issue
--holder "dev"` (one-time `keygen` first — see `tools/licensing/README.md`).

## CLI Usage

```bash
surroforge report run-001 --format html
```

The report is written to:

```text
reports/run-001.html
```

## DoE Generation

Generate a design-of-experiments sample table from a YAML spec:

```yaml
# doe.yaml
design: lhs            # full-factorial | lhs | sobol | random
samples: 50
seed: 1
factors:
  - {name: aoa, type: continuous, bounds: [-5, 16]}
  - {name: speed, type: discrete, levels: [20, 40, 60, 80, 100]}
```

```bash
surroforge doe run doe.yaml --out ./design
```

This writes `design/samples.csv` (a `case` column plus one column per factor)
and `design/doe.manifest.json` (a reproducibility record). For specs with
**numeric factors only** (continuous/discrete), the CSV is directly consumable
by `surroforge data add-openfoam --samples`. Specs that include **categorical**
factors produce string columns that `add-openfoam` cannot ingest — encode them
(e.g. one-hot or label) before passing to that command. LHS/Sobol need the extra:
`pip install 'surroforge-professional[doe]'`.

## Python API

```python
from surroforge import Project
from surroforge.services import generate_report

project = Project.open("beam-study")
run = project.run("run-001")
path, html = generate_report(project, run, fmt="html")
print(path)
```

## v1 Scope

- Static single-file HTML.
- Embedded CSS.
- No required JavaScript.
- No external network assets.
- Markdown report parity plus metric cards and simple visual structure.
- Optional Field-ROM details when the run carries complete metrics and lineage.

PDF audit reports and interactive charts are outside the v1 package.
