"""Write a generated design to samples.csv plus a reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


from surroforge_professional import __version__
from surroforge_professional.doe.sampling import generate
from surroforge_professional.doe.spec import DoeError, DoeSpec

CSV_NAME = "samples.csv"
MANIFEST_NAME = "doe.manifest.json"


def _case_ids(count: int, prefix: str) -> list[str]:
    width = max(3, len(str(count))) if count > 0 else 3
    return [f"{prefix}_{i:0{width}d}" for i in range(count)]


def write_design(spec: DoeSpec, out_dir: str | Path, force: bool = False) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    csv_path = out_dir / CSV_NAME
    manifest_path = out_dir / MANIFEST_NAME

    if not force:
        for path in (csv_path, manifest_path):
            if path.exists():
                raise DoeError(f"{path} already exists; pass force=True (or --force) to overwrite.")

    frame = generate(spec)
    frame.insert(0, "case", _case_ids(len(frame), spec.case_prefix))

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    table_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "design": spec.design,
        "samples": int(len(frame)),
        "seed": spec.seed,
        "case_prefix": spec.case_prefix,
        "factors": [factor.model_dump() for factor in spec.factors],
        "table_sha256": table_hash,
        "plugin_version": __version__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return csv_path, manifest_path
