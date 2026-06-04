"""Registered dataset metadata (PRD 7.3).

A dataset records the file path (relative to the project), a content hash, the
column schema, and shape. It is stored inside ``surroforge.yaml`` and reloaded
when a training config references it by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dataset:
    name: str
    path: str  # relative to the project root
    hash: str  # sha256 of the dataframe content
    n_rows: int
    n_cols: int
    schema: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "hash": self.hash,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            path=data["path"],
            hash=data["hash"],
            n_rows=int(data.get("n_rows", 0)),
            n_cols=int(data.get("n_cols", 0)),
            schema=dict(data.get("schema", {})),
        )
