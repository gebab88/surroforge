"""autoencoder-field guard module — the entry-point target, resolvable without torch."""

from __future__ import annotations

from typing import Any

try:
    from surroforge_enterprise.trainers._autoencoder_impl import (
        AutoencoderFieldTrainer,  # pyright: ignore[reportAssignmentType]
    )
except ModuleNotFoundError as exc:  # torch (or a torch submodule) missing
    if exc.name is None or not exc.name.split(".")[0] == "torch":
        raise  # a real bug in the impl module must not be masked

    from surroforge.errors import PluginError

    class AutoencoderFieldTrainer:  # type: ignore[no-redef]
        """Stub: keeps resolution/gating working; construction explains the extra."""

        name = "autoencoder-field"
        required_tier = "enterprise"
        export_formats: frozenset[str] = frozenset()

        def __init__(self, **_: Any) -> None:
            raise PluginError(
                "autoencoder-field requires the torch extra: "
                "pip install 'surroforge_enterprise[torch]'"
            )


__all__ = ["AutoencoderFieldTrainer"]
