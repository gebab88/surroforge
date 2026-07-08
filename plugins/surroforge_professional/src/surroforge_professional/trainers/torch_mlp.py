"""torch-mlp guard module — the entry-point target, resolvable without torch.

The real implementation (``_torch_impl``) subclasses ``nn.Module`` at module
level, so importing it requires torch. This guard lets ``ep.load()`` (and the
licence gate, which runs at resolution) succeed without the ``[torch]`` extra;
constructing the stub then explains exactly what to install.
"""

from __future__ import annotations

from typing import Any

try:
    from surroforge_professional.trainers._torch_impl import (
        TorchMLPTrainer,  # pyright: ignore[reportAssignmentType]
    )
except ModuleNotFoundError as exc:  # torch (or a torch submodule) missing
    if exc.name is None or not exc.name.split(".")[0] == "torch":
        raise  # a real bug in the impl module must not be masked

    from surroforge.errors import PluginError

    class TorchMLPTrainer:  # type: ignore[no-redef]
        """Stub: keeps resolution/gating working; construction explains the extra."""

        name = "torch-mlp"
        required_tier = "professional"
        export_formats: frozenset[str] = frozenset()

        def __init__(self, **_: Any) -> None:
            raise PluginError(
                "torch-mlp requires the torch extra: "
                "pip install 'surroforge_professional[torch]'"
            )


__all__ = ["TorchMLPTrainer"]
