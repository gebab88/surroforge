"""surroforge-torch — a PyTorch ``torch-mlp`` surrogate plugin for SurroForge (Epic R7).

An external, entry-points-only distribution: installing it registers the ``torch-mlp``
method through the ``surroforge.trainers`` entry point so SurroForge resolves it with no
change to Core. Switch between backends with ``model.method: mlp`` (scikit-learn, Core) and
``model.method: torch-mlp`` (PyTorch, this plugin).
"""

from surroforge_torch.trainer import TorchMLPTrainer

__all__ = ["TorchMLPTrainer"]
__version__ = "0.1.0"
