"""Plugin discovery for trainers, exporters, and reporters."""

from surroforge.plugins import gate
from surroforge.plugins.registry import (
    available_command_names,
    available_commands,
    available_comparison_reporters,
    available_exporters,
    available_reporters,
    available_trainers,
    get_comparison_reporter,
    get_exporter,
    get_reporter,
    get_trainer,
)
from surroforge.plugins.seam import (
    PLUGIN_API_VERSION,
    SUPPORTED_API_VERSIONS,
    PluginMetadata,
)

__all__ = [
    "PLUGIN_API_VERSION",
    "SUPPORTED_API_VERSIONS",
    "PluginMetadata",
    "available_command_names",
    "available_commands",
    "available_comparison_reporters",
    "available_exporters",
    "available_reporters",
    "available_trainers",
    "gate",
    "get_comparison_reporter",
    "get_exporter",
    "get_reporter",
    "get_trainer",
]
