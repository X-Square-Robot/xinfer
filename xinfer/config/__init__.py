"""Runtime configuration resolution (docs/configuration.md)."""

from xinfer.config.resolve import (
    DEFAULT_FRONTENDS,
    DEFAULT_LANES,
    DEFAULT_MAX_QUEUED_REQUESTS,
    DEFAULT_MAX_RUNNING_REQUESTS,
    ConfigError,
    ExplicitSettings,
    ResolvedConfig,
    Setting,
    apply_image_preprocess_device,
    cpu_quota,
    resolve,
)
from xinfer.config.tuned import TunedTable, find_tuned, load_tuned

__all__ = ["DEFAULT_FRONTENDS", "DEFAULT_LANES", "DEFAULT_MAX_QUEUED_REQUESTS", "DEFAULT_MAX_RUNNING_REQUESTS",
           "ConfigError", "ExplicitSettings", "ResolvedConfig", "Setting", "TunedTable",
           "apply_image_preprocess_device", "cpu_quota", "find_tuned", "load_tuned", "resolve"]
