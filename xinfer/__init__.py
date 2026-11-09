"""XInfer: inference serving for vision-language-action models.

The package root stays light: importing ``xinfer`` does not import torch or
initialize a device. ``Policy`` is resolved lazily.
"""

__version__ = "0.0.1.dev0"

__all__ = ["Policy", "__version__"]


def __getattr__(name):
    """Import ``Policy`` (and with it torch) on first access to ``xinfer.Policy``.

    Raises:
        AttributeError: for any other name.
    """
    if name == "Policy":
        from xinfer.entrypoints.policy import Policy

        return Policy
    raise AttributeError(name)
