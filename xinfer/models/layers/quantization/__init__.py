"""Quantization methods for linear layers (docs/design.md §12).

A ``QuantConfig`` maps module groups (dot-separated names the model assigns
to its linear layers, e.g. ``"vision.blocks"`` or ``"prefill"``) to methods.
Layers pick their method while they are constructed under ``use_quant``;
after the checkpoint is loaded, ``process_weights(model)`` converts each
layer's weights once (before they are shared or captured).
"""

from xinfer.models.layers.quantization.base import (
    METHODS,
    GatedLayout,
    QKVLayout,
    QuantConfig,
    QuantMethod,
    Unquantized,
    active_quant,
    make_method,
    process_weights,
    use_quant,
)

__all__ = ["METHODS", "GatedLayout", "QKVLayout", "QuantConfig", "QuantMethod", "Unquantized", "active_quant",
           "make_method", "process_weights", "use_quant"]
