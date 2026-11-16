"""Build-time guard for the tested Bisheng packed host launch ABI.

The 9.3.0 host launcher packs argument bytes consecutively, while device entry
loads respect scalar alignment. Reject layouts requiring implicit padding;
an odd number of 32-bit scalars before a 64-bit argument is unsafe. This is
checked against the lowered device entry order, not the original PrimFunc.
"""

_TYPES = {
    "handle": (8, 8),
    "int32": (4, 4),
    "uint32": (4, 4),
    "float32": (4, 4),
    "int64": (8, 8),
    "uint64": (8, 8),
    "float64": (8, 8),
}


def validate_packed_launch_layout(parameters, kernel_name="kernel"):
    """Return a reviewable layout or fail before compiling unsafe parameters.

    ``parameters`` is an ordered iterable of ``(name, TIR dtype string)``.
    Unvalidated scalar widths are rejected explicitly rather than guessed.
    """
    offset = 0
    layout = []
    for name, dtype in parameters:
        if dtype not in _TYPES:
            raise ValueError(
                f"{kernel_name}: unsupported launch parameter {name} dtype {dtype}; "
                "use pointers or validated 32/64-bit scalar arguments"
            )
        size, alignment = _TYPES[dtype]
        if offset % alignment:
            aligned = (offset + alignment - 1) // alignment * alignment
            raise ValueError(
                f"{kernel_name}: unsafe packed launch ABI for {name} ({dtype}): "
                f"host byte offset {offset}, device alignment requires {aligned}. "
                "Use int64 scalar parameters (encode float32 bits in int64 when needed)."
            )
        layout.append(
            {
                "name": name,
                "dtype": dtype,
                "offset": offset,
                "size": size,
                "alignment": alignment,
            }
        )
        offset += size
    return {"parameters": layout, "packed_bytes": offset}
