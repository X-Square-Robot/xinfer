"""CUDA kernel selection shared by attention and flow operators."""

_listed = {}  # (probe, device, layout) -> whether xcompute lists a kernel


def takes(probe, key, tensor, *args, **kwargs) -> bool:
    """Whether ``probe(*args, **kwargs)``, an xcompute ``*_backend`` query, finds a kernel (it raises ValueError
    otherwise).

    The first call for each device and layout is eager; captured calls reuse its result.
    """
    key = (probe, tensor.device, key)
    if key not in _listed:
        try:
            probe(tensor, *args, **kwargs)
            _listed[key] = True
        except ValueError:
            _listed[key] = False
    return _listed[key]
