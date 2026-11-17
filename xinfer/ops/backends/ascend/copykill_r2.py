"""COPYKILL-r2 integration flags; default off, no default flip."""
import os

_FAMILIES = ("k2", "k3", "k4", "k5", "k7")
_raw = os.environ.get("XINFER_COPYKILL_R2", "off").strip().lower()
if _raw in ("1", "all", "on"):
    _ENABLED = frozenset(_FAMILIES)
elif _raw in ("", "0", "off"):
    _ENABLED = frozenset()
else:
    _ENABLED = frozenset(x.strip() for x in _raw.split(",") if x.strip())
_unknown = _ENABLED - frozenset(_FAMILIES)
if _unknown:
    raise RuntimeError(f"XINFER_COPYKILL_R2 unknown families: {sorted(_unknown)}")

def enabled(family):
    return family in _ENABLED
