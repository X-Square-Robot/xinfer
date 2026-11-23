"""Layout and scratch-buffer reuse selected by the Wall-OSS recipe."""

import os


_FAMILIES = ("contiguous_qkv", "cache_fp32_bias", "fuse_bias_cast", "reuse_query_buffer")


def _selected(value):
    value = value.strip().lower()
    if value in ("1", "all", "on"):
        return frozenset(_FAMILIES)
    if value in ("", "0", "off"):
        return frozenset()
    families = frozenset(item.strip() for item in value.split(",") if item.strip())
    unknown = families.difference(_FAMILIES)
    if unknown:
        raise RuntimeError(f"XINFER_ASCEND_LAYOUT_REUSE unknown families: {sorted(unknown)}")
    return families


_ENABLED = _selected(os.environ.get("XINFER_ASCEND_LAYOUT_REUSE", "off"))


def enabled(family):
    return family in _ENABLED
