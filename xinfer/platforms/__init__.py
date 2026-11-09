from xinfer.platforms.interface import Platform, PlatformCapabilities

__all__ = ["Platform", "PlatformCapabilities", "resolve_platform"]


def resolve_platform(device=None):
    from xinfer.platforms.detect import resolve_platform as _resolve

    return _resolve(device)
