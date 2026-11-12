"""Device dispatch for the ``wallx2b`` registry entry.

``wallx2b`` names two serving lines of the Wall-X 2B family: the GPU baseline
(``Wallx2b``, this package) and the wallx2b RPU TP8 profile (``WallX2B`` in
``rpu/``). The registry entry resolves to this dispatcher and the device picks
the implementation: ``rpu``/``rpu:0`` loads the RPU profile, anything else the
GPU model.
"""

from __future__ import annotations


class Wallx2bServing:
    """The ``wallx2b`` registry entry; ``from_checkpoint`` dispatches on the device."""

    @classmethod
    def from_checkpoint(cls, path, *, device, **options):
        """Load ``wallx2b`` from a checkpoint onto ``device``.

        Args:
            path: checkpoint directory.
            device: ``rpu``/``rpu:0`` selects the RPU profile; other devices the GPU model.
            options: forwarded to the selected implementation unchanged.

        Returns:
            The loaded model (``WallX2B`` on RPU, ``Wallx2b`` otherwise).
        """
        if isinstance(device, str) and device.split(":", 1)[0] == "rpu":
            from xinfer.models.wallx2b.rpu.model import WallX2B

            return WallX2B.from_checkpoint(path, device=device, **options)
        from xinfer.models.wallx2b.model import Wallx2b

        return Wallx2b.from_checkpoint(path, device=device, **options)
