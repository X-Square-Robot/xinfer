"""Shared-memory buffers carrying prepared requests from a frontend to the engine.

Each frontend process owns a ring of fixed-size shared-memory buffers. It packs
a request's CPU tensors, and the data of its ``DeviceImage`` values, into a free
buffer and sends a small manifest (offsets, shapes, dtypes, host values) over
its pipe; the engine maps the buffer once and builds tensors over it without
copying. On CUDA the engine registers the buffer as pinned host memory, so the
upload reads it directly. A buffer is reused only after its response arrived,
which happens after the upload completed.
"""

from __future__ import annotations

import asyncio
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch

from xinfer.models.interface import DeviceImage

ALIGN = 256


def _dtype(name: str) -> torch.dtype:
    """Map a dtype string from a manifest (``"torch.float32"`` or ``"float32"``) to the ``torch.dtype``."""
    return getattr(torch, name.removeprefix("torch."))


class ShmRing:
    """Frontend side: create, fill and lease buffers.

    Used from the frontend's asyncio event loop. A buffer stays leased from
    ``acquire`` until its request's response arrived (``release``).

    Attributes:
        buffer_bytes: size of each buffer.
        buffers: the ``SharedMemory`` segments, named ``<prefix>-<i>``.
    """

    def __init__(self, prefix: str, count: int, buffer_bytes: int):
        """Create ``count`` shared-memory segments of ``buffer_bytes`` each.

        Args:
            prefix: unique name prefix (``xinfer serve`` uses ``xinfer-<random>-f<frontend index>``).
            count: number of buffers (``shm_buffers``).
            buffer_bytes: bytes per buffer (``--shm-buffer-mib``).

        Raises:
            FileExistsError: a segment with one of the names already exists.
        """
        self.buffer_bytes = buffer_bytes
        self.buffers = [shared_memory.SharedMemory(name=f"{prefix}-{i}", create=True, size=buffer_bytes)
                        for i in range(count)]
        self._free: asyncio.Queue | None = None

    def _queue(self) -> asyncio.Queue:
        """Return the queue of free buffer indices, created on first use (inside the event loop)."""
        if self._free is None:
            self._free = asyncio.Queue()
            for i in range(len(self.buffers)):
                self._free.put_nowait(i)
        return self._free

    async def acquire(self) -> int:
        """Lease a free buffer, waiting while all are leased.

        Returns:
            The buffer index, for ``pack`` and ``release``.
        """
        return await self._queue().get()

    def release(self, index: int) -> None:
        """Return buffer ``index`` to the free queue; call it only after the engine replied."""
        self._queue().put_nowait(index)

    def pack(self, index: int, tensors: dict[str, Any]) -> dict | None:
        """Write tensors into buffer ``index``; return the manifest, or None if they do not fit.

        Tensors, and the data of ``DeviceImage`` values, are copied as contiguous bytes at 256-byte aligned
        offsets; other values go into the manifest as host values.

        Args:
            index: a leased buffer.
            tensors: ``PreparedRequest.tensors``; tensors must have a NumPy-convertible dtype, other values
                must be picklable.

        Returns:
            ``{"buffer": <segment name>, "tensors": {name: entry}}`` with entries ``("t", offset, shape, dtype)``,
            ``("d", offset, shape, dtype, sizes, resize)`` (a ``DeviceImage``) or ``("v", value)``, where
            ``dtype`` is the tensor dtype as a string. ``None`` if the tensors exceed the buffer.
        """
        memory = self.buffers[index].buf
        manifest, offset = {}, 0
        for name, value in tensors.items():
            image = value if isinstance(value, DeviceImage) else None
            if image is not None:
                value = image.data
            if torch.is_tensor(value):
                array = np.ascontiguousarray(value.detach().cpu().numpy())
                offset = (offset + ALIGN - 1) // ALIGN * ALIGN
                if offset + array.nbytes > self.buffer_bytes:
                    return None
                target = np.frombuffer(memory, dtype=np.uint8, count=array.nbytes, offset=offset)
                target[:] = array.view(np.uint8).reshape(-1)
                entry = (offset, tuple(array.shape), str(value.dtype))
                manifest[name] = ("t", *entry) if image is None else ("d", *entry, image.sizes, image.resize)
                offset += array.nbytes
            else:
                manifest[name] = ("v", value)
        return {"buffer": self.buffers[index].name, "tensors": manifest}

    def close(self) -> None:
        """Close and unlink every buffer (frontend shutdown)."""
        for shm in self.buffers:
            shm.close()
            shm.unlink()


class ShmReader:
    """Engine side: map buffers by name and view packed tensors.

    Not thread-safe: ``xinfer serve`` shares one reader among the bridge
    threads under a lock.
    """

    def __init__(self, register_cuda: bool):
        """Create an empty reader.

        Args:
            register_cuda: register each mapped buffer as pinned host memory (``cudaHostRegister``), so
                uploads read it directly.
        """
        self.register_cuda = register_cuda
        self._maps: dict[str, shared_memory.SharedMemory] = {}

    def _map(self, name: str) -> shared_memory.SharedMemory:
        """Attach the segment ``name`` once (and register it with CUDA) and cache it.

        Raises:
            RuntimeError: ``cudaHostRegister`` failed.
            FileNotFoundError: no segment has this name.
        """
        shm = self._maps.get(name)
        if shm is None:
            shm = shared_memory.SharedMemory(name=name)
            if self.register_cuda:
                address = np.frombuffer(shm.buf, dtype=np.uint8).ctypes.data
                status = torch.cuda.cudart().cudaHostRegister(address, shm.size, 0)
                if int(status) != 0:
                    raise RuntimeError(f"cudaHostRegister failed for {name}: {status}")
            self._maps[name] = shm
        return shm

    def close(self) -> None:
        """Unregister and unmap all buffers (tensors over them must be dead)."""
        for shm in self._maps.values():
            if self.register_cuda:
                address = np.frombuffer(shm.buf, dtype=np.uint8).ctypes.data
                torch.cuda.cudart().cudaHostUnregister(address)
            try:
                shm.close()
            except BufferError:
                pass
        self._maps.clear()

    def unpack(self, packed: dict) -> dict[str, Any]:
        """Build a request's tensors over its shared-memory buffer, without copying.

        Args:
            packed: a manifest from ``ShmRing.pack``.

        Returns:
            Name to CPU tensor (a view of the buffer, valid until the frontend reuses the buffer, i.e.
            after the response), ``DeviceImage`` over such a view, or host value.
        """
        shm = self._map(packed["buffer"])
        out = {}
        for name, entry in packed["tensors"].items():
            if entry[0] == "v":
                out[name] = entry[1]
                continue
            kind, offset, shape, dtype = entry[:4]
            torch_dtype = _dtype(dtype)
            count = int(np.prod(shape)) if shape else 1
            tensor = torch.frombuffer(shm.buf, dtype=torch_dtype, count=count, offset=offset).view(shape)
            out[name] = tensor if kind == "t" else DeviceImage(tensor, *entry[4:])
        return out
