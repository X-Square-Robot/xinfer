"""Wire encoding: msgpack with numpy arrays in the msgpack-numpy layout.

The x2robot clients patch msgpack with ``msgpack_numpy``; arrays travel as
maps ``{nd: True, type, kind, shape, data}``. Using the same hooks keeps the
wire format of the existing robot service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy

# ``status`` of every response; error responses also carry ``error`` (a message).
OK = "ok"
INVALID_REQUEST = "invalid_request"  # the request could not be decoded or prepared
QUEUE_FULL = "queue_full"  # the engine's admission queue is full; retry later
PER_ROBOT_LIMIT = "per_robot_limit"  # the robot already has its maximum of requests in flight
UNAVAILABLE = "unavailable"  # the server is shutting down
TIMEOUT = "timeout"  # the engine did not finish the request within --request-timeout
INTERNAL_ERROR = "internal_error"  # the server failed while serving the request
# Engine admission refusals (``AdmitError.code``) as response statuses.
ADMIT_STATUS = {"queue_full": QUEUE_FULL, "per_robot_limit": PER_ROBOT_LIMIT, "stopped": UNAVAILABLE}


def packb(obj: Any) -> bytes:
    """Encode ``obj`` as msgpack; NumPy arrays use the msgpack-numpy layout, bytes the bin type.

    Args:
        obj: a msgpack-encodable object, possibly containing NumPy arrays.

    Returns:
        The encoded message.
    """
    return msgpack.packb(obj, default=msgpack_numpy.encode, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    """Decode a msgpack message, restoring msgpack-numpy arrays as ``np.ndarray`` and strings as ``str``.

    Args:
        data: one encoded message.

    Returns:
        The decoded object.
    """
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


def read_requests(path) -> list[dict]:
    """The requests in a request file (``xinfer-serve --warmup``, the benchmarks).

    A file is JSON when its name ends in ``.json`` and msgpack otherwise, and holds one request map or a list of
    them. JSON keeps the requests readable; msgpack can also hold NumPy arrays (msgpack-numpy), such as raw views.

    Args:
        path: the request file.

    Returns:
        Its requests, in order.
    """
    data = Path(path).read_bytes()
    value = json.loads(data) if str(path).endswith(".json") else unpackb(data)
    return value if isinstance(value, list) else [value]
