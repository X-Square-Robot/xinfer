"""Accelerator graph capture and replay of stage dispatches (docs/design.md §8.3).

Each lane owns one graph memory pool, shared by all graphs captured on that
lane (they run one at a time on the lane's stream). A graph is keyed by the
stage, the batch size, and the shapes, dtypes and host values of the inputs
the stage reads. Which inputs a stage reads is learned by tracking its
accesses during the eager run that precedes every capture; branches in stage
code must depend only on keyed values (shapes, host values, presence), so a
new branch also changes the key.

Static input and output buffers belong to the runtime, not to the model:
for every tensor name a lane keeps one flat buffer sized to the
largest shape seen (with headroom), and each graph uses a contiguous view at
its start; growing a static buffer drops the graphs that pointed into it. A
replay copies the dispatch's inputs into those views, replays, and returns
output views that stay valid until the lane's next replay; the caller copies
or hands them off before then.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from xinfer.models.interface import PreparedRequest, StageBatch, VLAModel
from xinfer.platforms.devices import device_module_for, graph_api_for

logger = logging.getLogger(__name__)

ON_MISS = ("capture", "eager", "error")


def warmup_requests(model: VLAModel, prepared: PreparedRequest) -> list[PreparedRequest]:
    """Warm finite bucket policies, widest first; exact-length requests stay as supplied."""
    variants = model.warmup_requests(prepared)
    widths = {}
    for name, spec in model.shape_buckets().items():
        if not spec.pads:
            continue
        expanded = []
        for variant in variants:
            length_range = variant.meta.get(f"{name}_len_range")
            if length_range is None:
                expanded.append(variant)
                continue
            selected = spec.widths(*length_range)
            widths.setdefault(name, set()).update(selected)
            expanded.extend(model.pad(variant, name, width) for width in selected)
        variants = expanded
    if widths:
        logger.info("warmup bucket widths: %s", {name: sorted(values, reverse=True) for name, values in widths.items()})
    return variants


@dataclass(frozen=True)
class CUDAGraphConfig:
    """How a lane uses CUDA graphs.

    ``max_graphs`` bounds the graphs kept per lane (least recently used are
    dropped). ``on_miss_after_warmup`` applies to keys first seen after ``finish_warmup()``:
    capture them, run them eagerly, or fail.

    Attributes:
        max_graphs: graphs kept per lane; capturing beyond it evicts the least recently used.
        on_miss_after_warmup: ``"capture"`` (capture and log a warning), ``"eager"`` (run without a graph) or
            ``"error"`` (raise ``CUDAGraphMiss``); any other value raises ``ValueError`` at construction.
    """

    max_graphs: int = 64
    on_miss_after_warmup: str = "capture"

    def __post_init__(self):
        if self.on_miss_after_warmup not in ON_MISS:
            raise ValueError(f"on_miss_after_warmup must be one of {ON_MISS}")


class CUDAGraphMiss(RuntimeError):
    """Raised by ``LaneCUDAGraphs.run`` for a dispatch without a graph after warmup when the policy is ``"error"``."""

    pass


class _ReadTracker(Mapping):
    """Records which input names a stage looks up (present or not).

    Wraps a stage's inputs during an eager run. Every lookup (``[]``, ``get``, and ``items()``/``values()``, which
    look up each name) and every ``in`` test adds the name to ``names``; iterating over the names alone and ``len``
    do not.

    Attributes:
        names: the names looked up so far.
    """

    def __init__(self, data: Mapping[str, Any]):
        """Wrap ``data`` (not copied)."""
        self._data = data
        self.names: set[str] = set()

    def __getitem__(self, name):
        self.names.add(name)
        return self._data[name]

    def __contains__(self, name):
        self.names.add(name)
        return name in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _freeze(value: Any):
    """Make a host input value hashable for a graph key.

    Lists and tuples become tuples and dicts become sorted ``(key, value)`` tuples, recursively; other values are
    returned unchanged (so they must be hashable themselves).
    """
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


@dataclass
class _Captured:
    """One captured dispatch.

    Attributes:
        graph: the ``torch.cuda.CUDAGraph``.
        inputs: static input view per tensor input the stage read; ``run`` copies the inputs into them.
        host: non-tensor inputs the stage read, as given at capture (they are part of the graph key).
        out: static output view per declared output; the graph writes into them.
        buffers: ``(kind, name, dtype)`` keys of the static buffers the graph points into.
        capture_ms: wall time of the capture, eager pre-run included.
    """

    graph: Any
    inputs: dict[str, torch.Tensor]  # name -> static view
    host: dict[str, Any]
    out: dict[str, torch.Tensor]
    buffers: frozenset
    capture_ms: float


class LaneCUDAGraphs:
    """Graphs of one lane: a shared pool, per-name static buffers and an LRU of captures.

    Not thread-safe; only the engine thread uses it. See the module docstring for the graph keys and
    the static buffers.

    Attributes:
        model: the model whose stages are captured.
        stream: the lane's stream; eager runs and captures run on it, input copies and replays on the caller's
            current stream.
        device: device of the static buffers and of eagerly allocated outputs.
        policy: the ``CUDAGraphConfig``.
        pool: graph memory pool handle shared by all graphs of this lane.
        warmed_up: set by ``finish_warmup``; misses afterwards follow ``policy.on_miss_after_warmup``.
        warmup_inputs: ``{stage: [inputs, ...]}`` of the graphs captured at batch size 1 before ``finish_warmup``,
            for ``capture_batch_sizes``; cleared by ``finish_warmup``.
        stats: counters ``captures`` (of which ``captures_after_warmup``), ``replays``, ``eager``, ``evictions`` (LRU
            and static-buffer growth) and ``capture_ms`` (total capture time, eager pre-runs included).
    """

    def __init__(self, model: VLAModel, stream: Any, device: str, policy: CUDAGraphConfig,
                 reads: dict[str, set[str]] | None = None):
        """Create an empty graph cache with its own graph memory pool.

        Args:
            model: the model whose stages are captured.
            stream: the lane's stream.
            device: device for static buffers and eager outputs.
            policy: graph settings.
            reads: ``{stage: input names}`` each stage reads; shared by an engine's lanes (and its batch keys) and
                updated by every capture. A new dict when None.
        """
        self.model = model
        self.stream = stream
        self.device = device
        self.policy = policy
        # Vendor-agnostic handle (CUDA, NPU) delivered by ``graph_api_for``: its ``graph_pool_handle`` /
        # ``CUDAGraph`` / ``graph(...)`` / ``stream(...)`` / ``synchronize`` match torch.cuda's surface.
        # A None from the factory means the engine never calls into it (CPU path uses no graph).
        self._device_module = device_module_for(device)
        self._graph_api = graph_api_for(device)
        device_context = (self._device_module.device(device) if str(device).partition(":")[0] == "npu"
                          else contextlib.nullcontext())
        with device_context:
            self.pool = self._graph_api.graph_pool_handle()
        self._static_buffers: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
        self._graphs: OrderedDict[tuple, _Captured] = OrderedDict()
        # Input names each stage reads; may be shared by the lanes of an engine.
        self._reads: dict[str, set[str]] = reads if reads is not None else {}
        self.warmed_up = False
        self.warmup_inputs: dict[str, list[Mapping[str, Any]]] = {}
        self.stats = dict(captures=0, captures_after_warmup=0, replays=0, eager=0, evictions=0, capture_ms=0.0)

    # -------------------------------------------------------------- keys
    def _key(self, stage: str, batch_size: int, inputs: Mapping[str, Any]) -> tuple:
        """Graph key of a dispatch.

        Looking up a tensor of a lazily merged ``batching.MergedInputs`` performs its merge.

        Args:
            stage: stage name.
            batch_size: number of requests in the batch.
            inputs: the dispatch's inputs.

        Returns:
            ``(stage, batch_size, parts)``, where ``parts`` has one entry per input name the stage reads, sorted
            by name: ``(name, shape, dtype)`` for a tensor (the full shape, so each padded length gets its own graph),
            ``(name, frozen value)`` for a host value and ``(name, None)`` for an absent name.
        """
        parts = []
        for name in sorted(self._reads.get(stage, ())):
            if name not in inputs:
                parts.append((name, None))
                continue
            value = inputs[name]
            if torch.is_tensor(value):
                parts.append((name, tuple(value.shape), value.dtype))
            else:
                parts.append((name, _freeze(value)))
        return (stage, batch_size, tuple(parts))

    def known(self, stage: str, batch_size: int, inputs: Mapping[str, Any]) -> bool:
        """Whether a graph is captured for this dispatch (False while the stage's reads are unknown)."""
        return stage in self._reads and self._key(stage, batch_size, inputs) in self._graphs

    # ------------------------------------------------------------ static buffers
    def _static_view(self, kind: str, name: str, shape, dtype) -> torch.Tensor:
        """View of ``shape`` at the start of the lane's flat static buffer for ``(kind, name, dtype)``.

        The buffer is allocated uninitialized on ``device`` when first needed. When it is too small it is replaced
        by a new one of ``max(numel, 1.5 * old numel)`` elements (contents not kept), and every graph pointing into
        the old buffer is dropped (counted in ``stats["evictions"]``).

        Args:
            kind: ``"in"`` or ``"out"``.
            name: input or output name.
            shape: shape of the view.
            dtype: element type.

        Returns:
            A contiguous view of the current buffer.
        """
        numel = 1
        for d in shape:
            numel *= d
        key = (kind, name, dtype)
        buffer = self._static_buffers.get(key)
        if buffer is None or buffer.numel() < numel:
            if buffer is not None:  # graphs that point into the old buffer are invalid
                for k in [k for k, c in self._graphs.items() if key in c.buffers]:
                    del self._graphs[k]
                    self.stats["evictions"] += 1
            capacity = max(numel, 1) if buffer is None else max(numel, buffer.numel() * 3 // 2)
            buffer = torch.empty(capacity, dtype=dtype, device=self.device)
            self._static_buffers[key] = buffer
        return buffer[:numel].view(shape)

    # ------------------------------------------------------------ run
    def run(self, stage: str, batch_size: int, inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Execute one dispatch on this lane's stream; returns output views.

        Call it with the lane's stream current (the engine does so in ``Lane.use_stream()``): input copies and
        replays are enqueued on the current stream.

        A known key is replayed after each tensor input the graph reads is copied into its static view. An unknown
        key is first captured (``_capture``) and stored, evicting the least recently used graphs beyond
        ``max_graphs``. After ``finish_warmup`` an unknown key is instead captured with a warning, run eagerly or
        rejected, per ``on_miss_after_warmup``.

        Args:
            stage: stage name.
            batch_size: number of requests in the batch (``StageBatch.batch_size``).
            inputs: the dispatch's merged inputs (device tensors and host values).

        Returns:
            ``{name: tensor}`` for every declared output. After a replay these are the lane's static output views,
            which the lane's next replay overwrites: read or copy them on the lane's stream before that. An eager
            run returns new tensors.

        Raises:
            CUDAGraphMiss: unknown key after warmup with ``on_miss_after_warmup="error"``.
        """
        stage_function = self.model.stage_fn(self.model.stage_graph[stage])
        key = self._key(stage, batch_size, inputs) if stage in self._reads else None
        captured = self._graphs.get(key) if key is not None else None
        if captured is None:
            if self.warmed_up and self.policy.on_miss_after_warmup != "capture":
                if self.policy.on_miss_after_warmup == "error":
                    raise CUDAGraphMiss(f"no graph for {stage} {key[2] if key is not None else ()}")
                self.stats["eager"] += 1
                return self._eager(stage, batch_size, inputs, stage_function)
            if self.warmed_up:
                logger.warning("capturing a graph after warmup for %s: %s", stage, key[2] if key is not None else ())
                self.stats["captures_after_warmup"] += 1
            captured = self._capture(stage, batch_size, inputs, stage_function)
            key = self._key(stage, batch_size, inputs)  # reads may have grown during capture
            if not self.warmed_up and batch_size == 1:
                self.warmup_inputs.setdefault(stage, []).append(inputs)
            self._graphs[key] = captured
            while len(self._graphs) > self.policy.max_graphs:
                self._graphs.popitem(last=False)
                self.stats["evictions"] += 1
        else:
            self._graphs.move_to_end(key)
        for name, view in captured.inputs.items():
            source = inputs[name]
            if source.data_ptr() != view.data_ptr():
                view.copy_(source, non_blocking=True)
        captured.graph.replay()
        self.stats["replays"] += 1
        return captured.out

    def _alloc_outputs(self, stage, batch, *, static: bool) -> dict[str, torch.Tensor]:
        """Allocate the outputs ``model.stage_outputs`` declares for ``batch``.

        Args:
            stage: stage name.
            batch: the dispatch's ``StageBatch`` (inputs set, ``out`` empty).
            static: return the lane's static output views (for a capture) instead of new tensors on ``device``.

        Returns:
            ``{name: tensor}`` with the declared shapes and dtypes; contents are not initialized.
        """
        specs = self.model.stage_outputs(stage, batch)
        if static:
            return {n: self._static_view("out", n, s.shape, s.dtype) for n, s in specs.items()}
        return {n: torch.empty(s.shape, dtype=s.dtype, device=self.device) for n, s in specs.items()}

    def _eager(self, stage, batch_size, inputs, stage_function) -> dict[str, torch.Tensor]:
        """Run the stage eagerly on the lane's stream into new output tensors (no graph, no read tracking).

        Args:
            stage: stage name.
            batch_size: number of requests in the batch.
            inputs: the dispatch's inputs.
            stage_function: the model's stage method.

        Returns:
            ``{name: tensor}`` with the stage's outputs.
        """
        batch = StageBatch(stage, batch_size, MappingProxyType(dict(inputs)))
        batch.out = self._alloc_outputs(stage, batch, static=False)
        with self._device_module.stream(self.stream), torch.inference_mode():
            stage_function(batch)
        return batch.out

    def _capture(self, stage, batch_size, inputs, stage_function) -> _Captured:
        """Capture a graph for one dispatch; the caller replays it.

        The stage first runs eagerly on the real inputs through a ``_ReadTracker``: this learns which inputs this
        branch reads (added to the shared reads) and does lazy initialization (handles, workspaces) outside the
        capture. Each tensor input read then gets a static input view filled with its value, the outputs get static
        views, and the stage is captured on the lane's stream into the lane's pool. Host values read are fixed in
        the graph. The eager outputs are discarded, so a captured dispatch runs the stage twice.

        Args:
            stage: stage name.
            batch_size: number of requests in the batch.
            inputs: the dispatch's inputs.
            stage_function: the model's stage method.

        Returns:
            The ``_Captured`` graph with its static views and the static buffers it uses.
        """
        start = time.perf_counter()
        # Eager run on the real inputs: learns which inputs this branch reads
        # and does lazy initialization (handles, workspaces) outside capture.
        tracked = _ReadTracker(inputs)
        batch = StageBatch(stage, batch_size, tracked)
        batch.out = self._alloc_outputs(stage, batch, static=False)
        with self._device_module.stream(self.stream), torch.inference_mode():
            stage_function(batch)
        self._reads.setdefault(stage, set()).update(tracked.names)
        static_in, host, used = {}, {}, set()
        for name in tracked.names:
            if name not in inputs:
                continue
            value = inputs[name]
            if torch.is_tensor(value):
                static_in[name] = self._static_view("in", name, tuple(value.shape), value.dtype)
                used.add(("in", name, value.dtype))
            else:
                host[name] = value
        view_inputs = dict(host)
        view_inputs.update(static_in)
        batch = StageBatch(stage, batch_size, MappingProxyType(view_inputs))
        batch.out = self._alloc_outputs(stage, batch, static=True)
        used.update(("out", n, t.dtype) for n, t in batch.out.items())
        with self._device_module.stream(self.stream):
            for name, view in static_in.items():
                view.copy_(inputs[name], non_blocking=True)
        graph = self._graph_api.CUDAGraph()
        with torch.inference_mode(), self._graph_api.graph(graph, pool=self.pool, stream=self.stream,
                                                           capture_error_mode="thread_local"):
            stage_function(batch)
        elapsed = (time.perf_counter() - start) * 1000
        self.stats["captures"] += 1
        self.stats["capture_ms"] += elapsed
        return _Captured(graph, static_in, host, batch.out, frozenset(used), elapsed)

    def finish_warmup(self) -> None:
        """Mark the warmup as done: later unknown keys follow ``policy.on_miss_after_warmup``."""
        self.warmed_up = True
        self.warmup_inputs.clear()

    def graph_count(self) -> int:
        return len(self._graphs)


def capture_batch_sizes(lanes: Mapping[str, list], batch_sizes: Mapping[str, list[int]], merge, *,
                        device: str = "cuda") -> None:
    """Capture the inputs recorded during warmup on every lane of their stage at every allowed batch size.

    The recorded inputs are repeated to fill each batch and merged by ``merge``, as serving merges a batch, so the
    graph keys are the serving keys. A batch of mixed ragged lengths is padded to its longest member, so it uses the
    graph of a warmed length. The largest inputs go first: a capture that grows a lane's static buffer drops the
    graphs that use the old one.

    Args:
        lanes: ``{stage: [lane, ...]}``; each lane has ``cudagraphs`` (a ``LaneCUDAGraphs``) and ``use_stream()``.
        batch_sizes: ``{stage: allowed batch sizes}``.
        merge: ``merge(stage, [inputs, ...])`` builds one batch's inputs from per-request inputs.
        device: device to synchronize after capture.
    """
    tasks = []
    for stage, stage_lanes in lanes.items():
        for lane in stage_lanes:
            for inputs in lane.cudagraphs.warmup_inputs.get(stage, []):
                footprint = sum(v.numel() * v.element_size() for v in inputs.values() if torch.is_tensor(v))
                tasks += [(footprint * batch_size, stage, batch_size, inputs) for batch_size in batch_sizes[stage]]
    tasks.sort(key=lambda task: task[0], reverse=True)
    for _, stage, batch_size, inputs in tasks:
        for lane in lanes[stage]:
            with lane.use_stream(), torch.inference_mode():
                batch = merge(stage, [inputs] * batch_size)
                if not lane.cudagraphs.known(stage, batch_size, batch):
                    lane.cudagraphs.run(stage, batch_size, batch)
    # Wait for the vendor runtime to finish capture work on all lanes used above.
    # No CUDA graph vocabulary leaks beyond ``graph_api_for``; lane._device_module (torch.npu etc.)
    # answers the same ``synchronize`` surface as torch.cuda.
    module = next((lane._device_module for stage_lanes in lanes.values() for lane in stage_lanes
                   if lane._device_module is not None), None)
    if module is not None:
        module.synchronize()

