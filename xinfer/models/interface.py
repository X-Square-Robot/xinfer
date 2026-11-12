"""The contract between a model and the runtime (docs/design.md §11).

A model declares its stages (``StageGraph``), what crosses each stage boundary
(``HandoffSpec``), what it supports (``Capabilities``) and its wire-level
inputs and outputs (``IOSchema``). It provides CPU input/output processors
and one forward method per stage. The runtime decides where and when stages
run; a model never sees processes, streams or scheduler state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class BucketSpec:
    """Pad lengths to a grid or to a list of widths; neither keeps the exact length.

    ``max_length`` caps a grid width after the model has truncated its input. ``sizes`` lists the only widths
    a model takes, ascending (its kernels accept those shapes only, for example): a length pads to the smallest
    that holds it, and a longer one is rejected.
    """

    granularity: int = 0
    max_length: int | None = None
    sizes: tuple[int, ...] = ()

    def __post_init__(self):
        for name, value, minimum in (("granularity", self.granularity, 0),
                                     ("max_length", self.max_length, 1)):
            if value is None and name != "granularity":
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        if self.sizes:
            if self.granularity or self.max_length is not None:
                raise ValueError("sizes cannot be combined with granularity or max_length")
            if any(type(s) is not int or s < 1 for s in self.sizes) or list(self.sizes) != sorted(set(self.sizes)):
                raise ValueError(f"sizes must be increasing positive integers, got {self.sizes!r}")

    @property
    def pads(self) -> bool:
        """Whether lengths are padded (a grid or a list of widths)."""
        return bool(self.granularity or self.sizes)

    def width(self, valid: int) -> int:
        """The physical width for a valid length; tensor padding stays in the model.

        Raises:
            ValueError: ``valid`` exceeds the largest of ``sizes``.
        """
        if self.sizes:
            for size in self.sizes:
                if size >= valid:
                    return size
            raise ValueError(f"length {valid} exceeds the largest width {self.sizes[-1]}")
        width = -(-valid // self.granularity) * self.granularity if self.granularity else valid
        return min(width, self.max_length) if self.max_length is not None else width

    def widths(self, shortest: int, longest: int) -> list[int]:
        """Distinct widths for an inclusive length range, largest first for warmup."""
        if shortest < 1 or longest < shortest:
            raise ValueError(f"invalid length range: {(shortest, longest)}")
        if self.sizes:
            longest = min(longest, self.sizes[-1])  # longer lengths are rejected, so there is nothing to warm
        return sorted({self.width(n) for n in range(shortest, longest + 1)}, reverse=True)


# --------------------------------------------------------------------------
# Stage graph
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """A unit of model computation, run by the runtime as one dispatch.

    The runtime calls the model method named ``method`` with a ``StageBatch``
    once per dispatch (see ``VLAModel`` for the stage-method contract). A
    stage is queued for a request once every stage in ``depends_on`` has been
    dispatched for it; the lane then waits for their events on the GPU.

    Attributes:
        name: stage name, unique in the graph; used in settings (``lanes.<name>``, ``batch.<name>``),
            ``Capabilities`` and ``HandoffSpec``.
        method: name of the model method implementing the stage (resolved by ``VLAModel.stage_fn``).
        depends_on: names of the stages this one depends on; the constructor also accepts one name as a ``str``.
    """

    name: str
    method: str  # name of the model method implementing the stage
    depends_on: tuple[str, ...] = ()

    def __init__(self, name: str, method: str, depends_on: str | tuple[str, ...] = ()):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "depends_on", (depends_on,) if isinstance(depends_on, str) else tuple(depends_on))


class StageGraph:
    """A DAG of stages with exactly one terminal stage.

    Iterating a graph (and ``len``) follows ``order``; ``graph[name]`` returns
    the ``Stage`` called ``name``. The terminal stage has no consumers; its
    outputs are returned to the caller. Process placement is not part of the graph.
    """

    def __init__(self, *stages: Stage):
        """Validate the stages and compute their execution order.

        Args:
            *stages: the model's stages; declaration order breaks ties in ``order``.

        Raises:
            ValueError: no stages, duplicate names, a dependency on an unknown stage, a cycle,
                or more than one terminal stage.
        """
        if not stages:
            raise ValueError("a stage graph needs at least one stage")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")
        known = set(names)
        for s in stages:
            missing = set(s.depends_on) - known
            if missing:
                raise ValueError(f"stage {s.name!r} depends on unknown stages {sorted(missing)}")
        self._stages = {s.name: s for s in stages}
        self._order = self._topological(stages)
        terminal_stages = [s.name for s in stages if not self.consumers(s.name)]
        if len(terminal_stages) != 1:
            raise ValueError("a stage graph requires exactly one terminal stage")

    @staticmethod
    def _topological(stages) -> tuple[Stage, ...]:
        """Order ``stages`` so that every stage follows its dependencies.

        Each pass emits all stages whose dependencies are already emitted, in
        declaration order.

        Args:
            stages: the validated stages (unique names, known dependencies).

        Returns:
            The stages in execution order.

        Raises:
            ValueError: the dependencies form a cycle.
        """
        remaining = {s.name: set(s.depends_on) for s in stages}
        by_name = {s.name: s for s in stages}
        order = []
        while remaining:
            ready = [n for n in (s.name for s in stages) if n in remaining and not remaining[n]]
            if not ready:
                raise ValueError(f"stage graph has a cycle among {sorted(remaining)}")
            for name in ready:
                order.append(by_name[name])
                del remaining[name]
                for deps in remaining.values():
                    deps.discard(name)
        return tuple(order)

    @property
    def order(self) -> tuple[Stage, ...]:
        """Stages in a valid execution order."""
        return self._order

    def __getitem__(self, name: str) -> Stage:
        return self._stages[name]

    def __iter__(self):
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def consumers(self, name: str) -> tuple[str, ...]:
        """Names of the stages that list ``name`` in their ``depends_on``, in execution order.

        Args:
            name: a stage name.

        Returns:
            The direct downstream stages; empty for a final stage.
        """
        return tuple(s.name for s in self._order if name in s.depends_on)


# --------------------------------------------------------------------------
# Handoff, capabilities, IO schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """Layout of one tensor crossing a stage boundary.

    The runtime uses the axes to gather requests into a batch and split them
    again; it does not interpret the values.

    When requests are batched, the consumer receives the tensors concatenated
    along ``batch_axis``, and the producer's output is split back into
    per-request rows along the same axis. With ``seq_axis`` set, rows may
    differ in length along that axis: the batch left-pads every row with
    zeros to the longest, and the consumer also receives ``<name>_lengths``
    (``int32 [B]``, each row's own length), at batch size 1 too.

    Attributes:
        batch_axis: axis that indexes requests (negative values count from the end).
        seq_axis: variable-length axis, or ``None`` when all requests must agree on every non-batch axis.
    """

    batch_axis: int = 0
    seq_axis: int | None = None  # variable-length axis, if rows may differ in length


@dataclass(frozen=True)
class HandoffSpec:
    """Tensors produced by one stage and read by another.

    The engine checks at construction that both stages exist and that
    ``consumer`` depends on ``producer`` (directly or transitively). The
    producer's ``stage_outputs`` must declare every named tensor. After each
    producer dispatch the runtime copies each request's row of these tensors
    into request-owned device tensors (other outputs of a non-final stage are
    not kept), and the consumer finds them in ``StageBatch.inputs`` under the
    same names. A handoff name must not equal a prepared input name.

    Attributes:
        producer: name of the stage that writes the tensors.
        consumer: name of the stage that reads them.
        tensors: tensor name to its batch layout.
    """

    producer: str
    consumer: str
    tensors: Mapping[str, TensorSpec]


@dataclass(frozen=True)
class Capabilities:
    """What a built model supports; checked when a service is built.

    ``xinfer.config.resolve`` turns ``batch_sizes`` into the ``batch_sizes.<stage>``
    settings; it and the engine call ``VLAModel.capabilities()`` once at startup.

    Attributes:
        batch_sizes: per stage, the batch sizes its method accepts; the stage batches up to the largest by
            default. Size 1 is always allowed; a stage missing from the mapping is treated as B1 only. A model
            declares sizes above 1 only for a stage whose batched results pass its precision gate.
        num_steps: denoising steps of the action stage, as declared by the model (the v1 runtime
            does not read it).
        rtc: whether the model accepts real-time-chunking requests.
        rtc_delay_range: inclusive range of accepted RTC inference delays, when ``rtc`` is True.
        stage_ops: per stage, the operators its method calls. ``Policy`` and ``xinfer-serve`` warn at startup
            when one of them has no backend kernel loaded and runs on a fused operator's torch reference.
    """

    batch_sizes: Mapping[str, tuple[int, ...]]
    num_steps: int
    rtc: bool = False
    rtc_delay_range: tuple[int, int] | None = None  # inclusive, when rtc is True
    # Operators each stage calls, for the startup check of slow fallbacks.
    stage_ops: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraSpec:
    """One camera view a request may carry.

    Attributes:
        name: wire key of the image in the request.
        required: whether a request must include this view.
    """

    name: str
    required: bool = True


@dataclass(frozen=True)
class IOSchema:
    """Wire-level inputs and outputs of a model, declared for request validation.

    The v1 runtime does not validate requests against it yet; ``Policy``
    exposes it as ``Policy.io_schema``.

    Attributes:
        cameras: the camera views a request may carry.
        state_keys: request keys holding robot state.
        action_keys: response keys holding actions.
        action_dim: width of one action row.
        action_horizon: action rows per response.
        instruction: whether a request carries a language instruction.
        extra: model-specific schema entries.
    """

    cameras: tuple[CameraSpec, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    action_dim: int
    action_horizon: int
    instruction: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Requests and processors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceImage:
    """A camera view that the engine decodes and resizes on the CUDA device before the request's first stage.

    An InputProcessor puts it into ``PreparedRequest.tensors`` under the name the stages read (``image_<i>``),
    in place of the resized ``[h, w, 3]`` uint8 view; the stages see that view, never this value.

    Attributes:
        data: uint8 CPU tensor: the JPEG bytes ``[n]`` (decoded with nvJPEG), or an RGB frame ``[H, W, 3]``.
        sizes: ``(h, w)`` after each resize step; the last is the view's size.
        resize: per step, ``(mode, antialias)`` of ``torch.nn.functional.interpolate``. A step whose size differs
            from its input's runs in FP32 and stores uint8, rounding half up; a step of the same size is skipped.
        prefetch: engine-internal mutable slot of this view's in-flight device decode (NPU per-view pipelining):
            ``device_image`` claims and submits it when this process's engine bound its device-image queue, and the
            device-image thread fills the view (or the exception the decode raised, and the device it decoded for)
            ahead of the request's admission item. ``None`` (frontend processes, CPU paths, before the engine
            starts) leaves the engine decoding the view at admission as before.
    """

    data: Any
    sizes: tuple[tuple[int, int], ...]
    resize: tuple[tuple[str, bool], ...]
    prefetch: Any = None


# The engine's per-view hand-off hook (installed by ``xinfer.engine.core`` from
# ``xinfer.engine.image_prefetch`` when an engine binds this process's device-image queue):
# a callable taking a frozen ``DeviceImage`` and returning it with its ``prefetch`` slot claimed
# and the ticket submitted. ``None`` (frontend processes, CPU-only tools, before the engine
# starts) leaves the engine decoding each view at admission as before. This slot lives on the
# interface module so the models and the engine share it without either importing the other's
# internals (tests/unit/test_layering.py).
_prefetch_hook = None


@dataclass
class PreparedRequest:
    """CPU-side model inputs produced by an InputProcessor.

    ``tensors`` holds CPU tensors that the runtime moves to the device once per
    request. ``meta`` carries host values the OutputProcessor needs (for
    example the reference pose for relative actions).

    Attributes:
        tensors: the request's stage inputs. CPU tensors are uploaded at admission (non-blocking, on a
            separate stream on CUDA); a ``DeviceImage`` becomes its view on the device before the request's
            first stage; other values (ints, bools, lists) are host values passed to the
            stages unchanged. Each tensor holds one request along its batch axis (``VLAModel.batch_axes``,
            default 0). The shapes and host values a stage reads select its CUDA graph and decide which
            requests share a batch. Under ``xinfer serve`` they cross from the frontend through shared
            memory, so tensors need a NumPy-convertible dtype and host values must be picklable.
        meta: host values for the OutputProcessor, plus ``<dimension>_len_range`` for bucket warmup.
            Only warmup reads these ranges; during serving, metadata stays in the frontend process.
    """

    tensors: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class InputProcessor(Protocol):
    """Turns one wire request into model inputs on the CPU.

    It runs in a frontend process under ``xinfer serve`` (and in the engine
    process for ``--warmup`` files), or in the calling thread under ``Policy``.
    It must not use the GPU: frontend processes hide all CUDA devices. A
    frontend calls it from several processing threads at once, one request per
    call, so it must be safe to call concurrently; shared state such as a noise
    generator is then used in the order the calls reach it.

    A processor that keeps state per robot connection also has
    ``connect()``, which returns the processor of one connection; the entry points take it through ``connect``.
    A connection's requests come one at a time.
    """

    def __call__(self, observation: Mapping[str, Any]) -> PreparedRequest:
        """Decode, resize, tokenize and normalize one request.

        Args:
            observation: the decoded msgpack request (arrays as NumPy arrays).

        Returns:
            The request's ``PreparedRequest``.
        """
        ...


def connect(processor: InputProcessor) -> InputProcessor:
    """The InputProcessor of one robot connection: ``processor.connect()`` for a processor with per-connection
    state, else ``processor``. ``xinfer serve`` calls it for every WebSocket connection, ``Policy`` once."""
    method = getattr(processor, "connect", None)
    return processor if method is None else method()


class OutputProcessor(Protocol):
    """Builds the wire response from one request's raw model outputs, on the CPU.

    It runs where the matching ``InputProcessor`` ran (a frontend processing
    thread or the ``Policy`` caller) and, like it, must be safe to call concurrently.
    """

    def __call__(self, outputs: Mapping[str, Any], request: PreparedRequest) -> dict[str, Any]:
        """Denormalize and convert the actions into the response.

        Args:
            outputs: every final-stage output, restricted to this request's row (the batch axis is kept,
                with length 1). NumPy arrays under ``xinfer serve``; CPU tensors under ``Policy``.
            request: the ``PreparedRequest`` the InputProcessor returned for this request, with its ``meta``.

        Returns:
            The response fields. Under ``xinfer serve`` they are msgpack-encoded (NumPy arrays allowed)
            after the frontend adds ``server_timing``.
        """
        ...


@dataclass(frozen=True)
class OutputSpec:
    """Shape and dtype of one stage output, declared before the stage runs.

    The runtime allocates an uninitialized device tensor (``torch.empty``) of
    this shape and dtype, or a view of a lane's static buffer under CUDA
    graphs; the stage must write every element it relies on.

    Attributes:
        shape: full output shape for the dispatch, including the batch axis when the output has one.
        dtype: a ``torch.dtype``.
    """

    shape: tuple[int, ...]
    dtype: Any  # torch.dtype


@dataclass
class StageBatch:
    """One stage dispatch: its inputs and the runtime-owned output buffers.

    ``inputs`` maps names to device tensors: the request's prepared tensors and
    the outputs of upstream stages. ``out`` maps each output the model declared
    in ``stage_outputs`` to a buffer the runtime allocated; the stage writes
    its results into these buffers and returns nothing. The runtime decides
    where the buffers live (per request, or static buffers reused by a
    captured CUDA graph), so stage code is the same in both cases.

    For ``batch_size`` > 1 the requests' tensors are concatenated along their batch
    axes (``TensorSpec.batch_axis`` for handoffs, ``VLAModel.batch_axes`` for
    prepared inputs, default 0). All requests in a batch have equal shapes
    (apart from the batch axis and declared ``seq_axis``) and equal host
    values for every input the stage reads; inputs the requests cannot share
    are absent. The first dispatch of each stage always runs alone, because
    it learns which inputs the stage reads.

    Attributes:
        stage: name of the stage being dispatched.
        batch_size: number of requests in the dispatch, one of the stage's resolved batch sizes.
        inputs: read-only mapping of the batch's inputs: device tensors and host values. Lookups (``[]``
            and ``in``) are recorded to learn which inputs the stage reads; ragged handoffs add
            ``<name>_lengths``. Merged lazily, so reading only what the stage needs is cheapest.
        out: output buffers keyed as in ``stage_outputs``; empty while ``stage_outputs`` runs.
    """

    stage: str
    batch_size: int
    inputs: Mapping[str, Any]
    out: Mapping[str, Any] = field(default_factory=dict)


class VLAModel:
    """Base class for models served by XInfer.

    Subclasses are ``torch.nn.Module`` instances as well (the mixin order is
    ``class M(VLAModel, torch.nn.Module)``) and define the class attributes and
    methods below. ``stage_outputs`` declares each stage's output buffers from
    its inputs; each stage method writes into ``batch.out`` and returns None.

    Stage methods must be capturable: work only on the current stream, read
    no device values on the host, and keep shapes a function of the inputs'
    shapes and host-side values (which select the CUDA graph).

    The runtime calls a stage method on the lane's stream under
    ``torch.inference_mode()``. With CUDA graphs, for each new key (stage,
    batch size, and the shapes, dtypes and host values of the inputs the
    stage reads) it first runs the method eagerly on the real inputs
    (learning which inputs it reads and doing lazy initialization outside
    capture), then captures it; later dispatches with the same key replay the
    graph. Branches in stage code must therefore depend only on those keyed
    values.

    Construction and stage calls happen in the engine process (the calling
    process under ``Policy``); the CPU processors (``build_processors``) also
    run in frontend processes, which never load weights.

    Attributes:
        stage_graph: the model's stages and their dependencies.
        handoffs: tensors passed between stages, with their batch layout.
        batch_axes: batch axis of prepared inputs and final-stage outputs by name (default 0); handoff
            tensors use their ``TensorSpec`` instead.
        pad_prompt: ``pad_prompt(tensors, width)``: the prepared tensors with the prompt padded (or truncated)
            to ``width`` tokens; the default ``pad`` applies it to the ``prompt`` dimension.
        checkpoint, processing: set by ``from_checkpoint``; the default ``input_processor`` and
            ``output_processor`` pass them to ``build_processors``.
    """

    stage_graph: ClassVar[StageGraph]
    handoffs: ClassVar[tuple[HandoffSpec, ...]] = ()
    # Batch axis of prepared inputs and final outputs when requests are
    # batched (default 0); handoff tensors use their TensorSpec.
    batch_axes: ClassVar[Mapping[str, int]] = {}
    # Pads the prepared prompt tensors to a bucket width (the default ``pad``).
    pad_prompt: ClassVar[Callable[[Mapping[str, Any], int], dict[str, Any]] | None] = None

    @classmethod
    def from_checkpoint(cls, path: str, *, device: str, ops, **options) -> VLAModel:
        """Build the model and load its weights (engine process, or the ``Policy`` caller).

        The runtime calls it inside ``xinfer.ops.use_ops(ops)``, so layers
        bind the selected operators while they are constructed.

        Args:
            path: checkpoint path (``--checkpoint``).
            device: torch device string of the platform, e.g. ``"cuda:0"`` or ``"cpu"``.
            ops: the operator table selected for the platform.
            **options: model options (``--model-options`` JSON, or ``Policy``'s extra keyword arguments).

        Returns:
            The model on ``device`` with its weights loaded, ready for inference.
        """
        raise NotImplementedError

    def capabilities(self) -> Capabilities:
        """Defaults: every stage takes one request at a time.

        Override to declare batch sizes, the operators each stage calls and RTC
        support. Called at startup by the resolver, the engine and ``Policy``.

        Returns:
            The model's ``Capabilities``; the default has B1 only for every stage and ``num_steps=0``.
        """
        return Capabilities(batch_sizes={s.name: (1,) for s in self.stage_graph}, num_steps=0)

    def io_schema(self) -> IOSchema:
        """Describe the wire-level request and response fields.

        Returns:
            The model's ``IOSchema``.
        """
        raise NotImplementedError

    @classmethod
    def build_processors(cls, checkpoint: str, **options) -> tuple[InputProcessor, OutputProcessor, dict]:
        """CPU processors and the wire metadata frame, without loading weights.

        Frontend processes call this; ``options`` are the same model options
        ``from_checkpoint`` receives. The metadata dict is sent to each client
        when it connects.

        It must not use the GPU (frontend processes hide all CUDA devices).
        ``xinfer serve`` also calls it in the engine process to prepare the
        ``--warmup`` requests.

        Args:
            checkpoint: checkpoint path, as given to ``from_checkpoint``.
            **options: the model options.

        Returns:
            ``(input_processor, output_processor, metadata)``. The frontend adds ``env`` to ``metadata``
            before sending it.
        """
        raise NotImplementedError

    def input_processor(self) -> InputProcessor:
        """Return the InputProcessor for in-process use (``Policy``).

        Returns:
            The first element of ``build_processors(self.checkpoint, processing=self.processing)``.
        """
        return self.build_processors(self.checkpoint, processing=self.processing)[0]

    def output_processor(self) -> OutputProcessor:
        """Return the OutputProcessor for in-process use (``Policy``).

        Returns:
            The second element of ``build_processors(self.checkpoint, processing=self.processing)``.
        """
        return self.build_processors(self.checkpoint, processing=self.processing)[1]

    def shape_buckets(self) -> Mapping[str, BucketSpec]:
        """Policies shared with input processing, keyed by variable dimension (for example ``prompt``).

        The processor supplies ``meta["<dimension>_len_range"]``. The engine enumerates its widths and
        calls ``pad`` during warmup. Models without buckets return an empty mapping.
        """
        return {}

    def warmup_requests(self, prepared: PreparedRequest) -> list[PreparedRequest]:
        """Model-specific warmup variants, such as trained history lengths.

        Shape bucketing is expanded by the engine afterwards; models should declare
        ``shape_buckets`` and implement ``pad`` instead of enumerating prompt widths here.
        """
        return [prepared]

    def pad(self, prepared: PreparedRequest, dimension: str, width: int) -> PreparedRequest:
        """Resize one declared dimension for warmup, preserving model-specific masks and positions.

        Smaller widths may truncate valid input for warmup only. Outputs are discarded; metadata and
        unrelated inputs should be shared with ``prepared``. The default pads the ``prompt`` dimension with
        ``pad_prompt``.
        """
        if dimension != "prompt" or self.pad_prompt is None:
            raise NotImplementedError(f"padding is not implemented for {dimension!r}")
        return PreparedRequest(self.pad_prompt(prepared.tensors, width), prepared.meta)

    def stage_outputs(self, stage: str, batch: StageBatch) -> dict[str, OutputSpec]:
        """Output buffers ``stage`` needs for ``batch`` (``batch.out`` is empty).

        Called before every eager run and every capture of the stage (not
        before graph replays). Shapes must follow from ``batch.batch_size`` and the
        shapes and host values of ``batch.inputs``; lookups here count as
        reads of the stage.

        Args:
            stage: the stage name.
            batch: the dispatch, with ``inputs`` set and ``out`` empty.

        Returns:
            Output name to ``OutputSpec``. It must include every tensor the stage hands off; for a final
            stage every declared output is returned to the caller, split into per-request rows along its
            batch axis.
        """
        raise NotImplementedError

    def stage_fn(self, stage: Stage) -> Callable[[StageBatch], None]:
        """Return the callable that runs ``stage``.

        Args:
            stage: a stage of ``stage_graph``.

        Returns:
            The bound method named ``stage.method``; it takes a ``StageBatch``, fills ``batch.out`` and
            returns None.
        """
        return getattr(self, stage.method)
