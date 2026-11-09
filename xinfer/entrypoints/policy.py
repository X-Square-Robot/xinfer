"""In-process policy API: ``Policy(model, checkpoint).infer(observation)``.

Runs the model's input processor, the colocated engine and the output
processor in the calling process. It is the M0 entry point for evaluation and
on-robot integration (docs/design.md §15). The engine settings are resolved
as for ``xinfer serve`` (xinfer.config); ``cudagraphs`` overrides the default
(on where the platform supports capture). Unlike ``xinfer serve``, the views
are decoded and resized on the device by default where the platform supports it
(``image_preprocess_device``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xinfer import ops
from xinfer.config import ConfigError, ExplicitSettings, resolve, resolve_image_preprocess_device
from xinfer.engine.core import Engine
from xinfer.engine.cudagraph import CUDAGraphConfig
from xinfer.models.interface import PreparedRequest, connect
from xinfer.models.registry import get_model_class
from xinfer.platforms import resolve_platform


class Policy:
    """One model served in the calling process: processors and a colocated engine.

    The constructor loads the model and starts the engine and completion
    threads. ``infer`` is blocking; calls from several threads run
    concurrently in the engine (and may share an action batch), as requests
    from several robots do under ``xinfer serve``. ``engine.shutdown()``
    releases CUDA graphs and their pools.

    Attributes:
        platform: the resolved ``Platform``.
        op_table: the operator selection the model was built with.
        model: the loaded ``VLAModel``.
        capabilities: ``model.capabilities()``.
        io_schema: ``model.io_schema()``.
        resolved: the resolved settings (``xinfer.config.ResolvedConfig``).
        engine: the ``Engine`` running the stages.
    """

    def __init__(
        self,
        model: str,
        checkpoint: str,
        *,
        device: str | None = None,
        op_backends: Mapping[str, str] | None = None,
        reference_ops: bool = False,
        cudagraphs: bool | CUDAGraphConfig | None = None,
        stream_priorities: Mapping[str, int] | None = None,
        image_preprocess_device: str | None = None,
        **model_options: Any,
    ):
        """Load ``model`` from ``checkpoint`` and build its engine.

        Args:
            model: registered model name, entry point name, or import path ``"module:Class"``.
            checkpoint: checkpoint path for ``from_checkpoint`` and the processors.
            device: ``"cuda"``, ``"cuda:N"``, ``"cpu"``, or ``None`` for the current CUDA device when
                available, else CPU.
            op_backends: operator name to backend name, pinning the selection; unpinned operators use the
                highest-priority supported backend.
            reference_ops: select only the torch reference operators.
            cudagraphs: ``None`` uses the default (on where the platform supports capture); a bool sets
                it explicitly; a ``CUDAGraphConfig`` turns graphs on with that policy.
            stream_priorities: CUDA stream priority by stage; lower integers have higher priority. Omitted stages
                use -1 for the final stage on CUDA and 0 otherwise. Nonzero priorities require platform support.
            image_preprocess_device: where the views are decoded and resized: ``"gpu"`` by the engine on the
                device, ``"cpu"`` in the calling thread, or ``None`` for the platform's default (``"gpu"`` on a
                CUDA device with torchvision; ``xinfer.config.resolve_image_preprocess_device``). Sets the model's
                ``processing.image_preprocess_device``.
            **model_options: passed to ``from_checkpoint``. A string ``quantization`` (default ``"bf16"``)
                also selects the tuned table. ``processing``, if given, is a dict.

        Raises:
            ConfigError: CUDA graphs requested on a platform without capture; ``image_preprocess_device`` set in
                ``processing`` (it is this argument), or ``"gpu"`` on a platform without device images.
            KeyError: unknown model name.
            RuntimeError: a CUDA device was requested but CUDA is not available.
            ValueError: unsupported device type.
        """
        self.platform = resolve_platform(device)
        ops.registry.load_plugins()
        self.op_table = ops.registry.build_table(
            self.platform, backends=op_backends, reference_only=reference_ops
        )
        model_cls = get_model_class(model)
        processing = model_options.get("processing") or {}
        if "image_preprocess_device" in processing:
            raise ConfigError("image_preprocess_device is a Policy argument, not a processing field")
        if resolve_image_preprocess_device(self.platform, image_preprocess_device) == "gpu":  # "cpu": the default
            model_options["processing"] = {**processing, "image_preprocess_device": "gpu"}
        with ops.use_ops(self.op_table):
            self.model = model_cls.from_checkpoint(
                checkpoint, device=self.platform.device, ops=self.op_table, **model_options
            )
        self.capabilities = self.model.capabilities()
        self.io_schema = self.model.io_schema()
        self._prepare = connect(self.model.input_processor())
        self._finish = self.model.output_processor()
        quantization = model_options.get("quantization", "bf16")
        self.resolved = resolve(self.model, self.platform, model_name=model,
                                quantization=quantization if isinstance(quantization, str) else None,
                                explicit=ExplicitSettings(cudagraphs=None if cudagraphs is None else bool(cudagraphs),
                                                          stream_priorities=stream_priorities or {}))
        policy = cudagraphs if isinstance(cudagraphs, CUDAGraphConfig) else None
        self.engine = Engine(self.model, self.platform, self.resolved.engine_config(policy))

    def prepare(self, observation: Mapping[str, Any]) -> PreparedRequest:
        """Run the model's InputProcessor on one observation (CPU, calling thread).

        Args:
            observation: a request in the wire format (arrays as NumPy arrays).

        Returns:
            The ``PreparedRequest`` for ``run_prepared``.
        """
        return self._prepare(observation)

    def warmup(self, observations) -> None:
        """Prepare representative requests and warm the engine up with them (``Engine.warmup``).

        Args:
            observations: iterable of wire-format requests.
        """
        self.engine.warmup([self._prepare(o) for o in observations])

    def run_prepared(self, prepared: PreparedRequest) -> dict:
        """Execute the model on prepared inputs; returns raw final-stage outputs.

        Blocks until the final stage completes.

        Args:
            prepared: the output of ``prepare``.

        Returns:
            Final-stage output name to CPU tensor (pinned on CUDA), as declared in ``stage_outputs``.

        Raises:
            AdmitError: the engine did not admit the request (``queue_full`` or ``stopped``).
            Exception: an exception raised while running a stage is re-raised here.
        """
        return self.engine.run(prepared)

    def finish(self, outputs: dict, prepared: PreparedRequest) -> dict[str, Any]:
        """Run the model's OutputProcessor on one request's raw outputs (CPU, calling thread).

        Args:
            outputs: the outputs ``run_prepared`` returned for ``prepared``.
            prepared: the request's ``PreparedRequest``.

        Returns:
            The response fields, as ``infer`` returns them.
        """
        return self._finish(outputs, prepared)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare one observation, run the model and build the response (blocking).

        Args:
            observation: a request in the wire format (arrays as NumPy arrays).

        Returns:
            The OutputProcessor's response fields (without ``server_timing``).
        """
        prepared = self._prepare(observation)
        raw = self.engine.run(prepared)
        return self._finish(raw, prepared)

    def describe(self) -> dict[str, Any]:
        """Platform, operator selection and capabilities, for startup reports.

        Returns:
            A dict with ``platform`` (name), ``device``, ``ops`` (one row per operator), ``capabilities``
            and ``config`` (the resolved settings as ``ResolvedConfig.rows()``).
        """
        return dict(
            platform=self.platform.name,
            device=self.platform.device,
            ops=self.op_table.report(),
            capabilities=self.capabilities,
            config=self.resolved.rows(),
        )
