"""What the engine reports for each completed request: plain data, no metrics library (docs/metrics.md).

With ``Engine.stats_sink`` set, the engine fills a ``RequestStats`` when a request completes and hands it to the
sink. Consumers turn it into their own records: ``xinfer-serve`` into its Prometheus metrics
(``xinfer.observability.metrics.EngineMetrics.record``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageStats:
    """One stage of a completed request: its times in seconds and its dispatch's batch size.

    Attributes:
        wait: from when the stage can start (the request's inputs on the device, or its upstream stages finished)
            to the start of its dispatch.
        execution: the device time of the stage dispatch that served the request.
        batch_size: requests in that dispatch.
    """

    wait: float
    execution: float
    batch_size: int


@dataclass(frozen=True)
class RequestStats:
    """A completed request's time in the engine.

    Attributes:
        admission_wait: submission to admission, in seconds (host clock).
        stages: ``{stage: StageStats}``, measured on one clock: CUDA timing events, or host times on the CPU. Along
            the request the stages' waits and executions add up to its inputs being ready to its final stage
            finishing.
    """

    admission_wait: float
    stages: dict[str, StageStats]
