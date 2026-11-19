"""RPU prepared executions, per-invocation bindings and completion waits."""

from __future__ import annotations

from .resources import RpuOperation, _Resource


class RpuCompletion:
    """One submission, retaining its execution and all buffer owners until wait."""

    def __init__(self, execution):
        self.execution = execution
        self.done = False
        self.error = None

    def wait(self):
        context = self.execution.context
        with context._lock:
            if not self.done:
                try:
                    self.execution.native.wait()
                except BaseException as error:
                    self.error = error
                    context.needs_reset = True
                finally:
                    self.done = True
                    context._pending = None
                    context.stats["completion_waits"] += 1
            if self.error is not None:
                raise self.error
        return self


class RpuExecution(_Resource):
    """Prepared batch. Every replay returns a distinct real completion owner."""

    def __init__(self, context, native, operations):
        super().__init__(context, native)
        self.operations = tuple(operations)
        self.epoch = context.epoch

    def execute(self):
        context = self.context
        with context._lock:
            self._check()
            if context.needs_reset or self.epoch != context.epoch:
                raise RuntimeError("RPU execution is stale; recover and rebuild its bindings")
            completion = RpuCompletion(self)
            try:
                # This is the public native worker/future path, not a CPU
                # event pretending device completion has already happened.
                self.native.execute(async_=True)
            except BaseException:
                context.needs_reset = True
                raise
            context._pending = completion
            context.stats["logical_submissions"] += 1
            return completion

    def bind(self, updates):
        with self.context._lock:
            self._check()
            for operation in updates:
                if not isinstance(operation, RpuOperation) or operation not in self.operations:
                    raise ValueError("bindings must target operations retained by this execution")
                operation._check()
            return RpuBoundExecution(self, updates)


class RpuBoundExecution(RpuExecution):
    """An invocation retaining a shared batch and fixed parameter updates.

    It allocates no SDK queue. Closing it releases this invocation, while the
    prepared execution remains owned by its plan. Updates occur under the same
    device lock as submission; a failed patch submits no work and requires
    explicit recovery before the partially rebound plan can be reused.
    """

    def __init__(self, execution, updates):
        _Resource.__init__(self, execution.context, execution.native)
        self.owner = execution
        self.operations, self.epoch = execution.operations, execution.epoch
        self.updates = {op: {name: dict(values) for name, values in patch.items()}
                        for op, patch in updates.items()}

    def execute(self):
        with self.context._lock:
            self._check()
            self.owner._check()
            try:
                for operation, patch in self.updates.items():
                    operation.rebind(**patch)
                return self.owner.execute()
            except BaseException:
                self.context.needs_reset = True
                raise

    def close(self):
        with self.context._lock:
            if self.closed:
                return
            self._check()
            self.closed = True
            self.context._resources.remove(self)
