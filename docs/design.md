# XInfer design (review draft v1)

English | [简体中文](design.zh-CN.md)

- **Scope**: overall architecture, core interfaces, first models and hardware, development process

Existing code and records this design draws on:

- earlier internal serving prototypes of these models and their measurements;
- current main source of vLLM and SGLang.

The [performance table](performance.md) lists measurement conditions for Wall-OSS and Wall-X 2B and links to their validation results.

---

## 0. Decisions

| Item | Decision |
|---|---|
| Project name | XInfer |
| Repository | X-Square-Robot/xinfer on GitHub. |
| Kernels | Stay in the existing xcompute repository, owned by the operator team; XInfer integrates through the xcompute binding in the ops layer |
| Operator binding | xcompute uses TVM-FFI by default; where there is a concrete PyTorch compile or export need, `torch.library` adapters are added as needed and reuse the underlying implementation (§14.2) |
| Deployment scenario | Each accelerator runs a model instance serving one or more robots. |
| Bundled models | Wall-OSS and Wall-X 2B |
| First hardware | RTX 5090 (sm120), Jetson Thor (sm110) |
| GPU process layout | Default `colocated`: one Engine per GPU, with Vision / Prefill / Action in one process; v1 implements only this layout, `split` is not implemented for now |
| Stage and deployment boundary | Logical stages are separate from the process mapping; StageGraph declares computation and dependencies, and the runtime decides which process a stage runs in (§9.3) |
| Hardware direction | v1 implements only the first hardware; other chips connect through Platform plugins and operator backends (§13, §18.3). No graph-compiling execution backend (§9.4) |
| Out of scope | SLO schedulers; only the scheduling-policy plugin interface is kept |

---

## Implementation status (2026-11-04)

The other sections describe the target design. The table below shows how far each design item is implemented on main; items marked "not implemented" are still reviewed against the original text until they are implemented.

| Design item | Status | Current state |
|---|---|---|
| §4 Package dependency direction | Implemented | Checked by `tests/unit/test_layering.py` (including `frontend/` and `observability/`); the two rules that `engine/` does not depend on `ops/` or `frontend/` are not checked yet |
| §4 Device boundary | Partially implemented | `engine/` no longer writes `torch.cuda.*` directly: every device primitive is reached through `xinfer/platforms/devices.py`'s `device_module_for` / `graph_api_for` / `jpeg_decoder_for`, so a torch-vendor runtime (CUDA today, ``torch.npu`` tomorrow) binds the same engine surface. The platform factory narrows vendor-specific strings to `platforms/detect.py`, `platforms/{cuda,cpu,npu}.py` and `platforms/rpu/`. Engine/policy/serve/config dispatch reads `capabilities.{execution_model,op_selection,cuda_interop}`, not `is_<vendor>` predicates; `frontend/shm.py` still calls `cudaHostRegister` directly, gated by `capabilities.cuda_interop`; `test_layering.py` only checks that `platforms/interface.py` and `models/interface.py` do not import torch |
| §5 Cancellation and per-request errors | Partially implemented | Cancelled requests that are not yet dispatched are dropped; dispatched ones run to completion and their results are discarded. An error during stage execution, during the upload at admission or while taking a batch fails only the affected requests and releases their slots. If the engine thread or the completion thread stops, `xinfer-serve` exits with status 1; recognizing sticky device errors (`Platform.classify_error`) and DRAINING are not implemented |
| §6.3 Three scheduling rules | Implemented | A request enters the downstream queue as soon as its upstream is dispatched, and the downstream lane waits for the event on the GPU; zero-wait batching; the completion thread waits only for the final event (`xinfer/engine/core.py`) |
| §6.4 Batched precision | Implemented | The Wall-OSS reference action stage supports B1, B2, B4 and B8. The optimized composition, vision and prefill run at B1. |
| §7.1 Admission | Partially implemented | `submit` returns a Future, which on rejection holds an `AdmitError` (`queue_full`, `per_robot_limit`, `stopped`); `admission_timeout` is not implemented |
| §7.2–§7.4 Stage queues, batching, dispatch | Implemented | FIFO; the runtime derives the compatibility key from the inputs a stage actually reads; handoff tensors with different sequence lengths are left-padded and batched; batches are joined with torch concatenation, not through Platform copy primitives |
| §7.5 Policy plugins | Not implemented | FIFO is built into the engine |
| §8.1 Slot state machine | Partially implemented | `SlotPool` only counts admissions and checks generations (FREE / ACTIVE / RELEASING); handoff tensors are held by the request, and each stage makes its own copy |
| §8.2 Memory planner | Not implemented | |
| §8.3 Graph management | Partially implemented | Each lane owns its graph pool and static buffers. Wall-OSS uses 16-token prompt buckets by default. Warmup enumerates the reachable widths and allowed batch sizes for each lane. |
| §9.1 Engine threads | Partially implemented | Three threads: the engine thread, the completion thread and the device-image thread (decodes and resizes, on its own stream, the views of requests with `DeviceImage` inputs); OMP / MKL / NUMBA thread counts default to 1; freezing GC after warmup is not implemented |
| §9.2 Frontend processes | Implemented | 2 processes by default, SO_REUSEPORT, 4 processing threads per process; shared memory plus pipe notification (not ZMQ); does not initialize CUDA, but imports torch. In-process calls are provided by `Policy` |
| §10 Configuration resolution | Partially implemented | Explicit value > tuning table > model / platform rules > default, with the source of each item printed at startup (docs/configuration.md); no tuning tables ship with the package yet; `xinfer tune`, memory-planner input to derivation, and `Platform.derive` / `validate` are not implemented |
| §11 Model integration | Implemented | Models declare stages, handoff tensors, capabilities, input and output processing, and shape buckets. The frontend imports the model class's `build_processors`; the engine calls the model's `pad` hook for warmup. |
| §11.4 Bundled models | Implemented | Wall-OSS supports reference and optimized BF16 compositions. The optimized composition on RTX 5090 passes the internal precision gate. Wall-X 2B uses a Qwen3.5 backbone and supports RTC on the desktop dual-arm robot. |
| §12 Quantization | Partially implemented | The framework supports per-group `QuantConfig`. Wall-OSS runs in BF16. MXFP8 and NVFP4 have PyTorch reference operators. See [quantization](quantization.md). |
| §13 Platform | Partially implemented | Only device capability descriptions and CUDA / CPU resolution (`xinfer/platforms/`); streams, events, copy primitives, `register_host` (`frontend/shm.py` currently calls `cudaHostRegister` directly), `classify_error`, `derive` / `validate` and plugin activation are not implemented; Thor (sm110) is unverified |
| §14 Operators | Partially implemented | `OpSpec` and contracts, the registry, build-time selection and the selection table are implemented; backends are xcompute (TVM-FFI) and flash-attn, registered by the `cuda` backend modules and loaded with the `xinfer.ops` entry points (§14.2); `xinfer.ops.conformance` is not implemented |
| §15 Protocol and frontend | Partially implemented | WebSocket + msgpack, compatible with the x2robot client; `xinfer.Policy` in-process API; Client SDK and compatibility adapters not implemented |
| §16 Observability and benchmark tools | Partially implemented | The service exposes [Prometheus metrics](metrics.md). Use the [benchmarks](benchmark.md) for single-request latency and closed-loop capacity, and [xinfer-profile](profiling.md) to measure operator, queue and stage times. |
| §17 Testing | Partially implemented | CPU unit and contract tests; GPU and asset tests need local assets and are run manually; CI (GitHub Actions, `.github/workflows/ci.yml`) runs ruff, the repository and documentation checks and the CPU tests |
| §18.1 Repository layout and command line | Differs from the plan | The repository is authoritative for directories; the service entry point is `xinfer-serve` (that is, `python -m xinfer.entrypoints.serve`) and the profiling entry point `xinfer-profile` (`ops`); the unified `xinfer` command is not implemented |

---

## 1. Background: how VLA serving differs from LLM serving

| Aspect | LLM serving | VLA serving |
|---|---|---|
| Request shape | One prefill + long autoregressive decode | One-shot computation: vision encoding → VLM prefill → action denoising (fixed step count), no long decode |
| Arrival pattern | Users arrive at random | Each robot sends periodically in a closed loop, usually with one request in flight; next send = max(last send + period, response arrival) |
| State lifetime | KV lives across many decode steps and needs paged management | Prefix KV lives only until the request's denoising finishes; per-request slot allocation is enough |
| Shapes | Sequence length varies widely | Camera count and resolution are fixed and prompt length is bounded, which suits shape buckets plus CUDA Graph |
| Bottleneck | Memory bandwidth, KV capacity | Low GPU utilization at small batch sizes; CPU preprocessing and dispatch overhead; large compute differences between stages |
| Metrics | TTFT, TPOT, throughput | Single-request latency; per-GPU capacity N\* (the number of robots served under a given latency definition) |

So VLA serving cannot simply reuse the token-level scheduling and paged KV of vLLM/SGLang.

---

## 2. Goals and non-goals

### 2.1 Goals

| Goal | Acceptance criterion | Existing reference |
|---|---|---|
| Thor single request | Wall-OSS on Thor, same input | No higher than the earlier Thor port of the model |
| Highest per-GPU capacity | 5090, uniform input, client P50 < 150 ms, three runs | Same-condition capacity of the earlier internal serving runtime (baseline numbers to be rechecked under §20 M1) |
| Same results alone and under full load | The same input run alone and under full load | Both pass the model's precision gate; bitwise-equal outputs where the stages' operators are deterministic (§6.4) |
| Easy to add new models | When the backbone and action-head building blocks exist | Write only the model file and weight mapping, no runtime changes |
| Easy to add new hardware and quantization | A new platform or quantization method | Implement only the platform layer, an operator backend or a QuantMethod, no model-file changes |

### 2.2 Non-goals (v1)

- **SLO schedulers.** The core keeps only the policy plugin interface (§7.5).
- **Training, RL, online weight updates.**
- **Multi-GPU tensor parallelism.** The first models fit on one GPU; multiple GPUs use data-parallel replicas plus a Router.
- **GPU stages in separate processes (`split`).** v1 implements only `colocated`, with no inter-stage CUDA IPC, copy-ACK or MPS lifecycle; multi-process CPU Frontends remain in scope. The responsibility boundaries are kept, but no remote-execution framework is built in advance (§9.3).
- **Numerics-changing optimizations enabled by default.** Examples are approximate stage overlap (reusing approximate KV across stages, which lowers task success) or fewer denoising steps. Quantization is offered as explicit configuration, with a precision gate.
- **Autoregressive action-token models** (FAST, OpenVLA-style). The stage abstraction leaves room for a decode stage; v1 does not implement it.

---

## 3. Terminology

| Term | Meaning |
|---|---|
| Stage | A piece of computation declared by the model, for example `vision`, `prefill`, `action`. A unit in code |
| StageGraph | The DAG of computation stages declared by the model, without process placement. The first models follow the logical chain vision → prefill → action; each model declares its own stages and handoffs |
| Lane | One concurrent execution slot of a stage: one stream + one set of static workspaces + one graph pool |
| Shape bucket | A discrete shape a stage input can take, for example Wall-OSS's prompt width in multiples of 16 |
| Slot | Per-request handoff storage for cross-stage tensors (such as the prefix KV), with a generation |
| Handoff | The set of tensors passed between stages, declared by a HandoffSpec |
| Chained dispatch | When an upstream stage is dispatched, its downstream stage is dispatched with it, linked by GPU events (§6) |
| Tuning table | The measured best resource configuration, stored per (model, platform, quantization) (§10) |

### 3.1 Device, platform and toolchain naming

Three layers take three words, and one object never takes two names inside one layer. This is what keeps
`npu` and `ascend` from reading as synonyms.

| Layer | Word | Written as | Not used for |
|---|---|---|---|
| torch device | `npu` | device strings `npu:0`, the `torch.npu` / `torch_npu` API, `platforms/npu.py` (the module that resolves a torch device), `Platform.is_npu`, `NpuJpegDecoder` | backend or vendor identity |
| platform and backend | `ascend` | `PlatformCapabilities.device_type == "ascend"`, platform name `ascend-<arch>`, `xinfer/ops/backends/ascend/`, selection names `ascend:xcompute` and `ascend:torch` | torch device strings |
| toolchain | `cann` | prose and error text about DVPP, ACLNN and FIA; `load_npu_media` imports CANN's Python ACL | selectors of any kind |

Rules:

1. A torch device string, or the torch API that consumes one, is `npu`.
2. Whatever identifies *which hardware and backend* serves a request — `device_type`, the platform name, the
   backend package, a selection name — is `ascend`.
3. The vendor software stack is `cann`, named only in prose and error messages, never as a selector.
4. `Platform.name` is therefore `ascend-<arch>`, for example `ascend-950dt9582`, matching its `device_type` and
   its backend. The `npu` in `Platform.is_npu` and in `platforms/npu.py` answers the device-layer question
   "is this a torch NPU device", which is why those two keep it.
5. External names keep their spelling: `npu-smi`, `torch_npu`, `ASCEND_HOME`, and vendor error text are quoted
   verbatim rather than normalised to the rule above.

---

## 4. Overall architecture

```text
 Robot-side Client SDK (trigger rule, RTC, chunk alignment, timing)
        │ WebSocket + msgpack
 ┌──────▼────────────────────────────────────────────────────────────────────────────────┐
 │ Frontend process ×N (CPU only, no GPU visible)                                        │
 │   Protocol and sessions · InputProcessor (decode/resize/tokenize/normalize)           │
 │   OutputProcessor (denormalization, representation conversion, RTC alignment)         │
 └──────┬─────────────────────────────────────────────────────────▲──────────────────────┘
        │ shared-memory slot (lease + generation) + notification  │
 ┌──────▼─────────────────────────────────────────────────────────┴──────────────────────┐
 │ Engine process (one per GPU)                                                          │
 │   Scheduler: admission · stage queues · batching · chaining · cancel · policy plugins │
 │   StageExecutor ×3: k lanes per stage; GraphManager                                   │
 │   Memory: read-only weights · request slot pool · graph pool budget · memory planner  │
 │   Completion thread: observes final events, hands back results                        │
 └──────┬────────────────────────────────────────────────────────────────────────────────┘
        │ stage functions
 ┌──────▼────────────────────────────────────────────────────────────────────────────────┐
 │ models/ (with models/layers/, models/layers/quantization/) · loader/                  │
 └──────┬────────────────────────────────────────────────────────────────────────────────┘
        │ ops.<name>(...)
 ┌──────▼────────────────────────────────────────────────────────────────────────────────┐
 │ platforms/ (cuda: sm120, sm110; cpu reference) · ops/ (contracts + reference ops)     │
 │ xcompute (existing operator repo, ships high-performance kernels per platform)        │
 └───────────────────────────────────────────────────────────────────────────────────────┘
 Optional: Router (session affinity, failure retry) · instance lifecycle · health checks
```

### Dependency direction

Logical stages are separate from the process mapping. In v1 the three GPU stages all live in one Engine, each with its own queue, lanes, streams and graph pool; stages that progress independently do not need separate processes. Models declare their computation and handoff contracts through StageGraph and HandoffSpec, and hold no process, IPC or scheduler-internal handles.

Two principles:

- **Keep dependencies between layers few.** A new cross-package dependency must first be added to the table below, with a reason.
- **Pay for abstraction at build and startup time.** Operator selection, configuration resolution and graph capture all happen at startup; hot paths such as dispatch, completion observation and handoff go through only a thin layer of calls, with no per-call lookup, validation or extra copies; the overhead is measured as in §6.5.

| Module | May depend on | Must not depend on |
|---|---|---|
| `engine/` | `platforms/`, the model interface (abstract classes), `observability.annotate` (stage spans) | Any concrete model, quantization method or kernel package; `config/`, `models/layers/`, `loader/`, `ops/`, `frontend/` |
| `models/` | `models/layers/` (including the `models/layers/quantization/` interface), `ops/`, `loader/`; platform contracts in backend-specific implementations | `engine/` internals; concrete platforms in shared model code |
| `models/layers/` (including `models/layers/quantization/`) | `ops/` | Concrete kernel packages (only through ops registration) |
| `ops/` | `platforms/` | Models, `observability/` |
| `frontend/` | The model's Input/OutputProcessor | `engine/` and other runtime internals; `torch.cuda` and any device runtime (Frontend processes do not initialize CUDA or any other device runtime) |
| `config/` | `platforms/`, `engine/` (`EngineConfig`), the built model | `frontend/`, `models/layers/`, `loader/` |
| `loader/` | None | `models/`, `models/layers/`, `engine/` |
| `observability/` | `ops/`, `platforms/` | `engine/`, `models/`, `models/layers/`, `frontend/`, `loader/`, `config/` |
| `entrypoints/` | Any package above | None (no other package depends on it) |

The startup order is: build the model (operators are selected here) → `config.resolve` reads the built model's `capabilities()` and the platform capabilities → create the Engine. So what the built model supports, such as its batch sizes, reaches the engine through the model interface, and the engine depends on neither ops nor config; the Frontend's shared memory and the Engine are connected by `entrypoints/serve.py`, so the engine does not depend on frontend.

**RPU package boundaries.** `models/wallx2b/rpu/` owns model configuration, weights, request processing and resident stage plans. `engine/rpu/engine.py` owns request admission, submission, completion, reset and shutdown through the model's RPU hooks. The platform package owns device resources: `platforms/rpu/platform.py` describes capabilities, `resources.py` owns buffers and operations, `execution.py` owns ordered executions, and `context.py` owns context lifecycle. `platforms/rpu/plan.py` defines the RPU request contract consumed by the backend model and engine; the shared model interface stays independent of vendor execution types. `config/rpu.py` supplies RPU request and lane defaults while the generic resolver keeps frontend and shared-memory settings. The package facades preserve the existing engine and platform imports.

`test_layering.py` checks the package dependency directions in the table above, and that `platforms/interface.py` and `models/interface.py` do not import torch. The device boundary below is a target; it is neither implemented nor checked.

**Device boundary (target).** Device runtime APIs (`torch.cuda`, `torch.npu`, etc.), device strings and architecture checks (`sm_`, `gfx`, etc.) may appear only in the concrete platform modules or packages under `platforms/` and in `ops/backends/`. The ROCm build of PyTorch also uses the `torch.cuda` interface, so the boundary is drawn by platform implementation, not by API name. Core code decides by Platform capabilities, not by platform name (§13.1). Current state: `engine/` no longer writes `torch.cuda.*` directly — stream/event/graph primitives go through `platforms/devices.device_module_for`, and entry-point dispatch reads `capabilities.{execution_model,op_selection,cuda_interop}` — so a torch-vendor runtime (CUDA, `torch.npu`) plugs into the same engine; vendor identity queries are sealed inside `platforms/{cuda,npu,cpu}.py` and `platforms/rpu/`. `frontend/shm.py` still calls `cudaHostRegister` directly, gated by `capabilities.cuda_interop`; each model's `processing.py` outputs torch CPU tensors, so `frontend/` imports torch, but it does not initialize CUDA.

---

## 5. Request lifecycle

```text
Client ──obs──▶ Frontend: validate IOSchema → InputProcessor → write shared-memory slot → notify Engine
Engine Scheduler: admit (allocate request slot) → H2D (uint8 images, tokens, state)
   vision lane:  encode → write slot.image_embeds                     (event v)
   prefill lane: wait(v) → prefill → write slot.prompt_kv …           (event p)
   action lane:  wait(p) → gather → N denoising steps → D2H to pinned (event a)
Completion thread: wait(a) → raw actions back to Frontend → OutputProcessor → Client
Recycle: each lane is free after its stage event; a slot is recycled after its last reader's event (generation +1)
```

### Ownership rules

- **Weights**: read-only, frozen after loading and quantization conversion.
- **Slot**: owned by the request. The producing stage writes it and consuming stages only read it; it is recycled only after the last reader's event completes, recycling increments the generation by 1, and stale handles are detected.
- **Lane workspace**: owned by the lane. A stage output that another stage needs must be written to a slot, not left in the lane workspace.
- **Graph outputs**: must be copied or handed off before the next replay.

### Cancellation and errors

- **Cancellation/disconnect**: requests not yet dispatched are dropped; dispatched ones run to completion on the GPU, their results are discarded, and the slot is recycled after its events.
- **Per-request errors** (input validation failure, shape outside the buckets): fail only that request.
- **Sticky device errors** (CUDA sticky errors, or equivalent unrecoverable errors on other platforms; `Platform.classify_error` tells request-level from device-level errors): the instance enters DRAINING and is restarted by the supervisor process.

---

## 6. Unified execution model: one implementation for a single request and a fleet

### 6.1 Conclusion

There is only one staged structure. The single-request path is simply the fleet path with nothing queued; it has no separate code and no mode switch.

### 6.2 Basis: where each kind of overhead comes from

| Overhead | Affects | Evidence |
|---|---|---|
| Host relay between stages (observe completion → enqueue → dispatch again) | Single-request latency | Completion observation, stage handoff and return |
| Waiting to fill a batch | Single-request latency | A request waits even when a lane is free. |
| Holding a lane for the entire request | Fleet throughput | Each stage loses the opportunity to serve another ready request. |
| Stages advance in lockstep | Fleet throughput | Stages cannot advance independently as work becomes ready. |

The first two concern "how stages are linked" and the last two "at what granularity work runs", so they do not conflict: keep the staged granularity and change only how stages are linked.

### 6.3 Three scheduling rules

1. **Zero-wait batching.** A dispatch takes the compatible requests already in the queue and never waits to fill a batch. When idle this is naturally B1; with a queue, batches form naturally.
2. **Chained dispatch when downstream is idle.** When stage s is dispatched, if the downstream stage's queue is empty and it has a free lane, the downstream stage is dispatched immediately too, linked by a device-side event wait (`stream.wait_event`, which is `cudaStreamWaitEvent` on CUDA); the chained path does not wait for the host to observe the intermediate completion before submitting downstream. If the downstream stage already has a queue, the request is enqueued normally, keeping FIFO and batching.
3. **Returning results waits only for the final completion.** The completion thread waits for the last stage's event and returns the result; the scheduling thread still checks each stage's events without blocking, to recycle lanes promptly, advance non-chained requests and maintain slot lifetimes.

**Effect:** with 1 robot, each request is dispatched from vision to action in one go, which is the fastest single-request path. With more robots, downstream queues form and execution becomes a pipeline with batching automatically. Each request is decided by the queue state at its dispatch time, so the transition from idle to full load is continuous.

**No whole-chain graphs.** Every deployment serves multiple robots from one GPU, and capturing the whole chain as one graph would remove pipelining and batching between stages; the fastest single-request path comes from chained dispatch. The earlier Thor port of the model uses a whole-call graph, and Thor's CPU is also weaker; if M0 shows that "three stage graphs + events" on Thor is clearly slower than the whole-call graph (§6.5), a whole-chain graph used only when idle will be evaluated then.

### 6.4 Correctness contract: running alone and under full load

- **Same kernels.** Both paths use the same set of stage graphs. Operators need not be deterministic (§14.1): results under full load must pass the model's precision gate as results run alone do, and where every operator of a stage is deterministic the two are bitwise equal.
- A model declares a stage's supported batch sizes after checking the batched results against its precision gate. Batching may change rounding. Wall-OSS supports action batching in the reference composition and uses B1 in the optimized composition.
- **Scope.** Bitwise equality holds only within one platform and one set of operator implementations, not across chips; alignment across platforms and with the reference implementation uses tolerances, and each platform keeps its own golden outputs.

### 6.5 Risks to quantify in M0 / M1

All measurements below use `colocated`; the capacity and batching risks are verified once M1's multi-lane runtime and fleet benchmark exist, and no `split` comparison is required.

| Risk | How to measure | Milestone |
|---|---|---|
| Chained dispatch reserves downstream lanes early and may lower throughput | Chaining on vs off, compare same-condition capacity | M1 |
| Chained requests no longer join downstream batches | Near the critical load, compare batching rate and capacity | M1 |
| "Three stage graphs + events" on Thor is slower than the whole-call graph | Compare single-request latency on the same input | M0 |
| Host dispatch overhead | Measure the time from "request ready" to "whole chain dispatched", target on the order of 100 µs | M0 |

---

## 7. Scheduler

### 7.1 Admission

- In-flight requests never exceed the number of slots; by default each robot has at most one request in flight.
- Admission and allocation happen in one step, returning `AdmitReady(slot)`, `AdmitDelay` or `AdmitFail(code)`, which avoids the race of "probe first, allocate later" (a lesson from an earlier prototype).
- Queues are bounded. Error codes distinguish `queue_full`, `per_robot_limit` and `admission_timeout`.

### 7.2 Stage queues

`q_vision → q_prefill → q_action`, all FIFO by arrival by default.

### 7.3 Batching

- The compat key is computed from the HandoffSpec and the model's declarations; only requests with the same key can be batched.
- Only captured batch sizes are used, and no dummy rows are padded.
- Zero-wait: only requests already in the queue are taken.
- In v1 the action stage gathers the contexts of several slots into the lane's batch workspace. This could later become a kernel that reads by slot index, saving the copy. The gather is done by the Platform's asynchronous copy primitive (with strided copy support, §13.1).

### 7.4 Dispatch

- A stage is dispatched when it has a free lane, and §6.3 decides whether to chain.
- Every dispatch records an event; the lane is released after the stage's event completes.

### 7.5 Policy plugins

```python
class StagePolicy(Protocol):
    def select(self, stage: str, queue: StageQueue, free_lanes: int,
               now: float) -> list[Dispatch] | Hold: ...
```

- **Division of responsibility**: a policy decides only "which requests to dispatch, and whether to dispatch now or wait". Lane allocation, chained dispatch, batch assembly and recycling are all done by the runtime.
- **Open-source edition**: ships only `fifo`.
- **SLO schedulers**: implemented as plugins, without changing the core, so SLO code does not get intertwined with generic scheduling.

---

## 8. Memory and ownership

### 8.1 Slot state machine

Reuses the slot state machine proven in an earlier internal service:

```text
FREE → WRITING → READY → READING → RELEASING(awaiting events) → FREE (generation+1)
```

### 8.2 Memory planner

Checked at startup:

```text
weights + Σ lane workspaces + Σ graph pools + slot count × slot size + headroom ≤ available device memory
```

- The planner should report feasible lane and slot counts if the memory check fails.
- Thor has unified memory; the budget uses the available amount reported by the platform.

### 8.3 Graph management

- Graphs are captured per (stage, lane, shape bucket, batch). Different shapes in one lane share a graph pool; lanes do not share pools.
- Warmup captures all declared combinations and then seals. An undeclared shape at runtime raises an error by default, configurable to fall back to eager with a warning.
- Operators that need a workspace must make their first allocation during warmup (§14.1).
- Shape-bucket granularity is chosen by the platform: the model declares the legal range, and the platform trades padding waste against the number of captures.

---

## 9. Process and thread model

### 9.1 Engine process (one per GPU, default `colocated` layout)

v1 implements only `colocated`. Vision, Prefill and Action share one Engine process, while execution resources and queues are still managed per stage; GPU work from different requests can overlap on different streams. A single request always follows the StageGraph's dependency order.

| Thread | Responsibility |
|---|---|
| Engine thread (exactly one) | Admission, queues, batching, chaining decisions, non-blocking completion checks of stage events, and all CUDA submissions (H2D, graph replay, events). Scheduling and dispatch share one thread, saving a thread switch |
| Completion thread | Blocks on the final event (without holding the GIL) and returns the result |

- **Thread hygiene** (the default on all platforms, not hidden in some hardware configuration and not routed through Platform): at process start, set the OMP/MKL/NUMBA thread counts to 1; freeze GC after warmup. Evidence: NUMBA thread oversubscription once caused timeouts at 14 robots.
- **Dispatch hot path**: CPU pre- and post-processing is left to the Frontend; the Engine thread does no blocking waits for GPU completion, and no memory allocation or graph capture after warmup. M0 measures submission overhead; M1 measures Engine thread occupancy, lock waits, and the delay from a request becoming runnable to its actual submission; single-process capacity must be accepted by measurement.

### 9.2 Frontend processes

- N processes by default, sharing the port through SO_REUSEPORT, without initializing CUDA.
- Each process uses several processing threads (4 by default) to handle requests from different connections at the same time: image decoding and resize release the GIL, so an uneven spread of connections across processes does not become a bottleneck. InputProcessor and OutputProcessor must therefore be safe to call concurrently.
- Communicates with the Engine through shared-memory slots (lease + generation) plus ZMQ/pipe notification.
- The Engine process maps the shared memory once and registers it as pinned through `Platform.register_host`, so H2D reads directly from it. An asynchronous copy from pageable memory would block the dispatch thread and add a host copy.
- Single-robot deployments can use in-process threads instead, saving the cross-process overhead.

### 9.3 Logical stages separate from the process mapping; `split` deferred

- **Model boundary**: StageGraph declares only stages and dependencies, and HandoffSpec only cross-stage tensor layouts; model files are unaware of which process a stage runs in or how data is transferred.
- **Runtime boundary**: stage queues, batching and lanes are managed per logical stage; slots are owned by requests, and their reads, writes and recycling follow stage dependencies. The runtime assembles execution resources and handles handoffs; v1 maps all GPU stages to one Engine and hands off through request slots and GPU events.
- **Current scope**: no `split`, and no process-layout switch, remote StageExecutor, inter-stage IPC channel or MPS start/stop code. The shared memory and notification channels of the CPU Frontend are not affected by this restriction. Only the responsibility boundaries above are kept; no empty implementations are built in advance.
- **Later extension**: if measurements show that colocated is still limited by CPU submission after blocking and hot-path overhead are removed, process splitting will be reviewed separately. A two-group mapping, Vision + Prefill / Action, could be considered, reusing the logical stages and numerical contracts; this is not part of the implementation or acceptance commitments of the current M0–M3.

Colocating stages keeps request slots in one process and links dispatches using device events. Measure the latency and capacity of this layout under the intended load.

### 9.4 Execution

Each stage calls the model's stage function directly on a stream (submitting through ops) and is captured as a graph as in §8.3. There is no graph-compiling execution backend (tracing and exporting to TensorRT or an NPU compiler), and platforms that offer only a graph-compiling runtime are out of scope; multiple stages are not merged into one graph either (§6.3).

---

## 10. Configuration resolution

### 10.1 Following vLLM / SGLang

Neither of them has **named profiles**:

- Parameters are flat and mostly default to None or auto; at startup only the ones the user did not set are filled in.
- vLLM's `get_batch_defaults()` sets defaults by memory tier and usage scenario.
- SGLang's `resolve_once()` runs device, platform, memory and model handlers in turn.
- vLLM's `--performance-mode interactivity|throughput` adjusts only derived defaults (which batch sizes get graphs, the batch limit), not code paths.
- Measured kernel-level tuning tables ship with the repository (vLLM's fused MoE has hundreds of JSON files named after devices).

### 10.2 Resolution order (highest priority first)

1. **Explicit values**: always take effect; a conflict with platform capabilities is an error, never silently rewritten.
2. **Tuning table**: stores the measured best resource configuration, such as lane counts, per (model, platform, quantization). Generated by `xinfer tune`, shipped with the repository, and records the commit, date and measurement conditions.
3. **Derivation rules**: derived from platform capabilities, model declarations (stage graph, batch sizes, shape buckets) and the memory planner. Shape-bucket granularity and the default `QuantConfig` are also resolved per (model, platform) at this level, not written in model files. Platform limits that caps cannot express come from `Platform.derive` and `Platform.validate` (§13.1), and are marked `platform:<name>` in the startup report; core rules contain no vendor special cases.
4. **Fixed defaults**: independent of load.

There are no load-derived intent parameters (such as the number of robots served). The number of robots an instance serves changes all the time, while the engine adapts to arriving requests by itself: it batches only requests already queued, idle lanes use no compute, and CUDA graphs are captured for the keys that actually occur. So the in-flight request limit (the counterpart of vLLM's `max_num_seqs`, which is likewise a fixed default limit), the allowed batch sets per stage, the lane counts and the number of frontend processes all take limits or fixed values that suit any load (docs/configuration.md).

Runtime behaviors such as chained dispatch and zero-wait batching are not configurable; they are always on, with only debug switches kept for ablations.

The v1 GPU process layout is fixed to `colocated`; neither the tuning table nor the load state switches it to `split`.

### 10.3 Example and audit

```bash
xinfer serve --model wall_oss --checkpoint CKPT                     # defaults
xinfer serve --model wall_oss --checkpoint CKPT --lanes vision=1    # override one item explicitly
```

At startup the resolved values are printed and saved to disk, each with its source:

```text
max_running_requests  32         default
lanes.vision          2          tuned:wall_oss/sm120-mxfp8@<commit>
batch_sizes.vision    [1]        model:declares B1 only
batch_sizes.action    [1,2,4,8]  model:declared sizes
cudagraphs            True       platform:supports graph capture
frontends             2          default, ≤ cpus/2 = 12
frontend_workers      4          default, frontends × workers ≤ cpus/2 = 12
```

### 10.4 Why no named profiles

- Combinations multiply with "model × hardware × deployment" and easily go stale.
- The source of a value is invisible.
- Hardware-independent settings easily get hidden in hardware profiles. For example, an earlier service's `rtx5090` profile set `OMP_NUM_THREADS=1`, which should be the default on every platform.

---

## 11. Model integration

### 11.1 What a model author implements

| Component | Responsibility | Runs in |
|---|---|---|
| `IOSchema` | Inputs: camera names and resolutions, state dimension, instruction, RTC fields and their legal ranges; outputs: action dimension and representation | Protocol validation |
| `InputProcessor` | Decodes views, resizes, tokenizes and normalizes state. Processors with per-connection state can implement `connect()`. A model may return `DeviceImage` for the engine to decode and resize before the first stage. | Frontend |
| `VLAModel` | Declares the StageGraph, shape buckets (legal ranges), HandoffSpec and Capabilities; implements each stage's forward pass | Engine |
| `OutputProcessor` | Denormalization, representation conversion, RTC alignment | Frontend |
| `WeightMapping` | Mapping from checkpoint keys to parameters | Loading |

```python
class WallOSS(VLAModel):
    stage_graph = StageGraph(
        Stage("vision",  "encode_images"),
        Stage("prefill", "prefill", depends_on="vision"),
        Stage("action",  "denoise", depends_on="prefill"),
    )
    handoffs = (
        HandoffSpec("vision", "prefill", {"image_embeds": TensorSpec(batch_axis=0)}),
        HandoffSpec("prefill", "action", {
            "prompt_kv": TensorSpec(batch_axis=2, seq_axis=4)}),  # the model declares the axes for the selected attention backend (§11.2)
    )
    def shape_buckets(self, cfg): ...          # planned: legal range, e.g. prompt width ≤ 200, step 16; the platform picks the actual buckets (§8.3)
    pad_prompt = staticmethod(pad_prompt)      # pads a prepared prompt to a bucket width; the base class's pad() applies it
    def capabilities(self): ...                # RTC, batch sizes, each stage's operators
    def stage_outputs(self, stage, b): ...     # shape and dtype of each stage's output buffers, allocated by the runtime

    def encode_images(self, b: StageBatch) -> None: ...   # writes b.out["image_embeds"]
    def prefill(self, b: StageBatch) -> None: ...         # writes b.out["prompt_kv"]
    def denoise(self, b: StageBatch) -> None: ...         # all N steps in one go, writes b.out["actions"]
```

**Registration is lazy.** A model registers with a name plus two "module:Class" strings: one points to processing (IOSchema, Input/OutputProcessor) and one to the model.

```python
register_model("wall_oss",
               processing="xinfer.models.wall_oss.processing:WallOSSInputProcessor",
               model="xinfer.models.wall_oss.model:WallOSS")
```

The Frontend imports only processing, never model code; only the Engine imports the model. An import failure in one model does not affect other models. Out-of-tree models register in the same form through the `xinfer.models` entry point.

### 11.2 Stage function constraints

- Shapes are static within each shape bucket; outputs are written only to slots or workspaces provided by the runtime.
- Submit only on the current stream; no host synchronization; no memory allocation after warmup.
- No mutable state held across requests.
- Denoising runs as a whole (N steps in one graph). The step count is recorded in Capabilities, leaving an interface for step-level scheduling later.

`HandoffSpec` declares the batch and sequence axes of each tensor. The runtime uses these declarations when gathering and scattering stage outputs. Producer and consumer layouts must agree.

### 11.3 Reusable building blocks

| Category | v1 | Later |
|---|---|---|
| VLM backbone | Qwen2.5-VL | — |
| Vision tower | Qwen2.5 ViT | — |
| Action head | Action expert with adaptive RMS normalization and attention over prefix K/V | DiT cross-attention head (GR00T-style) |
| Sampler | flow-matching Euler; RTC (frozen prefix) | — |
| Preprocessing | Multi-camera images, normalizers, prompt templates | — |

### 11.4 Bundled models

Wall-OSS uses a Qwen2.5-VL vision tower, a prefill expert and a flow-matching action expert. See the [Wall-OSS design](models/wall_oss_design.md) for tensor layouts, execution stages and numerical configurations.

Wall-X 2B pairs a Qwen3.5 vision encoder and hybrid language backbone with a flow-matching action expert. For the checkpoint layout, desktop dual-arm protocol and serving command, see the [Wall-X 2B guide](models/wallx2b.md).

---

## 12. Quantization

- **`QuantConfig`** sets the method for each module group. Wall-OSS currently supports BF16 only. See [quantization methods](quantization.md).
- **`QuantMethod` interface**:
  - `create_weights`;
  - `process_weights_after_loading`: conversion, packing and swizzling, which must finish before the weights are frozen;
  - `apply`: calls the GEMM for the format in ops.
- **Two sources**: online quantization at load time (MXFP8); offline pre-quantized checkpoints (NVFP4 from QAT, M2).
- **Adding a method**: one QuantMethod plus, if needed, an operator backend for its format, with no model-file changes.
- **Precision gate** `xinfer eval quant` (judged separately per model, platform and method):
  - compares against the BF16 reference and reports open-loop action error and drift relative to BF16;
  - also reports the noise floor from a group of perturbed repeats;
  - methods that fail the gate are marked experimental.
- **v1 scope**: BF16, MXFP8. NVFP4 is in M2.

---

## 13. Hardware platforms

### 13.1 Platform interface

```python
class Platform:
    name: str                 # "cuda-sm120", "cuda-sm110", "cpu"
    caps: PlatformCaps        # supported_dtypes, unified_memory, memory, compute_units,
                              # power_limit, graph_capture, stream_priorities, supported_quant
    # Required contract: every primitive the engine actually depends on
    def stream(self, priority=0): ...             # async execution queue; stream.wait_event(evt) waits on the device
    def event(self): ...                          # evt.query() does not block; evt.synchronize() blocks without holding the GIL
    def alloc(self, nbytes, kind): ...            # kind: device | pinned_host; returns a torch tensor
    def copy_async(self, dst, src, stream): ...   # H2D / D2H / D2D, strided copies supported
    def register_host(self, mem): ...             # registers mapped host memory (Frontend shared memory) for DMA, so H2D reads it directly (§9.2)
    def memory_budget(self): ...
    def device_count(self): ...                   # number of available devices; serve starts one Engine per device
    def device_info(self, i): ...                 # name, uuid, total memory; written to the startup report
    def bind_env(self, i): ...                    # environment variables to set before a child process starts (visible devices, etc.)
    def classify_error(self, exc): ...            # request-level error vs sticky device error (§5)
    # Configuration participation (§10.2)
    def derive(self, cfg): ...                    # fills in defaults that caps cannot express, each with source platform:<name>
    def validate(self, cfg): ...                  # platform-specific limits; errors on conflict
    # Optional capabilities
    def graph(self, pool=None): ...               # graph capture and replay (caps.graph_capture, §8.3)
```

A platform declares its capabilities at build time. The runtime derives the configuration from the capabilities; an unsupported explicit configuration is an error.

- **The required contract demands asynchronous execution.** The scheduler's single-threaded dispatch, the completion thread and chained dispatch all rely on asynchronous streams and device-side event waits; platforms that block synchronously are out of scope.
- **dtypes and quantization are capabilities.** If `supported_dtypes` lacks a dtype the model needs (for example, some NPUs have no bf16), or `supported_quant` does not include the selected quantization method, configuration resolution fails, without waiting for model build.
- **External platforms** register through the `xinfer.platforms` entry point (§18.3); the built-in cuda and cpu platforms use the same mechanism.
- `ipc`, `mps` and `spatial_partition` are read by no code in v1 (§9.3); they will be added when needed.
- **The core decides by capability, not by identity.** Platform provides no identity predicates such as `is_cuda()`; `engine/`, `models/layers/` and `models/` read only caps, and undeclared caps take conservative values (unsupported). Every Platform method must have a caller in the core and a contract test, or it is not added (SGLang's platform interface has a set of methods the core never calls, while the core still hard-codes `torch.cuda`, so overriding them has no effect).

### 13.2 First platforms

| Capability | RTX 5090 (sm120) | Jetson Thor (sm110) |
|---|---|---|
| Typical deployment | Fleet, multiple robots per GPU | Fleet, multiple robots per GPU |
| Memory | 32 GB discrete GPU memory | Unified memory: H2D can be zero-copy or mapped |
| CPU | Server CPU (24-core quota in the test container) | ARM, weaker: dispatch overhead is a larger share, and Frontend preprocessing is also more likely to become the bottleneck under fleet load |
| MXFP8 GEMM | PyTorch reference | PyTorch reference |
| MPS | Supported; v1 colocated does not depend on it | To be confirmed; v1 colocated does not depend on it |
| Known references | Earlier internal serving runtimes | The earlier Thor port of the model (ViT BF16, whole-call graph, RTC off) |

AMD GPUs are also a target platform; the timing and GPU models are to be decided (§22). The Platform interface is designed with HIP's differences in mind: HIP Graph, no MPS, CU masks; the available quantization formats depend on the GPU model and are expressed through `supported_quant`.

### 13.3 External stable surface and vendor self-certification

As an open-source project, XInfer integrates chips from different vendors whose packages are maintained out of tree, so it must state what is stable and how a vendor proves that its integration is correct.

- **Stable interface surface**: Platform, the operator signatures and numerical contracts of `ops`, VLAModel and HandoffSpec. This set carries an API version number; plugins declare the version range they need, and a mismatch is an error at activation that states both versions. There is no migration layer and no compatibility shims; breaking changes bump the major version.
- **Vendor self-certification**:
  - `xinfer.platforms.conformance` checks the Platform contract: `wait_event` ordering, `query` not blocking, `synchronize` releasing the GIL, strided `copy_async`, copies after `register_host` not blocking the host, both kinds of `alloc`, and error classification;
  - `xinfer.ops.conformance` (§14.1) checks operator implementations.
- **The out-of-tree path itself must be tested.** CI installs the CPU mock platform as a separate out-of-tree-style package and runs discovery, activation and contract tests through the entry point; otherwise the first vendor would be the first to hit the problems on the entry-point path.
- **Support tiers** (written to `docs/platforms/`): Tier 1 is covered by the main repository's CI (5090, Thor); Tier 2 is maintained by the vendor, passes the suites above, and must state the API version it uses; everything else is community-maintained.

---

## 14. Operators and the kernel repository

### 14.1 Operator contracts (in the XInfer repository, `xinfer/ops/`)

Each operator defines:

- **Signature**: outputs are written to an `out` provided by the caller; a workspace, if needed, is also provided by the caller.
- **torch reference implementation**.
- **Numerical contract**: bitwise equal to the reference or within a given tolerance; whether it is deterministic.

Determinism is declared, not required. An operator may give results that differ at rounding level between runs, for example stream-K GEMMs whose partial sums meet in a different order under concurrent load. What a model requires of its operators is that it passes its precision gate (docs/precision.md); for VLA models that is mainly the open-loop gate against recorded trajectories. Declaring `deterministic=False` tells tests and readers not to expect bitwise reproducibility.

Calling conventions:

- Submit only on the current stream; no implicit synchronization; no internal memory allocation.
- Operators that need workspace allocate it during warmup to avoid allocating memory during CUDA graph capture.
- The conventions above on `out`, workspace, the current stream, and no implicit synchronization or allocation do not change with the TVM-FFI or `torch.library` binding.

Validation requirements:

- Every binding must be validated for numerical agreement with the reference implementation, for its declared determinism, and for its declared CUDA Graph capture and replay support; the binding method by itself guarantees none of these properties, nor any performance gain.
- When a `torch.library` adapter is introduced, additionally validate the registration contract with `torch.library.opcheck`, and test the compile or export paths actually supported, covering input mutation and alias declarations, FakeTensor/shape inference and similar behavior. `opcheck` does not replace numerical agreement tests.

### 14.2 Backend registration and selection

A backend is the hardware an implementation runs on (`cuda`); the torch reference (`reference`) runs on every platform. Each backend is a package under `xinfer/ops/backends/` with one module per operator category (`cuda/attn.py`, `gemm.py`, `norm.py`, `flow.py`). Every module imports the kernel libraries it calls as it would any library and declares its implementations at module level with the same `xinfer.ops.register`, passing the library as `requires`. A library that does not import is None: its implementations are not registered and the operators run on their references. xcompute, which will serve more backends than CUDA, is imported once by `backends/_xcompute.py`, which checks that its Python package matches the shared library and passes the architectures the build targets as `platforms`.

```python
try:
    import flash_attn
except ImportError:
    flash_attn = None

@register("causal_attention", backend="cuda", requires=flash_attn)
def causal_attention(q, k, v, *, scale=None): ...
```

`register` adds to the process-wide registry, so importing a backend module registers its implementations; `OpRegistry.load_plugins()` imports the built-in backend modules, then the modules named by the `xinfer.ops` entry point group. `OpRegistry.add`, the one way into a registry, checks that each implementation takes the parameters of the operator's reference and registers it as `backend:library` (`cuda:xcompute`). `XINFER_XCOMPUTE=0` and `XINFER_FLASH_ATTN=0` leave a library out. The process-wide registry loads the plugins before its first `build_table`. A pin (`--op-backends`) names a backend, a library or both.

- The implementation is selected at model build time, not dispatched on every call. An implementation that supports only some calls of an operator runs the others on the reference itself, as xcompute's `prefix_kvcache_attention` does for the cache shapes its kernel does not take.
- When operators a model calls run on a fused reference on a CUDA device, one warning names them. `XINFER_OPS_STRICT=1` turns this warning into an error (CI and benchmarks).
- The selection, library versions and hashes are written to the engine metadata, and an "operator selection table" is printed at startup.
- A missing implementation fails explicitly; it can also be configured to fall back to the reference implementation with a warning.

**Operator binding strategy:**

- **Default path**: xcompute integrates through TVM-FFI; `xinfer/ops/backends/_xcompute.py` imports it and the modules of `xinfer/ops/backends/cuda/` register its operators in XInfer's operator registry.
- **Adapters as needed**: when an operator has a concrete PyTorch integration need such as `torch.compile` or `torch.export`, a `torch.library` registration is added. The wrapper reuses the existing underlying implementation and can call the same xcompute kernel through TVM-FFI, so two sets of kernels need not be maintained.
- **Responsibility boundary**: the `torch.library` adapter covers the operator signature, input mutation and alias declarations, and the required FakeTensor/shape inference rules; hardware implementation selection and numerical policy stay with XInfer's operator registry and contracts.
- **Unified call surface**: models and layers call only XInfer's operator interface and do not depend directly on the binding method. Registering with XInfer is not the same as registering a PyTorch custom operator.
- **Delivery scope**: an operator does not have to provide both entry points, and the default execution path does not have to go through two layers of wrappers. xcompute already uses TVM-FFI; `torch.library` adapters are implemented and validated as concrete needs arise, with no placeholder modules built in advance and no claim of a complete dual path.

### 14.3 Division of work

| | XInfer repository (framework team) | xcompute repository (operator team) |
|---|---|---|
| Content | Operator signatures, reference implementations, conformance test suite, registration mechanism | High-performance implementations for each platform (CUDA/CuTe/Triton/HIP), with a wheel per platform |
| Testing | Model-level alignment, runtime consistency | Runs `xinfer.ops.conformance` in CI (numerics, determinism, graph capture) and provides microbenchmarks |
| Deliverables | The operator list and shapes for each model | A theoretical lower-bound (roofline) table for each implementation |
| Versioning | Declares the compatible xcompute version range; checks the build fingerprint at startup | Semantic versioning; ABI changes bump the major version |

Tool: `xinfer profile ops --model wall_oss` reports each operator's time and theoretical lower bound for a model, so the operator team can set priorities.

Vendor NPU implementations do not need to be merged into xcompute: a separate package that registers through the `xinfer.ops` and `xinfer.platforms` entry points is enough (§18.3).

### 14.4 Wall-OSS operator list

- GEMM: BF16, MXFP8; plus fused quantization (RMSNorm + quant, SiLU-mul + quant).
- Attention:
  - bidirectional vision attention (Qwen ViT window/full);
  - prefix prefill;
  - joint attention from actions to the prefix (per-row valid lengths).
- Other: RMSNorm / adaRMS, RoPE / M-RoPE, patch embed, uint8 image → normalized patches, flow Euler step (with RTC).

---

## 15. Protocol, frontend and client SDK

- **Protocol**: WebSocket + msgpack, the same as the existing services.
  - Request: metadata plus the observation (multi-camera JPEG or raw images named per the IOSchema, state, instruction, optional RTC fields: the previous chunk, `inference_delay`).
  - Response: action chunk, action-space description, per-phase timing.
- **Compatibility adapters (M3, optional)**: LeRobot asynchronous inference.
- **Offline API**: `xinfer.Policy(model, checkpoint).infer(obs)`. It uses the same Engine and the same configuration resolution in process, for evaluation (such as LIBERO) and onboard integration.
- **Client SDK** (a separate small package):
  - real-robot trigger rule: next send = max(last send + period, response arrival);
  - RTC delay estimation (for example, P75 of the last 10 latencies);
  - chunk alignment by capture time;
  - client response latency timing.

---

## 16. Observability and benchmark tools

- **Per-request phase timing**: admission wait, per-stage queueing and GPU time, handoff, return.
- **Metrics and tracing**: Prometheus metrics; Perfetto trace export.
- **Benchmark tools**:
  - `xinfer bench latency`: single-request latency.
  - `xinfer bench fleet`: a virtual robot fleet (200 Hz control, model steps, execution length, trigger rule, RTC) that searches for per-GPU capacity N\*, repeated three times.
  - `xinfer tune`: sweeps resource configurations and generates tuning tables.
  - `xinfer eval`: bitwise/tolerance comparison with the reference implementation, reporting open-loop trajectory error; see §12 for `eval quant`.
- **Primary latency definition**: client response latency (D-17).

These tools decide whether others can reproduce "how many robots per GPU", and they matter as much as the engine.

---

## 17. Correctness and testing

| Level | Content | Environment |
|---|---|---|
| Contract tests | A fake model covers admission, slot lifecycle, cancellation, batching, chained dispatch and error paths; checks dependency direction (the device boundary is added with the device abstraction, §4). The CPU platform simulates asynchronous streams and events with threads | CPU, every MR |
| Out-of-tree plugin path | A separate dummy platform package goes through discovery, activation and contract tests via the entry point (§13.3) | CPU, every MR |
| Operator conformance | Each backend implementation against the reference implementation; every binding follows the execution and validation contracts of §14.1 | xcompute repository CI (GPU) |
| PyTorch operator adapters (when introduced) | `torch.library.opcheck`, the compile/export paths actually supported, and numerical agreement after adaptation (§14.1) | Main repository CI, on the devices declared as supported |
| Model alignment | Frozen inputs, noise and reference outputs, compared with the original implementation | GPU, every MR (small sample) |
| Runtime consistency | Chained vs non-chained, B1 vs Bn, idle vs full-load replay: outputs pass the precision gate, and are bitwise equal at B1 where the operators are deterministic | GPU, nightly |
| Capacity regression | Thresholds maintained per (model, platform) | GPU, nightly or weekly |

---

## 18. Repository and development process

- XInfer integrates xcompute through the operator contracts, bindings and conformance checks in §14. External publication requires license and contribution-process approval.
- **Conventions**:
  - MR descriptions must state the problem, changes, validation and limitations;
  - raw experiment data stays out of Git;
  - performance claims state their measurement conditions and evidence location;
  - defaults must be backed by same-condition evidence.
- **Language**: code, comments and external docs in English; internal review documents may be in Chinese before going public.
- **CI**: every pull request and `main` run CPU checks in GitHub Actions (`.github/workflows/ci.yml`: ruff, the repository and documentation checks, CPU tests); GPU tests are run manually.

### 18.1 XInfer repository layout

The following is the planned full directory layout, which differs from the directories of the current code (see "Implementation status" for the current state).

```text
XInfer/
├── README.md
├── CONTRIBUTING.md                 # contribution conventions, evidence rules
├── pyproject.toml                  # package name to be decided (§22); extras: test / bench / kernels
├── .github/workflows/ci.yml        # CPU checks on every pull request
├── docs/
│   ├── design.md                   # this document
│   ├── architecture/               # subsystem designs: scheduler, memory, ops contracts, config resolution
│   ├── models/                     # three pages per model: usage / design / validation
│   ├── platforms/                  # rtx5090.md, thor.md
│   ├── guides/                     # adding a new model, platform, quantization method or kernel
│   └── results/                    # results index and short validation reports (raw data in external storage)
├── xinfer/
│   ├── __init__.py                 # public API: Policy, serve
│   ├── entrypoints/
│   │   ├── cli.py                  # xinfer serve | bench | tune | eval | profile
│   │   ├── serve.py                # resolve config → start an Engine per device via Platform.device_count / bind_env, start Frontends → supervise lifecycle
│   │   └── policy.py               # in-process API: Policy(model, ckpt).infer(obs)
│   ├── config/
│   │   ├── schema.py               # config dataclasses (strict; unknown fields are errors)
│   │   ├── resolve.py              # resolution pipeline: explicit > tuning table > derived > default (§10)
│   │   ├── rules.py                # derivation rules: platform caps, model declarations, memory planning; no vendor special cases (those come from Platform.derive)
│   │   └── report.py               # prints and saves each value and its source
│   ├── frontend/                   # Frontend processes; do not initialize CUDA
│   │   ├── server.py               # WebSocket server (SO_REUSEPORT), sessions
│   │   ├── protocol.py             # msgpack encoding/decoding; validation against IOSchema
│   │   ├── worker.py               # main loop: calls the model's Input/OutputProcessor
│   │   ├── shm.py                  # shared-memory slots (lease + generation)
│   │   └── channel.py              # notification channel to the Engine
│   ├── engine/                     # depends only on models/interface.py; knows no concrete model
│   │   ├── engine.py               # assembly, warmup, sealing, ready state
│   │   ├── request.py              # request object and state machine
│   │   ├── scheduler/
│   │   │   ├── scheduler.py        # scheduling/dispatch main loop (single thread)
│   │   │   ├── admission.py        # AdmitReady/Delay/Fail, per-robot limit, bounded queues
│   │   │   ├── batching.py         # zero-wait batching, compat key
│   │   │   ├── chaining.py         # chained dispatch when downstream is idle
│   │   │   └── policy.py           # StagePolicy protocol + fifo
│   │   ├── stage/
│   │   │   ├── executor.py         # StageExecutor: lane set and dispatch
│   │   │   ├── lane.py             # stream + workspace + graph pool
│   │   │   ├── graphs.py           # graph capture, sealing, bounded cache
│   │   │   └── completion.py       # completion thread
│   │   └── memory/
│   │       ├── slots.py            # slot pool and state machine
│   │       ├── handoff.py          # gather / scatter per HandoffSpec
│   │       ├── weights.py          # read-only weight freezing
│   │       └── planner.py          # memory planner
│   ├── models/
│   │   ├── interface.py            # VLAModel, StageGraph, HandoffSpec, Capabilities,
│   │   │                           # IOSchema, InputProcessor, OutputProcessor
│   │   ├── registry.py             # lazy registration (name → module:Class string) and discovery, including plugin entry points
│   │   ├── processing.py           # input processing shared by models: concurrent views, segment tokenizer
│   │   ├── wall_oss/
│   │   │   ├── config.py           # structural parameters from the checkpoint config
│   │   │   ├── model.py            # stage graph and the three stage functions (assembled from layers)
│   │   │   ├── processing.py       # IOSchema, Input/OutputProcessor
│   │   │   └── weights.py          # weight mapping
│   │   └── layers/                 # building blocks reused across models; call only ops
│   │       ├── linear.py           # quantization-aware Linear (delegates to QuantMethod)
│   │       ├── attention.py        # bidirectional vision / prefix prefill / joint attention
│   │       ├── norm.py             # RMSNorm, adaRMS
│   │       ├── rotary.py           # RoPE, M-RoPE
│   │       ├── mlp.py
│   │       ├── image_pack.py       # uint8 views → normalized images / patches (on device)
│   │       ├── vision/             # qwen2_5_vit.py
│   │       ├── backbones/          # qwen2_5.py
│   │       ├── action/             # wallx_expert.py, flow.py (with RTC)
│   │       └── quantization/       # how a Linear stores its weight and which operators it calls
│   │           ├── base.py         # QuantMethod, QuantConfig, per-module policy matching
│   │           ├── bf16.py
│   │           ├── mxfp8.py
│   │           └── nvfp4.py        # M2
│   ├── loader/
│   │   ├── safetensors.py          # sharded reads, dtype conversion
│   │   └── mapping.py              # weight-mapping rule engine (rename, split, merge)
│   ├── platforms/
│   │   ├── interface.py            # Platform, PlatformCaps
│   │   ├── detect.py               # detects sm120 / sm110 / cpu, and discovers external platforms registered via the xinfer.platforms entry point
│   │   ├── cuda.py                 # stream, event, graphs, memory budget; per-architecture capability table
│   │   ├── cpu.py                  # reference platform (for tests): threads simulate async streams and events
│   │   └── conformance.py          # platform contract suite, run by vendors in their own CI (§13.3)
│   ├── ops/
│   │   ├── spec.py                 # operator signatures and numerical contracts
│   │   ├── registry.py             # backend registration, build-time selection, selection table
│   │   ├── reference/              # torch reference implementations on every platform: attn.py, gemm.py, norm.py, flow.py, rope.py, elementwise.py, conv.py, mxfp8.py, nvfp4.py
│   │   ├── layout/                 # storage layouts of the low-precision formats: mxfp8.py, nvfp4.py
│   │   ├── backends/
│   │   │   ├── _xcompute.py        # imports the xcompute library (currently via TVM-FFI), shared by the backends
│   │   │   └── cuda/               # one module per operator category: attn.py, gemm.py, norm.py, flow.py
│   │   └── conformance/            # conformance test suite (also run in the xcompute repository CI)
│   │       ├── cases.py            # per-operator shape and dtype cases, taken from the first models
│   │       └── checks.py           # numerics, determinism, graph capture checks
│   ├── observability/
│   │   ├── timing.py               # per-request phase timing
│   │   ├── metrics.py              # Prometheus
│   │   ├── trace.py                # Perfetto export
│   │   └── roofline.py             # per-operator time and theoretical lower bound
│   ├── benchmarks/
│   │   ├── latency.py              # xinfer bench latency
│   │   ├── fleet.py                # xinfer bench fleet: virtual robot fleet, capacity search
│   │   ├── tune.py                 # xinfer tune: sweeps and generates tuning tables
│   │   └── eval.py                 # xinfer eval: alignment, open-loop error, quantization gate
│   └── tuned/                      # tuning tables, shipped with the package
│       ├── wall_oss/sm120-mxfp8.json   # values + commit + date + measurement conditions
│       └── wall_oss/sm110-mxfp8.json
├── client/                         # separate small package xinfer-client, no torch dependency
│   ├── pyproject.toml
│   └── xinfer_client/
│       ├── client.py               # WebSocket client
│       ├── trigger.py              # trigger rule: max(last send + period, response arrival)
│       ├── rtc.py                  # RTC delay estimation, chunk alignment
│       └── timing.py               # client response latency
├── tests/
│   ├── unit/                       # no GPU: scheduling, slots, batching, chaining, config resolution
│   ├── contract/                   # end-to-end with a fake model; test_layering.py checks dependency direction
│   ├── plugin/                     # out-of-tree-style dummy platform package: discovery, activation and contract tests via entry points (§13.3)
│   ├── models/                     # GPU: alignment with reference implementations (assets fetched externally per manifest)
│   ├── consistency/                # GPU: chained/non-chained, B1/Bn, idle/full load within the precision gate
│   └── fixtures/                   # small deterministic fixtures, noting the tests that use them
├── examples/                       # 5090 and Thor fleet deployment, offline Policy
├── tools/
│   ├── check_layering.py
│   └── fetch_assets.py             # fetches checkpoints and reference outputs and verifies hashes
└── docker/
    ├── Dockerfile.sm120
    └── Dockerfile.thor             # based on JetPack/L4T
```

### 18.2 The xcompute repository and the integration boundary

The existing xcompute repository and its engineering structure are kept, maintained by the operator team; no separate operator repository is created. XInfer's integration and delivery requirements are:

| Item | Convention |
|---|---|
| Implementation | xcompute provides high-performance operators for each platform, such as GEMM, fused quantization, attention, norm/RoPE and image processing, with the scope driven by the first models' operator list |
| Binding | By default wrapped and registered through TVM-FFI by XInfer's `ops/backends/cuda/`; `torch.library` adapters that reuse the underlying implementation are added for concrete PyTorch integration needs (§14.2) |
| Validation | Besides xcompute's own tests, its CI runs `xinfer.ops.conformance`, and it delivers microbenchmarks and roofline tables; XInfer adds the §14.1 and §17 validation for new `torch.library` adapters |
| Versioning and release | Build artifacts delivered per platform, recording the library version, hash and supported capabilities; XInfer checks the compatible range and the build fingerprint (§14.3) |
| Third-party dependencies | xcompute keeps records of sources and licenses, reviewed before public release |

### 18.3 Extensions: plugin entry points

XInfer does not import external extensions directly; it discovers them at startup through Python entry points. vLLM also uses plugin entry points to bring in out-of-tree models and platforms.

| Entry-point group | Purpose | Example |
|---|---|---|
| `xinfer.ops` | Register operator implementations | The xcompute binding and other vendors' operator packages |
| `xinfer.models` | Out-of-tree models | Private or third-party models |
| `xinfer.policies` | Scheduling policies | SLO-aware schedulers |
| `xinfer.platforms` | Register a Platform | Vendor NPU packages; the built-in cuda and cpu platforms use the same mechanism |
| `xinfer.quantization` | Register a QuantMethod | Vendor quantization formats |

This way SLO schedulers, private models and NDA-bound chip SDK bindings can live in separate packages, with no need to fork the repository.

**Activation protocol** (the same approach as vLLM and SGLang):

- **Entry functions** take no arguments and are reentrant. A platform entry returns `None` when the current environment is not supported and the fully qualified name of the Platform class as a string when it is; discovery does not import vendor SDKs.
- **Only one platform is activated at a time.** If several activate, it is an error asking the user to choose with `XINFER_PLATFORM`; once set, only that entry is loaded, and startup fails if it is unavailable. `XINFER_PLUGINS` filters the other entry-point groups by name.
- **Failure policy.** If an entry (platform, ops, quantization) in the selected platform's distribution fails to load, startup aborts without falling back to another platform; load failures of entries from other packages only produce a warning.
- **The startup report** prints the activated platform, the package and version each entry comes from, and the API versions of both sides (§13.3).

### 18.4 Milestones and directories

M0–M3 all use `colocated` as the GPU execution layout and include no split execution backend or inter-stage IPC/MPS implementation; the CPU multi-process communication in `frontend/` is still implemented as planned.

| Milestone | Mainly in |
|---|---|
| M0 | `platforms/`, `ops/`, `models/layers/`, `models/wall_oss`, engine stages and slots, policy and benchmark entry points |
| M1 | `frontend/` (multi-process), all of `engine/scheduler/`, `memory/planner.py`, `config/resolve.py`, `benchmarks/fleet.py`, `tune.py`, `tuned/`, `client/` |
| M2 | `models/layers/quantization/nvfp4.py`, precision gates and building blocks for model extensions |
| M3 | Compatibility adapters, Router, docs, Docker images, GitHub release |

---

## 19. Relationship to existing code

**A new repository instead of evolving an existing one.** The earlier internal serving code mixed ported private code and SLO scheduling with generic scheduling, depended on undeclared builds, and ported kernels one by one for each model and GPU. XInfer keeps what proved itself there:

- generic admission, the queue chain, lane limits, slot allocation and recycling, draining on failure, and the slot state machine;
- the batching rules, the frontend process pool and shared memory;
- executor/engine separation (prepare → engine → decode), build-time capability checks, sealing graphs after warmup and strict configuration loading;
- admission results (`AdmitReady` / `AdmitDelay` / `AdmitFail`), slot handle generations, and `/ready` returning 503 until warmup completes.

What is carried over are the logical stages and the batching and ownership rules; this version does not carry over the GPU-stage ActionProcess, the CUDA IPC transfer or the MPS start/stop implementation. The CPU Frontend process pool and shared memory are kept as originally planned.

### Coupling left behind

- the hard-coded RTC delay range (`3 <= delay <= 8`);
- the assumption of a fixed handle structure;
- threads and environment variables named after Wall-X;
- the hard-coded 5.6 Hz;
- ablation gates imported on the hot path;
- camera names and dual arms hard-coded in generic types.

---

## 20. Milestones and acceptance

| Milestone | Content | Acceptance |
|---|---|---|
| M-1 Review and repository setup | Main repository created, xcompute confirmed as the operator repository; review of this document and of the operator contracts pending; CPU CI set up | Repository arrangement decided; review and acceptance conclusions recorded once the other items of this milestone are done |
| M0 Data plane | Platform, operator backends, layers, Wall-OSS, single-lane dispatch and benchmarks | Wall-OSS meets its precision gate and records latency and graph dispatch costs on the target platform. |
| M1 Fleet runtime | Multi-lane dispatch, zero-wait batching, frontend workers, slot lifecycle and configuration resolution | Measure fleet capacity, batched precision and dispatch overhead under load. |
| M2 Quantization and extension | NVFP4 and model extensions | Record numerical results for each model and quantization method. |
| M3 Release | Docs, Client SDK, compatibility adapters, Router; license approval; GitHub release | External users can reproduce single-request latency and per-GPU capacity by following the docs |

---

## 21. Main risks

| Risk | Impact | Mitigation |
|---|---|---|
| Chained dispatch lowers capacity near the critical load | The premise of the unified model | Measure on the colocated multi-lane path in M1; adjust the chaining condition if needed, keeping a single path |
| Single-process dispatch becomes the bottleneck (Python overhead, GIL) | Fleet capacity | M1 measures dispatch-thread occupancy, lock waits and submission delay; first remove blocking and heavy CPU work from the hot path, and if needed evaluate moving the submission logic to a lower level. If the capacity bar is not met, keep optimizing and record the limits, without relying on the unimplemented split as a fallback; process splitting needs a separate review |
| Thor's graphs, unified memory and CPU submission overhead differ from 5090 | Thor performance | Derive configuration from platform capabilities; separate M0 measurements on Thor. Thor is also deployed as a fleet, so M1 measures capacity and Frontend CPU usage on Thor separately. v1 does not depend on MPS, and confirming MPS support does not block this version |

---

## 22. Open decisions

1. **External name conflict**: `xinfer` on PyPI is taken by dnth/xinfer (a computer-vision inference framework), and Xorbits' Xinference has a similar name. The repository stays XInfer; the pip package name is decided before going public.
2. **Model weights**: external distribution of checkpoint files requires separate approval.
3. **Scope and versions of PyTorch operator adapters**: the strategy of TVM-FFI by default with `torch.library` added as needed is settled (§0, §14.2); confirm with the operator team which operators need adapters first, which compile/export paths are actually supported, and the PyTorch, TVM-FFI and xcompute version ranges.
4. **License**: to be approved before M3.
5. **External stable surface and support tiers**: the starting API version number and the rules for breaking changes, and the admission criteria for Tier 2 (§13.3).
6. **Timing and GPU models for AMD GPUs**: whether they enter v1; the GPU model determines the available quantization formats and kernel sources (§13.2).

