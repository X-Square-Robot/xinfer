# Cluster serving design

English | [简体中文](cluster_design.zh-CN.md)

This page designs serving one robot fleet from many GPUs on one or more machines. It covers:
- a router in front of single-GPU workers;
- failover, workers joining and leaving, and health checks;
- pluggable routing policies.

It is a design for review. The worker (§4), the router (§5) and example Kubernetes manifests (§6) are implemented; the [design](design.md) status table tracks them. It extends
design §2.2 ("multiple GPUs use data-parallel replicas plus a Router") and the optional Router of §4.

## 1. Goals and scope

| Goal | Criterion |
|---|---|
| Serve a fleet from many GPUs | A pool of k workers serves at least 0.95 × k × the single-GPU capacity |
| Survive a failure | When a worker, its GPU or its machine fails, each robot loses at most one response (a stateful connection starts over once, §4.5), and no robot connection closes |
| Add and remove GPUs without a restart | Workers join and leave a running pool; a drained worker leaves without a failed response |
| Health checks | A dead, hung or draining worker receives no requests |
| Pluggable routing policies | Built-in policies, and more through an entry point |
| Keep the single-GPU path | One worker's single-request latency and capacity do not change |

Out of scope:
- tensor parallelism;
- stages on different GPUs (design §9.3);
- an `xinfer-serve` that runs several GPUs in one process group. A worker is one GPU even when a machine has
  several, and the router serves both cases;
- SLO schedulers in the core: they can be routing policy plugins;
- a process supervisor or service registry of our own: Kubernetes provides them (§6).

## 2. Workload facts the design rests on

| Fact | Consequence |
|---|---|
| No engine keeps state between requests (no cross-request KV cache) | Any worker can serve any request of a stateless pool, and a request can be retried on another worker |
| A model may keep per-connection state in the frontend's InputProcessor (`connect`, design §11), such as real-time chunking plans; the bundled models keep none | Such a model's requests must stay with one worker for the connection's life (affinity) |
| A robot sends in a closed loop with one request in flight; one GPU serves 4–15 robots ([performance](performance.md)) | Load is counted in robots and requests in flight; admission caps robots per pool |
| Requests are large (three camera views) and latency-critical | The router forwards messages without decoding them; its added latency is an acceptance criterion (§8) |

## 3. Architecture

```text
 robots ──WebSocket + msgpack──▶ xinfer-router ×2 (stateless; one listening port per pool)
                                   │ per request (stateless pools) or per connection (stateful)
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
          worker: xinfer-serve on one GPU (unchanged single-GPU server: frontends + engine)
          GET /healthz /ready /info /metrics; SIGTERM drains
 discovery: Kubernetes headless Service (DNS: ready workers only) or a static list
```

| Term | Meaning |
|---|---|
| worker | one `xinfer-serve` process group on one GPU: its frontends and its engine |
| pool | the workers serving one model configuration; robots of a pool connect to one router port, and every worker of a pool must send the same metadata frame |
| router | `xinfer-router`: terminates robot connections and forwards each request to a worker of the pool |

## 4. Worker changes

### 4.1 Endpoints

| Endpoint | Answer |
|---|---|
| `GET /healthz`, `GET /ready` | 200 while the worker listens. A worker whose engine can no longer serve exits (§4.3), and a draining one stops listening (§4.2), so a refused connection means the worker is not ready |
| `GET /info` | JSON: model, model options, quantization, numerics, platform, XInfer version, the metadata frame's hash, `connection_state` (whether the model's InputProcessor keeps per-connection state), `capacity` (robots, from `--capacity`: the single-GPU capacity measured for the GPU model and model configuration, as in [performance](performance.md); null when not set) |
| `GET /metrics` | unchanged ([metrics](metrics.md)) |

### 4.2 Draining

SIGTERM drains the worker ([serving](serving.md)):
- its frontends stop listening and answer further requests `unavailable`, which the router retries elsewhere (§5.5);
- they answer the requests they had received, then close the connections with code 1001;
- the worker exits once those are answered, at most `--drain-timeout`.

On Kubernetes, `terminationGracePeriodSeconds` must exceed `--drain-timeout`.

### 4.3 Watchdog and request timeout

- **The engine watchdog** is `Engine.health` with `--hang-timeout` ([serving](serving.md)): when a request admitted longer ago than the timeout has not finished, or the device reports a sticky error, the worker exits with status 1, as when an engine thread stops.
  - A hung kernel can outlive its process.
  - Recovering the GPU is left to the platform (§6).
- **Request timeout.** The frontend answers a request that the engine has not finished within `--request-timeout` with the status `timeout`, so the router can retry it.

### 4.4 Robot identity

A request carries the robot's id as the top-level field `robot_id`, beside `state` and `views`. The worker uses it for its per-robot limit, across connections. A request without it is limited per connection, as today.

### 4.5 Stateful connections after a failover

A model that keeps per-connection state, such as the real-time chunking plans it returned, loses it when the robot's connection moves to another worker. The new worker answers as for a new connection, and the model's protocol tells the robot to start over (for example, to re-anchor with a plain request). Avoiding that would need the state to follow the robot and a change to the client's protocol.

## 5. Router

### 5.1 Connections and forwarding

Robots connect to the router as they connect to a server today: the same protocol, and the router's address in place of the server's. One router process serves one pool on one port ([router](router.md)).

The router terminates the robot's WebSocket:
- It sends the pool's metadata frame, read from the first worker. A worker joins the pool only if its `/info` carries the same metadata hash.
- It forwards each request message unchanged and sends the worker's answer back. It reads only top-level fields, without decoding the others: an answer's `status`, and an affinity pool's `robot_id` (§4.4) from a connection's first request.
- Each robot connection has its own connection to each worker it uses, which carries one request at a time.

### 5.2 Routing

| Pool | Rule |
|---|---|
| `connection_state` false (Wall-OSS, Wall-X 2B) | per request: the worker with the fewest of the router's requests in flight (§5.7) relative to its `capacity`; ties go round |
| `connection_state` true | affinity: at its first request, a robot connection is placed by rendezvous hashing on its `robot_id` among the workers below capacity (without an id: the fewest robots relative to capacity), and stays there; if that worker fails, the connection is placed again, and the robot starts over on it (§4.5) |

### 5.3 Discovery

- **Sources.** A pool's workers come from a static list (`host:port`, for development and machines without Kubernetes) or from a DNS name resolved every second (`dns:<name>:<port>`). A Kubernetes headless Service returns only ready pods.
- **Joining and leaving.** A new address joins once its `/info` answers with the pool's metadata hash; a mismatching worker is left out and logged. An address that DNS no longer returns leaves.

### 5.4 Health

The router checks workers passively:
- A connection error, a closed upstream connection, or an answer with a retried status (§5.5) counts as a failure; a success clears the count.
- After 3 failures in a row a worker gets no requests for 5 s. Then the next request probes it, and one more failure pauses it again.
- When every worker is paused, the router routes to all of them rather than refusing every robot.

On Kubernetes, readiness probes take workers that do not answer `/ready` out of DNS.

### 5.5 Retries

A request is retried on another worker only when every condition holds. For an affinity pool, that worker is the connection's new one (§5.2):
- the robot has received nothing for it yet;
- a worker of the pool has not been tried for it yet: each worker is tried at most once, so a request is served
  while several workers drain at once;
- it was received less than `--retry-deadline-ms` ago (default 30 ms).

Otherwise the router answers the failure at once and the robot sends a new request. With real-time chunking the robot keeps executing while it waits, so a late action is of no use.

| Retried | Not retried |
|---|---|
| transport failure, closed upstream connection, `timeout`, `queue_full`, `unavailable` | `invalid_request`, model protocol errors (`error_info`), `internal_error` |

### 5.6 Admission

The pool's capacity is the sum of its workers' `capacity`, when every worker reports one. A new robot connection beyond it is refused at the handshake with HTTP 503, as is any connection while no worker has joined.

### 5.7 Load signal

The router counts its own requests in flight to each worker, from receipt to the worker's answer, so preprocessing is included. The count is exact while one router serves the pool. With several routers, each sees its share of the traffic; workers report no load of their own.

### 5.8 Availability and data plane

- **Router failure.** Routers are stateless and run as two or more replicas behind one address. The affinity hash is the same in every replica. When a router fails, its robots reconnect to the address.
- **Data plane.** The router is Python asyncio. If its added latency fails §8, for example with raw camera frames, the data plane moves to a compiled language.
- **Added when needed:** other routing policies (such as the power of two choices for many routers), router metrics, active health checks and immediate `overloaded` answers.

## 6. Kubernetes deployment

| Object | Settings |
|---|---|
| worker Deployment (one per pool) | `nvidia.com/gpu: 1` per pod; a CPU request sized for the frontends; `startupProbe` on `/ready` long enough for warmup; `readinessProbe` `/ready` every second; `livenessProbe` `/healthz`; `terminationGracePeriodSeconds` beyond the drain (§4.2); `minReadySeconds` beyond the routers' discovery by DNS (a few seconds), so that a rolling update drains the next old worker only after the routers found the new one; the checkpoint on a read-only volume; `hostNetwork` optional, to avoid the overlay network's latency |
| headless Service (one per pool) | selects the pool's workers; `publishNotReadyAddresses: false`; the router's discovery name |
| router Deployment and Service | two replicas; the Service is the address robots connect to, exposed as a server's address is today |
| PodDisruptionBudget | at most one worker of a pool disrupted at a time |
| GPU health | the GPU Operator's DCGM health checks mark a failed GPU unschedulable, so a restarted worker does not land on it again |

Manifests and an image ship as examples ([router](router.md#kubernetes)). Outside Kubernetes, the same router and workers run with a static list.

| Operation | What happens |
|---|---|
| add GPUs | raise the worker replicas; new pods become ready after warmup and join through discovery |
| remove a worker | the pod's SIGTERM drains it (§4.2) |
| rolling update | the Deployment replaces workers one at a time (`maxSurge: 0` without a spare GPU); each old worker drains, the next one after the new worker has been ready for `minReadySeconds` |
| worker crash or GPU hang | liveness fails, Kubernetes restarts the pod; the router's passive checks stop routing to it within a few failed requests |
| machine failure | its pods are rescheduled; the router's checks drop them meanwhile |

## 7. Failure handling

| Failure | Detected by | What robots see | Recovery |
|---|---|---|---|
| worker process exits | closed upstream connections | the request is retried, at most one late or failed response; a stateful connection starts over on its new worker (§4.5) | Kubernetes restarts the pod |
| GPU hang | watchdog (§4.3), request timeout | as above | the watchdog exits the worker; the pod restarts; a failed GPU is marked unschedulable |
| worker overloaded | requests in flight, `queue_full` | the request goes to another worker | — |
| machine failure | active and passive checks | as for a worker exit, for its workers | pods rescheduled |
| router failure | robot connection closes | robots reconnect to the Service address | Kubernetes restarts the router |
| pool full | admission | new robots refused at the handshake | add workers |

## 8. Validation and acceptance

Phases C1 and C2 are tested on one machine with three RTX 5090s and a static list. Phase C3 is tested on a single-node k3s cluster where two worker pods share one RTX 5090 (time-slicing in the NVIDIA device plugin), which checks the mechanisms, not capacity.

| Criterion | Measurement |
|---|---|
| Single-GPU path unchanged | one worker's single-request latency and capacity, against main, within run-to-run variation |
| Router latency | client response latency with and without the router at capacity, three-camera JPEG requests: P50 added at most 1 ms |
| Scaling | k workers serve at least 0.95 × k × the single-GPU capacity, by the repository's load convention |
| Worker failure | kill a worker under load: each robot loses at most one response, and no robot connection closes; report the time until the router stops routing to it |
| Drain and rolling update | no failed response |
| Router logic | CPU unit tests with stub workers: routing, retries, circuits, discovery, admission |

## 9. Phases

| Phase | Content |
|---|---|
| C1 worker | §4: `/info`, request timeout, robot identity |
| C2 router | §5: forwarding, both routing modes, discovery (static, DNS), passive health checks, retries, admission |
| C3 Kubernetes | §6: example manifests, probes, draining on rollout, GPU health; validation on a GPU cluster |
| C4 | as needed: `EndpointSlice` discovery if DNS is too slow, a compiled data plane, the additions of §5.8, SLO policy plugins |

## 10. Related systems

From the sources at vLLM `9b38c83` and SGLang `62eab14`:

| | vLLM data parallelism | SGLang routers (`sgl-router`, `sgl-model-gateway`) | This design |
|---|---|---|---|
| Topology | every frontend connects to every engine; engine load reaches the frontends through a coordinator (on change, at most every 100 ms) | separate Rust processes in front of independent workers | separate router in front of independent single-GPU workers |
| Engine failure | the whole server exits | circuit breaker; `sgl-model-gateway` also checks `/health` every 60 s | the worker is paused after 3 failures in a row, its requests retried |
| Load balancing | lowest waiting plus running, adjusted locally after each pick | power of two choices, cache-aware and others | §5.2 |
| Membership | fixed size | static list, Kubernetes; the gateway also has a workers API | static list, DNS (Kubernetes headless Service) |
| Draining | `--shutdown-timeout` | readiness 503, a drain window, then in-flight requests | readiness 503, a drain window, then in-flight requests |
