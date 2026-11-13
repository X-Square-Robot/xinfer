# Router

English | [简体中文](router.zh-CN.md)

`xinfer-router` puts one port in front of the single-GPU servers (`xinfer-serve`) of one pool, on one machine or
several. Robots connect to it as they connect to a server. The design is in [cluster serving](cluster_design.md).

## Run

Start one server per GPU, each with the capacity measured for its GPU and model configuration
([performance](performance.md)), then the router:

```bash
CUDA_VISIBLE_DEVICES=0 xinfer-serve --model wallx2b --checkpoint /path/to/ckpt --port 34001 --capacity 15 &
CUDA_VISIBLE_DEVICES=1 xinfer-serve --model wallx2b --checkpoint /path/to/ckpt --port 34002 --capacity 15 &
xinfer-router --workers 127.0.0.1:34001 127.0.0.1:34002 --port 33985
```

On Kubernetes, give `--workers` the pool's headless Service: `dns:<service>:<port>` ([Kubernetes](#kubernetes)).

| Option | Meaning |
|---|---|
| `--workers` | the servers: `host:port`, or `dns:<name>:<port>` for every address of a name, resolved every second |
| `--host`, `--port` | where robots connect (default `0.0.0.0:33985`) |
| `--retry-deadline-ms` | a failed request is sent to the other servers in turn while younger than this, else its failure is answered at once (default 30) |

## Behavior

- **Pool.** A server joins once its `GET /info` answers. The first sets the pool's metadata frame, which the router
  sends to robots; a server with other metadata is left out.
- **Routing.**
  - A model without per-connection state: each request goes to the server with the fewest of the router's requests
    in flight relative to its capacity.
  - A model whose connections keep state (`connection_state` in `GET /info`): a robot's connection stays with one
    server, chosen at its first request from the request's `robot_id`. If that server fails, the connection moves to
    another, which has none of the connection's state ([cluster serving](cluster_design.md#45-stateful-connections-after-a-failover)).
- **Failures.**
  - A server that fails 3 requests in a row (connection error, closed connection, or `timeout`, `queue_full`,
    `unavailable`) gets no requests for 5 s.
  - A failed request is sent to the other servers in turn, each once, while it is younger than `--retry-deadline-ms`,
    so it is served when several servers drain at once; else its failure is answered at once (`unavailable` when no
    server answered).
- **Admission.** When every server reports `--capacity`, a robot connection beyond their sum is refused with HTTP 503.
- **Health.** `GET /healthz` returns 200; `GET /ready` returns 200 once a server has joined.

## Kubernetes

[examples/kubernetes](../examples/kubernetes) runs one pool of Wall-X 2B servers behind two routers:
- `Dockerfile`: one image for the servers and the router, with torch 2.10 (CUDA 13.0 wheels), flash-attn built for
  the RTX 5090 (`FLASH_ATTN_CUDA_ARCHS`), the xcompute wheel, and XInfer;
- `xinfer.yaml`:
  - the server Deployment: one GPU per pod, probes on `/ready` and `/healthz`, a termination grace period beyond
    the drain;
  - the headless Service `xinfer-workers`, which the routers resolve;
  - a PodDisruptionBudget;
  - the router Deployment, with node port 30985 for the robots.

The cluster needs the NVIDIA device plugin, so that pods can request `nvidia.com/gpu`. The image build takes the
xcompute wheel for the deployment GPUs ([kernels](kernels.md)) from the build context `xcompute`, a directory that
holds only that wheel. The checkpoint and the warmup requests come from host paths under `/srv/xinfer`; adapt them,
or use your cluster's volumes. Set `--capacity` to the model's measured capacity on the GPU
([performance](performance.md)).

```bash
docker build -f examples/kubernetes/Dockerfile --build-context xcompute=/path/to/wheel-dir -t xinfer:latest .
# make the image available to the nodes: push it to a registry, or import it (k3s: k3s ctr images import)
kubectl apply -f examples/kubernetes/xinfer.yaml
kubectl rollout status deployment/xinfer-worker
```

- **Joining.** A server pod is ready once its port answers, after loading and warmup. Only then does
  `xinfer-workers` resolve to it, and the routers add it within a few seconds (a 5 s DNS TTL, resolved every second).
- **Scaling.** `kubectl scale deployment/xinfer-worker --replicas=N`. A removed server drains
  ([serving](serving.md)); the routers send its requests to the others.
- **Updates.** A new image or `kubectl rollout restart` replaces the servers one at a time (`maxSurge: 0`,
  `maxUnavailable: 1`), which needs no spare GPU; the pool runs one server short meanwhile. `minReadySeconds: 10`
  keeps the next old server serving until the routers have found the new one; without it, a two-server pool can be
  left with no server for a few seconds. With a spare GPU, `maxSurge: 1` and `maxUnavailable: 0` keep the capacity.
- **Failures.** Kubernetes restarts a server that exits (engine failure, hang, crash), and it rejoins once ready; the
  routers send its requests to the others meanwhile.
- **Routers.** The node port spreads robot connections over the router replicas. A robot whose router stops
  reconnects.

## Limitations

- One router process serves one pool; start one router per model configuration.
- The router counts only its own requests in flight and robot connections: with several routers, each sees its
  share of the traffic, and each admits robots up to the pool's capacity.
- DNS names resolve to IPv4 addresses only.
