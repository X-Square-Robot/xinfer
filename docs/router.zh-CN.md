# Router

[English](router.md) | 简体中文

`xinfer-router` 在同一个池的单卡服务（`xinfer-serve`）前面提供一个端口，这些服务可以在一台机器上，也可以分布在多台机器上。机器人连接它的方式与连接服务端相同。设计见[集群服务](cluster_design.zh-CN.md)。

## 运行

每张 GPU 启动一个服务，各自带上按其 GPU 和模型配置实测的容量（[性能](performance.zh-CN.md)），然后启动 router：

```bash
CUDA_VISIBLE_DEVICES=0 xinfer-serve --model wallx2b --checkpoint /path/to/ckpt --port 34001 --capacity 15 &
CUDA_VISIBLE_DEVICES=1 xinfer-serve --model wallx2b --checkpoint /path/to/ckpt --port 34002 --capacity 15 &
xinfer-router --workers 127.0.0.1:34001 127.0.0.1:34002 --port 33985
```

在 Kubernetes 上，`--workers` 写池的 headless Service：`dns:<service>:<port>`（[Kubernetes](#kubernetes)）。

| 选项 | 含义 |
|---|---|
| `--workers` | 服务：`host:port`，或 `dns:<name>:<port>`（名称的所有地址，每秒解析一次） |
| `--host`, `--port` | 机器人连接的地址（默认 `0.0.0.0:33985`） |
| `--retry-deadline-ms` | 失败的请求在不超过这个时长时依次改发给其他服务，超过则立即返回失败（默认 30） |

## 行为

- **池。** 服务的 `GET /info` 返回后即加入池。第一个服务决定池的元数据帧，router 把它发给机器人；元数据不同的服务不加入。
- **路由。**
  - 没有按连接状态的模型：每个请求发给 router 在途请求数相对其容量最少的服务。
  - 连接上有状态的模型（`GET /info` 中的 `connection_state`）：一台机器人的连接一直留在同一个服务上，在第一个请求时按请求的 `robot_id` 选定。该服务故障时，连接转到另一个服务，那里没有这条连接的状态（[集群服务](cluster_design.zh-CN.md#45-故障转移后有状态的连接)）。
- **故障。**
  - 连续 3 个请求失败（连接错误、连接被关闭，或 `timeout`、`queue_full`、`unavailable`）的服务，5 s 内不再收到请求。
  - 失败的请求在不超过 `--retry-deadline-ms` 时依次改发给其他服务，每个服务最多一次，所以几个服务同时排空时也能得到服务；否则立即返回失败（没有任何服务回答时返回 `unavailable`）。
- **准入。** 每个服务都报告了 `--capacity` 时，超出容量之和的机器人连接以 HTTP 503 拒绝。
- **健康检查。** `GET /healthz` 返回 200；有服务加入后 `GET /ready` 返回 200。

## Kubernetes

[examples/kubernetes](../examples/kubernetes) 在两个 router 后面运行一个 Wall-X 2B 服务池：
- `Dockerfile`：服务和 router 共用的镜像，包含 torch 2.10（CUDA 13.0 wheel）、为 RTX 5090 编译的 flash-attn（`FLASH_ATTN_CUDA_ARCHS`）、xcompute wheel 和 XInfer；
- `xinfer.yaml`：
  - 服务的 Deployment：每个 Pod 一张 GPU，探针访问 `/ready` 和 `/healthz`，终止宽限期大于排空时间；
  - headless Service `xinfer-workers`，供 router 解析；
  - PodDisruptionBudget；
  - router 的 Deployment，机器人连接节点端口 30985。

集群需要装 NVIDIA device plugin，Pod 才能申请 `nvidia.com/gpu`。构建镜像时，xcompute wheel（为部署所用 GPU 编译，见 [kernel](kernels.zh-CN.md)）来自构建上下文 `xcompute`，它是一个只放这一个 wheel 的目录。checkpoint 和预热请求来自 `/srv/xinfer` 下的宿主机路径；按实际情况修改，或换成集群自己的卷。`--capacity` 填该模型在这种 GPU 上实测的容量（[性能](performance.zh-CN.md)）。

```bash
docker build -f examples/kubernetes/Dockerfile --build-context xcompute=/path/to/wheel-dir -t xinfer:latest .
# 让节点能拿到镜像：推到镜像仓库，或者导入（k3s：k3s ctr images import）
kubectl apply -f examples/kubernetes/xinfer.yaml
kubectl rollout status deployment/xinfer-worker
```

- **加入。** 服务 Pod 加载并预热完、端口开始应答后就绪。此后 `xinfer-workers` 才解析到它，router 在几秒内把它加入池（DNS TTL 为 5 s，router 每秒解析一次）。
- **扩缩。** `kubectl scale deployment/xinfer-worker --replicas=N`。被移除的服务先排空（[服务](serving.zh-CN.md)），router 把它的请求发给其他服务。
- **升级。** 换镜像或 `kubectl rollout restart` 会逐个替换服务（`maxSurge: 0`，`maxUnavailable: 1`），不需要空闲 GPU；替换期间池里少一个服务。`minReadySeconds: 10` 让下一个旧服务一直工作到 router 发现新服务为止；不设的话，两个服务的池可能有几秒钟没有可用的服务。有空闲 GPU 时，`maxSurge: 1`、`maxUnavailable: 0` 可以保持容量不降。
- **故障。** 服务退出（引擎故障、挂起、崩溃）后由 Kubernetes 重启，就绪后重新加入；在此期间 router 把它的请求发给其他服务。
- **Router。** 节点端口把机器人连接分散到各个 router 副本上。router 停止后，连在它上面的机器人重新连接。

## 限制

- 一个 router 进程服务一个池；每种模型配置启动一个 router。
- router 只统计自己的在途请求和机器人连接：有多个 router 时，每个只看到自己那部分流量，并且每个都按池的全部容量准入机器人。
- DNS 名称只解析为 IPv4 地址。
