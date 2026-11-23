# Performance

English | [简体中文](performance.zh-CN.md)

Single-accelerator measurements for WALL-OSS-1.0-Nano from `benchmarks/performance.py` using the production robot request format (three camera views per request). Single-request P50 and fleet-load figures come from separate runs.

<table>
  <thead>
    <tr>
      <th scope="col" rowspan="2">Model</th>
      <th scope="col" rowspan="2">Precision and stages</th>
      <th scope="col" rowspan="2">Accelerator</th>
      <th scope="col">Single-request latency</th>
      <th scope="col" colspan="3">Closed-loop fleet serving</th>
    </tr>
    <tr>
      <th scope="col">P50</th>
      <th scope="col">Capacity (robots)</th>
      <th scope="col">Server P50 / P95 at capacity</th>
      <th scope="col">Peak throughput</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td>RTX 5090</td>
      <td>20.1 ms</td>
      <td>14</td>
      <td>85 / 107 ms</td>
      <td>81.3 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>MXFP8 vision and prefill</td>
      <td>RTX 5090</td>
      <td>15.1 ms</td>
      <td>25</td>
      <td>67 / 88 ms</td>
      <td>141.0 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td><a href="https://github.com/X-Square-Robot/xinfer/pull/33">Ascend 950DT</a></td>
      <td>20.6 ms</td>
      <td>15</td>
      <td>129 / 163 ms</td>
      <td>83.9 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td>Jetson AGX Thor</td>
      <td>71.0 ms</td>
      <td>2</td>
      <td>82 / 88 ms</td>
      <td>16.0 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>NVFP4 vision and prefill</td>
      <td>RTX 5090</td>
      <td>TBD</td>
      <td colspan="3">TBD</td>
    </tr>
  </tbody>
</table>

Capacity and server P50 / P95 describe the same robot count. Peak throughput is the highest steady-state request completion rate across all tested robot counts and rounds.

## Measurement protocol

- **Single-request P50** measures in-process latency from a serialized request to its response. It includes image decoding and resizing, model inference, and output processing.
- **Robots per accelerator**, **server P50 / P95**, and **peak throughput** report closed-loop serving load. See the [benchmark protocol](benchmark.md#performance-table) for the capacity rule and aggregation.

## RTX 5090

The two single-request P50s were measured on November 21, 2026 on one idle RTX 5090 with SM120 kernel revision `4a62de1`, PyTorch 2.10.0+cu130, and FlashAttention 2.8.3.post1. Camera views were decoded and resized on the GPU. The single-request input set contained 216 recorded requests with three camera JPEGs each.

The BF16 fleet figures are from the November 13 run on SM120 kernel revision `7336b0d`. The MXFP8 fleet figures are from the November 17 run on revision `f4cdb84`. Both used synthetic robot observations with real-time chunking. The [MXFP8 preset](models/wall_oss_1_0_nano_design.md#mxfp8) quantizes selected vision and prefill linear layers and keeps the action stage in BF16.

## Jetson AGX Thor

On November 23, 2026, the public SM110 build from XInfer `453e82e` (library SHA-256 prefix `dc44beaf6a32`) ran Nano BF16 with GPU image processing and PyTorch 2.10.0+cu130. The single-request P50 uses 50 timed runs. The closed-loop sweep ran eight rounds at each of one, two and three robots. At three robots the median server P50 rose from 82.3 to 180.5 ms, crossing the capacity threshold. All rounds completed with zero errors and zero missing requests.

## Ascend 950DT

Public build artifacts and reproduction steps for this BF16 measurement are **TBD**. See the [Ascend guide](ascend950dt.md) for device setup and operator configuration. In-tree Ascend kernels are tracked in [PR #33](https://github.com/X-Square-Robot/xinfer/pull/33).
