# ros2-gpu-pointcloud

[![CI](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml)

A ROS 2 Jazzy node that range-filters and voxel-downsamples point clouds, with a single-threaded C++17 backend and a CUDA backend that produce the same output, plus a benchmark of the two.

**Work in progress.** The filter, both backends, the node and the benchmark exist. The DDS vs Zenoh transport comparison and the full README are not done yet; see [docs/STATUS.md](docs/STATUS.md).

## Benchmark

Median of 200 timed iterations after 20 warmup iterations, default parameters (voxel 0.05 m, range 0.3 to 10 m), synthetic depth-camera cloud. Raw data: [results/bench.csv](results/bench.csv), [results/stages.csv](results/stages.csv), [results/env.txt](results/env.txt).

Environment: NVIDIA GeForce RTX 4070 Laptop GPU (driver 616.64, CUDA 12.6), Intel Core i9-14900HX, Docker on WSL2 (kernel 5.15.167.4), laptop on AC power.

| Resolution | Points | CPU ms | CUDA end-to-end ms | CUDA GPU-only ms | Speedup end-to-end | Speedup GPU-only | Output points |
|---|---:|---:|---:|---:|---:|---:|---:|
| 320x240 | 76,800 | 1.825 | 0.461 | 0.232 | 3.96x | 7.86x | 20,785 |
| 640x480 | 307,200 | 7.914 | 1.049 | 0.370 | 7.55x | 21.37x | 34,654 |
| 1280x720 | 921,600 | 25.683 | 2.439 | 0.586 | 10.53x | 43.80x | 39,514 |

99th percentile (ms):

| Resolution | CPU | CUDA end-to-end | CUDA GPU-only |
|---|---:|---:|---:|
| 320x240 | 2.094 | 0.624 | 0.365 |
| 640x480 | 9.462 | 1.313 | 0.592 |
| 1280x720 | 31.828 | 12.256 | 3.738 |

How to read this:

- **CPU** is a single-threaded `-O3` baseline (key, `std::sort`, linear reduce). It is not multi-threaded and not PCL, so the speedups are against that baseline only.
- **End-to-end** is host input to host output: the copy into pinned memory, the host-to-device transfer, the device pipeline, the transfer back and the final synchronisation, timed with `std::chrono::steady_clock`.
- **GPU-only** is the device pipeline between the two transfers, timed with cudaEvents.
- At 1280x720 the input transfer is the largest single stage (1.66 ms of 2.44 ms), so the end-to-end speedup is bounded by the copy, not by the kernels.
- The CUDA tail at 1280x720 (p99 12.3 ms against a 2.4 ms median) is real and unexplained; this is a laptop GPU shared with the Windows desktop through WSL2.
- Both backends return the same voxel count, and on these synthetic clouds identical centroids. The tested tolerance is 1e-4 m per axis.
- The clouds are synthetic. Real sensor data will have a different ratio of input points to occupied voxels.

Reproduce inside the dev container:

```
docker compose -f docker/compose.dev.yml up -d --build
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build && colcon test && colcon test-result --verbose'
docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/bench.sh
```

## Tests and CI

CI builds and tests the CPU path and compiles the CUDA code. GitHub runners have no GPU, so the CUDA parity tests run only locally (they skip themselves when no GPU is visible).
