# ros2-gpu-pointcloud

[![CI](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml)

A ROS 2 Jazzy node that range-filters and voxel-downsamples point clouds. It has a single-threaded C++17 backend, an OpenMP backend and a CUDA backend that produce the same output, and a benchmark that compares them with each other and with PCL.

**Work in progress.** See [docs/STATUS.md](docs/STATUS.md) for what exists and what is next.

## Benchmark

Median of 200 timed iterations after 20 warmup iterations, default parameters (voxel 0.05 m, range 0.3 to 10 m), synthetic depth-camera cloud, packed xyz input (12 bytes per point). Raw data is in [results/](results): `bench.csv`, `cuda_launches.csv`, `cpu_omp_threads.csv`, `stages.csv`, `cuda_input_modes.csv`, `env.txt`.

Environment: NVIDIA GeForce RTX 4070 Laptop GPU (driver 616.64, CUDA 12.6), Intel Core i9-14900HX with 20 logical CPUs visible to the container, PCL 1.14.0, Docker on WSL2 (kernel 5.15.167.4), laptop on AC power.

### Headline: CUDA against the best CPU baseline

The best CPU baseline at every resolution is this repository's OpenMP backend on 20 threads.

| Resolution | Points | Best CPU (OpenMP, 20 threads) ms | CUDA end-to-end ms | CUDA GPU-only ms | End-to-end speedup | GPU-only speedup |
|---|---:|---:|---:|---:|---:|---:|
| 320x240 | 76,800 | 0.247 | 0.430 | 0.244 | 0.57x | 1.01x |
| 640x480 | 307,200 | 0.876 | 0.792 | 0.335 | 1.11x | 2.62x |
| 1280x720 | 921,600 | 2.774 | 1.702 | 0.617 | 1.63x | 4.49x |

Measured end to end on this machine, the GPU is slower than 20 CPU threads at 320x240, 11% faster at 640x480 and 1.6x faster at 1280x720.

**The CUDA end-to-end time depends on the process launch.** The same binary run seven times under the same protocol gave these medians ([results/cuda_launches.csv](results/cuda_launches.csv)):

| Resolution | Fastest launch ms | Median launch ms | Slowest launch ms | Speedup vs best CPU, slowest to fastest |
|---|---:|---:|---:|---:|
| 320x240 | 0.371 | 0.394 | 0.512 | 0.48x to 0.67x |
| 640x480 | 0.697 | 0.769 | 1.051 | 0.83x to 1.26x |
| 1280x720 | 1.471 | 1.521 | 2.397 | 1.16x to 1.89x |

At 1280x720 the launches fall into two groups, five near 1.5 ms and two near 2.4 ms. The difference is entirely in the host-to-device transfer stage, which takes about 0.93 ms or about 1.74 ms for the same 11 MB. In the slow launches plain host `memcpy` and the CPU filter run at their normal speed, and the same transfer measured outside the filter is always fast, so this is not host memory, CPU core speed or buffer allocation. It sits in the driver's transfer path under WSL2 and was not traced further ([docs/DESIGN.md](docs/DESIGN.md), "Input path"). The headline table is the main run, which landed between the two groups.

### All backends

Median ms (99th percentile in brackets), from the main run:

| Resolution | cpu (1 thread) | pcl (1 thread) | cpu_omp (20 threads) | cuda end-to-end | cuda GPU-only | Output points |
|---|---:|---:|---:|---:|---:|---:|
| 320x240 | 1.900 (2.312) | 1.552 (1.777) | 0.247 (0.392) | 0.430 (0.556) | 0.244 (0.336) | 20,785 |
| 640x480 | 7.964 (8.780) | 4.987 (5.581) | 0.876 (1.283) | 0.792 (1.031) | 0.335 (0.512) | 34,654 |
| 1280x720 | 24.909 (27.972) | 13.869 (15.163) | 2.774 (3.391) | 1.702 (2.024) | 0.617 (0.762) | 39,514 |

CUDA end-to-end speedup against each single-threaded baseline:

| Resolution | vs cpu (1 thread) | vs pcl (1 thread) |
|---|---:|---:|
| 320x240 | 4.42x | 3.61x |
| 640x480 | 10.05x | 6.30x |
| 1280x720 | 14.64x | 8.15x |

How to read this:

- **cpu** is the single-threaded reference: key every point, `std::sort`, linear reduce, `-O3`.
- **cpu_omp** is the same algorithm on OpenMP (parallel classify, parallel sort, parallel reduce). Its output is bit-identical to `cpu`. 20 threads was the fastest setting at every resolution in the thread sweep.
- **pcl** is a radial range loop followed by `pcl::VoxelGrid`, single-threaded. Converting into and out of PCL point types is not timed, which favours PCL. Its voxel counts equal ours on all nine tested cases; its centroids are float-accumulated and it is not bit-compatible (details in [docs/DESIGN.md](docs/DESIGN.md)).
- **cuda end-to-end** is host input to host output: host-to-device transfer straight from the input buffer, device pipeline, transfer back, final synchronisation, timed with `std::chrono::steady_clock`.
- **cuda GPU-only** is the device pipeline between the two transfers, timed with cudaEvents.
- CPU and CUDA backends return the same voxel count and, on these synthetic clouds, identical centroids. The tested tolerance is 1e-4 m per axis.
- The clouds are synthetic. Real sensor data will have a different ratio of input points to occupied voxels.
- This is a laptop: 20 CPU threads against a mobile GPU reached through WSL2. The balance will differ on other hardware.

### The bottleneck: profiled, changed, measured again

The first per-stage breakdown showed that getting the cloud onto the GPU cost more than everything the GPU then did with it: 1.62 ms of 2.36 ms at 1280x720. The input path at that point was a `memcpy` into a pinned staging buffer followed by one asynchronous transfer.

What the measurements showed ([docs/DESIGN.md](docs/DESIGN.md) has the numbers and method):

- The host `memcpy` was only 0.45 ms of the 1.62 ms. The transfer itself was the slow part.
- A transfer straight from the caller's ordinary (pageable) memory takes about as long as the transfer from pinned memory did, so the staging `memcpy` bought nothing here and could be removed.
- For 16-byte points, uploading the raw bytes and reading x, y, z by offset on the device beat gathering x, y, z on the host first at every size. For 32-byte points the host gather was faster at 320x240 and 640x480 and slightly slower at 1280x720 ([results/cuda_input_modes.csv](results/cuda_input_modes.csv)).

So every backend now takes a strided view of the caller's buffer (the layout of a PointCloud2), and the CUDA backend uploads it without a staging copy: raw for points up to 16 bytes, host-gathered for wider ones. The pinned input buffer is gone.

| | 320x240 | 640x480 | 1280x720 |
|---|---:|---:|---:|
| Input stage before, ms | 0.127 | 0.517 | 1.618 |
| Input stage after, ms | 0.102 | 0.346 | 0.987 |
| CUDA end-to-end before, ms | 0.442 | 0.998 | 2.356 |
| CUDA end-to-end after, main run, ms | 0.430 | 0.792 | 1.702 |
| CUDA end-to-end after, median of 7 launches, ms | 0.394 | 0.769 | 1.521 |

In the slower launch state the 1280x720 time (about 2.4 ms) is no better than before the change. The "before" numbers come from single launches whose state was not recorded, so the size of the improvement is known only for the common, faster state: about 2.36 ms to about 1.5 ms. The input transfer is still the largest stage (0.99 ms of 1.70 ms).

### History

Earlier runs are kept in [results/history](results/history) so changes can be compared. CUDA end-to-end median ms:

| Run | What changed | 320x240 | 640x480 | 1280x720 |
|---|---|---:|---:|---:|
| [m4](results/history/m4) | First benchmark | 0.461 | 1.049 | 2.439 |
| [a-baselines](results/history/a-baselines) | OpenMP and PCL baselines added; no change to the CUDA code | 0.442 | 0.998 | 2.356 |
| current | Direct upload of the input buffer, fields read on the device | 0.430 | 0.792 | 1.702 |

The M4 run reported speedups of 3.96x, 7.55x and 10.53x. Those were against the single-threaded backend, the only baseline at the time.

### Reproduce

Inside the dev container:

```
docker compose -f docker/compose.dev.yml up -d --build
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build --cmake-args -DWITH_PCL=ON && colcon test && colcon test-result --verbose'
docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/bench.sh
```

## Tests and CI

CI builds and tests the CPU paths and compiles the CUDA code. GitHub runners have no GPU, so the CUDA parity tests run only locally (they skip themselves when no GPU is visible). The PCL comparison test is built only with `-DWITH_PCL=ON`, which CI does not use.
