# ros2-gpu-pointcloud

[![CI](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml)

A ROS 2 Jazzy node that range-filters and voxel-downsamples point clouds. It has a single-threaded C++17 backend, an OpenMP backend and a CUDA backend that produce the same output, and a benchmark that compares them with each other and with PCL.

**Work in progress.** See [docs/STATUS.md](docs/STATUS.md) for what exists and what is next.

## Benchmark

Median of 200 timed iterations after 20 warmup iterations, default parameters (voxel 0.05 m, range 0.3 to 10 m), synthetic depth-camera cloud. Raw data: [results/bench.csv](results/bench.csv), [results/cpu_omp_threads.csv](results/cpu_omp_threads.csv), [results/stages.csv](results/stages.csv), [results/env.txt](results/env.txt).

Environment: NVIDIA GeForce RTX 4070 Laptop GPU (driver 616.64, CUDA 12.6), Intel Core i9-14900HX with 20 logical CPUs visible to the container, PCL 1.14.0, Docker on WSL2 (kernel 5.15.167.4), laptop on AC power.

### Headline: CUDA against the best CPU baseline

The best CPU baseline at every resolution is this repository's OpenMP backend on 20 threads.

| Resolution | Points | Best CPU (OpenMP, 20 threads) ms | CUDA end-to-end ms | CUDA GPU-only ms | End-to-end speedup | GPU-only speedup |
|---|---:|---:|---:|---:|---:|---:|
| 320x240 | 76,800 | 0.241 | 0.442 | 0.233 | 0.55x | 1.04x |
| 640x480 | 307,200 | 0.856 | 0.998 | 0.358 | 0.86x | 2.39x |
| 1280x720 | 921,600 | 2.630 | 2.356 | 0.567 | 1.12x | 4.64x |

So, measured end to end on this machine, **the GPU is slower than 20 CPU threads at 320x240 and 640x480 and 12% faster at 1280x720.** The device pipeline alone is 1.0x to 4.6x faster; the difference is the cost of getting the cloud onto the GPU, which is the largest stage at every size ([results/stages.csv](results/stages.csv)).

### All backends

Median ms (99th percentile in brackets):

| Resolution | cpu (1 thread) | pcl (1 thread) | cpu_omp (20 threads) | cuda end-to-end | cuda GPU-only | Output points |
|---|---:|---:|---:|---:|---:|---:|
| 320x240 | 1.918 (2.295) | 1.574 (1.810) | 0.241 (0.527) | 0.442 (0.548) | 0.233 (0.329) | 20,785 |
| 640x480 | 7.995 (8.882) | 4.896 (5.557) | 0.856 (1.197) | 0.998 (1.161) | 0.358 (0.504) | 34,654 |
| 1280x720 | 25.788 (27.121) | 13.885 (16.188) | 2.630 (3.326) | 2.356 (11.405) | 0.567 (3.058) | 39,514 |

CUDA end-to-end speedup against each single-threaded baseline:

| Resolution | vs cpu (1 thread) | vs pcl (1 thread) |
|---|---:|---:|
| 320x240 | 4.34x | 3.56x |
| 640x480 | 8.01x | 4.91x |
| 1280x720 | 10.95x | 5.89x |

How to read this:

- **cpu** is the single-threaded reference: key every point, `std::sort`, linear reduce, `-O3`.
- **cpu_omp** is the same algorithm on OpenMP (parallel classify, parallel sort, parallel reduce). Its output is bit-identical to `cpu`. 20 threads was the fastest setting at every resolution in the thread sweep.
- **pcl** is a radial range loop followed by `pcl::VoxelGrid`, single-threaded. Converting into and out of PCL point types is not timed, which favours PCL. PCL is faster than our single-threaded backend at every size. Its voxel counts equal ours on all nine tested cases; its centroids are float-accumulated and it is not bit-compatible (details in [docs/DESIGN.md](docs/DESIGN.md)).
- **cuda end-to-end** is host input to host output: copy into pinned memory, host-to-device transfer, device pipeline, transfer back, final synchronisation, timed with `std::chrono::steady_clock`.
- **cuda GPU-only** is the device pipeline between the two transfers, timed with cudaEvents.
- The CUDA tail at 1280x720 (p99 11.4 ms against a 2.4 ms median) is real and not yet explained.
- CPU and CUDA backends return the same voxel count and, on these synthetic clouds, identical centroids. The tested tolerance is 1e-4 m per axis.
- The clouds are synthetic. Real sensor data will have a different ratio of input points to occupied voxels.
- This is a laptop: 20 CPU threads against a mobile GPU reached through WSL2. The balance will differ on other hardware.

### History

Earlier runs are kept in [results/history](results/history) so changes can be compared:

| Run | What changed | 320x240 | 640x480 | 1280x720 |
|---|---|---:|---:|---:|
| [m4](results/history/m4) | First benchmark. CUDA end-to-end ms | 0.461 | 1.049 | 2.439 |
| current | Baselines added (no change to the CUDA code). CUDA end-to-end ms | 0.442 | 0.998 | 2.356 |

The M4 run reported speedups of 3.96x, 7.55x and 10.53x. Those were against the single-threaded backend only, which was the only baseline at the time.

### Reproduce

Inside the dev container:

```
docker compose -f docker/compose.dev.yml up -d --build
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build --cmake-args -DWITH_PCL=ON && colcon test && colcon test-result --verbose'
docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/bench.sh
```

## Tests and CI

CI builds and tests the CPU paths and compiles the CUDA code. GitHub runners have no GPU, so the CUDA parity tests run only locally (they skip themselves when no GPU is visible). The PCL comparison test is built only with `-DWITH_PCL=ON`, which CI does not use.
