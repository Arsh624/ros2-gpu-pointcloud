# ros2-gpu-pointcloud

[![CI](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/Arsh624/ros2-gpu-pointcloud/actions/workflows/ci.yml)

A ROS 2 Jazzy node that range-filters and voxel-downsamples point clouds. It has a single-threaded C++17 backend, an OpenMP backend and a CUDA backend that produce the same output, and a benchmark that compares them with each other and with PCL.

**Work in progress.** See [docs/STATUS.md](docs/STATUS.md) for what exists and what is next.

## Benchmark

Default parameters (voxel 0.05 m, range 0.3 to 10 m), synthetic depth-camera cloud, packed xyz input (12 bytes per point). Each number is the median of 200 timed iterations after a warmup of 20 iterations and at least 500 ms. Raw data is in [results/](results).

Environment: NVIDIA GeForce RTX 4070 Laptop GPU (driver 616.64, CUDA 12.6), Intel Core i9-14900HX with 20 logical CPUs visible to the container, PCL 1.14.0, Docker on WSL2 (kernel 5.15.167.4), laptop on AC power.

There are two different questions here and they have different answers:

1. **Back to back**, how fast is each backend when it is called in a tight loop?
2. **At a camera's frame rate**, how long does one call take when the backend sat idle since the last frame?

### 1. Back to back: CUDA against the best CPU baseline

The best CPU baseline at every resolution is this repository's OpenMP backend on 20 threads.

| Resolution | Points | Best CPU (OpenMP, 20 threads) ms | CUDA end-to-end ms | CUDA GPU-only ms | End-to-end speedup | GPU-only speedup |
|---|---:|---:|---:|---:|---:|---:|
| 320x240 | 76,800 | 0.246 | 0.415 | 0.218 | 0.59x | 1.13x |
| 640x480 | 307,200 | 0.914 | 0.705 | 0.256 | 1.30x | 3.57x |
| 1280x720 | 921,600 | 2.889 | 1.641 | 0.435 | 1.76x | 6.64x |

In a tight loop the GPU is slower than 20 CPU threads at 320x240, 1.3x faster at 640x480 and 1.8x faster at 1280x720, end to end.

**The CUDA end-to-end time also depends on the process launch.** The same binary run seven times gave these medians ([results/cuda_launches.csv](results/cuda_launches.csv)):

| Resolution | Fastest launch ms | Median launch ms | Slowest launch ms | Speedup vs best CPU, slowest to fastest |
|---|---:|---:|---:|---:|
| 320x240 | 0.384 | 0.489 | 0.512 | 0.48x to 0.64x |
| 640x480 | 0.704 | 0.733 | 1.122 | 0.81x to 1.30x |
| 1280x720 | 1.504 | 1.666 | 2.423 | 1.19x to 1.92x |

At 1280x720 four launches were between 1.50 and 1.67 ms and three were near 2.4 ms. The difference is in the host-to-device transfer stage, which takes about 0.93 ms or about 1.74 ms for the same 11 MB. Host memory speed, CPU core type, buffer allocation and the bare transfer call were each ruled out by measurement; it sits in the driver's transfer path under WSL2 and was not traced further ([docs/DESIGN.md](docs/DESIGN.md), "Two transfer rates").

### 2. At a frame rate: idle gaps change the picture

A benchmark loop never lets the GPU rest. A camera does. `bench_idle_gap` sleeps before every call and times the call ([results/idle_gap.csv](results/idle_gap.csv)). Median ms per call:

| Idle gap before each call | cpu_omp 640x480 | cuda 640x480 | cpu_omp 1280x720 | cuda 1280x720 |
|---|---:|---:|---:|---:|
| none (tight loop) | 0.91 | 0.71 | 2.69 | 1.49 |
| 10 ms | 1.20 | 1.28 | 2.98 | 2.11 |
| 33 ms (30 Hz) | 1.37 | 1.94 | 3.21 | 3.03 |
| 100 ms (10 Hz) | 1.72 | 1.95 | 4.71 | 6.96 |
| 300 ms | 7.03 | 5.94 | 9.39 | 12.71 |
| 1 s | 6.74 | 6.05 | 7.33 | 13.00 |

Both backends slow down when they have been idle, and the GPU slows down more:

- **At 30 Hz the two are about level** (CUDA 6% faster at 1280x720, 40% slower at 640x480).
- **At 10 Hz the CPU is faster at both sizes** (1280x720: 4.7 ms against 7.0 ms).
- After a long idle a 1280x720 CUDA call takes about 12.7 ms, eight times its tight-loop time.

The cause on the GPU side is its power management. `nvidia-smi` reports the idle GPU in performance state P8 at 210 MHz core / 405 MHz memory clock and 2 W; under the benchmark it is in P0 at 2625 MHz / 7000 to 8200 MHz and 33 W. After 3 s of idle it takes roughly a quarter to a third of a second of continuous work before calls are back to full speed ([results/idle_recovery.csv](results/idle_recovery.csv)). A tiny GPU call every 2 ms in the background does not prevent this ([results/idle_gap_keepalive.csv](results/idle_gap_keepalive.csv)): the clocks follow load, not activity. The CPU side slows because the OpenMP worker threads go to sleep; its worst single calls after a gap reached 20 to 70 ms.

So on this laptop, **the GPU backend only pays off when it is kept busy**: at high frame rates, with large clouds, or when several streams share it. For one 640x480 camera at 10 to 30 Hz the OpenMP backend is the better choice here. A desktop or embedded GPU with different power management may behave differently; that has not been measured.

### 3. Inside the ROS node: latency and CPU cost per frame

`scripts/node_bench.sh` runs `pointcloud_filter_node` against `cloud_publisher_node` in one container, at 10 and 30 Hz, and logs every callback ([results/node_latency.csv](results/node_latency.csv)). 100 messages at 10 Hz and 300 at 30 Hz per row, after 20 start-up messages; every message arrived. "Callback" is callback entry to the return of `publish()` (PointCloud2 view, filter, output message). "CPU" is the CPU time of the whole node process (all threads) during the callback.

| Backend | Resolution | 10 Hz callback p50 ms | 10 Hz p99 ms | 10 Hz CPU p50 ms | 10 Hz avg cores | 30 Hz callback p50 ms | 30 Hz p99 ms | 30 Hz CPU p50 ms | 30 Hz avg cores |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 320x240 | 2.60 | 6.73 | 2.65 | 0.97 | 3.86 | 6.75 | 3.70 | 0.96 |
| cpu | 640x480 | 12.04 | 16.71 | 11.57 | 0.96 | 10.15 | 14.80 | 9.77 | 0.96 |
| cpu | 1280x720 | 27.47 | 32.99 | 26.42 | 0.96 | 27.09 | 34.63 | 26.06 | 0.97 |
| cpu_omp, 12 threads | 320x240 | 1.19 | 14.06 | 1.23 | 4.08 | 1.03 | 1.70 | 1.05 | 1.41 |
| cpu_omp, 12 threads | 640x480 | 4.03 | 11.54 | 4.47 | 4.53 | 2.34 | 9.76 | 2.31 | 2.59 |
| cpu_omp, 12 threads | 1280x720 | 6.46 | 18.22 | 10.35 | 4.17 | 6.68 | 10.99 | 29.57 | 4.55 |
| cpu_omp, 20 threads | 320x240 | 1.43 | 20.71 | 1.47 | 7.34 | 0.97 | 11.64 | 1.01 | 4.70 |
| cpu_omp, 20 threads | 640x480 | 3.06 | 49.53 | 3.46 | 8.13 | 1.84 | 11.42 | 2.12 | 3.83 |
| cpu_omp, 20 threads | 1280x720 | 7.17 | 19.18 | 65.21 | 9.03 | 5.10 | 11.21 | 11.76 | 6.14 |
| cuda | 320x240 | 2.67 | 4.56 | 2.56 | 0.97 | 1.83 | 4.45 | 1.77 | 0.97 |
| cuda | 640x480 | 2.90 | 6.99 | 2.79 | 0.97 | 2.50 | 7.13 | 2.41 | 0.97 |
| cuda | 1280x720 | 11.07 | 13.58 | 10.65 | 0.96 | 3.48 | 12.09 | 3.35 | 0.96 |

"Avg cores" is total CPU time over total callback time for the row; the CPU p50 column is the median of single callbacks, so the two do not divide into each other exactly.

What this shows:

- **CUDA costs one core, OpenMP costs four to nine.** At 1280x720 and 30 Hz, CUDA takes 3.48 ms per frame on 0.96 cores; OpenMP on 12 threads takes 6.68 ms on 4.55 cores. CUDA's CPU time equals its wall time, which fits the host thread spinning while it waits for the GPU (the CUDA runtime's default wait policy); a blocking wait was not tried.
- **At 30 Hz CUDA is the fastest at 1280x720** (3.48 ms against 5.10 ms for the best OpenMP setting). At 640x480 OpenMP on 20 threads has the lower median (1.84 against 2.50 ms); CUDA has the lowest p99 at 640x480 at both rates (7.13 and 6.99 ms).
- **At 10 Hz the GPU's idle clock-down shows up again**: CUDA 1280x720 goes from 3.48 ms at 30 Hz to 11.07 ms at 10 Hz (section 2 explains why).
- **OpenMP tails are long inside a node.** Its p99 is between 10 and 50 ms in 10 of its 12 rows. The cause was not isolated; without ROS, section 2 already showed single OpenMP calls of 20 to 70 ms after an idle gap.
- The single-threaded `cpu` backend needs 27 ms of the 33 ms frame budget at 1280x720 and 30 Hz.

**Middleware setting used here.** Transport is not what this measures, so both processes use Fast DDS over shared memory with a segment large enough for one cloud ([docker/fastdds_shm_large.xml](docker/fastdds_shm_large.xml)). With middleware defaults most clouds of 0.9 MB and more were dropped in this container: Fast DDS's default 512 KB shared-memory segment is smaller than one sample, and the UDP path is limited by the Docker VM's 208 KB socket buffer cap. Middleware behaviour with defaults across containers is the subject of the networking comparison.

### All backends, back to back

Median ms (99th percentile in brackets), from the main run:

| Resolution | cpu (1 thread) | pcl (1 thread) | cpu_omp (20 threads) | cuda end-to-end | cuda GPU-only | Output points |
|---|---:|---:|---:|---:|---:|---:|
| 320x240 | 1.894 (2.174) | 1.603 (1.972) | 0.246 (0.490) | 0.415 (0.676) | 0.218 (0.376) | 20,785 |
| 640x480 | 8.732 (12.042) | 5.011 (6.015) | 0.914 (1.583) | 0.705 (0.815) | 0.256 (0.339) | 34,654 |
| 1280x720 | 26.156 (29.344) | 14.242 (17.667) | 2.889 (12.901) | 1.641 (1.917) | 0.435 (0.547) | 39,514 |

CUDA end-to-end speedup against each single-threaded baseline:

| Resolution | vs cpu (1 thread) | vs pcl (1 thread) |
|---|---:|---:|
| 320x240 | 4.56x | 3.86x |
| 640x480 | 12.38x | 7.11x |
| 1280x720 | 15.93x | 8.68x |

How to read this:

- **cpu** is the single-threaded reference: key every point, `std::sort`, linear reduce, `-O3`.
- **cpu_omp** is the same algorithm on OpenMP (parallel classify, parallel sort, parallel reduce). Its output is bit-identical to `cpu`. 20 threads was the fastest setting at every resolution in the thread sweep.
- **pcl** is a radial range loop followed by `pcl::VoxelGrid`, single-threaded. Converting into and out of PCL point types is not timed, which favours PCL. Its voxel counts equal ours on all nine tested cases; its centroids are float-accumulated and it is not bit-compatible (details in [docs/DESIGN.md](docs/DESIGN.md)).
- **cuda end-to-end** is host input to host output: host-to-device transfer straight from the input buffer, device pipeline, transfer back, final synchronisation, timed with `std::chrono::steady_clock`.
- **cuda GPU-only** is the device pipeline between the two transfers, timed with cudaEvents.
- CPU and CUDA backends return the same voxel count and, on these synthetic clouds, identical centroids. The tested tolerance is 1e-4 m per axis.
- The clouds are synthetic. Real sensor data will have a different ratio of input points to occupied voxels.

### The input bottleneck: profiled, changed, measured again

The first per-stage breakdown showed that getting the cloud onto the GPU cost more than everything the GPU then did with it: 1.62 ms of 2.36 ms at 1280x720. The input path at that point was a `memcpy` into a pinned staging buffer followed by one asynchronous transfer.

What the measurements showed ([docs/DESIGN.md](docs/DESIGN.md), "Input path"):

- The host `memcpy` was 0.45 ms of the 1.62 ms.
- A transfer straight from the caller's ordinary (pageable) memory takes about as long as the transfer from pinned memory did, so the staging copy bought nothing here and could be removed.
- For 16-byte points, uploading the raw bytes and reading x, y, z by offset on the device beat gathering x, y, z on the host first at every size. For 32-byte points the host gather was faster at 320x240 and 640x480 and slightly slower at 1280x720.

So every backend now takes a strided view of the caller's buffer (the layout of a PointCloud2), and the CUDA backend uploads it without a staging copy: raw for points up to 16 bytes, host-gathered for wider ones.

| | 320x240 | 640x480 | 1280x720 |
|---|---:|---:|---:|
| Input stage before, ms | 0.127 | 0.517 | 1.618 |
| Input stage after, ms | 0.110 | 0.342 | 1.053 |
| CUDA end-to-end before, ms | 0.442 | 0.998 | 2.356 |
| CUDA end-to-end after, main run, ms | 0.415 | 0.705 | 1.641 |
| CUDA end-to-end after, median of 7 launches, ms | 0.489 | 0.733 | 1.666 |

In the slower launch state the 1280x720 time (about 2.4 ms) is no better than before the change. The "before" numbers come from single launches whose state was not recorded, so the size of the improvement is only established for the faster state. The input transfer is still the largest stage (1.05 ms of 1.64 ms).

### The p99 tail: explained

Earlier runs showed a CUDA 99th percentile of 11 to 12 ms at 1280x720 against a 2.4 ms median. Per-iteration samples ([results/p99_tail_example.csv](results/p99_tail_example.csv)) show these were the first few timed iterations, each about 11.5 ms, after which every iteration was normal. The GPU had idled for several seconds while the CPU backends ran in the same process, and the 20 warmup iterations ended before it was back at full clock. It is the same effect as in section 2, not a recurring stall and not an allocation or synchronisation in the filter.

The benchmark now warms up for at least 500 ms as well as 20 iterations. Without that, 2 of 6 runs had slow timed iterations (13 and 8 of them); with it, 0 of 8 runs had any, and the CUDA p99 at 1280x720 is 1.9 ms.

### History

Earlier runs are kept in [results/history](results/history) so changes can be compared. CUDA end-to-end median ms:

| Run | What changed | 320x240 | 640x480 | 1280x720 |
|---|---|---:|---:|---:|
| [m4](results/history/m4) | First benchmark | 0.461 | 1.049 | 2.439 |
| [a-baselines](results/history/a-baselines) | OpenMP and PCL baselines added; no change to the CUDA code | 0.442 | 0.998 | 2.356 |
| [b-input-path](results/history/b-input-path) | Direct upload of the input buffer, fields read on the device | 0.430 | 0.792 | 1.702 |
| current | Warmup of at least 500 ms; no change to the CUDA code | 0.415 | 0.705 | 1.641 |

The M4 run reported speedups of 3.96x, 7.55x and 10.53x. Those were against the single-threaded backend, the only baseline at the time, and in a tight loop.

### Reproduce

Inside the dev container:

```
docker compose -f docker/compose.dev.yml up -d --build
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build --cmake-args -DWITH_PCL=ON && colcon test && colcon test-result --verbose'
docker exec gpc-dev /ros_entrypoint.sh bash /ws/src/ros2-gpu-pointcloud/scripts/bench.sh
```

## Tests and CI

CI builds and tests the CPU paths and compiles the CUDA code. GitHub runners have no GPU, so the CUDA parity tests run only locally (they skip themselves when no GPU is visible). The PCL comparison test is built only with `-DWITH_PCL=ON`, which CI does not use.
