# Design notes

Decisions and the reasons for them. Interfaces are listed in [STATUS.md](STATUS.md).

## Filter semantics

Both backends apply the same five steps, in this order.

1. **Non-finite drop.** A point with any NaN or Inf coordinate is dropped.
2. **Range.** `r = sqrt(x*x + y*y + z*z)`, computed in double from the float32 coordinates. Keep the point when `min_range <= r <= max_range`, both ends inclusive.
3. **Voxel index.** Per axis `i = floor(coord / voxel_size)`, computed in float32. A point whose index falls outside `[-2^20, 2^20)` on any axis is dropped and counted in `FilterStats::index_overflow`. With the defaults that limit is about +/-52 km, so it only matters for a very small `voxel_size`.
4. **Key.** `key = (ix + 2^20) << 42 | (iy + 2^20) << 21 | (iz + 2^20)`, 21 bits per axis in a uint64.
5. **Centroid.** One output point per key: the mean of its points, summed in double, emitted as float32. Output is in ascending key order.

### Why these choices

- **Ascending key order** makes the output deterministic and lets the parity test compare the two backends element by element. No matching step is needed.
- **Float32 division for the index, double for the range.** IEEE 754 division, multiplication, addition and square root are correctly rounded on both x86 and the GPU, so the same inputs give the same bits provided the compiler does not fuse a multiply and an add into one FMA. Contraction is therefore switched off on both sides (`-ffp-contract=off` for GCC, `--fmad=false` for nvcc). This is what makes "same voxel count exactly" achievable.
- **Double accumulation.** Centroid sums are the only place where the two backends may add in a different order. In double the difference is many orders of magnitude below the 1e-4 m parity tolerance.
- **A point exactly on a voxel face** belongs to the voxel `floor` gives it: `x = 0.05` with `voxel_size = 0.05` has index 1, and `x = -0.0001` has index -1.

## Toolchain

Image: `nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a` plus the ROS 2 Jazzy apt repository.

Installed versions in the image built on 2026-11-16 (`dpkg -l`):

| Package | Version |
|---|---|
| cuda-nvcc-12-6 | 12.6.85-1 |
| cuda-cudart-dev-12-6 | 12.6.77-1 |
| cuda-sanitizer-12-6 | 12.6.77-1 |
| nsight-systems-2024.5.1 | 2024.5.1.113 |
| g++-13 | 13.2.0-23ubuntu4 |
| cmake | 3.28.3-1build7 |
| ros-jazzy-ros-base | 0.11.0-1noble.20261105.164630 |
| ros-jazzy-rclcpp | 28.1.23-1noble.20261105.084818 |
| ros-jazzy-sensor-msgs | 5.3.8-1noble.20261105.060814 |
| ros-jazzy-rmw-zenoh-cpp | 0.2.13-1noble.20261104.171211 |
| ros-jazzy-rmw-cyclonedds-cpp | 2.2.4-1noble.20261105.063358 |
| ros-jazzy-ament-cmake-gtest | 2.5.7-1noble.20261025.230436 |

The base image is pinned by digest. The ROS packages come from the rolling Jazzy apt repository and are not pinned, so a rebuild on a later date may pick up newer patch versions.

## Working in the dev container

`docker exec` does not run the image entrypoint, so ROS is not sourced by default. Run commands through it:

```
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build'
```

From Git Bash on Windows, set `MSYS_NO_PATHCONV=1` first so `/ros_entrypoint.sh` is not rewritten to a Windows path.

## CUDA pipeline

Per frame, on one stream:

| Stage | How | Why |
|---|---|---|
| H2D | `memcpy` into a pinned buffer, `cudaMemcpyAsync` | Pinned memory makes the copy asynchronous and avoids a staging copy inside the driver |
| Classify | one kernel: finite test, range test, voxel key, keep flag | One pass over the input; uses the same `detail::classify` as the CPU code |
| Select | `cub::DeviceSelect::Flagged` over a counting iterator | Yields the indices of kept points, so later stages never touch rejected ones |
| Gather keys | small kernel | Keys of kept points, contiguous, ready to sort |
| Sort | `cub::DeviceRadixSort::SortPairs` on (key, index) with `DoubleBuffer` | Brings equal keys together; ascending key order is the output order |
| Reduce | `cub::DeviceReduce::ReduceByKey` over a transform iterator that loads `(x, y, z, 1)` as doubles by index | Sums and counts per voxel without materialising a sorted copy of the points |
| Centroid | small kernel: sum / count, cast to float32 | |
| D2H | `cudaMemcpyAsync` into a pinned buffer, then `memcpy` into the caller's vector | |

Decisions:

- **Nothing is allocated per frame.** Device buffers, both pinned host buffers and the CUB temporary storage are sized once for `reserve_points` (the node reserves 1280x720). A larger cloud reallocates once and the new size is kept. The three CUB stages share one temporary buffer sized for the largest of them.
- **Narrowed sort key.** A kept point has `|coord| <= max_range`, so its voxel index lies in `[-L, L]` with `L = ceil(max_range / voxel_size) + 1`. The kernel rebases each 21-bit field by `2^20 - L` and repacks with just enough bits per axis. The order is the same as for the canonical key, but the radix sort only processes `3 * bits` bits: 27 with the defaults instead of 63. When `L` does not fit in fewer than 21 bits the canonical key is used unchanged.
- **Reject counters.** Each block counts its rejected points per class in shared memory and adds the totals to three global counters, so there is one global atomic per block and class instead of one per rejected point.
- **Two host syncs inside the pipeline.** CUB takes item counts as host values, so the host must read the kept count before the sort and the voxel count before the output copy. Both are a few bytes read through a pinned buffer.
- **Double sums on the GPU.** `ReduceByKey` may add in a different order than the sequential loop on the CPU. With double accumulation the measured difference in the parity tests is 0 m on every tested cloud; the guaranteed bound stays 1e-4 m.
- **Device time.** `FilterStats::device_ms` is measured with cudaEvents recorded after the input copy is queued and before the output copy, so it covers the device pipeline including the two small syncs but excludes H2D and D2H of the cloud.
- **NVTX ranges** wrap each stage so a profile shows the stages by name.

## Known verification gap: compute-sanitizer

`compute-sanitizer --tool memcheck` does not run in this environment (CUDA 12.6 in Docker on WSL2):

```
========= Error: Failed to initialize WDDM debugger interface. Please run EnableDebuggerInterface.bat as an administrator
========= Error: Device not supported. Please refer to the "Supported Devices" section of the sanitizer documentation
```

The two errors in its summary are these start-up errors, not memory errors: the tool never attached to the device. The CUDA code has therefore **not** been checked by compute-sanitizer. Running it on native Linux, or after enabling the Windows debugger interface, is tracked in [issue #3](https://github.com/Arsh624/ros2-gpu-pointcloud/issues/3).

## Benchmark methodology

`bench_filter` (no ROS) runs each resolution and backend for 20 warmup iterations and 200 timed iterations on the same synthetic cloud, with one filter instance per run so buffers are already allocated.

| Timing | Clock | Covers |
|---|---|---|
| cpu / end_to_end | `std::chrono::steady_clock` around `apply` | The whole CPU filter |
| cuda / end_to_end | `std::chrono::steady_clock` around `apply` | Host input to host output: copy into pinned memory, H2D, device pipeline, D2H, final stream synchronisation, copy into the output vector |
| cuda / gpu_only | cudaEvents (`FilterStats::device_ms`) | Device pipeline between the two transfers, including the two small host syncs |
| per stage | cudaEvents at each stage boundary | One interval per stage; `h2d_copy` also contains the host `memcpy` into the pinned buffer, and the two `host_sync+...` stages contain the host round trip |

Reported statistics are the median and the nearest-rank 99th percentile. Speedup is CPU median divided by CUDA median. Input points per second is the input size divided by the median time.

Sanity checks applied before publishing numbers: CPU time per point is roughly constant across resolutions (24 to 28 ns per point in the committed run), GPU-only time is below end-to-end time at every size, and the per-stage medians add up to about the end-to-end median.

## Profiling under WSL2

`scripts/profile.sh` runs `nsys profile -t cuda,nvtx`. In this environment (Docker on WSL2) Nsight Systems records the NVTX ranges and the CUDA API calls, but no GPU-side data:

```
SKIPPED: profile.sqlite does not contain CUDA kernel data.
SKIPPED: profile.sqlite does not contain GPU memory data.
```

So `results/nsys_summary.txt` shows host-side durations only. A host-side NVTX range around an asynchronous launch measures the launch, not the work: the `sort` range is short because the sort's GPU time is paid inside the next range that synchronises. For stage costs use `results/stages.csv`, which comes from cudaEvents recorded on the stream and is therefore on the GPU clock.
