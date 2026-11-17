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
| H2D | `cudaMemcpyAsync` straight from the input buffer (see "Input path") | Measured: a pinned staging buffer added a host copy and did not speed up the transfer |
| Classify | one kernel: read x, y, z by offset, finite test, range test, voxel key, keep flag | One pass over the input; uses the same `detail::classify` as the CPU code |
| Select | `cub::DeviceSelect::Flagged` over a counting iterator | Yields the indices of kept points, so later stages never touch rejected ones |
| Gather keys | small kernel | Keys of kept points, contiguous, ready to sort |
| Sort | `cub::DeviceRadixSort::SortPairs` on (key, index) with `DoubleBuffer` | Brings equal keys together; ascending key order is the output order |
| Reduce | `cub::DeviceReduce::ReduceByKey` over a transform iterator that loads `(x, y, z, 1)` as doubles by index | Sums and counts per voxel without materialising a sorted copy of the points |
| Centroid | small kernel: sum / count, cast to float32 | |
| D2H | `cudaMemcpyAsync` into a pinned buffer, then `memcpy` into the caller's vector | |

Decisions:

- **Nothing is allocated per frame.** Device buffers, the pinned output buffer and the CUB temporary storage are sized once for `reserve_points` (the node reserves 1280x720). A larger cloud reallocates once and the new size is kept. The three CUB stages share one temporary buffer sized for the largest of them.
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
| cuda / end_to_end | `std::chrono::steady_clock` around `apply` | Host input to host output: H2D from the input buffer (after a host gather for wide points), device pipeline, D2H, final stream synchronisation, copy into the output vector |
| cuda / gpu_only | cudaEvents (`FilterStats::device_ms`) | Device pipeline between the two transfers, including the two small host syncs |
| per stage | cudaEvents at each stage boundary | One interval per stage; `h2d_copy` also contains the host gather when wide points are repacked, and the two `host_sync+...` stages contain the host round trip |

Reported statistics are the median and the nearest-rank 99th percentile. Speedup is CPU median divided by CUDA median. Input points per second is the input size divided by the median time.

Sanity checks applied before publishing numbers: CPU time per point is roughly constant across resolutions (24 to 28 ns per point in the committed run), GPU-only time is below end-to-end time at every size, and the per-stage medians add up to about the end-to-end median.

## Profiling under WSL2

`scripts/profile.sh` runs `nsys profile -t cuda,nvtx`. In this environment (Docker on WSL2) Nsight Systems records the NVTX ranges and the CUDA API calls, but no GPU-side data:

```
SKIPPED: profile.sqlite does not contain CUDA kernel data.
SKIPPED: profile.sqlite does not contain GPU memory data.
```

So `results/nsys_summary.txt` shows host-side durations only. A host-side NVTX range around an asynchronous launch measures the launch, not the work: the `sort` range is short because the sort's GPU time is paid inside the next range that synchronises. For stage costs use `results/stages.csv`, which comes from cudaEvents recorded on the stream and is therefore on the GPU clock.

## CPU baselines

The benchmark has three CPU rows so the CUDA numbers are judged against the strongest one.

| Baseline | What it is | Notes |
|---|---|---|
| `cpu` | This repository's single-threaded filter | The semantic reference |
| `cpu_omp` | The same algorithm on OpenMP | Bit-identical to `cpu` at any thread count (tested at 1, 2, 7 and 32) |
| `pcl` | Radial range loop + `pcl::VoxelGrid` (PCL 1.14), single-threaded | Built only with `-DWITH_PCL=ON` |

### OpenMP backend

1. Classify points in parallel over fixed chunks.
2. Compact the kept (key, index) pairs in input order, using per-chunk counts and a prefix sum.
3. Sort with `__gnu_parallel::sort` (libstdc++ parallel mode, a multiway mergesort on OpenMP).
4. Reduce in parallel: the sorted array is cut into one slice per thread, each cut moved forward to the next run start so no voxel is split, then each thread averages its runs into a precomputed output offset.

The pairs are unique, so the sort order is total and the summation order inside a voxel is the input order, as in the serial backend. The loops run over chunk numbers rather than thread ids, so the result does not depend on how many threads the runtime grants.

`results/cpu_omp_threads.csv` holds a thread sweep under the benchmark protocol. On the test machine the container sees 20 logical CPUs and 20 threads was fastest at all three resolutions.

### PCL baseline and how its semantics differ

PCL has no radial range filter (`pcl::CropBox` is a box, `pcl::PassThrough` is one axis), so the baseline uses the plain loop a PCL user would write: drop non-finite points and points outside `[min_range, max_range]`, then `pcl::VoxelGrid` with a cubic leaf of `voxel_size`. The conversions between our point type and `pcl::PointXYZ` are outside the timed region, which favours PCL.

Differences from our filter, from reading `voxel_grid.hpp` and from the comparison test:

- **Same lattice.** `VoxelGrid` computes `floor(coord * inverse_leaf_size)`, a grid anchored at the origin like ours. It multiplies by a reciprocal where we divide, so a point on a voxel face can fall on the other side.
- **Float centroids.** PCL accumulates in float; we accumulate in double.
- **Different output order.** PCL sorts by a linear index with z most significant; we emit ascending key order with x most significant.
- **`min_points_per_voxel`** defaults to 0, so every occupied voxel is emitted, like ours.
- **Index overflow.** PCL refuses to filter when the grid's bounding box has more than 2^31 cells; we use 21 bits per axis independent of the bounding box.

Measured on the synthetic clouds at 3 resolutions x voxel sizes {0.02, 0.05, 0.1} (`test_pcl_baseline`):

- Voxel counts are equal in all nine cases.
- Every voxel we emit is also occupied in PCL's result.
- Centroids agree to within 5e-6 m in seven cases. In the two largest clouds at voxel 0.02 the worst difference is 1e-2 m and 4e-3 m. That is the size expected when a boundary point is assigned to the neighbouring voxel, which fits the reciprocal-multiply difference, but the cause was not traced point by point.

Because of these differences the test does not demand parity with PCL. It requires the counts to agree within 1% and at least 99% of our voxels to be occupied in PCL's result, which is enough to say the two do comparable work.

## Input path

This section replaces the H2D row of the pipeline table above: there is no pinned input buffer any more.

### Strided views

Every backend takes a `CloudView`: a pointer, a count, a stride and the byte offsets of float32 x, y, z. That is the layout of a `sensor_msgs/PointCloud2` without row padding, so a message can be filtered where it lies. Points are read with `detail::PointReader`, which uses `memcpy` and therefore works for any alignment on the host. Packed `PointXYZ` arrays are the special case stride 12, offsets 0/4/8.

### What was measured

The first stage breakdown (`results/history/a-baselines/stages.csv`) had `h2d_copy` at 1.62 ms of a 2.36 ms end-to-end at 1280x720. That stage was a `memcpy` into a pinned buffer plus one `cudaMemcpyAsync`.

Exploratory runs on the old code (100 iterations each, not committed as results) separated the parts:

| Variant at 1280x720 | Input stage ms |
|---|---:|
| `memcpy` to pinned + async transfer (the old path) | 1.61 to 1.69 |
| host `memcpy` only, no transfer | 0.45 |
| async transfer straight from the caller's pageable buffer | 0.93 to 0.97, or 1.77 to 1.79 (see below) |
| pinned staging in 64k-point chunks | 0.96, or 1.81 |

`gpu_pointcloud/bench/bench_h2d.cu` then measured the bare transfer of 11 MB with no filter around it: about 1.0 ms (11.5 GB/s) from `malloc` memory, from memory with huge pages disabled, from huge-page-aligned memory and from pinned memory alike, on every buffer tried.

Conclusions:

- The transfer costs about 1 ms for 11 MB whatever the source memory is. Pinning does not make it faster on this platform.
- The old path therefore paid 0.45 ms for a staging copy that bought nothing. Transferring directly from the caller's buffer removes it.
- Chunking and two-stream overlap were not pursued: the kernels that could overlap with the transfer (classify, 0.2 ms) are small next to it, and chunked pinned transfers measured the same as the direct transfer.

### Raw upload or host repack for wide points

With a point stride above 12 bytes there is a choice: upload all the bytes and read fields on the device (`CudaInput::kRaw`), or gather x, y, z on the host and upload 12 bytes per point (`CudaInput::kRepack`). `bench_cuda_input` runs both in one process, alternating per iteration (`results/cuda_input_modes.csv`, medians over three launches):

| Point step | Resolution | Raw ms | Repack ms |
|---|---|---:|---:|
| 16 | 320x240 | 0.41 | 0.43 |
| 16 | 640x480 | 0.82 | 0.93 |
| 16 | 1280x720 | 3.03 | 3.42 |
| 32 | 320x240 | 0.53 | 0.47 |
| 32 | 640x480 | 1.26 | 1.08 |
| 32 | 1280x720 | 3.34 | 3.49 |

`CudaInput::kAuto` therefore uploads raw up to 16 bytes per point and repacks wider points. Layouts whose stride or offsets are not multiples of 4 are always repacked, because the device cannot read misaligned floats. This rule is provisional: the two modes run in separate filter instances, and the state effect described next disturbs the comparison. An earlier six-launch run of the same tool gave 1.84 ms for 16-byte raw upload at 1280x720 where this table has 3.03 ms.

### Two transfer rates (open, environmental)

The input stage of the same binary takes one of two values. At 1280x720 it is about 0.93 ms or about 1.74 ms. `results/cuda_launches.csv` has seven launches of the benchmark: five at about 1.5 ms end-to-end, two at about 2.4 ms. Within a launch, different filter instances can be in different states.

What was ruled out, each by a direct measurement:

- **Host memory bandwidth and CPU core type.** In 14 launches of a test that timed the filter, a plain 11 MB `memcpy` and the CPU filter side by side, three launches had the slow input stage. `memcpy` (0.41 to 0.46 ms) and the CPU filter (24.5 to 26.0 ms) were the same in slow and fast launches.
- **CPU affinity.** Pinning the process to a given logical CPU with `taskset` produced both states on the same CPU number.
- **Huge pages and buffer alignment.** `bench_h2d` shows the same rate for every allocation method, and the amount of transparent-huge-page backing did not track the state.
- **The transfer call itself.** Outside the filter, synchronous and stream-asynchronous transfers to 16 different device buffers all took 0.96 to 0.98 ms.

What that leaves: something in the driver's transfer path under WSL2 that is fixed when a filter instance (its stream and device buffers) is created. It was not traced further, and nothing in this repository's code selects between the two states. On native Linux the effect may not exist; that has not been tested.

Consequence for reading the benchmark: a single launch can land in either state, so `scripts/bench.sh` also records seven separate launches, and the README reports their range.

## Idle GPU: the p99 tail and what a frame rate does

### The symptom

The M4 and baseline runs reported a CUDA p99 of 12.3 ms and 11.4 ms at 1280x720 against medians of 2.4 ms.

### What it was

`bench_filter --samples-csv` writes every timed iteration. In a run that shows the tail (`results/p99_tail_example.csv`), iterations 0 to 4 take 11.4 to 11.6 ms, iteration 5 takes 8.5 ms, and from iteration 6 on everything is at 1.9 to 2.2 ms. In the slow iterations the input stage is 6.8 ms and the device pipeline 4.1 ms: everything is slow, not one stage.

The tail appeared only when all backends and all resolutions ran in one process, never in CUDA-only or single-resolution runs. In the all-backends run the GPU idles for about six seconds at 1280x720 while `cpu` and `cpu_omp` are measured, and then gets 20 warmup iterations.

### Ruled out

- **Per-frame allocation.** The filter allocates nothing per frame, and the slow iterations are the first ones, not scattered.
- **First touch.** The 20 warmup iterations have already used every buffer.
- **Sync points.** The two host syncs are in every iteration; they do not explain 5 slow iterations followed by 195 normal ones.
- **A recurring stall.** 2000 consecutive frames in a CUDA-only process had a maximum of 3.1 ms (old input path: 4.6 ms).

### The cause: GPU power state

`nvidia-smi --query-gpu=pstate,clocks.current.graphics,clocks.current.memory,power.draw`:

| | Performance state | Core clock | Memory clock | Power |
|---|---|---:|---:|---:|
| Idle | P8 | 210 MHz | 405 MHz | 2 W |
| During the benchmark | P0 | 2625 MHz | 7001 to 8201 MHz | 33 W |

`bench_idle_gap` measures a call after a controlled idle gap (`results/idle_gap.csv`) and the recovery after 3 s of idle (`results/idle_recovery.csv`):

- With no gap a 1280x720 CUDA call takes 1.49 ms. With a 100 ms gap before every call the median is 6.96 ms, and with 300 ms or more it is about 12.7 ms.
- After 3 s idle, 28 consecutive 1280x720 calls took 10.5 to 12.9 ms before the time dropped to 1.7 ms: about 320 ms of work. In another run it was 17 calls, about 220 ms. At 640x480 one run recovered after 48 calls of about 5 ms (about 245 ms) and another had not recovered after 60 calls (about 300 ms).
- A background thread making a tiny CUDA call every 2 ms does not keep the device fast (`results/idle_gap_keepalive.csv`): 11.5 ms at 300 ms gaps. Every 10 ms and every 50 ms were also tried, with the same result. The clocks follow load, not the mere presence of work.

So this is the laptop GPU's dynamic power management. Nothing in the filter causes it and nothing cheap in the filter prevents it. Options outside this repository that were not tested: the "prefer maximum performance" power mode in the NVIDIA control panel on the Windows host, or a desktop or embedded GPU with different power management.

The OpenMP backend has a smaller version of the same problem: its worker threads sleep when idle. With a 100 ms gap a 1280x720 call takes 4.7 ms instead of 2.7 ms, and individual calls after a gap reached 20 to 70 ms.

### What changed in this repository

- `bench_filter --warmup-ms`: warmup lasts until both the iteration count and this time are reached. `scripts/bench.sh` uses 20 iterations and 500 ms. Checked by counting CUDA iterations above 5 ms over repeated all-backend runs: without it 2 of 6 runs had some (13 and 8), with it 0 of 8.
- `bench_idle_gap` and its three result files, so the tight-loop numbers are not the only ones published.
- A side effect worth knowing when comparing runs: with the longer warmup the GPU-only medians dropped (1280x720: 0.62 ms to 0.44 ms). The earlier runs were measuring a GPU that was not always fully clocked up.
