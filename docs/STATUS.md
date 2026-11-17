# Status and implementation plan

**Goal:** a ROS 2 Jazzy node that range-filters and voxel-downsamples point clouds on CPU or CUDA with identical results, plus an honest benchmark and a DDS vs Zenoh comparison.

**Architecture:** one ament_cmake package, `gpu_pointcloud`. A ROS-free core library (`gpu_pointcloud_core`: CPU filter, CUDA filter, synthetic scene) is used by the tests, the benchmark and the node. A thin ROS layer (`cloud_conv`, `filter_node`) converts PointCloud2 to and from the core's point type.

**Tech stack:** C++17, CUDA 12.6 + CUB, ROS 2 Jazzy, gtest (ament), Docker (`nvidia/cuda:12.6.3-devel-ubuntu24.04`), GitHub Actions.

## Current state

| Milestone | State | PR | Notes |
|---|---|---|---|
| M0 zenoh smoke test | done | n/a | Passed, see below |
| M1 skeleton + CI | done | [#1](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/1) | CI on main: https://github.com/Arsh624/ros2-gpu-pointcloud/actions/runs/44019966683 |
| M2 CPU filter | done | [#2](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/2) | 24 tests pass, node verified with backend cpu |
| M3 CUDA filter | done | [#4](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/4) | Parity exact on RTX 4070; compute-sanitizer could not run under WSL2 (see cuts) |
| M4 benchmark + profile | done | [#7](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/7) | 3.96x / 7.55x / 10.53x end-to-end at 320x240 / 640x480 / 1280x720; nsys has no GPU kernel data under WSL2 |
| M5 networking | done | [#18](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/18) | Fast DDS defaults drop 13-37% across containers; Zenoh and Fast DDS LARGE_DATA 0% |
| M6 README + polish | done | [#17](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/17) | README rewrite, before/after plot, description and topics |
| M7 release | done | [#17](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/17) | v0.1.0: https://github.com/Arsh624/ros2-gpu-pointcloud/releases/tag/v0.1.0 |

## Global constraints

- Build and run everything inside Docker; source is volume-mounted, the image is not rebuilt for source changes.
- Core files (`filter_*`, `synthetic`) must not include any ROS header.
- CPU and CUDA must give the same voxel count exactly and centroids within 1e-4 m per axis.
- Floating-point contraction is off on both compilers (`-ffp-contract=off`, nvcc `-fmad=false`) so keys match bit for bit.
- Defaults: `voxel_size=0.05`, `min_range=0.3`, `max_range=10.0` (metres).
- One branch `mN-<slug>` and one PR per milestone, merge commit, conventional commit prefixes.
- No number goes into any file unless it was measured in this repo.

## Interfaces (fixed here, used by every task)

`include/gpu_pointcloud/filter.hpp`, namespace `gpu_pointcloud`:

```cpp
struct PointXYZ { float x, y, z; };                       // 12 bytes, no padding
struct FilterParams { float voxel_size = 0.05f; float min_range = 0.3f; float max_range = 10.0f; };
struct FilterStats { std::size_t in, non_finite, out_of_range, index_overflow, out; };

void validate(const FilterParams&);                       // throws std::invalid_argument
bool voxel_key(const PointXYZ&, float voxel_size, std::uint64_t& key);  // false = index outside [-2^20, 2^20)

class Filter {
 public:
  virtual ~Filter() = default;
  virtual FilterStats apply(const PointXYZ* in, std::size_t n, std::vector<PointXYZ>& out) = 0;
};
std::unique_ptr<Filter> make_cpu_filter(const FilterParams&);
std::unique_ptr<Filter> make_cuda_filter(const FilterParams&, std::size_t reserve_points);
bool cuda_compiled();     // false when built with WITH_CUDA=OFF
bool cuda_device_present();
```

Semantics, in order:

1. Drop a point if any coordinate is non-finite.
2. `r = sqrt(x*x + y*y + z*z)` computed in double; keep `min_range <= r <= max_range`.
3. Per axis `i = floor(coord / voxel_size)` in float32. Drop and count the point if any `i` is outside `[-2^20, 2^20)`.
4. `key = (ix + 2^20) << 42 | (iy + 2^20) << 21 | (iz + 2^20)`.
5. Output one centroid per key, summed in double, emitted as float32, in ascending key order.

`make_cuda_filter` throws `std::runtime_error("gpu_pointcloud was built without CUDA")` when `cuda_compiled()` is false.

`include/gpu_pointcloud/synthetic.hpp`:

```cpp
std::vector<PointXYZ> make_synthetic_cloud(int width, int height, std::uint32_t seed = 42);
```

Pinhole camera at the origin looking along +z, `fx = fy = 0.8 * width`, principal point at the centre. Scene: floor plane, back wall beyond `max_range` in places, 3 boxes and 2 spheres. Gaussian depth noise sigma 5 mm, 5% of pixels NaN, `std::mt19937`. Always returns `width * height` points.

`include/gpu_pointcloud/cloud_conv.hpp` (ROS):

```cpp
void from_pointcloud2(const sensor_msgs::msg::PointCloud2&, std::vector<PointXYZ>& out);  // throws std::invalid_argument
void to_pointcloud2(const PointXYZ* pts, std::size_t n, const std_msgs::msg::Header&, sensor_msgs::msg::PointCloud2& out);
```

CMake: target `gpu_pointcloud_core` (static). Options `WITH_CUDA` (default ON when a CUDA compiler is found) and `CORE_ONLY` (default OFF; skips ament and ROS targets, used by the CUDA compile job in CI). `CMAKE_CUDA_ARCHITECTURES` defaults to 89.

## Review focus

Inputs the brief does not spell out that are likely to bite; each has a test in the task named.

1. `voxel_size <= 0`, non-finite params or `min_range > max_range`: `validate` throws (M2, `test_filter_cpu`).
2. PointCloud2 whose `data` is shorter than `width * height * point_step`, or without float32 x/y/z: `from_pointcloud2` throws (M2, `test_cloud_conv`).
3. Big-endian PointCloud2 on a little-endian host: `from_pointcloud2` throws rather than misreading (M2, `test_cloud_conv`).
4. A cloud larger than the CUDA filter's `reserve_points`: buffers grow once, result still correct (M3, `test_parity_cuda`).
5. A tiny `voxel_size` that pushes indices past 21 bits: points dropped and counted in `index_overflow`, identically on both backends (M2 and M3).

## Tasks

### M0: zenoh smoke test
- [x] Two containers plus `rmw_zenohd` on one bridge network, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, talker and listener from `demo_nodes_cpp`. Pass = listener prints `I heard`.
- [x] Record the result and the needed config here.

Result (rmw_zenoh_cpp 0.2.13, `ros:jazzy-ros-base`): passed, the listener printed `I heard: [Hello World: N]` once per second. M5 compares Fast DDS with Zenoh as planned.

Config needed on the talker and listener containers (the router needs none):

```
RMW_IMPLEMENTATION=rmw_zenoh_cpp
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/router:7447"];listen/endpoints=["tcp/0.0.0.0:0"]'
```

`connect/endpoints` alone is not enough. With the default `listen` endpoint (`tcp/localhost:0`) topics are discovered through the router but no data arrives, because the nodes advertise a loopback address that the other container cannot reach (`Unable to connect to any locator of scouted peer ... [tcp/[::1]:port]`).

### M1: skeleton + CI (branch `m1-skeleton`)
- [x] `.gitignore`, `.gitattributes`, `LICENSE`; `git init`; first commit excludes `BRIEF.local.md`.
- [x] `docker/Dockerfile` (digest-pinned) and `docker/compose.dev.yml`; image builds.
- [x] `gpu_pointcloud/package.xml`, `CMakeLists.txt` with the options above, headers, and a compiling core. Pass = `colcon build` inside the dev container.
- [x] `docs/ARCHITECTURE.md`, `docs/DESIGN.md` (with installed package versions), README stub, `CHANGELOG.md`.
- [x] `.github/workflows/ci.yml`: job `ros-cpu` in `ros:jazzy-ros-base` (build with `-DWITH_CUDA=OFF`, test, `colcon test-result --verbose`); job `cuda-compile` in the CUDA devel image (`cmake -DCORE_ONLY=ON` then build).
- [x] Create the GitHub repo, push, PR, CI green, merge, link the run here.

### M2: CPU filter (branch `m2-cpu-filter`)
- [x] `test/test_filter_cpu.cpp` first. Cases: empty input; all out of range; NaN and Inf dropped; exact min and max range kept; two points in one voxel give their mean; `x = 0.05` lands in index 1 and `x = -0.0001` in index -1; output ascending by key; 4x4x1 grid with hand-computed centroids; invalid params throw; index overflow counted. Run and show it failing.
- [x] `src/filter_cpu.cpp`: keys, `std::sort` of (key, index), linear reduce. Tests pass.
- [x] `src/synthetic.cpp` with a determinism test (two calls give identical bytes; size is `width * height`; contains NaN and beyond-range points).
- [x] `test/test_cloud_conv.cpp` first (rgb field, `point_step` 32, round trip; short data, missing field and big-endian throw), then `src/cloud_conv.cpp`.
- [x] `src/filter_node.cpp` with `backend`, `voxel_size`, `min_range`, `max_range`. Verify in the container with a one-shot publisher and `ros2 topic echo --once /output --field width`.
- [x] PR with output, merge.

### M3: CUDA filter (branch `m3-cuda-filter`)
- [x] `test/test_parity_cuda.cpp` first: 3 resolutions x voxel sizes {0.02, 0.05, 0.1}, equal count and max |delta| <= 1e-4; the M2 edge cases through CUDA; reserve-growth case; `GTEST_SKIP` when `cuda_device_present()` is false.
- [x] `src/filter_cuda.cu`: key + flag kernel, `DeviceSelect::Flagged`, `DeviceRadixSort::SortPairs`, `DeviceReduce::ReduceByKey`, divide kernel; preallocated device buffers and CUB temp storage, pinned host buffers, one stream.
- [ ] `compute-sanitizer` on the 640x480 case: 0 errors.
- [x] Node verified with `backend:=cuda`. PR, merge.

### M4: benchmark + profile (branch `m4-benchmark`)
- [x] `bench/bench_filter.cpp`: flags `--backend`, `--res`, `--iters`, `--warmup`, `--csv`; 20 warmup + 200 timed; CUDA end-to-end and GPU-only timings; NVTX ranges per stage.
- [x] `scripts/bench.sh` writes `results/bench.csv` and `results/env.txt`. Sanity-check scaling before reporting.
- [x] `scripts/profile.sh`: `nsys profile -t cuda,nvtx`, then `nsys stats` to `results/nsys_summary.txt`. Fallback: per-stage cudaEvent timings.
- [x] PR with the table, merge.

### M5 to M7
Not started. Each has a GitHub issue with the exact task and acceptance criteria (linked in the table above). Other follow-ups: compute-sanitizer run (#3), input copy removal (#5), Nsight timeline screenshot (#6), PCL baseline (#8), multi-threaded CPU baseline (#9).

## Cuts and deviations



- **`cloud_publisher_node` arrived in M2, not M5.** M2 needs a one-shot publisher of the synthetic cloud to verify the filter node, so the basic publisher was written now. M5 adds the latency listener and any rate or QoS options it needs.
- **Two files beyond the brief's layout:** `include/gpu_pointcloud/detail/voxel_math.hpp` (per-point classification shared by the CPU code and the CUDA kernel, so both run identical arithmetic) and `test/filter_cases.hpp` (the edge-case suite, instantiated once per backend).
- **compute-sanitizer not run (M3).** The tool cannot attach to the GPU under WSL2 without an administrator registry change on Windows ("Failed to initialize WDDM debugger interface", "Device not supported"). The M3 requirement "sanitizer clean" is not met; tracked in issue #3. Details in `docs/DESIGN.md`.
- **`FilterStats::device_ms` added (M3)** so the benchmark can report GPU-only time without a second code path.
- **nsys captures no GPU kernel data under WSL2 (M4).** NVTX ranges and CUDA API calls are recorded; kernel and memory rows are not. Per-stage cost is therefore measured with cudaEvents (`results/stages.csv`), as the brief's fallback allows. `results/nsys_summary.txt` is still committed, with host-side durations only.
- **Extra M4 outputs:** `results/stages.csv` and `results/bench.md` (the benchmark's stdout), and `Filter::last_stage_times()`.

## Owner tasks

- Open `results/profile.nsys-rep` in Nsight Systems on Windows and save `results/nsys_timeline.png` (issue #6). The file is not committed; regenerate it with `scripts/profile.sh` if needed.
- Enable the NVIDIA debugger interface or use a native Linux machine to run compute-sanitizer (issue #3).
- Read the README before the v0.1.0 release.

## Next session

M5 to M7 are not started. Start with M5 using the Zenoh config recorded under M0. `cloud_publisher_node` already exists; it still needs the latency listener, a `compose.net.yml` and `scripts/net_bench.sh`.

## Rigor pass (after M4)

Priority changed after M4: strengthen the evidence before adding features. Items, each its own PR:

| Item | State | PR | Key result |
|---|---|---|---|
| A. PCL and OpenMP CPU baselines (#8, #9) | done | [#13](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/13) | Best CPU is OpenMP on 20 threads. CUDA end-to-end vs best CPU: 0.55x / 0.86x / 1.12x at 320x240 / 640x480 / 1280x720. GPU-only: 1.04x / 2.39x / 4.64x |
| B. Fix the input copy bottleneck (#5) | done | [#14](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/14) | CUDA end-to-end 0.442 / 0.998 / 2.356 ms -> 0.430 / 0.792 / 1.702 ms (main run); vs best CPU 0.57x / 1.11x / 1.63x. Over 7 launches 1280x720 is ~1.5 ms (5) or ~2.4 ms (2) |
| C. Explain the CUDA p99 tail at 1280x720 | done | [#15](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/15) | Environmental: GPU idles to P8 (210 MHz) during the CPU runs and needs ~250-320 ms to clock up; the tail was the first timed iterations. Benchmark warmup now also lasts 500 ms. With idle gaps CUDA 1280x720 is 3.0 ms at 30 Hz and 7.0 ms at 10 Hz (cpu_omp: 3.2 and 4.7 ms) |
| D. Node-level latency and CPU cost per frame | done | [#16](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/16) | 1280x720 at 30 Hz: CUDA 3.48 ms on 0.96 cores, cpu_omp 12 threads 6.68 ms on 4.55 cores; at 10 Hz CUDA 11.07 ms (idle clock-down). Image lacked Fast DDS (nodes ran on Cyclone); with middleware defaults large clouds dropped in-container, fixed for node runs with `docker/fastdds_shm_large.xml` |
| E. M5 Fast DDS vs Zenoh (#10) | done | [#18](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/18) | Raw 3.7 MB cloud across containers: Fast DDS defaults drop 37.4% / 32.5% at 10 / 30 Hz; Zenoh 0%, p50 4.28 / 4.00 ms; Fast DDS LARGE_DATA (tuned) 0%, p50 4.15 / 4.14 ms |
| F. M6 README rewrite (#11), M7 release (#12) | done | [#17](https://github.com/Arsh624/ros2-gpu-pointcloud/pull/17) | README rewritten (results led by node latency and cores), quick start run once in a fresh container, `results/before_after.png`, CHANGELOG v0.1.0, release v0.1.0 |

Notes for whoever continues:

- Build with `colcon build --cmake-args -DWITH_PCL=ON` in the dev container; `scripts/bench.sh` expects the PCL baseline.
- Earlier benchmark runs live in `results/history/<name>/`. Move the current `results/*.csv`, `bench.md` and `env.txt` there before a run that changes the code under test, and add a row to the README history table.
- Python on the Windows host is `python3`; write files in binary mode from it (text mode writes CRLF, which breaks the shell scripts inside the container).
- The CUDA input stage has two speeds that are fixed per filter instance (about 0.93 or 1.74 ms at 1280x720). Host memory, CPU core, huge pages and the bare transfer call were ruled out; see `docs/DESIGN.md`, "Two transfer rates". Always compare CUDA variants inside one process (`bench_cuda_input`) or over several launches (`results/cuda_launches.csv`), never from two single launches.
- Tight-loop numbers are not what a node gets. After an idle gap both backends are slower, the GPU much more so (`results/idle_gap.csv`, `docs/DESIGN.md` "Idle GPU"). Item D must measure at real publish rates (10 and 30 Hz), not only at the maximum rate, and should expect the OpenMP backend to win at 10 Hz on this laptop.
- Do not run anything else in the container while a benchmark runs; the results are sensitive to GPU and CPU load.

## Where this stopped, and exact next steps

Items D, E and F are done and v0.1.0 is released. Next: README network section and v0.2.0. Everything below is what a new session needs; nothing else is in anyone's head.

### Before anything else

```
docker compose -f docker/compose.dev.yml up -d
docker exec gpc-dev /ros_entrypoint.sh bash -c 'cd /ws && colcon build --cmake-args -DWITH_PCL=ON && colcon test && colcon test-result'
```

Expect 81 tests, 0 failures, 1 skipped. From Git Bash on Windows set `MSYS_NO_PATHCONV=1` first. The container's `/ws/build` is lost when the container is recreated; the build takes about a minute.

### Item D: done (PR #16)

The two discarded runs were not caused by stale shared-memory files: a fresh container dropped just as much. Root cause and fix are in `docs/DESIGN.md`, "Middleware for same-host node runs". In short: the image had no Fast DDS, so the default RMW was Cyclone DDS; with defaults, best-effort clouds of 0.9 MB and more do not fit the UDP path (Docker VM `net.core.rmem_max = 212992`) or Fast DDS's 512 KB SHM segment. `node_bench.sh` now runs Fast DDS with `docker/fastdds_shm_large.xml` and warns if a row is missing messages. The image must be rebuilt once (`docker compose -f docker/compose.dev.yml build`) to get Fast DDS.

### Item E: done (PR #18)

`latency_listener_node` (stats in `latency_stats.hpp`, tested), publisher `filter:=true`, `docker/compose.net.yml`, `scripts/net_bench.sh` (runs on the host), `results/net.csv` with 12 rows: 8 with middleware defaults plus 4 for the one labelled tuning (Fast DDS `LARGE_DATA`), added because the defaults dropped messages. Method and choices: `docs/DESIGN.md`, "Network benchmark". Deviation from the plan above: the filtered payload is filtered once by the publisher, not by a filter node in the talker container.

### Item F: done (PR #17)

README rewritten; every quick-start command was run once in a fresh dev container (build, test, node with each backend, `bench.sh`, `node_bench.sh`, plot). `bench_filter --dump` and `scripts/plot_before_after.py` make `results/before_after.png`. The dev container now has a 1 GB `/dev/shm` and sets `FASTRTPS_DEFAULT_PROFILES_FILE` (each Fast DDS process takes a 16 MB segment; with 64 MB, a fourth participant such as `ros2 topic echo` failed to start). Repo description and topics set; v0.1.0 released.

After E: add a "Networking" section to the README (before Limitations) from `results/net.csv`, replace the networking limitation bullet, add a v0.2.0 CHANGELOG section and `gh release create v0.2.0`.

### Open questions worth an hour each, in order of value

1. The rising-rate sweep (`RATE_SWEEP=1 bash scripts/node_bench.sh`) was written but not run. It gives the maximum sustainable input rate per backend.
2. Why the CUDA transfer stage has two speeds per filter instance (`docs/DESIGN.md`, "Two transfer rates"). Next thing to try: whether it follows the stream (create several streams in one process and time the same transfer on each).
3. Whether the Windows "prefer maximum performance" GPU power mode removes the idle slowdown (owner must change it; then rerun `bench_idle_gap`).
