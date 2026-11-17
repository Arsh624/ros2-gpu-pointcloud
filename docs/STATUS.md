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
| M3 CUDA filter | in review | | Parity exact on RTX 4070; compute-sanitizer could not run under WSL2 (see cuts) |
| M4 benchmark + profile | pending | | |
| M5 networking | pending | | |
| M6 README + polish | pending | | |
| M7 release | pending | | |

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
- [ ] `bench/bench_filter.cpp`: flags `--backend`, `--res`, `--iters`, `--warmup`, `--csv`; 20 warmup + 200 timed; CUDA end-to-end and GPU-only timings; NVTX ranges per stage.
- [ ] `scripts/bench.sh` writes `results/bench.csv` and `results/env.txt`. Sanity-check scaling before reporting.
- [ ] `scripts/profile.sh`: `nsys profile -t cuda,nvtx`, then `nsys stats` to `results/nsys_summary.txt`. Fallback: per-stage cudaEvent timings.
- [ ] PR with the table, merge.

### M5 to M7
As written in the brief; expanded here when M4 is merged.

## Cuts and deviations



- **`cloud_publisher_node` arrived in M2, not M5.** M2 needs a one-shot publisher of the synthetic cloud to verify the filter node, so the basic publisher was written now. M5 adds the latency listener and any rate or QoS options it needs.
- **Two files beyond the brief's layout:** `include/gpu_pointcloud/detail/voxel_math.hpp` (per-point classification shared by the CPU code and the CUDA kernel, so both run identical arithmetic) and `test/filter_cases.hpp` (the edge-case suite, instantiated once per backend).
- **compute-sanitizer not run (M3).** The tool cannot attach to the GPU under WSL2 without an administrator registry change on Windows ("Failed to initialize WDDM debugger interface", "Device not supported"). The M3 requirement "sanitizer clean" is not met; tracked in issue #3. Details in `docs/DESIGN.md`.
- **`FilterStats::device_ms` added (M3)** so the benchmark can report GPU-only time without a second code path.
