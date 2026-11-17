// Range + voxel-grid point cloud filter with CPU and CUDA backends.
// This header is ROS-free on purpose: tests and the benchmark use it without a ROS graph.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gpu_pointcloud {

struct PointXYZ {
  float x, y, z;
};
static_assert(sizeof(PointXYZ) == 12, "PointXYZ must be 3 packed floats");

struct FilterParams {
  float voxel_size = 0.05f;  // metres
  float min_range = 0.3f;    // metres, inclusive
  float max_range = 10.0f;   // metres, inclusive
};

// Where every input point went. in == non_finite + out_of_range + index_overflow + (points kept).
struct FilterStats {
  std::size_t in = 0;
  std::size_t non_finite = 0;
  std::size_t out_of_range = 0;
  std::size_t index_overflow = 0;
  std::size_t out = 0;  // occupied voxels
  // CUDA only: time of the device pipeline for this call, measured with cudaEvents and excluding
  // the host-to-device input copy and device-to-host output copy. 0 on the CPU backend.
  double device_ms = 0.0;
};

struct StageTime {
  const char* name;
  double ms;
};

// Throws std::invalid_argument unless voxel_size > 0, 0 <= min_range <= max_range, all finite.
void validate(const FilterParams& params);

// Packs the voxel index of `p` into a sortable key: 21 bits per axis, x in the high bits.
// Returns false (key untouched) when an index is outside [-2^20, 2^20). `p` must be finite.
bool voxel_key(const PointXYZ& p, float voxel_size, std::uint64_t& key);

// Filters a cloud: drop non-finite points, keep min_range <= |p| <= max_range, then emit one
// centroid per occupied voxel in ascending key order. See docs/DESIGN.md for exact semantics.
class Filter {
 public:
  virtual ~Filter() = default;
  // `out` is replaced. `in` may be null when n == 0. Not thread-safe; reuse one instance per stream.
  virtual FilterStats apply(const PointXYZ* in, std::size_t n, std::vector<PointXYZ>& out) = 0;
  // Per-stage timings of the last non-empty apply() call, in pipeline order. Empty on backends
  // that do not measure stages (CPU).
  virtual std::vector<StageTime> last_stage_times() const { return {}; }
};

// Both factories validate `params` and throw std::invalid_argument when they are bad.
std::unique_ptr<Filter> make_cpu_filter(const FilterParams& params);
// `reserve_points` sizes the preallocated device buffers; larger clouds grow them once.
// Throws std::runtime_error when built without CUDA or when no device is usable.
std::unique_ptr<Filter> make_cuda_filter(const FilterParams& params, std::size_t reserve_points);

// True when the library was compiled with the CUDA backend (WITH_CUDA=ON).
bool cuda_compiled();
// True when a usable CUDA device is visible at run time. Always false without cuda_compiled().
bool cuda_device_present();

}  // namespace gpu_pointcloud
