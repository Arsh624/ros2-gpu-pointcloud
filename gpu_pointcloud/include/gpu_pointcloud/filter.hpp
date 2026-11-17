// Range + voxel-grid point cloud filter with CPU and CUDA backends.
// This header is ROS-free on purpose: tests and the benchmark use it without a ROS graph.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
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

// A read-only view of `count` points inside someone else's buffer: point i starts at
// data + i * stride and holds float32 x, y, z at the given byte offsets. This is the layout of a
// sensor_msgs/PointCloud2 without row padding, so a message can be filtered without repacking.
// Offsets and stride need no particular alignment.
struct CloudView {
  const std::uint8_t* data = nullptr;
  std::size_t count = 0;
  std::size_t stride = sizeof(PointXYZ);
  std::size_t x_offset = 0, y_offset = 4, z_offset = 8;
};

inline CloudView as_view(const PointXYZ* points, std::size_t n) {
  return {reinterpret_cast<const std::uint8_t*>(points), n, sizeof(PointXYZ), 0, 4, 8};
}

// Throws std::invalid_argument unless voxel_size > 0, 0 <= min_range <= max_range, all finite.
void validate(const FilterParams& params);
// Throws std::invalid_argument when a field would extend past `stride`, or when count > 0 and
// data is null.
void validate(const CloudView& view);

// Packs the voxel index of `p` into a sortable key: 21 bits per axis, x in the high bits.
// Returns false (key untouched) when an index is outside [-2^20, 2^20). `p` must be finite.
bool voxel_key(const PointXYZ& p, float voxel_size, std::uint64_t& key);

// Filters a cloud: drop non-finite points, keep min_range <= |p| <= max_range, then emit one
// centroid per occupied voxel in ascending key order. See docs/DESIGN.md for exact semantics.
class Filter {
 public:
  virtual ~Filter() = default;
  // Filters points read in place from a strided buffer. `out` is replaced.
  // Throws std::invalid_argument when the view is malformed (see validate(const CloudView&)).
  // Not thread-safe; reuse one instance per stream.
  virtual FilterStats apply(const CloudView& in, std::vector<PointXYZ>& out) = 0;
  // Convenience for packed points. `in` may be null when n == 0.
  FilterStats apply(const PointXYZ* in, std::size_t n, std::vector<PointXYZ>& out) {
    return apply(as_view(in, n), out);
  }
  // Per-stage timings of the last non-empty apply() call, in pipeline order. Empty on backends
  // that do not measure stages (CPU).
  virtual std::vector<StageTime> last_stage_times() const { return {}; }
};

// Both factories validate `params` and throw std::invalid_argument when they are bad.
std::unique_ptr<Filter> make_cpu_filter(const FilterParams& params);
// Multi-threaded CPU backend (OpenMP). Same output as make_cpu_filter, bit for bit.
// `threads` = 0 uses default_cpu_threads(); a negative count throws std::invalid_argument.
std::unique_ptr<Filter> make_cpu_omp_filter(const FilterParams& params, int threads = 0);
// Number of threads OpenMP uses by default on this machine.
int default_cpu_threads();
// How the CUDA backend gets a strided cloud onto the device.
enum class CudaInput {
  kAuto,    // the measured best: same as kRaw (data in docs/DESIGN.md, "Input path")
  kRaw,     // upload the bytes as they are and read x, y, z by offset on the device; layouts
            // that are not 4-byte aligned are repacked, since the device cannot read them
  kRepack,  // gather x, y, z into a packed host buffer first, then upload 12 bytes per point
};
// `reserve_points` sizes the preallocated device buffers; larger clouds grow them once.
// Throws std::runtime_error when built without CUDA or when no device is usable.
std::unique_ptr<Filter> make_cuda_filter(const FilterParams& params, std::size_t reserve_points,
                                         CudaInput input = CudaInput::kAuto);

// True when the library was compiled with the CUDA backend (WITH_CUDA=ON).
bool cuda_compiled();
// True when a usable CUDA device is visible at run time. Always false without cuda_compiled().
bool cuda_device_present();

}  // namespace gpu_pointcloud
