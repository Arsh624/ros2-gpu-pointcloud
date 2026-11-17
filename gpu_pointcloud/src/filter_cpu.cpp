#include "gpu_pointcloud/filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "gpu_pointcloud/detail/voxel_math.hpp"

namespace gpu_pointcloud {

void validate(const FilterParams& p) {
  const bool ok = std::isfinite(p.voxel_size) && p.voxel_size > 0.0f && std::isfinite(p.min_range) &&
                  p.min_range >= 0.0f && std::isfinite(p.max_range) && p.max_range >= p.min_range;
  if (!ok) {
    throw std::invalid_argument(
        "invalid filter params: need voxel_size > 0 and 0 <= min_range <= max_range, all finite (got voxel_size=" +
        std::to_string(p.voxel_size) + ", min_range=" + std::to_string(p.min_range) +
        ", max_range=" + std::to_string(p.max_range) + ")");
  }
}

bool voxel_key(const PointXYZ& p, float voxel_size, std::uint64_t& key) {
  return detail::pack_key(p, voxel_size, key);
}

namespace {

// Single-threaded reference: key every point, sort (key, index), then one linear pass that
// averages each run of equal keys. Sorting pairs makes ties break by input index, so the
// summation order within a voxel is the input order.
class CpuFilter final : public Filter {
 public:
  explicit CpuFilter(const FilterParams& params) : params_(params) { validate(params_); }

  FilterStats apply(const PointXYZ* in, std::size_t n, std::vector<PointXYZ>& out) override {
    if (n > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error("cloud has more than 2^32 - 1 points");
    }
    FilterStats stats;
    stats.in = n;
    out.clear();
    keyed_.clear();

    const double min_range = params_.min_range, max_range = params_.max_range;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint64_t key;
      switch (detail::classify(in[i], params_.voxel_size, min_range, max_range, key)) {
        case detail::kKept: keyed_.emplace_back(key, static_cast<std::uint32_t>(i)); break;
        case detail::kNonFinite: ++stats.non_finite; break;
        case detail::kOutOfRange: ++stats.out_of_range; break;
        case detail::kIndexOverflow: ++stats.index_overflow; break;
      }
    }
    std::sort(keyed_.begin(), keyed_.end());

    for (std::size_t i = 0; i < keyed_.size();) {
      const std::uint64_t key = keyed_[i].first;
      double sx = 0.0, sy = 0.0, sz = 0.0;
      std::size_t count = 0;
      for (; i < keyed_.size() && keyed_[i].first == key; ++i, ++count) {
        const PointXYZ& p = in[keyed_[i].second];
        sx += p.x;
        sy += p.y;
        sz += p.z;
      }
      const double inv = 1.0 / static_cast<double>(count);
      out.push_back({static_cast<float>(sx * inv), static_cast<float>(sy * inv), static_cast<float>(sz * inv)});
    }
    stats.out = out.size();
    return stats;
  }

 private:
  FilterParams params_;
  std::vector<std::pair<std::uint64_t, std::uint32_t>> keyed_;  // reused across frames
};

}  // namespace

std::unique_ptr<Filter> make_cpu_filter(const FilterParams& params) {
  return std::make_unique<CpuFilter>(params);
}

#ifdef GPU_POINTCLOUD_WITH_CUDA
bool cuda_compiled() { return true; }
#else
bool cuda_compiled() { return false; }
bool cuda_device_present() { return false; }
std::unique_ptr<Filter> make_cuda_filter(const FilterParams&, std::size_t) {
  throw std::runtime_error("gpu_pointcloud was built without CUDA (WITH_CUDA=OFF); use backend 'cpu'");
}
#endif

}  // namespace gpu_pointcloud
