#include "gpu_pointcloud/filter.hpp"

#include <stdexcept>

namespace gpu_pointcloud {

// Not implemented yet: the tests in test/test_filter_cpu.cpp are written first.
void validate(const FilterParams&) {}

bool voxel_key(const PointXYZ&, float, std::uint64_t&) { return false; }

namespace {
class CpuFilter final : public Filter {
 public:
  FilterStats apply(const PointXYZ*, std::size_t, std::vector<PointXYZ>&) override { return {}; }
};
}  // namespace

std::unique_ptr<Filter> make_cpu_filter(const FilterParams&) { return std::make_unique<CpuFilter>(); }

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
