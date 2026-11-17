#include "gpu_pointcloud/filter.hpp"

#include <stdexcept>

namespace gpu_pointcloud {

// Not implemented yet: tests first.
int default_cpu_threads() { return 0; }

std::unique_ptr<Filter> make_cpu_omp_filter(const FilterParams&, int) {
  throw std::runtime_error("OpenMP CPU backend is not implemented yet");
}

}  // namespace gpu_pointcloud
