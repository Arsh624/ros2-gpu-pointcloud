#include "gpu_pointcloud/filter.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace gpu_pointcloud {

bool cuda_device_present() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Implemented in M3.
std::unique_ptr<Filter> make_cuda_filter(const FilterParams&, std::size_t) {
  throw std::runtime_error("CUDA backend is not implemented yet");
}

}  // namespace gpu_pointcloud
