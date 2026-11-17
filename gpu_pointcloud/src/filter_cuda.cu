#include "gpu_pointcloud/filter.hpp"

#include <cuda_runtime.h>

namespace gpu_pointcloud {

bool cuda_device_present() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace gpu_pointcloud
