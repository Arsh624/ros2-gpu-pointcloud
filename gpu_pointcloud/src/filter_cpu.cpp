#include "gpu_pointcloud/filter.hpp"

namespace gpu_pointcloud {

#ifdef GPU_POINTCLOUD_WITH_CUDA
bool cuda_compiled() { return true; }
#else
bool cuda_compiled() { return false; }
bool cuda_device_present() { return false; }
#endif

}  // namespace gpu_pointcloud
