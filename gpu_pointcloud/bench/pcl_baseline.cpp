#include "pcl_baseline.hpp"

#include <cmath>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace gpu_pointcloud::bench {

struct PclBaseline::Impl {
  using Cloud = pcl::PointCloud<pcl::PointXYZ>;
  FilterParams params;
  Cloud::Ptr input{new Cloud};
  Cloud::Ptr in_range{new Cloud};
  Cloud output;
  pcl::VoxelGrid<pcl::PointXYZ> grid;
};

PclBaseline::PclBaseline(const FilterParams& params) : impl_(new Impl) {
  validate(params);
  impl_->params = params;
  impl_->grid.setLeafSize(params.voxel_size, params.voxel_size, params.voxel_size);
}

PclBaseline::~PclBaseline() = default;

void PclBaseline::set_input(const PointXYZ* points, std::size_t n) {
  auto& cloud = *impl_->input;
  cloud.points.resize(n);
  for (std::size_t i = 0; i < n; ++i) cloud.points[i] = pcl::PointXYZ(points[i].x, points[i].y, points[i].z);
  cloud.width = static_cast<std::uint32_t>(n);
  cloud.height = 1;
  cloud.is_dense = false;
}

FilterStats PclBaseline::run() {
  FilterStats stats;
  const auto& in = impl_->input->points;
  auto& kept = *impl_->in_range;
  stats.in = in.size();

  const double min_range = impl_->params.min_range, max_range = impl_->params.max_range;
  kept.points.clear();
  for (const pcl::PointXYZ& p : in) {
    if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))) {
      ++stats.non_finite;
      continue;
    }
    const double x = p.x, y = p.y, z = p.z;
    const double r = std::sqrt(x * x + y * y + z * z);
    if (!(r >= min_range && r <= max_range)) {
      ++stats.out_of_range;
      continue;
    }
    kept.points.push_back(p);
  }
  kept.width = static_cast<std::uint32_t>(kept.points.size());
  kept.height = 1;
  kept.is_dense = true;

  impl_->grid.setInputCloud(impl_->in_range);
  impl_->grid.filter(impl_->output);
  stats.out = impl_->output.points.size();
  return stats;
}

void PclBaseline::copy_output(std::vector<PointXYZ>& out) const {
  out.clear();
  for (const pcl::PointXYZ& p : impl_->output.points) out.push_back({p.x, p.y, p.z});
}

}  // namespace gpu_pointcloud::bench
