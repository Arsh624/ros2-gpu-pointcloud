// PCL reference pipeline for the benchmark: radial range filter + pcl::VoxelGrid.
// Built only with -DWITH_PCL=ON. Not part of the core library, which has no PCL dependency.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "gpu_pointcloud/filter.hpp"

namespace gpu_pointcloud::bench {

class PclBaseline {
 public:
  explicit PclBaseline(const FilterParams& params);
  ~PclBaseline();

  // Converts the input into a pcl::PointCloud<pcl::PointXYZ>. Not part of the timed work: a PCL
  // user would already hold their data in this form.
  void set_input(const PointXYZ* points, std::size_t n);

  // The timed work: drop non-finite and out-of-range points into a second cloud (PCL has no
  // radial range filter; this is the loop a PCL user would write), then pcl::VoxelGrid::filter.
  FilterStats run();

  // Copies the last result out, in PCL's order. Not timed.
  void copy_output(std::vector<PointXYZ>& out) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu_pointcloud::bench
