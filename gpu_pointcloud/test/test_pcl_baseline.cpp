// Checks that the PCL baseline in the benchmark does comparable work to our filter, so its
// timing is a fair reference. PCL is not expected to match bit for bit: see docs/DESIGN.md.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "../bench/pcl_baseline.hpp"
#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"

using namespace gpu_pointcloud;

namespace {

// PCL multiplies by 1/leaf where we divide by voxel_size, and it accumulates centroids in
// float, so a point on a voxel face can land in the neighbouring voxel. The voxel counts must
// agree within 1% and at least 99% of our voxels must also be occupied in PCL's result.
constexpr double kCountTolerance = 0.01;
constexpr double kMatchedFraction = 0.99;

struct Agreement {
  std::size_t ours = 0, pcl = 0;
  std::size_t same_voxel = 0;   // our centroids whose voxel also holds a PCL centroid
  double worst_delta = 0.0;     // largest per-axis centroid difference among those, metres
};

Agreement compare(int width, int height, float voxel_size) {
  FilterParams params;
  params.voxel_size = voxel_size;
  const auto cloud = make_synthetic_cloud(width, height);

  std::vector<PointXYZ> ours, theirs;
  make_cpu_filter(params)->apply(cloud.data(), cloud.size(), ours);
  bench::PclBaseline pcl(params);
  pcl.set_input(cloud.data(), cloud.size());
  pcl.run();
  pcl.copy_output(theirs);

  std::unordered_map<std::uint64_t, PointXYZ> by_voxel;
  for (const PointXYZ& p : theirs) {
    std::uint64_t key;
    if (voxel_key(p, voxel_size, key)) by_voxel[key] = p;
  }
  Agreement a;
  a.ours = ours.size();
  a.pcl = theirs.size();
  for (const PointXYZ& p : ours) {
    std::uint64_t key;
    if (!voxel_key(p, voxel_size, key)) continue;
    const auto it = by_voxel.find(key);
    if (it == by_voxel.end()) continue;
    ++a.same_voxel;
    a.worst_delta = std::max({a.worst_delta, std::fabs(double(p.x) - it->second.x),
                              std::fabs(double(p.y) - it->second.y), std::fabs(double(p.z) - it->second.z)});
  }
  return a;
}

}  // namespace

TEST(PclBaseline, ProducesComparableVoxelsToOurFilter) {
  for (const auto& res : {std::pair<int, int>{320, 240}, {640, 480}, {1280, 720}}) {
    for (float voxel_size : {0.02f, 0.05f, 0.1f}) {
      const Agreement a = compare(res.first, res.second, voxel_size);
      const double count_diff = std::fabs(double(a.ours) - double(a.pcl)) / double(a.ours);
      const double matched = double(a.same_voxel) / double(a.ours);
      std::printf("  %dx%d voxel=%.2f: ours=%zu pcl=%zu count diff=%.4f%% matched=%.4f%% worst delta=%.3g m\n",
                  res.first, res.second, voxel_size, a.ours, a.pcl, 100.0 * count_diff, 100.0 * matched,
                  a.worst_delta);
      EXPECT_GT(a.pcl, 0u);
      EXPECT_LE(count_diff, kCountTolerance);
      EXPECT_GE(matched, kMatchedFraction);
    }
  }
}
