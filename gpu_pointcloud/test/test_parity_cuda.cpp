// CUDA backend tests: the shared edge cases, then CPU-vs-CUDA parity on synthetic and random
// clouds. Everything is skipped when no CUDA device is visible, so CI (no GPU) stays green.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "filter_cases.hpp"
#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"

using namespace gpu_pointcloud;

namespace {

constexpr double kTolerance = 1e-4;  // metres per axis; the tolerance stated in the README

std::unique_ptr<Filter> make_cuda_small(const FilterParams& p) { return make_cuda_filter(p, 1024); }

#define SKIP_WITHOUT_GPU() \
  if (!cuda_device_present()) GTEST_SKIP() << "no CUDA device visible; GPU tests run only on a machine with a GPU"

// Runs both backends and requires identical stats and voxel count, and centroids within tolerance.
// Returns the largest per-axis difference seen.
double expect_parity(Filter& cpu, Filter& cuda, const std::vector<PointXYZ>& cloud) {
  std::vector<PointXYZ> a, b;
  const FilterStats sa = cpu.apply(cloud.data(), cloud.size(), a);
  const FilterStats sb = cuda.apply(cloud.data(), cloud.size(), b);
  EXPECT_EQ(sa.in, sb.in);
  EXPECT_EQ(sa.non_finite, sb.non_finite);
  EXPECT_EQ(sa.out_of_range, sb.out_of_range);
  EXPECT_EQ(sa.index_overflow, sb.index_overflow);
  EXPECT_EQ(sa.out, sb.out);
  EXPECT_EQ(a.size(), b.size());
  if (a.size() != b.size()) return -1.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max({worst, std::fabs(double(a[i].x) - b[i].x), std::fabs(double(a[i].y) - b[i].y),
                      std::fabs(double(a[i].z) - b[i].z)});
  }
  EXPECT_LE(worst, kTolerance);
  return worst;
}

}  // namespace

using gpu_pointcloud::cases::FilterCases;
INSTANTIATE_TEST_SUITE_P(Cuda, FilterCases,
                         ::testing::Values(gpu_pointcloud::cases::Backend{"cuda", true, &make_cuda_small}),
                         ::testing::PrintToStringParamName());

class Parity : public ::testing::TestWithParam<std::tuple<std::pair<int, int>, float>> {};

TEST_P(Parity, SyntheticCloudMatchesCpu) {
  SKIP_WITHOUT_GPU();
  const auto [res, voxel_size] = GetParam();
  FilterParams p;
  p.voxel_size = voxel_size;
  const auto cloud = make_synthetic_cloud(res.first, res.second);
  auto cpu = make_cpu_filter(p);
  auto cuda = make_cuda_filter(p, cloud.size());
  const double worst = expect_parity(*cpu, *cuda, cloud);
  std::vector<PointXYZ> out;
  EXPECT_GT(cuda->apply(cloud.data(), cloud.size(), out).out, 1000u) << "parity on an empty result proves nothing";
  RecordProperty("max_abs_delta_m", std::to_string(worst));
  std::printf("  %dx%d voxel=%.2f: max |delta| = %.3g m\n", res.first, res.second, voxel_size, worst);
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions, Parity,
    ::testing::Combine(::testing::Values(std::make_pair(320, 240), std::make_pair(640, 480),
                                         std::make_pair(1280, 720)),
                       ::testing::Values(0.02f, 0.05f, 0.1f)));

TEST(ParityExtra, RandomCloudMatchesCpu) {
  // Uniform points in a cube that straddles the origin and both range limits: far more voxel
  // boundaries and negative indices than the camera scene has.
  SKIP_WITHOUT_GPU();
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> coord(-8.f, 8.f);
  std::vector<PointXYZ> cloud(300000);
  for (auto& p : cloud) p = {coord(rng), coord(rng), coord(rng)};
  for (float voxel_size : {0.01f, 0.05f, 0.5f}) {
    FilterParams p;
    p.voxel_size = voxel_size;
    auto cpu = make_cpu_filter(p);
    auto cuda = make_cuda_filter(p, cloud.size());
    expect_parity(*cpu, *cuda, cloud);
  }
}

TEST(ParityExtra, GrowsBeyondReservedSize) {
  SKIP_WITHOUT_GPU();
  const auto cloud = make_synthetic_cloud(320, 240);
  auto cpu = make_cpu_filter({});
  auto cuda = make_cuda_filter({}, 16);  // far smaller than the cloud
  expect_parity(*cpu, *cuda, cloud);
}

TEST(ParityExtra, ReusedAcrossFramesOfDifferentSizes) {
  SKIP_WITHOUT_GPU();
  const auto big = make_synthetic_cloud(640, 480);
  const auto small = make_synthetic_cloud(320, 240, 7);
  auto cpu = make_cpu_filter({});
  auto cuda = make_cuda_filter({}, big.size());
  for (const auto* cloud : {&big, &small, &big, &small}) expect_parity(*cpu, *cuda, *cloud);
}

TEST(ParityExtra, WideKeyRangeFallsBackToFullWidthSort) {
  // A tiny voxel with a large range needs the full 21 bits per axis.
  SKIP_WITHOUT_GPU();
  FilterParams p;
  p.voxel_size = 1e-4f;
  p.max_range = 100.f;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> coord(-90.f, 90.f);
  std::vector<PointXYZ> cloud(50000);
  for (auto& q : cloud) q = {coord(rng), coord(rng), coord(rng)};
  auto cpu = make_cpu_filter(p);
  auto cuda = make_cuda_filter(p, cloud.size());
  expect_parity(*cpu, *cuda, cloud);
}

TEST(ParityExtra, ReportsDeviceTime) {
  SKIP_WITHOUT_GPU();
  const auto cloud = make_synthetic_cloud(320, 240);
  std::vector<PointXYZ> out;
  const FilterStats s = make_cuda_filter({}, cloud.size())->apply(cloud.data(), cloud.size(), out);
  EXPECT_GT(s.device_ms, 0.0);
}

TEST(ParityExtra, StageTimesCoverTheDevicePipeline) {
  SKIP_WITHOUT_GPU();
  const auto cloud = make_synthetic_cloud(640, 480);
  std::vector<PointXYZ> out;
  auto cuda = make_cuda_filter({}, cloud.size());
  const FilterStats s = cuda->apply(cloud.data(), cloud.size(), out);
  const auto stages = cuda->last_stage_times();
  ASSERT_EQ(stages.size(), 8u);
  EXPECT_STREQ(stages.front().name, "h2d_copy");
  EXPECT_STREQ(stages.back().name, "d2h_copy");
  double device = 0.0;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    EXPECT_GE(stages[i].ms, 0.0) << stages[i].name;
    if (i != 0 && i + 1 != stages.size()) device += stages[i].ms;
  }
  // device_ms spans exactly the stages between the two copies.
  EXPECT_NEAR(device, s.device_ms, 0.01 + 0.02 * s.device_ms);
  EXPECT_TRUE(make_cpu_filter({})->last_stage_times().empty());
}
