#include <gtest/gtest.h>

#include <cstdint>

#include "filter_cases.hpp"
#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"

using namespace gpu_pointcloud;

namespace {

constexpr std::uint64_t kOffset = 1u << 20;

std::unique_ptr<Filter> make_cpu(const FilterParams& p) { return make_cpu_filter(p); }

}  // namespace

using gpu_pointcloud::cases::FilterCases;
INSTANTIATE_TEST_SUITE_P(Cpu, FilterCases,
                         ::testing::Values(gpu_pointcloud::cases::Backend{"cpu", false, &make_cpu}),
                         ::testing::PrintToStringParamName());

TEST(VoxelKey, PacksFloorIndicesWithOffset) {
  std::uint64_t key = 0;
  ASSERT_TRUE(voxel_key({0.05f, -0.0001f, 0.f}, 0.05f, key));
  EXPECT_EQ(key >> 42, kOffset + 1);                // x = 0.05   -> index 1
  EXPECT_EQ((key >> 21) & 0x1FFFFF, kOffset - 1);   // y = -1e-4  -> index -1
  EXPECT_EQ(key & 0x1FFFFF, kOffset);               // z = 0      -> index 0
}

TEST(VoxelKey, RejectsIndicesOutsideTwentyOneBits) {
  std::uint64_t key = 0;
  EXPECT_TRUE(voxel_key({-1048576.f, 0.f, 1048575.f}, 1.f, key));  // both ends of the valid range
  EXPECT_EQ(key >> 42, 0u);
  EXPECT_EQ(key & 0x1FFFFF, 0x1FFFFFu);
  EXPECT_FALSE(voxel_key({1048576.f, 0.f, 0.f}, 1.f, key));
  EXPECT_FALSE(voxel_key({0.f, -1048577.f, 0.f}, 1.f, key));
}

TEST(CudaFactory, FailsClearlyWhenBuiltWithoutCuda) {
  if (cuda_compiled()) GTEST_SKIP() << "built with CUDA";
  EXPECT_FALSE(cuda_device_present());
  EXPECT_THROW(make_cuda_filter({}, 1024), std::runtime_error);
}

TEST(Synthetic, IsDeterministicAndHasTheAdvertisedDefects) {
  const auto a = make_synthetic_cloud(320, 240);
  const auto b = make_synthetic_cloud(320, 240);
  ASSERT_EQ(a.size(), 320u * 240u);
  ASSERT_EQ(a.size(), b.size());

  std::size_t nan = 0, beyond = 0, same = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const bool a_nan = std::isnan(a[i].x);
    if (a_nan && std::isnan(b[i].x)) ++same;
    if (!a_nan && a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z) ++same;
    if (a_nan) { ++nan; continue; }
    if (std::sqrt(a[i].x * a[i].x + a[i].y * a[i].y + a[i].z * a[i].z) > 10.f) ++beyond;
  }
  EXPECT_EQ(same, a.size()) << "two runs with the same seed must match exactly";
  const double nan_fraction = static_cast<double>(nan) / a.size();
  EXPECT_GT(nan_fraction, 0.04);
  EXPECT_LT(nan_fraction, 0.06);
  EXPECT_GT(beyond, 0u) << "scene must contain returns beyond the default max_range";

  const auto c = make_synthetic_cloud(320, 240, 43);
  std::size_t differs = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!std::isnan(a[i].x) && !std::isnan(c[i].x) && a[i].z != c[i].z) ++differs;
  EXPECT_GT(differs, a.size() / 2) << "a different seed must give different noise";
}

TEST(Synthetic, FilteredCloudIsMuchSmallerThanInput) {
  const auto cloud = make_synthetic_cloud(320, 240);
  std::vector<PointXYZ> out;
  const FilterStats s = make_cpu_filter({})->apply(cloud.data(), cloud.size(), out);
  EXPECT_GT(s.out, 1000u);
  EXPECT_LT(s.out, s.in - s.non_finite - s.out_of_range) << "voxels must merge some points";
  EXPECT_EQ(s.index_overflow, 0u);
  EXPECT_GT(s.non_finite, 0u);
  EXPECT_GT(s.out_of_range, 0u);
}
