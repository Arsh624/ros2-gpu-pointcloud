// bench_cuda_input: compares the two CUDA input paths (raw upload vs host repack) inside one
// process, alternating them iteration by iteration.
//
//   bench_cuda_input [LABEL]
//
// Why a separate tool: on the development machine the host-to-device transfer rate takes one of
// two values per process launch (see docs/DESIGN.md, "Input path"), so two modes measured in two
// different launches cannot be compared. Interleaving them in one process can.
//
// Prints CSV rows: label,resolution,point_step,raw_ms,repack_ms,raw_h2d_ms,repack_h2d_ms
// (medians of 200 timed iterations per mode after 20 warmup iterations per mode).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"

using namespace gpu_pointcloud;

namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Samples {
  std::vector<double> wall_ms, h2d_ms;
};

void step(Filter& filter, const CloudView& view, std::vector<PointXYZ>& out, Samples* samples) {
  const auto t0 = std::chrono::steady_clock::now();
  filter.apply(view, out);
  const auto t1 = std::chrono::steady_clock::now();
  if (samples) {
    samples->wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    samples->h2d_ms.push_back(filter.last_stage_times().front().ms);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* label = argc > 1 ? argv[1] : "run";
  if (!cuda_device_present()) {
    std::fprintf(stderr, "error: no CUDA device available\n");
    return 1;
  }
  const int warmup = 20, iters = 200;
  const std::pair<int, int> resolutions[] = {{320, 240}, {640, 480}, {1280, 720}};
  for (const auto& [w, h] : resolutions) {
    const auto cloud = make_synthetic_cloud(w, h);
    for (std::size_t point_step : {12u, 16u, 32u}) {
      std::vector<std::uint8_t> bytes(cloud.size() * point_step, 0xCD);
      for (std::size_t i = 0; i < cloud.size(); ++i) std::memcpy(&bytes[i * point_step], &cloud[i], sizeof(PointXYZ));
      const CloudView view{bytes.data(), cloud.size(), point_step, 0, 4, 8};

      auto raw = make_cuda_filter({}, cloud.size(), CudaInput::kRaw);
      auto repack = make_cuda_filter({}, cloud.size(), CudaInput::kRepack);
      std::vector<PointXYZ> out;
      for (int i = 0; i < warmup; ++i) {
        step(*raw, view, out, nullptr);
        step(*repack, view, out, nullptr);
      }
      Samples a, b;
      for (int i = 0; i < iters; ++i) {
        step(*raw, view, out, &a);
        step(*repack, view, out, &b);
      }
      std::printf("%s,%dx%d,%zu,%.4f,%.4f,%.4f,%.4f\n", label, w, h, point_step, median(a.wall_ms),
                  median(b.wall_ms), median(a.h2d_ms), median(b.h2d_ms));
    }
  }
  return 0;
}
