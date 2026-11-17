// bench_idle_gap: how long does one filter call take when the backend has been idle for a while?
//
//   bench_idle_gap [--csv FILE] [--recovery-csv FILE] [--backend cpu_omp|cuda] [--keepalive-ms N]
//
// A benchmark loop keeps the GPU (or the CPU thread pool) busy without pause. A robot does not:
// at 10 Hz the filter runs for a millisecond or two and then nothing happens for 98 ms. This
// tool measures both situations.
//
// Part 1, per backend, resolution and gap: sleep for `gap` before every call, time the call.
// Part 2, recovery: idle for 3 s, then run 60 calls back to back and record each one, to show
// how many calls it takes to get back to full speed.
//
// --keepalive-ms N runs a second, tiny CUDA filter call (64 points) every N ms on a background
// thread for the whole run, to test whether a trickle of GPU work keeps the device clocked up.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"

using namespace gpu_pointcloud;
using Clock = std::chrono::steady_clock;

namespace {

double call_ms(Filter& filter, const std::vector<PointXYZ>& cloud, std::vector<PointXYZ>& out) {
  const auto t0 = Clock::now();
  filter.apply(cloud.data(), cloud.size(), out);
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv_path, recovery_path, only_backend;
  int keepalive_ms = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--csv")) csv_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--recovery-csv")) recovery_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--backend")) only_backend = argv[i + 1];
    else if (!std::strcmp(argv[i], "--keepalive-ms")) keepalive_ms = std::atoi(argv[i + 1]);
  }
  std::FILE* csv = csv_path.empty() ? stdout : std::fopen(csv_path.c_str(), "w");
  std::FILE* recovery = recovery_path.empty() ? stdout : std::fopen(recovery_path.c_str(), "w");
  if (!csv || !recovery) {
    std::fprintf(stderr, "error: cannot open an output file\n");
    return 1;
  }

  std::vector<std::string> backends;
  if (only_backend.empty() || only_backend == "cpu_omp") backends.push_back("cpu_omp");
  if ((only_backend.empty() || only_backend == "cuda") && cuda_device_present()) backends.push_back("cuda");

  std::atomic<bool> stop{false};
  std::thread keepalive;
  if (keepalive_ms > 0 && cuda_device_present()) {
    keepalive = std::thread([&] {
      const std::vector<PointXYZ> tiny(64, PointXYZ{1.f, 1.f, 1.f});
      auto filter = make_cuda_filter({}, tiny.size());
      std::vector<PointXYZ> out;
      while (!stop) {
        filter->apply(tiny.data(), tiny.size(), out);
        std::this_thread::sleep_for(std::chrono::milliseconds(keepalive_ms));
      }
    });
  }
  const std::pair<int, int> resolutions[] = {{640, 480}, {1280, 720}};
  // Gap before each call, and how many calls to time at that gap (fewer for long gaps).
  const std::pair<int, int> gaps[] = {{0, 100}, {10, 50}, {33, 40}, {100, 30}, {300, 15}, {1000, 8}, {3000, 5}};

  std::fprintf(csv, "keepalive_ms,backend,resolution,gap_ms,calls,median_ms,min_ms,max_ms\n");
  std::fprintf(recovery, "keepalive_ms,backend,resolution,call_after_3s_idle,ms\n");
  for (const std::string& backend : backends) {
    for (const auto& [w, h] : resolutions) {
      const std::string res = std::to_string(w) + "x" + std::to_string(h);
      const auto cloud = make_synthetic_cloud(w, h);
      std::unique_ptr<Filter> filter =
          backend == "cuda" ? make_cuda_filter({}, cloud.size()) : make_cpu_omp_filter({});
      std::vector<PointXYZ> out;
      for (int i = 0; i < 200; ++i) call_ms(*filter, cloud, out);  // well past any start-up effect

      for (const auto& [gap_ms, calls] : gaps) {
        std::vector<double> ms;
        for (int i = 0; i < calls; ++i) {
          if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
          ms.push_back(call_ms(*filter, cloud, out));
        }
        std::fprintf(csv, "%d,%s,%s,%d,%d,%.4f,%.4f,%.4f\n", keepalive_ms, backend.c_str(), res.c_str(), gap_ms, calls, median(ms),
                     *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end()));
        std::fflush(csv);
      }

      std::this_thread::sleep_for(std::chrono::seconds(3));
      for (int i = 0; i < 60; ++i) {
        std::fprintf(recovery, "%d,%s,%s,%d,%.4f\n", keepalive_ms, backend.c_str(), res.c_str(), i,
                     call_ms(*filter, cloud, out));
      }
      std::fflush(recovery);
    }
  }
  stop = true;
  if (keepalive.joinable()) keepalive.join();
  return 0;
}
