// bench_filter: times the CPU and CUDA filters on the synthetic clouds. No ROS.
//
//   bench_filter [--backend cpu|cuda|all] [--res WxH|all] [--iters N] [--warmup N] [--csv FILE]
//                [--stages-csv FILE]
//
// Timings reported per resolution:
//   cpu  / end_to_end : steady_clock around Filter::apply
//   cuda / end_to_end : steady_clock around Filter::apply, i.e. host input to host output,
//                       including both copies and the final stream synchronisation
//   cuda / gpu_only   : cudaEvent time of the device pipeline (FilterStats::device_ms),
//                       excluding the input and output copies
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"

using namespace gpu_pointcloud;

namespace {

struct Options {
  std::string backend = "all";
  std::string res = "all";
  int iters = 200;
  int warmup = 20;
  std::string csv;
  std::string stages_csv;
};

struct Row {
  std::string resolution;
  std::size_t points;
  std::string backend, timing;
  double median_ms, p99_ms, points_per_sec;
  std::size_t out_points;
};

[[noreturn]] void usage(const char* error) {
  std::fprintf(stderr, "error: %s\nusage: bench_filter [--backend cpu|cuda|all] [--res WxH|all] [--iters N] "
                       "[--warmup N] [--csv FILE]\n", error);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) usage(("missing value for " + a).c_str());
    const std::string v = argv[++i];
    if (a == "--backend") o.backend = v;
    else if (a == "--res") o.res = v;
    else if (a == "--iters") o.iters = std::atoi(v.c_str());
    else if (a == "--warmup") o.warmup = std::atoi(v.c_str());
    else if (a == "--csv") o.csv = v;
    else if (a == "--stages-csv") o.stages_csv = v;
    else usage(("unknown option " + a).c_str());
  }
  if (o.backend != "cpu" && o.backend != "cuda" && o.backend != "all") usage("--backend must be cpu, cuda or all");
  if (o.iters < 1 || o.warmup < 0) usage("--iters must be >= 1 and --warmup >= 0");
  return o;
}

std::vector<std::pair<int, int>> resolutions(const std::string& res) {
  if (res == "all") return {{320, 240}, {640, 480}, {1280, 720}};
  int w = 0, h = 0;
  if (std::sscanf(res.c_str(), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) usage("--res must look like 640x480");
  return {{w, h}};
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Nearest-rank 99th percentile: the smallest sample with at least 99% of samples at or below it.
double p99(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t rank = static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(v.size())));
  return v[std::max<std::size_t>(rank, 1) - 1];
}

Row make_row(const std::string& res, std::size_t points, const char* backend, const char* timing,
             const std::vector<double>& ms, std::size_t out_points) {
  const double med = median(ms);
  return {res, points, backend, timing, med, p99(ms), static_cast<double>(points) / (med * 1e-3), out_points};
}

// Runs warmup + timed iterations. `wall_ms` gets one steady_clock sample per timed iteration and
// `device_ms` the matching FilterStats::device_ms.
std::size_t run(Filter& filter, const std::vector<PointXYZ>& cloud, const Options& o, std::vector<double>& wall_ms,
                std::vector<std::vector<double>>& stage_ms,
                std::vector<double>& device_ms) {
  std::vector<PointXYZ> out;
  for (int i = 0; i < o.warmup; ++i) filter.apply(cloud.data(), cloud.size(), out);
  wall_ms.clear();
  device_ms.clear();
  stage_ms.clear();
  for (int i = 0; i < o.iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const FilterStats s = filter.apply(cloud.data(), cloud.size(), out);
    const auto t1 = std::chrono::steady_clock::now();
    wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    device_ms.push_back(s.device_ms);
    const auto stages = filter.last_stage_times();
    stage_ms.resize(stages.size());
    for (std::size_t k = 0; k < stages.size(); ++k) stage_ms[k].push_back(stages[k].ms);
  }
  return out.size();
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const bool want_cpu = o.backend != "cuda";
  bool want_cuda = o.backend != "cpu";
  if (want_cuda && !cuda_device_present()) {
    if (o.backend == "cuda") {
      std::fprintf(stderr, "error: --backend cuda requested but %s\n",
                   cuda_compiled() ? "no CUDA device is visible" : "this build has no CUDA support");
      return 1;
    }
    std::fprintf(stderr, "note: no CUDA device available, benchmarking the CPU backend only\n");
    want_cuda = false;
  }

  const FilterParams params;  // defaults: voxel 0.05 m, range 0.3 to 10 m
  std::vector<Row> rows;
  std::vector<std::string> speedups;
  struct StageRow {
    std::string resolution, stage;
    double median_ms;
  };
  std::vector<StageRow> stage_rows;

  for (const auto& [w, h] : resolutions(o.res)) {
    const std::string res = std::to_string(w) + "x" + std::to_string(h);
    const auto cloud = make_synthetic_cloud(w, h);
    std::vector<double> wall, device;
    std::vector<std::vector<double>> stage;
    double cpu_median = 0.0;

    if (want_cpu) {
      auto filter = make_cpu_filter(params);
      const std::size_t out = run(*filter, cloud, o, wall, stage, device);
      rows.push_back(make_row(res, cloud.size(), "cpu", "end_to_end", wall, out));
      cpu_median = rows.back().median_ms;
    }
    if (want_cuda) {
      auto filter = make_cuda_filter(params, cloud.size());
      const std::size_t out = run(*filter, cloud, o, wall, stage, device);
      rows.push_back(make_row(res, cloud.size(), "cuda", "end_to_end", wall, out));
      const double e2e = rows.back().median_ms;
      rows.push_back(make_row(res, cloud.size(), "cuda", "gpu_only", device, out));
      const double gpu = rows.back().median_ms;
      if (want_cpu) {
        char line[160];
        std::snprintf(line, sizeof(line), "| %s | %.2fx | %.2fx |", res.c_str(), cpu_median / e2e, cpu_median / gpu);
        speedups.push_back(line);
      }
      const auto names = filter->last_stage_times();
      for (std::size_t k = 0; k < names.size() && k < stage.size(); ++k) {
        stage_rows.push_back({res, names[k].name, median(stage[k])});
      }
    }
  }

  std::printf("iterations: %d timed after %d warmup, voxel_size=%.2f m, range %.1f to %.1f m\n\n", o.iters, o.warmup,
              params.voxel_size, params.min_range, params.max_range);
  std::printf("| resolution | points | backend | timing | median ms | p99 ms | input points/s | output points |\n");
  std::printf("|---|---:|---|---|---:|---:|---:|---:|\n");
  for (const Row& r : rows) {
    std::printf("| %s | %zu | %s | %s | %.3f | %.3f | %.3g | %zu |\n", r.resolution.c_str(), r.points,
                r.backend.c_str(), r.timing.c_str(), r.median_ms, r.p99_ms, r.points_per_sec, r.out_points);
  }
  if (!speedups.empty()) {
    std::printf("\nSpeedup = CPU median / CUDA median.\n\n");
    std::printf("| resolution | vs CUDA end_to_end | vs CUDA gpu_only |\n|---|---:|---:|\n");
    for (const auto& s : speedups) std::printf("%s\n", s.c_str());
  }

  if (!o.csv.empty()) {
    std::ofstream csv(o.csv);
    if (!csv) {
      std::fprintf(stderr, "error: cannot write %s\n", o.csv.c_str());
      return 1;
    }
    csv << "resolution,points,backend,timing,median_ms,p99_ms,points_per_sec,out_points\n";
    char line[256];
    for (const Row& r : rows) {
      std::snprintf(line, sizeof(line), "%s,%zu,%s,%s,%.4f,%.4f,%.0f,%zu\n", r.resolution.c_str(), r.points,
                    r.backend.c_str(), r.timing.c_str(), r.median_ms, r.p99_ms, r.points_per_sec, r.out_points);
      csv << line;
    }
  }

  if (!stage_rows.empty()) {
    std::printf("\nCUDA stages, median ms per stage (cudaEvent intervals on the stream).\n\n");
    std::printf("| resolution | stage | median ms |\n|---|---|---:|\n");
    for (const StageRow& r : stage_rows) {
      std::printf("| %s | %s | %.3f |\n", r.resolution.c_str(), r.stage.c_str(), r.median_ms);
    }
  }
  if (!o.stages_csv.empty()) {
    std::ofstream csv(o.stages_csv);
    if (!csv) {
      std::fprintf(stderr, "error: cannot write %s\n", o.stages_csv.c_str());
      return 1;
    }
    csv << "resolution,stage,median_ms\n";
    char line[160];
    for (const StageRow& r : stage_rows) {
      std::snprintf(line, sizeof(line), "%s,%s,%.4f\n", r.resolution.c_str(), r.stage.c_str(), r.median_ms);
      csv << line;
    }
  }
  return 0;
}
