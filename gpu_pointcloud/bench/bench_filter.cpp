// bench_filter: times the filter backends on the synthetic clouds. No ROS.
//
//   bench_filter [--backend LIST|all] [--res WxH|all] [--iters N] [--warmup N] [--threads N]
//                [--point-step BYTES] [--cuda-input auto|raw|repack] [--csv FILE] [--stages-csv FILE]
//
// LIST is a comma-separated subset of: cpu, cpu_omp, pcl, cuda.
// --point-step lays the cloud out like a PointCloud2 with that many bytes per point (x, y, z
// first, the rest filler) and filters it in place; 12 (the default) is packed xyz.
//
// Backends and what their timing covers:
//   cpu     / end_to_end : single-threaded reference; steady_clock around Filter::apply
//   cpu_omp / end_to_end : OpenMP backend with --threads threads (default: all); same clock
//   pcl     / end_to_end : radial range loop + pcl::VoxelGrid on a ready pcl::PointCloud; the
//                          conversions into and out of PCL types are NOT timed (WITH_PCL builds)
//   cuda    / end_to_end : host input to host output, including both copies and the final
//                          stream synchronisation; steady_clock around Filter::apply
//   cuda    / gpu_only   : cudaEvent time of the device pipeline (FilterStats::device_ms),
//                          excluding the input and output copies
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_pointcloud/filter.hpp"
#include "gpu_pointcloud/synthetic.hpp"
#ifdef GPU_POINTCLOUD_WITH_PCL
#include "pcl_baseline.hpp"
#endif

using namespace gpu_pointcloud;

namespace {

struct Options {
  std::vector<std::string> backends;
  std::string res = "all";
  int iters = 200;
  int warmup = 20;
  int threads = 0;
  int point_step = 12;
  CudaInput cuda_input = CudaInput::kAuto;
  std::string cuda_input_name = "auto";
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

struct StageRow {
  std::string resolution, stage;
  double median_ms;
};

// One backend prepared for one cloud. `step` is the timed work.
struct Runner {
  std::function<FilterStats()> step;
  std::function<std::vector<StageTime>()> stages = [] { return std::vector<StageTime>{}; };
  bool has_device_time = false;
};

const char* const kUsage =
    "usage: bench_filter [--backend LIST|all] [--res WxH|all] [--iters N] [--warmup N] [--threads N]\n"
    "                    [--point-step BYTES] [--cuda-input auto|raw|repack] [--csv FILE] [--stages-csv FILE]\n"
    "       LIST is a comma-separated subset of: cpu, cpu_omp, pcl, cuda\n";

[[noreturn]] void usage(const std::string& error) {
  std::fprintf(stderr, "error: %s\n%s", error.c_str(), kUsage);
  std::exit(2);
}

bool pcl_compiled() {
#ifdef GPU_POINTCLOUD_WITH_PCL
  return true;
#else
  return false;
#endif
}

Options parse(int argc, char** argv) {
  Options o;
  std::string backend = "all";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) usage("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--backend") backend = v;
    else if (a == "--res") o.res = v;
    else if (a == "--iters") o.iters = std::atoi(v.c_str());
    else if (a == "--warmup") o.warmup = std::atoi(v.c_str());
    else if (a == "--threads") o.threads = std::atoi(v.c_str());
    else if (a == "--point-step") o.point_step = std::atoi(v.c_str());
    else if (a == "--cuda-input") {
      o.cuda_input_name = v;
      if (v == "auto") o.cuda_input = CudaInput::kAuto;
      else if (v == "raw") o.cuda_input = CudaInput::kRaw;
      else if (v == "repack") o.cuda_input = CudaInput::kRepack;
      else usage("--cuda-input must be auto, raw or repack");
    }
    else if (a == "--csv") o.csv = v;
    else if (a == "--stages-csv") o.stages_csv = v;
    else usage("unknown option " + a);
  }
  if (o.iters < 1 || o.warmup < 0 || o.threads < 0) usage("--iters must be >= 1, --warmup and --threads >= 0");
  if (o.point_step < 12) usage("--point-step must be at least 12");

  if (backend == "all") {
    // Everything this build and machine can run; unavailable backends are skipped with a note.
    o.backends = {"cpu", "cpu_omp"};
    if (pcl_compiled()) o.backends.push_back("pcl");
    else std::fprintf(stderr, "note: built without PCL (WITH_PCL=OFF), skipping the pcl baseline\n");
    if (cuda_device_present()) o.backends.push_back("cuda");
    else std::fprintf(stderr, "note: no CUDA device available, skipping the cuda backend\n");
  } else {
    std::stringstream list(backend);
    for (std::string name; std::getline(list, name, ',');) {
      if (name != "cpu" && name != "cpu_omp" && name != "pcl" && name != "cuda") usage("unknown backend " + name);
      if (name == "pcl" && !pcl_compiled()) usage("this build has no PCL support (configure with -DWITH_PCL=ON)");
      if (name == "cuda" && !cuda_device_present()) {
        usage(cuda_compiled() ? "no CUDA device is visible" : "this build has no CUDA support");
      }
      o.backends.push_back(name);
    }
    if (o.backends.empty()) usage("--backend list is empty");
  }
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

Row make_row(const std::string& res, std::size_t points, const std::string& backend, const char* timing,
             const std::vector<double>& ms, std::size_t out_points) {
  const double med = median(ms);
  return {res, points, backend, timing, med, p99(ms), static_cast<double>(points) / (med * 1e-3), out_points};
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const FilterParams params;  // defaults: voxel 0.05 m, range 0.3 to 10 m
  const int threads = o.threads > 0 ? o.threads : default_cpu_threads();

  std::vector<Row> rows;
  std::vector<StageRow> stage_rows;
  std::vector<std::string> speedups;

  for (const auto& [w, h] : resolutions(o.res)) {
    const std::string res = std::to_string(w) + "x" + std::to_string(h);
    const auto cloud = make_synthetic_cloud(w, h);
    // The filters read this view in place. With --point-step 12 it is the cloud itself.
    std::vector<std::uint8_t> strided;
    CloudView view = as_view(cloud.data(), cloud.size());
    if (o.point_step != 12) {
      strided.assign(cloud.size() * o.point_step, 0xCD);
      for (std::size_t i = 0; i < cloud.size(); ++i) std::memcpy(&strided[i * o.point_step], &cloud[i], sizeof(PointXYZ));
      view = {strided.data(), cloud.size(), static_cast<std::size_t>(o.point_step), 0, 4, 8};
    }
    double cpu_median = 0.0, best_cpu_median = 0.0, cuda_e2e = 0.0, cuda_gpu = 0.0;
    std::string best_cpu;

    for (const std::string& backend : o.backends) {
      // Everything a runner needs stays alive in these until the backend is done.
      std::unique_ptr<Filter> filter;
      std::vector<PointXYZ> out;
      Runner runner;
#ifdef GPU_POINTCLOUD_WITH_PCL
      std::unique_ptr<bench::PclBaseline> pcl;
      if (backend == "pcl") {
        pcl = std::make_unique<bench::PclBaseline>(params);
        pcl->set_input(cloud.data(), cloud.size());
        runner.step = [&] { return pcl->run(); };
      }
#endif
      if (backend != "pcl") {
        if (backend == "cpu") filter = make_cpu_filter(params);
        else if (backend == "cpu_omp") filter = make_cpu_omp_filter(params, threads);
        else filter = make_cuda_filter(params, cloud.size(), o.cuda_input);
        runner.step = [&] { return filter->apply(view, out); };
        runner.stages = [&] { return filter->last_stage_times(); };
        runner.has_device_time = backend == "cuda";
      }

      for (int i = 0; i < o.warmup; ++i) runner.step();
      std::vector<double> wall, device;
      std::vector<std::vector<double>> stage;
      std::size_t out_points = 0;
      for (int i = 0; i < o.iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const FilterStats s = runner.step();
        const auto t1 = std::chrono::steady_clock::now();
        wall.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        device.push_back(s.device_ms);
        out_points = s.out;
        const auto stages = runner.stages();
        stage.resize(stages.size());
        for (std::size_t k = 0; k < stages.size(); ++k) stage[k].push_back(stages[k].ms);
      }

      rows.push_back(make_row(res, cloud.size(), backend, "end_to_end", wall, out_points));
      const double med = rows.back().median_ms;
      if (backend == "cuda") {
        cuda_e2e = med;
        rows.push_back(make_row(res, cloud.size(), backend, "gpu_only", device, out_points));
        cuda_gpu = rows.back().median_ms;
        const auto names = runner.stages();
        for (std::size_t k = 0; k < names.size() && k < stage.size(); ++k) {
          stage_rows.push_back({res, names[k].name, median(stage[k])});
        }
      } else {
        if (backend == "cpu") cpu_median = med;
        if (best_cpu.empty() || med < best_cpu_median) {
          best_cpu = backend;
          best_cpu_median = med;
        }
      }
    }

    if (cuda_e2e > 0.0 && !best_cpu.empty()) {
      char line[256];
      char single[2][32] = {"n/a", "n/a"};
      if (cpu_median > 0.0) {
        std::snprintf(single[0], sizeof(single[0]), "%.2fx", cpu_median / cuda_e2e);
        std::snprintf(single[1], sizeof(single[1]), "%.2fx", cpu_median / cuda_gpu);
      }
      std::snprintf(line, sizeof(line), "| %s | %s | %.3f | %.2fx | %.2fx | %s | %s |", res.c_str(), best_cpu.c_str(),
                    best_cpu_median, best_cpu_median / cuda_e2e, best_cpu_median / cuda_gpu, single[0], single[1]);
      speedups.push_back(line);
    }
  }

  std::printf("iterations: %d timed after %d warmup, voxel_size=%.2f m, range %.1f to %.1f m, cpu_omp threads: %d, "
              "point step: %d bytes, cuda input: %s\n\n",
              o.iters, o.warmup, params.voxel_size, params.min_range, params.max_range, threads, o.point_step,
              o.cuda_input_name.c_str());
  std::printf("| resolution | points | backend | timing | median ms | p99 ms | input points/s | output points |\n");
  std::printf("|---|---:|---|---|---:|---:|---:|---:|\n");
  for (const Row& r : rows) {
    std::printf("| %s | %zu | %s | %s | %.3f | %.3f | %.3g | %zu |\n", r.resolution.c_str(), r.points,
                r.backend.c_str(), r.timing.c_str(), r.median_ms, r.p99_ms, r.points_per_sec, r.out_points);
  }
  if (!speedups.empty()) {
    std::printf("\nSpeedup = CPU median / CUDA median. \"Best CPU\" is the fastest CPU baseline measured in this "
                "run at that resolution.\n\n");
    std::printf("| resolution | best CPU | best CPU ms | CUDA end_to_end vs best CPU | CUDA gpu_only vs best CPU "
                "| CUDA end_to_end vs single-threaded cpu | CUDA gpu_only vs single-threaded cpu |\n");
    std::printf("|---|---|---:|---:|---:|---:|---:|\n");
    for (const auto& s : speedups) std::printf("%s\n", s.c_str());
  }
  if (!stage_rows.empty()) {
    std::printf("\nCUDA stages, median ms per stage (cudaEvent intervals on the stream).\n\n");
    std::printf("| resolution | stage | median ms |\n|---|---|---:|\n");
    for (const StageRow& r : stage_rows) {
      std::printf("| %s | %s | %.3f |\n", r.resolution.c_str(), r.stage.c_str(), r.median_ms);
    }
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
