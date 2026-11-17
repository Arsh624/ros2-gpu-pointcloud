#include "gpu_pointcloud/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace gpu_pointcloud {
namespace {

// Camera frame: x right, y down, z forward. Rays are (dx, dy, 1), so the ray parameter t is depth.
struct Ray {
  double dx, dy;
};

struct Box {
  double lo[3], hi[3];
};

struct Sphere {
  double c[3], r;
};

constexpr double kFloorY = 1.5;   // camera is 1.5 m above the floor
constexpr double kWallZ = 9.5;    // r > 10 m towards the image edges, so some returns are out of range
constexpr double kNoiseSigma = 0.005;
constexpr double kInvalidFraction = 0.05;

constexpr Box kBoxes[] = {
    {{-1.6, 0.7, 3.0}, {-0.8, 1.5, 3.8}},
    {{0.5, 0.9, 4.5}, {1.5, 1.5, 5.5}},
    {{-0.4, 1.1, 2.0}, {0.2, 1.5, 2.5}},
};
constexpr Sphere kSpheres[] = {
    {{1.2, 1.0, 2.8}, 0.5},
    {{-0.3, 0.2, 6.0}, 0.6},
};

// Slab test. Returns the entry depth, or infinity when the ray misses.
double hit_box(const Ray& ray, const Box& b) {
  const double d[3] = {ray.dx, ray.dy, 1.0};
  double t_near = 0.0, t_far = std::numeric_limits<double>::infinity();
  for (int a = 0; a < 3; ++a) {
    if (d[a] == 0.0) {
      if (0.0 < b.lo[a] || 0.0 > b.hi[a]) return std::numeric_limits<double>::infinity();
      continue;
    }
    double t0 = b.lo[a] / d[a], t1 = b.hi[a] / d[a];
    if (t0 > t1) std::swap(t0, t1);
    t_near = std::max(t_near, t0);
    t_far = std::min(t_far, t1);
  }
  return t_near <= t_far && t_near > 0.0 ? t_near : std::numeric_limits<double>::infinity();
}

// Nearest root of |t*d - c|^2 = r^2, or infinity.
double hit_sphere(const Ray& ray, const Sphere& s) {
  const double a = ray.dx * ray.dx + ray.dy * ray.dy + 1.0;
  const double b = ray.dx * s.c[0] + ray.dy * s.c[1] + s.c[2];
  const double c = s.c[0] * s.c[0] + s.c[1] * s.c[1] + s.c[2] * s.c[2] - s.r * s.r;
  const double disc = b * b - a * c;
  if (disc < 0.0) return std::numeric_limits<double>::infinity();
  const double t = (b - std::sqrt(disc)) / a;
  return t > 0.0 ? t : std::numeric_limits<double>::infinity();
}

double scene_depth(const Ray& ray) {
  double t = kWallZ;  // every ray reaches the wall plane unless something is in front
  if (ray.dy > 0.0) t = std::min(t, kFloorY / ray.dy);
  for (const Box& b : kBoxes) t = std::min(t, hit_box(ray, b));
  for (const Sphere& s : kSpheres) t = std::min(t, hit_sphere(ray, s));
  return t;
}

}  // namespace

std::vector<PointXYZ> make_synthetic_cloud(int width, int height, std::uint32_t seed) {
  std::vector<PointXYZ> cloud;
  if (width <= 0 || height <= 0) return cloud;
  cloud.reserve(static_cast<std::size_t>(width) * height);

  const double f = 0.8 * width;
  const double cx = 0.5 * width, cy = 0.5 * height;
  const float nan = std::numeric_limits<float>::quiet_NaN();

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, kNoiseSigma);

  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      // Both draws happen for every pixel so the random stream does not depend on the scene.
      const bool invalid = uniform(rng) < kInvalidFraction;
      const double dz = noise(rng);
      if (invalid) {
        cloud.push_back({nan, nan, nan});
        continue;
      }
      const Ray ray{(u + 0.5 - cx) / f, (v + 0.5 - cy) / f};
      const double z = scene_depth(ray) + dz;
      cloud.push_back({static_cast<float>(ray.dx * z), static_cast<float>(ray.dy * z), static_cast<float>(z)});
    }
  }
  return cloud;
}

}  // namespace gpu_pointcloud
