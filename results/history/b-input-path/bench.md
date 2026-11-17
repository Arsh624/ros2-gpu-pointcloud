iterations: 200 timed after 20 warmup, voxel_size=0.05 m, range 0.3 to 10.0 m, cpu_omp threads: 20, point step: 12 bytes, cuda input: auto

| resolution | points | backend | timing | median ms | p99 ms | input points/s | output points |
|---|---:|---|---|---:|---:|---:|---:|
| 320x240 | 76800 | cpu | end_to_end | 1.900 | 2.312 | 4.04e+07 | 20785 |
| 320x240 | 76800 | cpu_omp | end_to_end | 0.247 | 0.392 | 3.11e+08 | 20785 |
| 320x240 | 76800 | pcl | end_to_end | 1.552 | 1.777 | 4.95e+07 | 20785 |
| 320x240 | 76800 | cuda | end_to_end | 0.430 | 0.556 | 1.79e+08 | 20785 |
| 320x240 | 76800 | cuda | gpu_only | 0.244 | 0.336 | 3.15e+08 | 20785 |
| 640x480 | 307200 | cpu | end_to_end | 7.964 | 8.780 | 3.86e+07 | 34654 |
| 640x480 | 307200 | cpu_omp | end_to_end | 0.876 | 1.283 | 3.51e+08 | 34654 |
| 640x480 | 307200 | pcl | end_to_end | 4.987 | 5.581 | 6.16e+07 | 34654 |
| 640x480 | 307200 | cuda | end_to_end | 0.792 | 1.031 | 3.88e+08 | 34654 |
| 640x480 | 307200 | cuda | gpu_only | 0.335 | 0.512 | 9.18e+08 | 34654 |
| 1280x720 | 921600 | cpu | end_to_end | 24.909 | 27.972 | 3.7e+07 | 39514 |
| 1280x720 | 921600 | cpu_omp | end_to_end | 2.774 | 3.391 | 3.32e+08 | 39514 |
| 1280x720 | 921600 | pcl | end_to_end | 13.869 | 15.163 | 6.65e+07 | 39514 |
| 1280x720 | 921600 | cuda | end_to_end | 1.702 | 2.024 | 5.42e+08 | 39514 |
| 1280x720 | 921600 | cuda | gpu_only | 0.617 | 0.762 | 1.49e+09 | 39514 |

Speedup = CPU median / CUDA median. "Best CPU" is the fastest CPU baseline measured in this run at that resolution.

| resolution | best CPU | best CPU ms | CUDA end_to_end vs best CPU | CUDA gpu_only vs best CPU | CUDA end_to_end vs single-threaded cpu | CUDA gpu_only vs single-threaded cpu |
|---|---|---:|---:|---:|---:|---:|
| 320x240 | cpu_omp | 0.247 | 0.57x | 1.01x | 4.42x | 7.80x |
| 640x480 | cpu_omp | 0.876 | 1.11x | 2.62x | 10.05x | 23.81x |
| 1280x720 | cpu_omp | 2.774 | 1.63x | 4.49x | 14.64x | 40.35x |

CUDA stages, median ms per stage (cudaEvent intervals on the stream).

| resolution | stage | median ms |
|---|---|---:|
| 320x240 | h2d_copy | 0.102 |
| 320x240 | classify | 0.043 |
| 320x240 | select | 0.010 |
| 320x240 | host_sync+gather_keys | 0.025 |
| 320x240 | radix_sort | 0.088 |
| 320x240 | reduce_by_key | 0.032 |
| 320x240 | host_sync+centroid | 0.029 |
| 320x240 | d2h_copy | 0.044 |
| 640x480 | h2d_copy | 0.346 |
| 640x480 | classify | 0.089 |
| 640x480 | select | 0.012 |
| 640x480 | host_sync+gather_keys | 0.030 |
| 640x480 | radix_sort | 0.091 |
| 640x480 | reduce_by_key | 0.075 |
| 640x480 | host_sync+centroid | 0.031 |
| 640x480 | d2h_copy | 0.058 |
| 1280x720 | h2d_copy | 0.987 |
| 1280x720 | classify | 0.199 |
| 1280x720 | select | 0.015 |
| 1280x720 | host_sync+gather_keys | 0.034 |
| 1280x720 | radix_sort | 0.141 |
| 1280x720 | reduce_by_key | 0.148 |
| 1280x720 | host_sync+centroid | 0.031 |
| 1280x720 | d2h_copy | 0.058 |
