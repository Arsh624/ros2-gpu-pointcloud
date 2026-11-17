cmake_minimum_required(VERSION 3.22)
project(gpu_pointcloud LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CORE_ONLY "Build only the ROS-free core library" OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(_default_with_cuda ON)
else()
  set(_default_with_cuda OFF)
endif()
option(WITH_CUDA "Build the CUDA backend" ${_default_with_cuda})

if(WITH_CUDA)
  # Must be set before enable_language(CUDA).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
endif()

set(CORE_SOURCES
  src/filter_cpu.cpp
  src/filter_cpu_omp.cpp
  src/synthetic.cpp
  # --- core sources are appended here in later milestones ---
)
if(WITH_CUDA)
  list(APPEND CORE_SOURCES src/filter_cuda.cu)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(gpu_pointcloud_core STATIC ${CORE_SOURCES})
set_target_properties(gpu_pointcloud_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gpu_pointcloud_core PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(gpu_pointcloud_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(gpu_pointcloud_core PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-O3;-ffp-contract=off>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
if(WITH_CUDA)
  target_compile_definitions(gpu_pointcloud_core PUBLIC GPU_POINTCLOUD_WITH_CUDA)
  target_link_libraries(gpu_pointcloud_core PUBLIC CUDA::cudart)
endif()

add_executable(bench_filter bench/bench_filter.cpp)
target_compile_options(bench_filter PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(bench_filter gpu_pointcloud_core)

add_executable(bench_idle_gap bench/bench_idle_gap.cpp)
target_compile_options(bench_idle_gap PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(bench_idle_gap gpu_pointcloud_core)

if(WITH_CUDA)
  add_executable(bench_cuda_input bench/bench_cuda_input.cpp)
  target_compile_options(bench_cuda_input PRIVATE -Wall -Wextra -Wpedantic -O3)
  target_link_libraries(bench_cuda_input gpu_pointcloud_core)
  add_executable(bench_h2d bench/bench_h2d.cu)
  target_link_libraries(bench_h2d CUDA::cudart)
endif()

# Optional PCL reference pipeline for the benchmark. OFF by default so the package has no PCL
# dependency; the dev image has PCL installed for it.
option(WITH_PCL "Add a pcl::VoxelGrid baseline to bench_filter" OFF)
if(WITH_PCL)
  find_package(PCL REQUIRED COMPONENTS common filters)
  add_library(gpu_pointcloud_pcl_baseline STATIC bench/pcl_baseline.cpp)
  target_compile_options(gpu_pointcloud_pcl_baseline PRIVATE -O3)
  target_include_directories(gpu_pointcloud_pcl_baseline SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
  target_link_libraries(gpu_pointcloud_pcl_baseline PUBLIC gpu_pointcloud_core ${PCL_LIBRARIES})
  target_compile_definitions(bench_filter PRIVATE GPU_POINTCLOUD_WITH_PCL)
  target_link_libraries(bench_filter gpu_pointcloud_pcl_baseline)
endif()

message(STATUS "gpu_pointcloud: WITH_CUDA=${WITH_CUDA} WITH_PCL=${WITH_PCL} CORE_ONLY=${CORE_ONLY}")

if(NOT CORE_ONLY)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)

  add_library(gpu_pointcloud_conv STATIC src/cloud_conv.cpp)
  set_target_properties(gpu_pointcloud_conv PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(gpu_pointcloud_conv PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(gpu_pointcloud_conv PUBLIC gpu_pointcloud_core)
  ament_target_dependencies(gpu_pointcloud_conv PUBLIC sensor_msgs std_msgs)

  foreach(node pointcloud_filter_node cloud_publisher_node)
    add_executable(${node})
    target_compile_options(${node} PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(${node} gpu_pointcloud_conv)
    ament_target_dependencies(${node} rclcpp sensor_msgs)
  endforeach()
  target_sources(pointcloud_filter_node PRIVATE src/filter_node.cpp)
  target_sources(cloud_publisher_node PRIVATE src/cloud_publisher_node.cpp)
  if(WITH_CUDA)
    install(TARGETS bench_cuda_input RUNTIME DESTINATION lib/${PROJECT_NAME})
  endif()
  install(TARGETS pointcloud_filter_node cloud_publisher_node bench_filter bench_idle_gap RUNTIME DESTINATION lib/${PROJECT_NAME})

  install(DIRECTORY include/ DESTINATION include)
  install(TARGETS gpu_pointcloud_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_filter_cpu test/test_filter_cpu.cpp)
    target_link_libraries(test_filter_cpu gpu_pointcloud_core)
    if(WITH_PCL)
      ament_add_gtest(test_pcl_baseline test/test_pcl_baseline.cpp)
      target_link_libraries(test_pcl_baseline gpu_pointcloud_pcl_baseline)
    endif()
    if(WITH_CUDA)
      # Skips itself at run time when no CUDA device is visible.
      ament_add_gtest(test_parity_cuda test/test_parity_cuda.cpp)
      target_link_libraries(test_parity_cuda gpu_pointcloud_core)
    endif()
    ament_add_gtest(test_latency_stats test/test_latency_stats.cpp)
    target_include_directories(test_latency_stats PRIVATE include)
    ament_add_gtest(test_cloud_conv test/test_cloud_conv.cpp)
    target_link_libraries(test_cloud_conv gpu_pointcloud_conv)
  endif()

  ament_package()
endif()
