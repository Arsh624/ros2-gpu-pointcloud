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
  src/synthetic.cpp
  # --- core sources are appended here in later milestones ---
)
if(WITH_CUDA)
  list(APPEND CORE_SOURCES src/filter_cuda.cu)
endif()

add_library(gpu_pointcloud_core STATIC ${CORE_SOURCES})
set_target_properties(gpu_pointcloud_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
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

message(STATUS "gpu_pointcloud: WITH_CUDA=${WITH_CUDA} CORE_ONLY=${CORE_ONLY}")

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


  install(DIRECTORY include/ DESTINATION include)
  install(TARGETS gpu_pointcloud_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_filter_cpu test/test_filter_cpu.cpp)
    target_link_libraries(test_filter_cpu gpu_pointcloud_core)
    ament_add_gtest(test_cloud_conv test/test_cloud_conv.cpp)
    target_link_libraries(test_cloud_conv gpu_pointcloud_conv)
  endif()

  ament_package()
endif()
