cmake_minimum_required(VERSION 3.20)
project(stereo_vo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui features2d video calib3d)

add_library(svo_core
  src/camera.cpp
  src/dataset_kitti.cpp
  src/estimator.cpp
  src/frontend.cpp
  src/map.cpp
  src/stereo_initializer.cpp
  src/tracker.cpp
)

target_include_directories(svo_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(svo_core
  PUBLIC Eigen3::Eigen ${OpenCV_LIBS}
)

target_compile_options(svo_core
  PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(run_kitti
  app/run_kitti.cpp
)

target_link_libraries(run_kitti
  PRIVATE svo_core
)

target_compile_options(run_kitti
  PRIVATE -Wall -Wextra -Wpedantic
)