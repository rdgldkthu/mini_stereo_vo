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

set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "" FORCE)
set(SOPHUS_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/Sophus)
get_target_property(_sophus_inc sophus INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(sophus PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_sophus_inc}")

find_package(Ceres REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui features2d video calib3d)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)
FetchContent_Declare(
  rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/download/0.22.1/rerun_cpp_sdk.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(rerun_sdk)

add_library(svo_core
  src/camera.cpp
  src/dataset_kitti.cpp
  src/estimator.cpp
  src/frontend.cpp
  src/map.cpp
  src/motion_model.cpp
  src/pose_writer.cpp
  src/stereo_initializer.cpp
  src/tracker.cpp
  src/rerun_viewer.cpp
)

target_include_directories(svo_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(svo_core
  PUBLIC Eigen3::Eigen Sophus::Sophus Ceres::ceres ${OpenCV_LIBS} rerun_sdk
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

# ---------------------------------------------------------------------------
# Regression test — skips automatically when KITTI data or evo is absent
# ---------------------------------------------------------------------------
enable_testing()

set(REGRESSION_MAX_FRAMES "100" CACHE STRING
    "Number of KITTI frames used by the regression_kitti_05 CTest")
set(REGRESSION_APE_THRESHOLD "2.0" CACHE STRING
    "APE RMSE ceiling in metres for the regression_kitti_05 CTest")

add_executable(test_geometry tests/test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE svo_core)
add_test(NAME unit_geometry COMMAND test_geometry)

add_test(
  NAME regression_kitti_05
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_regression.sh
          "05" "${REGRESSION_MAX_FRAMES}" "${REGRESSION_APE_THRESHOLD}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(regression_kitti_05 PROPERTIES
  SKIP_RETURN_CODE 77
)