#
# NOTE: Unlike all other CMakeLists.txt in the apps/ folder, this
# is deliberately intended to be standalone (not included from the toplevel)
# in order to show the minimum scaffolding necessary to use ahead-of-time
# Generators in a simple app.
#
# To use:
# mkdir cmake_build && cd cmake_build && cmake .. && make -j8 && ./bin/wavelet ../../images/gray.png .

project(2dblur-sched)
cmake_minimum_required(VERSION 3.1.3)

# Define the camera pipe app
add_executable(2dblur_sched_process "${CMAKE_CURRENT_SOURCE_DIR}/process.cpp")
#include("${HALIDE_DISTRIB_DIR}/halide.cmake")
set_target_properties(2dblur_sched_process PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(2dblur_sched_process PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(2dblur_sched_process PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")

halide_use_image_io(2dblur_sched_process)
target_link_directories(2dblur_sched_process PUBLIC ${HALIDE_DISTRIB_DIR}/bin)
target_link_libraries(2dblur_sched_process PRIVATE Halide)
# Define a halide_library() for each generator we have, and link each one into 2dblur


