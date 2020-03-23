project(HalidePlayGround)
cmake_minimum_required(VERSION 3.1.3)

# Requires to build with Clang
### Set Toolchain (This should get fixed, there should be a better way of doing this)
#set(CMAKE_C_COMPILER    "/usr/bin/clang")
#set(CMAKE_CXX_COMPILER  "/usr/bin/clang++")
#set(CMAKE_AR            "/usr/bin/llvm-ar")
#set(CMAKE_LINKER        "/usr/bin/llvm-ld")
#set(CMAKE_NM            "/usr/bin/llvm-nm")
#set(CMAKE_OBJDUMP       "/usr/bin/llvm-objdump")
#set(CMAKE_RANLIB        "/usr/bin/llvm-ranlib")

find_package(Threads REQUIRED THREADS_PREFER_PTHREAD_FLAG)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/LLVM-IR-CMAKE/cmake")
message(STATUS ${CMAKE_MODULE_PATH})
include(LLVMIRUtil)


# halide.cmake requires that HALIDE_DISTRIB_DIR be set to point to the Halide distribution folder we use.
# This assumes it's built at the toplevel via 'make distrib' (ironically, the CMake build isn't
# yet capable of producing a distrib folder).
# set(HALIDE_DISTRIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../distrib" CACHE PATH "Path to distrib folder.")
if(NOT DEFINED HALIDE_DISTRIB_DIR)
    message(FATAL_ERROR "The variable HALIDE_DISTRIB_DIR should pass to the cmake.")
else()
    message(STATUS "HALIDE_DISTRIB_DIR : ${HALIDE_DISTRIB_DIR}")
endif(NOT DEFINED HALIDE_DISTRIB_DIR)
# halide.cmake defaults to using the shared-library version of libHalide;
# we can make it prefer the static-library version by setting this:
# set(HALIDE_DISTRIB_USE_STATIC_LIBRARY TRUE)

include("${HALIDE_DISTRIB_DIR}/halide.cmake")

add_subdirectory(camera_pipe)
add_subdirectory(gaussian)
add_subdirectory(harris)
add_subdirectory(lens_blur)
add_subdirectory(stereo)
add_subdirectory(wavelet)
add_subdirectory(2d_blur)
add_subdirectory(2dblur_sched)

