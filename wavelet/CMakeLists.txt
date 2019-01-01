#
# NOTE: Unlike all other CMakeLists.txt in the apps/ folder, this
# is deliberately intended to be standalone (not included from the toplevel)
# in order to show the minimum scaffolding necessary to use ahead-of-time
# Generators in a simple app.
#
# To use:
# mkdir cmake_build && cd cmake_build && cmake .. && make -j8 && ./bin/wavelet ../../images/gray.png .

project(wavelet)
cmake_minimum_required(VERSION 3.1.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../LLVM-IR-CMAKE/cmake")

message(STATUS ${CMAKE_MODULE_PATH})
include(LLVMIRUtil)


# halide.cmake requires that HALIDE_DISTRIB_DIR be set to point to the Halide distribution folder we use.
# This assumes it's built at the toplevel via 'make distrib' (ironically, the CMake build isn't
# yet capable of producing a distrib folder).
#et(HALIDE_DISTRIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../distrib")
set(HALIDE_DISTRIB_DIR "/packages/halide-lib")
# halide.cmake defaults to using the shared-library version of libHalide;
# we can make it prefer the static-library version by setting this:
# set(HALIDE_DISTRIB_USE_STATIC_LIBRARY TRUE)

include("${HALIDE_DISTRIB_DIR}/halide.cmake")

# Define the wavelet app
add_executable(wavelet "${CMAKE_CURRENT_SOURCE_DIR}/wavelet.cpp")
set_target_properties(wavelet PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(wavelet PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(wavelet PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_use_image_io(wavelet)

add_custom_target(All_bc)
# Define a halide_library() for each generator we have, and link each one into wavelet
file(GLOB GENS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*_generator.cpp")
SET(listVar "")
SET(incVar "")
foreach(GEN_SRC ${GENS})
    string(REPLACE "_generator.cpp" "" GEN_NAME "${GEN_SRC}")
    string(REPLACE "_generator.cpp" ".bc" BC_NAME "${GEN_SRC}")
    _halide_genfiles_dir("${GEN_NAME}" GEN_DIR)
    LIST(APPEND listVar "${GEN_DIR}/${BC_NAME}")
    halide_library("${GEN_NAME}" SRCS ${GEN_SRC})
    target_link_libraries(wavelet PUBLIC "${GEN_NAME}")
    LIST(APPEND incVar "${GEN_DIR}")
endforeach()

set_target_properties(wavelet PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${incVar}")
llvmir_attach_bc_target(wavelet_bc wavelet)
add_dependencies(wavelet_bc wavelet)
get_property(wavelet_bc_dir TARGET wavelet_bc PROPERTY LLVMIR_DIR)
get_property(wavelet_bc_file TARGET wavelet_bc PROPERTY LLVMIR_FILES)
LIST(APPEND listVar "${wavelet_bc_dir}/${wavelet_bc_file}")


message("${listVar}")
message("${RUNTIME_NAME}")
#message("${CMAKE_CURRENT_BINARY_DIR}/${LLVMIR_DIR}/wavelet_bc/wavelet.bc")

set_target_properties(All_bc PROPERTIES DEPENDS "${listVar}")
# this property is required by our parasitic targets
set_target_properties(All_bc PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(All_bc PROPERTIES LLVMIR_TYPE LLVMIR_BINARY)
set_target_properties(All_bc PROPERTIES LLVMIR_DIR "")
set_target_properties(All_bc PROPERTIES LLVMIR_FILES "${listVar}")

llvmir_attach_link_target(
  TARGET All
DEPENDS All_bc)
