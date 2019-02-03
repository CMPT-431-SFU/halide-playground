#
# NOTE: Unlike all other CMakeLists.txt in the apps/ folder, this
# is deliberately intended to be standalone (not included from the toplevel)
# in order to show the minimum scaffolding necessary to use ahead-of-time
# Generators in a simple app.
#
# To use:
# mkdir cmake_build && cd cmake_build && cmake .. && make -j8 && ./bin/wavelet ../../images/gray.png .

project(gaussian_pipe)
cmake_minimum_required(VERSION 3.1.3)

# Define the camera pipe app
add_executable(gaussian_pipe_process "${CMAKE_CURRENT_SOURCE_DIR}/process.cpp")
set_target_properties(gaussian_pipe_process PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(gaussian_pipe_process PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(gaussian_pipe_process PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_use_image_io(gaussian_pipe_process)

add_custom_target(bc_files_gaussian_linked)
# Define a halide_library() for each generator we have, and link each one into gaussian_pipe
file(GLOB GENS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*_generator.cpp")
SET(listVar "")
SET(incVar "")
halide_generator(gaussian_pipe.generator SRCS gaussian_pipe_generator.cpp)
SET(GEN_SRC gaussian_pipe_generator.cpp)

string(REPLACE "_generator.cpp" ".bc" BC_NAME "${GEN_SRC}")
set(LIB gaussian_pipe)
# Create the generator library
halide_library_from_generator(${LIB}
                                  GENERATOR gaussian_pipe.generator)

string(REPLACE "_generator.cpp" "" GEN_NAME "${GEN_SRC}")
_halide_genfiles_dir("${GEN_NAME}" GEN_DIR)
   
LIST(APPEND listVar "${GEN_DIR}/${BC_NAME}")
LIST(APPEND incVar  "${GEN_DIR}")
target_link_libraries(gaussian_pipe_process PRIVATE ${LIB} Threads::Threads)


set_target_properties(gaussian_pipe_process PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${incVar}")
llvmir_attach_bc_target(gaussian_pipe_process_bc gaussian_pipe_process)
add_dependencies(gaussian_pipe_process_bc gaussian_pipe_process)
get_property(gaussian_pipe_process_bc_dir TARGET gaussian_pipe_process_bc PROPERTY LLVMIR_DIR)
get_property(gaussian_pipe_process_bc_file TARGET gaussian_pipe_process_bc PROPERTY LLVMIR_FILES)
LIST(APPEND listVar "${gaussian_pipe_process_bc_dir}/${gaussian_pipe_process_bc_file}")



set_target_properties(bc_files_gaussian_linked PROPERTIES DEPENDS "${listVar}")
# this property is required by our parasitic targets
set_target_properties(bc_files_gaussian_linked PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(bc_files_gaussian_linked PROPERTIES LLVMIR_TYPE LLVMIR_BINARY)
set_target_properties(bc_files_gaussian_linked PROPERTIES LLVMIR_DIR "")
set_target_properties(bc_files_gaussian_linked PROPERTIES LLVMIR_FILES "${listVar}")

llvmir_attach_link_target(
  TARGET gaussian_pipe_exe
DEPENDS bc_files_gaussian_linked)


