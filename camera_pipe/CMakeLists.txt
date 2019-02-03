#
# NOTE: Unlike all other CMakeLists.txt in the apps/ folder, this
# is deliberately intended to be standalone (not included from the toplevel)
# in order to show the minimum scaffolding necessary to use ahead-of-time
# Generators in a simple app.
#
# To use:
# mkdir cmake_build && cd cmake_build && cmake .. && make -j8 && ./bin/wavelet ../../images/gray.png .

project(camera_pipe)
cmake_minimum_required(VERSION 3.1.3)

# Define the camera pipe app
add_executable(camera_pipe_process "${CMAKE_CURRENT_SOURCE_DIR}/process.cpp")
set_target_properties(camera_pipe_process PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(camera_pipe_process PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(camera_pipe_process PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_use_image_io(camera_pipe_process)

add_custom_target(bc_files_camera_pipe_linked)
# Define a halide_library() for each generator we have, and link each one into camera_pipe
file(GLOB GENS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*_generator.cpp")
SET(listVar "")
SET(incVar "")
halide_generator(camera_pipe.generator SRCS camera_pipe_generator.cpp)
SET(GEN_SRC camera_pipe_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    # Determine auto schedule and determine bc file name
    if(${AUTO_SCHEDULE})
        set(LIB camera_pipe_auto_schedule)
        string(REPLACE "_generator.cpp" "_auto_schedule.bc" BC_NAME "${GEN_SRC}")
    else()
        string(REPLACE "_generator.cpp" ".bc" BC_NAME "${GEN_SRC}")
        set(LIB camera_pipe)
    endif()
   # Create the generator library
    halide_library_from_generator(${LIB}
                                  GENERATOR camera_pipe.generator
                                  GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})

    string(REPLACE "_generator.cpp" "" GEN_NAME "${GEN_SRC}")
    _halide_genfiles_dir("${GEN_NAME}" GEN_DIR)
   
    if(${AUTO_SCHEDULE})
        LIST(APPEND listVar "${GEN_DIR}_auto_schedule/${BC_NAME}")
        LIST(APPEND incVar  "${GEN_DIR}_auto_schedule")
        else()
        LIST(APPEND listVar "${GEN_DIR}/${BC_NAME}")
        LIST(APPEND incVar  "${GEN_DIR}")
    endif()
    target_link_libraries(camera_pipe_process PRIVATE ${LIB} Threads::Threads)
endforeach()


set_target_properties(camera_pipe_process PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${incVar}")
llvmir_attach_bc_target(camera_pipe_process_bc camera_pipe_process)
add_dependencies(camera_pipe_process_bc camera_pipe_process)
get_property(camera_pipe_process_bc_dir TARGET camera_pipe_process_bc PROPERTY LLVMIR_DIR)
get_property(camera_pipe_process_bc_file TARGET camera_pipe_process_bc PROPERTY LLVMIR_FILES)
LIST(APPEND listVar "${camera_pipe_process_bc_dir}/${camera_pipe_process_bc_file}")


#message("${listVar}")
#message("${RUNTIME_NAME}")
#message("${CMAKE_CURRENT_BINARY_DIR}/${LLVMIR_DIR}/wavelet_bc/wavelet.bc")

set_target_properties(bc_files_camera_pipe_linked PROPERTIES DEPENDS "${listVar}")
# this property is required by our parasitic targets
set_target_properties(bc_files_camera_pipe_linked PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(bc_files_camera_pipe_linked PROPERTIES LLVMIR_TYPE LLVMIR_BINARY)
set_target_properties(bc_files_camera_pipe_linked PROPERTIES LLVMIR_DIR "")
set_target_properties(bc_files_camera_pipe_linked PROPERTIES LLVMIR_FILES "${listVar}")

llvmir_attach_link_target(
  TARGET camera_pipe_exe
DEPENDS bc_files_camera_pipe_linked)

# add_executable(camera_pipe_process process.cpp)
# halide_use_image_io(camera_pipe_process)

# halide_generator(camera_pipe.generator SRCS camera_pipe_generator.cpp)

# foreach(AUTO_SCHEDULE false true)
#     if(${AUTO_SCHEDULE})
#         set(LIB camera_pipe_auto_schedule)
#     else()
#         set(LIB camera_pipe)
#     endif()
#     halide_library_from_generator(${LIB}
#                                   GENERATOR camera_pipe.generator
#                                   GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
#     target_link_libraries(camera_pipe_process PRIVATE ${LIB} ${curved_lib})
# endforeach()
