#
# NOTE: Unlike all other CMakeLists.txt in the apps/ folder, this
# is deliberately intended to be standalone (not included from the toplevel)
# in order to show the minimum scaffolding necessary to use ahead-of-time
# Generators in a simple app.
#
# To use:
# mkdir cmake_build && cd cmake_build && cmake .. && make -j8 && ./bin/wavelet ../../images/gray.png .

project(lens_blur)
cmake_minimum_required(VERSION 3.1.3)

# Define the lens_blur app
add_executable(lens_blur_process "${CMAKE_CURRENT_SOURCE_DIR}/process.cpp")
set_target_properties(lens_blur_process PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(lens_blur_process PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(lens_blur_process PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_use_image_io(lens_blur_process)

add_custom_target(bc_files_lens_blur)
# Define a halide_library() for each generator we have, and link each one into lens_blur
file(GLOB GENS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*_generator.cpp")
SET(listVar "")
SET(incVar "")
halide_generator(lens_blur.generator SRCS lens_blur_generator.cpp)
SET(GEN_SRC lens_blur_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    # Determine auto schedule and determine bc file name
    if(${AUTO_SCHEDULE})
        set(LIB lens_blur_auto_schedule)
        string(REPLACE "_generator.cpp" "_auto_schedule.bc" BC_NAME "${GEN_SRC}")
    else()
        string(REPLACE "_generator.cpp" ".bc" BC_NAME "${GEN_SRC}")
        set(LIB lens_blur)
    endif()
   # Create the generator library
    halide_library_from_generator(${LIB}
                                  GENERATOR lens_blur.generator
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
    target_link_libraries(lens_blur_process PRIVATE ${LIB} Threads::Threads)
endforeach()


# foreach(GEN_SRC ${GENS})
#     string(REPLACE "_generator.cpp" "" GEN_NAME "${GEN_SRC}")
#     string(REPLACE "_generator.cpp" ".bc" BC_NAME "${GEN_SRC}")
#     _halide_genfiles_dir("${GEN_NAME}" GEN_DIR)
#     LIST(APPEND listVar "${GEN_DIR}/${BC_NAME}")
#     halide_library("${GEN_NAME}" SRCS ${GEN_SRC})
#     target_link_libraries(lens_blur PUBLIC "${GEN_NAME}")
#     LIST(APPEND incVar "${GEN_DIR}")
# endforeach()

set_target_properties(lens_blur_process PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${incVar}")
llvmir_attach_bc_target(lens_blur_process_bc lens_blur_process)
add_dependencies(lens_blur_process_bc lens_blur_process)
get_property(lens_blur_process_bc_dir TARGET lens_blur_process_bc PROPERTY LLVMIR_DIR)
get_property(lens_blur_process_bc_file TARGET lens_blur_process_bc PROPERTY LLVMIR_FILES)
LIST(APPEND listVar "${lens_blur_process_bc_dir}/${lens_blur_process_bc_file}")


#message("${listVar}")
#message("${RUNTIME_NAME}")
#message("${CMAKE_CURRENT_BINARY_DIR}/${LLVMIR_DIR}/wavelet_bc/wavelet.bc")

set_target_properties(bc_files_lens_blur PROPERTIES DEPENDS "${listVar}")
# this property is required by our parasitic targets
set_target_properties(bc_files_lens_blur PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(bc_files_lens_blur PROPERTIES LLVMIR_TYPE LLVMIR_BINARY)
set_target_properties(bc_files_lens_blur PROPERTIES LLVMIR_DIR "")
set_target_properties(bc_files_lens_blur PROPERTIES LLVMIR_FILES "${listVar}")

llvmir_attach_link_target(
  TARGET lens_blur_exe
DEPENDS bc_files_lens_blur)






