# The cuFFT store callbacks are compiled to LTO-IR, embedded in the library, and
# handed to cuFFT, which links them into its own kernels when a plan is made.
set(archs ${CMAKE_CUDA_ARCHITECTURES})
if(archs STREQUAL "native")
  set(archs ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
elseif(archs STREQUAL "all" OR archs STREQUAL "all-major")
  message(FATAL_ERROR "QORAPHASECU_CUFFT_LTO needs explicit CMAKE_CUDA_ARCHITECTURES")
endif()
set(gencode)
foreach(arch IN LISTS archs)
  string(REGEX REPLACE "-.*" "" arch "${arch}")
  list(APPEND gencode "-gencode=arch=compute_${arch},code=lto_${arch}")
endforeach()
list(REMOVE_DUPLICATES gencode)

set(source ${PROJECT_SOURCE_DIR}/src/cuda/callbacks.cu)
set(fatbin ${CMAKE_CURRENT_BINARY_DIR}/cufft_callbacks.fatbin)
set(embedded ${CMAKE_CURRENT_BINARY_DIR}/cufft_callbacks.cpp)
add_custom_command(
  OUTPUT ${fatbin}
  COMMAND ${CMAKE_CUDA_COMPILER} -std=c++20 -dc -fatbin ${gencode}
    --expt-relaxed-constexpr -I${PROJECT_SOURCE_DIR}/include
    "$<$<BOOL:${CMAKE_CUDA_HOST_COMPILER}>:-ccbin=${CMAKE_CUDA_HOST_COMPILER}>"
    "$<$<BOOL:${MSVC}>:-Xcompiler=/Zc:preprocessor>"
    -o ${fatbin} ${source}
  DEPENDS ${source} ${PROJECT_SOURCE_DIR}/src/cuda/device.cuh
    ${PROJECT_SOURCE_DIR}/src/cuda/kernels.cuh ${PROJECT_SOURCE_DIR}/src/core/pointwise.hpp
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_command(
  OUTPUT ${embedded}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${fatbin} -DOUTPUT=${embedded} -DNAME=qpcu_cufft_callbacks
    -P ${PROJECT_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS ${fatbin} ${PROJECT_SOURCE_DIR}/cmake/embed.cmake
  VERBATIM)
target_sources(qoraphasecu PRIVATE ${embedded})
target_compile_definitions(qoraphasecu PUBLIC QORAPHASECU_HAS_CUFFT_LTO)
