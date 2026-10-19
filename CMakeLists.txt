cmake_minimum_required(VERSION 3.28)

project(QoraPhaseCU VERSION 0.1.0 LANGUAGES CXX)

option(QORAPHASECU_CUDA "Build the CUDA backend" OFF)
option(QORAPHASECU_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})
option(QORAPHASECU_BENCH "Build the benchmark" ${PROJECT_IS_TOP_LEVEL})
option(QORAPHASECU_EXAMPLES "Build the C++ examples" ${PROJECT_IS_TOP_LEVEL})
option(QORAPHASECU_PYTHON "Build the nanobind module" OFF)
option(QORAPHASECU_WERROR "Treat compiler warnings as errors" OFF)
option(QORAPHASECU_CUFFT_LTO "Build the cuFFT LTO callback variant when the toolkit supports it" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(QORAPHASECU_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  # LTO callbacks arrived in cuFFT 11.5 (CUDA 12.6); older headers lack the entry point.
  if(QORAPHASECU_CUFFT_LTO)
    find_file(QORAPHASECU_CUFFTXT_H cufftXt.h PATHS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(QORAPHASECU_CUFFTXT_H)
      file(STRINGS "${QORAPHASECU_CUFFTXT_H}" jit_callback REGEX "cufftXtSetJITCallback")
    endif()
    if(NOT jit_callback)
      message(STATUS "cuFFT without LTO callbacks: the callback variant is disabled")
      set(QORAPHASECU_CUFFT_LTO OFF)
    endif()
  endif()
endif()

include(FetchContent)
FetchContent_Declare(
  pocketfft
  GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
  GIT_TAG 4d865cba24359b05b37da4e75aee24498c3b78e7 # branch cpp
  GIT_SHALLOW FALSE
  SOURCE_SUBDIR none)
FetchContent_MakeAvailable(pocketfft)

add_library(qoraphasecu_warnings INTERFACE)
if(MSVC)
  target_compile_options(qoraphasecu_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- $<$<BOOL:${QORAPHASECU_WERROR}>:/WX>>)
else()
  target_compile_options(qoraphasecu_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow
      $<$<BOOL:${QORAPHASECU_WERROR}>:-Werror>>)
endif()
if(QORAPHASECU_CUDA)
  target_compile_options(qoraphasecu_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<BOOL:${QORAPHASECU_WERROR}>:-Werror=all-warnings>>)
endif()

add_library(qoraphasecu
  src/core/metrics.cpp
  src/core/problem.cpp
  src/core/recipe.cpp
  src/core/solver.cpp
  src/cpu/cpu_solver.cpp
  src/cpu/fft.cpp
  src/cpu/kernels.cpp
  src/cpu/operators.cpp)
add_library(qoraphasecu::qoraphasecu ALIAS qoraphasecu)
target_include_directories(qoraphasecu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(qoraphasecu SYSTEM PRIVATE ${pocketfft_SOURCE_DIR})
target_link_libraries(qoraphasecu PRIVATE qoraphasecu_warnings)
target_compile_features(qoraphasecu PUBLIC cxx_std_20)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(qoraphasecu PRIVATE Threads::Threads)
endif()

if(QORAPHASECU_CUDA)
  target_sources(qoraphasecu PRIVATE
    src/cuda/cuda_solver.cu
    src/cuda/half.cu
    src/cuda/kernels.cu
    src/cuda/operators.cu)
  target_compile_definitions(qoraphasecu PUBLIC QORAPHASECU_HAS_CUDA)
  target_link_libraries(qoraphasecu PRIVATE CUDA::cudart_static CUDA::cufft)
  target_compile_options(qoraphasecu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(QORAPHASECU_CUFFT_LTO)
    include(cmake/cufft_callbacks.cmake)
  endif()
else()
  target_sources(qoraphasecu PRIVATE src/cpu/no_cuda.cpp)
endif()

if(QORAPHASECU_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(QORAPHASECU_BENCH)
  add_subdirectory(bench)
endif()
if(QORAPHASECU_EXAMPLES)
  add_subdirectory(examples)
endif()
if(QORAPHASECU_PYTHON)
  add_subdirectory(python)
endif()
