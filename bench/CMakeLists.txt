add_executable(qpcu_bench bench.cpp)
target_link_libraries(qpcu_bench PRIVATE qoraphasecu qoraphasecu_warnings)
if(QORAPHASECU_CUDA)
  target_sources(qpcu_bench PRIVATE fft_floor.cu)
  target_link_libraries(qpcu_bench PRIVATE CUDA::cudart_static CUDA::cufft)
else()
  target_sources(qpcu_bench PRIVATE fft_floor_none.cpp)
endif()

if(QORAPHASECU_CUDA)
  add_executable(qpcu_profile profile.cu)
  target_link_libraries(qpcu_profile PRIVATE qoraphasecu qoraphasecu_warnings CUDA::cudart_static
                                             CUDA::nvtx3)
endif()
