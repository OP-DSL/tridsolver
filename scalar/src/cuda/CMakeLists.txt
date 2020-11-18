if (BUILD_FOR_GPU)
  # need for cublas support until cmake 3.17
  find_package(CUDA REQUIRED)
  include(FindCUDA)

  # architecture
  set(ARCHITECTURE
      "AUTO"
      CACHE STRING "Your target CUDA architecture")
  if (${ARCHITECTURE} STREQUAL "AUTO")
    set(CUDA_ARCH_LIST Auto)
    cuda_select_nvcc_arch_flags(ARCH_LIST ${CUDA_ARCH_LIST})
    string(
      REPLACE ";"
              " "
              ARCH_FLAGS
              "${ARCH_LIST}")
  else ()
    string(
      REPLACE "NUM"
              ${ARCHITECTURE}
              ARCH_FLAGS
              "-gencode arch=compute_NUM,code=sm_NUM")
  endif ()

  set(cuda_flags "-m64 -Xptxas -dlcm=cg ${ARCH_FLAGS}")

  # does not have effect TODO find working solution:

  # set(CUDA_PROPAGATE_HOST_FLAGS OFF) set(CUDA_NVCC_FLAGS_DEBUG
  # "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false")

  if (USE_PROFILING)
    # PROFILE disables macros with the suffix 2 to achieve lower overhead to use with profiler

    # DETAILED enables all macros and performs deatailed time measures, but has bigger overhead
    set(PROFILING_LEVEL
        "DETAILED"
        CACHE STRING "Level of details for profiling")
    set_property(CACHE PROFILING_LEVEL PROPERTY STRINGS PROFILE DETAILED)
    add_library(profiling_cuda SHARED ./cuda_timing.cu)
    set_target_properties(profiling_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(profiling_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include/)
    if (BUILD_FOR_MPI)
      target_compile_definitions(profiling_cuda PRIVATE -DUSE_MPI)
      target_link_libraries(profiling_cuda PUBLIC MPI::MPI_CXX)
    endif (BUILD_FOR_MPI)
    if (${PROFILING_LEVEL} STREQUAL "PROFILE")
      target_compile_definitions(profiling_cuda PUBLIC -DPROFILING=1)
    else (${PROFILING_LEVEL} STREQUAL "PROFILE")
      target_compile_definitions(profiling_cuda PUBLIC -DPROFILING=2)
    endif (${PROFILING_LEVEL} STREQUAL "PROFILE")
    target_link_directories(profiling_cuda PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    target_link_libraries(profiling_cuda PRIVATE nvToolsExt)
  endif (USE_PROFILING)

  add_library(tridcuda SHARED ./trid_cuda.cu)
  cuda_add_cublas_to_target(tridcuda)
  target_link_libraries(tridcuda trid_common_options)
  set_target_properties(tridcuda PROPERTIES COMPILE_FLAGS "${cuda_flags}")
  target_include_directories(tridcuda PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda/ ${PROJECT_SOURCE_DIR}/src/cuda/generics/)

  install(
    TARGETS tridcuda
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)

  install(
    FILES ${PROJECT_SOURCE_DIR}/include/trid_cuda.h
          ${PROJECT_SOURCE_DIR}/include/trid_common.h
          ${PROJECT_SOURCE_DIR}/include/trid_util.h
          ${PROJECT_SOURCE_DIR}/include/cutil_inline.h
    DESTINATION ${CMAKE_BINARY_DIR}/include)

  if (BUILD_FOR_MPI)
    option(USE_CUDA_AWARE_MPI "Build library with CUDA-aware MPI" OFF)
    option(USE_NCCL "Build library with NCCL for MPI comms" OFF)
    find_package(MPI REQUIRED)
    find_package(OpenMP REQUIRED)

    if (USE_NCCL)
      find_package(NCCL)
      if (NCCL_FOUND)
        add_definitions(-DTRID_NCCL)
        include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
      endif ()
    endif ()

    add_library(tridcuda_mpi SHARED ./trid_cuda_mpi.cu)
    target_include_directories(tridcuda_mpi PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda/
                                                    ${PROJECT_SOURCE_DIR}/src/cuda/generics/)
    set_target_properties(tridcuda_mpi PROPERTIES COMPILE_FLAGS "${cuda_flags}")

    if (USE_CUDA_AWARE_MPI)
      target_compile_definitions(tridcuda_mpi PRIVATE TRID_CUDA_AWARE_MPI)
    endif (USE_CUDA_AWARE_MPI)

    target_link_libraries(
      tridcuda_mpi
      PRIVATE trid_common_options
      PUBLIC tridcuda
      PUBLIC MPI::MPI_CXX)
    if (USE_PROFILING)
      target_link_libraries(tridcuda_mpi PUBLIC profiling_cuda)
    endif (USE_PROFILING)
    install(
      TARGETS tridcuda_mpi
      LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
      ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)

    install(FILES ${PROJECT_SOURCE_DIR}/include/trid_mpi_cuda.hpp
                  ${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp DESTINATION ${CMAKE_BINARY_DIR}/include)
  endif (BUILD_FOR_MPI)

endif (BUILD_FOR_GPU)
