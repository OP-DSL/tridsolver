cmake_minimum_required(VERSION 2.8.8)

if (BUILD_FOR_GPU)
  find_package(CUDA REQUIRED)
  include(FindCUDA)

  set(CUDA_64_BIT_DEVICE_CODE ON) # -m64 compiler option
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  # architecture
  set(ARCHITECTURE
      "AUTO"
      CACHE STRING "What is your target CUDA architecture?")
  if (${ARCHITECTURE} STREQUAL "AUTO")
    set(CUDA_ARCH_LIST Auto)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
    string(
      REPLACE ";"
              " "
              ARCH_FLAGS
              "${CUDA_NVCC_FLAGS}")
  else ()
    string(
      REPLACE "NUM"
              ${ARCHITECTURE}
              ARCH_FLAGS
              "-gencode arch=compute_NUM,code=sm_NUM")
  endif ()

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -dlcm=cg -std=c++11")
  set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3")

  cuda_include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/cuda/
                           ${PROJECT_SOURCE_DIR}/src/cuda/generics/)
  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so) # , ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so)

  if (USE_PROFILING)
    # PROFILE disables macros with the suffix 2 to achieve lower overhead to use with profiler

    # DETAILED enables all macros and performs deatailed time measures, but has bigger overhead
    set(PROFILING_LEVEL
        "DETAILED"
        CACHE STRING "Level of details for profiling")
    set_property(CACHE PROFILING_LEVEL PROPERTY STRINGS PROFILE DETAILED)
    cuda_add_library(profiling_cuda SHARED ./cuda_timing.cu)
    target_compile_options(profiling_cuda PRIVATE -fPIC)
    if (BUILD_FOR_MPI)
      target_compile_definitions(profiling_cuda PRIVATE -DUSE_MPI)
    endif (BUILD_FOR_MPI)
    if (${PROFILING_LEVEL} STREQUAL "PROFILE")
      target_compile_definitions(profiling_cuda PUBLIC -DPROFILING=1)
    else (${PROFILING_LEVEL} STREQUAL "PROFILE")
      target_compile_definitions(profiling_cuda PUBLIC -DPROFILING=2)
    endif (${PROFILING_LEVEL} STREQUAL "PROFILE")
    target_link_directories(profiling_cuda PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    target_link_libraries(profiling_cuda nvToolsExt)
  endif (USE_PROFILING)

  cuda_add_library(
    tridcuda
    ${PROJECT_SOURCE_DIR}/src/cuda/trid_cuda.cu
    SHARED
    OPTIONS
    -Xcompiler
    -fPIC) # Make binary relocatable

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

    cuda_add_library(
      tridcuda_mpi
      ${PROJECT_SOURCE_DIR}/src/cuda/trid_cuda_mpi.cu
      SHARED
      OPTIONS
      -Xcompiler
      -fPIC
      -Xcompiler) # Make binary relocatable

    if (USE_CUDA_AWARE_MPI)
      target_compile_definitions(tridcuda_mpi PRIVATE TRID_CUDA_AWARE_MPI)
    endif (USE_CUDA_AWARE_MPI)

    target_link_libraries(tridcuda_mpi tridcuda)
    if (USE_PROFILING)
      target_link_libraries(tridcuda_mpi profiling_cuda)
    endif (USE_PROFILING)
    install(
      TARGETS tridcuda_mpi
      LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
      ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)

    install(FILES ${PROJECT_SOURCE_DIR}/include/trid_mpi_cuda.hpp
                  ${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp DESTINATION ${CMAKE_BINARY_DIR}/include)
  endif (BUILD_FOR_MPI)

endif (BUILD_FOR_GPU)
