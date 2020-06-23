cmake_minimum_required(VERSION 3.9)

if (BUILD_FOR_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Could not find CUDA compiler, cannot build for GPU")
  endif ()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(CUDA_64_BIT_DEVICE_CODE ON) # -m64 compiler option
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}         -arch=sm_35 -gencode arch=compute_60,code=sm_60 -Xptxas -dlcm=cg")
  set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false               ")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3                                            ")

  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so)

  add_library(tridcuda ${PROJECT_SOURCE_DIR}/src/cuda/trid_cuda.cu SHARED)
  # Make binary relocatable
  set_target_properties(tridcuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(tridcuda PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/cuda/
                                              ${PROJECT_SOURCE_DIR}/src/cuda/generics/)

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
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})

    add_library(tridcuda_mpi ${PROJECT_SOURCE_DIR}/src/cuda/trid_cuda_mpi.cu SHARED)
    target_include_directories(tridcuda_mpi PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/cuda/
                                                    ${PROJECT_SOURCE_DIR}/src/cuda/generics/)

    # Make binary relocatable
    set_target_properties(tridcuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(tridcuda_mpi ${MPI_CXX_LIBRARIES})
    # target_link_libraries(tridcuda_mpi tridcuda)
    install(
      TARGETS tridcuda_mpi
      LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
      ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)

    install(FILES ${PROJECT_SOURCE_DIR}/include/trid_mpi_cuda.hpp
                  ${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp DESTINATION ${CMAKE_BINARY_DIR}/include)
  endif (BUILD_FOR_MPI)

endif (BUILD_FOR_GPU)
