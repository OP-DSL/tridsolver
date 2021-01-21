if (BUILD_FOR_GPU)
  # find cublas, nvToolsExt, etc.
  find_package(CUDAToolkit)

  set(cuda_flags "-m64 -Xptxas -dlcm=cg ")

  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}   -g -G -O0 ")

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
    target_link_libraries(profiling_cuda PRIVATE CUDA::nvToolsExt)
  endif (USE_PROFILING)

  set(${PROJECT_NAME}_components "${${PROJECT_NAME}_components} tridcuda")
  set(${PROJECT_NAME}_components
      ${${PROJECT_NAME}_components}
      PARENT_SCOPE)
  add_library(tridcuda SHARED ./trid_cuda.cu)
  add_library(${PROJECT_NAME}::tridcuda ALIAS tridcuda)
  target_link_libraries(
    tridcuda
    PRIVATE trid_common_options
    PUBLIC trid_global_options CUDA::cublas)
  set_target_properties(tridcuda PROPERTIES COMPILE_FLAGS "${cuda_flags}")
  set_target_properties(tridcuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_include_directories(tridcuda PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda/ ${PROJECT_SOURCE_DIR}/src/cuda/generics/)

  install(
    TARGETS tridcuda
    EXPORT tridcudaTarget
    LIBRARY DESTINATION ${PROJECT_NAME}/lib
    ARCHIVE DESTINATION ${PROJECT_NAME}/lib)

  install(
    FILES ${PROJECT_SOURCE_DIR}/include/trid_cuda.h
          ${PROJECT_SOURCE_DIR}/include/trid_common.h
          ${PROJECT_SOURCE_DIR}/include/trid_util.h
          ${PROJECT_SOURCE_DIR}/include/cutil_inline.h
    DESTINATION ${PROJECT_NAME}/include)

  install(
    EXPORT tridcudaTarget
    FILE tridcudaTarget.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${PROJECT_NAME}/lib/cmake)

  if (BUILD_FOR_MPI)
    # MPI: use pure MPI communication

    # CUDA_AWARE_MPI: use CUDA aware MPI communication, might save the cost of moving data from device to host

    # NCCL: use NCCL library to send data
    set(CUDA_COMMUNICATION_LIBRARY
        "MPI"
        CACHE STRING "Communication options")
    set_property(
      CACHE CUDA_COMMUNICATION_LIBRARY
      PROPERTY STRINGS
               MPI
               CUDA_AWARE_MPI
               NCCL)

    set(${PROJECT_NAME}_components "${${PROJECT_NAME}_components} tridcuda_mpi")
    set(${PROJECT_NAME}_components
        ${${PROJECT_NAME}_components}
        PARENT_SCOPE)
    add_library(tridcuda_mpi SHARED ./trid_cuda_mpi.cu)
    add_library(${PROJECT_NAME}::tridcuda_mpi ALIAS tridcuda_mpi)
    target_include_directories(tridcuda_mpi PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda/
                                                    ${PROJECT_SOURCE_DIR}/src/cuda/generics/)
    set_target_properties(tridcuda_mpi PROPERTIES COMPILE_FLAGS "${cuda_flags}")
    set_target_properties(tridcuda_mpi PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(
      tridcuda_mpi
      PRIVATE trid_common_options
      PUBLIC tridcuda trid_global_options MPI::MPI_CXX)

    if (CUDA_COMMUNICATION_LIBRARY STREQUAL "NCCL")
      find_package(NCCL REQUIRED)
      target_compile_definitions(tridcuda_mpi PUBLIC -DTRID_NCCL)
      target_include_directories(tridcuda_mpi PUBLIC ${NCCL_INCLUDE_DIR})
      target_link_libraries(tridcuda_mpi PUBLIC ${NCCL_LIBRARY})
    elseif (CUDA_COMMUNICATION_LIBRARY STREQUAL "CUDA_AWARE_MPI")
      target_compile_definitions(tridcuda_mpi PRIVATE TRID_CUDA_AWARE_MPI)
    elseif (
      NOT
      CUDA_COMMUNICATION_LIBRARY
      STREQUAL
      "MPI")
      get_property(
        COMM_MODES
        CACHE CUDA_COMMUNICATION_LIBRARY
        PROPERTY STRINGS)
      message(
        FATAL_ERROR
          "Unknown communication mode for CUDA: ${CUDA_COMMUNICATION_LIBRARY}. CUDA_COMMUNICATION_LIBRARY must be one of the folowing: ${COMM_MODES}"
      )
    endif ()

    if (USE_PROFILING)
      target_link_libraries(tridcuda_mpi PUBLIC profiling_cuda)
    endif (USE_PROFILING)

    install(
      TARGETS tridcuda_mpi
      EXPORT tridcuda_mpiTarget
      LIBRARY DESTINATION ${PROJECT_NAME}/lib
      ARCHIVE DESTINATION ${PROJECT_NAME}/lib)

    install(
      FILES ${PROJECT_SOURCE_DIR}/include/trid_cuda.h
            ${PROJECT_SOURCE_DIR}/include/trid_common.h
            ${PROJECT_SOURCE_DIR}/include/trid_mpi_cuda.hpp
            ${PROJECT_SOURCE_DIR}/include/trid_mpi_solver_params.hpp
      DESTINATION ${PROJECT_NAME}/include)
    install(
      EXPORT tridcuda_mpiTarget
      FILE tridcuda_mpiTarget.cmake
      NAMESPACE ${PROJECT_NAME}::
      DESTINATION ${PROJECT_NAME}/lib/cmake)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_GPU)
