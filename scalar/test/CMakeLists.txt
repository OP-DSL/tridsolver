add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

if (BUILD_FOR_CPU)
  set(TEST_SOURCES catch_main.cpp test_cpu.cpp)

  add_executable(tests ${TEST_SOURCES})
  target_link_libraries(
    tests
    Catch
    tridcpu
    OpenMP::OpenMP_CXX)
  target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)

  add_test(
    NAME catch_test
    COMMAND tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  if (BUILD_FOR_MPI)
    set(MPI_TEST_NUM_PROC
        "2"
        CACHE STRING "The number of processes to launch in MPI tests.")
    add_executable(mpi_tests catch_mpi_main.cpp test_cpu_mpi.cpp)
    target_link_libraries(
      mpi_tests
      Catch
      tridcpu
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX)
    target_include_directories(mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
    add_test(
      NAME catch_mpi_test
      COMMAND "mpirun" -np ${MPI_TEST_NUM_PROC} $<TARGET_FILE:mpi_tests>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    if (USE_PROFILING)
      find_package(OpenMP REQUIRED)
      add_executable(mpi_perf_test performance_test.cpp)
      target_include_directories(mpi_perf_test PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
      target_link_libraries(
        mpi_perf_test
        tridcpu
        profiling
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX)
    endif (USE_PROFILING)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_CPU)

if (BUILD_FOR_GPU)
  set(CUDA_64_BIT_DEVICE_CODE ON) # -m64 compiler option
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}         -arch sm_35 -Xptxas -dlcm=cg ")
  set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false ")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3 ")

  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
  add_executable(cuda_tests ${PROJECT_SOURCE_DIR}/test/catch_main.cpp ${PROJECT_SOURCE_DIR}/test/test_cuda.cu)
  # OPTIONS -Xcompiler -fPIC)

  target_link_libraries(cuda_tests Catch tridcuda)
  target_include_directories(cuda_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
  add_test(
    NAME catch_test_cuda
    COMMAND cuda_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  if (BUILD_FOR_MPI)
    option(USE_NCCL "Build library with NCCL for MPI comms" OFF)
    if (USE_NCCL)
      find_package(NCCL)
      if (NCCL_FOUND)
        add_definitions(-DTRID_NCCL)
        include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
      endif ()
    endif ()
    set(MPI_TEST_NUM_PROC
        "2"
        CACHE STRING "The number of processes to launch in MPI tests.")
    find_package(MPI REQUIRED)
    add_executable(cuda_mpi_tests catch_mpi_main.cpp test_cuda_mpi.cu)
    target_include_directories(cuda_mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/test/
                                                      ${PROJECT_SOURCE_DIR}/src/cuda)
    # cuda_add_cublas_to_target(cuda_mpi_tests)
    target_link_libraries(
      cuda_mpi_tests
      Catch
      tridcuda_mpi
      MPI::MPI_CXX
      ${NCCL_LIBRARY})
    add_test(
      NAME catch_cuda_mpi_test
      COMMAND "mpirun" -np ${MPI_TEST_NUM_PROC} $<TARGET_FILE:cuda_mpi_tests>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    if (USE_PROFILING)
      # performance_test.cu is a symlink to performance_test.cpp TODO fix compilation with cpp (compile with nvcc)
      add_executable(cuda_mpi_perf_test performance_test.cu)
      target_include_directories(cuda_mpi_perf_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/test/
                                                            ${PROJECT_SOURCE_DIR}/src/cuda)
      cuda_add_cublas_to_target(cuda_mpi_perf_test)
      target_compile_definitions(cuda_mpi_perf_test PRIVATE TRID_PERF_CUDA)

      target_link_libraries(
        cuda_mpi_perf_test
        tridcuda_mpi
        profiling_cuda
        MPI::MPI_CXX
        ${NCCL_LIBRARY})
    endif (USE_PROFILING)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_GPU)
