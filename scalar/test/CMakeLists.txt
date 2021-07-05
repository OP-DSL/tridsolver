add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
add_library(test_common_options INTERFACE)
target_compile_features(test_common_options INTERFACE cxx_std_17)

if (BUILD_FOR_CPU)
  if (BUILD_FOR_SN)
    set(TEST_SOURCES catch_main.cpp test_cpu.cpp)

    add_executable(tests ${TEST_SOURCES})
    target_link_libraries(
      tests
      Catch
      tridcpu
      OpenMP::OpenMP_CXX
      test_common_options)
    target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)

    add_test(
      NAME catch_test
      COMMAND tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif (BUILD_FOR_SN)
  if (BUILD_FOR_MPI)
    set(MPI_TEST_NUM_PROC
        "2"
        CACHE STRING "The number of processes to launch in MPI tests.")
    add_executable(mpi_tests catch_mpi_main.cpp test_cpu_mpi.cpp)
    target_link_libraries(
      mpi_tests
      Catch
      tridcpu_mpi
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX
      test_common_options)
    target_include_directories(mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
    add_test(
      NAME catch_mpi_test
      COMMAND "mpirun" -np ${MPI_TEST_NUM_PROC} $<TARGET_FILE:mpi_tests>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    if (USE_PROFILING)
      add_executable(mpi_perf_test performance_test.cpp)
      target_include_directories(mpi_perf_test PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
      target_link_libraries(
        mpi_perf_test
        tridcpu_mpi
        profiling
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        test_common_options)
    endif (USE_PROFILING)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_CPU)

if (BUILD_FOR_GPU)
  set(cuda_flags -m64 -Xptxas -dlcm=cg)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0 ")
  if (CMAKE_CXX_COMPILER_ID MATCHES "NVCC")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -ftz=false -fmad=false")
  endif ()

  add_executable(cuda_tests ${PROJECT_SOURCE_DIR}/test/catch_main.cpp ${PROJECT_SOURCE_DIR}/test/test_cuda.cu)
  set_target_properties(cuda_tests PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)

  target_link_libraries(
    cuda_tests
    Catch
    tridcuda
    OpenMP::OpenMP_CXX
    test_common_options)
  target_include_directories(cuda_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
  add_test(
    NAME catch_test_cuda
    COMMAND cuda_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  if (BUILD_FOR_MPI)
    set(MPI_TEST_NUM_PROC
        "2"
        CACHE STRING "The number of processes to launch in MPI tests.")
    add_executable(cuda_mpi_tests catch_mpi_main.cpp test_cuda_mpi.cu)
    set_target_properties(cuda_mpi_tests PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(cuda_mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/test/
                                                      ${PROJECT_SOURCE_DIR}/src/cuda)
    target_compile_options(cuda_mpi_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)
    target_link_libraries(
      cuda_mpi_tests
      Catch
      tridcuda_mpi
      OpenMP::OpenMP_CXX
      MPI::MPI_CXX
      test_common_options)
    add_test(
      NAME catch_cuda_mpi_test
      COMMAND "mpirun" -np ${MPI_TEST_NUM_PROC} $<TARGET_FILE:cuda_mpi_tests>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    if (USE_PROFILING)
      add_executable(cuda_mpi_perf_test performance_test.cpp)
      set_source_files_properties(performance_test.cpp PROPERTIES LANGUAGE CUDA)
      target_include_directories(cuda_mpi_perf_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/test/)
      target_compile_definitions(cuda_mpi_perf_test PRIVATE TRID_PERF_CUDA)

      target_link_libraries(
        cuda_mpi_perf_test
        tridcuda_mpi
        profiling_cuda
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        test_common_options)
    endif (USE_PROFILING)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_GPU)
