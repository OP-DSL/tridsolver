cmake_minimum_required(VERSION 3.9)
project("libtrid" C CXX)

option(BUILD_FOR_CPU "Build library for CPU architecture." OFF)
option(BUILD_FOR_MIC "Build library for MIC architecture." OFF)
option(BUILD_FOR_GPU "Build library for GPU architecture." OFF)
option(BUILD_FOR_MPI "Build library for MPI combnied with the above architectures." OFF)
option(BUILD_TESTING "Build tests for the library." OFF)
option(USE_PROFILING "Build timing library for to measure timings." OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  option(PRODUCE_OPT_REPORT "Print intel optimisation reports" OFF)

  # Set compiler flags/options for Intel compiler
  set(FLAGS_INTEL_DEFAULT "-xHost -m64 -parallel -qno-offload")
  if (PRODUCE_OPT_REPORT)
    set(FLAGS_INTEL_DEFAULT "${FLAGS_INTEL_DEFAULT} -qopt-report=2 -qopt-report-phase=vec -qopt-report-file=stdout")
  endif (PRODUCE_OPT_REPORT)

  # cmake-format: off
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}     ${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS}       ${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  # cmake-format: on
else (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # Set compiler flags/options for GCC compiler
  set(FLAGS_GCC_DEFAULT "-march=native ") # -march=core-avx2
  set(FLAGS_GCC_DEBUG "-O0")
  set(FLAGS_GCC_RELEASE "-flto -fstrict-aliasing -finline-functions") # -ffast-math -fargument-noalias -fargument-
                                                                      # noalias-global

  # cmake-format: off
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_GCC_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_GCC_DEBUG}  ")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}     ${FLAGS_GCC_RELEASE}")

  set(CMAKE_C_FLAGS           "${CMAKE_CXX_FLAGS}       ${FLAGS_GCC_DEFAULT} -std=c99")
  set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_GCC_DEBUG}   -std=c99")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_CXX_RELEASE}     ${FLAGS_GCC_RELEASE} -std=c99")
  # cmake-format: on
endif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")

# Define maximal length of a scalar tridiagonal system
set(N_MAX
    1024
    CACHE STRING
          "Maximal length of the internal buffer for storing intermediate c and d vectors of the Thomas algorithm")
set(N_MPI_MAX 128 CACHE STRING "Maximum number of MPI processes in a dimension")
add_library(trid_common_options INTERFACE)
target_compile_features(trid_common_options INTERFACE cxx_std_14)
target_compile_definitions(trid_common_options INTERFACE -DN_MAX=${N_MAX})
target_compile_definitions(trid_common_options INTERFACE -DN_MPI_MAX=${N_MPI_MAX})
target_include_directories(trid_common_options INTERFACE ${PROJECT_SOURCE_DIR}/include)

if (BUILD_FOR_MPI)
  find_package(MPI REQUIRED)
  # TODO check it
endif (BUILD_FOR_MPI)

if (BUILD_FOR_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
  else (CMAKE_CUDA_COMPILER)
    # TODO ecking
  endif (CMAKE_CUDA_COMPILER)
endif (BUILD_FOR_GPU)

# Invoke subprojects
add_subdirectory(src/cpu)
add_subdirectory(src/cuda)
if (BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif (BUILD_TESTING)
