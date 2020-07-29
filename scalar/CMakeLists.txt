cmake_minimum_required(VERSION 2.8.8)

option(INTEL_CC "Use the Intel compiler." ON)
option(BUILD_FOR_CPU "Build library for CPU architecture." OFF)
option(BUILD_FOR_MIC "Build library for MIC architecture." OFF)
option(BUILD_FOR_GPU "Build library for GPU architecture." OFF)
option(BUILD_FOR_MPI "Build library for MPI combnied with the above architectures." OFF)
option(USE_PROFILING "Build timing library for to measure timings." OFF)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

if (INTEL_CC)
  option(PRODUCE_OPT_REPORT "Print intel optimisation reports" OFF)
  # Detect/find Intel compilers
  find_program(CMAKE_C_COMPILER NAMES mpicc)
  find_program(CMAKE_CXX_COMPILER NAMES mpicxx)
  find_program(CMAKE_AR NAMES xiar)
  find_program(CMAKE_LINKER NAMES xild)

  set(INTEL_PATH
      $ENV{INTEL_PATH}
      CACHE FILEPATH "Intel compiler suite path")

  # set(CMAKE_C_COMPILER   ${INTEL_PATH}/bin/intel64/icc  CACHE FILEPATH "Intel compiler is prefered")

  # set(CMAKE_CXX_COMPILER ${INTEL_PATH}/bin/intel64/icpc CACHE FILEPATH "Intel compiler is prefered")

  if (CMAKE_C_COMPILER MATCHES CMAKE_C_COMPILER-NOTFOUND
      OR CMAKE_CXX_COMPILER MATCHES CMAKE_CXX_COMPILER-NOTFOUND
      OR CMAKE_AR MATCHES CMAKE_AR-NOTFOUND
      OR CMAKE_LINKER MATCHES CMAKE_LINKER-NOTFOUND)
    message(FATAL_ERROR "Cannot find Intel compiler.  You may need to run `/opt/intel/bin/compilervars.sh intel64'")
  endif ()
endif (INTEL_CC)

project("libtrid")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
message(${PROJECT_SOURCE_DIR}/cmake)

if (INTEL_CC)
  # Set compiler flags/options for Intel compiler
  set(FLAGS_INTEL_DEFAULT "-xHost -m64 -parallel -qopenmp -qno-offload")
  if (PRODUCE_OPT_REPORT)
    set(FLAGS_INTEL_DEFAULT "${FLAGS_INTEL_DEFAULT} -qopt-report=2 -qopt-report-phase=vec -qopt-report-file=stdout")
  endif (PRODUCE_OPT_REPORT)

  # cmake-format: off
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} 		${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}       ${FLAGS_INTEL_DEFAULT}")
  # cmake-format: on
else (INTEL_CC)
  # Set compiler flags/options for GCC compiler
  set(FLAGS_GCC_DEFAULT "-march=native -fopenmp -g") # -march=core-avx2
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
endif (INTEL_CC)

# Define maximal length of a scalar tridiagonal system
set(N_MAX
    1024
    CACHE STRING
          "Maximal length of the internal buffer for storing intermediate c and d vectors of the Thomas algorithm")
add_definitions(-DN_MAX=${N_MAX})

# Invoke subprojects
add_subdirectory(src/cpu)
add_subdirectory(src/cuda)
enable_testing()
add_subdirectory(test)
