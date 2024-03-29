cmake_minimum_required(VERSION 3.18)
project(libtrid CXX)

option(BUILD_FOR_CPU "Build library for CPU architecture." OFF)
option(BUILD_FOR_GPU "Build library for GPU architecture." OFF)
option(BUILD_FOR_MPI "Build library for MPI combnied with the above architectures." OFF)
option(BUILD_FOR_SN "Build library for a single node combnied with the above architectures." OFF)
option(BUILD_TESTING "Build tests for the library." OFF)
option(BUILD_STATIC "Build static library" OFF)
option(USE_PROFILING "Build timing library for to measure timings." OFF)

if (BUILD_STATIC)
  set(BUILD_TYPE STATIC)
else ()
  set(BUILD_TYPE SHARED)
endif ()

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
  set(FLAGS_GCC_RELEASE "-flto -fstrict-aliasing -finline-functions") # -ffast-math -fargument-noalias -fargument-
                                                                      # noalias-global

  # cmake-format: off
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         ${FLAGS_GCC_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FLAGS_GCC_RELEASE}")

  set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS}         ${FLAGS_GCC_DEFAULT}")
  set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}   -O0")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${FLAGS_GCC_RELEASE}")
  # cmake-format: on
endif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")

# Define maximal length of a scalar tridiagonal system
set(N_MAX
    1024
    CACHE STRING
          "Maximal length of the internal buffer for storing intermediate c and d vectors of the Thomas algorithm")
set(N_MPI_MAX
    128
    CACHE STRING "Maximum number of MPI processes in a dimension")
# internal options that shouldn't be exported
add_library(trid_common_options INTERFACE)
target_compile_features(trid_common_options INTERFACE cxx_std_14)
target_compile_definitions(trid_common_options INTERFACE -DN_MAX=${N_MAX})
target_compile_definitions(trid_common_options INTERFACE -DN_MPI_MAX=${N_MPI_MAX})
# tridsolver public options that should be exported with the library
add_library(trid_global_options INTERFACE)
target_include_directories(trid_global_options INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                         $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)

install(
  TARGETS trid_global_options
  EXPORT tridCommonTarget
  LIBRARY DESTINATION ${PROJECT_NAME}/lib
  ARCHIVE DESTINATION ${PROJECT_NAME}/lib)

install(FILES ${PROJECT_SOURCE_DIR}/include/trid_common.h
              ${PROJECT_SOURCE_DIR}/include/tridsolver.h
              ${PROJECT_SOURCE_DIR}/include/trid_params.h 
              DESTINATION ${PROJECT_NAME}/include)

install(
  EXPORT tridCommonTarget
  FILE trid_commonTarget.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${PROJECT_NAME}/lib/cmake)

find_package(OpenMP REQUIRED)
if (BUILD_FOR_MPI)
  find_package(MPI REQUIRED)
endif (BUILD_FOR_MPI)

if (BUILD_FOR_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER)
    message(ERROR "Couldn't find CUDA compiler")
  endif (NOT CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CXX_EXTENSIONS OFF)
  # use dynamic linking to CUDA runtime to avoid error with CUDA aware MPI
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  # architecture
  set(ARCHITECTURE
      "AUTO"
      CACHE STRING "Your target CUDA architecture")
  include(cmake/CUDA_ARCH.cmake)
  select_cuda_architectures("${ARCHITECTURE}" ARCH_LIST)
  message(STATUS "CUDA_ARCH: ${ARCH_LIST}")
  set(CMAKE_CUDA_ARCHITECTURES
      "${ARCH_LIST}"
      CACHE STRING "CUDA architectures" FORCE)

  if (BUILD_FOR_MPI)
    # cmake-format: off
    # MPI: use pure MPI communication
    # CUDA_AWARE_MPI: use CUDA aware MPI communication, might save the cost of moving data from device to host
    # NCCL: use NCCL library to send data
    # cmake-format: on
    set(CUDA_COMMUNICATION_LIBRARY
        "MPI"
        CACHE STRING "Communication options")
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_GPU)

set(libtrid_components "trid_common")
# Invoke subprojects
add_subdirectory(src/cpu)
add_subdirectory(src/cuda)
if (BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif (BUILD_TESTING)

include(CMakePackageConfigHelpers)
# generate the config file that includes the exports
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${PROJECT_NAME}/lib/cmake"
  NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# install the configuration file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake DESTINATION ${PROJECT_NAME}/lib/cmake/)
