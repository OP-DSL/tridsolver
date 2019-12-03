cmake_minimum_required(VERSION 2.8.8)

option(INTEL_CC "Use the Intel compiler." ON) 
option(BUILD_FOR_CPU "Build library for CPU architecture." OFF) 
option(BUILD_FOR_MIC "Build library for MIC architecture." OFF) 
option(BUILD_FOR_GPU "Build library for GPU architecture." OFF) 

if (INTEL_CC) 
  # Detect/find Intel compilers
  find_program(CMAKE_C_COMPILER NAMES   icc) 
  find_program(CMAKE_CXX_COMPILER NAMES icpc) 
  find_program(CMAKE_AR NAMES           xiar) 
  find_program(CMAKE_LINKER NAMES       xild) 

  set(INTEL_PATH $ENV{INTEL_PATH} CACHE FILEPATH "Intel compiler suite path")
  #set(CMAKE_C_COMPILER   ${INTEL_PATH}/bin/intel64/icc  CACHE FILEPATH "Intel compiler is prefered")
  #set(CMAKE_CXX_COMPILER ${INTEL_PATH}/bin/intel64/icpc CACHE FILEPATH "Intel compiler is prefered")

  if (CMAKE_C_COMPILER MATCHES CMAKE_C_COMPILER-NOTFOUND OR 
      CMAKE_CXX_COMPILER MATCHES CMAKE_CXX_COMPILER-NOTFOUND OR 
      CMAKE_AR MATCHES CMAKE_AR-NOTFOUND OR 
      CMAKE_LINKER MATCHES CMAKE_LINKER-NOTFOUND) 
    message(FATAL_ERROR "Cannot find Intel compiler.  You may need to run `. /opt/intel/bin/compilervars.sh intel64'") 
  endif () 
endif (INTEL_CC) 

project("libtrid")

if (INTEL_CC) 
  # Set compiler flags/options for Intel compiler
  #set(FLAGS_INTEL_DEFAULT "-xCORE_AVX2 -parallel -qopenmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par -opt-report-phase=offload")
  set(FLAGS_INTEL_DEFAULT "-O3 -xMIC-AVX512 -xHost -m64 -parallel -qopenmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report-file=stdout") #-xCORE_AVX2 
  set(FLAGS_INTEL_DEBUG   "-O3 -xMIC-AVX512 -xHost -m64 -parallel -qopenmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report-file=stdout -g")
  set(FLAGS_INTEL_RELEASE "-O3 -xMIC-AVX512 -xHost -m64 -parallel -qopenmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report-file=stdout") #-fp-model fast 

  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} 			${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_INTEL_DEBUG}  ")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}		 	${FLAGS_INTEL_RELEASE}") 

  set(CMAKE_C_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_INTEL_DEBUG}  ")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}		 	${FLAGS_INTEL_RELEASE}") 
else (INTEL_CC) 
  # Set compiler flags/options for GCC compiler
  set(FLAGS_GCC_DEFAULT "-mavx -fopenmp") #-march=core-avx2
  set(FLAGS_GCC_DEBUG   "-g -O0")
  set(FLAGS_GCC_RELEASE "-O3 -flto -fstrict-aliasing -finline-functions") # -ffast-math -fargument-noalias -fargument-noalias-global

  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_GCC_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_GCC_DEBUG}  ")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}     ${FLAGS_GCC_RELEASE}") 

  set(CMAKE_C_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_GCC_DEFAULT} -std=c99")
  set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_GCC_DEBUG}   -std=c99")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}     ${FLAGS_GCC_RELEASE} -std=c99") 
endif (INTEL_CC) 

# Define maximal length of a scalar tridiagonal system 
set(N_MAX 1024 CACHE STRING "Maximal length of the internal buffer for storing intermediate c and d vectors of the Thomas algorithm") 
add_definitions(-DN_MAX=${N_MAX}) 

# Invoke subprojects
add_subdirectory(src/cpu)
add_subdirectory(src/cuda)
