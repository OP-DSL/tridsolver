cmake_minimum_required(VERSION 2.8.8)

option(INTEL_CC "Use the Intel compiler." ON) 
option(BUILD_FOR_CPU "Build library for CPU architecture." ON) 
option(BUILD_FOR_MIC "Build library for MIC architecture." OFF) 
option(BUILD_FOR_GPU "Build library for GPU architecture." ON) 

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

project("block")

if (INTEL_CC) 
  # Set compiler flags/options for Intel compiler
  #set(FLAGS_INTEL_DEFAULT "-xCORE_AVX2 -parallel -openmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par -opt-report-phase=offload")
  set(FLAGS_INTEL_DEFAULT "-xAVX -m64 -parallel -openmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par") #-xCORE_AVX2 
  set(FLAGS_INTEL_DEBUG   "-g -O0")
  set(FLAGS_INTEL_RELEASE "-O3 -ip -restrict") #-fp-model fast 

  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_INTEL_DEBUG}  ")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}     ${FLAGS_INTEL_RELEASE}") 

  set(CMAKE_C_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${FLAGS_INTEL_DEBUG}  ")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}     ${FLAGS_INTEL_RELEASE}") 
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
set(N_MAX 1024 CACHE STRING "Maximal lenght of the internal buffer for storing intermediate c and d vectors of the Thomas algorithm") 
add_definitions(-DN_MAX=${N_MAX}) 

if (BUILD_FOR_CPU)
  add_library(tridblkcpu SHARED ${PROJECT_SOURCE_DIR}/src/blktrid_cpu.cpp)
  target_include_directories(tridblkcpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
  #target_compile_options(tridblkcpu PRIVATE -xAVX -m64 -parallel -openmp -no-offload)
  #set_target_properties(tridblkcpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
  install(TARGETS tridblkcpu 
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)
  install(FILES ${PROJECT_SOURCE_DIR}/include/blktrid_cpu.h 
                ${PROJECT_SOURCE_DIR}/include/blktrid_common.h 
                ${PROJECT_SOURCE_DIR}/include/blktrid_util.h 
                ${PROJECT_SOURCE_DIR}/include/cutil_inline.h
          DESTINATION ${CMAKE_BINARY_DIR}/include)
endif (BUILD_FOR_CPU)

if (BUILD_FOR_GPU)
  find_package(CUDA REQUIRED)
  include(FindCUDA)

  set(CUDA_64_BIT_DEVICE_CODE ON) # -m64 compiler option
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  #$(NVCC) $(NVCCFLAGS) $(INCS) $(INCS_CUDA) $(INCS_MKL) $(ADI_PARAMS) -c blktrid_gpu.cu          $(LIBS) $(LIBS_CUDA) $(LIBS_MKL) -o blktrid_gpu.o

  cuda_include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/../scalar/src/cuda/ ${PROJECT_SOURCE_DIR}/../scalar/src/cuda/generics/ ${PROJECT_SOURCE_DIR}/../scalar/include )

  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS}         -arch sm_35 --ptxas-options=-v -Xptxas -dlcm=cg ")
  set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false                ")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3                                             ")

  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
  cuda_add_library(tridblkcuda ${PROJECT_SOURCE_DIR}/src/blktrid_gpu.cu SHARED OPTIONS -Xcompiler -fPIC) # Make binary relocatable

  #set_target_properties(tridblkcuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

  install(TARGETS tridblkcuda    
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)
  install(FILES ${PROJECT_SOURCE_DIR}/include/blktrid_gpu.h 
                ${PROJECT_SOURCE_DIR}/include/blktrid_common.h 
                ${PROJECT_SOURCE_DIR}/include/blktrid_util.h 
                ${PROJECT_SOURCE_DIR}/include/cutil_inline.h
    DESTINATION ${CMAKE_BINARY_DIR}/include)
endif (BUILD_FOR_GPU)
