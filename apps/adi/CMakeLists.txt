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

project("adi")

if (INTEL_CC) 
  # Set compiler flags/options for Intel compiler
  set(FLAGS_INTEL_DEFAULT "-xCORE_AVX2 -parallel -openmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par -opt-report-phase=offload")
  #set(FLAGS_INTEL_DEFAULT "-xAVX -m64 -parallel -openmp -no-offload -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par") #-xCORE_AVX2 
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


#set(CMAKE_GENERATOR_TOOLSET "Intel C++ Compiler XE 15.0" CACHE STRING "Platform Toolset" FORCE)

set(LIBTRID_PATH "/home/mgiles/mudalige/TriDiagonal/Tridiag-Git/tridsolver/scalar/build")

set(N_MAX 1024 CACHE STRING "Maximum length of the internal buffer for storing intermediate c and d vectors of the Thomas algorithm") 
add_definitions(-DN_MAX=${N_MAX}) 

set(FPPREC 1 CACHE STRING "0 - single precision build; 1 - double precision build") 
add_definitions(-DFPPREC=${FPPREC}) 

#include_directories(${LIBTRID_PATH}/scalar/include ./include ../libtrid/ ../libtrid/src )

if (BUILD_FOR_CPU)
  add_executable(adi_cpu src/adi_cpu.cpp)
  #target_compile_options(adi_cpu PRIVATE -xAVX -parallel -openmp -no-offload)
  target_include_directories(adi_cpu PRIVATE ${LIBTRID_PATH}/include ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(adi_cpu PRIVATE -L${LIBTRID_PATH}/lib -ltridcpu -L${INTEL_PATH}/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts)

  #/opt/intel/composer_xe_2015.3.187/bin/intel64/icc -O3 -xAVX -ip -restrict -parallel -fopenmp -openmp-report2 -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par  -DFPPREC=0  -DN_MAX=1024  -I./include -I./libtrid  -I./include -I./libtrid -I/opt/intel/composer_xe_2015.3.187/mkl/include   -D__LAPACKE__ src/adi_cpu.cpp -L/opt/intel/composer_xe_2015.3.187/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L/opt/intel/composer_xe_2015.3.187/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts -o adi_mkl
  add_executable(adi_mkl src/adi_cpu.cpp)
  target_compile_definitions(adi_mkl PRIVATE -D__MKL__)
  target_compile_options(adi_mkl PRIVATE -xAVX -parallel -openmp -no-offload)
  target_include_directories(adi_mkl PRIVATE ${INTEL_PATH}/mkl/include ${LIBTRID_PATH}/include ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(adi_mkl -L${LIBTRID_PATH}/lib -ltridcpu -L${INTEL_PATH}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts)

endif (BUILD_FOR_CPU)

if (BUILD_FOR_MIC)
  add_executable(adi_mic_offload src/adi_cpu.cpp)
  target_link_libraries(adi_mic_offload ${LIBTRID_PATH}/lib/libtridmic_offload.so)
  target_include_directories(adi_mic_offload PRIVATE ../libtrid/include)
  target_compile_options(adi_mic_offload PRIVATE -xAVX -parallel -openmp -opt-report-phase=offload -offload-attribute-target=mic -mkl -offload-option,mic,ld,"-L${INTEL_PATH}/mkl/lib/mic -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/mic -limf -lintlc -lsvml -lirng -liomp5 -loffload -lcilkrts" -I./include -I./libtrid  -D__OFFLOAD__ -L./libtrid/lib -limf -lintlc -lsvml -lirng)

  add_executable(adi_mic_offload_mkl src/adi_cpu.cpp)
  target_compile_definitions(adi_mic_offload_mkl PRIVATE -D__MKL__)
  target_compile_options(adi_mic_offload_mkl PRIVATE -xAVX -parallel -openmp -offload)
  target_include_directories(adi_mic_offload_mkl PRIVATE -I/opt/intel/composer_xe_2015.3.187/mkl/include)
  target_link_libraries(adi_mic_offload_mkl ${LIBTRID_PATH}/lib/libtridcpu.so -L${INTEL_PATH}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts)



  add_executable(adi_mic_native src/adi_cpu.cpp)
  #target_compile_definitions(adi_mic_native PRIVATE -D__MKL__)
  target_compile_options(adi_mic_native PRIVATE -mmic -parallel -openmp )
  #target_include_directories(adi_mic_native PRIVATE -I${INTEL_PATH}/mkl/include)
  target_link_libraries(adi_mic_native ${LIBTRID_PATH}/lib/libtridmic_native.so)


  add_executable(adi_mic_native_mkl src/adi_cpu.cpp)
  target_compile_definitions(adi_mic_native_mkl PRIVATE -D__MKL__)
  target_compile_options(adi_mic_native_mkl PRIVATE -mmic -parallel -openmp )
  target_include_directories(adi_mic_native_mkl PRIVATE -I${INTEL_PATH}/mkl/include)
  target_link_libraries(adi_mic_native_mkl ${LIBTRID_PATH}/lib/libtridmic_native.so -L./libtrid/lib -limf -lintlc -lsvml -lirng) #-L$(INTEL_PATH)/mkl/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_rt -liomp5 -lpthread -lm -lgfortran)
endif (BUILD_FOR_MIC)

if (BUILD_FOR_GPU)
  include(FindCUDA)
  #find_package(CUDA)

  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS         "-arch sm_35 --ptxas-options=-v -Xptxas -dlcm=cg")
  set(CUDA_NVCC_FLAGS_DEBUG   "-g -G -O0 -ftz=false -fmad=false")
  set(CUDA_NVCC_FLAGS_RELEASE "-O3")

  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
  cuda_include_directories(${LIBTRID_PATH}/include ${PROJECT_SOURCE_DIR}/include)
  cuda_add_executable(adi_cuda src/adi_cuda.cu src/preproc.cu)
  target_link_libraries(adi_cuda -L${LIBTRID_PATH}/lib -ltridcuda)
endif (BUILD_FOR_GPU)



#INCS_MKL = -I./include -I./libtrid -I$(INTEL_PATH)/mkl/include  
#LIBS_MKL = -L$(INTEL_PATH)/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L$(INTEL_PATH)/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts
#LIBS_MKL = $(INTEL_PATH)/mkl/lib/intel64/libmkl_intel_lp64.a \
#	   $(INTEL_PATH)/mkl/lib/intel64/libmkl_intel_thread.a \
#	   $(INTEL_PATH)/mkl/lib/intel64/libmkl_core.a

#INCS_MKL_MIC = -I./include -I./libtrid -I$(INTEL_PATH)/mkl/include  
#LIBS_MKL_MIC = -L$(INTEL_PATH)/mkl/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L$(INTEL_PATH)/compiler/lib/mic -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts

#INCS_PHI = -I./include -I./libtrid 
#LIBS_PHI = -L./libtrid/lib -limf -lintlc -lsvml -lirng #-ltrid_phi

#INCS_PHI_LAPACKE = -I./include -I./libtrid -I$(INTEL_PATH)/mkl/include
#LIBS_PHI_LAPACKE = -L./libtrid/lib -limf -lintlc -lsvml -lirng -L$(INTEL_PATH)/mkl/lib/mic -lmkl_core -lmkl_rt -lmkl_lapack95_ilp64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lgfortran #-ltrid_phi
#LIBS_PHI_LAPACKE = -L./libtrid/lib -limf -lintlc -lsvml -lirng -L$(INTEL_PATH)/mkl/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_rt -liomp5 -lpthread -lm -lgfortran #-ltrid_phi
#LIBS_PHI_LAPACKE = -L./libtrid/lib -limf -lintlc -lsvml -lirng #-L$(INTEL_PATH)/mkl/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_rt -liomp5 -lpthread -lm -lgfortran #-ltrid_phi





#target_link_libraries(adi_cpu tridcpu)
add_executable(adi_orig ${PROJECT_SOURCE_DIR}/src/adi_orig.c)

add_executable(compare ${PROJECT_SOURCE_DIR}/src/compare.c)
target_include_directories(compare PRIVATE ${LIBTRID_PATH}/include)



#install(TARGETS libtridcuda DESTINATION lib)
#install(FILES src/cuda/trid_cuda.h DESTINATION include)
