cmake_minimum_required(VERSION 3.18)
project("adi" C CXX)

option(BUILD_FOR_CPU "Build library for CPU architecture." OFF)
option(BUILD_FOR_MIC "Build library for MIC architecture." OFF)
option(BUILD_FOR_GPU "Build library for GPU architecture." OFF)
option(BUILD_FOR_MPI "Build library for MPI combnied with the above architectures." OFF)
option(USE_NCCL "TDMA library built with NCCL" OFF)
set(LIBTRID_PATH
    $ENV{TRIDSOLVER_INSTALL_PATH}
    CACHE STRING "Tridsolver (scalar) installation directory")

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
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}       ${FLAGS_GCC_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_RELEASE}     ${FLAGS_GCC_RELEASE}")

  set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS}         ${FLAGS_GCC_DEFAULT}")
  set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}   -O0")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_RELEASE}       ${FLAGS_GCC_RELEASE}")
  # cmake-format: on
endif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")

set(FPPREC
    1
    CACHE STRING "0 - single precision build; 1 - double precision build")
set_property(CACHE FPPREC PROPERTY STRINGS 0 1)
add_definitions(-DFPPREC=${FPPREC})

if (BUILD_FOR_CPU)
  find_package(OpenMP REQUIRED)
  find_package(
    libtrid
    REQUIRED
    PATHS
    ${LIBTRID_PATH})
  # the tridsolver config file should set TRIDSOLVER_N_MAX value
  add_definitions(-DN_MAX=${TRIDSOLVER_N_MAX})
  add_executable(adi_cpu src/adi_cpu.cpp)
  target_include_directories(adi_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(adi_cpu PRIVATE libtrid::tridcpu OpenMP::OpenMP_CXX)

  # cmake-format: off
  add_executable(adi_mkl src/adi_cpu.cpp)
  target_compile_definitions(adi_mkl PRIVATE -D__MKL__)
  target_compile_options(adi_mkl PRIVATE -xAVX -parallel -qopenmp -no-offload)
  target_include_directories(adi_mkl PRIVATE ${INTEL_PATH}/mkl/include ${LIBTRID_PATH}/include ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(adi_mkl -L${LIBTRID_PATH}/lib -ltridcpu -L${INTEL_PATH}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts)
  # cmake-format: on

  if (BUILD_FOR_MPI)
    find_package(MPI REQUIRED)
    add_executable(adi_mpi_cpu src/adi_mpi_cpu.cpp)
    target_include_directories(adi_mpi_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(adi_mpi_cpu PRIVATE libtrid::tridcpu_mpi OpenMP::OpenMP_CXX MPI::MPI_CXX)
  endif (BUILD_FOR_MPI)

endif (BUILD_FOR_CPU)

# cmake-format: off
if (BUILD_FOR_MIC)
  add_executable(adi_mic_offload src/adi_cpu.cpp)
  target_link_libraries(adi_mic_offload ${LIBTRID_PATH}/lib/libtridmic_offload.so)
  target_include_directories(adi_mic_offload PRIVATE ../libtrid/include)
  target_compile_options(adi_mic_offload PRIVATE -xAVX -parallel -qopenmp -opt-report-phase=offload -offload-attribute-target=mic -mkl -offload-option,mic,ld,"-L${INTEL_PATH}/mkl/lib/mic -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/mic -limf -lintlc -lsvml -lirng -liomp5 -loffload -lcilkrts" -I./include -I./libtrid  -D__OFFLOAD__ -L./libtrid/lib -limf -lintlc -lsvml -lirng)

  add_executable(adi_mic_offload_mkl src/adi_cpu.cpp)
  target_compile_definitions(adi_mic_offload_mkl PRIVATE -D__MKL__)
  target_compile_options(adi_mic_offload_mkl PRIVATE -xAVX -parallel -qopenmp -offload)
  target_include_directories(adi_mic_offload_mkl PRIVATE -I/opt/intel/composer_xe_2015.3.187/mkl/include)
  target_link_libraries(adi_mic_offload_mkl ${LIBTRID_PATH}/lib/libtridcpu.so -L${INTEL_PATH}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L${INTEL_PATH}/compiler/lib/intel64 -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts)



  add_executable(adi_mic_native src/adi_cpu.cpp)
  #target_compile_definitions(adi_mic_native PRIVATE -D__MKL__)
  target_compile_options(adi_mic_native PRIVATE -mmic -parallel -qopenmp )
  #target_include_directories(adi_mic_native PRIVATE -I${INTEL_PATH}/mkl/include)
  target_link_libraries(adi_mic_native ${LIBTRID_PATH}/lib/libtridmic_native.so)


  add_executable(adi_mic_native_mkl src/adi_cpu.cpp)
  target_compile_definitions(adi_mic_native_mkl PRIVATE -D__MKL__)
  target_compile_options(adi_mic_native_mkl PRIVATE -mmic -parallel -qopenmp )
  target_include_directories(adi_mic_native_mkl PRIVATE -I${INTEL_PATH}/mkl/include)
  target_link_libraries(adi_mic_native_mkl ${LIBTRID_PATH}/lib/libtridmic_native.so -L./libtrid/lib -limf -lintlc -lsvml -lirng) #-L$(INTEL_PATH)/mkl/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_rt -liomp5 -lpthread -lm -lgfortran)
endif (BUILD_FOR_MIC)
# cmake-format: on

if (BUILD_FOR_GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
  else (CMAKE_CUDA_COMPILER)
    message(ERROR "Couldn't find CUDA compiler")
  endif (CMAKE_CUDA_COMPILER)

  find_package(
    libtrid
    REQUIRED
    PATHS
    ${LIBTRID_PATH})
  find_package(CUDAToolkit)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xptxas -dlcm=cg")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}   -g -G -O0 ")

  add_executable(adi_cuda std/adi_cuda.cu std/preproc.cu)
  target_include_directories(adi_cuda ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(adi_cuda libtrid::tridcuda CUDA::cublas)

  find_package(MPI REQUIRED)

  if (USE_NCCL)
    find_package(NCCL)

    if (NCCL_FOUND)
      add_definitions(-DTRID_NCCL)
      include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
    endif ()
  endif ()
  add_executable(adi_mpi_cuda src/adi_mpi_cuda.cu)
  target_include_directories(adi_mpi_cuda ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(
    adi_mpi_cuda
    libtrid::trid_cuda_mpi
    MPI::MPI_CXX
    CUDA::cublas
    ${NCCL_LIBRARY})
endif (BUILD_FOR_GPU)

# cmake-format: off
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
# cmake-format: on
