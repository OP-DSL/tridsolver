cmake_minimum_required(VERSION 2.8.8)

if (BUILD_FOR_GPU)
	find_package(CUDA REQUIRED)
	include(FindCUDA)

	set(CUDA_64_BIT_DEVICE_CODE ON) # -m64 compiler option
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)

	set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS}         -arch sm_35 --ptxas-options=-v -Xptxas -dlcm=cg ")
	set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false                ")
	set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3                                             ")

	cuda_include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/cuda/ ${PROJECT_SOURCE_DIR}/src/cuda/generics/)
	set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
	cuda_add_library(tridcuda ${PROJECT_SOURCE_DIR}/src/cuda/trid_cuda.cu SHARED OPTIONS -Xcompiler -fPIC) # Make binary relocatable

	install(TARGETS tridcuda    
		      LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
		      ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/lib)

	install(FILES ${PROJECT_SOURCE_DIR}/include/trid_cuda.h 
		            ${PROJECT_SOURCE_DIR}/include/trid_common.h
		            ${PROJECT_SOURCE_DIR}/include/trid_util.h 
		            ${PROJECT_SOURCE_DIR}/include/cutil_inline.h    
		      DESTINATION ${CMAKE_BINARY_DIR}/include)
endif (BUILD_FOR_GPU)