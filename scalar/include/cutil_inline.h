#ifndef __CUTIL_INLINE_H
#define __CUTIL_INLINE_H

//
// stripped-down version of cutil_inline_runtime.h from CUDA 3.2 SDK
//

// CUDA headers
#include <cuda.h>
#include <cufft.h>
#include <cublas.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

// System headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define cudaSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
#define cufftSafeCall(err)  __cufftSafeCall(err,__FILE__,__LINE__)
#define cublasSafeCall(err) __cublasSafeCall(err,__FILE__,__LINE__)
#define cudaCheckMsg(msg)  __cudaCheckMsg(msg,__FILE__,__LINE__)

inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}

inline void __cufftSafeCall(cufftResult err,
                            const char *file, const int line){
  if(CUFFT_SUCCESS != err) {
    printf("%s(%i) : cufftSafeCall() CUFFT error.\n", file, line);
    exit(-1);
  }
}

inline void __cublasSafeCall(cublasStatus err,
                            const char *file, const int line){
  if(CUBLAS_STATUS_SUCCESS != err) {
    printf("%s(%i) : cublasSafeCall() CUBLAS error.\n", file, line);
    exit(-1);
  }
}

inline void __cudaCheckMsg(const char *errorMessage,
                            const char *file, const int line) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    printf("%s(%i) : cudaCheckMsg() error : %s : %s.\n",
           file, line, errorMessage, cudaGetErrorString(err) );
    exit(-1);
  }
}

// this one is modified from its original form

inline void cutilDeviceInit(int argc, char **argv) {
  int            dev, deviceCount;
  cudaDeviceProp devProp;

  // optional selection of CUDA device dependent on Makefile

#ifdef CUDA_DEVICE
  printf("\nSetting CUDA device %d\n",CUDA_DEVICE);
  cudaSetDevice(CUDA_DEVICE);
#endif

  cudaSafeCall(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("cutil error: no devices supporting CUDA\n");
    exit(-1);
  }
  cudaSafeCall(cudaFree(0));
  cudaSafeCall(cudaGetDevice(&dev));
  cudaSafeCall(cudaGetDeviceProperties(&devProp,dev));
  printf("\nUsing CUDA device %d: %s\n\n", dev,devProp.name);
}

#endif







// #ifndef __CUTIL_INLINE_H
// #define __CUTIL_INLINE_H

// //
// // stripped-down version of cutil_inline_runtime.h from CUDA 3.2 SDK
// //

// #include <cuda.h>
// #include <cufft.h>
// #include <cublas.h>
// #include <cuda_runtime_api.h>
// #include <math_constants.h>


// #define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
// #define cufftSafeCall(err)  __cufftSafeCall(err,__FILE__,__LINE__)
// #define cublasSafeCall(err) __cublasSafeCall(err,__FILE__,__LINE__)
// #define cutilCheckMsg(msg)  __cutilCheckMsg(msg,__FILE__,__LINE__)

// inline void __cudaSafeCall(cudaError err,
//                            const char *file, const int line){
//   if(cudaSuccess != err) {
//     printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
//            file, line, cudaGetErrorString(err) );
//     exit(-1);
//   }
// }

// inline void __cufftSafeCall(cufftResult err,
//                             const char *file, const int line){
//   if(CUFFT_SUCCESS != err) {
//     printf("%s(%i) : cufftSafeCall() CUFFT error.\n", file, line);
//     exit(-1);
//   }
// }

// inline void __cublasSafeCall(cublasStatus err,
//                             const char *file, const int line){
//   if(CUBLAS_STATUS_SUCCESS != err) {
//     printf("%s(%i) : cublasSafeCall() CUBLAS error.\n", file, line);
//     exit(-1);
//   }
// }

// inline void __cutilCheckMsg(const char *errorMessage,
//                             const char *file, const int line) {
//   cudaError_t err = cudaGetLastError();
//   if( cudaSuccess != err) {
//     printf("%s(%i) : cutilCheckMsg() error : %s : %s.\n",
//            file, line, errorMessage, cudaGetErrorString(err) );
//     exit(-1);
//   }
// }

// // this one is modified from its original form

// inline void cutilDeviceInit(int argc, char **argv) {
//   int            dev, deviceCount;
//   cudaDeviceProp devProp;

//   // optional selection of CUDA device dependent on Makefile

// #ifdef CUDA_DEVICE
//   printf("\n Setting CUDA device %d\n",CUDA_DEVICE);
//   cudaSetDevice(CUDA_DEVICE);
// #endif

//   cutilSafeCall(cudaGetDeviceCount(&deviceCount));
//   if (deviceCount == 0) {
//     printf("cutil error: no devices supporting CUDA\n");
//     exit(-1);
//   }

//   cutilSafeCall(cudaGetDevice(&dev));
//   cutilSafeCall(cudaGetDeviceProperties(&devProp,dev));
//   printf("\n Using CUDA device %d: %s\n\n", dev,devProp.name);
// }

// #endif
