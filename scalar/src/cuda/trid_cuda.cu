/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Endre László and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Endre László may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Endre László ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Endre László BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk,
// 2013-2014

#include "trid_common.h"
#include "trid_cuda.h"

//#define WARP_SIZE 32
//#define ALIGN 32               // 32 byte alignment is required
//#define ALIGN_FLOAT  (ALIGN/4) // 32 byte/ 4bytes/float = 8
////#define ALIGN_DOUBLE (ALIGN/2) // 32 byte/ 8bytes/float = 4
//#define ALIGN_DOUBLE (ALIGN/8) // 32 byte/ 8bytes/float = 4
//
//#define MAXDIM 8 // Maximal dimension that can be used in the library. Defines
//static arrays
//
//#define CUDA_ALIGN_BYTE 32 // 32 byte alignment is used on CUDA-enabled GPUs

// CUDA constant memory setup for multidimension related index calculations

#include "trid_linear.hpp"
#include "trid_linear_shared.hpp"
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if
//register shuffle intrinsics exist on the device
#include "trid_linear_reg16_float4.hpp"
#include "trid_linear_reg8_double2.hpp"
#include "trid_thomaspcr.hpp"
//#endif
//#include "trid_cusparse.hpp"
#include "trid_strided_multidim.hpp"

#include "cutil_inline.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

// cusparseHandle_t handle_sp; // Handle for cuSPARSE setup





//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if
//register shuffle intrinsics exist on the device
template <typename REAL>
void trid_linear_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *d_ax,
                     const REAL *d_bx, const REAL *d_cx, REAL *d_du, REAL *d_u,
                     int sys_size, int sys_pads, int sys_n) {
  if (sizeof(REAL) == 4) {
    trid_linear_reg16_float4<<<dimGrid_x, dimBlock_x>>>(
        (float *)d_ax, (float *)d_bx, (float *)d_cx, (float *)d_du,
        (float *)d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_reg16_float4 execution failed\n");
  } else if (sizeof(REAL) == 8) {
    trid_linear_reg8_double2<<<dimGrid_x, dimBlock_x>>>(
        (double *)d_ax, (double *)d_bx, (double *)d_cx, (double *)d_du,
        (double *)d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_reg8_double2 execution failed\n");
  }
}
//#endif

//
// Tridiagonal solver for linear (coniguous) system layout. Optimization options
// may be used to select optimization algorithm
//
template <typename REAL, int INC>
void trid_linearlayout_cuda(const REAL **d_ax, const REAL **d_bx,
                            const REAL **d_cx, REAL **d_du, REAL **d_u,
                            int sys_size, int sys_pads, int sys_n, int opt) {
  // int sys_stride = 1; // Linear layout -> stride = 1

  // Set up the execution configuration
  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx = dimgrid % 65536;            // can go up to max 65535 on Fermi
  int dimgridy = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  int shared_mem_size = ((8 + 1) * dimBlock_x.x * dimBlock_x.y) * sizeof(REAL);

  // Arguments for solveBatchedTrid() function call
  int numTrids = sys_n;
  int length = sys_size;
  int stride1 = 1;
  int stride2 = sys_size;
  int subBatchSize = numTrids;
  int subBatchStride = 0;

  switch (opt) {
  case 1:
    trid_linear<REAL,INC><<<dimGrid_x, dimBlock_x>>>(
        *d_ax, *d_bx, *d_cx, *d_du, *d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear execution failed\n");
    break;
  case 2:
    trid_linear_shared<REAL><<<dimGrid_x, dimBlock_x, shared_mem_size>>>(
        *d_ax, *d_bx, *d_cx, *d_du, *d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_shared execution failed\n");
    break;
  case 3:
    //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if
    //register shuffle intrinsics exist on the device
    trid_linear_reg<REAL>(dimGrid_x, dimBlock_x, *d_ax, *d_bx, *d_cx, *d_du,
                          *d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_reg16_float4 execution failed\n");
    //#else
    //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
    //#endif
    break;
  case 4:
    //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if
    //register shuffle intrinsics exist on the device
    // trid_linear_thomaspcr<REAL, WARP_SIZE>(*d_ax, *d_bx, *d_cx, *d_du, *d_u,
    // sys_size, sys_pads, sys_n);
    // solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2,
    // subBatchSize, subBatchStride, *d_ax, *d_bx, *d_cx, *d_du, *d_u);
    solveBatchedTrid<REAL, INC>(numTrids, length, stride1, stride2,
                                subBatchSize, subBatchStride, *d_ax, *d_bx,
                                *d_cx, *d_du, *d_du);
    cudaCheckMsg("trid_linear_thomaspcr execution failed\n");
    //#else
    //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
    //#endif
    break;
  case 5:
    // trid_linear_cusparse(handle_sp, d_ax, d_bx, d_cx, d_du, d_u, sys_stride,
    // sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_cusparse execution failed\n");
    break;
  default:
    printf("Wrong optimization argument OPT is given. Exiting.\n");
    exit(-1);
  }
}

//
// Host function for selecting the proper setup for solve in a specific
// dimension
//

template<typename REAL>
void transpose(cublasHandle_t &handle, size_t mRows, size_t nCols, const REAL *in, REAL *out) {
}
template<>
void transpose<float>(cublasHandle_t &handle, size_t mRows, size_t nCols, const float *in, float *out) {
   float alpha = 1.;
   float beta = 0.;
   cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mRows, nCols, &alpha, in, nCols, &beta, in, nCols, out, mRows);
}
template<>
void transpose<double>(cublasHandle_t &handle, size_t mRows, size_t nCols, const double *in, double *out) {
   double alpha = 1.;
   double beta = 0.;
   cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mRows, nCols, &alpha, in, nCols, &beta, in, nCols, out, mRows);
}

template <typename REAL, int INC>
void tridMultiDimBatchSolve(const REAL *d_a, const int *a_pads,
                            const REAL *d_b, const int *b_pads,
                            const REAL *d_c, const int *c_pads,
                            REAL *d_d, const int *d_pads,
                            REAL *d_u, const int *u_pads,
                            int ndim, int solvedim,
                            int *dims, int *opts, int sync) {

  //int sys_n = cumdims[ndim] / dims[solvedim]; // Number of systems to be solved
  int sys_n = 1;
  for ( int i = 0 ; i < ndim ; i++ ) {
     sys_n *= (i == solvedim) ? 1 : dims[i];
  }


  if (solvedim == 0) {

    /* If there is padding in the Y-dimension, this solver will break down */
    bool foundBadPadding = false;
    for ( int i = 1 ; i < ndim ; i++ ) {
        foundBadPadding |= (a_pads[i] != dims[i]);
    }
    if ( foundBadPadding ) {
        printf("CUDA solver for Trid MultiDimBatchSolve is not currently "
                "capable of solving systems with padding in the 'Y' "
                "dimension.\n");
        exit(-1);
    }

    /* Padded size of solve dimension */
    int sys_pads = a_pads[0];
    /* Number of systems to solve */
    int sys_n_lin = 1;
    for ( int i = 1 ; i < ndim ; i++ ) {
        sys_n_lin *= dims[i];
    }

    if ( ndim == 1 || opts[0] != 0) {

        int sys_size = dims[0]; // Size (length) of a system
        int opt = opts[0];
        trid_linearlayout_cuda<REAL, INC>(&d_a, &d_b, &d_c, &d_d, &d_u, sys_size,
                sys_pads, sys_n_lin, opt);
    } else {
        static REAL *aT = NULL;
        static REAL *bT = NULL;
        static REAL *cT = NULL;
        static REAL *dT = NULL;
        static REAL *uT = NULL;
        static int alloc_size = 0;

        static cublasHandle_t handle = 0;

        if ( !handle ) cublasCreate(&handle);

        int size_needed = sizeof(REAL)*sys_pads*sys_n_lin;

        if ( size_needed > alloc_size ) {
            if ( aT ) cudaFree( aT ); cudaMalloc(&aT, size_needed);
            if ( bT ) cudaFree( bT ); cudaMalloc(&bT, size_needed);
            if ( cT ) cudaFree( cT ); cudaMalloc(&cT, size_needed);
            if ( dT ) cudaFree( dT ); cudaMalloc(&dT, size_needed);
            if ( INC ) { if ( uT ) cudaFree( uT ); cudaMalloc(&uT, size_needed); }

            alloc_size = size_needed;
        }

        size_t m = sys_n_lin;  /* Maybe need to swap? */
        size_t n = sys_pads;
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        transpose(handle, m, n, d_a, aT);
        transpose(handle, m, n, d_b, bT);
        transpose(handle, m, n, d_c, cT);
        transpose(handle, m, n, d_d, dT);
        if ( INC) transpose(handle, m, n, d_u, uT);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        assert(ndim <= 3);
        int newDims[MAXDIM] = {sys_n_lin, dims[0]};
        int newPads[MAXDIM] = {sys_n_lin, a_pads[0]}; /* Assumption of no actual padding in original Y (& Z) dimensions */
        int newNumDim = max(ndim-1, 2); /* Linearized Y&Z dimensions, so potentially reduced to 2D problem */
        /* TODO:  Better pads arrays */
        tridMultiDimBatchSolve<REAL, INC>(aT, newPads, bT, newPads, cT, newPads, dT, newPads, uT, newPads, newNumDim, 1, newDims, opts, sync);

        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        if ( INC ) transpose(handle, n, m, uT, d_u);
        else transpose(handle, n, m, dT, d_d);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    }
  } else {
    if (solvedim == 1 && opts[1] == 3) { // If y-solve and ThomasPCR
     
      int numTrids = dims[0] * dims[2];
      int length = dims[1];
      int stride1 = dims[0];
      int stride2 = 1;
      int subBatchSize = dims[0]; // dims[1];
      int subBatchStride = dims[0] * dims[1];
      //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if
      //register shuffle intrinsics exist on the device
      // y: stride1 = x, stride2 = 1, subBatchSize = x, subBatchStride = x*y
      solveBatchedTrid<REAL, INC>(numTrids, length, stride1, stride2,
                                  subBatchSize, subBatchStride, d_a, d_b, d_c,
                                  d_d, d_u);
      // solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2,
      // subBatchSize, subBatchStride, d_a, d_b, d_c, d_d, d_u);
      //#else
      //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
      //#endif
    } else if (solvedim == 2 && opts[2] == 3) { // If z-solve and ThomasPCR
     
      int numTrids = dims[0] * dims[1];
      int length = dims[2];
      int stride1 = dims[0] * dims[1];
      int stride2 = 1;
      int subBatchSize = dims[0] * dims[1];
      int subBatchStride = 0;
      //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if
      //register shuffle intrinsics exist on the device
      // z: stride1 = x*y, stride2 = 1, subBatchSize = x*y, subBatchStride = 0
      solveBatchedTrid<REAL, INC>(numTrids, length, stride1, stride2,
                                  subBatchSize, subBatchStride, d_a, d_b, d_c,
                                  d_d, d_d); //last param was d_d
      //#else
      //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
      //#endif
    } else {
     
      // Test if data is aligned
      long isaligned = 0;
      isaligned =
          (long)d_a % CUDA_ALIGN_BYTE; // Check if base pointers are aligned
      isaligned += (long)d_b % CUDA_ALIGN_BYTE;
      isaligned += (long)d_c % CUDA_ALIGN_BYTE;
      isaligned += (long)d_d % CUDA_ALIGN_BYTE;
      if (d_u != NULL)
        isaligned += (long)d_u % CUDA_ALIGN_BYTE;
      isaligned += (dims[0] * sizeof(REAL)) %
                   CUDA_ALIGN_BYTE; // Check if X-dimension allows alignment

      if (isaligned == 0) { // If any of the above is non-zero, vector loads can not be used
        if (sizeof(REAL) == 4) {
         
          // Kernel launch configuration
          int sys_n_float4 = sys_n / 4;
          int blockdimx_float4 =
              128; // Has to be the multiple of 32(or maybe 4??)
          int blockdimy_float4 = 1;
          int dimgrid_float4 =
              1 + (sys_n_float4 - 1) / blockdimx_float4; // can go up to 65535
          int dimgridx_float4 =
              dimgrid_float4 % 65536; // can go up to max 65535 on Fermi
          int dimgridy_float4 = 1 + dimgrid_float4 / 65536;
          dim3 dimGrid_float4(dimgridx_float4, dimgridy_float4);
          dim3 dimBlock_float4(blockdimx_float4, blockdimy_float4);

          // Setup dimension and padding according to float4 loads/stores
          int new_dims[MAXDIM];
          int new_a_pads[MAXDIM];
          int new_b_pads[MAXDIM];
          int new_c_pads[MAXDIM];
          int new_d_pads[MAXDIM];
          int new_u_pads[MAXDIM];
          memcpy(new_dims, dims, sizeof(int)*ndim);
          memcpy(new_a_pads, a_pads, sizeof(int)*ndim);
          memcpy(new_b_pads, b_pads, sizeof(int)*ndim);
          memcpy(new_c_pads, c_pads, sizeof(int)*ndim);
          memcpy(new_d_pads, d_pads, sizeof(int)*ndim);
          if (INC) memcpy(new_u_pads, u_pads, sizeof(int)*ndim);
          new_dims[0] = dims[0] / 4;
          new_a_pads[0] /= 4;
          new_b_pads[0] /= 4;
          new_c_pads[0] /= 4;
          new_d_pads[0] /= 4;
          new_u_pads[0] /= 4;
          // trid_set_consts(ndim, dims, pads);

          trid_strided_multidim<float, float4, INC>(
              dimGrid_float4, dimBlock_float4,
                  (float4 *)d_a, new_a_pads, (float4 *)d_b, new_b_pads,
                  (float4 *)d_c, new_c_pads, (float4 *)d_d, new_d_pads,
                  (float4 *)d_u, new_u_pads, ndim, solvedim, sys_n_float4, new_dims);

          // trid_set_consts(ndim, dims, a_pads);
        } else if (sizeof(REAL) == 8) {

          // Kernel launch configuration
          int sys_n_double2 = sys_n / 2;
          int blockdimx_double2 =
              128; // Has to be the multiple of 32(or maybe 4??)
          int blockdimy_double2 = 1;
          int dimgrid_double2 =
              1 + (sys_n_double2 - 1) / blockdimx_double2; // can go up to 65535
          int dimgridx_double2 =
              dimgrid_double2 % 65536; // can go up to max 65535 on Fermi
          int dimgridy_double2 = 1 + dimgrid_double2 / 65536;
          dim3 dimGrid_double2(dimgridx_double2, dimgridy_double2);
          dim3 dimBlock_double2(blockdimx_double2, blockdimy_double2);
          // Setup dimension and padding according to double2 loads/stores
          int new_dims[MAXDIM];
          int new_a_pads[MAXDIM];
          int new_b_pads[MAXDIM];
          int new_c_pads[MAXDIM];
          int new_d_pads[MAXDIM];
          int new_u_pads[MAXDIM];
          memcpy(new_dims, dims, sizeof(int)*ndim);
          memcpy(new_a_pads, a_pads, sizeof(int)*ndim);
          memcpy(new_b_pads, b_pads, sizeof(int)*ndim);
          memcpy(new_c_pads, c_pads, sizeof(int)*ndim);
          memcpy(new_d_pads, d_pads, sizeof(int)*ndim);
          if (INC) memcpy(new_u_pads, u_pads, sizeof(int)*ndim);
          new_dims[0] /= 2;
          new_a_pads[0] /= 2;
          new_b_pads[0] /= 2;
          new_c_pads[0] /= 2;
          new_d_pads[0] /= 2;
          new_u_pads[0] /= 2;
          // trid_set_consts(ndim, dims, a_pads);

          trid_strided_multidim<double, double2, INC>(
              dimGrid_double2, dimBlock_double2,
                  (double2 *)d_a, new_a_pads, (double2 *)d_b, new_b_pads,
                  (double2 *)d_c, new_c_pads, (double2 *)d_d, new_d_pads,
                  (double2 *)d_u, new_u_pads, ndim, solvedim,
                  sys_n_double2, new_dims);

          // trid_set_consts(ndim, dims, pads);
        }
      } else {

        // Kernel launch configuration
        int blockdimx = 128; // Has to be the multiple of 32(or maybe 4??)
        int blockdimy = 1;
        int dimgrid = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
        int dimgridx = dimgrid % 65536; // can go up to max 65535 on Fermi
        int dimgridy = 1 + dimgrid / 65536;
        dim3 dimGrid(dimgridx, dimgridy);
        dim3 dimBlock(blockdimx, blockdimy);

        trid_strided_multidim<REAL, REAL, INC>(dimGrid, dimBlock,
            d_a, a_pads, d_b, b_pads, d_c, c_pads, d_d, d_pads, d_u, u_pads,
            ndim, solvedim, sys_n, dims);
      }
    }
    //      break;
    //    default:
    //      printf("Wrong optimization argument OPT is given. Exiting.\n");
    //      exit(-1);
  }
  if (sync == 1)
    cudaSafeCall(cudaDeviceSynchronize());
  
}

//------------------------------------------------------------------------------------------------------------------
//
// API calls
//
//------------------------------------------------------------------------------------------------------------------

// tridStatus_t tridSgtsvStridedBatch(int sys_size, const float* a, const float
// *b, const float *c, float *d, int num_sys, int sys_stride, int opt) {
//  trid_linearlayout_cuda<float,0>(a, b, c, d, NULL, sys_size, sys_stride,
//  num_sys, opt);
//  return TRID_STATUS_SUCCESS;
//}
//
// tridStatus_t tridDgtsvStridedBatch(int sys_size, const double* a, const
// double *b, const double *c, double *d, int num_sys, int sys_stride, int opt)
// {
//  trid_linearlayout_cuda<double,0>(a, b, c, d, NULL, sys_size, sys_stride,
//  num_sys, opt);
//  return TRID_STATUS_SUCCESS;
//}

// tridStatus_t tridCgtsvStridedBatch(int sys_size, const complexf* a, const
// complexf *b, const complexf *c, complexf *d, int num_sys, int sys_stride) {
//  trid_linearlayout_cuda<complexf,0>(a, b, c, d, NULL, sys_size, sys_stride,
//  num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

// tridStatus_t tridZgtsvStridedBatch(int sys_size, const complexd* a, const
// complexd *b, const complexd *c, complexd *d, int num_sys, int sys_stride) {
//  trid_linearlayout_cuda<complexd,0>(a, b, c, d, NULL, sys_size, sys_stride,
//  num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

// tridStatus_t tridSgtsvStridedBatchInc(int sys_size, const float* a, const
// float *b, const float *c, float *d, float *u, int num_sys, int sys_stride,
// int* opts) {
//  trid_linearlayout_cuda<float,1>(a, b, c, d, u, sys_size, sys_stride,
//  num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}
//
// tridStatus_t tridDgtsvStridedBatchInc(int sys_size, const double* a, const
// double *b, const double *c, double *d, double *u, int num_sys, int
// sys_stride, int* opts) {
//  trid_linearlayout_cuda<double,1>(a, b, c, d, u, sys_size, sys_stride,
//  num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

// tridStatus_t tridCgtsvStridedBatchInc(int sys_size, const complexf* a, const
// complexf *b, const complexf *c, complexf *d, complexf *u, int num_sys, int
// sys_stride) {
//  trid_linearlayout_cuda<complexf,1>(a, b, c, d, u, sys_size, sys_stride,
//  num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

// tridStatus_t tridZgtsvStridedBatchInc(int sys_size, const complexd* a, const
// complexd *b, const complexd *c, complexd *d, complexd *u, int num_sys, int
// sys_stride) {
//  trid_linearlayout_cuda<complexd,1>(a, b, c, d, u, sys_size, sys_stride,
//  num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}


tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b,
                                   const float *c, float *d, float *u, int ndim,
                                   int solvedim, int *dims, int *pads,
                                   int *opts, int sync) {
  tridMultiDimBatchSolve<float, 0>(a, pads, b, pads, c, pads, d, pads, NULL, pads,
                                     ndim, solvedim, dims, opts, 1);
  return TRID_STATUS_SUCCESS;
}


tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b,
                                   const double *c, double *d, double *u,
                                   int ndim, int solvedim, int *dims, int *pads,
                                   int *opts, int sync) {
  tridMultiDimBatchSolve<double, 0>(a, pads, b, pads, c, pads, d, pads, NULL, pads,
                                    ndim, solvedim, dims, opts, 1);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchPadded(const double *a, const int *a_pads,
                                   const double *b, const int *b_pads,
                                   const double *c, const int *c_pads,
                                   double *d, const int *d_pads,
                                   double *u, const int *u_pads,
                                   int ndim, int solvedim, int *dims,
                                   int *opts, int sync) {
  tridMultiDimBatchSolve<double, 0>(a, a_pads, b, b_pads, c, c_pads, d, d_pads, NULL, u_pads,
                                    ndim, solvedim, dims, opts, 1);
  return TRID_STATUS_SUCCESS;
}

// tridStatus_t tridCmtsvStridedBatch(int ndim, int* sys_size, const complexf*
// a, const complexf *b, const complexf *c, complexf *d, int *sys_stride, int
// solvedim) {
//  tridMultiDimBatchSolve<complexf,0>(a, b, c, d, NULL, ndim, solvedim,
//  sys_size, sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}
//
// tridStatus_t tridZmtsvStridedBatch(int ndim, int* sys_size, const complexd*
// a, const complexd *b, const complexd *c, complexd*d, int *sys_stride, int
// solvedim) {
//  tridMultiDimBatchSolve<complexd,0>(a, b, c, d, NULL, ndim, solvedim,
//  sys_size, sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}


tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads, int *opts, int sync) {
  tridMultiDimBatchSolve<float, 1>(a, pads, b, pads, c, pads, d, pads, u, pads,
                                     ndim, solvedim, dims, opts, 1);
  return TRID_STATUS_SUCCESS;
}


tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads, int *opts, int sync) {
  tridMultiDimBatchSolve<double, 1>(a, pads, b, pads, c, pads, d, pads, u, pads,
                                    ndim, solvedim, dims, opts, 1);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchPaddedInc(const double *a, const int *a_pads,
                                   const double *b, const int *b_pads,
                                   const double *c, const int *c_pads,
                                   double *d, const int *d_pads,
                                   double *u, const int *u_pads,
                                   int ndim, int solvedim, int *dims,
                                   int *opts, int sync) {
  tridMultiDimBatchSolve<double, 1>(a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads,
                                    ndim, solvedim, dims, opts, 1);
  return TRID_STATUS_SUCCESS;
}

// tridStatus_t tridCmtsvStridedBatchInc(int ndim, int* sys_size, const
// complexf* a, const complexf *b, const complexf *c, complexf *d, complexf *u,
// int *sys_stride, int solvedim) {
//  tridMultiDimBatchSolve<complexf,1>(a, b, c, d, u, ndim, solvedim, sys_size,
//  sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}
//
// tridStatus_t tridZmtsvStridedBatchInc(int ndim, int* sys_size, const
// complexd* a, const complexd *b, const complexd *c, complexd *d, complexd *u,
// int *sys_stride, int solvedim) {
//  tridMultiDimBatchSolve<complexd,1>(a, b, c, d, u, ndim, solvedim, sys_size,
//  sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}



void initTridMultiDimBatchSolve(int ndim, int* dims, int* pads) { }
