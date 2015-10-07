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

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#include "trid_cuda.h"
#include "trid_common.h"

//#define WARP_SIZE 32
//#define ALIGN 32               // 32 byte alignment is required
//#define ALIGN_FLOAT  (ALIGN/4) // 32 byte/ 4bytes/float = 8
////#define ALIGN_DOUBLE (ALIGN/2) // 32 byte/ 8bytes/float = 4
//#define ALIGN_DOUBLE (ALIGN/8) // 32 byte/ 8bytes/float = 4
//
//#define MAXDIM 8 // Maximal dimension that can be used in the library. Defines static arrays
//
//#define CUDA_ALIGN_BYTE 32 // 32 byte alignment is used on CUDA-enabled GPUs

// CUDA constant memory setup for multidimension related index calculations
__constant__ int d_dims[MAXDIM];    // Dimension sizes
__constant__ int d_pads[MAXDIM];    // Padding sizes
__constant__ int d_cumdims[MAXDIM+1]; // Cummulative-multiplication of dimensions
__constant__ int d_cumpads[MAXDIM+1]; // Cummulative-multiplication of paddings

#include "trid_linear.hpp"
#include "trid_linear_shared.hpp"
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if register shuffle intrinsics exist on the device
  #include "trid_linear_reg16_float4.hpp"
  #include "trid_linear_reg8_double2.hpp"
  #include "trid_thomaspcr.hpp"
//#endif
//#include "trid_cusparse.hpp"
#include "trid_strided_multidim.hpp"

#include "cutil_inline.h"
#include <cusparse_v2.h>

//cusparseHandle_t handle_sp; // Handle for cuSPARSE setup

int opts[MAXDIM];

int cumdims[MAXDIM+1]; // Cummulative-multiplication of dimensions
int cumpads[MAXDIM+1]; // Cummulative-multiplication of paddings

int initialised = 0;
int prev_dims[MAXDIM];
int prev_pads[MAXDIM];

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if register shuffle intrinsics exist on the device
template<typename REAL>
void trid_linear_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL* d_ax, const REAL* d_bx, const REAL* d_cx, REAL* d_du, REAL* d_u, int sys_size, int sys_pads, int sys_n) {
  if(sizeof(REAL)==4) {
    trid_linear_reg16_float4<<<dimGrid_x, dimBlock_x>>>((float*)d_ax, (float*)d_bx, (float*)d_cx, (float*)d_du, (float*)d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_reg16_float4 execution failed\n");
  } else if(sizeof(REAL)==8) {
    trid_linear_reg8_double2<<<dimGrid_x, dimBlock_x>>>((double*)d_ax, (double*)d_bx, (double*)d_cx, (double*)d_du, (double*)d_u, sys_size, sys_pads, sys_n);
    cudaCheckMsg("trid_linear_reg8_double2 execution failed\n");
  }
}
//#endif

//
// Tridiagonal solver for linear (coniguous) system layout. Optimization options may be used to select optimization algorithm
//
template<typename REAL, int INC>
void trid_linearlayout_cuda(const REAL** d_ax, const REAL** d_bx, const REAL** d_cx, REAL** d_du, REAL** d_u, int sys_size, int sys_pads, int sys_n, int opt) {
  //int sys_stride = 1; // Linear layout -> stride = 1

  // Set up the execution configuration
  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid   = 1 + (sys_n-1)/blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx,blockdimy);
  int  shared_mem_size = ((8+1) * dimBlock_x.x * dimBlock_x.y) * sizeof(REAL);

  // Arguments for solveBatchedTrid() function call
  int numTrids = sys_n;
  int length   = sys_size;
  int stride1  = 1;
  int stride2  = sys_size;
  int subBatchSize   = numTrids;
  int subBatchStride = 0;

  switch(opt) {
    case 0:
      trid_linear<REAL><<<dimGrid_x, dimBlock_x>>>(*d_ax, *d_bx, *d_cx, *d_du, *d_u, sys_size, sys_pads, sys_n);
      cudaCheckMsg("trid_linear execution failed\n");
      break;
    case 1:
      trid_linear_shared<REAL><<<dimGrid_x, dimBlock_x, shared_mem_size>>>(*d_ax, *d_bx, *d_cx, *d_du, *d_u, sys_size, sys_pads, sys_n);
      cudaCheckMsg("trid_linear_shared execution failed\n");
      break;
    case 2:
      //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if register shuffle intrinsics exist on the device
        trid_linear_reg<REAL>(dimGrid_x, dimBlock_x, *d_ax, *d_bx, *d_cx, *d_du, *d_u, sys_size, sys_pads, sys_n);
        cudaCheckMsg("trid_linear_reg16_float4 execution failed\n");
      //#else 
      //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
      //#endif   
      break;
    case 3:
      //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if register shuffle intrinsics exist on the device
        //trid_linear_thomaspcr<REAL, WARP_SIZE>(*d_ax, *d_bx, *d_cx, *d_du, *d_u, sys_size, sys_pads, sys_n);
        //solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2, subBatchSize, subBatchStride, *d_ax, *d_bx, *d_cx, *d_du, *d_u);
        solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2, subBatchSize, subBatchStride, *d_ax, *d_bx, *d_cx, *d_du, *d_du);
        cudaCheckMsg("trid_linear_thomaspcr execution failed\n");
      //#else 
      //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
      //#endif   
      break;
    case 4:
      //trid_linear_cusparse(handle_sp, d_ax, d_bx, d_cx, d_du, d_u, sys_stride, sys_size, sys_pads, sys_n);
      cudaCheckMsg("trid_linear_cusparse execution failed\n");
      break;
    default:
      printf("Wrong optimization argument OPT is given. Exiting.\n");
      exit(-1);
  }
}

//
// Initialize solver environment. Set CUDA __constant__ variables specifing dimension and padding sizes.
//
void initTridMultiDimBatchSolve(int ndim, int *dims, int *pads) {
  
//  int changed = 0;
//  for (int i = 0; i < ndim; i++) {
//    changed = changed || !(dims[i]==prev_dims[i] && pads[i]==prev_pads[i]);
//  }
//  
//  if (changed == 1 || !initialised) {
//    initialised = 1;
    
    // Initialize CUDA cuSPARSE libraries
    //if(cusparseCreate(&handle_sp) != CUSPARSE_STATUS_SUCCESS) exit(-1);

    // Set CUDA __constant__ variables
    cumdims[0] = 1;
    cumpads[0] = 1;
    for(int i=0; i<ndim; i++) {
      cumdims[i+1] = cumdims[i]*dims[i];
      cumpads[i+1] = cumpads[i]*pads[i];
    }
    cudaSafeCall(cudaMemcpyToSymbol(d_dims,    dims,    ndim*sizeof(int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(d_pads,    pads,    ndim*sizeof(int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(d_cumdims, cumdims, (ndim+1)*sizeof(int), 0, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(d_cumpads, cumpads, (ndim+1)*sizeof(int), 0, cudaMemcpyHostToDevice));
//  }
}

//
// Host function for selecting the proper setup for solve in a specific dimension
//
template<typename REAL, int INC>
void tridMultiDimBatchSolve(const REAL* d_a, const REAL* d_b, const REAL* d_c, REAL* d_d, REAL* d_u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync) {
  int sys_n = cumdims[ndim]/dims[solvedim]; // Number of systems to be solved

  if(solvedim == 0) {
    //int sys_stride = 1;       // Stride between the consecutive elements of a system
    int sys_size   = dims[0]; // Size (length) of a system
    int sys_pads   = pads[0]; // Padded sizes along each ndim number of dimensions
    int sys_n_lin  = dims[1]*dims[2]; // = cumdims[solve] // Number of systems to be solved
    int opt        = opts[0];
    trid_linearlayout_cuda<REAL,0>(&d_a, &d_b, &d_c, &d_d, &d_u, sys_size, sys_pads, sys_n_lin, opt);
  }
  else {
    if(solvedim==1 && opts[1]==3) { // If y-solve and ThomasPCR
      int numTrids = dims[0]*dims[2];
      int length   = dims[1];
      int stride1  = dims[0];
      int stride2  = 1;
      int subBatchSize   = dims[0];//dims[1];
      int subBatchStride = dims[0]*dims[1];
      //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if register shuffle intrinsics exist on the device
        // y: stride1 = x, stride2 = 1, subBatchSize = x, subBatchStride = x*y
        solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2, subBatchSize, subBatchStride, d_a, d_b, d_c, d_d, d_d);
        //solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2, subBatchSize, subBatchStride, d_a, d_b, d_c, d_d, d_u);
      //#else 
      //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
      //#endif   
    } else if(solvedim==2 && opts[2]==3) { // If z-solve and ThomasPCR
      int numTrids = dims[0]*dims[1];
      int length   = dims[2];
      int stride1  = dims[0]*dims[1];
      int stride2  = 1;
      int subBatchSize   = dims[0]*dims[1];
      int subBatchStride = 0;
      //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >=300) // Only compiler if register shuffle intrinsics exist on the device
        // z: stride1 = x*y, stride2 = 1, subBatchSize = x*y, subBatchStride = 0
        solveBatchedTrid<REAL,INC>(numTrids, length, stride1, stride2, subBatchSize, subBatchStride, d_a, d_b, d_c, d_d, d_u);
      //#else 
      //  printf("This option is only valid for __CUDA_ARCH__ >= 300\n");
      //#endif   
    } else {
      // Test if data is aligned
      long isaligned = 0;
      isaligned  = (long)d_a % CUDA_ALIGN_BYTE;            // Check if base pointers are aligned
      isaligned += (long)d_b % CUDA_ALIGN_BYTE;
      isaligned += (long)d_c % CUDA_ALIGN_BYTE;
      isaligned += (long)d_d % CUDA_ALIGN_BYTE;
      if(d_u != NULL) isaligned += (long)d_u % CUDA_ALIGN_BYTE;
      isaligned += (dims[0]*sizeof(REAL)) % CUDA_ALIGN_BYTE; // Check if X-dimension allows alignment

      if(isaligned==0) { // If any of the above is non-zero, vector loads can not be used
        if(sizeof(REAL) == 4) {
          // Kernel launch configuration
          int sys_n_float4 = sys_n/4;
          int blockdimx_float4 = 128; // Has to be the multiple of 32(or maybe 4??)
          int blockdimy_float4 = 1;
          int dimgrid_float4   = 1 + (sys_n_float4-1)/blockdimx_float4; // can go up to 65535
          int dimgridx_float4  = dimgrid_float4 % 65536;         // can go up to max 65535 on Fermi
          int dimgridy_float4  = 1 + dimgrid_float4 / 65536;
          dim3 dimGrid_float4(dimgridx_float4, dimgridy_float4);
          dim3 dimBlock_float4(blockdimx_float4,blockdimy_float4);

          // Setup dimension and padding according to float4 loads/stores
          dims[0] = dims[0]/4;
          pads[0] = dims[0];
          //trid_set_consts(ndim, dims, pads);
          initTridMultiDimBatchSolve(ndim, dims, pads);

          trid_strided_multidim<float,float4,INC><<<dimGrid_float4, dimBlock_float4>>>((float4*)d_a, (float4*)d_b, (float4*)d_c, (float4*)d_d, (float4*)d_u, ndim, solvedim, sys_n_float4);

          dims[0] = dims[0]*4;
          pads[0] = dims[0];
          //trid_set_consts(ndim, dims, pads);
          initTridMultiDimBatchSolve(ndim, dims, pads);
        } else if(sizeof(REAL) == 8) {
          // Kernel launch configuration
          int sys_n_double2 = sys_n/2;
          int blockdimx_double2 = 128; // Has to be the multiple of 32(or maybe 4??)
          int blockdimy_double2 = 1;
          int dimgrid_double2  = 1 + (sys_n_double2-1)/blockdimx_double2; // can go up to 65535
          int dimgridx_double2  = dimgrid_double2 % 65536;         // can go up to max 65535 on Fermi
          int dimgridy_double2  = 1 + dimgrid_double2 / 65536;
          dim3 dimGrid_double2(dimgridx_double2, dimgridy_double2);
          dim3 dimBlock_double2(blockdimx_double2,blockdimy_double2);
          // Setup dimension and padding according to double2 loads/stores
          dims[0] = dims[0]/2;
          pads[0] = dims[0];
          //trid_set_consts(ndim, dims, pads);
          initTridMultiDimBatchSolve(ndim, dims, pads);

          trid_strided_multidim<double,double2,INC><<<dimGrid_double2, dimBlock_double2>>>((double2*)d_a, (double2*)d_b, (double2*)d_c, (double2*)d_d, (double2*)d_u, ndim, solvedim, sys_n_double2);

          dims[0] = dims[0]*2;
          pads[0] = dims[0];
          //trid_set_consts(ndim, dims, pads);
          initTridMultiDimBatchSolve(ndim, dims, pads);
        }
      } else {
        // Kernel launch configuration
        int blockdimx = 128; // Has to be the multiple of 32(or maybe 4??)
        int blockdimy = 1;
        int dimgrid   = 1 + (sys_n-1)/blockdimx; // can go up to 65535
        int dimgridx  = dimgrid % 65536;         // can go up to max 65535 on Fermi
        int dimgridy  = 1 + dimgrid / 65536;
        dim3 dimGrid(dimgridx, dimgridy);
        dim3 dimBlock(blockdimx,blockdimy);

        trid_strided_multidim<REAL,REAL,INC><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, d_d, d_u, ndim, solvedim, sys_n);
      }
    }
//      break;
//    default:
//      printf("Wrong optimization argument OPT is given. Exiting.\n");
//      exit(-1);
  }
  if(sync == 1) cudaSafeCall( cudaDeviceSynchronize() );
}

//------------------------------------------------------------------------------------------------------------------
//
// API calls
//
//------------------------------------------------------------------------------------------------------------------

//tridStatus_t tridSgtsvStridedBatch(int sys_size, const float* a, const float *b, const float *c, float *d, int num_sys, int sys_stride, int opt) {
//  trid_linearlayout_cuda<float,0>(a, b, c, d, NULL, sys_size, sys_stride, num_sys, opt);
//  return TRID_STATUS_SUCCESS;
//}
//
//tridStatus_t tridDgtsvStridedBatch(int sys_size, const double* a, const double *b, const double *c, double *d, int num_sys, int sys_stride, int opt) {
//  trid_linearlayout_cuda<double,0>(a, b, c, d, NULL, sys_size, sys_stride, num_sys, opt);
//  return TRID_STATUS_SUCCESS;
//}

//tridStatus_t tridCgtsvStridedBatch(int sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, int num_sys, int sys_stride) {
//  trid_linearlayout_cuda<complexf,0>(a, b, c, d, NULL, sys_size, sys_stride, num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

//tridStatus_t tridZgtsvStridedBatch(int sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, int num_sys, int sys_stride) {
//  trid_linearlayout_cuda<complexd,0>(a, b, c, d, NULL, sys_size, sys_stride, num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

//tridStatus_t tridSgtsvStridedBatchInc(int sys_size, const float* a, const float *b, const float *c, float *d, float *u, int num_sys, int sys_stride, int* opts) {
//  trid_linearlayout_cuda<float,1>(a, b, c, d, u, sys_size, sys_stride, num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}
//
//tridStatus_t tridDgtsvStridedBatchInc(int sys_size, const double* a, const double *b, const double *c, double *d, double *u, int num_sys, int sys_stride, int* opts) {
//  trid_linearlayout_cuda<double,1>(a, b, c, d, u, sys_size, sys_stride, num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

//tridStatus_t tridCgtsvStridedBatchInc(int sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, complexf *u, int num_sys, int sys_stride) {
//  trid_linearlayout_cuda<complexf,1>(a, b, c, d, u, sys_size, sys_stride, num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

//tridStatus_t tridZgtsvStridedBatchInc(int sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, complexd *u, int num_sys, int sys_stride) {
//  trid_linearlayout_cuda<complexd,1>(a, b, c, d, u, sys_size, sys_stride, num_sys, opts[0]);
//  return TRID_STATUS_SUCCESS;
//}

tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b, const float *c, float *d, float* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync) {
  tridMultiDimBatchSolve<float,0>(a, b, c, d, NULL, ndim, solvedim, dims, pads, opts, 1);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b, const double *c, double *d, double* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync) {
  tridMultiDimBatchSolve<double,0>(a, b, c, d, NULL, ndim, solvedim, dims, pads, opts, 1);
  return TRID_STATUS_SUCCESS;
}

//tridStatus_t tridCmtsvStridedBatch(int ndim, int* sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, int *sys_stride, int solvedim) {
//  tridMultiDimBatchSolve<complexf,0>(a, b, c, d, NULL, ndim, solvedim, sys_size, sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}
//
//tridStatus_t tridZmtsvStridedBatch(int ndim, int* sys_size, const complexd* a, const complexd *b, const complexd *c, complexd*d, int *sys_stride, int solvedim) {
//  tridMultiDimBatchSolve<complexd,0>(a, b, c, d, NULL, ndim, solvedim, sys_size, sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}

tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b, const float *c, float *d, float* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync) {
  tridMultiDimBatchSolve<float,1>(a, b, c, d, u, ndim, solvedim, dims, pads, opts, 1);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b, const double *c, double *d, double* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync) {
  tridMultiDimBatchSolve<double,1>(a, b, c, d, u, ndim, solvedim, dims, pads, opts, 1);
  return TRID_STATUS_SUCCESS;
}

//tridStatus_t tridCmtsvStridedBatchInc(int ndim, int* sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, complexf *u, int *sys_stride, int solvedim) {
//  tridMultiDimBatchSolve<complexf,1>(a, b, c, d, u, ndim, solvedim, sys_size, sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}
//
//tridStatus_t tridZmtsvStridedBatchInc(int ndim, int* sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, complexd *u, int *sys_stride, int solvedim) {
//  tridMultiDimBatchSolve<complexd,1>(a, b, c, d, u, ndim, solvedim, sys_size, sys_stride, opts, NULL, 1, 1);
//  return TRID_STATUS_SUCCESS;
//}

int* get_opts() {return opts;}

