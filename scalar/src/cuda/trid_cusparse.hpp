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

#ifndef __TRID_CUSPARSE_HPP
#define __TRID_CUSPARSE_HPP
//#include "adi_cuda.h"
//#include "trid_params.h"
#include <cusparse_v2.h>

////
//// Add contribution
////
//__global__ void add_contrib(FP* d, FP* u, int sys_stride, int sys_size, int sys_pads, int sys_n) {
//  int   i;
//  //
//  // set up indices for main block
//  //
//  int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
//  int ind = sys_pads*tid;
//
//  for(i=0; i<sys_size; i++) {
//    ind    += sys_stride;
//    u[ind] = ind;//+= d[ind];
//  }
//}

//
// Add contribution kernel
//
template<typename REAL>
__global__ void add_contrib_kernel(REAL* d, REAL* u, int nx, int ny, int nz) {
  long   i, j, k, ind, off;
  //
  // set up indices for main block
  //
  i   = threadIdx.x + blockIdx.x*blockDim.x;
  j   = threadIdx.y + blockIdx.y*blockDim.y;
  ind = i+j*nx;
  off = nx*ny;

  for(k=0; k<nz; k++) {
    ind    += off;
    u[ind] += d[ind];
  }
}

//
// Add contribution host function
//
template<typename REAL>
void add_contrib(REAL* d_d, REAL* d_u, int nx, int ny, int nz) {
  dim3 dimGrid(1+(nx-1)/32, 1+(ny-1)/4);
  dim3 dimBlock(32,4);
  add_contrib_kernel<<<dimGrid, dimBlock>>>(d_d, d_u, nx, ny, nz);
}


#define TILE_DIM   32
#define BLOCK_ROWS  8
//
// GPU kernel to transpose data on a slice of a hypercube. Out of place transpose is performed.
//
__global__ void transpose(FP* a, FP* buffer, int in_stride, int out_stride, int matrix_stride){
  __shared__ FP tile[TILE_DIM][TILE_DIM+1];

  int offset   = blockIdx.z * matrix_stride; // offset is used for the z-dimension

  int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = offset + xIndex + (yIndex)*in_stride;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = offset + xIndex + (yIndex)*out_stride;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[threadIdx.y+i][threadIdx.x] = a[index_in+i*in_stride];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    buffer[index_out+i*out_stride] = tile[threadIdx.x][threadIdx.y+i];
  }
}

//
// CPU code to preform data transposition along dimension Z for every X-Y plane
//   Uses a buffer to do out of place transposition
//   dir - direction: 0 - transpose X-Y plane, 1 - transpose X-Z plane
//
inline void transpose(FP** array, FP** buffer, int nx, int ny, int nz, int dir) {
  dim3 dimGrid(1+(nx-1)/TILE_DIM, 1+(ny-1)/TILE_DIM,nz);
  dim3 dimBlock(TILE_DIM,BLOCK_ROWS,1);
  // Perform out of place transpose
  if(dir==0)      transpose<<<dimGrid, dimBlock>>>(*array, *buffer, nx, ny, nx*ny);
  else if(dir==1) transpose<<<dimGrid, dimBlock>>>(*array, *buffer, nx*ny, nx*ny, nz);
  // Switch pointers
  FP* tmp = *array;
  *array = *buffer;
  *buffer = tmp;
}

// 
// Tridiagonal solver along dimension that is linearly layed down
//
template<typename REAL>
void trid_linear_cusparse(cusparseHandle_t *handle_sp, const REAL** d_ax, const REAL** d_bx, const REAL** d_cx, REAL** d_du, REAL** d_u, int sys_stride, int sys_size, int sys_pads, int sys_n) {
  if(sizeof(REAL)==4) {
    if(cusparseSgtsvStridedBatch(*handle_sp, sys_size, (float*)*d_ax, (float*)*d_bx, (float*)*d_cx, (float*)*d_du, sys_n, sys_pads) != CUSPARSE_STATUS_SUCCESS) exit(-1);
  }else if(sizeof(REAL)==8) {
    if(cusparseDgtsvStridedBatch(*handle_sp, sys_size, (double*)*d_ax, (double*)*d_bx, (double*)*d_cx, (double*)*d_du, sys_n, sys_pads) != CUSPARSE_STATUS_SUCCESS) exit(-1);
  }
}

// 
// tridiagonal-y solver
//
//void trid_y_cusparse(cusparseHandle_t *handle_sp, FP** d_ay, FP** d_by, FP** d_cy, FP** d_du, FP** d_u, int sys_stride, int sys_size, int sys_pads, int sys_n, int nx, int ny, int nz, FP** d_buffer) {
template<typename REAL>
void trid_y_cusparse(cusparseHandle_t *handle_sp, REAL** d_ay, REAL** d_by, REAL** d_cy, REAL** d_du, REAL** d_u, int nx, int ny, int nz, REAL** d_buffer) {
  // Transpose X-Y matrices
  // Only transpose data that is needed for solving Y
  transpose(d_du, d_buffer, nx, ny, nz, 0);
  transpose(d_ay, d_buffer, nx, ny, nz, 0);
  transpose(d_by, d_buffer, nx, ny, nz, 0);
  transpose(d_cy, d_buffer, nx, ny, nz, 0);
  // Solve tridiagonal systems of dimension Y
  if(sizeof(REAL)==4) {
    if(cusparseSgtsvStridedBatch(*handle_sp, ny, (float*)*d_ay, (float*)*d_by, (float*)*d_cy, (float*)*d_du, nx*nz, nx) != CUSPARSE_STATUS_SUCCESS) exit(-1);
  }else if(sizeof(REAL)==8) {
    if(cusparseDgtsvStridedBatch(*handle_sp, ny, (double*)*d_ay, (double*)*d_by, (double*)*d_cy, (double*)*d_du, nx*nz, nx) != CUSPARSE_STATUS_SUCCESS) exit(-1);
  }
  // Transpose X-Y matrices
  // Only transpose back array u since ay, by and cy will not be used later
  transpose(d_du, d_buffer, nx, ny, nz, 0);
}

// 
// tridiagonal-z solver
//
template<typename REAL>
void trid_z_cusparse(cusparseHandle_t *handle_sp, REAL** d_az, REAL** d_bz, REAL** d_cz, REAL** d_du, REAL** d_u, int nx, int ny, int nz, REAL** d_buffer) {
  // Only transpose data that is needed for solving Z
  transpose(d_du, d_buffer, nx, ny, nz, 1);
  transpose(d_az, d_buffer, nx, ny, nz, 1);
  transpose(d_bz, d_buffer, nx, ny, nz, 1);
  transpose(d_cz, d_buffer, nx, ny, nz, 1);
  // Solve tridiagonal systems of dimension Z
  if(sizeof(REAL)==4) {
    if(cusparseSgtsvStridedBatch(*handle_sp, nz, (float*)*d_az, (float*)*d_bz, (float*)*d_cz, (float*)*d_du, nx*ny, nx*ny) != CUSPARSE_STATUS_SUCCESS) exit(-1);
  }else if(sizeof(REAL)==8) {
    if(cusparseDgtsvStridedBatch(*handle_sp, nz, (double*)*d_az, (double*)*d_bz, (double*)*d_cz, (double*)*d_du, nx*ny, nx*ny) != CUSPARSE_STATUS_SUCCESS) exit(-1);
  }
  // Transpose X-Y matrices
  // Only transpose back array u since ay, by and cy will not be used later
  transpose(d_du, d_buffer, nx, ny, nz, 1);
}

//
////#include "adi_cuda.h"
//#include "trid_cuda.h"
//
////
//// Add contribution
////
//__global__ void add_contrib(FP* d, FP* u, int nx, int ny, int nz) {
//  int   i, j, k, ind, off;
//  //
//  // set up indices for main block
//  //
//  i   = threadIdx.x + blockIdx.x*blockDim.x;
//  j   = threadIdx.y + blockIdx.y*blockDim.y;
//  ind = i+j*nx;
//  off = nx*ny;
//
//  for(k=0; k<nz; k++) {
//    ind    += off;
//    u[ind] += d[ind];
//  }
//}
//
//// Transpose an array of a cube
//#define TILE_DIM   32
//#define BLOCK_ROWS  8
////
//// GPU kernel to transpose data on X-Y plane
////
//__global__ void transpose(FP*  a, FP*  buffer, int in_stride, int out_stride, int matrix_stride){
//    __shared__ FP tile[TILE_DIM][TILE_DIM+1];
//
//    int offset = blockIdx.z * matrix_stride;
//
//    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
//    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
//    int index_in = offset + xIndex + (yIndex)*in_stride;
//
//    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//    int index_out = offset + xIndex + (yIndex)*out_stride;
//
//    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
//    {
//      tile[threadIdx.y+i][threadIdx.x] = a[index_in+i*in_stride];
//    }
//
//    __syncthreads();
//
//    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
//    {
//      buffer[index_out+i*out_stride] = tile[threadIdx.x][threadIdx.y+i];
//    }
//}
//
////
//// CPU code to preform data transposition along dimension Z for every X-Y plane
////   Uses a buffer to do out of place transposition
////   dir - direction: 0 - transpose X-Y plane, 1 - transpose X-Z plane
////
//inline void transpose(FP**  array, FP**  buffer, int nx, int ny, int nz, int dir) {
//  dim3 dimGrid(1+(nx-1)/TILE_DIM, 1+(ny-1)/TILE_DIM,nz);
//  dim3 dimBlock(TILE_DIM,BLOCK_ROWS,1);
//  // Perform out of place transpose
//  if(dir==0)      transpose<<<dimGrid, dimBlock>>>(*array, *buffer, nx, ny, nx*ny);
//  else if(dir==1) transpose<<<dimGrid, dimBlock>>>(*array, *buffer, nx*ny, nx*ny, nz);
//  // Switch pointers
//  FP* tmp = *array;
//  *array = *buffer;
//  *buffer = tmp;
//}
//
////
//// tridiagonal-x solver
////
//void trid_x_cusparse(cusparseHandle_t *handle_sp, FP**  d_ax, FP**  d_bx, FP**  d_cx, FP**  d_du, FP**  d_u, int nx, int ny, int nz) {
//#if FPPREC==0
//  if(cusparseSgtsvStridedBatch(*handle_sp, nx, *d_ax, *d_bx, *d_cx, *d_du, ny*nz, nx) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//#elif FPPREC==1
//  if(cusparseDgtsvStridedBatch(*handle_sp, nx, *d_ax, *d_bx, *d_cx, *d_du, ny*nz, nx) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//#endif
//}
//
////
//// tridiagonal-y solver
////
//void trid_y_cusparse(cusparseHandle_t *handle_sp, FP**  d_ay, FP**  d_by, FP**  d_cy, FP**  d_du, FP**  d_u, int nx, int ny, int nz, FP**  d_buffer) {
//    // Transpose X-Y matrices
//    // Only transpose data that is needed for solving Y
//      transpose(d_du, d_buffer, nx, ny, nz, 0);
//      transpose(d_ay, d_buffer, nx, ny, nz, 0);
//      transpose(d_by, d_buffer, nx, ny, nz, 0);
//      transpose(d_cy, d_buffer, nx, ny, nz, 0);
//      // Solve tridiagonal systems of dimension Y
//      #if FPPREC==0
//        if(cusparseSgtsvStridedBatch(*handle_sp, ny, *d_ay, *d_by, *d_cy, *d_du, nx*nz, ny) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//        //if(cusparseSgtsvStridedBatch(*handle_sp, ny, *d_ay, *d_by, *d_cy, *d_du, 255*256, 256) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//      #elif FPPREC==1
//        if(cusparseDgtsvStridedBatch(*handle_sp, ny, *d_ay, *d_by, *d_cy, *d_du, nx*nz, ny) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//      #endif
//      // Transpose X-Y matrices
//      // Only transpose back array u since ay, by and cy will not be used later
//      transpose(d_du, d_buffer, nx, ny, nz, 0);
//}
//
////
//// tridiagonal-z solver
////
//void trid_z_cusparse(cusparseHandle_t *handle_sp, FP**  d_az, FP**  d_bz, FP**  d_cz, FP**  d_du, FP**  d_u, int nx, int ny, int nz, FP**  d_buffer) {
//    // Only transpose data that is needed for solving Z
//      transpose(d_du, d_buffer, nx, ny, nz, 1);
//      transpose(d_az, d_buffer, nx, ny, nz, 1);
//      transpose(d_bz, d_buffer, nx, ny, nz, 1);
//      transpose(d_cz, d_buffer, nx, ny, nz, 1);
//      // Solve tridiagonal systems of dimension Z
//      #if FPPREC==0
//        if(cusparseSgtsvStridedBatch(*handle_sp, nz, *d_az, *d_bz, *d_cz, *d_du, nx*ny, nz) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//      #elif FPPREC==1
//        if(cusparseDgtsvStridedBatch(*handle_sp, nz, *d_az, *d_bz, *d_cz, *d_du, nx*ny, nz) != CUSPARSE_STATUS_SUCCESS) exit(-1);
//      #endif
//      // Transpose X-Y matrices
//      // Only transpose back array u since ay, by and cy will not be used later
//      transpose(d_du, d_buffer, nx, ny, nz, 1);
//}

#endif
