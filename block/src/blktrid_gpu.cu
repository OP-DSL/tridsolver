/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the block-tridiagonal solver distribution.
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

// Block tridiagonal solver
#include "blktrid_gpu.cuh"
#include "blktrid_common.h"
#include "blktrid_util.h"

// Scalar tridiagonal solver
#include "trid_common.h"
#include "trid_cuda.h" 
#include "trid_cuda.cu" 

#include <cuda.h>

#include "cutil_inline.h"

#include "stdlib.h"
#include "stdio.h"

// s - single
// d - double
// b - block
// t - tridiagonal
// sv- solver
void sbtsv_gpu(float *A, float *B, float *C, float *d, float *Cstar, float *dstar, float *u, const int N, const int P, const int blkdim) {
  const int  PROBS_PER_WARP = (int)(WARP_SIZE/blkdim);
  const dim3 blockDim(TBLOCK,1,1);
  const dim3 gridDim(1 + (P - 1) / (PROBS_PER_WARP * (blockDim.x / WARP_SIZE)),1,1);
  printf("Running kernel using blockDim.x = %d, gridDim.x = %d, PROBS_PER_WARP = %d, WARP_SIZE = %d\n", blockDim.x, gridDim.x, PROBS_PER_WARP, WARP_SIZE);





      // Tridiagonal solver option arguemnt's setup
      int ndim = 2;          // Number of dimensions of the (hyper)cubic data structure.
      int dims[MAXDIM];      // Array containing the sizes of each ndim dimensions. size(dims) == ndim <=MAXDIM
      int pads[MAXDIM];      // Padded sizes along each ndim number of dimensions
      dims[0] = P;
      dims[1] = N;
      //dims[2] = nz;
      pads[0] = dims[0];
      pads[1] = dims[1];
      //pads[2] = dims[2];

      int opts[MAXDIM];// = {nx, ny, nz,...};
      opts[0] = 0;
      opts[1] = 0;
      //opts[2] = 0;

      //trid_set_consts(ndim, dims, pads);
      initTridMultiDimBatchSolve(ndim, dims, pads);
  
      int solvedim = 1;
      float * d_buffer = NULL;
      int sync = 1; // Host-synchronous kernel execution





  // timer variable and elapsed time
  double timer, elapsed; 
elapsed_time(&timer); // initialise timer
  switch(blkdim) {
    case 1:
      //tridMultiDimBatchSolve<float,0>(A, B, C, d, u, ndim, solvedim, dims, pads, opts, &d_buffer, sync);
      blk_thomas_gpu<float,1,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 2:
      blk_thomas_gpu<float,2,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 3:
      blk_thomas_gpu<float,3,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 4:
      blk_thomas_gpu<float,4,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 5:
      blk_thomas_gpu<float,5,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 6:
      blk_thomas_gpu<float,6,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 7:
      blk_thomas_gpu<float,7,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 8:
      blk_thomas_gpu<float,8,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 9:
      blk_thomas_gpu<float,9,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 10:
      blk_thomas_gpu<float,10,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    default:
      printf("Only BLK_DIM block dimension <= 10 is supported!");
      break;
  }
  
    //cudaCheckMsg("kernel failed to launch:\n");
    //cudaSafeCall(cudaDeviceSynchronize());
    cudaDeviceSynchronize();
    elapsed = elapsed_time(&timer);
    printf("elapsed = %lf\n",elapsed);
}

void dbtsv_gpu(double *A, double *B, double *C, double *d, double *Cstar, double *dstar, double *u, const int N, const int P, const int blkdim) {
  const int  PROBS_PER_WARP = (int)(WARP_SIZE/blkdim);
  const dim3 blockDim(TBLOCK,1,1);
  const dim3 gridDim(1 + (P - 1) / (PROBS_PER_WARP * (blockDim.x / WARP_SIZE)),1,1);
  printf("Running kernel using blockDim.x = %d, gridDim.x = %d, PROBS_PER_WARP = %d, WARP_SIZE = %d\n", blockDim.x, gridDim.x, PROBS_PER_WARP, WARP_SIZE);

  switch(blkdim) {
    case 2:
      blk_thomas_gpu<double,2,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 3:
      blk_thomas_gpu<double,3,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 4:
      blk_thomas_gpu<double,4,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 5:
      blk_thomas_gpu<double,5,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 6:
      blk_thomas_gpu<double,6,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 7:
      blk_thomas_gpu<double,7,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 8:
      blk_thomas_gpu<double,8,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 9:
      blk_thomas_gpu<double,9,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    case 10:
      blk_thomas_gpu<double,10,COMMUNICATION,WARP_SIZE,TBLOCK><<<gridDim,blockDim>>>(A, B, C, d, Cstar, dstar, u, N, P);
      break;
    default:
      printf("Only BLK_DIM block dimension <= 10 is supported!");
      break;
  }
}
