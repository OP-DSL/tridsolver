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

#ifndef __TRID_MULTIDIM_H
#define __TRID_MULTIDIM_H

#include "helper_math.h"

//
// Tridiagonal solver for multidimensional batch problems
//
template <typename REAL, typename VECTOR, int INC>
__device__ void trid_strided_multidim_kernel(const VECTOR* __restrict__ a, int ind_a, int stride_a,
                                      const VECTOR* __restrict__ b, int ind_b, int stride_b,
                                      const VECTOR* __restrict__ c, int ind_c, int stride_c,
                                      VECTOR* __restrict__ d, int ind_d, int stride_d,
                                      VECTOR* __restrict__ u, int ind_u, int stride_u,
                                      int sys_size) {
   VECTOR aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
   //
   // forward pass
   //
   bb    = (static_cast<REAL>(1.0))  / b[ind_b];
   cc    = bb*c[ind_c];
   dd    = bb*d[ind_d];
   c2[0] = cc;
   d2[0] = dd;
   for(int j=1; j<sys_size; j++) {
      ind_a += stride_a;
      ind_b += stride_b;
      ind_c += stride_c;
      ind_d += stride_d;
      if(INC) ind_u += stride_u;
      aa    = a[ind_a];
      bb    = b[ind_b] - aa*cc;
      dd    = d[ind_d] - aa*dd;
      bb    = (static_cast<REAL>(1.0))  / bb;
      cc    = bb*c[ind_c];
      dd    = bb*dd;
      c2[j] = cc;
      d2[j] = dd;
   }
   //
   // reverse pass
   //
   if(INC==0) d[ind_d]  = dd;
   else       u[ind_u] += dd;
   //u[ind] = dd;
   for(int j=sys_size-2; j>=0; j--) {
      if (INC==0) ind_d -= stride_d;
      else ind_u -= stride_u;
      dd     = d2[j] - c2[j]*dd;
      if(INC==0) d[ind_d]  = dd;
      else       u[ind_u] += dd;
   }
}

typedef struct {
   int v[8];
} DIM_V;


template<typename REAL, typename VECTOR, int INC>
__global__ void trid_strided_multidim(
      const VECTOR* __restrict__ a, const DIM_V a_pads,
      const VECTOR* __restrict__ b, const DIM_V b_pads,
      const VECTOR* __restrict__ c, const DIM_V c_pads,
      VECTOR* __restrict__ d, const DIM_V d_pads,
      VECTOR* __restrict__ u, const DIM_V u_pads,
      int ndim, int solvedim, int sys_n,
      const DIM_V dims) {

   int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
   if ( solvedim < 1 || solvedim > ndim ) return; /* Just hints to the compiler */

   int __shared__ d_cumdims[MAXDIM + 1];
   int __shared__ d_cumpads[5][MAXDIM + 1];


   /* Build up d_cumpads and d_cumdims */
   if ( tid < 6 ) {
       int *tgt = (tid == 0) ? d_cumdims : d_cumpads[tid-1];
       const int *src = NULL;
       switch (tid) {
           case 0: src = dims.v;    break;
           case 1: src = a_pads.v;  break;
           case 2: src = b_pads.v;  break;
           case 3: src = c_pads.v;  break;
           case 4: src = d_pads.v;  break;
           case 5: src = u_pads.v;  break;
       }

      tgt[0] = 1;
      for ( int i = 0 ; i < ndim ; i++ ) {
         tgt[i+1] = tgt[i] * src[i];
      }
   }
   __syncthreads();

   //
   // set up indices for main block
   //
   tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system

   int ind_a = 0;
   int ind_b = 0;
   int ind_c = 0;
   int ind_d = 0;
   int ind_u = 0;

   for ( int j = 0; j < solvedim; j++) {
      ind_a += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[0][j];
      ind_b += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[1][j];
      ind_c += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[2][j];
      ind_d += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[3][j];
      if (INC) ind_u += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[4][j];
   }
   for ( int j = solvedim+1; j < ndim; j++) {
      ind_a += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[0][j];
      ind_b += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[1][j];
      ind_c += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[2][j];
      ind_d += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[3][j];
      if (INC) ind_u += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[4][j];
   }


   int stride_a   = d_cumpads[0][solvedim];
   int stride_b   = d_cumpads[1][solvedim];
   int stride_c   = d_cumpads[2][solvedim];
   int stride_d   = d_cumpads[3][solvedim];
   int stride_u   = d_cumpads[4][solvedim];
   int sys_size = dims.v[solvedim];


   if( tid<sys_n ) {
#if 0
      if ( tid == 0 ) {
          printf("a_pad: [% 4d  % 4d  % 4d]\n", a_pads.v[0], a_pads.v[1], a_pads.v[2]);
          printf("b_pad: [% 4d  % 4d  % 4d]\n", b_pads.v[0], b_pads.v[1], b_pads.v[2]);
          printf("c_pad: [% 4d  % 4d  % 4d]\n", c_pads.v[0], c_pads.v[1], c_pads.v[2]);
          printf("d_pad: [% 4d  % 4d  % 4d]\n", d_pads.v[0], d_pads.v[1], d_pads.v[2]);
          printf("u_pad: [% 4d  % 4d  % 4d]\n", u_pads.v[0], u_pads.v[1], u_pads.v[2]);
          printf("dims:  [% 4d  % 4d  % 4d]\n", dims.v[0], dims.v[1], dims.v[2]);
          printf("strides:  [% 4d   % 4d   % 4d   % 4d   % 4d]\n", stride_a, stride_b, stride_c, stride_d, stride_u);
          printf("solvedim:  %d    sys_size:  %d\n", solvedim, sys_size);
      }
      printf("tid: % 8d   ind: [% 6d  % 6d  % 6d  % 6d  % 6d]\n", tid, ind_a, ind_b, ind_c, ind_d, ind_u);
#endif

      trid_strided_multidim_kernel<REAL, VECTOR, INC>(
                a, ind_a, stride_a,
                b, ind_b, stride_b,
                c, ind_c, stride_c,
                d, ind_d, stride_d,
                u, ind_u, stride_u,
                sys_size);
   }
}


/* Rumor has it that the GPU kernels can take along about 256 bytes of arguments
 * If we assume that includes a Program Counter pointer, that means that we have:
 * 8 bytes PC
 * 5*8 bytes (a, b, c, d, u)
 * 4 bytes (ndim)
 * 4 bytes (solvedim)
 * 4 bytes (sys_n)
 * ===
 * 60 bytes
 *
 * That leaves ~196B.
 *
 * We need to pass 8 bytes per supported dimension (4 dim, 4 pad)
 * For a MAXDIM of 8, that is 64 bytes, well within the range.
 *
 * So, our API:
 * __host__ launchSolve(a,b,c,d,u, ndim, int*dims, int*pads, solvedim, sys_n)
 */
#if MAXDIM > 8
#error "Code needs updated to support DIMS > 8... Verify GPU can handle it"
#endif

template<typename REAL, typename VECTOR, int INC>
void trid_strided_multidim(const dim3 &grid, const dim3 & block,
                           const VECTOR* __restrict__ a, const int* __restrict__ a_pads,
                           const VECTOR* __restrict__ b, const int* __restrict__ b_pads,
                           const VECTOR* __restrict__ c, const int* __restrict__ c_pads,
                           VECTOR* __restrict__ d, const int* __restrict__ d_pads,
                           VECTOR* __restrict__ u, const int* __restrict__ u_pads,
                           int ndim, int solvedim, int sys_n,
                           const int* __restrict__ dims) {

   DIM_V v_dims, va_pads, vb_pads, vc_pads, vd_pads, vu_pads;
   memcpy(v_dims.v, dims, ndim*sizeof(int));
   memcpy(va_pads.v, a_pads, ndim*sizeof(int));
   memcpy(vb_pads.v, b_pads, ndim*sizeof(int));
   memcpy(vc_pads.v, c_pads, ndim*sizeof(int));
   memcpy(vd_pads.v, d_pads, ndim*sizeof(int));
   if (INC) memcpy(vu_pads.v, u_pads, ndim*sizeof(int));
   else memset(vu_pads.v, '\0', ndim*sizeof(int));

   trid_strided_multidim<REAL, VECTOR, INC><<<grid, block>>>(
        a, va_pads, b, vb_pads, c, vc_pads, d, vd_pads, u, vu_pads,
        ndim, solvedim, sys_n, v_dims);
}

/*
    Functions to deal with x solve when there is y padding.
    This does the same as the above functions except that the padding values
    are not operated on.
*/

template<typename REAL, typename VECTOR, int INC>
__global__ void trid_strided_multidim_x_solve_y_padding(
      const VECTOR* __restrict__ a, const DIM_V a_pads,
      const VECTOR* __restrict__ b, const DIM_V b_pads,
      const VECTOR* __restrict__ c, const DIM_V c_pads,
      VECTOR* __restrict__ d, const DIM_V d_pads,
      VECTOR* __restrict__ u, const DIM_V u_pads,
      int ndim, int solvedim, int sys_n,
      const DIM_V dims, const int y_size, const int y_pads) {

   int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

   if ( solvedim < 1 || solvedim > ndim ) return; /* Just hints to the compiler */

   int __shared__ d_cumdims[MAXDIM + 1];
   int __shared__ d_cumpads[5][MAXDIM + 1];


   /* Build up d_cumpads and d_cumdims */
   if ( tid < 6 ) {
       int *tgt = (tid == 0) ? d_cumdims : d_cumpads[tid-1];
       const int *src = NULL;
       switch (tid) {
           case 0: src = dims.v;    break;
           case 1: src = a_pads.v;  break;
           case 2: src = b_pads.v;  break;
           case 3: src = c_pads.v;  break;
           case 4: src = d_pads.v;  break;
           case 5: src = u_pads.v;  break;
       }

      tgt[0] = 1;
      for ( int i = 0 ; i < ndim ; i++ ) {
         tgt[i+1] = tgt[i] * src[i];
      }
   }
   __syncthreads();

   //
   // set up indices for main block
   //
   tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system

   int in_padding = (tid % y_pads) >= y_size;

   int ind_a = 0;
   int ind_b = 0;
   int ind_c = 0;
   int ind_d = 0;
   int ind_u = 0;

   for ( int j = 0; j < solvedim; j++) {
      ind_a += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[0][j];
      ind_b += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[1][j];
      ind_c += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[2][j];
      ind_d += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[3][j];
      if (INC) ind_u += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[4][j];
   }
   for ( int j = solvedim+1; j < ndim; j++) {
      ind_a += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[0][j];
      ind_b += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[1][j];
      ind_c += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[2][j];
      ind_d += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[3][j];
      if (INC) ind_u += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[4][j];
   }


   int stride_a   = d_cumpads[0][solvedim];
   int stride_b   = d_cumpads[1][solvedim];
   int stride_c   = d_cumpads[2][solvedim];
   int stride_d   = d_cumpads[3][solvedim];
   int stride_u   = d_cumpads[4][solvedim];
   int sys_size = dims.v[solvedim];


   //printf("tid: %d, in_padding: %d, y_size: %d, y_pads: %d, val: %d\n", tid, in_padding, y_size, y_pads, (tid % y_pads));
   if( tid<sys_n && !in_padding ) {
#if 0
      if ( tid == 0 ) {
          printf("a_pad: [% 4d  % 4d  % 4d]\n", a_pads.v[0], a_pads.v[1], a_pads.v[2]);
          printf("b_pad: [% 4d  % 4d  % 4d]\n", b_pads.v[0], b_pads.v[1], b_pads.v[2]);
          printf("c_pad: [% 4d  % 4d  % 4d]\n", c_pads.v[0], c_pads.v[1], c_pads.v[2]);
          printf("d_pad: [% 4d  % 4d  % 4d]\n", d_pads.v[0], d_pads.v[1], d_pads.v[2]);
          printf("u_pad: [% 4d  % 4d  % 4d]\n", u_pads.v[0], u_pads.v[1], u_pads.v[2]);
          printf("dims:  [% 4d  % 4d  % 4d]\n", dims.v[0], dims.v[1], dims.v[2]);
          printf("strides:  [% 4d   % 4d   % 4d   % 4d   % 4d]\n", stride_a, stride_b, stride_c, stride_d, stride_u);
          printf("solvedim:  %d    sys_size:  %d\n", solvedim, sys_size);
      }
      printf("tid: % 8d   ind: [% 6d  % 6d  % 6d  % 6d  % 6d]\n", tid, ind_a, ind_b, ind_c, ind_d, ind_u);
#endif

      trid_strided_multidim_kernel<REAL, VECTOR, INC>(
                a, ind_a, stride_a,
                b, ind_b, stride_b,
                c, ind_c, stride_c,
                d, ind_d, stride_d,
                u, ind_u, stride_u,
                sys_size);
   }
}

template<typename REAL, typename VECTOR, int INC>
void trid_strided_multidim_x_solve_y_padding(const dim3 &grid, const dim3 & block,
                  const VECTOR* __restrict__ a, const int* __restrict__ a_pads,
                  const VECTOR* __restrict__ b, const int* __restrict__ b_pads,
                  const VECTOR* __restrict__ c, const int* __restrict__ c_pads,
                  VECTOR* __restrict__ d, const int* __restrict__ d_pads,
                  VECTOR* __restrict__ u, const int* __restrict__ u_pads,
                  int ndim, int solvedim, int sys_n, const int* __restrict__ dims,
                  const int y_size, const int y_pads) {

   DIM_V v_dims, va_pads, vb_pads, vc_pads, vd_pads, vu_pads;
   memcpy(v_dims.v, dims, ndim*sizeof(int));
   memcpy(va_pads.v, a_pads, ndim*sizeof(int));
   memcpy(vb_pads.v, b_pads, ndim*sizeof(int));
   memcpy(vc_pads.v, c_pads, ndim*sizeof(int));
   memcpy(vd_pads.v, d_pads, ndim*sizeof(int));
   if (INC) memcpy(vu_pads.v, u_pads, ndim*sizeof(int));
   else memset(vu_pads.v, '\0', ndim*sizeof(int));

   trid_strided_multidim_x_solve_y_padding<REAL, VECTOR, INC><<<grid, block>>>(
        a, va_pads, b, vb_pads, c, vc_pads, d, vd_pads, u, vu_pads,
        ndim, solvedim, sys_n, v_dims, y_size, y_pads);
}

#endif
