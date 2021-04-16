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

// Written by Gabor Daniel Balogh, Pazmany Peter Catholic University,
// balogh.gabor.daniel@itk.ppke.hu, 2020
// Simple implementations of MPI solver along hiher dimensions.

#ifndef TRID_STRIDED_MULTIDIM_GPU_MPI__
#define TRID_STRIDED_MULTIDIM_GPU_MPI__

typedef struct {
  int v[8];
} DIM_V;

/*
 * Modified Thomas forward pass in y or higher dimensions.
 * Each array should have a size of sys_n*sys_size.
 * The layout and indexing of aa, cc, and dd are the same as of a, c, d
 * respectively
 * The boundaries array has a size of sys_n*6 and will hold the first and last
 * elements of aa, cc, and dd for each system
 *
 */
template <typename REAL>
__device__ void trid_strided_multidim_forward_kernel(
    const REAL *__restrict__ a, int ind_a, int stride_a,
    const REAL *__restrict__ b, int ind_b, int stride_b,
    const REAL *__restrict__ c, int ind_c, int stride_c,
    const REAL *__restrict__ d, int ind_d, int stride_d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ dd, REAL *__restrict__ boundaries,
    int ind_bound, int sys_size) {
  //
  // forward pass
  //
  REAL bb;
  for (int i = 0; i < 2; ++i) {
    bb                       = static_cast<REAL>(1.0) / b[ind_b + i * stride_b];
    cc[ind_c + i * stride_c] = bb * c[ind_c + i * stride_c];
    aa[ind_a + i * stride_a] = bb * a[ind_a + i * stride_a];
    dd[ind_d + i * stride_d] = bb * d[ind_d + i * stride_d];
  }

  if (sys_size >= 3) {
    // Eliminate lower off-diagonal
    for (int i = 2; i < sys_size; ++i) {
      bb = static_cast<REAL>(1.0) /
           (b[ind_b + i * stride_b] -
            a[ind_a + i * stride_a] * cc[ind_c + (i - 1) * stride_c]);
      dd[ind_d + i * stride_d] =
          (d[ind_d + i * stride_d] -
           a[ind_a + i * stride_a] * dd[ind_d + (i - 1) * stride_d]) *
          bb;
      aa[ind_a + i * stride_a] =
          (-a[ind_a + i * stride_a] * aa[ind_a + (i - 1) * stride_a]) * bb;
      cc[ind_c + i * stride_c] = c[ind_c + i * stride_c] * bb;
    }
    // Eliminate upper off-diagonal
    for (int i = sys_size - 3; i > 0; --i) {
      dd[ind_d + i * stride_d] =
          dd[ind_d + i * stride_d] -
          cc[ind_c + i * stride_c] * dd[ind_d + (i + 1) * stride_d];
      aa[ind_a + i * stride_a] =
          aa[ind_a + i * stride_a] -
          cc[ind_c + i * stride_c] * aa[ind_a + (i + 1) * stride_a];
      cc[ind_c + i * stride_c] =
          -cc[ind_c + i * stride_c] * cc[ind_c + (i + 1) * stride_c];
    }
    bb = static_cast<REAL>(1.0) /
         (static_cast<REAL>(1.0) - cc[ind_c] * aa[ind_a + stride_a]);
    dd[ind_d] = bb * (dd[ind_d] - cc[ind_c] * dd[ind_d + stride_d]);
    aa[ind_a] = bb * aa[ind_a];
    cc[ind_c] = bb * (-cc[ind_c] * cc[ind_c + stride_c]);
  }
  // prepare boundaries for communication
  boundaries[ind_bound + 0] = aa[ind_a];
  boundaries[ind_bound + 1] = aa[ind_a + (sys_size - 1) * stride_a];
  boundaries[ind_bound + 2] = cc[ind_c];
  boundaries[ind_bound + 3] = cc[ind_c + (sys_size - 1) * stride_c];
  boundaries[ind_bound + 4] = dd[ind_d];
  boundaries[ind_bound + 5] = dd[ind_d + (sys_size - 1) * stride_d];
}

template <typename REAL>
__global__ void trid_strided_multidim_forward(
    const REAL *__restrict__ a, const DIM_V a_pads, const REAL *__restrict__ b,
    const DIM_V b_pads, const REAL *__restrict__ c, const DIM_V c_pads,
    const REAL *__restrict__ d, const DIM_V d_pads, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ dd, REAL *__restrict__ boundaries,
    int ndim, int solvedim, int sys_n, const DIM_V dims, int sys_offset = 0) {
  // thread ID in block
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  if (solvedim < 1 || solvedim > ndim) return; /* Just hints to the compiler */

  int __shared__ d_cumdims[MAXDIM + 1];
  int __shared__ d_cumpads[4][MAXDIM + 1];

  /* Build up d_cumpads and d_cumdims */
  if (tid < 5) {
    int *tgt       = (tid == 0) ? d_cumdims : d_cumpads[tid - 1];
    const int *src = NULL;
    switch (tid) {
    case 0: src = dims.v; break;
    case 1: src = a_pads.v; break;
    case 2: src = b_pads.v; break;
    case 3: src = c_pads.v; break;
    case 4: src = d_pads.v; break;
    }

    tgt[0] = 1;
    for (int i = 0; i < ndim; i++) {
      tgt[i + 1] = tgt[i] * src[i];
    }
  }
  __syncthreads();
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  tid = sys_offset + threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  int ind_a     = 0;
  int ind_b     = 0;
  int ind_c     = 0;
  int ind_d     = 0;
  int ind_bound = tid * 6;

  for (int j = 0; j < solvedim; j++) {
    ind_a += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[0][j];
    ind_b += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[1][j];
    ind_c += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[2][j];
    ind_d += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[3][j];
  }
  for (int j = solvedim + 1; j < ndim; j++) {
    ind_a += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[0][j];
    ind_b += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[1][j];
    ind_c += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[2][j];
    ind_d += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[3][j];
  }
  int stride_a = d_cumpads[0][solvedim];
  int stride_b = d_cumpads[1][solvedim];
  int stride_c = d_cumpads[2][solvedim];
  int stride_d = d_cumpads[3][solvedim];
  int sys_size = dims.v[solvedim];

  if (tid < sys_offset + sys_n) {
    trid_strided_multidim_forward_kernel<REAL>(
        a, ind_a, stride_a, b, ind_b, stride_b, c, ind_c, stride_c, d, ind_d,
        stride_d, aa, cc, dd, boundaries, ind_bound, sys_size);
  }
}

/*
 * Modified Thomas backward pass in y or higher dimensions.
 * Each array should have a size of sys_n*sys_size.
 * The layout and indexing of aa, cc, and dd are the same as of a, c, d
 * respectively
 * The boundaries array has a size of sys_n*2 and hold the first and last
 * elements of dd for each system
 *
 */
template <typename REAL, int INC>
__device__ void trid_strided_multidim_backward_kernel(
    const REAL *__restrict__ aa, int ind_a, int stride_a,
    const REAL *__restrict__ cc, int ind_c, int stride_c,
    const REAL *__restrict__ dd, REAL *__restrict__ d, int ind_d, int stride_d,
    REAL *__restrict__ u, int ind_u, int stride_u,
    const REAL *__restrict__ boundaries, int ind_bound, int sys_size) {
  //
  // reverse pass
  //
  REAL dd0 = boundaries[ind_bound], dd_last = boundaries[ind_bound + 1];
  if (INC == 0)
    d[ind_d] = dd0;
  else
    u[ind_u] += dd0;
  for (int i = 1; i < sys_size - 1; i++) {
    REAL res = dd[ind_d + i * stride_d] - aa[ind_a + i * stride_a] * dd0 -
               cc[ind_c + i * stride_c] * dd_last;
    if (INC == 0)
      d[ind_d + i * stride_d] = res;
    else
      u[ind_u + i * stride_u] += res;
  }
  if (INC == 0)
    d[ind_d + (sys_size - 1) * stride_d] = dd_last;
  else
    u[ind_u + (sys_size - 1) * stride_u] += dd_last;
}

template <typename REAL, int INC>
__global__ void trid_strided_multidim_backward(
    const REAL *__restrict__ aa, const DIM_V a_pads,
    const REAL *__restrict__ cc, const DIM_V c_pads,
    const REAL *__restrict__ dd, REAL *__restrict__ d, const DIM_V d_pads,
    REAL *__restrict__ u, const DIM_V u_pads,
    const REAL *__restrict__ boundaries, int ndim, int solvedim, int sys_n,
    const DIM_V dims, int sys_offset = 0) {
  // thread ID in block
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  if (solvedim < 1 || solvedim > ndim) return; /* Just hints to the compiler */

  int __shared__ d_cumdims[MAXDIM + 1];
  int __shared__ d_cumpads[4][MAXDIM + 1];

  /* Build up d_cumpads and d_cumdims */
  if (tid < 5) {
    int *tgt       = (tid == 0) ? d_cumdims : d_cumpads[tid - 1];
    const int *src = NULL;
    switch (tid) {
    case 0: src = dims.v; break;
    case 1: src = a_pads.v; break;
    case 2: src = c_pads.v; break;
    case 3: src = d_pads.v; break;
    case 4: src = u_pads.v; break;
    }

    tgt[0] = 1;
    for (int i = 0; i < ndim; i++) {
      tgt[i + 1] = tgt[i] * src[i];
    }
  }
  __syncthreads();
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  tid = sys_offset + threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  int ind_a     = 0;
  int ind_c     = 0;
  int ind_d     = 0;
  int ind_u     = 0;
  int ind_bound = tid * 2; // 2 values per system since it hold only dd

  for (int j = 0; j < solvedim; j++) {
    ind_a += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[0][j];
    ind_c += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[1][j];
    ind_d += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[2][j];
    if (INC) ind_u += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[3][j];
  }
  for (int j = solvedim + 1; j < ndim; j++) {
    ind_a += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[0][j];
    ind_c += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[1][j];
    ind_d += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[2][j];
    if (INC)
      ind_u += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
               d_cumpads[3][j];
  }
  int stride_a = d_cumpads[0][solvedim];
  int stride_c = d_cumpads[1][solvedim];
  int stride_d = d_cumpads[2][solvedim];
  int stride_u = d_cumpads[3][solvedim];
  int sys_size = dims.v[solvedim];

  if (tid < sys_offset + sys_n) {
    trid_strided_multidim_backward_kernel<REAL, INC>(
        aa, ind_a, stride_a, cc, ind_c, stride_c, dd, d, ind_d, stride_d, u,
        ind_u, stride_u, boundaries, ind_bound, sys_size);
  }
}

#endif /* ifndef TRID_STRIDED_MULTIDIM_GPU_MPI__ */
