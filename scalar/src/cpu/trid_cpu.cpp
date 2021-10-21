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

#include "trid_cpu.h"
#include "transpose.hpp"
#include "trid_common.h"
#include "trid_simd.h"
#include <assert.h>
#include <stdio.h>

template <typename REAL>
inline void load(simd_reg_t<REAL> *__restrict__ dst, const REAL *__restrict__ src, int n,
                 int pad) {
  for (int i = 0; i < simd_vec_l<REAL>; i++) {
    dst[i] = simd_load_p(&src[i * pad + n]);
  }
}

template <typename REAL>
inline void load_mask(simd_reg_t<REAL> *__restrict__ dst, const REAL *__restrict__ src,
                      int n, int pad, const simd_mask_t<REAL> &m) {
  for (int i = 0; i < simd_vec_l<REAL>; i++) {
    dst[i] = simd_load_p_m(&src[i * pad + n], m);
  }
}

template <typename REAL>
inline void store(REAL *__restrict__ dst, simd_reg_t<REAL> *__restrict__ src, int n,
                  int pad) {
  for (int i = 0; i < simd_vec_l<REAL>; i++) {
    simd_store_p(&dst[i * pad + n], src[i]);
  }
}

template <typename REAL>
inline void store_mask(REAL *__restrict__ dst, simd_reg_t<REAL> *__restrict__ src, int n,
                       int pad, const simd_mask_t<REAL> &m) {
  for (int i = 0; i < simd_vec_l<REAL>; i++) {
    simd_store_p_m(&dst[i * pad + n], m, src[i]);
  }
}

#ifdef __AVX512F__
template <typename REAL>
void transpose(simd_reg_t<REAL> reg[simd_vec_l<REAL>]){
  if constexpr(std::is_same_v<REAL, double>) {
    transpose8x8_intrinsic(reg);
  } else {
    transpose16x16_intrinsic(reg);
  }
}
#elif __AVX__
template <typename REAL>
void transpose(simd_reg_t<REAL> reg[simd_vec_l<REAL>]){
  if constexpr(std::is_same_v<REAL, double>) {
    transpose4x4_intrinsic(reg);
  } else {
    transpose8x8_intrinsic(reg);
  }
}
#endif

#define LOAD(reg, array, n, N)                                                 \
  load(reg, array, n, N);                                                      \
  transpose<REAL>(reg);
#define LOAD_M2(reg, array, n, N, endmask)                                     \
  load_mask(reg, array, n, N, endmask);                                        \
  transpose<REAL>(reg);
#define STORE(array, reg, n, N)                                                \
  transpose<REAL>(reg);                                                        \
  store(array, reg, n, N);
#define STORE_M2(array, reg, n, N, endmask)                                    \
  transpose<REAL>(reg);                                                        \
  store_mask(array, reg, n, N, endmask);

//
// tridiagonal-x solver; vectorised solution where the system dimension is the
// same as the vectorisation and sys_size < SIMD_VEC
//
template <typename REAL, bool INC, typename MASKTYPE>
void trid_x_transpose_short(const REAL *__restrict a, const REAL *__restrict b,
                            const REAL *__restrict c, REAL *__restrict d,
                            REAL *__restrict u, int sys_size, int sys_pad,
                            const MASKTYPE &endmask, const MASKTYPE &cmask) {

  assert(sys_pad % simd_vec_l<REAL> == 0);
  assert(sys_size <= simd_vec_l<REAL>);

  simd_reg_t<REAL> aa;
  simd_reg_t<REAL> bb;
  simd_reg_t<REAL> cc;
  simd_reg_t<REAL> dd;

  simd_reg_t<REAL> a_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> b_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> c_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> d_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> u_reg[simd_vec_l<REAL>];

  simd_reg_t<REAL> c2[simd_vec_l<REAL>];
  simd_reg_t<REAL> d2[simd_vec_l<REAL>];

  //
  // forward pass
  //
  simd_reg_t<REAL> ones = simd_set1_p<REAL>(1.0F);

  LOAD_M2(a_reg, a, 0, sys_pad, simd_mask_and<REAL>(mask_first<REAL>(), endmask));
  LOAD_M2(b_reg, b, 0, sys_pad, endmask);
  LOAD_M2(c_reg, c, 0, sys_pad, cmask);
  LOAD_M2(d_reg, d, 0, sys_pad, endmask);

  bb    = b_reg[0];
  bb    = ones / bb;
  cc    = c_reg[0];
  cc    = bb * cc;
  dd    = d_reg[0];
  dd    = bb * dd;
  c2[0] = cc;
  d2[0] = dd;

  for (int i = 1; i < sys_size; i++) {
    aa    = a_reg[i];
    bb    = b_reg[i] - aa * cc;
    dd    = d_reg[i] - aa * dd;
    bb    = ones / bb;
    cc    = bb * c_reg[i];
    dd    = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }

  //
  // backward pass
  //

  // backward on last chunk
  d_reg[sys_size - 1] = dd;
  for (int i = sys_size - 2; i >= 0; i--) {
    dd       = d2[i] - c2[i] * dd;
    d_reg[i] = dd;
  }
  if (INC) {
    LOAD_M2(u_reg, u, 0, sys_pad, endmask);
    for (int j = 0; j < sys_size; j++)
      u_reg[j] = u_reg[j] + d_reg[j];
    STORE_M2(u, u_reg, 0, sys_pad, endmask);
  } else {
    STORE_M2(d, d_reg, 0, sys_pad, endmask);
  }
}

//
// tridiagonal-x solver; vectorised solution where the system dimension is the
// same as the vectorisation dimension
//
template <typename REAL, bool INC, typename MASKTYPE>
void trid_x_transpose(const REAL *__restrict a, const REAL *__restrict b,
                      const REAL *__restrict c, REAL *__restrict d,
                      REAL *__restrict u, int sys_size, int sys_pad,
                      const MASKTYPE &endmask, const MASKTYPE &cmask) {
  assert(sys_pad % simd_vec_l<REAL> == 0);

  if (sys_size <= simd_vec_l<REAL>) {
    trid_x_transpose_short<REAL, INC>(a, b, c, d, u, sys_size, sys_pad, endmask,
                                      cmask);
    return;
  }

  simd_reg_t<REAL> aa;
  simd_reg_t<REAL> bb;
  simd_reg_t<REAL> cc;
  simd_reg_t<REAL> dd;

  simd_reg_t<REAL> a_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> b_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> c_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> d_reg[simd_vec_l<REAL>];
  simd_reg_t<REAL> u_reg[simd_vec_l<REAL>];

  simd_reg_t<REAL> c2[N_MAX];
  simd_reg_t<REAL> d2[N_MAX];

  //
  // forward pass
  //
  simd_reg_t<REAL> ones = simd_set1_p<REAL>(1.0F);

  LOAD_M2(a_reg, a, 0, sys_pad, mask_first<REAL>());
  LOAD(b_reg, b, 0, sys_pad); // load this way all except a, and do a separately
  LOAD(c_reg, c, 0, sys_pad);
  LOAD(d_reg, d, 0, sys_pad);

  bb    = b_reg[0];
  bb    = ones / bb;
  cc    = c_reg[0];
  cc    = bb * cc;
  dd    = d_reg[0];
  dd    = bb * dd;
  c2[0] = cc;
  d2[0] = dd;

  for (int i = 1; i < simd_vec_l<REAL>; i++) {
    aa = a_reg[i];
    bb = b_reg[i] - aa * cc;
    dd = d_reg[i] - aa * dd;
    bb    = ones / bb;
    cc    = bb * c_reg[i];
    dd    = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }

  // forward pass on inner SIMD vectors
  for (int n = simd_vec_l<REAL>; n < ROUND_DOWN(sys_size - 1, simd_vec_l<REAL>);
       n += simd_vec_l<REAL>) {
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    for (int i = 0; i < simd_vec_l<REAL>; i++) {
      aa = a_reg[i];
      bb = b_reg[i] - aa * cc;
      dd = d_reg[i] - aa * dd;
      bb        = ones / bb;
      cc        = bb * c_reg[i];
      dd        = bb * dd;
      c2[n + i] = cc;
      d2[n + i] = dd;
    }
  }

  // forward on remainder
  // perform forward on incomplete SIMD vector
  // Loads are guarded with endmask
  int n = ROUND_DOWN(sys_size - 1, simd_vec_l<REAL>);
  LOAD_M2(a_reg, a, n, sys_pad, endmask);
  LOAD_M2(b_reg, b, n, sys_pad, endmask);
  LOAD_M2(c_reg, c, n, sys_pad, cmask);
  LOAD_M2(d_reg, d, n, sys_pad, endmask);
  for (int i = 0; (n + i) < sys_size; i++) {
    aa = a_reg[i];
    bb = b_reg[i] - aa * cc;
    dd = d_reg[i] - aa * dd;
    bb        = ones / bb;
    cc        = bb * c_reg[i];
    dd        = bb * dd;
    c2[n + i] = cc;
    d2[n + i] = dd;
  }

  // backward on last chunk
  n = ROUND_DOWN(sys_size, sys_pad);
  if (sys_size != sys_pad) {
    d_reg[sys_size - 1 - n] = dd;
    for (int i = sys_size - n - 2; i >= 0; i--) {
      dd       = d2[n + i] - c2[n + i] * dd;
      d_reg[i] = dd;
    }
    if (INC) {
      LOAD_M2(u_reg, u, n, sys_pad, endmask);
      for (int j = 0; j < sys_size; j++)
        u_reg[j] = u_reg[j] + d_reg[j];
      STORE_M2(u, u_reg, n, sys_pad, endmask);
    } else {
      STORE_M2(d, d_reg, n, sys_pad, endmask);
    }
  } else {
    d_reg[simd_vec_l<REAL> - 1] = dd;
    n -= simd_vec_l<REAL>;
    for (int i = simd_vec_l<REAL> - 2; i >= 0; i--) {
      dd       = d2[n + i] - c2[n + i] * dd;
      d_reg[i] = dd;
    }
    if (INC) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < simd_vec_l<REAL>; j++)
        u_reg[j] = u_reg[j] + d_reg[j];
      STORE(u, u_reg, n, sys_pad);
    } else {
      STORE(d, d_reg, n, sys_pad);
    }
  }
  n -= simd_vec_l<REAL>;

  //
  // backward pass
  //

  for (; n >= 0; n -= simd_vec_l<REAL>) {
    for (int i = (simd_vec_l<REAL> - 1); i >= 0; i--) {
      dd       = d2[n + i] - c2[n + i] * dd;
      d_reg[i] = dd;
    }
    if (INC) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < simd_vec_l<REAL>; j++)
        u_reg[j] = u_reg[j] + d_reg[j];
      STORE(u, u_reg, n, sys_pad);
    } else {
      STORE(d, d_reg, n, sys_pad);
    }
  }
}

//
// tridiagonal solver; vectorised solution where the system dimension is not
// the same as the vectorisation dimension
//
template <typename REAL, bool INC>
void trid_scalar_vec(const REAL *__restrict h_a, const REAL *__restrict h_b,
                     const REAL *__restrict h_c, REAL *__restrict h_d,
                     REAL *__restrict h_u, int N, int stride) {

  int ind = 0;
  simd_reg_t<REAL> aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];

  simd_reg_t<REAL> ones = simd_set1_p<REAL>(1.0f);

  //
  // forward pass
  //
  bb    = ones / simd_load_p(&h_b[0]);
  cc    = bb * simd_load_p(&h_c[0]);
  dd    = bb * simd_load_p(&h_d[0]);
  c2[0] = cc;
  d2[0] = dd;

  for (int i = 1; i < N - 1; i++) {
    ind   = ind + stride;
    aa    = simd_load_p(&h_a[ind]);
    bb    = simd_load_p(&h_b[ind]) - aa * cc;
    dd    = simd_load_p(&h_d[ind]) - aa * dd;
    bb    = ones / bb;
    cc    = bb * simd_load_p(&h_c[ind]);
    dd    = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }

  {
    // last iteration access to c leads to segfault
    ind = ind + stride;
    aa  = simd_load_p(&h_a[ind]);
    bb  = simd_load_p(&h_b[ind]) - aa * cc;
    dd  = simd_load_p(&h_d[ind]) - aa * dd;
    bb  = ones / bb;
    dd  = bb * dd;
  }
  //
  // reverse pass
  //
  if (INC)
    simd_store_p(&h_u[ind], simd_load_p(&h_u[ind]) + dd);
  else
    simd_store_p(&h_d[ind], dd);
  for (int i = N - 2; i >= 0; i--) {
    ind = ind - stride;
    dd  = d2[i] - c2[i] * dd;
    if (INC)
      simd_store_p(&h_u[ind], simd_load_p(&h_u[ind]) + dd);
    else
      simd_store_p(&h_d[ind], dd);
  }
}

//
// tridiagonal solver; simple non-vectorised solution
//
template <typename REAL, bool INC>
void trid_scalar(const REAL *__restrict a, const REAL *__restrict b,
                 const REAL *__restrict c, REAL *__restrict d,
                 REAL *__restrict u, int N, int stride) {
  int ind = 0;
  REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = 1.0F / b[0];
  cc    = bb * c[0];
  dd    = bb * d[0];
  c2[0] = cc;
  d2[0] = dd;

  for (int i = 1; i < N - 1; i++) {
    ind   = ind + stride;
    aa    = a[ind];
    bb    = b[ind] - aa * cc;
    dd    = d[ind] - aa * dd;
    bb    = 1.0F / bb;
    cc    = bb * c[ind];
    dd    = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  {
    // last iteration access to c leads to segfault
    ind = ind + stride;
    aa  = a[ind];
    bb  = b[ind] - aa * cc;
    dd  = d[ind] - aa * dd;
    bb  = 1.0F / bb;
    dd  = bb * dd;
  }

  //
  // reverse pass
  //
  if (INC)
    u[ind] += dd;
  else
    d[ind] = dd;
  for (int i = N - 2; i >= 0; i--) {
    ind = ind - stride;
    dd  = d2[i] - c2[i] * dd;
    if (INC)
      u[ind] += dd;
    else
      d[ind] = dd;
  }
}

//
// Function for selecting the proper setup for solve in a specific dimension
//
template <typename REAL, bool INC>
void tridMultiDimBatchSolve(const REAL *a, const REAL *b, const REAL *c,
                            REAL *d, REAL *u, int ndim, int solvedim,
                            const int *dims_p, const int *pads_p) {

  int dims[3] = {1, 1, 1};
  int pads[3] = {1, 1, 1};
  for (int i = 0; i < ndim; i++) {
    dims[i] = dims_p[i];
    pads[i] = pads_p[i];
  }
  if (solvedim == 0) {
    int sys_stride = 1; // Stride between the consecutive elements of a system
    int sys_size   = dims[0]; // Size (length) of a system
    int sys_pads = pads[0]; // Padded sizes along each ndim number of dimensions

    if (sys_pads % simd_vec_l<REAL> == 0) {
      simd_mask_t<REAL> endmask = create_endmask<REAL>(sys_size);
      simd_mask_t<REAL> cmask   = create_cmask<REAL>(sys_size);
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < ROUND_DOWN(dims[1], simd_vec_l<REAL>); j += simd_vec_l<REAL>) {
          int ind = k * pads[0] * pads[1] + j * pads[0];
          trid_x_transpose<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind],
                                      &u[ind], sys_size, sys_pads, endmask,
                                      cmask);
        }
      }
      if (ROUND_DOWN(dims[1], simd_vec_l<REAL>) <
          dims[1]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
        for (int k = 0; k < dims[2]; k++) {
          for (int j = ROUND_DOWN(dims[1], simd_vec_l<REAL>); j < dims[1]; j++) {
            int ind = k * pads[0] * pads[1] + j * pads[0];
            trid_scalar<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                   sys_size, sys_stride);
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < dims[1]; j++) {
          int ind = k * pads[0] * pads[1] + j * pads[0];
          trid_scalar<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                 sys_size, sys_stride);
        }
      }
    }
  } else if (solvedim == 1) {
    int sys_stride =
        pads[0]; // Stride between the consecutive elements of a system
    int sys_size = dims[1]; // Size (length) of a system

#pragma omp parallel for collapse(2)
    for (int k = 0; k < dims[2]; k++) {
      for (int i = 0; i < ROUND_DOWN(dims[0], simd_vec_l<REAL>); i += simd_vec_l<REAL>) {
        int ind = k * pads[0] * pads[1] + i;
        trid_scalar_vec<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind],
                                           &u[ind], sys_size, sys_stride);
      }
    }
    if (ROUND_DOWN(dims[0], simd_vec_l<REAL>) <
        dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int i = ROUND_DOWN(dims[0], simd_vec_l<REAL>); i < dims[0]; i++) {
          int ind = k * pads[0] * pads[1] + i;
          trid_scalar<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                 sys_size, sys_stride);
        }
      }
    }
  } else if (solvedim == 2) {
    int sys_stride =
        pads[0] *
        pads[1]; // Stride between the consecutive elements of a system
    int sys_size = dims[2]; // Size (length) of a system

#pragma omp parallel for collapse(2) // Interleaved scheduling for better data
                                     // locality and thus lower TLB miss rate
    for (int j = 0; j < dims[1]; j++) {
      for (int i = 0; i < ROUND_DOWN(dims[0], simd_vec_l<REAL>); i += simd_vec_l<REAL>) {
        int ind = j * pads[0] + i;
        trid_scalar_vec<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind],
                                           &u[ind], sys_size, sys_stride);
      }
    }
    if (ROUND_DOWN(dims[0], simd_vec_l<REAL>) <
        dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int j = 0; j < dims[1]; j++) {
        for (int i = ROUND_DOWN(dims[0], simd_vec_l<REAL>); i < dims[0]; i++) {
          int ind = j * pads[0] + i;
          trid_scalar<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                 sys_size, sys_stride);
        }
      }
    }
  }
}

tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b,
                                   const float *c, float *d, float *u, int ndim,
                                   int solvedim, const int *dims,
                                   const int *pads) {
  tridMultiDimBatchSolve<float, false>(a, b, c, d, u, ndim, solvedim, dims,
                                       pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads) {
  tridMultiDimBatchSolve<float, true>(a, b, c, d, u, ndim, solvedim, dims,
                                      pads);
  return TRID_STATUS_SUCCESS;
}

void trid_scalarS(const float *__restrict a, const float *__restrict b,
                  const float *__restrict c, float *__restrict d,
                  float *__restrict u, int N, int stride) {

  trid_scalar<float, false>(a, b, c, d, u, N, stride);
}

void trid_x_transposeS(const float *__restrict a, const float *__restrict b,
                       const float *__restrict c, float *__restrict d,
                       float *__restrict u, int sys_size, int sys_pad) {
  simd_mask_t<float> endmask = create_endmask<double>(sys_size);
  simd_mask_t<float> cmask   = create_cmask<double>(sys_size);
  trid_x_transpose<float, false>(a, b, c, d, u, sys_size, sys_pad, endmask,
                                 cmask);
}

void trid_scalar_vecS(const float *__restrict a, const float *__restrict b,
                      const float *__restrict c, float *__restrict d,
                      float *__restrict u, int N, int stride) {

  trid_scalar_vec<float, false>(a, b, c, d, u, N, stride);
}

void trid_scalar_vecSInc(const float *__restrict a, const float *__restrict b,
                         const float *__restrict c, float *__restrict d,
                         float *__restrict u, int N, int stride) {

  trid_scalar_vec<float, true>(a, b, c, d, u, N, stride);
}

tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b,
                                   const double *c, double *d, double *u,
                                   int ndim, int solvedim, const int *dims,
                                   const int *pads) {
  tridMultiDimBatchSolve<double, false>(a, b, c, d, u, ndim, solvedim, dims,
                                        pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads) {
  tridMultiDimBatchSolve<double, true>(a, b, c, d, u, ndim, solvedim, dims,
                                       pads);
  return TRID_STATUS_SUCCESS;
}

void trid_scalarD(const double *__restrict a, const double *__restrict b,
                  const double *__restrict c, double *__restrict d,
                  double *__restrict u, int N, int stride) {

  trid_scalar<double, false>(a, b, c, d, u, N, stride);
}

void trid_x_transposeD(const double *__restrict a, const double *__restrict b,
                       const double *__restrict c, double *__restrict d,
                       double *__restrict u, int sys_size, int sys_pad) {

  simd_mask_t<double> endmask = create_endmask<double>(sys_size);
  simd_mask_t<double> cmask   = create_cmask<double>(sys_size);
  trid_x_transpose<double, false>(a, b, c, d, u, sys_size, sys_pad, endmask,
                                  cmask);
}

void trid_scalar_vecD(const double *__restrict a, const double *__restrict b,
                      const double *__restrict c, double *__restrict d,
                      double *__restrict u, int N, int stride) {

  trid_scalar_vec<double, false>(a, b, c, d, u, N, stride);
}

void trid_scalar_vecDInc(const double *__restrict a, const double *__restrict b,
                         const double *__restrict c, double *__restrict d,
                         double *__restrict u, int N, int stride) {

  trid_scalar_vec<double, true>(a, b, c, d, u, N, stride);
}
