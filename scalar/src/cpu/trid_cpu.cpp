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

// #include "trid_cpu.h"
#include "tridsolver.h"
#include "trid_cpu.h"
#include "transpose.hpp"
#include "trid_common.h"
#include "trid_simd.h"
#include <assert.h>
#include <stdio.h>
#define ROUND_DOWN(N, step) (((N) / (step)) * step)

inline void load(SIMD_REG *__restrict__ dst, const FP *__restrict__ src, int n,
                 int pad) {
  for (int i = 0; i < SIMD_VEC; i++) {
    dst[i] = SIMD_LOAD_P(&src[i * pad + n]);
  }
}

inline void store(FP *__restrict__ dst, SIMD_REG *__restrict__ src, int n,
                  int pad) {
  for (int i = 0; i < SIMD_VEC; i++) {
    SIMD_STORE_P(&dst[i * pad + n], src[i]);
  }
}

#ifdef __AVX512F__
#  if FPPREC == 0
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose16x16_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose16x16_intrinsic(reg);                                           \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#elif __AVX__
#  if FPPREC == 0
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose4x4_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose4x4_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#endif

//
// tridiagonal-x solver; vectorised solution where the system dimension is the
// same as the vectorisation dimension
//
template <typename REAL, bool INC>
void trid_x_transpose(const REAL *__restrict a, const REAL *__restrict b,
                      const REAL *__restrict c, REAL *__restrict d,
                      REAL *__restrict u, int sys_size, int sys_pad) {
  assert(sys_pad % SIMD_VEC == 0);

  SIMD_REG aa;
  SIMD_REG bb;
  SIMD_REG cc;
  SIMD_REG dd;

  SIMD_REG a_reg[SIMD_VEC];
  SIMD_REG b_reg[SIMD_VEC];
  SIMD_REG c_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  SIMD_REG u_reg[SIMD_VEC];

  SIMD_REG c2[N_MAX];
  SIMD_REG d2[N_MAX];

  //
  // forward pass
  //
  SIMD_REG ones = SIMD_SET1_P(1.0F);

  LOAD(a_reg, a, 0, sys_pad);
  LOAD(b_reg, b, 0, sys_pad);
  LOAD(c_reg, c, 0, sys_pad);
  LOAD(d_reg, d, 0, sys_pad);

  bb    = b_reg[0];
  bb    = SIMD_DIV_P(ones, bb);
  cc    = c_reg[0];
  cc    = SIMD_MUL_P(bb, cc);
  dd    = d_reg[0];
  dd    = SIMD_MUL_P(bb, dd);
  c2[0] = cc;
  d2[0] = dd;

  for (int i = 1; i < SIMD_VEC; i++) {
    aa = a_reg[i];
#ifdef __AVX512F__
    bb = SIMD_FNMADD_P(aa, cc, b_reg[i]);
    dd = SIMD_FNMADD_P(aa, dd, d_reg[i]);
#else
    bb = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa, cc));
    dd = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa, dd));
#endif
    bb    = SIMD_DIV_P(ones, bb);
    cc    = SIMD_MUL_P(bb, c_reg[i]);
    dd    = SIMD_MUL_P(bb, dd);
    c2[i] = cc;
    d2[i] = dd;
  }

  for (int n = SIMD_VEC; n < ROUND_DOWN(sys_size, SIMD_VEC); n += SIMD_VEC) {
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    for (int i = 0; i < SIMD_VEC; i++) {
      aa = a_reg[i];
#ifdef __AVX512F__
      bb = SIMD_FNMADD_P(aa, cc, b_reg[i]);
      dd = SIMD_FNMADD_P(aa, dd, d_reg[i]);
#else
      bb = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa, cc));
      dd = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa, dd));
#endif
      bb        = SIMD_DIV_P(ones, bb);
      cc        = SIMD_MUL_P(bb, c_reg[i]);
      dd        = SIMD_MUL_P(bb, dd);
      c2[n + i] = cc;
      d2[n + i] = dd;
    }
  }

  // forward on remainder

  if (sys_size != sys_pad) {
    // perform a noncomplete forward
    // Loads are safe since sys_pads must be a multiple of SIMD_WIDTH, and we
    // don't use data in paddings

    int n = ROUND_DOWN(sys_size, SIMD_VEC);
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    for (int i = 0; (n + i) < sys_size; i++) {
      aa = a_reg[i];
#ifdef __AVX512F__
      bb = SIMD_FNMADD_P(aa, cc, b_reg[i]);
      dd = SIMD_FNMADD_P(aa, dd, d_reg[i]);
#else
      bb = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa, cc));
      dd = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa, dd));
#endif
      bb        = SIMD_DIV_P(ones, bb);
      cc        = SIMD_MUL_P(bb, c_reg[i]);
      dd        = SIMD_MUL_P(bb, dd);
      c2[n + i] = cc;
      d2[n + i] = dd;
    }
  }

  // backward on last chunk
  int n = ROUND_DOWN(sys_size, SIMD_VEC);
  if (sys_size != sys_pad) {
    d_reg[sys_size - 1 - n] = dd;
    for (int i = sys_size - n - 2; i >= 0; i--) {
      dd       = SIMD_SUB_P(d2[n + i], SIMD_MUL_P(c2[n + i], dd));
      d_reg[i] = dd;
    }
    if (INC) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < sys_size - n; j++)
        u_reg[j] = SIMD_ADD_P(u_reg[j], d_reg[j]);
      STORE(u, u_reg, n, sys_pad);
    } else {
      STORE(d, d_reg, n, sys_pad);
    }
  } else {
    d_reg[SIMD_VEC - 1] = dd;
    n -= SIMD_VEC;
    for (int i = SIMD_VEC - 2; i >= 0; i--) {
      dd       = SIMD_SUB_P(d2[n + i], SIMD_MUL_P(c2[n + i], dd));
      d_reg[i] = dd;
    }
    if (INC) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < SIMD_VEC; j++)
        u_reg[j] = SIMD_ADD_P(u_reg[j], d_reg[j]);
      STORE(u, u_reg, n, sys_pad);
    } else {
      STORE(d, d_reg, n, sys_pad);
    }
  }
  n -= SIMD_VEC;

  //
  // backward pass
  //

  for (; n >= 0; n -= SIMD_VEC) {
    for (int i = (SIMD_VEC - 1); i >= 0; i--) {
      dd       = SIMD_SUB_P(d2[n + i], SIMD_MUL_P(c2[n + i], dd));
      d_reg[i] = dd;
    }
    if (INC) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < SIMD_VEC; j++)
        u_reg[j] = SIMD_ADD_P(u_reg[j], d_reg[j]);
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
template <typename REAL, typename VECTOR, bool INC>
void trid_scalar_vec(const REAL *__restrict h_a, const REAL *__restrict h_b,
                     const REAL *__restrict h_c, REAL *__restrict h_d,
                     REAL *__restrict h_u, int N, int stride) {

  int i, ind = 0;
  VECTOR aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];

  VECTOR ones = SIMD_SET1_P(1.0f);

  //
  // forward pass
  //
  bb    = ones / SIMD_LOAD_P(&h_b[0]);
  cc    = bb * SIMD_LOAD_P(&h_c[0]);
  dd    = bb * SIMD_LOAD_P(&h_d[0]);
  c2[0] = cc;
  d2[0] = dd;

  for (i = 1; i < N; i++) {
    ind   = ind + stride;
    aa    = SIMD_LOAD_P(&h_a[ind]);
    bb    = SIMD_LOAD_P(&h_b[ind]) - aa * cc;
    dd    = SIMD_LOAD_P(&h_d[ind]) - aa * dd;
    bb    = ones / bb;
    cc    = bb * SIMD_LOAD_P(&h_c[ind]);
    dd    = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  if (INC)
    SIMD_STORE_P(&h_u[ind], SIMD_LOAD_P(&h_u[ind]) + dd);
  else
    SIMD_STORE_P(&h_d[ind], dd);
  for (i = N - 2; i >= 0; i--) {
    ind = ind - stride;
    dd  = d2[i] - c2[i] * dd;
    if (INC)
      SIMD_STORE_P(&h_u[ind], SIMD_LOAD_P(&h_u[ind]) + dd);
    else
      SIMD_STORE_P(&h_d[ind], dd);
  }
}

//
// tridiagonal solver; simple non-vectorised solution
//
template <typename REAL, bool INC>
void trid_scalar(const REAL *__restrict a, const REAL *__restrict b,
                 const REAL *__restrict c, REAL *__restrict d,
                 REAL *__restrict u, int N, int stride) {
  int i, ind = 0;
  REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = 1.0F / b[0];
  cc    = bb * c[0];
  dd    = bb * d[0];
  c2[0] = cc;
  d2[0] = dd;

  for (i = 1; i < N; i++) {
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
  //
  // reverse pass
  //
  if (INC)
    u[ind] += dd;
  else
    d[ind] = dd;
  for (i = N - 2; i >= 0; i--) {
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

    if (sys_pads % SIMD_VEC == 0) {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < ROUND_DOWN(dims[1], SIMD_VEC); j += SIMD_VEC) {
          int ind = k * pads[0] * pads[1] + j * pads[0];
          trid_x_transpose<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind],
                                      &u[ind], sys_size, sys_pads);
        }
      }
      if (ROUND_DOWN(dims[1], SIMD_VEC) <
          dims[1]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
        for (int k = 0; k < dims[2]; k++) {
          for (int j = ROUND_DOWN(dims[1], SIMD_VEC); j < dims[1]; j++) {
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
      for (int i = 0; i < ROUND_DOWN(dims[0], SIMD_VEC); i += SIMD_VEC) {
        int ind = k * pads[0] * pads[1] + i;
        trid_scalar_vec<REAL, VECTOR, INC>(&a[ind], &b[ind], &c[ind], &d[ind],
                                           &u[ind], sys_size, sys_stride);
      }
    }
    if (ROUND_DOWN(dims[0], SIMD_VEC) <
        dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int i = ROUND_DOWN(dims[0], SIMD_VEC); i < dims[0]; i++) {
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
      for (int i = 0; i < ROUND_DOWN(dims[0], SIMD_VEC); i += SIMD_VEC) {
        int ind = j * pads[0] + i;
        trid_scalar_vec<REAL, VECTOR, INC>(&a[ind], &b[ind], &c[ind], &d[ind],
                                           &u[ind], sys_size, sys_stride);
      }
    }
    if (ROUND_DOWN(dims[0], SIMD_VEC) <
        dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int j = 0; j < dims[1]; j++) {
        for (int i = ROUND_DOWN(dims[0], SIMD_VEC); i < dims[0]; i++) {
          int ind = j * pads[0] + i;
          trid_scalar<REAL, INC>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                 sys_size, sys_stride);
        }
      }
    }
  }
}

#if FPPREC == 0

tridStatus_t tridSmtsvStridedBatch(const TridParams *, const float *a,
                                   const float *b, const float *c, float *d,
                                   float *u, int ndim, int solvedim,
                                   const int *dims, const int *pads) {
  tridMultiDimBatchSolve<float, false>(a, b, c, d, u, ndim, solvedim, dims,
                                       pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchInc(const TridParams *, const float *a,
                                      const float *b, const float *c, float *d,
                                      float *u, int ndim, int solvedim,
                                      const int *dims, const int *pads) {
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

  trid_x_transpose<float, false>(a, b, c, d, u, sys_size, sys_pad);
}

void trid_scalar_vecS(const float *__restrict a, const float *__restrict b,
                      const float *__restrict c, float *__restrict d,
                      float *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, false>(a, b, c, d, u, N, stride);
}

void trid_scalar_vecSInc(const float *__restrict a, const float *__restrict b,
                         const float *__restrict c, float *__restrict d,
                         float *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, true>(a, b, c, d, u, N, stride);
}

#elif FPPREC == 1

tridStatus_t tridDmtsvStridedBatch(const TridParams *, const double *a,
                                   const double *b, const double *c, double *d,
                                   double *u, int ndim, int solvedim,
                                   const int *dims, const int *pads) {
  tridMultiDimBatchSolve<double, false>(a, b, c, d, u, ndim, solvedim, dims,
                                        pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchInc(const TridParams *, const double *a,
                                      const double *b, const double *c,
                                      double *d, double *u, int ndim,
                                      int solvedim, const int *dims,
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

  trid_x_transpose<double, false>(a, b, c, d, u, sys_size, sys_pad);
}

void trid_scalar_vecD(const double *__restrict a, const double *__restrict b,
                      const double *__restrict c, double *__restrict d,
                      double *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, false>(a, b, c, d, u, N, stride);
}

void trid_scalar_vecDInc(const double *__restrict a, const double *__restrict b,
                         const double *__restrict c, double *__restrict d,
                         double *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, true>(a, b, c, d, u, N, stride);
}
#endif
