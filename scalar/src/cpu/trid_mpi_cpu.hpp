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
// Extended by Toby Flynn, University of Warwick, T.Flynn@warwick.ac.uk, 2020

#ifndef __TRID_MPI_CPU_HPP
#define __TRID_MPI_CPU_HPP

#include "math.h"

//
// Thomas solver for reduced system
//
template<typename REAL>
inline void thomas_on_reduced(
    const REAL* __restrict__ aa_r,
    const REAL* __restrict__ cc_r,
          REAL* __restrict__ dd_r,
    const int N,
    const int stride) {
  int   i, ind = 0;
  REAL aa, bb, cc, dd, c2[2 * N_MPI_MAX], d2[2 * N_MPI_MAX];
  //
  // forward pass
  //
  bb    = static_cast<REAL>(1.0);
  cc    = cc_r[0];
  dd    = dd_r[0];
  c2[0] = cc;
  d2[0] = dd;

  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = aa_r[ind];
    bb    = static_cast<REAL>(1.0) - aa*cc;
    dd    = dd_r[ind] - aa*dd;
    bb    = static_cast<REAL>(1.0)/bb;
    cc    = bb*cc_r[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  dd_r[ind] = dd;
  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
    dd_r[ind] = dd;
  }
}

//
// Modified Thomas forwards pass
//
template<typename REAL>
inline void thomas_forward(
    const REAL *__restrict__ a,
    const REAL *__restrict__ b,
    const REAL *__restrict__ c,
          REAL *__restrict__ d,
          REAL *__restrict__ aa,
          REAL *__restrict__ cc,
    const int sys_index,
    const int N,
    const int stride) {

  REAL bbi;
  int ind = 0;

  if(N >=2) {
    // Start lower off-diagonal elimination
    for(int i=0; i<2; i++) {
      ind = sys_index + i * stride;
      bbi   = static_cast<REAL>(1.0) / b[ind];
      d[ind] = d[ind] * bbi;
      aa[ind] = a[ind] * bbi;
      cc[ind] = c[ind] * bbi;
    }
    if(N >=3 ) {
      // Eliminate lower off-diagonal
      for(int i=2; i<N; i++) {
        ind = sys_index + i * stride;
        bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]);
        d[ind] = (d[ind] - a[ind]*d[ind - stride]) * bbi;
        aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
        cc[ind] =                 c[ind]  * bbi;
      }
      // Eliminate upper off-diagonal
      for(int i=N-3; i>0; i--) {
        ind = sys_index + i * stride;
        d[ind] = d[ind] - cc[ind]*d[ind + stride];
        aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
        cc[ind] =       - cc[ind]*cc[ind + stride];
      }
      bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[sys_index]*aa[sys_index + stride]);
      d[sys_index] =  bbi * ( d[sys_index] - cc[sys_index]*d[sys_index + stride] );
      aa[sys_index] =  bbi *   aa[sys_index];
      cc[sys_index] =  bbi * (       - cc[sys_index]*cc[sys_index + stride] );
    }
  }
  else {
    exit(-1);
  }
}

//
// Modified Thomas forwards pass
//
template<typename REAL>
inline void thomas_forward_vec_strip(
    const REAL *__restrict__ a,
    const REAL *__restrict__ b,
    const REAL *__restrict__ c,
          REAL *__restrict__ d,
          REAL *__restrict__ aa,
          REAL *__restrict__ cc,
    const int sys_index,
    const int N,
    const int stride,
    const int strip_len) {

  int ind = 0;
  int base = 0;

  REAL bbi;

  for(int i = 0; i < 2; i++) {
    base = sys_index + i * stride;
    // Compiler seems to have an issue with #pragma omp simd aligned(aa, cc, dd: SIMD_WIDTH)
    #pragma omp simd aligned(aa, cc: 64)
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      bbi   = static_cast<REAL>(1.0) / b[ind];
      d[ind] = d[ind] * bbi;
      aa[ind] = a[ind] * bbi;
      cc[ind] = c[ind] * bbi;
    }
  }

  for(int i = 2; i < N; i++) {
    base = sys_index + i * stride;
    #pragma omp simd aligned(aa, cc: 64)
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]);
      d[ind] = (d[ind] - a[ind]*d[ind - stride]) * bbi;
      aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
      cc[ind] =                 c[ind]  * bbi;
    }
  }

  for(int i = N - 3; i > 0; i--) {
    base = sys_index + i * stride;
    #pragma omp simd aligned(aa, cc: 64)
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      d[ind] = d[ind] - cc[ind]*d[ind + stride];
      aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
      cc[ind] =       - cc[ind]*cc[ind + stride];
    }
  }

  #pragma omp simd aligned(aa, cc: 64)
  for(int j = 0; j < strip_len; j++) {
    bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[sys_index + j]*aa[sys_index + stride + j]);
    d[sys_index + j] =  bbi * ( d[sys_index + j] - cc[sys_index + j]*d[sys_index + stride + j] );
    aa[sys_index + j] =  bbi *   aa[sys_index + j];
    cc[sys_index + j] =  bbi * (       - cc[sys_index + j]*cc[sys_index + stride + j] );
  }
}

template <typename REAL, int INC>
inline void thomas_backward_vec_strip(const REAL *__restrict__ aa,
                                      const REAL *__restrict__ cc,
                                      REAL *__restrict__ d,
                                      REAL *__restrict__ u, const int sys_index,
                                      const int N, const int stride,
                                      const int strip_len) {
  int base = 0;
  if (INC) {
    for (int j = 0; j < strip_len; j++) {
      u[sys_index + j] += d[sys_index + j];
    }
  }

  if (INC) {
    for (int i = 1; i < N - 1; i++) {
      base = sys_index + i * stride;
      #pragma omp simd aligned(aa, cc: 64)
      for (int j = 0; j < strip_len; j++) {
        u[base + j] += d[base + j] - aa[base + j] * d[sys_index + j] -
                       cc[base + j] * d[sys_index + (N - 1) * stride + j];
      }
    }
  } else {
    for (int i = 1; i < N - 1; i++) {
      base = sys_index + i * stride;
      #pragma omp simd aligned(aa, cc: 64)
      for (int j = 0; j < strip_len; j++) {
        d[base + j] = d[base + j] - aa[base + j] * d[sys_index + j] -
                      cc[base + j] * d[sys_index + (N - 1) * stride + j];
      }
    }
  }

  if (INC) {
    for (int j = 0; j < strip_len; j++) {
      u[sys_index + (N - 1) * stride + j] += d[sys_index + (N - 1) * stride + j];
    }
  }
}

template <typename REAL, int INC>
inline void
thomas_backward(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                REAL *__restrict__ d, REAL *__restrict__ u,
                const int sys_index, const int N, const int stride) {
  if (INC) {
    u[sys_index] += d[sys_index];
  }

  if(INC) {
    #pragma omp simd aligned(aa, cc: 64)
    for (int i = 1; i < N - 1; i++) {
      int ind = sys_index + i * stride;
      u[ind] += d[ind] - aa[ind] * d[sys_index] - cc[ind] * d[sys_index + (N - 1) * stride];
    }
  } else {
    #pragma omp simd aligned(aa, cc: 64)
    for (int i = 1; i < N - 1; i++) {
      int ind = sys_index + i * stride;
      d[ind] = d[ind] - aa[ind] * d[sys_index] - cc[ind] * d[sys_index + (N - 1) * stride];
    }
  }

  if (INC) {
    u[sys_index + (N - 1) * stride] += d[sys_index + (N - 1) * stride];
  }
}
#endif
