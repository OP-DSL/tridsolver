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

#include "trid_simd.h"
#include "transpose.hpp"
#include "math.h"

#define N_MPI_MAX 128

////
//// Thomas solver for reduced system
////
//template<typename REAL>
//inline void pcr_on_reduced(void** a, void** c, void** d, int N, int stride) {
//  
//  REAL b;
//  int s=1;
//  REAL a2_array[N_MPI_MAX], c2_array[N_MPI_MAX], d2_array[N_MPI_MAX];
//  REAL *a2 = a2_array;
//  REAL *c2 = c2_array;
//  REAL *d2 = d2_array;
//
//  REAL *a_tmp, *c_tmp, *d_tmp;
//
//  int P = ceil(log2((double)N));
//
//  for(int i=0; i<N; i+=2) {
//    b       = static_cast<REAL>(1.0) - a[i+1] * c[i] - c[i+1] * a[i+2];
//    b       = static_cast<REAL>(1.0) / b;
//    d2[i+1] =   b * (d[i+1] - a[i+1] * d[i] - c[i+1] * d[i+2]);
//    a2[i+1] = - b * a[i+1] * a[i];
//    c2[i+1] = - b * c[i+1] * c[i+2];
//  }
//
//  for(int p=1; p<P; p++) {
//    int s = 1 << p;
//    for(int i=0; i<N; i+=2) {
//      b       = static_cast<REAL>(1.0) - a[i+1] * c[i+s+1] - c[i+1] * a[i+s+2];
//      b       = static_cast<REAL>(1.0) / b;
//      d2[i+1] = 1;//  b * (d[i+1] - a[i+1] * d[i-s+1] - c[i+1] * d[i+s+1]);
//      a2[i+1] = - b * a[i+1] * a[i+1-s];
//      c2[i+1] = - b * c[i+1] * c[i+1+s];
//    }    
//    a_tmp = a2;
//    a2    = a;
//    a     = a_tmp;
//    c_tmp = c2;
//    c2    = c;
//    c     = c_tmp;
//    d_tmp = d2;
//    d2    = d;
//    d     = d_tmp;
//  }
//}


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
  REAL aa, bb, cc, dd, c2[N_MPI_MAX], d2[N_MPI_MAX];
  // REAL aa, bb, cc, dd, c2[N], d2[N];
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
    const REAL *__restrict__ d, 
          REAL *__restrict__ aa, 
          REAL *__restrict__ cc, 
          REAL *__restrict__ dd, 
    const int N, 
    const int stride) {

  REAL bbi;
  int ind = 0;

  if(N >=2) {
    // Start lower off-diagonal elimination
    for(int i=0; i<2; i++) {
      ind = i * stride;
      bbi   = static_cast<REAL>(1.0) / b[ind];
      //dd[i] = 66;//d[i] * bbi;
      dd[ind] = d[ind] * bbi;
      aa[ind] = a[ind] * bbi;
      cc[ind] = c[ind] * bbi;
    }
    if(N >=3 ) {
      // Eliminate lower off-diagonal
      for(int i=2; i<N; i++) {
        ind = i * stride;
        bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
        //dd[i] = 77;//(d[i] - a[i]*dd[i-1]) * bbi;
        dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
        aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
        cc[ind] =                 c[ind]  * bbi;
      }
      // Eliminate upper off-diagonal
      for(int i=N-3; i>0; i--) {
        ind = i * stride;
        //dd[i] = 88;//dd[i] - cc[i]*dd[i+1];
        dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
        aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
        cc[ind] =       - cc[ind]*cc[ind + stride];
      }
      bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[0]*aa[stride]);
      dd[0] =  bbi * ( dd[0] - cc[0]*dd[stride] );
      aa[0] =  bbi *   aa[0];
      cc[0] =  bbi * (       - cc[0]*cc[stride] );
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
    const REAL *__restrict__ d, 
          REAL *__restrict__ aa, 
          REAL *__restrict__ cc, 
          REAL *__restrict__ dd, 
    const int N, 
    const int stride,
    const int strip_len) {

  int ind = 0;
  int base = 0;
  
  REAL bbi;
  
  for(int i = 0; i < 2; i++) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      bbi   = static_cast<REAL>(1.0) / b[ind];
      //dd[i] = 66;//d[i] * bbi;
      dd[ind] = d[ind] * bbi;
      aa[ind] = a[ind] * bbi;
      cc[ind] = c[ind] * bbi;
    }
  }
  
  for(int i = 2; i < N; i++) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
      //dd[i] = 77;//(d[i] - a[i]*dd[i-1]) * bbi;
      dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
      aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
      cc[ind] =                 c[ind]  * bbi;
    }
  }
  
  for(int i = N - 3; i > 0; i--) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
      aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
      cc[ind] =       - cc[ind]*cc[ind + stride];
    }
  }
  
  #pragma omp simd
  for(int j = 0; j < strip_len; j++) {
    bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[j]*aa[stride + j]);
    dd[j] =  bbi * ( dd[j] - cc[j]*dd[stride + j] );
    aa[j] =  bbi *   aa[j];
    cc[j] =  bbi * (       - cc[j]*cc[stride + j] );
  }
}

template <typename REAL, int INC>
inline void thomas_backward_vec_strip(const REAL *__restrict__ aa,
                                      const REAL *__restrict__ cc,
                                      const REAL *__restrict__ dd,
                                      REAL *__restrict__ d,
                                      REAL *__restrict__ u, const int N,
                                      const int stride, const int strip_len) {
  int base = 0;

  #pragma omp simd
  for (int j = 0; j < strip_len; j++) {
    if (INC) {
      u[j] += dd[j];
    } else {
      d[j] = dd[j];
    }
  }

  for (int i = 1; i < N - 1; i++) {
    base = i * stride;
    #pragma omp simd
    for (int j = 0; j < strip_len; j++) {
      if (INC) {
        u[base + j] += dd[base + j] - aa[base + j] * dd[j] -
                       cc[base + j] * dd[(N - 1) * stride + j];
      } else {
        d[base + j] = dd[base + j] - aa[base + j] * dd[j] -
                      cc[base + j] * dd[(N - 1) * stride + j];
      }
    }
  }

  #pragma omp simd
  for (int j = 0; j < strip_len; j++) {
    if (INC) {
      u[(N - 1) * stride + j] += dd[(N - 1) * stride + j];
    } else {
      d[(N - 1) * stride + j] = dd[(N - 1) * stride + j];
    }
  }
}

template <typename REAL, int INC>
inline void
thomas_backward(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                const REAL *__restrict__ dd, REAL *__restrict__ d,
                REAL *__restrict__ u, const int N, const int stride) {
  if (INC) {
    u[0] += dd[0];
  } else {
    d[0] = dd[0];
  }
#pragma omp simd
  for (int i = 1; i < N - 1; i++) {
    int ind = i * stride;
    if (INC) {
      u[ind] += dd[ind] - aa[ind] * dd[0] - cc[ind] * dd[(N - 1) * stride];
    } else {
      d[ind] = dd[ind] - aa[ind] * dd[0] - cc[ind] * dd[(N - 1) * stride];
    }
  }
  if (INC) {
    u[(N - 1) * stride] += dd[(N - 1) * stride];
  } else {
    d[(N - 1) * stride] = dd[(N - 1) * stride];
  }
}
#endif
