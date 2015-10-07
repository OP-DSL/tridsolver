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

#include "trid_common.h"
#include "trid_simd.h"

#ifdef __MIC__ // Or #ifdef __KNC__ - more general option, future proof, __INTEL_OFFLOAD is another option
  __attribute__((target(mic)))
  inline void preproc_simd(FP lambda, FP* __restrict u, FP* __restrict du, FP* __restrict ax, FP* __restrict bx, FP* __restrict cx, FP* __restrict ay, FP* __restrict by, FP* __restrict cy, FP* __restrict az, FP* __restrict bz, FP* __restrict cz, int nx, int ny, int nz);

  __attribute__((target(mic)))
  inline void preproc(REAL lambda, REAL* __restrict u, REAL* __restrict du, REAL* __restrict ax, REAL* __restrict bx, REAL* __restrict cx, REAL* __restrict ay, REAL* __restrict by, REAL* __restrict cy, REAL* __restrict az, REAL* __restrict bz, REAL* __restrict cz, int nx, int nx_pad, int ny, int nz);
#endif

inline void preproc_simd(FP lambda, FP* __restrict u, FP* __restrict du, FP* __restrict ax, FP* __restrict bx, FP* __restrict cx, FP* __restrict ay, FP* __restrict by, FP* __restrict cy, FP* __restrict az, FP* __restrict bz, FP* __restrict cz, int nx, int ny, int nz) {
  int   i, j, k;
  __declspec(align(SIMD_WIDTH)) SIMD_REG *mm_u, *mm_du, *mm_ax, *mm_bx, *mm_cx,*mm_ay, *mm_by, *mm_cy,*mm_az, *mm_bz, *mm_cz;
  mm_u  = (SIMD_REG*) u ;
  mm_du = (SIMD_REG*) du;
  mm_ax = (SIMD_REG*) ax;
  mm_bx = (SIMD_REG*) bx;
  mm_cx = (SIMD_REG*) cx;
  mm_ay = (SIMD_REG*) ay;
  mm_by = (SIMD_REG*) by;
  mm_cy = (SIMD_REG*) cy;
  mm_az = (SIMD_REG*) az;
  mm_bz = (SIMD_REG*) bz;
  mm_cz = (SIMD_REG*) cz;
  #pragma omp parallel for collapse(2)
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      int   ind, mm_ind, n;
      __declspec(align(SIMD_WIDTH)) SIMD_REG a, b, c, d, tmp;
      FP  A,B,C,D;
      A = -0.5F * lambda;
      B =  1.0F + lambda;
      C = -0.5F * lambda;

      //#pragma simd
      //#pragma prefetch     u:1:12
      //#pragma prefetch     u:0:6

      //#pragma prefetch  mm_u:1:12
      //#pragma prefetch mm_ax:1:12
      //#pragma prefetch mm_bx:1:12
      //#pragma prefetch mm_cx:1:12
      //#pragma prefetch mm_ay:1:12
      //#pragma prefetch mm_by:1:12
      //#pragma prefetch mm_cy:1:12
      //#pragma prefetch mm_az:1:12
      //#pragma prefetch mm_bz:1:12
      //#pragma prefetch mm_cz:1:12
      //#pragma prefetch mm_du:1:12

      //#pragma prefetch  mm_u:0:4
      //#pragma prefetch mm_ax:0:4
      //#pragma prefetch mm_bx:0:4
      //#pragma prefetch mm_cx:0:4
      //#pragma prefetch mm_ay:0:4
      //#pragma prefetch mm_by:0:4
      //#pragma prefetch mm_cy:0:4
      //#pragma prefetch mm_az:0:4
      //#pragma prefetch mm_bz:0:4
      //#pragma prefetch mm_cz:0:4
      //#pragma prefetch mm_du:0:4

      for(i=0; i<nx; i+=SIMD_VEC) {   // i loop innermost for sequential memory access
        ind = k*nx*ny + j*nx + i;
        mm_ind = ind/SIMD_VEC;

        d = SIMD_MUL_P( SIMD_SET1_P(-6.0F), mm_u[mm_ind]);

        // Process a,b,c
        a = SIMD_SET1_P(A);
        b = SIMD_SET1_P(B);
        c = SIMD_SET1_P(C);

        if(i==0) {
          ((FP*)(&a))[0]          = 0.0F;
          ((FP*)(&b))[0]          = 1.0F;
          ((FP*)(&c))[0]          = 0.0F;
          ((FP*)(&d))[0]          = 0.0F;
          for(n=1; n<SIMD_VEC; n++) {
            ((FP*)(&d))[n] += u[ind+n-1] + u[ind+n+1];
          }
        }

        else if(i==nx-SIMD_VEC) {
          ((FP*)(&a))[SIMD_VEC-1] = 0.0F;
          ((FP*)(&b))[SIMD_VEC-1] = 1.0F;
          ((FP*)(&c))[SIMD_VEC-1] = 0.0F;
          ((FP*)(&d))[SIMD_VEC-1] = 0.0F;
          for(n=0; n<SIMD_VEC-1; n++) {
            ((FP*)(&d))[n] += u[ind+n-1] + u[ind+n+1];
          }
        }

        else {
          for(n=0; n<SIMD_VEC; n++) {
            ((FP*)(&d))[n] += u[ind+n-1] + u[ind+n+1];
          }
        }

        if(j==0 || j==ny-1 || k==0 || k==nz-1)	{
          a = SIMD_SET1_P(0.0F);
          b = SIMD_SET1_P(1.0F);
          c = SIMD_SET1_P(0.0F);
          d = SIMD_SET1_P(0.0F);
        } else {
          d = SIMD_ADD_P(d,mm_u[mm_ind-(nx/SIMD_VEC)]);
          d = SIMD_ADD_P(d,mm_u[mm_ind+(nx/SIMD_VEC)]);
          d = SIMD_ADD_P(d,mm_u[mm_ind-ny*(nx/SIMD_VEC)]);
          d = SIMD_ADD_P(d,mm_u[mm_ind+ny*(nx/SIMD_VEC)]);
          d = SIMD_MUL_P( SIMD_SET1_P(lambda), d);
          if(i==0) {
            ((FP*)(&d))[0]          = 0.0F;
          }
          else if(i==nx-SIMD_VEC) {
            ((FP*)(&d))[SIMD_VEC-1] = 0.0F;
          }
        }

        mm_du[mm_ind] = d;
        mm_ax[mm_ind] = a;
        mm_bx[mm_ind] = b;
        mm_cx[mm_ind] = c;
        mm_ay[mm_ind] = a;
        mm_by[mm_ind] = b;
        mm_cy[mm_ind] = c;
        mm_az[mm_ind] = a;
        mm_bz[mm_ind] = b;
        mm_cz[mm_ind] = c;
      }
    }
  }
}

//
// calculate r.h.s. and set tri-diagonal coefficients
//
template<typename REAL>
inline void preproc(REAL lambda, REAL* __restrict u, REAL* __restrict du, REAL* __restrict ax, REAL* __restrict bx, REAL* __restrict cx, REAL* __restrict ay, REAL* __restrict by, REAL* __restrict cy, REAL* __restrict az, REAL* __restrict bz, REAL* __restrict cz, int nx, int nx_pad, int ny, int nz) {

  #ifndef VALID
    #pragma omp parallel for collapse(3) //private(i,j,k,ind,a,b,c,d)
  #endif

  for(int k=0; k<nz; k++) {
    for(int j=0; j<ny; j++) {
      // #pragma simd
      // #pragma vector nontemporal //aligned
      for(int i=0; i<nx; i++) {   // i loop innermost for sequential memory access
        int ind = k*nx_pad*ny + j*nx_pad + i;
        REAL a, b, c, d;
        if(i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
          d = 0.0f; // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          d = lambda*(  u[ind-1    ] + u[ind+1]
                      + u[ind-nx_pad   ] + u[ind+nx_pad]
                      + u[ind-nx_pad*ny] + u[ind+nx_pad*ny]
                      - 6.0f*u[ind]);
          a = -0.5f * lambda;
          b =  1.0f + lambda;
          c = -0.5f * lambda;
        }
        du[ind] = d;
        ax[ind] = a;
        bx[ind] = b;
        cx[ind] = c;
        ay[ind] = a;
        by[ind] = b;
        cy[ind] = c;
        az[ind] = a;
        bz[ind] = b;
        cz[ind] = c;
      }
    }
  }
}

