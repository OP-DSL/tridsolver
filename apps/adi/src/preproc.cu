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

/*
 * Create coefficient matrices
 */
__global__ void preproc(FP lambda, FP* u, FP* du, FP* ax, FP* bx, FP* cx, FP* ay, FP* by, FP* cy, FP* az, FP* bz, FP* cz, int nx, int ny, int nz) {
  int   i, j, k, ind, active;
  FP    a, b, c, d;
  //
  // set up indices for main block
  //
  i   = threadIdx.x + blockIdx.x*blockDim.x;
  j   = threadIdx.y + blockIdx.y*blockDim.y;
  ind = i + j*nx; 

  // Is the thread in active region?
  active = (i<nx) && (j<ny);

  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //
  //#pragma unroll 8
  for(k=0; k<nz; k++) {
    if(active) {
        if(i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
          d = 0.0f; // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          d = lambda*(  u[ind-1    ] + u[ind+1]
                      + u[ind-nx   ] + u[ind+nx]
                      + u[ind-nx*ny] + u[ind+nx*ny] 
                      - 6.0f*u[ind]);
          a = -0.5f * lambda;
          b =  1.0f + lambda;
          c = -0.5f * lambda;
        }
        du[ind] = d;
        //*((int*)&ax[ind]) = ind;//a;
        //ax[ind] = ind;//a;
        ax[ind] = a;
        bx[ind] = b;
        cx[ind] = c;
        ay[ind] = a;
        by[ind] = b;
        cy[ind] = c;
        az[ind] = a;
        bz[ind] = b;
        cz[ind] = c;

        ind += nx*ny;
      }
  }
}
