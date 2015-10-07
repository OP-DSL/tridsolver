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

//#include "trid_params.h"
/*
 * tridiagonal solve in x-direction
 */
template<typename REAL>
__global__ void trid_linear(const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c, REAL* __restrict__ d, REAL* __restrict__ u, int sys_size, int sys_pads, int sys_n) {
  int   i;//, ind;//, off;
  //REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  
  //
  // set up indices for main block
  //
  int off = 1;

  int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
  int ind = sys_pads*tid;

  if( tid<sys_n ) {
    //
    // forward pass
    //
    bb    = 1.0f/b[ind];
    cc    = bb*c[ind];
    dd    = bb*d[ind];
    c2[0] = cc;
    d2[0] = dd;

    //u[ind] = 0;//dd;

    for(i=1; i<sys_size; i++) {
      ind   = ind + off;
      aa    = a[ind];
      bb    = b[ind] - aa*cc;
      dd    = d[ind] - aa*dd;
      bb    = 1.0f/bb;
      cc    = bb*c[ind];
      dd    = bb*dd;
      c2[i] = cc;
      d2[i] = dd;
    
      //u[ind] = ind;//dd;

    }
    //
    // reverse pass
    //
    d[ind] = dd;

    //u[ind] = dd;

    for(i=sys_size-2; i>=0; i--) {
      ind    = ind - off;
      dd     = d2[i] - c2[i]*dd;
      d[ind] = dd;

      //u[ind] = dd;

    }
  }
}
