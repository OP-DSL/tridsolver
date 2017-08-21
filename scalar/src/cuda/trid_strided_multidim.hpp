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
__global__ void trid_strided_multidim(const VECTOR* __restrict__ a,
                                      const VECTOR* __restrict__ b,
                                      const VECTOR* __restrict__ c,
                                      VECTOR* __restrict__ d,
                                      VECTOR* __restrict__ u, int ndim,
                                      int solvedim, int sys_n) {
  int    j;
  VECTOR aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // set up indices for main block
  //
  int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
  int coord[MAXDIM];

  #pragma loop unroll(MAXDIM)
  for(j=0; j<ndim; j++) {
    if(j<=solvedim) coord[j] = ( tid /  d_cumdims[j] ) % d_dims[j];
    else            coord[j] = ( tid / (d_cumdims[j] / d_dims[solvedim])) % d_dims[j];
//    if(tid==256) {
//      printf("tid           = %d \n",tid);
//      printf("d_cumdims[%d] = %d \n",j,d_cumdims[j]);
//      printf("d_dims[%d]    = %d \n",j,d_dims[j]);
//      printf("coord[%d] = %d \n",j,coord[j]);
//      printf("d_cumpads[%d] = %d \n\n",j,d_cumpads[j]);
//    }
  }
  coord[solvedim] = 0;

  int ind = 0;
  #pragma loop unroll(MAXDIM)
  for(j=0; j<ndim; j++) ind += coord[j]*d_cumpads[j];

  int stride   = d_cumpads[solvedim];
  int sys_size = d_dims[solvedim];

//  Y-dim index setup
//  int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
//  int stride   = d_pads[0];
//  int sys_size = d_dims[1];
//  int sys_n    = d_dims[0]*d_dims[2];
//  int coord[3];
//  coord[0] =( tid % d_dims[0]);
//  coord[1] = 0;
//  coord[2] = (tid / d_dims[0] ) % d_dims[1];
//  int ind = coord[0] + coord[1]*d_pads[0] + coord[2]*d_pads[0]*d_pads[1];

//  //  Z-dim index setup
//  int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
//  int stride   = d_pads[0]*d_pads[1];
//  int sys_size = d_dims[2];
//  int sys_n    = d_dims[0]*d_dims[1];
//  int coord[3];
//  coord[0] = tid % d_dims[0];
//  coord[1] = (tid / d_dims[0] ) % d_dims[1];
//  coord[2] = 0;
//  int ind = coord[0] + coord[1]*d_pads[0] + coord[2]*d_pads[0]*d_pads[1];

  if( tid<sys_n ) {
    //
    // forward pass
    //
    bb    = (static_cast<REAL>(1.0))  / b[ind];
    cc    = bb*c[ind];
    dd    = bb*d[ind];
    c2[0] = cc;
    d2[0] = dd;
    for(j=1; j<sys_size; j++) {
      ind   = ind + stride;
      aa    = a[ind];
      bb    = b[ind] - aa*cc;
      dd    = d[ind] - aa*dd;
      bb    = (static_cast<REAL>(1.0))  / bb;
      cc    = bb*c[ind];
      dd    = bb*dd;
      c2[j] = cc;
      d2[j] = dd;
    }
    //
    // reverse pass
    //
    if(INC==0) d[ind]  = dd;
    else       u[ind] += dd;
    //u[ind] = dd;
    for(j=sys_size-2; j>=0; j--) {
      ind    = ind - stride;
      dd     = d2[j] - c2[j]*dd;
      if(INC==0) d[ind]  = dd;
      else       u[ind] += dd;
    }
  }
}

#endif
