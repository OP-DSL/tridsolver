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
 
#ifndef __TRID_LINEAR_SHARED_HPP
#define __TRID_LINEAR_SHARED_HPP

#include <assert.h>
//#include"adi_cuda.h"
//#include "trid_params.h"
#include<cuda_runtime.h>
#include<stdio.h>

#ifdef VEC
#undef VEC
#endif
#define VEC 8 
//#define REAL float
// ga - global array
// la - local array
template<typename REAL>
inline __device__ void load_array_shared(const REAL* ga,REAL* la, int n, volatile REAL* shared, int goffset, int soffset, int trow, int tcol, int sys_size, int sys_pads) {
  int gind; // Global memroy index of an element
  int sind; // Shared memory index of an element
  int roff; // Row offset 0..31 by steps of 32/VEC 
  int m;    // Iterator within the VEC registers
  if(n+tcol<sys_size) {
    for(roff=0; roff<WARP_SIZE; roff+=(WARP_SIZE/VEC)) { 
      gind = goffset + n + (roff+trow) * sys_pads + tcol;
      sind = soffset + (roff+trow) * (VEC+1)  + tcol; 
      shared[sind] = ga[gind]; 
      //if(n+tcol<nx) shared[sind] = ga[gind]; 
    } 
  }
  for(m=0; m<VEC; m++) { 
    la[m] = shared[soffset + (threadIdx.x % WARP_SIZE)*(VEC+1) + m]; 
  } 
}

template<typename REAL>
inline __device__ void store_array_shared(REAL* ga, REAL* la, int n, volatile REAL* shared, int goffset, int soffset, int trow, int tcol, int sys_size, int sys_pads) {
  int gind; // Global memroy index of an element
  int sind; // Shared memory index of an element
  int roff; // Row offset 0..31 by steps of 32/VEC 
  int m;    // Iterator within the VEC registers
//(threadIdx.x==1) {
  for(m=0; m<VEC; m++) { 
    //shared[soffset + threadIdx.x*(VEC+1) + m] = la[m]; 
    shared[soffset + (threadIdx.x % WARP_SIZE)*(VEC+1) + m] = la[m]; 
//    printf("soffset + threadIdx.x *(VEC+1) + m = %d + %d * (%d) + %d = %d\n", soffset, threadIdx.x, (VEC+1), m, soffset + threadIdx.x *(VEC+1) + m);
  } 
  if(n+tcol<sys_size) {
    for(roff=0; roff<WARP_SIZE; roff+=(WARP_SIZE/VEC)) { 
      gind = goffset + n + (roff+trow) * sys_pads + tcol;
      sind = soffset + (roff+trow) * (VEC+1)  + tcol; 
      ga[gind] = shared[sind]; 
      //if(n+tcol<nx) ga[gind] = shared[sind]; 
    } 
  }
//  else { 
//    for(roff=0; roff<WARP_SIZE; roff+=(WARP_SIZE/VEC)) { 
//      gind = goffset + n + (roff+trow) * nx + tcol; 
//      sind = soffset + (roff+trow) * (VEC+1)  + tcol; 
//      ga[gind] = 33;//shared[sind]; 
//      //if(n+tcol<nx) ga[gind] = shared[sind]; 
//    } 
//  }
}

//
// tridiagonal solve in x-direction
//
//__global__ void trid_linear_shared(const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c, REAL* __restrict__ d, REAL* __restrict__ u, int sys_size, int sys_pads, int sys_n, int nx, int ny, int nz) {
template<typename REAL>
__global__ void trid_linear_shared(const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c, REAL* __restrict__ d, REAL* __restrict__ u, int sys_size, int sys_pads, int sys_n) {
//__global__ void trid_x_shared(const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c, REAL* __restrict__ d, REAL* __restrict__ u, int nx, int ny, int nz) {
  int   i;
  REAL l_a[VEC], l_b[VEC], l_c[VEC], l_d[VEC];
  REAL aa, bb, cc, dd, c2[N_MAX],d2[N_MAX];
 
//  assert(blockDim.x == 32); // The algorithm is tuned for warp level optimization
  
  //extern __shared__ volatile REAL shared[]; // a thread block processes a 32xVEC block on the Y-Z plane; +1 is added to decrease bank conflicts;  size of the array = ((VEC+1) * dimBlock_x_shared.x * dimBlock_x_shared.y) * sizeof(REAL)
  extern __shared__ volatile char shared_char[]; // a thread block processes a 32xVEC block on the Y-Z plane; +1 is added to decrease bank conflicts;  size of the array = ((VEC+1) * dimBlock_x_shared.x * dimBlock_x_shared.y) * sizeof(REAL)
  REAL* shared = (REAL*)&shared_char[0];         // Pointer casting is needed due to some compiler malfunction that doesn't allow two definitions (float[] and double[]) for an extern __shared__ array within one compilation unit

  //
  // set up indices for main block
  //
//  j   = threadIdx.x + blockIdx.x*blockDim.x;
//  k   = threadIdx.y + blockIdx.y*blockDim.y;
//
//if( (j<ny) && (k<nz) ) {
  const int tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
  //ind = sys_pads*tid;

  const int wid     = tid / WARP_SIZE;            // Warp ID in global scope - the ID wich the thread belongs to
  const int goffset = wid * WARP_SIZE * sys_pads; // Global memory offset: unique to a warp; every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  //int stride  = nx;                    // Stride between systems
        //int ind     = tid * sys_pads;                                // First index a system being solved by thread tid.
        int n       = 0;                                             // Iterator for every 8th element along dimension X
  const int optimized_solve = ((tid/VEC)*VEC+VEC <= sys_n);                // These 8-threads do the regular memory read/write and data transpose on a 8x8 tile of scalar values
  const int boundary_solve  = !optimized_solve && ( tid < sys_n ); // Among these 8-threads are some that have to be deactivated from global memory read/write
  const int active_thread   = optimized_solve || boundary_solve;     // A thread is active only if it works on valid memory
  //const int aligned         = !(sys_pads % ALIGN_FLOAT);

  // Wrap 1 warp into 4 rows with 8 threads in a row
  const int tcol    = threadIdx.x % VEC;                    // Thread column ID within a warp
  const int trow    = (threadIdx.x % WARP_SIZE)/ VEC;       // Thread row ID within a warp
  //const int goffset = k*ny*nx + (j/WARP_SIZE)*WARP_SIZE*nx; // Global memory offset: unique to a warp; every thread in a warp calculates the same goffset, which is the "begining" of 3D tile
  //const int goffset = k*ny*nx + (j/WARP_SIZE)*WARP_SIZE*nx; // Global memory offset: unique to a warp; every thread in a warp calculates the same goffset, which is the "begining" of 3D tile
  //const int soffset = threadIdx.y * (blockDim.x/WARP_SIZE)*WARP_SIZE * (VEC+1) + (threadIdx.x/WARP_SIZE)*WARP_SIZE*(VEC+1); // Shared memory offset for the threads in the y block dimension
  const int soffset = (threadIdx.x/WARP_SIZE)*WARP_SIZE*(VEC+1); // Shared memory offset for the threads in the y block dimension

  //if( tid<sys_n ) {
  if(active_thread) {
//    if(optimized_solve) {
//      if(aligned) {
        // All threads iterate along dimension X by step size 8
        // 32 thread (a warp) that is originally mapped to dimension Y now loads data along dimension X
        // Warps within a block wrap into a 4x8 tile where indices are (y,x)=(threadIdx.x/8,threadIdx.y%8)
        // Such a 4x8 tile iterates 8 times along the dimension Y to read the data for the 32 rows.
        //
        // forward pass
        //
        n = 0;
        load_array_shared<REAL>(a,l_a,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
        load_array_shared<REAL>(b,l_b,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
        load_array_shared<REAL>(c,l_c,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
        load_array_shared<REAL>(d,l_d,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
        bb     = 1.0f/l_b[0];
        cc     = bb*l_c[0];
        dd     = bb*l_d[0];
        c2[0] = cc;
        d2[0] = dd;

        //l_d[0] = dd;//0;

        for(i=1; i<VEC; i++) {
          aa    = l_a[i];
          bb    = l_b[i] - aa*cc;
          dd    = l_d[i] - aa*dd;
          bb    = 1.0f/bb;
          cc    = bb*l_c[i];
          dd    = bb*dd;
          c2[i] = cc;
          d2[i] = dd;

          // u[ind+i] = d2[i];
          // l_d[i] = dd;//n+i;

        }
        //store_array_shared<REAL>(u,l_d,n,shared,goffset,soffset,trow,tcol,sys_size);

        for(n=VEC; n<sys_size; n+=VEC) {
          load_array_shared<REAL>(a,l_a,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
          load_array_shared<REAL>(b,l_b,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
          load_array_shared<REAL>(c,l_c,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
          load_array_shared<REAL>(d,l_d,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
          for(i=0; i<VEC; i++) {
            //if(isnan(l_a[i]) || isnan(l_b[i]) || isnan(l_c[i]) || isnan(l_d[i])) printf("a = %f ; b = %f ; c = %f ; d = %f;\n",l_a[i], l_b[i], l_c[i], l_d[i]);
            aa    = l_a[i];
            bb    = l_b[i] - aa*cc;
            dd    = l_d[i] - aa*dd;
            bb    = 1.0f/bb;
            cc    = bb*l_c[i];
            dd    = bb*dd;
            c2[n+i] = cc;
            d2[n+i] = dd;

            //u[ind+n+i] = d2[n+i];

            // l_d[i] = dd;//n+i;
          }
          //store_array_shared<REAL>(u,l_d,n,shared,goffset,soffset,trow,tcol,sys_size);
        }
        //
        // reverse pass
        //
        int sys_off = n-sys_size;
        dd = d2[sys_size-1];
        //l_d[VEC-sys_off-1] = dd;
        //l_d[VEC-1] = nx-1;
        //int diff = n-sys_size;
        //n = nx - VEC;
        n -= VEC;
        //printf("n = %d\n",n);
        //for(i=VEC-sys_off-2; i>=0; i--) {
        for(i=VEC-1; i>=0; i--) {
          //for(i=sys_off-2; i>=0; i--) {
          if(i==VEC-sys_off-1) l_d[i] = dd;
          if(i<VEC-sys_off-1) {
            dd      = d2[n+i] - c2[n+i]*dd;
            l_d[i] = dd;
            //l_d[i]  = n+i;
          }
        }
        store_array_shared<REAL>(d,l_d,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
        //store_array_shared<REAL>(u,l_d,n,shared,goffset,soffset,trow,tcol,sys_size);

        //for(n=sys_size-2*VEC; n>=0; n-=VEC) {
        for(n=n-VEC; n>=0; n-=VEC) {
          for(i=(VEC-1); i>=0; i--) {
            dd     = d2[n+i] - c2[n+i]*dd;
            l_d[i] = dd;
            //l_d[i] = n+i;
          }
          store_array_shared<REAL>(d,l_d,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
          //store_array_shared<REAL>(u,l_d,n,shared,goffset,soffset,trow,tcol,sys_size);
        }
      }
//    }
//  }
}

#endif
