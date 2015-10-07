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
 
#ifndef __TRID_LINEAR_REG8_DOUBLE2_HPP
#define __TRID_LINEAR_REG8_DOUBLE2_HPP

#include <assert.h>
#include <sm_35_intrinsics.h>
//#include <generics/shfl.h>
#include "cuda_shfl.h"
//#include "adi_cuda.h"
//#include "trid_params.h"

//
// tridiagonal solve in x-direction
//
#ifdef VEC
#undef VEC
#endif
#define VEC 8 
// ga - global array
// la - local array

//typedef union __align__(16) {
typedef union {
  double2 vec[VEC/2];
  double  f[VEC];
} double8;

// transpose4x4xor() - exchanges data between 4 consecutive threads
inline __device__ void transpose4x4xor(double8* la) {
  double2 tmp1;
  double2 tmp2;

  // Perform a 2-stage butterfly transpose

  // stage 1 (transpose each one of the 2x2 sub-blocks internally)
  if (threadIdx.x&1) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[2];
  } else {
    tmp1 = (*la).vec[1];
    tmp2 = (*la).vec[3];
  }

  tmp1.x = __shfl_xor(tmp1.x,1);
  tmp1.y = __shfl_xor(tmp1.y,1);
  //tmp1.z = __shfl_xor(tmp1.z,1);
  //tmp1.w = __shfl_xor(tmp1.w,1);

  tmp2.x = __shfl_xor(tmp2.x,1);
  tmp2.y = __shfl_xor(tmp2.y,1);
  //tmp2.z = __shfl_xor(tmp2.z,1);
  //tmp2.w = __shfl_xor(tmp2.w,1);

  if (threadIdx.x&1) {
    (*la).vec[0] = tmp1;
    (*la).vec[2] = tmp2;
  } else {
    (*la).vec[1] = tmp1;
    (*la).vec[3] = tmp2;
  }

  // stage 2 (swap off-diagonal 2x2 blocks)
  if (threadIdx.x&2) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[1];
  } else {
    tmp1 = (*la).vec[2];
    tmp2 = (*la).vec[3];
  }

  tmp1.x = __shfl_xor(tmp1.x,2);
  tmp1.y = __shfl_xor(tmp1.y,2);
  //tmp1.z = __shfl_xor(tmp1.z,2);
  //tmp1.w = __shfl_xor(tmp1.w,2);

  tmp2.x = __shfl_xor(tmp2.x,2);
  tmp2.y = __shfl_xor(tmp2.y,2);
  //tmp2.z = __shfl_xor(tmp2.z,2);
  //tmp2.w = __shfl_xor(tmp2.w,2);

  if (threadIdx.x&2) {
    (*la).vec[0] = tmp1;
    (*la).vec[1] = tmp2;
  } else {
    (*la).vec[2] = tmp1;
    (*la).vec[3] = tmp2;
  }
}

// ga - global array
// la - local array
inline __device__ void load_array_reg8_double2(const double* __restrict__ ga, double8* la, int n, int woffset, int sys_pads) {
  int gind; // Global memory index of an element
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
  int tcol =  threadIdx.x       % 4; // Threads' colum index within a warp

  // Load 4 double2 values (64bytes) from an X-line
  gind = woffset + (4*(trow)) * sys_pads + tcol*2 + n; // First index in the X-line; woffset - warp offset in global memory
  ////(*la).vec[0] = *((double2*)&ga[gind]); // Read a double2 from aligned index
  //(*la).vec[0] = __ldg( ((double2*)&ga[gind]) ); // Read a double2 from aligned index
  //gind += sys_pads;                           // stride to the next system
  ////(*la).vec[1] = *((double2*)&ga[gind]);
  //(*la).vec[1] = __ldg( ((double2*)&ga[gind]) );
  //gind += sys_pads;
  ////(*la).vec[2] = *((double2*)&ga[gind]);
  //(*la).vec[2] = __ldg( ((double2*)&ga[gind]) );
  //gind += sys_pads;
  ////(*la).vec[3] = *((double2*)&ga[gind]);
  //(*la).vec[3] = __ldg( ((double2*)&ga[gind]) );
  int i;
  //#pragma unroll(4)
  for(i=0; i<4; i++) {
  //  (*la).vec[i] = *((double2*)&ga[gind]);
    (*la).vec[i] = __ldg( ((double2*)&ga[gind]) );
    gind += sys_pads;
  }

  transpose4x4xor(la);
}

// Same as load_array_reg8() with the following exception: if sys_pads would cause unaligned access the index is rounded down to the its floor value to prevent missaligned access.
// ga - global array
// la - local array
inline __device__ void load_array_reg8_double2_unaligned(double const* __restrict__ ga, double8* la, int n, int tid, int sys_pads, int sys_length) {
  int gind; // Global memory index of an element
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  //int trow = (threadIdx.x % 32)/ 4; // Threads' row index within a warp
  int tcol = threadIdx.x % 4;       // Threads' colum index within a warp

  // Load 4 double2 values (64bytes) from an X-line
  //gind = (tid/4)*4 * sys_pads  + tcol*4 + n; // Global memory index for threads
  gind = (tid/4)*4 * sys_pads  + n; // Global memory index for threads

  int gind_floor;
  //int ind;
  int i;
  for(i=0; i<4; i++) {
    gind_floor   = (gind/ALIGN_DOUBLE)*ALIGN_DOUBLE + tcol*2; // Round index to floor
    //gind_floor   = (gind/4)*4; // Round index to floor
    //(*la).vec[i] = *((double2*)&ga[gind_floor]);    // Get aligned data
    (*la).vec[i] = __ldg( ((double2*)&ga[gind_floor]) );    // Get aligned data
    gind        += sys_pads;                         // Stride to the next system
  }

  transpose4x4xor(la);

}

//inline __device__ void load_array_reg8_boundary(double* ga, double8* la, int n, int woffset, int stride, int n_sys) {
//  int gind; // Global memory index of an element
//  // Array indexing can be decided in compile time -> arrays will stay in registers
//  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
//  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
//  int tcol =  threadIdx.x       % 4; // Threads' colum index within a warp
//
//  // Load 4 double2 values (64bytes) from an X-line
//  gind = woffset + (4*(trow)) * stride + tcol*4 + n; // First index in the X-line; woffset - warp offset in global memory
//  //(*la).vec[0] = *((double2*)&ga[gind]); // Read a double2 from aligned index
//  //gind += stride;                           // Stride to the next system
//  //(*la).vec[1] = *((double2*)&ga[gind]);
//  //gind += stride;
//  //(*la).vec[2] = *((double2*)&ga[gind]);
//  //gind += stride;
//  //(*la).vec[3] = *((double2*)&ga[gind]);
//  int tmp;
//  int bound = n_sys*stride;
//  #pragma unroll(4)
//  for(i=0; i<4; i++) {
//    (*la).vec[i] = *((double2*)&ga[gind]);
//    tmp = gind+stride;
//    //gind += stride; //= woffset + (4*(trow) + 1) * stride + tcol*4 + n;
//    gind = tmp<bound
//  }
//
//  transpose4x4xor(la);
//}

// Store a tile with 32x16 elements into 32 double8 struct allocated in registers. Every 4 consecutive threads cooperate to transpose and store a 4 x double2 sub-tile.
// ga - global array
// la - local array
inline __device__ void store_array_reg8_double2(double* __restrict__ ga, double8* la, int n, int woffset, int sys_pads) {
  int gind; // Global memory index of an element
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
  int tcol =  threadIdx.x     % 4;   // Threads' colum index within a warp

  transpose4x4xor(la);

  gind = woffset + (4*(trow)) * sys_pads + tcol*2 + n;
  *((double2*)&ga[gind]) = (*la).vec[0];
  gind += sys_pads;
  *((double2*)&ga[gind]) = (*la).vec[1];
  gind += sys_pads;
  *((double2*)&ga[gind]) = (*la).vec[2];
  gind += sys_pads;
  *((double2*)&ga[gind]) = (*la).vec[3];

  //int i;
  //#pragma unroll(4)
  //for(i=0; i<lines; i++) {
  //  //if(gind+4<stride*ny*nz) *((double2*)&ga[gind]) = (*la).vec[i];
  //  *((double2*)&ga[gind]) = (*la).vec[i];
  //  gind += stride;
  //}

}

// Same as store_array_reg8() with the following exception: if stride would cause unaligned access the index is rounded down to the its floor value to prevent missaligned access.
// ga - global array
// la - local array
inline __device__ void store_array_reg8_double2_unaligned(double* __restrict__ ga, double8* __restrict__ la, int n, int tid, int sys_pads, int sys_length) {
  int gind; // Global memory index of an element
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  //int trow = (threadIdx.x % 32)/ 4; // Threads' row index within a warp
  int tcol = threadIdx.x % 4;       // Threads' colum index within a warp

  transpose4x4xor(la);

  // Store 4 double2 values (64bytes) to an X-line
  //gind = (tid/4)*4 * sys_pads  + tcol*4 + n; // Global memory index for threads
  gind = (tid/4)*4 * sys_pads  + n; // Global memory index for threads

  int gind_floor;
  //int ind;
  int i;
  for(i=0; i<4; i++) {
    //gind_floor = (gind/ALIGN_DOUBLE)*ALIGN_DOUBLE ; // Round index to floor
    gind_floor = (gind/ALIGN_DOUBLE)*ALIGN_DOUBLE + tcol*2; // Round index to floor
    //gind_floor = (gind/4)*4; // Round index to floor double2
    *((double2*)&ga[gind_floor]) = (*la).vec[i];  // Put aligned data
    //*((double2*)&ga[gind_floor]) = (double2){gind_floor, gind_floor,gind_floor, gind_floor};  // Put aligned data
    gind += sys_pads;                              // Stride to the next system
  }
}

  // All threads iterate along dimension X by step size 16
    // 32 thread (a warp) that is originally mapped to dimension Y now loads data along dimension X
    // Warps within a block wrap into a 4x8 tile where indices are (y,x)=(threadIdx.x/8,threadIdx.y%8)
    // Such a 4x8 tile iterates 8 times along the dimension Y to read the data for the 32 rows.
//__global__ void trid_linear_reg8_double2(double* a, double* b, double* c, double* d, double* u, int sys_size, int sys_pads, int sys_n, int nx, int ny, int nz) {
  __global__ void trid_linear_reg8_double2(double* a, double* b, double* c, double* d, double* u, int sys_size, int sys_pads, int sys_n) {
  int     i;
  double8 l_a, l_b, l_c, l_d;
  double   aa, bb, cc, dd, c2[N_MAX],d2[N_MAX];

  //
  // set up indices for main block
  //
  const int tid     = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system,
  const int wid     = tid / WARP_SIZE;      // Warp ID in global scope - the ID wich the thread belongs to
  //const int woffset = wid * WARP_SIZE * nx; // Global memory offset: unique to a warp; every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads; // Global memory offset: unique to a warp; every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  //const int stride  = nx;                   // Stride between systems
  //      int ind     = tid * stride;         // First index a system being solved by thread tid.
        int ind     = tid * sys_pads;         // First index a system being solved by thread tid.
        int n       = 0;                                             // Iterator for every 16th element along dimension X
  //const int optimized_solve = ((tid/4)*4+4 <= ny*nz);                // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid/4)*4+4 <= sys_n);                // These 4-threads do the regular memory read/write and data transpose
  //const int boundary_solve  = !optimized_solve && ( tid < (ny*nz) ); // Among these 4-threads are some that have to be deactivated from global memory read/write
  const int boundary_solve  = !optimized_solve && ( tid < (sys_n) ); // Among these 4-threads are some that have to be deactivated from global memory read/write
  const int active_thread   = optimized_solve || boundary_solve;     // A thread is active only if it works on valid memory
  //const int aligned         = !(nx % ALIGN_DOUBLE);
  const int aligned         = !(sys_pads % ALIGN_DOUBLE);

  //
  // forward pass
  //

  if(active_thread) {
    if(optimized_solve) {
      if(aligned) { // If the data is aligned do an aligned, fully optimized solve
        // The 0th element is treated differently, so do the first VECtor separately
        load_array_reg8_double2(a,&l_a,n, woffset, sys_size);
        load_array_reg8_double2(b,&l_b,n, woffset, sys_size);
        load_array_reg8_double2(c,&l_c,n, woffset, sys_size);
        load_array_reg8_double2(d,&l_d,n, woffset, sys_size);
        bb    = 1.0f/l_b.f[0];
        cc    = bb*l_c.f[0];
        dd    = bb*l_d.f[0];
        c2[0] = cc;
        d2[0] = dd;

        #pragma unroll 16
        for(i=1; i<VEC; i++) {
          aa    = l_a.f[i];
          bb    = l_b.f[i] - aa*cc;
          dd    = l_d.f[i] - aa*dd;
          bb    = 1.0f/bb;
          cc    = bb*l_c.f[i];
          dd    = bb*dd;
          c2[i] = cc;
          d2[i] = dd;
        }
        // Process the rest of the VECtors
        for(n=VEC; n<sys_size; n+=VEC) {
          load_array_reg8_double2(a,&l_a,n, woffset, sys_size);
          load_array_reg8_double2(b,&l_b,n, woffset, sys_size);
          load_array_reg8_double2(c,&l_c,n, woffset, sys_size);
          load_array_reg8_double2(d,&l_d,n, woffset, sys_size);
          #pragma unroll 16
          for(i=0; i<VEC; i++) {
            aa      = l_a.f[i];
            bb      = l_b.f[i] - aa*cc;
            dd      = l_d.f[i] - aa*dd;
            bb      = 1.0f/bb;
            cc      = bb*l_c.f[i];
            dd      = bb*dd;
            c2[n+i] = cc;
            d2[n+i] = dd;
          }
        }
        //
        // reverse pass
        //
        // The last element is treated differently, so do the the last VECtor separately
        l_d.f[VEC-1] = dd;
        n = sys_size - VEC;
        for(i=VEC-2; i>=0; i--) {
          dd       = d2[n+i] - c2[n+i]*dd;
          l_d.f[i] = dd;
        }
        store_array_reg8_double2(d,&l_d,n, woffset, sys_size);

        // Process the rest of the VECtors
        for(n=sys_size-2*VEC; n>=0; n-=VEC) {
          for(i=VEC-1; i>=0; i--) {
            dd       = d2[n+i] - c2[n+i]*dd;
            l_d.f[i] = dd;
          }
          store_array_reg8_double2(d,&l_d,n, woffset, sys_size);
          //store_array_reg8(u,&l_d,n, woffset);
        }
      } else { // Else if the data is not aligned do a naive solve on the first and last elements
        //
        // forward pass
        //
        int aligned;
        //int bound     = stride*ny*nz;
        int ind_floor = (ind/ALIGN_DOUBLE)*ALIGN_DOUBLE;
        int sys_off   = ind - ind_floor; // Offset index from ind_floor, 0 for aligned systems

        for(i=0; i<ALIGN_DOUBLE; i++) {
          if(i<sys_off) {
            c2[i] = 0.0;
            d2[i] = 0.0;
          }
          if(i==sys_off) {
            //bb    = 1.0f/__ldg( &b[ind] );
            //cc    = bb*__ldg( &c[ind] );
            //dd    = bb*__ldg( &d[ind] );
            bb    = 1.0f/b[ind];
            cc    = bb*c[ind];
            dd    = bb*d[ind];
            c2[i] = cc;
            d2[i] = dd;
            //u[ind_floor+i] = dd;//a[ind_floor+i];//i;//dd;//a[ind];//aa;//i;//dd;//i;//dd;
          }
          if(i>sys_off) {
            aa    = __ldg( &a[ind_floor+i] );
            bb    = __ldg( &b[ind_floor+i] ) - aa*cc;
            dd    = __ldg( &d[ind_floor+i] ) - aa*dd;
            bb    = 1.0f/bb;
            cc    = bb*__ldg( &c[ind_floor+i] );
            dd    = bb*dd;
            c2[i] = cc;
            d2[i] = dd;//i;//dd;
            //u[ind_floor+i] = dd;//aa;//i;//dd;//aa;//i;//dd;//i;//dd;
          }
        }
        // Process the rest of the VECtors
        //for(aligned=ALIGN_DOUBLE; aligned<nx-2*ALIGN_DOUBLE; aligned+=VEC) {
        //for(aligned=ALIGN_DOUBLE; aligned<nx-ALIGN_DOUBLE; aligned+=VEC) {
        for(aligned=ALIGN_DOUBLE; aligned<sys_size-VEC; aligned+=VEC) {
        //aligned=ALIGN_DOUBLE;
        //{
          load_array_reg8_double2_unaligned(a,&l_a,aligned, tid, sys_size, sys_size);
          load_array_reg8_double2_unaligned(b,&l_b,aligned, tid, sys_size, sys_size);
          load_array_reg8_double2_unaligned(c,&l_c,aligned, tid, sys_size, sys_size);
          load_array_reg8_double2_unaligned(d,&l_d,aligned, tid, sys_size, sys_size);
          #pragma unroll 16
          for(i=0; i<VEC; i++) {
            aa      = l_a.f[i];
            bb      = l_b.f[i] - aa*cc;
            dd      = l_d.f[i] - aa*dd;
            bb      = 1.0f/bb;
            cc      = bb*l_c.f[i];
            dd      = bb*dd;
            c2[aligned+i] = cc;
            d2[aligned+i] = dd;
            //u[ind_floor+aligned+i] = dd;//aa;//ind_floor+aligned+i;//aa;//ind_floor+aligned+i;//dd;//aa;//aligned+i;//dd;//aligned+i;//i;//dd;
          }
        }
        //printf("aligned = %d\n",aligned);
//        n = aligned;
//        int last_aligned = aligned;
//        //for(i=n-ALIGN_DOUBLE; i<nx; i++) {
//        //for(i=n; i<nx; i++) {
//        for(i=n; i<nx+sys_off; i++) {
//          aa    = __ldg( &a[ind_floor+i] );
//          bb    = __ldg( &b[ind_floor+i] ) - aa*cc;
//          dd    = __ldg( &d[ind_floor+i] ) - aa*dd;
//          bb    = 1.0f/bb;
//          cc    = bb*__ldg( &c[ind_floor+i] );
//          dd    = bb*dd;
//        //c2[sys_off+i] = cc;
//        //d2[sys_off+i] = dd;
//          c2[i] = cc;
//          d2[i] = dd;
//                      //u[ind_floor+i] = dd;//aa;//ind_floor+i;//aa;//i;//dd;//aa;// i;//dd;//i;//dd;
//        }
        //n = aligned-VEC;
        n = aligned;
        int last_aligned = aligned;
        //int j;
        for(; n+1 < ((sys_size+sys_off)/2) * 2; n+=2) {
          //l_a.vec[0] = *(double2*) &a[ind_floor+n];
          //l_b.vec[0] = *(double2*) &b[ind_floor+n];
          //l_c.vec[0] = *(double2*) &c[ind_floor+n];
          //l_d.vec[0] = *(double2*) &d[ind_floor+n];
          l_a.vec[0] = __ldg( (double2*) &a[ind_floor+n] );
          l_b.vec[0] = __ldg( (double2*) &b[ind_floor+n] );
          l_c.vec[0] = __ldg( (double2*) &c[ind_floor+n] );
          l_d.vec[0] = __ldg( (double2*) &d[ind_floor+n] );
          for(i=0; i<2; i++) {
            aa      = l_a.f[i];
            bb      = l_b.f[i] - aa*cc;
            dd      = l_d.f[i] - aa*dd;
            bb      = 1.0f/bb;
            cc      = bb*l_c.f[i];
            dd      = bb*dd;
            l_c.f[i] = cc;
            l_d.f[i] = dd;
            //u[ind_floor+n+i] = dd;
          }
          *((double2*)&c2[n]) = l_c.vec[0];
          *((double2*)&d2[n]) = l_d.vec[0];
          //*((double2*)& u[ind_floor+n]) = (double2){ind_floor+n, ind_floor+n+1, ind_floor+n+2, ind_floor+n+3};//l_d.vec[0];
          //*((double2*)& u[ind_floor+n]) = l_d.vec[0];//l_a.vec[0];//(double2){n, n+1, n+2, n+3};//l_d.vec[0];
        }

        if(n == ((sys_size+sys_off)/2) * 2) {
          i = n;
        } else {
          i = n-2;
        }

        for(; i<sys_size+sys_off; i++) {
            aa    = a[ind_floor+i];
            bb    = b[ind_floor+i] - aa*cc;
            dd    = d[ind_floor+i] - aa*dd;
            bb    = 1.0f/bb;
            cc    = bb*c[ind_floor+i];
            dd    = bb*dd;
            c2[i] = cc;
            d2[i] = dd;
            //aa    = __ldg( &a[ind_floor+i] );
            //bb    = __ldg( &b[ind_floor+i] ) - aa*cc;
            //dd    = __ldg( &d[ind_floor+i] ) - aa*dd;
            //bb    = 1.0f/bb;
            //cc    = bb*__ldg( &c[ind_floor+i] );
            //dd    = bb*dd;
            //c2[i] = cc;
            //d2[i] = dd;
            // u[ind_floor+i] = dd;
        }
        //
        // reverse pass
        //

        //d[ind+nx-1] = dd;
        //u[ind+nx-1] = nx-1;//dd;//nx-1;//dd;
        for(i=sys_size+sys_off-1; i>=last_aligned; i--) {
          dd       = d2[i] - c2[i]*dd;
          d[ind_floor+i] = dd;
          //u[ind_floor+i] = dd;//i-sys_off;//dd;//d2[sys_off+i];//dd;
        }
        //for(n=ALIGN_DOUBLE; n<nx-2*ALIGN_DOUBLE; n+=VEC) {
        //for(aligned=nx-VEC-1; aligned>=ALIGN_DOUBLE; aligned--) {
        //  dd       = d2[aligned] - c2[aligned]*dd;
        //  d[ind_floor+aligned] = dd;
        //  //u[ind_floor+aligned] = dd;//n;//dd;//d2[sys_off+i];//dd;
        //  //u[ind_floor+aligned] = dd;//aligned;//n;//dd;//d2[sys_off+i];//dd;
        //}
        for(aligned=last_aligned-VEC; aligned>=ALIGN_DOUBLE; aligned-=VEC) {
          for(i=VEC-1; i>=0; i--) {
            dd       = d2[aligned+i] - c2[aligned+i]*dd;
            l_d.f[i] = dd;//aligned+i;//dd;//aligned+i;//dd;
            //u[ind_floor+aligned+i] = dd;//i-sys_off;//dd;//d2[sys_off+i];//dd;
          }
          //store_array_reg8_unaligned(u,&l_d,aligned, tid, nx, nx);
          store_array_reg8_double2_unaligned(d,&l_d,aligned, tid, sys_size, sys_size);
        }
        for(i=ALIGN_DOUBLE-1; i>=sys_off; i--) {
        //for(i=ALIGN_DOUBLE-1; i>=0; i--) {
          dd       = d2[i] - c2[i]*dd;
          d[ind_floor+i] = dd;
          //u[ind_floor+i] = dd;//n;//dd;//d2[sys_off+i];//dd;
        }
      }
    // Else if the system is a boundary one, use naive algorithm to solve it
    } else {
      //
      // forward pass
      //
      //bb    = 1.0f/__ldg( &b[ind] );
      //cc    = bb*__ldg( &c[ind] );
      //dd    = bb*__ldg( &d[ind] );
      bb    = 1.0f/b[ind];
      cc    = bb*c[ind];
      dd    = bb*d[ind];
      c2[0] = cc;
      d2[0] = dd;

      //u[ind] = 0;//dd;

      for(i=1; i<sys_size; i++) {
        ind   = ind + 1;
        aa    = __ldg( &a[ind] );
        bb    = __ldg( &b[ind] ) - aa*cc;
        dd    = __ldg( &d[ind] ) - aa*dd;
        bb    = 1.0f/bb;
        cc    = bb*__ldg( &c[ind] );
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
        ind    = ind - 1;
        dd     = d2[i] - c2[i]*dd;
        d[ind] = dd;
        //u[ind] = dd;
      }
    }
  }






//  int     i, j, k;             // Indices for 3D
//  double8 l_a, l_b, l_c, l_d;  // Register buffers for rapid execution
//  double  aa, bb, cc, dd;
//  double  c2[N_MAX],d2[N_MAX]; // Temporary arrays for Thomas algorithm
//
//  //assert(blockDim.x == 32); // The algorithm is tuned for warp level optimization
//
//  //
//  // set up indices for main block
//  //
//  j   = threadIdx.x + blockIdx.x*blockDim.x;
//  k   = threadIdx.y + blockIdx.y*blockDim.y;
//  //int ind = nx*(j+k*ny);
//  //int goffset = ind - threadIdx.x*nx;
//  int goffset = k*ny*nx + (j/WARP_SIZE)*WARP_SIZE*nx; // Global memory offset: unique to a warp; every thread in a warp calculates the same goffset, which is the "begining" of 3D tile
//  int n; // Iterator for every 16th element along dimension X
//  //
//  // forward pass
//  //
//  n=0;
//
//  load_array_reg8_double(a,&l_a,n, goffset, nx);
//  load_array_reg8_double(b,&l_b,n, goffset, nx);
//  load_array_reg8_double(c,&l_c,n, goffset, nx);
//  load_array_reg8_double(d,&l_d,n, goffset, nx);
//  bb    = 1.0f/l_b.f[0];
//  cc    = bb*l_c.f[0];
//  dd    = bb*l_d.f[0];
//  c2[0] = cc;
//  d2[0] = dd;
//
////  #pragma unroll 16
//  for(i=1; i<VEC; i++) {
//    aa    = l_a.f[i];
//    bb    = l_b.f[i] - aa*cc;
//    dd    = l_d.f[i] - aa*dd;
//    bb    = 1.0f/bb;
//    cc    = bb*l_c.f[i];
//    dd    = bb*dd;
//    c2[i] = cc;
//    d2[i] = dd;
//  }
//
//  for(n=VEC; n<nx; n+=VEC) {
//    load_array_reg8_double(a,&l_a,n, goffset, nx);
//    load_array_reg8_double(b,&l_b,n, goffset, nx);
//    load_array_reg8_double(c,&l_c,n, goffset, nx);
//    load_array_reg8_double(d,&l_d,n, goffset, nx);
////    #pragma unroll VEC
//    for(i=0; i<VEC; i++) {
//      aa      = l_a.f[i];
//      bb      = l_b.f[i] - aa*cc;
//      dd      = l_d.f[i] - aa*dd;
//      bb      = 1.0f/bb;
//      cc      = bb*l_c.f[i];
//      dd      = bb*dd;
//      c2[n+i] = cc;
//      d2[n+i] = dd;
//    }
//  }
//  //
//  // reverse pass
//  //
//  l_d.f[VEC-1] = dd;
//  n = nx - VEC;
//  for(i=VEC-2; i>=0; i--) {
//    dd        = d2[n+i] - c2[n+i]*dd;
//    l_d.f[i]  = dd;
//  }
//
//  store_array_reg8_double(d,&l_d,n, goffset, nx);
////  #pragma unroll
//  for(n=nx-2*VEC; n>=0; n-=VEC) {
////    #pragma unroll
//    for(i=VEC-1; i>=0; i--) {
//      dd       = d2[n+i] - c2[n+i]*dd;
//      l_d.f[i] = dd;
//    }
//    store_array_reg8_double(d,&l_d,n, goffset, nx);
//    //store_array_reg8(u,&l_d,n, goffset);
//  }
}

#endif
