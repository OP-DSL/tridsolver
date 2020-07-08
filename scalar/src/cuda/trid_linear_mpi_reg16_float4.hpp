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

// Written by Toby Flynn, University of Warwick, T.Flynn@warwick.ac.uk, 2020
// Based on trid_linear_reg16_float4.hpp written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014

#ifndef TRID_LINEAR_MPI_REG16_FLOAT4_GPU_MPI__
#define TRID_LINEAR_MPI_REG16_FLOAT4_GPU_MPI__

#define VEC_F 16
#define WARP_SIZE 32

#include <assert.h>
#include <sm_35_intrinsics.h>
#include <generics/generics/shfl.h>

#include "cuda_shfl.h"

typedef union {
  float4 vec[VEC_F/4];
  float  f[VEC_F];
} float16;

// transpose4x4xor() - exchanges data between 4 consecutive threads
inline __device__ void transpose4x4xor(float16* la) {
  float4 tmp1;
  float4 tmp2;

  // Perform a 2-stage butterfly transpose

  // stage 1 (transpose each one of the 2x2 sub-blocks internally)
  if (threadIdx.x&1) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[2];
  } else {
    tmp1 = (*la).vec[1];
    tmp2 = (*la).vec[3];
  }

  tmp1.x = trid_shfl_xor(tmp1.x,1);
  tmp1.y = trid_shfl_xor(tmp1.y,1);
  tmp1.z = trid_shfl_xor(tmp1.z,1);
  tmp1.w = trid_shfl_xor(tmp1.w,1);

  tmp2.x = trid_shfl_xor(tmp2.x,1);
  tmp2.y = trid_shfl_xor(tmp2.y,1);
  tmp2.z = trid_shfl_xor(tmp2.z,1);
  tmp2.w = trid_shfl_xor(tmp2.w,1);

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

  tmp1.x = trid_shfl_xor(tmp1.x,2);
  tmp1.y = trid_shfl_xor(tmp1.y,2);
  tmp1.z = trid_shfl_xor(tmp1.z,2);
  tmp1.w = trid_shfl_xor(tmp1.w,2);

  tmp2.x = trid_shfl_xor(tmp2.x,2);
  tmp2.y = trid_shfl_xor(tmp2.y,2);
  tmp2.z = trid_shfl_xor(tmp2.z,2);
  tmp2.w = trid_shfl_xor(tmp2.w,2);

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
inline __device__ void load_array_reg16(const float* __restrict__ ga, float16* la, int n, int woffset, int sys_pads) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
  int tcol =  threadIdx.x       % 4; // Threads' colum index within a warp

  // Load 4 float4 values (64bytes) from an X-line
  gind = woffset + (4*(trow)) * sys_pads + tcol*4 + n; // First index in the X-line; woffset - warp offset in global memory
  int i;
  for(i=0; i<4; i++) {
    (*la).vec[i] = __ldg( ((float4*)&ga[gind]) );
    gind += sys_pads;
  }

  transpose4x4xor(la);
}

// Same as load_array_reg16() with the following exception: if sys_pads would cause unaligned access the index is rounded down to the its floor value to prevent missaligned access.
// ga - global array
// la - local array
inline __device__ void load_array_reg16_unaligned(float const* __restrict__ ga, float16* la, int n, int tid, int sys_pads, int sys_length) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  //int trow = (threadIdx.x % 32)/ 4; // Threads' row index within a warp
  int tcol = threadIdx.x % 4;       // Threads' colum index within a warp

  // Load 4 float4 values (64bytes) from an X-line
  gind = (tid/4)*4 * sys_pads  + n; // Global memory index for threads

  int gind_floor;
  int i;
  for(i=0; i<4; i++) {
    gind_floor   = (gind/ALIGN_FLOAT)*ALIGN_FLOAT + tcol*4; // Round index to floor
    (*la).vec[i] = __ldg( ((float4*)&ga[gind_floor]) );    // Get aligned data
    gind        += sys_pads;                         // Stride to the next system
  }

  transpose4x4xor(la);

}

// Store a tile with 32x16 elements into 32 float16 struct allocated in registers. Every 4 consecutive threads cooperate to transpose and store a 4 x float4 sub-tile.
// ga - global array
// la - local array
inline __device__ void store_array_reg16(float* __restrict__ ga, float16* la, int n, int woffset, int sys_pads) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
  int tcol =  threadIdx.x     % 4;   // Threads' colum index within a warp

  transpose4x4xor(la);

  gind = woffset + (4*(trow)) * sys_pads + tcol*4 + n;
  *((float4*)&ga[gind]) = (*la).vec[0];
  gind += sys_pads;
  *((float4*)&ga[gind]) = (*la).vec[1];
  gind += sys_pads;
  *((float4*)&ga[gind]) = (*la).vec[2];
  gind += sys_pads;
  *((float4*)&ga[gind]) = (*la).vec[3];
}

// Same as store_array_reg16() with the following exception: if stride would cause unaligned access the index is rounded down to the its floor value to prevent missaligned access.
// ga - global array
// la - local array
inline __device__ void store_array_reg16_unaligned(float* __restrict__ ga, float16* __restrict__ la, int n, int tid, int sys_pads, int sys_length) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int tcol = threadIdx.x % 4;       // Threads' colum index within a warp

  transpose4x4xor(la);

  // Store 4 float4 values (64bytes) to an X-line
  gind = (tid/4)*4 * sys_pads  + n; // Global memory index for threads

  int gind_floor;
  int i;
  for(i=0; i<4; i++) {
    gind_floor = (gind/ALIGN_FLOAT)*ALIGN_FLOAT + tcol*4; // Round index to floor
    *((float4*)&ga[gind_floor]) = (*la).vec[i];  // Put aligned data
    gind += sys_pads;                              // Stride to the next system
  }
}

// Modified Thomas forward pass for X dimension.
// Uses register shuffle optimization, can handle both aligned and unaligned memory
__global__ void
trid_linear_forward_float(const float *__restrict__ a, const float *__restrict__ b,
                    const float *__restrict__ c, const float *__restrict__ d,
                    float *__restrict__ aa, float *__restrict__ cc,
                    float *__restrict__ dd, float *__restrict__ boundaries,
                    int sys_size, int sys_pads, int sys_n) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // Warp ID in global scope - the ID wich the thread belongs to
  const int wid = tid / WARP_SIZE;
  // Global memory offset: unique to a warp;
  // every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid/4)*4+4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global memory read/write
  const int boundary_solve  = !optimized_solve && ( tid < (sys_n) );
  // A thread is active only if it works on valid memory
  const int active_thread   = optimized_solve || boundary_solve;
  // Check if aligned memory
  //const int aligned         = (sys_pads % VEC_F) == 0;
  const int aligned = (sys_pads % ALIGN_FLOAT) == 0;

  int n = 0;
  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Local arrays used in the register shuffle
  float16 l_a, l_b, l_c, l_d, l_aa, l_cc, l_dd;
  float bb, a2, c2, d2;

  // Check that this is an active thread
  if(active_thread) {
    // Check that this thread can perform an optimized solve
    if(optimized_solve && sys_size >= 48) {
      // Check whether memory is aligned
      if(aligned) {
        // Process first vector separately
        load_array_reg16(a,&l_a,n, woffset, sys_pads);
        load_array_reg16(b,&l_b,n, woffset, sys_pads);
        load_array_reg16(c,&l_c,n, woffset, sys_pads);
        load_array_reg16(d,&l_d,n, woffset, sys_pads);

        for (int i = 0; i < 2; i++) {
          bb = 1.0f / l_b.f[i];
          d2 = bb * l_d.f[i];
          a2 = bb * l_a.f[i];
          c2 = bb * l_c.f[i];
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }

        for(int i = 2; i < VEC_F; i++) {
          bb = 1.0f / (l_b.f[i] - l_a.f[i] * c2);
          d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
          a2 = (-l_a.f[i] * a2) * bb;
          c2 = l_c.f[i] * bb;
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }

        store_array_reg16(dd,&l_dd,n, woffset, sys_pads);
        store_array_reg16(cc,&l_cc,n, woffset, sys_pads);
        store_array_reg16(aa,&l_aa,n, woffset, sys_pads);

        // Forward pass
        for(n = VEC_F; n < sys_size - VEC_F; n += VEC_F) {
          load_array_reg16(a,&l_a,n, woffset, sys_pads);
          load_array_reg16(b,&l_b,n, woffset, sys_pads);
          load_array_reg16(c,&l_c,n, woffset, sys_pads);
          load_array_reg16(d,&l_d,n, woffset, sys_pads);
          #pragma unroll 16
          for(int i=0; i<VEC_F; i++) {
            bb = 1.0f / (l_b.f[i] - l_a.f[i] * c2);
            d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
            a2 = (-l_a.f[i] * a2) * bb;
            c2 = l_c.f[i] * bb;
            l_dd.f[i] = d2;
            l_aa.f[i] = a2;
            l_cc.f[i] = c2;
          }
          store_array_reg16(dd,&l_dd,n, woffset, sys_pads);
          store_array_reg16(cc,&l_cc,n, woffset, sys_pads);
          store_array_reg16(aa,&l_aa,n, woffset, sys_pads);
        }

        // Finish off last part that may not fill an entire vector
        for(int i = n; i < sys_size; i++) {
          int loc_ind = ind + i;
          bb = 1.0f / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
          dd[loc_ind] = (d[loc_ind] - a[loc_ind] * dd[loc_ind - 1]) * bb;
          aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
          cc[loc_ind] = c[loc_ind] * bb;
        }

        // Backwards pass
        n -= VEC_F;

        a2 = aa[ind + sys_size - 2];
        c2 = cc[ind + sys_size - 2];
        d2 = dd[ind + sys_size - 2];

        // Do part that may not fit in vector
        for(int i = sys_size - 3; i >= n + VEC_F; i--) {
          int loc_ind = ind + i;
          d2 = dd[loc_ind] - cc[loc_ind] * d2;
          a2 = aa[loc_ind] - cc[loc_ind] * a2;
          c2 = -cc[loc_ind] * c2;
          dd[loc_ind] = d2;
          aa[loc_ind] = a2;
          cc[loc_ind] = c2;
        }

        // Backwards pass using vectors
        for(; n > 0; n -= VEC_F) {
          load_array_reg16(aa,&l_aa,n, woffset, sys_pads);
          load_array_reg16(cc,&l_cc,n, woffset, sys_pads);
          load_array_reg16(dd,&l_dd,n, woffset, sys_pads);

          for(int i = VEC_F - 1; i >= 0; i--) {
            d2 = l_dd.f[i] - l_cc.f[i] * d2;
            a2 = l_aa.f[i] - l_cc.f[i] * a2;
            c2 = -l_cc.f[i] * c2;
            l_dd.f[i] = d2;
            l_cc.f[i] = c2;
            l_aa.f[i] = a2;
          }

          store_array_reg16(dd,&l_dd,n, woffset, sys_pads);
          store_array_reg16(cc,&l_cc,n, woffset, sys_pads);
          store_array_reg16(aa,&l_aa,n, woffset, sys_pads);
        }

        // Final vector processed separately so that element 0 can be handled
        n = 0;

        load_array_reg16(aa,&l_aa,n, woffset, sys_pads);
        load_array_reg16(cc,&l_cc,n, woffset, sys_pads);
        load_array_reg16(dd,&l_dd,n, woffset, sys_pads);

        for(int i = VEC_F - 1; i > 0; i--) {
          d2 = l_dd.f[i] - l_cc.f[i] * d2;
          a2 = l_aa.f[i] - l_cc.f[i] * a2;
          c2 = -l_cc.f[i] * c2;
          l_dd.f[i] = d2;
          l_cc.f[i] = c2;
          l_aa.f[i] = a2;
        }

        bb = 1.0f / (1.0f - l_cc.f[0] * a2);
        l_dd.f[0] = bb * (l_dd.f[0] - l_cc.f[0] * d2);
        l_aa.f[0] = bb * l_aa.f[0];
        l_cc.f[0] = bb * (-l_cc.f[0] * c2);

        store_array_reg16(dd,&l_dd,n, woffset, sys_pads);
        store_array_reg16(cc,&l_cc,n, woffset, sys_pads);
        store_array_reg16(aa,&l_aa,n, woffset, sys_pads);

        // Store boundary values for communication
        int i = tid * 6;
        boundaries[i + 0] = aa[ind];
        boundaries[i + 1] = aa[ind + sys_size - 1];
        boundaries[i + 2] = cc[ind];
        boundaries[i + 3] = cc[ind + sys_size - 1];
        boundaries[i + 4] = dd[ind];
        boundaries[i + 5] = dd[ind + sys_size - 1];
      } else {
        // Memory is unaligned
        int ind_floor = (ind/ALIGN_FLOAT)*ALIGN_FLOAT;
        int sys_off   = ind - ind_floor;

        // Handle start of unaligned memory
        for(int i = 0; i < VEC_F; i++) {
          if(i >= sys_off) {
            int loc_ind = ind_floor + i;
            if(i - sys_off < 2) {
              bb = 1.0 / b[loc_ind];
              d2 = bb * d[loc_ind];
              a2 = bb * a[loc_ind];
              c2 = bb * c[loc_ind];
              dd[loc_ind] = d2;
              aa[loc_ind] = a2;
              cc[loc_ind] = c2;
            } else {
              bb = 1.0 / (b[loc_ind] - a[loc_ind] * c2);
              d2 = (d[loc_ind] - a[loc_ind] * d2) * bb;
              a2 = (-a[loc_ind] * a2) * bb;
              c2 = c[loc_ind] * bb;
              dd[loc_ind] = d2;
              aa[loc_ind] = a2;
              cc[loc_ind] = c2;
            }
          }
        }

        int n = VEC_F;
        // Back to normal
        for(; n < sys_size - VEC_F; n += VEC_F) {
          load_array_reg16_unaligned(a,&l_a,n, tid, sys_pads, sys_size);
          load_array_reg16_unaligned(b,&l_b,n, tid, sys_pads, sys_size);
          load_array_reg16_unaligned(c,&l_c,n, tid, sys_pads, sys_size);
          load_array_reg16_unaligned(d,&l_d,n, tid, sys_pads, sys_size);
          #pragma unroll 16
          for(int i=0; i<VEC_F; i++) {
            bb = 1.0 / (l_b.f[i] - l_a.f[i] * c2);
            d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
            a2 = (-l_a.f[i] * a2) * bb;
            c2 = l_c.f[i] * bb;
            l_dd.f[i] = d2;
            l_aa.f[i] = a2;
            l_cc.f[i] = c2;
          }
          store_array_reg16_unaligned(dd,&l_dd,n, tid, sys_pads, sys_size);
          store_array_reg16_unaligned(cc,&l_cc,n, tid, sys_pads, sys_size);
          store_array_reg16_unaligned(aa,&l_aa,n, tid, sys_pads, sys_size);
        }

        // Handle end of unaligned memory
        for(int i = n; i < sys_size + sys_off; i++) {
          int loc_ind = ind_floor + i;
          bb = 1.0 / (b[loc_ind] - a[loc_ind] * c2);
          d2 = (d[loc_ind] - a[loc_ind] * d2) * bb;
          a2 = (-a[loc_ind] * a2) * bb;
          c2 = c[loc_ind] * bb;
          dd[loc_ind] = d2;
          aa[loc_ind] = a2;
          cc[loc_ind] = c2;
        }

        // Backwards pass
        d2 = dd[ind_floor + sys_size + sys_off - 2];
        a2 = aa[ind_floor + sys_size + sys_off - 2];
        c2 = cc[ind_floor + sys_size + sys_off - 2];

        n -= VEC_F;

        // Start with end of unaligned memory
        for(int i = sys_size + sys_off - 3; i >= n; i--) {
          int loc_ind = ind_floor + i;
          d2 = dd[loc_ind] - cc[loc_ind] * d2;
          a2 = aa[loc_ind] - cc[loc_ind] * a2;
          c2 = -cc[loc_ind] * c2;
          dd[loc_ind] = d2;
          aa[loc_ind] = a2;
          cc[loc_ind] = c2;
        }

        n -= VEC_F;

        // Back to normal
        for(; n > 0; n -= VEC_F) {
          load_array_reg16_unaligned(aa,&l_aa,n, tid, sys_pads, sys_size);
          load_array_reg16_unaligned(cc,&l_cc,n, tid, sys_pads, sys_size);
          load_array_reg16_unaligned(dd,&l_dd,n, tid, sys_pads, sys_size);

          for(int i = VEC_F - 1; i >= 0; i--) {
            d2 = l_dd.f[i] - l_cc.f[i] * d2;
            a2 = l_aa.f[i] - l_cc.f[i] * a2;
            c2 = -l_cc.f[i] * c2;
            l_dd.f[i] = d2;
            l_cc.f[i] = c2;
            l_aa.f[i] = a2;
          }

          store_array_reg16_unaligned(dd,&l_dd,n, tid, sys_pads, sys_size);
          store_array_reg16_unaligned(cc,&l_cc,n, tid, sys_pads, sys_size);
          store_array_reg16_unaligned(aa,&l_aa,n, tid, sys_pads, sys_size);
        }

        for(int i = n + VEC_F - 1; i > sys_off; i--) {
          int loc_ind = ind_floor + i;
          d2 = dd[loc_ind] - cc[loc_ind] * d2;
          a2 = aa[loc_ind] - cc[loc_ind] * a2;
          c2 = -cc[loc_ind] * c2;
          dd[loc_ind] = d2;
          aa[loc_ind] = a2;
          cc[loc_ind] = c2;
        }

        bb = 1.0 / (1.0 - cc[ind] * a2);
        dd[ind] = bb * (dd[ind] - cc[ind] * d2);
        aa[ind] = bb * aa[ind];
        cc[ind] = bb * (-cc[ind] * c2);

        // Store boundary values for communication
        int i = tid * 6;
        boundaries[i + 0] = aa[ind];
        boundaries[i + 1] = aa[ind + sys_size - 1];
        boundaries[i + 2] = cc[ind];
        boundaries[i + 3] = cc[ind + sys_size - 1];
        boundaries[i + 4] = dd[ind];
        boundaries[i + 5] = dd[ind + sys_size - 1];
      }
    } else {
      // Normal modified Thomas if not optimized solve

      for (int i = 0; i < 2; ++i) {
        bb = 1.0f / b[ind + i];
        dd[ind + i] = bb * d[ind + i];
        aa[ind + i] = bb * a[ind + i];
        cc[ind + i] = bb * c[ind + i];
      }

      if (sys_size >= 3) {
        // eliminate lower off-diagonal
        for (int i = 2; i < sys_size; i++) {
          int loc_ind = ind + i;
          bb = 1.0f / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
          dd[loc_ind] = (d[loc_ind] - a[loc_ind] * dd[loc_ind - 1]) * bb;
          aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
          cc[loc_ind] = c[loc_ind] * bb;
        }
        // Eliminate upper off-diagonal
        for (int i = sys_size - 3; i > 0; --i) {
          int loc_ind = ind + i;
          dd[loc_ind] = dd[loc_ind] - cc[loc_ind] * dd[loc_ind + 1];
          aa[loc_ind] = aa[loc_ind] - cc[loc_ind] * aa[loc_ind + 1];
          cc[loc_ind] = -cc[loc_ind] * cc[loc_ind + 1];
        }
        bb = 1.0f / (1.0f - cc[ind] * aa[ind + 1]);
        dd[ind] = bb * (dd[ind] - cc[ind] * dd[ind + 1]);
        aa[ind] = bb * aa[ind];
        cc[ind] = bb * (-cc[ind] * cc[ind + 1]);
      }

      // Store boundary values for communication
      int i = tid * 6;
      boundaries[i + 0] = aa[ind];
      boundaries[i + 1] = aa[ind + sys_size - 1];
      boundaries[i + 2] = cc[ind];
      boundaries[i + 3] = cc[ind + sys_size - 1];
      boundaries[i + 4] = dd[ind];
      boundaries[i + 5] = dd[ind + sys_size - 1];
    }
  }
}

// Modified Thomas backwards pass for X dimension.
// Uses register shuffle optimization, can handle both aligned and unaligned memory
template <int INC>
__global__ void
trid_linear_backward_float(const float *__restrict__ aa, const float *__restrict__ cc,
                     const float *__restrict__ dd, float *__restrict__ d,
                     float *__restrict__ u, const float *__restrict__ boundaries,
                     int sys_size, int sys_pads, int sys_n) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // Warp ID in global scope - the ID wich the thread belongs to
  const int wid = tid / WARP_SIZE;
  // Global memory offset: unique to a warp;
  // every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid/4)*4+4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global memory read/write
  const int boundary_solve  = !optimized_solve && ( tid < (sys_n) );
  // A thread is active only if it works on valid memory
  const int active_thread   = optimized_solve || boundary_solve;
  // Check if aligned memory
  //const int aligned         = (sys_pads % VEC_F) == 0;
  const int aligned = (sys_pads % ALIGN_FLOAT) == 0;

  int n = 0;
  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Local arrays used in register shuffle
  float16 l_aa, l_cc, l_dd, l_d, l_u;

  // Check if active thread
  if(active_thread) {
    // Set start and end dd values
    double dd0 = boundaries[2 * tid];
    double ddn = boundaries[2 * tid + 1];
    // Check if optimized solve
    if(optimized_solve && sys_size >= 48) {
      // Check if aligned memory
      if(aligned) {
        if(INC) {
          // Handle first vector
          load_array_reg16(aa,&l_aa,n, woffset, sys_pads);
          load_array_reg16(cc,&l_cc,n, woffset, sys_pads);
          load_array_reg16(dd,&l_dd,n, woffset, sys_pads);
          load_array_reg16(u,&l_u,n, woffset, sys_pads);

          l_u.f[0] += dd0;

          for(int i = 1; i < VEC_F; i++) {
            l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }

          store_array_reg16(u,&l_u,n, woffset, sys_pads);

          // Iterate over remaining vectors
          for(n = VEC_F; n < sys_size - VEC_F; n += VEC_F) {
            load_array_reg16(aa,&l_aa,n, woffset, sys_pads);
            load_array_reg16(cc,&l_cc,n, woffset, sys_pads);
            load_array_reg16(dd,&l_dd,n, woffset, sys_pads);
            load_array_reg16(u,&l_u,n, woffset, sys_pads);
            for(int i = 0; i < VEC_F; i++) {
              l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16(u,&l_u,n, woffset, sys_pads);
          }

          // Handle last section separately as might not completely fit into a vector
          for(; n < sys_size - 1; n++) {
            u[ind + n] += dd[ind + n] - aa[ind + n] * dd0 - cc[ind + n] * ddn;
          }

          u[ind + sys_size - 1] += ddn;
        } else {
          // Handle first vector
          load_array_reg16(aa,&l_aa,n, woffset, sys_pads);
          load_array_reg16(cc,&l_cc,n, woffset, sys_pads);
          load_array_reg16(dd,&l_dd,n, woffset, sys_pads);

          l_d.f[0] = dd0;

          for(int i = 1; i < VEC_F; i++) {
            l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }

          store_array_reg16(d,&l_d,n, woffset, sys_pads);

          // Iterate over all remaining vectors
          for(n = VEC_F; n < sys_size - VEC_F; n += VEC_F) {
            load_array_reg16(aa,&l_aa,n, woffset, sys_pads);
            load_array_reg16(cc,&l_cc,n, woffset, sys_pads);
            load_array_reg16(dd,&l_dd,n, woffset, sys_pads);
            for(int i = 0; i < VEC_F; i++) {
              l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16(d,&l_d,n, woffset, sys_pads);
          }

          // Handle last section separately as might not completely fit into a vector
          for(int i = n; i < sys_size - 1; i++) {
            d[ind + i] = dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
          }

          d[ind + sys_size - 1] = ddn;
        }
      } else {
        // Unaligned memory
        if(INC) {
          int ind_floor = (ind/ALIGN_FLOAT)*ALIGN_FLOAT;
          int sys_off   = ind - ind_floor;

          // Handle start of unaligned memory
          for(int i = 0; i < VEC_F; i++) {
            if(i >= sys_off) {
              int loc_ind = ind_floor + i;
              if(i == sys_off) {
                u[loc_ind] += dd0;
              } else {
                u[loc_ind] += dd[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
              }
            }
          }

          n = VEC_F;
          // Back to normal
          for(; n < sys_size - VEC_F; n += VEC_F) {
            load_array_reg16_unaligned(aa,&l_aa,n, tid, sys_pads, sys_size);
            load_array_reg16_unaligned(cc,&l_cc,n, tid, sys_pads, sys_size);
            load_array_reg16_unaligned(dd,&l_dd,n, tid, sys_pads, sys_size);
            load_array_reg16_unaligned(u,&l_u,n, tid, sys_pads, sys_size);
            #pragma unroll 16
            for(int i=0; i<VEC_F; i++) {
              l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16_unaligned(u,&l_u,n, tid, sys_pads, sys_size);
          }

          // Handle end of unaligned memory
          for(int i = n; i < sys_size + sys_off - 1; i++) {
            int loc_ind = ind_floor + i;
            u[loc_ind] += dd[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
          }

          u[ind + sys_size - 1] += ddn;
        } else {
          int ind_floor = (ind/ALIGN_FLOAT)*ALIGN_FLOAT;
          int sys_off   = ind - ind_floor;

          // Handle start of unaligned memory
          for(int i = 0; i < VEC_F; i++) {
            if(i >= sys_off) {
              int loc_ind = ind_floor + i;
              if(i == sys_off) {
                d[loc_ind] = dd0;
              } else {
                d[loc_ind] = dd[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
              }
            }
          }

          n = VEC_F;
          // Back to normal
          for(; n < sys_size - VEC_F; n += VEC_F) {
            load_array_reg16_unaligned(aa,&l_aa,n, tid, sys_pads, sys_size);
            load_array_reg16_unaligned(cc,&l_cc,n, tid, sys_pads, sys_size);
            load_array_reg16_unaligned(dd,&l_dd,n, tid, sys_pads, sys_size);
            load_array_reg16_unaligned(d,&l_d,n, tid, sys_pads, sys_size);
            #pragma unroll 16
            for(int i=0; i<VEC_F; i++) {
              l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16_unaligned(d,&l_d,n, tid, sys_pads, sys_size);
          }

          // Handle end of unaligned memory
          for(int i = n; i < sys_size + sys_off - 1; i++) {
            int loc_ind = ind_floor + i;
            d[loc_ind] = dd[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
          }

          d[ind + sys_size - 1] = ddn;
        }
      }
    } else {
      // Normal modified Thomas backwards pass if not optimized solve
      if(INC) {
        u[ind] += dd0;

        for(int i = 1; i < sys_size - 1; i++) {
          u[ind + i] += dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }

        u[ind + sys_size - 1] += ddn;
      } else {
        d[ind] = dd0;

        for(int i = 1; i < sys_size - 1; i++) {
          d[ind + i] = dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }

        d[ind + sys_size - 1] = ddn;
      }
    }
  }
}

#endif
