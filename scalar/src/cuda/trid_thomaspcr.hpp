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

// Written by Jeremy Appleyard with contributions from Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#include <stdio.h>
#include "trid_thomaspcr_small.h"
#include "trid_thomaspcr_large.hpp"

// This is an extension of Mike Giles' 1D Black Scholes work (http://people.maths.ox.ac.uk/gilesm/codes/BS_1D/).
// The original code iterated over time within the kenrel without global memory interaction. This version
// loads from global memory at the start, performs one iteration, then writes back to global memory at the end.

// Note: We use warp synchronous programming, and this only works on devices with a warp size of 32.

// In this version the data is assumed to have a stride of one between consecutive elements (ie. contiguous).
template <typename REAL, int regStoreSize, int t_warpSize>
__device__ void loadDataIntoRegisters_contig(volatile REAL *smem, REAL *regArray, const REAL* __restrict__ devArray,
                                      int tid, int smemOff, const int length, int ID, int numTrids, REAL blank) {
   
   REAL regTmp[regStoreSize];
                                      
   // This step is required due to the volatile keyword. We want to pipeline the loads
   // but the volatile store into shared memory would forbid this if we went there directly.
   for (int i=0; i<regStoreSize; i++) {
      int gmemIdx = i * t_warpSize + tid;
         
      if (gmemIdx < length) regTmp[i] = devArray[gmemIdx];
      else                  regTmp[i] = blank;
   }
            
   for (int i=0; i<regStoreSize; i++) {
      int smemIdx = i * t_warpSize + tid;
         
      smem[smemIdx + smemOff] = regTmp[i];
   }
   
   for (int i=0; i<regStoreSize; i++) {
      regArray[i] = smem[i + tid * regStoreSize + smemOff];
   }

//   REAL regTmp[regStoreSize];
//
//   // This step is required due to the volatile keyword. We want to pipeline the loads
//   // but the volatile store into shared memory would forbid this if we went there directly.
//   for (int i=0; i<regStoreSize; i++) {
//      int gmemIdx = i * t_warpSize + tid;
//
//      if (gmemIdx < length) regTmp[i] = devArray[gmemIdx];
//      else                  regTmp[i] = blank;
//   }
//
//   for (int i=0; i<regStoreSize; i++) {
//      int smemIdx = i * (t_warpSize+4) + (tid/regStoreSize)*(regStoreSize+1) + tid%regStoreSize;
//
//      smem[smemIdx + smemOff] = regTmp[i];
//   }
//
//   for (int i=0; i<regStoreSize; i++) {
//      regArray[i] = smem[i + tid * (regStoreSize+1) + smemOff];
//   }
}

#define VEC 8
typedef union {
  float4 vec[VEC/4];
  float  f[VEC];
} float8;

inline __device__ void transpose2x2xor(float8* la) {
// Butterfly transpose with XOR
//  float4 tmp1;
//  if (threadIdx.x&1) {
//    tmp1 = (*la).vec[0];
//  } else {
//    tmp1 = (*la).vec[1];
//  }
//
//  tmp1.x = __shfl_xor(tmp1.x,1);
//  tmp1.y = __shfl_xor(tmp1.y,1);
//  tmp1.z = __shfl_xor(tmp1.z,1);
//  tmp1.w = __shfl_xor(tmp1.w,1);
//
//  if (threadIdx.x&1) {
//    (*la).vec[0] = tmp1;
//  } else {
//    (*la).vec[1] = tmp1;
//  }

  // Transpose with cyclic shuffle
  float4 tmp;
  (*la).vec[0] = __shfl((*la).vec[0],(threadIdx.x+1) % WARP_SIZE); // __shfl_down() could not be used, since it doesn't work in a round buffer fashion -> data would be lost
  if(threadIdx.x%2 == 0) {
    tmp = (*la).vec[0];
    (*la).vec[0] = (*la).vec[1];
    (*la).vec[1] = tmp;
  }
  (*la).vec[0] = __shfl((*la).vec[0],(threadIdx.x-1) % WARP_SIZE);
}

template <typename REAL, int regStoreSize, int t_warpSize>
__device__ void loadDataIntoRegisters_contig_shfl(volatile REAL *smem, REAL *regArray, const REAL* __restrict__ devArray,
                                      int tid, int smemOff, const int length, int ID, int numTrids, REAL blank) {

   //float8 *regArray8 = (float8*) regArray;
   //float8 *devArray8 = (float8*) devArray;

   //REAL regTmp[regStoreSize];
   //float8 regTmp8;

   // This step is required due to the volatile keyword. We want to pipeline the loads
   // but the volatile store into shared memory would forbid this if we went there directly.
   for (int i=0; i<regStoreSize/4; i++) {
//      int gmemIdx = i * t_warpSize + tid;
     int gmemIdx = i * regStoreSize/4 + (tid/2)*4 + tid%2;
     //int gmemIdx = tid * regStoreSize/4 + i;

//      if (gmemIdx < length) regTmp[i] = devArray[gmemIdx];
//      else                  regTmp[i] = blank;
      //if (gmemIdx*4 < length) (*regArray8).vec[i] = ((float4*)devArray)[gmemIdx];
     ((float8*)regArray)->vec[i] = ((const float4* __restrict__)devArray)[gmemIdx];
      //if (gmemIdx < length) ((float8*)regArray)->vec[i] = ((float4*)devArray)[gmemIdx];
      //else                    regTmp8.vec[i] = blank;
   }


   transpose2x2xor((float8*)regArray);
//   for (int i=0; i<regStoreSize; i++) {
//      int smemIdx = i * t_warpSize + tid;
//
//      smem[smemIdx + smemOff] = regTmp8.f[i];
//   }

//   for (int i=0; i<regStoreSize; i++) {
//      //regArray[i] = smem[i + tid * regStoreSize + smemOff];
//      regArray[i] = regTmp8.f[i];
//   }
}

// In this version the data is assumed to have a stride of one between consecutive elements (ie. contiguous).
template <typename REAL, int regStoreSize, int t_warpSize>
__device__ void loadDataIntoRegisters_contig(volatile REAL *smem, REAL *regArray,  REAL*  devArray,
                                      int tid, int smemOff, const int length, int ID, int numTrids, REAL blank) {

   REAL regTmp[regStoreSize];

   // This step is required due to the volatile keyword. We want to pipeline the loads
   // but the volatile store into shared memory would forbid this if we went there directly.
   for (int i=0; i<regStoreSize; i++) {
      int gmemIdx = i * t_warpSize + tid;

      if (gmemIdx < length) regTmp[i] = devArray[gmemIdx];
      else                  regTmp[i] = blank;
   }

   for (int i=0; i<regStoreSize; i++) {
      int smemIdx = i * t_warpSize + tid;

      smem[smemIdx + smemOff] = regTmp[i];
   }

   for (int i=0; i<regStoreSize; i++) {
      regArray[i] = smem[i + tid * regStoreSize + smemOff];
   }
}


template <typename REAL, int regStoreSize, int blockSize, int blocksPerSMX, int t_warpSize, int INC>
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
__global__ void batchedTrid_contig_ker(REAL*  x, const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c,
                              REAL* d,
                             const int length, const int stride, const int numTrids) {
   REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize], d_reg[regStoreSize],
        aa[regStoreSize], cc[regStoreSize], dd[regStoreSize]; 
        
   REAL bbi;
   
   //volatile REAL __shared__ smem[(regStoreSize+1) * blockSize];
   volatile REAL __shared__ smem[regStoreSize * blockSize];
   
   int ID = (blockSize * blockIdx.x + threadIdx.x) / t_warpSize;
   
   if (ID >= numTrids) return;
   
   int warpID = (threadIdx.x / t_warpSize);
   int tid = (threadIdx.x % t_warpSize);
       
   //int smemOff = (regStoreSize+1) * t_warpSize * warpID;
   int smemOff = regStoreSize * t_warpSize * warpID;
   
   a += stride * ID;
   b += stride * ID;
   c += stride * ID;
   d += stride * ID;
   x += stride * ID;
   
   // We use the final variable to initialise areas outside the length of the matrix to the identity matrix.
   loadDataIntoRegisters_contig<REAL, regStoreSize, t_warpSize>(smem, a_reg, a, tid, smemOff, length, ID, numTrids, (REAL)0.);
   loadDataIntoRegisters_contig<REAL, regStoreSize, t_warpSize>(smem, b_reg, b, tid, smemOff, length, ID, numTrids, (REAL)1.);
   loadDataIntoRegisters_contig<REAL, regStoreSize, t_warpSize>(smem, c_reg, c, tid, smemOff, length, ID, numTrids, (REAL)0.);
   loadDataIntoRegisters_contig<REAL, regStoreSize, t_warpSize>(smem, d_reg, d, tid, smemOff, length, ID, numTrids, (REAL)0.);

//   loadDataIntoRegisters_contig_shfl<REAL, regStoreSize, t_warpSize>(smem, a_reg, a, tid, smemOff, length, ID, numTrids, (REAL)0.);
//   loadDataIntoRegisters_contig_shfl<REAL, regStoreSize, t_warpSize>(smem, b_reg, b, tid, smemOff, length, ID, numTrids, (REAL)1.);
//   loadDataIntoRegisters_contig_shfl<REAL, regStoreSize, t_warpSize>(smem, c_reg, c, tid, smemOff, length, ID, numTrids, (REAL)0.);
//   loadDataIntoRegisters_contig_shfl<REAL, regStoreSize, t_warpSize>(smem, d_reg, d, tid, smemOff, length, ID, numTrids, (REAL)0.);


   if (regStoreSize >= 2) {
     for (int i=0; i<2; i++) {
        bbi  = 1.0f / b_reg[i];
        dd[i] = bbi * d_reg[i];
        aa[i] = bbi * a_reg[i];
        cc[i] = bbi * c_reg[i];
     }
     
     // The in-thread reduction here breaks down when the 
     // number of elements per thread drops below three. 
     if (regStoreSize >= 3) {
       for (int i=2; i<regStoreSize; i++) {
          bbi   = 1.0f / ( b_reg[i] - a_reg[i]*cc[i-1] );
          dd[i] =  bbi * ( d_reg[i] - a_reg[i]*dd[i-1] );
          aa[i] =  bbi * (          - a_reg[i]*aa[i-1] );
          cc[i] =  bbi *   c_reg[i];
       }

       for (int i=regStoreSize-3; i>0; i--) {
          dd[i] =  dd[i] - cc[i]*dd[i+1];
          aa[i] =  aa[i] - cc[i]*aa[i+1];
          cc[i] =        - cc[i]*cc[i+1];
       }

       bbi = 1.0f / (1.0f - cc[0]*aa[1]);
       dd[0] =  bbi * ( dd[0] - cc[0]*dd[1] );
       aa[0] =  bbi *   aa[0];
       cc[0] =  bbi * (       - cc[0]*cc[1] );
     }
     
#if (__CUDA_ARCH__ >= 300)
     trid2_warp<REAL>(aa[0],cc[0],dd[0],aa[regStoreSize-1],cc[regStoreSize-1],dd[regStoreSize-1]);
#else
     volatile REAL __shared__ smem_trid2[blockSize];
     trid2_warp_s<REAL>(aa[0],cc[0],dd[0],aa[regStoreSize-1],cc[regStoreSize-1],dd[regStoreSize-1],smem_trid2);
#endif   

     for (int i=1; i<regStoreSize-1; i++) {
       dd[i] = dd[i] - aa[i]*dd[0] - cc[i]*dd[regStoreSize-1];
     }
   }
   else {
      bbi  = 1.0f / b_reg[0];
      dd[0] = bbi * d_reg[0];
      aa[0] = bbi * a_reg[0];
      cc[0] = bbi * c_reg[0];

#if (__CUDA_ARCH__ >= 300)
      trid1_warp<REAL>(aa[0],cc[0],dd[0]);
#else
      volatile REAL __shared__ smem_trid1[blockSize];
      trid1_warp_shared<REAL>(aa[0],cc[0],dd[0],smem_trid1);
#endif         
   }
   
   REAL regTmp[regStoreSize];
   
   // Solved. Reorder in smem
   for (int i=0; i<regStoreSize; i++) {
      smem[i + tid * regStoreSize + smemOff] = dd[i];
   }
   
   for (int i=0; i<regStoreSize; i++) {
      int smemIdx = i * t_warpSize + tid;
         
      regTmp[i] = smem[smemIdx + smemOff];
   }

   // Write to gmem
   for (int i=0; i<regStoreSize; i++) {
      int index = i * t_warpSize + tid;
      if (index < length) {
        if(INC==0) x[index]  = regTmp[i];
        else       x[index] += regTmp[i];

      }
   }
}

// ------------------------------------------------------------------------------------------


// In this version the data is assumed to have a large stride between elements

/* Each group of threads loads consecutive locations from the same set of trids. 
*  The data is loaded such that each thread group has n consecutive elements of n trids. 
* For example, with four groups:
*  Block 0, Threads 0-7 load elements 0-3 of trids 0-7
*  Block 0, Threads 8-15 load elements 4-7 of trids 0-7
*  Block 0, Threads 16-23 load elements 8-11 of trids 0-7
*  etc.
*  Block 1, Threads 0-7 load elements 0-3 of trids 8-15
*  Block 1, Threads 8-15 load elements 4-7 of trids 8-15
*  etc.
*
*  After loading these elements each thread then reduces its set of elements for each trid. 
*  This gives a reduced system which is placed in shared memory.
*
*  This reduced system is pulled out of shared memory such that each warp solves one trid.
*  After solving the reduced system we redistrubte the results, and each thread finalises
*  and writes back to memory in the same what that it was read.
*
*  Block sizing depends upon number of tridiagonals to be solved. In the above example,
*  this is 8, requiring an 8*32=256 thread block. The number of registers pre vector 
*  depends upon the length of the tridiagonal. The above example must have length 128
*  tridiagonals as each group of 8 threads only loads 4 elements. 256*4/8 = 128. If we
*  were instead to want to solve 256 length tridiagonals, we'd load 8 elements of each
*  per thread.  
*/

template <typename REAL, int regStoreSize, int t_warpSize>
__device__ void loadDataIntoRegisters_strided(REAL *regArray, const REAL* __restrict__ devArray,
                                              int tridiag, int subWarpIdx, int subThreadIdx,
                                              const int length, const int numTrids, const int stride, 
                                              int subBatchID, int subBatchTrid, const int subBatchSize,
                                              const int subBatchStride, const REAL blank) {
         
   for (int i=0; i<regStoreSize; i++) {   
      int element = subWarpIdx * regStoreSize + i;
 
      int gmemIdx = subBatchID * subBatchStride + subBatchTrid + stride * element;
      //int gmemIdx = tridiag + stride * element;
    
      if (element < length && tridiag < numTrids)
         regArray[i] = devArray[gmemIdx];
      else                                        
         regArray[i] = blank;
   }
}

template <typename REAL, int regStoreSize, int t_warpSize>
__device__ void loadDataIntoRegisters_strided(REAL *regArray,  REAL*  devArray,
                                              int tridiag, int subWarpIdx, int subThreadIdx,
                                              const int length, const int numTrids, const int stride,
                                              int subBatchID, int subBatchTrid, const int subBatchSize,
                                              const REAL blank) {

   for (int i=0; i<regStoreSize; i++) {
      int element = subWarpIdx * regStoreSize + i;

      int gmemIdx = subBatchTrid + subBatchID * subBatchSize * stride + stride * element;
      //int gmemIdx = tridiag + stride * element;

      if (element < length && tridiag < numTrids)
         regArray[i] = devArray[gmemIdx];
      else
         regArray[i] = blank;
   }
}

template <typename REAL, int regStoreSize, int groupsPerWarp, int blockSize, int blocksPerSMX, int t_warpSize, int INC>
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
__global__ void batchedTrid_strided_ker(REAL* __restrict__ x, const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c,
                             const REAL* __restrict__ d,
                             const int length, const int numTrids, 
                             const int stride, const int subBatchSize, const int subBatchStride) {
   REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize], d_reg[regStoreSize],
        aa[regStoreSize], cc[regStoreSize], dd[regStoreSize]; 
        
   REAL bbi;
   
   int warpIdx = threadIdx.x / t_warpSize;
   int subWarpIdx = threadIdx.x / (t_warpSize / groupsPerWarp);
   int subThreadIdx = threadIdx.x % (t_warpSize / groupsPerWarp);
   
   int tridiag = blockIdx.x * (t_warpSize / groupsPerWarp) + subThreadIdx;
   
   // subBatchSize is not a compile-time constant.
   int subBatchID = tridiag / subBatchSize;
   int subBatchTrid = tridiag % subBatchSize;
      
#if (__CUDA_ARCH__ >= 300)      
   REAL __shared__ smem[3 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   
   REAL *a0_s = &smem[0];
   REAL *an_s = &smem[1 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   REAL *c0_s = &smem[2 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   
   REAL *cn_s = &smem[0 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   REAL *d0_s = &smem[1 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   REAL *dn_s = &smem[2 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
#else
   REAL __shared__ smem[6 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   
   REAL *a0_s = &smem[0];
   REAL *an_s = &smem[1 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   REAL *c0_s = &smem[2 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   
   REAL *cn_s = &smem[3 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   REAL *d0_s = &smem[4 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
   REAL *dn_s = &smem[5 * ((t_warpSize + 1) * (blockSize / t_warpSize))];
#endif   

   // We use the final variable to initialise areas outside the length of the matrix to the identity matrix.
   loadDataIntoRegisters_strided<REAL, regStoreSize, t_warpSize>(a_reg, a, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)0.);
   loadDataIntoRegisters_strided<REAL, regStoreSize, t_warpSize>(b_reg, b, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)1.);
   loadDataIntoRegisters_strided<REAL, regStoreSize, t_warpSize>(c_reg, c, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)0.);
   loadDataIntoRegisters_strided<REAL, regStoreSize, t_warpSize>(d_reg, d, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)0.);        
      
   // Reduce the system
   if (regStoreSize >= 2) {
     for (int i=0; i<2; i++) {
        bbi  = 1.0f / b_reg[i];
        dd[i] = bbi * d_reg[i];
        aa[i] = bbi * a_reg[i];
        cc[i] = bbi * c_reg[i];
     }
     
     // The in-thread reduction here breaks down when the 
     // number of elements per thread drops below three. 
     if (regStoreSize >= 3) {
       for (int i=2; i<regStoreSize; i++) {
          bbi   = 1.0f / ( b_reg[i] - a_reg[i]*cc[i-1] );
          dd[i] =  bbi * ( d_reg[i] - a_reg[i]*dd[i-1] );
          aa[i] =  bbi * (          - a_reg[i]*aa[i-1] );
          cc[i] =  bbi *   c_reg[i];
       }

       for (int i=regStoreSize-3; i>0; i--) {
          dd[i] =  dd[i] - cc[i]*dd[i+1];
          aa[i] =  aa[i] - cc[i]*aa[i+1];
          cc[i] =        - cc[i]*cc[i+1];
       }

       bbi = 1.0f / (1.0f - cc[0]*aa[1]);
       dd[0] =  bbi * ( dd[0] - cc[0]*dd[1] );
       aa[0] =  bbi *   aa[0];
       cc[0] =  bbi * (       - cc[0]*cc[1] );
     }
   }
   else {
     bbi  = 1.0f / b_reg[0];
     dd[0] = bbi * d_reg[0];
     aa[0] = bbi * a_reg[0];
     cc[0] = bbi * c_reg[0];      
   }
   
   // Push intermediate values into shared memory
   // then pull from shared into one-warp-per-trid format     
   // Attempted to minimise syncs without limiting occ by smem
   int smemIdx1 = threadIdx.x + warpIdx;
   
   // Index to group without padding.
   int smemIdx2 = (threadIdx.x % t_warpSize) * (t_warpSize / groupsPerWarp);
   // Add the padding
   smemIdx2 += smemIdx2 / t_warpSize;
   // Offset to our trid
   smemIdx2 += warpIdx;
   
#if (__CUDA_ARCH__ >= 300)  
   a0_s[smemIdx1] = aa[0];
   an_s[smemIdx1] = aa[regStoreSize-1];
   
   c0_s[smemIdx1] = cc[0];
   
   __syncthreads();
   REAL aa0 = a0_s[smemIdx2];
   REAL aan = an_s[smemIdx2];
   
   REAL cc0 = c0_s[smemIdx2];
   
   __syncthreads();
   cn_s[smemIdx1] = cc[regStoreSize-1];
   
   d0_s[smemIdx1] = dd[0];
   dn_s[smemIdx1] = dd[regStoreSize-1];        
   
   __syncthreads();
   REAL ccn = cn_s[smemIdx2];
   
   REAL dd0 = d0_s[smemIdx2];   
   REAL ddn = dn_s[smemIdx2];    
#else
   a0_s[smemIdx1] = aa[0];
   an_s[smemIdx1] = aa[regStoreSize-1];
   
   c0_s[smemIdx1] = cc[0];
   cn_s[smemIdx1] = cc[regStoreSize-1];
   
   d0_s[smemIdx1] = dd[0];
   dn_s[smemIdx1] = dd[regStoreSize-1];  
   
   __syncthreads();
   REAL aa0 = a0_s[smemIdx2];
   REAL aan = an_s[smemIdx2];
   
   REAL cc0 = c0_s[smemIdx2];
   REAL ccn = cn_s[smemIdx2];
   
   REAL dd0 = d0_s[smemIdx2];   
   REAL ddn = dn_s[smemIdx2]; 
#endif
    
   // Cyclic reduction
   if (regStoreSize >= 2) {
#if (__CUDA_ARCH__ >= 300)
      trid2_warp<REAL>(aa0,cc0,dd0,aan,ccn,ddn);
#else
      REAL volatile __shared__ smem_CR[blockSize];
      trid2_warp_s<REAL>(aa0,cc0,dd0,aan,ccn,ddn,smem_CR);
#endif
   }
   else {
#if (__CUDA_ARCH__ >= 300)
      trid1_warp<REAL>(aan,ccn,ddn);
#else
      REAL volatile __shared__ smem_CR[blockSize];
      trid1_warp_shared<REAL>(aan,ccn,ddn,smem_CR);
#endif   
   }
   
   // Reorder back. We only need dd0 and ddn
   d0_s[smemIdx2] = dd0; 
   dn_s[smemIdx2] = ddn; 
   
   __syncthreads();
   
   dd[0]              = d0_s[smemIdx1];
   dd[regStoreSize-1] = dn_s[smemIdx1];
     
   // Compute the solution
   for (int i=1; i<regStoreSize-1; i++) {
     dd[i] = dd[i] - aa[i]*dd[0] - cc[i]*dd[regStoreSize-1];
   }
 
   // Solved. Write to output
   for (int i=0; i<regStoreSize; i++) {   
      int element = subWarpIdx * regStoreSize + i;
      int gmemIdx = subBatchID * subBatchStride + subBatchTrid + stride * element;
      //int gmemIdx = tridiag + stride * element;
      
      if (element < length && tridiag < numTrids) {
        if(INC==0) x[gmemIdx] = dd[i];
        else       x[gmemIdx] += dd[i];
      }
      //if (element < length && tridiag < numTrids) x[gmemIdx] = dd[i];
   }
}



template <typename REAL, int regStoreSize, int blockSize, int blocksPerSMX, int t_warpSize, int INC>
void cyclicRed_contig(REAL* x, const REAL* a, const REAL* b, const REAL* c, REAL* d, int length, int stride, int numTrids, int nBlocks, int nThreads) {
  if (sizeof(REAL) == 4) cudaFuncSetSharedMemConfig(batchedTrid_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, t_warpSize, INC>, cudaSharedMemBankSizeFourByte);
  else                   cudaFuncSetSharedMemConfig(batchedTrid_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, t_warpSize, INC>, cudaSharedMemBankSizeEightByte);

  cudaFuncSetCacheConfig(batchedTrid_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, t_warpSize, INC>, cudaFuncCachePreferShared);

  
  batchedTrid_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, t_warpSize, INC><<<nBlocks,nThreads>>>(x, a, b, c, d, length, stride, numTrids);
}



template <typename REAL, int INC>
int solveBatchedTrid_contig(int numTrids, int length, int stride, const REAL *a, const REAL *b, const REAL *c, REAL *d, REAL *x) {
   // Template args are as follows:
   // <Type, regStoreSize, blockSize, blocksPerSMX, t_warpSize>
   //
   // By scaling the length and reducing the number of tridiagonals we can combine multiple
   // tridiagonals into one connected system. This allows us to more efficiently solve small
   // systems which aren't near multiples of 32 by combining them into fewer larger systems.

  int nThreads = 128;            
  int nBlocks = (numTrids+(nThreads/32)-1) / (nThreads/32); 
  
  // There is probably a better way to do this
  int canDiv2 = (numTrids & 1 == 0) && (length == stride);
  int canDiv4 = (numTrids & 3 == 0) && (length == stride);
  
  if      (length <= 32)             cyclicRed_contig<REAL, 1,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 48 && canDiv2)  cyclicRed_contig<REAL, 3,  128, 6, 32, INC>(x, a, b, c, d, length * 2, stride, numTrids / 2, (nBlocks + 1) / 2,nThreads);
  else if (length <= 56 && canDiv4)  cyclicRed_contig<REAL, 7,  128, 6, 32, INC>(x, a, b, c, d, length * 4, stride, numTrids / 4, (nBlocks + 3) / 4,nThreads);                  
  else if (length <= 64)             cyclicRed_contig<REAL, 2,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 80 && canDiv2)  cyclicRed_contig<REAL, 5,  128, 6, 32, INC>(x, a, b, c, d, length * 2, stride, numTrids / 2, (nBlocks + 1) / 2,nThreads);
  else if (length <= 96)             cyclicRed_contig<REAL, 3,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 112 && canDiv2) cyclicRed_contig<REAL, 7,  128, 6, 32, INC>(x, a, b, c, d, length * 2, stride, numTrids / 2, (nBlocks + 1) / 2,nThreads);
  else if (length <= 128)            cyclicRed_contig<REAL, 4,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 160)            cyclicRed_contig<REAL, 5,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 192)            cyclicRed_contig<REAL, 6,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 224)            cyclicRed_contig<REAL, 7,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 256)            cyclicRed_contig<REAL, 8,  128, 6, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 288)            cyclicRed_contig<REAL, 9,  128, 5, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 320)            cyclicRed_contig<REAL, 10, 128, 5, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 352)            cyclicRed_contig<REAL, 11, 128, 5, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 384)            cyclicRed_contig<REAL, 12, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 416)            cyclicRed_contig<REAL, 13, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 448)            cyclicRed_contig<REAL, 14, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 480)            cyclicRed_contig<REAL, 15, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 512)            cyclicRed_contig<REAL, 16, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  else if (length <= 512)            cyclicRed_large_contig<REAL, 8, 256, 4, 64, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*2);
//  else if (length <= 352)            cyclicRed_large_contig<REAL, 6, 512, 2, 64, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);

//  else if (length <= 384)            cyclicRed_large_contig<REAL, 6, 512, 2, 64, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);

//  else if (length <= 416)            cyclicRed_large_contig<REAL, 8, 256, 4, 64, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
//  else if (length <= 448)            cyclicRed_large_contig<REAL, 3, 512, 2, 128, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
//  else if (length <= 480)            cyclicRed_large_contig<REAL, 4, 256, 4, 128, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
//  else if (length <= 512)            cyclicRed_large_contig<REAL, 4, 512, 2, 256, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
//  else if (length <= 576)            cyclicRed_large_contig<REAL, 4, 256, 2, 128, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
//  else if (length <= 640)            cyclicRed_large_contig<REAL, 5, 512, 2, 128, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);

  else if (length <= 768)            cyclicRed_large_contig<REAL, 12, 256, 4, 64, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*2);

//  else if (length <= 896)            cyclicRed_large_contig<REAL, 7, 512, 2, 128, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);

//  else if (length <= 1024)           cyclicRed_large_contig<REAL, 4, 512, 2, 256, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);

  //else if (length <= 544)            cyclicRed_contig<REAL, 17, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 576)            cyclicRed_contig<REAL, 18, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 608)            cyclicRed_contig<REAL, 19, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 640)            cyclicRed_contig<REAL, 20, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 672)            cyclicRed_contig<REAL, 21, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 704)            cyclicRed_contig<REAL, 22, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 736)            cyclicRed_contig<REAL, 23, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 768)            cyclicRed_contig<REAL, 24, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 800)            cyclicRed_contig<REAL, 25, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 832)            cyclicRed_contig<REAL, 26, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 864)            cyclicRed_contig<REAL, 27, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 896)            cyclicRed_contig<REAL, 28, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 928)            cyclicRed_contig<REAL, 29, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 960)            cyclicRed_contig<REAL, 30, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 992)            cyclicRed_contig<REAL, 31, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);
  //else if (length <= 1024)           cyclicRed_contig<REAL, 32, 128, 4, 32, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads);

  //else if (length <= 1024)           cyclicRed_large_contig<REAL, 16, 256, 4, 64, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*2);
  else if (length <= 1024)           cyclicRed_large_contig<REAL, 8, 512, 4, 128, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
  
  //else if (length <= 2048)           cyclicRed_large_contig<REAL, 4, 512, 2, 512, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);
  //else if (length <= 4096)           cyclicRed_large_contig<REAL, 8, 512, 2, 512, INC>(x, a, b, c, d, length, stride, numTrids, nBlocks,nThreads*4);

  else                   { printf("Contiguous, length > 1024, not supported\n"); return 1; }
  
  return 0;
}

template <typename REAL, int regStoreSize, int groupsPerWarp, int blockSize, int blocksPerSMX, int t_warpSize, int INC>
void cyclicRed_strided(REAL* x, const REAL* a, const REAL* b, const REAL* c, REAL* d, int length, int numTrids, int stride, int subBatchSize, int subBatchStride, int nBlocks, int nThreads) {
  if (sizeof(REAL) == 4) cudaFuncSetSharedMemConfig(batchedTrid_strided_ker<REAL, regStoreSize, groupsPerWarp, blockSize, blocksPerSMX, t_warpSize, INC>, cudaSharedMemBankSizeFourByte);
  else                   cudaFuncSetSharedMemConfig(batchedTrid_strided_ker<REAL, regStoreSize, groupsPerWarp, blockSize, blocksPerSMX, t_warpSize, INC>, cudaSharedMemBankSizeEightByte);

  cudaFuncSetCacheConfig(batchedTrid_strided_ker<REAL, regStoreSize, groupsPerWarp, blockSize, blocksPerSMX, t_warpSize, INC>, cudaFuncCachePreferShared);

  
  batchedTrid_strided_ker<REAL, regStoreSize, groupsPerWarp, blockSize, blocksPerSMX, t_warpSize, INC><<<nBlocks,nThreads>>>(x, a, b, c, d, length, numTrids, stride, subBatchSize, subBatchStride);
}

//template <typename REAL>
template <typename REAL, int INC>
int solveBatchedTrid_strided(int numTrids, int length, int stride1, int subBatchSize, int subBatchStride,
                                    const REAL *a, const REAL *b, const REAL *c, REAL *d, REAL *x, int fermi) {
  // Template args are as follows:
  // <Type, regStoreSize, blockSize, blocksPerSMX, warpSize, incremental>
  //
  // By scaling the length and reducing the number of tridiagonals we can combine multiple
  // tridiagonals into one connected system. This allows us to more efficiently solve small
  // systems which aren't near multiples of 32 by combining them into fewer larger systems.

  if (length > 1024) { printf("Strided, length > 1024, not supported\n"); return 1; }
  
  // For the case where subBatchSize >= numTrids
 
  // For the case where subBatchSize >= numTrids
  #define GROUPS_PER_WARP_SIMPLE_D 2 // Double
  #define GROUPS_PER_WARP_SIMPLE_F 1 // Float
  #define GROUPS_PER_WARP_SIMPLE_F_FERMI 2 // Float

  // For the full case
  #define GROUPS_PER_WARP_FULL_D 8 //4 // Double
  #define GROUPS_PER_WARP_FULL_F 4 //2 // Float
  
  // Simple case
  if (sizeof(REAL) == 4) {
    int nThreadsF = 32 * 32 / (GROUPS_PER_WARP_FULL_F);            
    int nBlocksF = (numTrids+(nThreadsF/32)-1) / (nThreadsF/32);

  
    if (subBatchSize >= numTrids) {
      if (fermi) {
        int nThreadsS = 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI);            
        int nBlocksS = (numTrids+(nThreadsS/32)-1) / (nThreadsS/32);

        if      (length <= 32)  cyclicRed_strided<REAL, 1,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 64)  cyclicRed_strided<REAL, 2,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 96)  cyclicRed_strided<REAL, 3,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 3, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 128) cyclicRed_strided<REAL, 4,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 160) cyclicRed_strided<REAL, 5,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 192) cyclicRed_strided<REAL, 6,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 224) cyclicRed_strided<REAL, 7,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 256) cyclicRed_strided<REAL, 8,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 288) cyclicRed_strided<REAL, 9,  GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 320) cyclicRed_strided<REAL, 10, GROUPS_PER_WARP_SIMPLE_F_FERMI, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F_FERMI), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      }
      else {
        int nThreadsS = 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F);            
        int nBlocksS = (numTrids+(nThreadsS/32)-1) / (nThreadsS/32);
        
        int nThreadsS2 = 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F * 2);            
        int nBlocksS2 = (numTrids+(nThreadsS2/32)-1) / (nThreadsS2/32);
        
        if      (length <= 32)  cyclicRed_strided<REAL, 1,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 64)  cyclicRed_strided<REAL, 2,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 96)  cyclicRed_strided<REAL, 3,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 128) cyclicRed_strided<REAL, 4,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 160) cyclicRed_strided<REAL, 5,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 192) cyclicRed_strided<REAL, 6,  GROUPS_PER_WARP_SIMPLE_F * 2, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F * 2), 3, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS2, nThreadsS2);
        else if (length <= 224) cyclicRed_strided<REAL, 7,  GROUPS_PER_WARP_SIMPLE_F * 2, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F * 2), 3, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS2, nThreadsS2);
        else if (length <= 256) cyclicRed_strided<REAL, 8,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 288) cyclicRed_strided<REAL, 9,  GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 320) cyclicRed_strided<REAL, 10, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);

        else if (length <= 352) cyclicRed_strided<REAL, 11, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 384) cyclicRed_strided<REAL, 12, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 416) cyclicRed_strided<REAL, 13, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 448) cyclicRed_strided<REAL, 14, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 480) cyclicRed_strided<REAL, 15, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
        else if (length <= 512) cyclicRed_strided<REAL, 16, GROUPS_PER_WARP_SIMPLE_F, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);

      }
    }
    else {
      if      (length <= 32)  cyclicRed_strided<REAL, 1,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 64)  cyclicRed_strided<REAL, 2,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 4, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 96)  cyclicRed_strided<REAL, 3,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 4, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 128) cyclicRed_strided<REAL, 4,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 4, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 160) cyclicRed_strided<REAL, 5,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 4, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 192) cyclicRed_strided<REAL, 6,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 3, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 224) cyclicRed_strided<REAL, 7,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 256) cyclicRed_strided<REAL, 8,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 288) cyclicRed_strided<REAL, 9,  GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 320) cyclicRed_strided<REAL, 10, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);

      else if (length <= 352) cyclicRed_strided<REAL, 11, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 384) cyclicRed_strided<REAL, 12, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 416) cyclicRed_strided<REAL, 13, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 448) cyclicRed_strided<REAL, 14, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 480) cyclicRed_strided<REAL, 15, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 512) {
//          #define GROUPS_PER_WARP_FULL_D 4 //2 // Float
//    	  int nThreadsF = 32 * 32 / (GROUPS_PER_WARP_FULL_D);
//    	  int nBlocksF = (numTrids+(nThreadsF/32)-1) / (nThreadsF/32);
//    	  cyclicRed_strided<REAL, 9, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      }
      else if (length <= 512) cyclicRed_strided<REAL, 16, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 544) cyclicRed_strided<REAL, 17, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 576) cyclicRed_strided<REAL, 18, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 608) cyclicRed_strided<REAL, 19, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 640) cyclicRed_strided<REAL, 20, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 672) cyclicRed_strided<REAL, 21, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 704) cyclicRed_strided<REAL, 22, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 736) cyclicRed_strided<REAL, 23, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 768) cyclicRed_strided<REAL, 24, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 800) cyclicRed_strided<REAL, 25, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 832) cyclicRed_strided<REAL, 26, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 864) cyclicRed_strided<REAL, 27, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 896) cyclicRed_strided<REAL, 28, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 928) cyclicRed_strided<REAL, 29, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 960) cyclicRed_strided<REAL, 30, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 992) cyclicRed_strided<REAL, 31, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 1024) cyclicRed_strided<REAL, 32, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 1024) cyclicRed_large_strided<REAL, 10, GROUPS_PER_WARP_FULL_F, 32 * 32 / (GROUPS_PER_WARP_FULL_F), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
    }
  }
  else {
    int nThreadsS = 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D);            
    int nBlocksS = (numTrids+(nThreadsS/32)-1) / (nThreadsS/32);
    
    int nThreadsF = 32 * 32 / (GROUPS_PER_WARP_FULL_D);            
    int nBlocksF = (numTrids+(nThreadsF/32)-1) / (nThreadsF/32);

  
    if (subBatchSize >= numTrids) {
      if      (length <= 32)  cyclicRed_strided<REAL, 1,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 64)  cyclicRed_strided<REAL, 2,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 96)  cyclicRed_strided<REAL, 3,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 3, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 128) cyclicRed_strided<REAL, 4,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 160) cyclicRed_strided<REAL, 5,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 192) cyclicRed_strided<REAL, 6,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 224) cyclicRed_strided<REAL, 7,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 256) cyclicRed_strided<REAL, 8,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 288) cyclicRed_strided<REAL, 9,  GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 320) cyclicRed_strided<REAL, 10, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);

      else if (length <= 352) cyclicRed_strided<REAL, 11, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 384) cyclicRed_strided<REAL, 12, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 416) cyclicRed_strided<REAL, 13, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 448) cyclicRed_strided<REAL, 14, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 480) cyclicRed_strided<REAL, 15, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
      else if (length <= 512) cyclicRed_strided<REAL, 16, GROUPS_PER_WARP_SIMPLE_D, 32 * 32 / (GROUPS_PER_WARP_SIMPLE_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksS, nThreadsS);
    }
    else {
      if      (length <= 32)  cyclicRed_strided<REAL, 1,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 0, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 64)  cyclicRed_strided<REAL, 2,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 6, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 96)  cyclicRed_strided<REAL, 3,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 6, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 128) cyclicRed_strided<REAL, 4,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 5, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 160) cyclicRed_strided<REAL, 5,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 4, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 192) cyclicRed_strided<REAL, 6,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 3, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 224) cyclicRed_strided<REAL, 7,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 2, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 256) cyclicRed_strided<REAL, 8,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 288) cyclicRed_strided<REAL, 9,  GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 320) cyclicRed_strided<REAL, 10, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);

//      else if (length <= 352) cyclicRed_strided<REAL, 11, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 384) cyclicRed_strided<REAL, 12, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 416) cyclicRed_strided<REAL, 13, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 448) cyclicRed_strided<REAL, 14, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 480) cyclicRed_strided<REAL, 15, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 512) cyclicRed_strided<REAL, 16, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 544) cyclicRed_strided<REAL, 17, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 576) cyclicRed_strided<REAL, 18, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 608) cyclicRed_strided<REAL, 19, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 640) cyclicRed_strided<REAL, 20, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 672) cyclicRed_strided<REAL, 21, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 704) cyclicRed_strided<REAL, 22, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 736) cyclicRed_strided<REAL, 23, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 768) cyclicRed_strided<REAL, 24, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 800) cyclicRed_strided<REAL, 25, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 832) cyclicRed_strided<REAL, 26, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 864) cyclicRed_strided<REAL, 27, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 892) cyclicRed_strided<REAL, 28, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 924) cyclicRed_strided<REAL, 29, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 960) cyclicRed_strided<REAL, 30, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
//      else if (length <= 992) cyclicRed_strided<REAL, 31, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);
      else if (length <= 1024) cyclicRed_large_strided<REAL, 10, GROUPS_PER_WARP_FULL_D, 32 * 32 / (GROUPS_PER_WARP_FULL_D), 1, 32, INC>(x, a, b, c, d, length, numTrids, stride1, subBatchSize, subBatchStride, nBlocksF, nThreadsF);    }
  }
  return 0;
}

// Solves Ax=d for many tridiagonal matrices concurrently
// Returns 0 if passed, 1 if failed.
/*
* numTrids:       The number of tridiagonals in the batch
* length:         Length of the tridiagonals
* stride1:        The stride between consecutive elements of a tridiagonal. 
* stride2:        The stride between starting elements of consecutive tridiagonals. 
* subBatchSize:   The number of tridiagonals in each sub-batch.
* subBatchStride: The stride between each sub-batch of tridiagonals.
* a:              Upper diagonal of A (first value must be 0)
* b:              Diagonal of A 
* c:              Lower diagonal of A (last value must be 0)
* d:              Right hand side
* x:              Solution vector to be filled
*/

/* 
* For example, a 3D cube with side length x, y, z would use:
*
* For the x direction:
*    stride1 = 1, stride2 = x, subBatchSize = y*z, subBatchStride = 0
*
* For the y direction:
*    stride1 = x, stride2 = 1, subBatchSize = x, subBatchStride = x*y
*
* For the z direction:
*    stride1 = x*y, stride2 = 1, subBatchSize = x*y, subBatchStride = 0
*
*
* Sub-batches are introduced to solve the 3D ADI type problem in the y direction where one
* has a set of consecutively starting tridiagonals followed by a big jump and then another
* set of consecutively starting tridiagonals.
*
*/

//template <typename REAL>
template <typename REAL, int INC>
int solveBatchedTrid(int numTrids, int length, int stride1, int stride2, int subBatchSize, int subBatchStride, 
                            const REAL *a, const REAL *b, const REAL *c, REAL *d, REAL *x) {
  static int firstCall = 1;
  static int major = 0;
  // Make sure our warp size is 32. Only do it once.
  if (firstCall) {
    int dev;
    cudaGetDevice(&dev);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
  
    if (deviceProp.warpSize != 32) {
      printf("solveBatchedTrid only supports devices with warp size 32. Warp size %d detected.\n", deviceProp.warpSize);
      return 1;
    }
    
    major = deviceProp.major;
    
    //if (deviceProp.major < 3) {
    //  printf("solveBatchedTrid only supports devices with compute capability >= 3.0. Compute capability %d.%d detected.\n", deviceProp.major, deviceProp.minor);
    //  return 1;
    //}
    
    firstCall = 0;
  }
  
  // Solve
  if      (stride1 == 1) return solveBatchedTrid_contig<REAL,INC>(numTrids, length, stride2, a, b, c, d, x);
  else if (stride2 == 1) return solveBatchedTrid_strided<REAL,INC>(numTrids, length, stride1, subBatchSize, subBatchStride, a, b, c, d, x, major < 3);
  else { printf("solveBatchedTrid: either stride1 or stride2 must equal 1 (currently %d, %d)\n", stride1, stride2); return 1; }
}

/*
template <typename REAL, int INC>
int runTest() {
   REAL *a, *b, *c, *d, *x;
   REAL *a_d, *b_d, *c_d, *d_d, *x_d;

   int length, numTrids;
   long long numReps;

   float time_ms, totalTime;
   cudaEvent_t start, stop;

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   numReps = 1000;
   for (int test=1; test<=3; test++) {
      printf("\n");
      for (length=256; length <= 256; length+=1) {
         numTrids = length * length;
         {
            totalTime = 0;

            a = (REAL *)malloc(length*numTrids*sizeof(REAL));
            b = (REAL *)malloc(length*numTrids*sizeof(REAL));
            c = (REAL *)malloc(length*numTrids*sizeof(REAL));
            d = (REAL *)malloc(length*numTrids*sizeof(REAL));
            x = (REAL *)malloc(length*numTrids*sizeof(REAL));

            // Initialise the same data for each mode.
            if (test == 1) {
              for (int trid=0; trid<numTrids; trid++) {
                 for (int i=0; i<length; i++) {
                    a[trid*length+i] = -1.+((double)i)/length;
                    b[trid*length+i] = i+1;
                    c[trid*length+i] = 1.+((double)i)/length;
                    d[trid*length+i] = trid+length-i;
                 }
                 a[trid*length] = 0;
                 c[trid*length+(length-1)] = 0;
              }
            }
            else if (test == 2) {
              for (int i=0; i<length; i++) {
                 for (int trid=0; trid<numTrids; trid++) {
                    a[numTrids*i+trid] = -1.+((double)i)/length;
                    b[numTrids*i+trid] = i+1;
                    c[numTrids*i+trid] = 1.+((double)i)/length;
                    d[numTrids*i+trid] = trid+length-i;

                    if (i == length-1) {
                      a[numTrids*0+trid] = 0;
                      c[numTrids*(length-1)+trid] = 0;
                    }
                 }
              }
            }
            else {
              for (int i=0; i<length; i++) {
                 for (int j=0; j<length; j++) {
                    for (int k=0; k<length; k++) {
                       int trid = j * length + k;

                       a[length*i+numTrids*j+k] = -1.+((double)i)/length;
                       b[length*i+numTrids*j+k] = i+1;
                       c[length*i+numTrids*j+k] = 1.+((double)i)/length;
                       d[length*i+numTrids*j+k] = trid+length-i;

                       if (i == length-1) {
                         a[length*0+numTrids*j+k] = 0;
                         c[length*(length-1)+numTrids*j+k] = 0;
                       }
                    }
                 }
              }
            }

            cudaMalloc((REAL **)&a_d, length*numTrids*sizeof(REAL));
            cudaMalloc((REAL **)&b_d, length*numTrids*sizeof(REAL));
            cudaMalloc((REAL **)&c_d, length*numTrids*sizeof(REAL));
            cudaMalloc((REAL **)&d_d, length*numTrids*sizeof(REAL));
            cudaMalloc((REAL **)&x_d, length*numTrids*sizeof(REAL));


            cudaMemcpy(a_d, a, length*numTrids*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(b_d, b, length*numTrids*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(c_d, c, length*numTrids*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(d_d, d, length*numTrids*sizeof(REAL), cudaMemcpyHostToDevice);

            // Zero the working & result arrays. Otherwise undefined reuslts can appear correct
            // as they often take the values of a previous correct run.
            cudaMemset(x_d, 0, length*numTrids*sizeof(REAL));
            cudaEventRecord(start);

            // Do the work.
            for (int repeat=0; repeat < numReps; repeat++) {
               switch (test) {
               // Solve in the x direction
               case 1:
                  solveBatchedTrid<REAL, INC>(numTrids, length, 1, length, numTrids, 0, a_d, b_d, c_d, d_d, x_d);
                  break;
               // Solve in the z direction
               case 2:
                  // z: stride1 = x*y, stride2 = 1, subBatchSize = x*y, subBatchStride = 0
                  solveBatchedTrid<REAL, INC>(numTrids, length, numTrids, 1, numTrids, 0, a_d, b_d, c_d, d_d, x_d);
                  break;
               // Solve in the y direction
               case 3:
                  // y: stride1 = x, stride2 = 1, subBatchSize = x, subBatchStride = x*y
                  solveBatchedTrid<REAL, INC>(numTrids, length, length, 1, length, numTrids, a_d, b_d, c_d, d_d, x_d);
                  break;

               default:
                  printf("Invalid case\n");
                  return 1;
               }

               cudaError_t ierr = cudaDeviceSynchronize();
               if (ierr != 0) {
                  printf("ERROR: %s\n", cudaGetErrorString(ierr));
                  return 1;
               }

            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            totalTime += time_ms;

            // Very basic check compared against a known correct solution.
            // if (CHECK_RESULT) {
               // cudaMemcpy(x, x_d, length*numTrids*sizeof(REAL), cudaMemcpyDeviceToHost);

               // double sum = 0;
               // for (int i=0; i<length*numTrids; i++) {
                  // sum += abs(x[i]);
               // }
               // printf("%10.7f\n", sum / numTrids);

               // Check Ax=b (TODO: Make this work for test 2 and 3)
<<<<<<< HEAD:scalar/libtrid/cuda/trid_thomaspcr.hpp
               /*if (test == 1) {
                  for (int trid=0; trid<numTrids; trid++) {
                     for (int i=0; i<length; i++) {
                        double dVal;
                        if (i == 0) {
                           dVal = b[trid*length+i] * x[trid*length+i] + c[trid*length+i] * x[trid*length+i+1];
                        }
                        else if (i == length - 1) {
                           dVal = a[trid*length+i] * x[trid*length+i-1] + b[trid*length+i] * x[trid*length+i];
                        }
                        else {
                           dVal = a[trid*length+i] * x[trid*length+i-1] + b[trid*length+i] * x[trid*length+i] + c[trid*length+i] * x[trid*length+i+1];
                        }

                        // 0.1% tolerance
                        if (dVal / d[trid*length+i] > 1.001 || d[trid*length+i] / dVal > 1.001) {
                           printf("Error, %f %f. %d %d\n", dVal, d[trid*length+i], trid, i);
                        }
                     }
                  }
               }
=======
               // if (test == 1) {
                  // for (int trid=0; trid<numTrids; trid++) {
                     // for (int i=0; i<length; i++) {
                        // double dVal;
                        // if (i == 0) {
                           // dVal = b[trid*length+i] * x[trid*length+i] + c[trid*length+i] * x[trid*length+i+1];
                        // }
                        // else if (i == length - 1) {
                           // dVal = a[trid*length+i] * x[trid*length+i-1] + b[trid*length+i] * x[trid*length+i];
                        // }
                        // else {
                           // dVal = a[trid*length+i] * x[trid*length+i-1] + b[trid*length+i] * x[trid*length+i] + c[trid*length+i] * x[trid*length+i+1];
                        // }

                        // // 0.1% tolerance
                        // if (dVal / d[trid*length+i] > 1.001 || d[trid*length+i] / dVal > 1.001) {
                           // printf("Error, %f %f. %d %d\n", dVal, d[trid*length+i], trid, i);
                        // }
                     // }
                  // }
               // }
>>>>>>> origin/master:scalar/libtrid/cuda/trid_strided_thomaspcr.hpp
           // }


            cudaFree(a_d);
            cudaFree(b_d);
            cudaFree(c_d);
            cudaFree(x_d);
            cudaFree(d_d);

            free(a);
            free(b);
            free(c);
            free(d);
            free(x);

            printf(" %4d, %6d, %E\n", length, numTrids, totalTime / (1000ll * numTrids * length * numReps));
         }
      }
   }

   return 0;
}

int main(int argc, char **argv) {
   return runTest<float, 1>();
}*/

