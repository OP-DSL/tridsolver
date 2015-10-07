/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Jeremy Appleyard and others. Please see the AUTHORS file in
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
 *     * The name of Jeremy Appleyard may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Jeremy Appleyard ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Jeremy Appleyard BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Written by Jeremy Appleyard with contributions from Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

// This is an extension of Mike Giles' 1D Black Scholes work (http://people.maths.ox.ac.uk/gilesm/codes/BS_1D/).
// The original code iterated over time within the kenrel without global memory interaction. This version
// loads from global memory at the start, performs one iteration, then writes back to global memory at the end.

// Note: We use warp synchronous programming, and this only works on devices with a warp size of 32.

// In this version the data is assumed to have a stride of one between consecutive elements (ie. contiguous).
template <typename REAL, int regStoreSize, int tridSolveSize>
__device__ void loadDataIntoRegisters_large_contig(volatile REAL *smem, REAL *regArray, const REAL* __restrict__ devArray,
                                      int tid, int smemOff, const int length, int ID, int numTrids, REAL blank) {
   
   REAL regTmp[regStoreSize];
                                      
   // This step is required due to the volatile keyword. We want to pipeline the loads
   // but the volatile store into shared memory would forbid this if we went there directly.
   for (int i=0; i<regStoreSize; i++) {
      int gmemIdx = i * tridSolveSize + tid;
         
      if (gmemIdx < length) regTmp[i] = devArray[gmemIdx];
      else                  regTmp[i] = blank;
   }
            
   for (int i=0; i<regStoreSize; i++) {
      int smemIdx = i * tridSolveSize + tid;
         
      smem[smemIdx + smemOff] = regTmp[i];
   }
   
   if (tridSolveSize > warpSize) __syncthreads();
   
   for (int i=0; i<regStoreSize; i++) {
      regArray[i] = smem[i + tid * regStoreSize + smemOff];
   }
   
   if (tridSolveSize > warpSize) __syncthreads();
}

// In this version the data is assumed to have a stride of one between consecutive elements (ie. contiguous).
template <typename REAL, int regStoreSize, int tridSolveSize>
__device__ void loadDataIntoRegisters_large_contig(volatile REAL *smem, REAL *regArray,  REAL*  devArray,
                                      int tid, int smemOff, const int length, int ID, int numTrids, REAL blank) {

   REAL regTmp[regStoreSize];

   // This step is required due to the volatile keyword. We want to pipeline the loads
   // but the volatile store into shared memory would forbid this if we went there directly.
   for (int i=0; i<regStoreSize; i++) {
      int gmemIdx = i * tridSolveSize + tid;

      if (gmemIdx < length) regTmp[i] = devArray[gmemIdx];
      else                  regTmp[i] = blank;
   }

   for (int i=0; i<regStoreSize; i++) {
      int smemIdx = i * tridSolveSize + tid;

      smem[smemIdx + smemOff] = regTmp[i];
   }
   
   if (tridSolveSize > warpSize) __syncthreads();

   for (int i=0; i<regStoreSize; i++) {
      regArray[i] = smem[i + tid * regStoreSize + smemOff];
   }
   
   if (tridSolveSize > warpSize) __syncthreads();
}


template <typename REAL, int regStoreSize, int blockSize, int blocksPerSMX, int tridSolveSize, int INC>
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
__global__ void batchedTrid_large_contig_ker(REAL*  x, const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c,
                              REAL* d,
                             const int length, const int stride, const int numTrids) {
   REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize], d_reg[regStoreSize],
        aa[regStoreSize], cc[regStoreSize], dd[regStoreSize]; 
        
   REAL bbi;
   
   volatile REAL __shared__ smem[(regStoreSize < 3 ? 3 : regStoreSize) * blockSize];
   
   int ID = (blockSize * blockIdx.x + threadIdx.x) / tridSolveSize;
   
   if (ID >= numTrids) return;
   
   int warpID = (threadIdx.x / tridSolveSize);
   int tid = (threadIdx.x % tridSolveSize);
       
   int smemOff = regStoreSize * tridSolveSize * warpID;
   
   a += stride * ID;
   b += stride * ID;
   c += stride * ID;
   d += stride * ID;
   x += stride * ID;
   
   // We use the final variable to initialise areas outside the length of the matrix to the identity matrix.
   loadDataIntoRegisters_large_contig<REAL, regStoreSize, tridSolveSize>(smem, a_reg, a, tid, smemOff, length, ID, numTrids, (REAL)0.);
   loadDataIntoRegisters_large_contig<REAL, regStoreSize, tridSolveSize>(smem, b_reg, b, tid, smemOff, length, ID, numTrids, (REAL)1.);
   loadDataIntoRegisters_large_contig<REAL, regStoreSize, tridSolveSize>(smem, c_reg, c, tid, smemOff, length, ID, numTrids, (REAL)0.);
   loadDataIntoRegisters_large_contig<REAL, regStoreSize, tridSolveSize>(smem, d_reg, d, tid, smemOff, length, ID, numTrids, (REAL)0.);
      
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
      if (tridSolveSize == 32) trid2_warp<REAL>(aa[0],cc[0],dd[0],aa[regStoreSize-1],cc[regStoreSize-1],dd[regStoreSize-1]);
      else                     trid2_warp_large<REAL, tridSolveSize, blockSize>(aa[0],cc[0],dd[0],aa[regStoreSize-1],cc[regStoreSize-1],dd[regStoreSize-1], smem);
#else
     volatile REAL __shared__ smem_trid2[blockSize];
     //trid2_warp_s<REAL>(aa[0],cc[0],dd[0],aa[regStoreSize-1],cc[regStoreSize-1],dd[regStoreSize-1],smem_trid2);
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
      //trid1_warp<REAL>(aa[0],cc[0],dd[0]);
#else
      volatile REAL __shared__ smem_trid1[blockSize];
      //trid1_warp_shared<REAL>(aa[0],cc[0],dd[0],smem_trid1);
#endif         
   }
   
   REAL regTmp[regStoreSize];
   
   if (tridSolveSize > warpSize) __syncthreads();   
   
   // Solved. Reorder in smem
   for (int i=0; i<regStoreSize; i++) {
      smem[i + tid * regStoreSize + smemOff] = dd[i];
   }
   
   if (tridSolveSize > warpSize) __syncthreads();
   
   for (int i=0; i<regStoreSize; i++) {
      int smemIdx = i * tridSolveSize + tid;
         
      regTmp[i] = smem[smemIdx + smemOff];
   }

   if (tridSolveSize > warpSize) __syncthreads();   
   
   // Write to gmem
   for (int i=0; i<regStoreSize; i++) {
      int index = i * tridSolveSize + tid;
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


template <typename REAL, int regStoreSize, int tridSolveSize>
__device__ void loadDataIntoRegisters_large_strided(REAL *regArray, const REAL* __restrict__ devArray,
                                              int tridiag, int subWarpIdx, int subThreadIdx,
                                              const int length, const int numTrids, const int stride, 
                                              int subBatchID, int subBatchTrid, const int subBatchSize,
                                              const int subBatchStride, const REAL blank) {
         
   for (int i=0; i<regStoreSize; i++) {   
      int element = subWarpIdx * regStoreSize + i;
 
      int gmemIdx = subBatchID * subBatchStride + subBatchTrid + stride * element;
    
      if (element < length && tridiag < numTrids)
         regArray[i] = devArray[gmemIdx];
      else                                        
         regArray[i] = blank;
   }
}

template <typename REAL, int regStoreSize, int tridSolveSize>
__device__ void loadDataIntoRegisters_large_strided(REAL *regArray,  REAL*  devArray,
                                              int tridiag, int subWarpIdx, int subThreadIdx,
                                              const int length, const int numTrids, const int stride,
                                              int subBatchID, int subBatchTrid, const int subBatchSize,
                                              const REAL blank) {

   for (int i=0; i<regStoreSize; i++) {
      int element = subWarpIdx * regStoreSize + i;

      int gmemIdx = subBatchTrid + subBatchID * subBatchSize * stride + stride * element;

      if (element < length && tridiag < numTrids)
         regArray[i] = devArray[gmemIdx];
      else
         regArray[i] = blank;
   }
}

template <typename REAL, int regStoreSize, int groupsPerTrid, int blockSize, int blocksPerSMX, int tridSolveSize, int INC>
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
__global__ void batchedTrid_large_strided_ker(REAL* __restrict__ x, const REAL* __restrict__ a, const REAL* __restrict__ b, const REAL* __restrict__ c,
                             const REAL* __restrict__ d,
                             const int length, const int numTrids, 
                             const int stride, const int subBatchSize, const int subBatchStride) {
   REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize], d_reg[regStoreSize],
        aa[regStoreSize], cc[regStoreSize], dd[regStoreSize]; 
        
   REAL bbi;
   
   int warpIdx = threadIdx.x / tridSolveSize;
   int subWarpIdx = threadIdx.x / (tridSolveSize / groupsPerTrid);
   int subThreadIdx = threadIdx.x % (tridSolveSize / groupsPerTrid);
   
   int tridiag = blockIdx.x * (tridSolveSize / groupsPerTrid) + subThreadIdx;
   
   // subBatchSize is not a compile-time constant.
   int subBatchID = tridiag / subBatchSize;
   int subBatchTrid = tridiag % subBatchSize;
      
#if (__CUDA_ARCH__ >= 300)      
   REAL __shared__ smem[3 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   
   REAL *a0_s = &smem[0];
   REAL *an_s = &smem[1 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   REAL *c0_s = &smem[2 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   
   REAL *cn_s = &smem[0 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   REAL *d0_s = &smem[1 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   REAL *dn_s = &smem[2 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
#else
   REAL __shared__ smem[6 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   
   REAL *a0_s = &smem[0];
   REAL *an_s = &smem[1 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   REAL *c0_s = &smem[2 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   
   REAL *cn_s = &smem[3 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   REAL *d0_s = &smem[4 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
   REAL *dn_s = &smem[5 * ((tridSolveSize + 1) * (blockSize / tridSolveSize))];
#endif   

   // We use the final variable to initialise areas outside the length of the matrix to the identity matrix.
   loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(a_reg, a, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)0.);
   loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(b_reg, b, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)1.);
   loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(c_reg, c, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
                                                                 stride, subBatchID, subBatchTrid, subBatchSize, subBatchStride, (REAL)0.);
   loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(d_reg, d, tridiag, subWarpIdx, subThreadIdx, length, numTrids,
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
   int smemIdx2 = (threadIdx.x % tridSolveSize) * (tridSolveSize / groupsPerTrid) + warpIdx;
   smemIdx2 += smemIdx2 / tridSolveSize;

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

   if (tridSolveSize > warpSize) __syncthreads();
    
   // Cyclic reduction
   if (regStoreSize >= 2) {
#if (__CUDA_ARCH__ >= 300)
      if (tridSolveSize == 32) trid2_warp<REAL>(aa0,cc0,dd0,aan,ccn,ddn);
      else                     trid2_warp_large<REAL, tridSolveSize, blockSize>(aa0,cc0,dd0,aan,ccn,ddn, smem);
#else
      REAL volatile __shared__ smem_CR[blockSize];
      //trid2_warp_s<REAL>(aa0,cc0,dd0,aan,ccn,ddn,smem_CR);
#endif
   }
   else {
#if (__CUDA_ARCH__ >= 300)
      //trid1_warp<REAL>(aan,ccn,ddn);
#else
      REAL volatile __shared__ smem_CR[blockSize];
      //trid1_warp_shared<REAL>(aan,ccn,ddn,smem_CR);
#endif   
   }
   
   if (tridSolveSize > warpSize) __syncthreads();   
   
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
      
      if (element < length && tridiag < numTrids) {
        if(INC==0) x[gmemIdx] = dd[i];
        else       x[gmemIdx] += dd[i];
      }
   }
}


template <typename REAL, int regStoreSize, int blockSize, int blocksPerSMX, int tridSolveSize, int INC>
void cyclicRed_large_contig(REAL* x, const REAL* a, const REAL* b, const REAL* c, REAL* d, int length, int stride, int numTrids, int nBlocks, int nThreads) {
  if (sizeof(REAL) == 4) cudaFuncSetSharedMemConfig(batchedTrid_large_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, tridSolveSize, INC>, cudaSharedMemBankSizeFourByte);
  else                   cudaFuncSetSharedMemConfig(batchedTrid_large_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, tridSolveSize, INC>, cudaSharedMemBankSizeEightByte);

  cudaFuncSetCacheConfig(batchedTrid_large_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, tridSolveSize, INC>, cudaFuncCachePreferShared);

  
  batchedTrid_large_contig_ker<REAL, regStoreSize, blockSize, blocksPerSMX, tridSolveSize, INC><<<nBlocks,nThreads>>>(x, a, b, c, d, length, stride, numTrids);
}

template <typename REAL, int regStoreSize, int groupsPerTrid, int blockSize, int blocksPerSMX, int tridSolveSize, int INC>
void cyclicRed_large_strided(REAL* x, const REAL* a, const REAL* b, const REAL* c, REAL* d, int length, int numTrids, int stride, int subBatchSize, int subBatchStride, int nBlocks, int nThreads) {
  if (sizeof(REAL) == 4) cudaFuncSetSharedMemConfig(batchedTrid_large_strided_ker<REAL, regStoreSize, groupsPerTrid, blockSize, blocksPerSMX, tridSolveSize, INC>, cudaSharedMemBankSizeFourByte);
  else                   cudaFuncSetSharedMemConfig(batchedTrid_large_strided_ker<REAL, regStoreSize, groupsPerTrid, blockSize, blocksPerSMX, tridSolveSize, INC>, cudaSharedMemBankSizeEightByte);

  cudaFuncSetCacheConfig(batchedTrid_large_strided_ker<REAL, regStoreSize, groupsPerTrid, blockSize, blocksPerSMX, tridSolveSize, INC>, cudaFuncCachePreferShared);

  
  batchedTrid_large_strided_ker<REAL, regStoreSize, groupsPerTrid, blockSize, blocksPerSMX, tridSolveSize, INC><<<nBlocks,nThreads>>>(x, a, b, c, d, length, numTrids, stride, subBatchSize, subBatchStride);
}
