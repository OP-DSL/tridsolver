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

#ifndef __CUDA_SHFL_H
#define __CUDA_SHFL_H

// CUDA 5.5 did not have certain definitions of the __shfl() intrinsic
#if CUDA_VERSION == 5500
  __forceinline__ __device__ double __shfl_up(double x, uint s) {
    int lo, hi;
    asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
    lo = __shfl_up(lo,s);
    hi = __shfl_up(hi,s);
    asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
    return x;
  }
  
  __forceinline__ __device__ double __shfl_down(double x, uint s) {
    int lo, hi;
    asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
    lo = __shfl_down(lo,s);
    hi = __shfl_down(hi,s);
    asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
    return x;
  }
  
  __forceinline__ __device__ double __shfl_xor(double x, int s) {
    int lo, hi;
    asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
    lo = __shfl_xor(lo,s);
    hi = __shfl_xor(hi,s);
    asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
    return x;
  }
#endif

#endif
