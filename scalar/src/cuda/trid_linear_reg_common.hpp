
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

// Written by Gabor Daniel Balogh, Pazmany Peter Catholic University,
// balogh.gabor.daniel@itk.ppke.hu, 2021

// This file contains templates to ease the use of linear solver with
// register blocking
#ifndef TRID_LINEAR_REG_COMMON_HPP__
#define TRID_LINEAR_REG_COMMON_HPP__

#include "trid_common.h"
#include "cuda_shfl.h"

#include <type_traits>

namespace {
template <typename REAL> constexpr int align      = ALIGN / sizeof(REAL);
template <typename REAL> constexpr int vec_length = 64 / sizeof(REAL);
template <typename REAL> constexpr int pack_len   = 16 / sizeof(REAL);
template <typename REAL>
using vec_t = typename std::conditional_t<std::is_same<REAL, double>::value,
                                          double2, float4>;

template <typename REAL> union vec_line_t {
  vec_t<REAL> vec[4]; // 8/2 for double or 16/4 for float
  REAL f[vec_length<REAL>];
};

template <typename REALPACK>
__device__ void trid_shfl_xor_vec(REALPACK *vec, int offset);
template <> __device__ void trid_shfl_xor_vec(vec_t<float> *vec, int offset) {
  vec->x = trid_shfl_xor(vec->x, offset);
  vec->y = trid_shfl_xor(vec->y, offset);
  vec->z = trid_shfl_xor(vec->z, offset);
  vec->w = trid_shfl_xor(vec->w, offset);
}
template <> __device__ void trid_shfl_xor_vec(vec_t<double> *vec, int offset) {
  vec->x = trid_shfl_xor(vec->x, offset);
  vec->y = trid_shfl_xor(vec->y, offset);
}
} // namespace

// transpose4x4xor() - exchanges data between 4 consecutive threads
template <typename REAL>
inline __device__ void transpose4x4xor(vec_line_t<REAL> *la) {
  vec_t<REAL> tmp1;
  vec_t<REAL> tmp2;

  // Perform a 2-stage butterfly transpose

  // stage 1 (transpose each one of the 2x2 sub-blocks internally)
  if (threadIdx.x & 1) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[2];
  } else {
    tmp1 = (*la).vec[1];
    tmp2 = (*la).vec[3];
  }

  trid_shfl_xor_vec(&tmp1, 1);
  trid_shfl_xor_vec(&tmp2, 1);

  if (threadIdx.x & 1) {
    (*la).vec[0] = tmp1;
    (*la).vec[2] = tmp2;
  } else {
    (*la).vec[1] = tmp1;
    (*la).vec[3] = tmp2;
  }

  // stage 2 (swap off-diagonal 2x2 blocks)
  if (threadIdx.x & 2) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[1];
  } else {
    tmp1 = (*la).vec[2];
    tmp2 = (*la).vec[3];
  }
  trid_shfl_xor_vec(&tmp1, 2);
  trid_shfl_xor_vec(&tmp2, 2);

  if (threadIdx.x & 2) {
    (*la).vec[0] = tmp1;
    (*la).vec[1] = tmp2;
  } else {
    (*la).vec[2] = tmp1;
    (*la).vec[3] = tmp2;
  }
}

// ga - global array
// la - local array
template <typename REAL>
inline __device__ void load_array_reg(const REAL *__restrict__ ga,
                                      vec_line_t<REAL> *la, int n, int woffset,
                                      int sys_pads) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in
  // registers If trow and tcol are taken as an argument, they are not know in
  // compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
  int tcol = threadIdx.x % 4;        // Threads' colum index within a warp

  // Load 4 double2 or float4 values (64bytes) from an X-line
  gind = woffset + (4 * (trow)) * sys_pads + tcol * pack_len<REAL> +
         n; // First index in the X-line; woffset - warp offset in global memory

  for (int i = 0; i < 4; i++) {
    (*la).vec[i] = __ldg(((vec_t<REAL> *)&ga[gind]));
    gind += sys_pads;
  }

  transpose4x4xor(la);
}

// Same as load_array_reg() with the following exception: if sys_pads would
// cause unaligned access the index is rounded down to the its floor value to
// prevent missaligned access.
// ga - global array
// la - local array
template <typename REAL>
inline __device__ void load_array_reg_unaligned(REAL const *__restrict__ ga,
                                                vec_line_t<REAL> *la, int n,
                                                int tid, int sys_pads,
                                                int sys_length, int offset) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in
  // registers If trow and tcol are taken as an argument, they are not know in
  // compile time -> no optimization
  // int trow = (threadIdx.x % 32)/ 4; // Threads' row index within a warp
  int tcol = threadIdx.x % 4; // Threads' colum index within a warp

  // Load 4 double2 or float4 values (64bytes) from an X-line
  gind = (tid / 4) * 4 * sys_pads + n; // Global memory index for threads

  for (int i = 0; i < 4; i++) {
    int gind_floor = ((gind + offset) / align<REAL>)*align<REAL> - offset +
                     tcol * pack_len<REAL>; // Round index to floor
    (*la).vec[i] = __ldg(((vec_t<REAL> *)&ga[gind_floor])); // Get aligned data
    gind += sys_pads; // Stride to the next system
  }

  transpose4x4xor(la);
}

// Store a tile with 32x16 elements into 32 float16 or double8 struct allocated
// in registers. Every 4 consecutive threads cooperate to transpose and store a
// 4 x float4 or 4 x double2 sub-tile.
// ga - global array
// la - local array
template <typename REAL>
inline __device__ void store_array_reg(REAL *__restrict__ ga,
                                       vec_line_t<REAL> *la, int n, int woffset,
                                       int sys_pads) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in
  // registers If trow and tcol are taken as an argument, they are not know in
  // compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp
  int tcol = threadIdx.x % 4;        // Threads' colum index within a warp

  transpose4x4xor(la);

  gind = woffset + (4 * (trow)) * sys_pads + tcol * pack_len<REAL> + n;
  *((vec_t<REAL> *)&ga[gind]) = (*la).vec[0];
  gind += sys_pads;
  *((vec_t<REAL> *)&ga[gind]) = (*la).vec[1];
  gind += sys_pads;
  *((vec_t<REAL> *)&ga[gind]) = (*la).vec[2];
  gind += sys_pads;
  *((vec_t<REAL> *)&ga[gind]) = (*la).vec[3];
}

// Same as store_array_reg() with the following exception: if stride would cause
// unaligned access the index is rounded down to the its floor value to prevent
// missaligned access.
// ga - global array
// la - local array
template <typename REAL>
inline __device__ void
store_array_reg_unaligned(REAL *__restrict__ ga,
                          vec_line_t<REAL> *__restrict__ la, int n, int tid,
                          int sys_pads, int sys_length, int offset) {
  // Global memory index of an element
  int gind;
  // Array indexing can be decided in compile time -> arrays will stay in
  // registers If trow and tcol are taken as an argument, they are not know in
  // compile time -> no optimization
  int tcol = threadIdx.x % 4; // Threads' colum index within a warp

  transpose4x4xor(la);

  // Store 4 float4 or double2 values (64bytes) to an X-line
  gind = (tid / 4) * 4 * sys_pads + n; // Global memory index for threads

  int gind_floor;
  int i;
  for (i = 0; i < 4; i++) {
    gind_floor = ((gind + offset) / align<REAL>)*align<REAL> - offset +
                 tcol * pack_len<REAL>;               // Round index to floor
    *((vec_t<REAL> *)&ga[gind_floor]) = (*la).vec[i]; // Put aligned data
    gind += sys_pads; // Stride to the next system
  }
}

#endif
