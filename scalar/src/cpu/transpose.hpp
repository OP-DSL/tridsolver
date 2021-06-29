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

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk,
// 2013-2014

#ifndef TRID_TRANSPOSE_H_INCLUDED
#define TRID_TRANSPOSE_H_INCLUDED

#include "trid_simd.h"

#ifdef __AVX512F__
#  if defined(__clang__)
inline void transpose8x8_intrinsic(__m512d reg[8]) {
#  else
inline void transpose8x8_intrinsic(__m512d __restrict__ reg[8]) {
#  endif
  __m512d tmp[8];

  tmp[0] = _mm512_unpacklo_pd(reg[0], reg[1]);
  tmp[1] = _mm512_unpackhi_pd(reg[0], reg[1]);
  tmp[2] = _mm512_unpacklo_pd(reg[2], reg[3]);
  tmp[3] = _mm512_unpackhi_pd(reg[2], reg[3]);
  tmp[4] = _mm512_unpacklo_pd(reg[4], reg[5]);
  tmp[5] = _mm512_unpackhi_pd(reg[4], reg[5]);
  tmp[6] = _mm512_unpacklo_pd(reg[6], reg[7]);
  tmp[7] = _mm512_unpackhi_pd(reg[6], reg[7]);

  // The result of the next pass is not perfect transpose of 2x2 blocks
  // The next pass will fix the order
  reg[0] = _mm512_shuffle_f64x2(tmp[0], tmp[2], _MM_SHUFFLE(2, 0, 2, 0));
  reg[1] = _mm512_shuffle_f64x2(tmp[1], tmp[3], _MM_SHUFFLE(2, 0, 2, 0));
  reg[2] = _mm512_shuffle_f64x2(tmp[0], tmp[2], _MM_SHUFFLE(3, 1, 3, 1));
  reg[3] = _mm512_shuffle_f64x2(tmp[1], tmp[3], _MM_SHUFFLE(3, 1, 3, 1));
  reg[4] = _mm512_shuffle_f64x2(tmp[4], tmp[6], _MM_SHUFFLE(2, 0, 2, 0));
  reg[5] = _mm512_shuffle_f64x2(tmp[5], tmp[7], _MM_SHUFFLE(2, 0, 2, 0));
  reg[6] = _mm512_shuffle_f64x2(tmp[4], tmp[6], _MM_SHUFFLE(3, 1, 3, 1));
  reg[7] = _mm512_shuffle_f64x2(tmp[5], tmp[7], _MM_SHUFFLE(3, 1, 3, 1));

#  pragma unroll
  for (int i = 0; i < 4; ++i) {
    tmp[i] = _mm512_shuffle_f64x2(reg[i], reg[4 + i], _MM_SHUFFLE(2, 0, 2, 0));
    tmp[4 + i] =
        _mm512_shuffle_f64x2(reg[i], reg[4 + i], _MM_SHUFFLE(3, 1, 3, 1));
  }

#  pragma unroll
  for (int i = 0; i < 8; ++i) {
    reg[i] = tmp[i];
  }
}


#  if defined(__clang__)
inline void transpose16x16_intrinsic(__m512 reg[16]) {
#  else
inline void transpose16x16_intrinsic(__m512 __restrict__ reg[16]) {
#  endif
  __m512 tmp[16];
// Transpose 8x8 blocks (block size is 2x2) within 16x16 matrix
// Not a true transpose:
// 1 2 3 4 -> 1 5 2 6
// 5 6 7 8 -> 3 6 4 8
// But we fix it in the next pass
#  pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp[2 * i]     = _mm512_unpacklo_ps(reg[2 * i], reg[2 * i + 1]);
    tmp[2 * i + 1] = _mm512_unpackhi_ps(reg[2 * i], reg[2 * i + 1]);
  }

  // Transpose 4x4 blocks (block size is 4x4) within 16x16 matrix
#  pragma unroll
  for (int i = 0; i < 4; ++i) {
    reg[4 * i + 0] = _mm512_shuffle_ps(tmp[4 * i + 0], tmp[4 * i + 2],
                                       _MM_SHUFFLE(1, 0, 1, 0));
    reg[4 * i + 1] = _mm512_shuffle_ps(tmp[4 * i + 0], tmp[4 * i + 2],
                                       _MM_SHUFFLE(3, 2, 3, 2));
    reg[4 * i + 2] = _mm512_shuffle_ps(tmp[4 * i + 1], tmp[4 * i + 3],
                                       _MM_SHUFFLE(1, 0, 1, 0));
    reg[4 * i + 3] = _mm512_shuffle_ps(tmp[4 * i + 1], tmp[4 * i + 3],
                                       _MM_SHUFFLE(3, 2, 3, 2));
  }

  // Transpose 2x2 blocks (block size is 8x8) within 16x16 matrix
  // Similarly to the first pass the shuffle mess up the transpose but we will
  // fix it in the next pass
#  pragma unroll
  for (int i = 0; i < 2; ++i) {
#  pragma unroll
    for (int j = 0; j < 4; ++j) {
      tmp[8 * i + j] = _mm512_shuffle_f32x4(reg[8 * i + j], reg[8 * i + 4 + j],
                                            _MM_SHUFFLE(2, 0, 2, 0));
      tmp[8 * i + 4 + j] = _mm512_shuffle_f32x4(
          reg[8 * i + j], reg[8 * i + 4 + j], _MM_SHUFFLE(3, 1, 3, 1));
    }
  }

  // Transpose 1 blocks (block size is 16x16) within 16x16 matrix
#  pragma unroll
  for (int i = 0; i < 8; ++i) {
    reg[i] = _mm512_shuffle_f32x4(tmp[i], tmp[8 + i], _MM_SHUFFLE(2, 0, 2, 0));
    reg[8 + i] =
        _mm512_shuffle_f32x4(tmp[i], tmp[8 + i], _MM_SHUFFLE(3, 1, 3, 1));
  }
}

#elif defined(__AVX__)
#  if defined(__clang__)
inline void transpose8x8_intrinsic(__m256 ymm[8]) {
#  else
inline void transpose8x8_intrinsic(__m256 __restrict__ ymm[8]) {
#  endif
  __m256 tmp[8];

  tmp[0] = _mm256_unpacklo_ps(ymm[0], ymm[1]);
  tmp[1] = _mm256_unpackhi_ps(ymm[0], ymm[1]);
  tmp[2] = _mm256_unpacklo_ps(ymm[2], ymm[3]);
  tmp[3] = _mm256_unpackhi_ps(ymm[2], ymm[3]);
  tmp[4] = _mm256_unpacklo_ps(ymm[4], ymm[5]);
  tmp[5] = _mm256_unpackhi_ps(ymm[4], ymm[5]);
  tmp[6] = _mm256_unpacklo_ps(ymm[6], ymm[7]);
  tmp[7] = _mm256_unpackhi_ps(ymm[6], ymm[7]);

  ymm[0] = _mm256_shuffle_ps(tmp[0], tmp[2], _MM_SHUFFLE(1, 0, 1, 0));
  ymm[1] = _mm256_shuffle_ps(tmp[0], tmp[2], _MM_SHUFFLE(3, 2, 3, 2));
  ymm[2] = _mm256_shuffle_ps(tmp[1], tmp[3], _MM_SHUFFLE(1, 0, 1, 0));
  ymm[3] = _mm256_shuffle_ps(tmp[1], tmp[3], _MM_SHUFFLE(3, 2, 3, 2));
  ymm[4] = _mm256_shuffle_ps(tmp[4], tmp[6], _MM_SHUFFLE(1, 0, 1, 0));
  ymm[5] = _mm256_shuffle_ps(tmp[4], tmp[6], _MM_SHUFFLE(3, 2, 3, 2));
  ymm[6] = _mm256_shuffle_ps(tmp[5], tmp[7], _MM_SHUFFLE(1, 0, 1, 0));
  ymm[7] = _mm256_shuffle_ps(tmp[5], tmp[7], _MM_SHUFFLE(3, 2, 3, 2));

  tmp[0] = _mm256_permute2f128_ps(ymm[0], ymm[4], 0x20);
  tmp[1] = _mm256_permute2f128_ps(ymm[1], ymm[5], 0x20);
  tmp[2] = _mm256_permute2f128_ps(ymm[2], ymm[6], 0x20);
  tmp[3] = _mm256_permute2f128_ps(ymm[3], ymm[7], 0x20);
  tmp[4] = _mm256_permute2f128_ps(ymm[0], ymm[4], 0x31);
  tmp[5] = _mm256_permute2f128_ps(ymm[1], ymm[5], 0x31);
  tmp[6] = _mm256_permute2f128_ps(ymm[2], ymm[6], 0x31);
  tmp[7] = _mm256_permute2f128_ps(ymm[3], ymm[7], 0x31);

#  pragma unroll
  for (int i = 0; i < 8; ++i) {
    ymm[i] = tmp[i];
  }
}

#  if defined(__clang__)
inline void transpose4x4_intrinsic(__m256d ymm[4]) {
#  else
inline void transpose4x4_intrinsic(__m256d __restrict__ ymm[4]) {
#  endif
  __m256d tmp[4];
  tmp[0] = _mm256_permute2f128_pd(ymm[0], ymm[2], 0b00100000);
  tmp[1] = _mm256_permute2f128_pd(ymm[1], ymm[3], 0b00100000);
  tmp[2] = _mm256_permute2f128_pd(ymm[2], ymm[0], 0b00010011);
  tmp[3] = _mm256_permute2f128_pd(ymm[3], ymm[1], 0b00010011);

  ymm[0] = _mm256_shuffle_pd(tmp[0], tmp[1], 0b00000000);
  ymm[1] = _mm256_shuffle_pd(tmp[0], tmp[1], 0b00001111);
  ymm[2] = _mm256_shuffle_pd(tmp[2], tmp[3], 0b00000000);
  ymm[3] = _mm256_shuffle_pd(tmp[2], tmp[3], 0b00001111);
}

#endif

#endif // TRID_TRANSPOSE_H_INCLUDED
