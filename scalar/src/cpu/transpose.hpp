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

#ifndef __TRANSPOSE_H
#define __TRANSPOSE_H

#include "trid_simd.h"

#ifdef __AVX__
// void transpose8x8_intrinsic(__m256 *ymm ) {
inline void transpose8x8_intrinsic(__m256 __restrict__ ymm[8]) {
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

  ymm[0] = tmp[0];
  ymm[1] = tmp[1];
  ymm[2] = tmp[2];
  ymm[3] = tmp[3];
  ymm[4] = tmp[4];
  ymm[5] = tmp[5];
  ymm[6] = tmp[6];
  ymm[7] = tmp[7];
}

// void transpose4x4_intrinsic(__m256d *ymm ) {
inline void transpose4x4_intrinsic(__m256d __restrict__ ymm[4]) {
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

#else
#  ifdef __MIC__
__attribute__((target(mic)))
// void transpose16x16_intrinsic( __m512 *zmm ) {
inline void
transpose16x16_intrinsic(__m512 __restrict__ zmm[16]) {
  __m512 tmp[16];

// Transpose 2x2 blocks (block size is 8x8) within 16x16 matrix
#    pragma unroll(16 / 2)
  for (int i = 0; i < 16 / 2; ++i) {
    tmp[i] = _mm512_mask_permute4f128_ps(zmm[i], 0b1111111100000000,
                                         zmm[(16 / 2) + i], _MM_PERM_BACD);
    tmp[(16 / 2) + i] = _mm512_mask_permute4f128_ps(
        zmm[(16 / 2) + i], 0b0000000011111111, zmm[i], _MM_PERM_BADC);
  }

// Transpose 2x2 blocks (block size is 8x8) within 16x16 matrix
// tmp[0 ] = _mm512_mask_permute4f128_ps(zmm[0 ], 0b1111111100000000, zmm[8 ],
// _MM_PERM_BACD); tmp[1 ] = _mm512_mask_permute4f128_ps(zmm[1 ],
// 0b1111111100000000, zmm[9 ], _MM_PERM_BACD); tmp[2 ] =
// _mm512_mask_permute4f128_ps(zmm[2 ], 0b1111111100000000, zmm[10],
// _MM_PERM_BACD); tmp[3 ] = _mm512_mask_permute4f128_ps(zmm[3 ],
// 0b1111111100000000, zmm[11], _MM_PERM_BACD); tmp[4 ] =
// _mm512_mask_permute4f128_ps(zmm[4 ], 0b1111111100000000, zmm[12],
// _MM_PERM_BACD); tmp[5 ] = _mm512_mask_permute4f128_ps(zmm[5 ],
// 0b1111111100000000, zmm[13], _MM_PERM_BACD); tmp[6 ] =
// _mm512_mask_permute4f128_ps(zmm[6 ], 0b1111111100000000, zmm[14],
// _MM_PERM_BACD); tmp[7 ] = _mm512_mask_permute4f128_ps(zmm[7 ],
// 0b1111111100000000, zmm[15], _MM_PERM_BACD);
//
// tmp[8 ] = _mm512_mask_permute4f128_ps(zmm[8 ], 0b0000000011111111, zmm[0 ],
// _MM_PERM_BADC); tmp[9 ] = _mm512_mask_permute4f128_ps(zmm[9 ],
// 0b0000000011111111, zmm[1 ], _MM_PERM_BADC); tmp[10] =
// _mm512_mask_permute4f128_ps(zmm[10], 0b0000000011111111, zmm[2 ],
// _MM_PERM_BADC); tmp[11] = _mm512_mask_permute4f128_ps(zmm[11],
// 0b0000000011111111, zmm[3 ], _MM_PERM_BADC); tmp[12] =
// _mm512_mask_permute4f128_ps(zmm[12], 0b0000000011111111, zmm[4 ],
// _MM_PERM_BADC); tmp[13] = _mm512_mask_permute4f128_ps(zmm[13],
// 0b0000000011111111, zmm[5 ], _MM_PERM_BADC); tmp[14] =
// _mm512_mask_permute4f128_ps(zmm[14], 0b0000000011111111, zmm[6 ],
// _MM_PERM_BADC); tmp[15] = _mm512_mask_permute4f128_ps(zmm[15],
// 0b0000000011111111, zmm[7 ], _MM_PERM_BADC);

// Transpose 4x4 blocks (block size is 4x4) within 16x16 matrix
#    pragma unroll(16 / 8)
  for (int j = 0; j < 16 / 8; ++j) {
#    pragma unroll(16 / 4)
    for (int i = 0; i < 16 / 4; ++i) {
      zmm[j * (16 / 2) + i] = _mm512_mask_permute4f128_ps(
          tmp[j * (16 / 2) + i], 0b1111000011110000,
          tmp[j * (16 / 2) + (16 / 4) + i], _MM_PERM_CBAD);
      zmm[j * (16 / 2) + (16 / 4) + i] = _mm512_mask_permute4f128_ps(
          tmp[j * (16 / 2) + (16 / 4) + i], 0b0000111100001111,
          tmp[j * (16 / 2) + +i], _MM_PERM_ADCB);
    }
  }
// Transpose 4x4 blocks (block size is 4x4) within 16x16 matrix
// zmm[0 ] = _mm512_mask_permute4f128_ps(tmp[0 ], 0b1111000011110000, tmp[4 ],
// _MM_PERM_CBAD); zmm[1 ] = _mm512_mask_permute4f128_ps(tmp[1 ],
// 0b1111000011110000, tmp[5 ], _MM_PERM_CBAD); zmm[2 ] =
// _mm512_mask_permute4f128_ps(tmp[2 ], 0b1111000011110000, tmp[6 ],
// _MM_PERM_CBAD); zmm[3 ] = _mm512_mask_permute4f128_ps(tmp[3 ],
// 0b1111000011110000, tmp[7 ], _MM_PERM_CBAD);
//
// zmm[4 ] = _mm512_mask_permute4f128_ps(tmp[4 ], 0b0000111100001111, tmp[0 ],
// _MM_PERM_ADCB); zmm[5 ] = _mm512_mask_permute4f128_ps(tmp[5 ],
// 0b0000111100001111, tmp[1 ], _MM_PERM_ADCB); zmm[6 ] =
// _mm512_mask_permute4f128_ps(tmp[6 ], 0b0000111100001111, tmp[2 ],
// _MM_PERM_ADCB); zmm[7 ] = _mm512_mask_permute4f128_ps(tmp[7 ],
// 0b0000111100001111, tmp[3 ], _MM_PERM_ADCB);
//
// zmm[8 ] = _mm512_mask_permute4f128_ps(tmp[8 ], 0b1111000011110000, tmp[12],
// _MM_PERM_CBAD); zmm[9 ] = _mm512_mask_permute4f128_ps(tmp[9 ],
// 0b1111000011110000, tmp[13], _MM_PERM_CBAD); zmm[10] =
// _mm512_mask_permute4f128_ps(tmp[10], 0b1111000011110000, tmp[14],
// _MM_PERM_CBAD); zmm[11] = _mm512_mask_permute4f128_ps(tmp[11],
// 0b1111000011110000, tmp[15], _MM_PERM_CBAD);
//
// zmm[12] = _mm512_mask_permute4f128_ps(tmp[12], 0b0000111100001111, tmp[8 ],
// _MM_PERM_ADCB); zmm[13] = _mm512_mask_permute4f128_ps(tmp[13],
// 0b0000111100001111, tmp[9 ], _MM_PERM_ADCB); zmm[14] =
// _mm512_mask_permute4f128_ps(tmp[14], 0b0000111100001111, tmp[10],
// _MM_PERM_ADCB); zmm[15] = _mm512_mask_permute4f128_ps(tmp[15],
// 0b0000111100001111, tmp[11], _MM_PERM_ADCB);

// Transpose 8x8 blocks (block size is 2x2) within 16x16 matrix
#    pragma unroll(16 / 4)
  for (int j = 0; j < 16 / 4; ++j) {
#    pragma unroll(16 / 8)
    for (int i = 0; i < 16 / 8; ++i) {
      tmp[j * (16 / 4) + i] = _mm512_mask_swizzle_ps(
          zmm[j * (16 / 4) + i], 0b1100110011001100,
          zmm[j * (16 / 4) + (16 / 8) + i], _MM_SWIZ_REG_BADC);
      tmp[j * (16 / 4) + (16 / 8) + i] = _mm512_mask_swizzle_ps(
          zmm[j * (16 / 4) + (16 / 8) + i], 0b0011001100110011,
          zmm[j * (16 / 4) + +i], _MM_SWIZ_REG_BADC);
    }
  }

// Transpose 8x8 blocks (block size is 2x2) within 16x16 matrix
// tmp[0 ] = _mm512_mask_swizzle_ps(zmm[0 ], 0b1100110011001100, zmm[2 ],
// _MM_SWIZ_REG_BADC); tmp[1 ] = _mm512_mask_swizzle_ps(zmm[1 ],
// 0b1100110011001100, zmm[3 ], _MM_SWIZ_REG_BADC);

// tmp[2 ] = _mm512_mask_swizzle_ps(zmm[2 ], 0b0011001100110011, zmm[0 ],
// _MM_SWIZ_REG_BADC); tmp[3 ] = _mm512_mask_swizzle_ps(zmm[3 ],
// 0b0011001100110011, zmm[1 ], _MM_SWIZ_REG_BADC);

// tmp[4 ] = _mm512_mask_swizzle_ps(zmm[4 ], 0b1100110011001100, zmm[6 ],
// _MM_SWIZ_REG_BADC); tmp[5 ] = _mm512_mask_swizzle_ps(zmm[5 ],
// 0b1100110011001100, zmm[7 ], _MM_SWIZ_REG_BADC);

// tmp[6 ] = _mm512_mask_swizzle_ps(zmm[6 ], 0b0011001100110011, zmm[4 ],
// _MM_SWIZ_REG_BADC); tmp[7 ] = _mm512_mask_swizzle_ps(zmm[7 ],
// 0b0011001100110011, zmm[5 ], _MM_SWIZ_REG_BADC);
//
// tmp[8 ] = _mm512_mask_swizzle_ps(zmm[8 ], 0b1100110011001100, zmm[10],
// _MM_SWIZ_REG_BADC); tmp[9 ] = _mm512_mask_swizzle_ps(zmm[9 ],
// 0b1100110011001100, zmm[11], _MM_SWIZ_REG_BADC);
//
// tmp[10] = _mm512_mask_swizzle_ps(zmm[10], 0b0011001100110011, zmm[8 ],
// _MM_SWIZ_REG_BADC); tmp[11] = _mm512_mask_swizzle_ps(zmm[11],
// 0b0011001100110011, zmm[9 ], _MM_SWIZ_REG_BADC);

// tmp[12] = _mm512_mask_swizzle_ps(zmm[12], 0b1100110011001100, zmm[14],
// _MM_SWIZ_REG_BADC); tmp[13] = _mm512_mask_swizzle_ps(zmm[13],
// 0b1100110011001100, zmm[15], _MM_SWIZ_REG_BADC);

// tmp[14] = _mm512_mask_swizzle_ps(zmm[14], 0b0011001100110011, zmm[12],
// _MM_SWIZ_REG_BADC); tmp[15] = _mm512_mask_swizzle_ps(zmm[15],
// 0b0011001100110011, zmm[13], _MM_SWIZ_REG_BADC);

// Transpose 8x8 blocks (block size is 2x2) within 16x16 matrix
#    pragma unroll(16 / 2)
  for (int j = 0; j < 16 / 2; ++j) {
    zmm[j * (16 / 8) + 0] =
        _mm512_mask_swizzle_ps(tmp[j * (16 / 8) + 0], 0b1010101010101010,
                               tmp[j * (16 / 8) + 1], _MM_SWIZ_REG_CDAB);
    zmm[j * (16 / 8) + 1] =
        _mm512_mask_swizzle_ps(tmp[j * (16 / 8) + 1], 0b0101010101010101,
                               tmp[j * (16 / 8) + 0], _MM_SWIZ_REG_CDAB);
  }

  // Transpose 8x8 blocks (block size is 2x2) within 16x16 matrix
  // zmm[0 ] = _mm512_mask_swizzle_ps(tmp[0 ], 0b1010101010101010, tmp[1 ],
  // _MM_SWIZ_REG_CDAB); zmm[1 ] = _mm512_mask_swizzle_ps(tmp[1 ],
  // 0b0101010101010101, tmp[0 ], _MM_SWIZ_REG_CDAB);
  //
  // zmm[2 ] = _mm512_mask_swizzle_ps(tmp[2 ], 0b1010101010101010, tmp[3 ],
  // _MM_SWIZ_REG_CDAB); zmm[3 ] = _mm512_mask_swizzle_ps(tmp[3 ],
  // 0b0101010101010101, tmp[2 ], _MM_SWIZ_REG_CDAB);
  //
  // zmm[4 ] = _mm512_mask_swizzle_ps(tmp[4 ], 0b1010101010101010, tmp[5 ],
  // _MM_SWIZ_REG_CDAB); zmm[5 ] = _mm512_mask_swizzle_ps(tmp[5 ],
  // 0b0101010101010101, tmp[4 ], _MM_SWIZ_REG_CDAB);
  //
  // zmm[6 ] = _mm512_mask_swizzle_ps(tmp[6 ], 0b1010101010101010, tmp[7 ],
  // _MM_SWIZ_REG_CDAB); zmm[7 ] = _mm512_mask_swizzle_ps(tmp[7 ],
  // 0b0101010101010101, tmp[6 ], _MM_SWIZ_REG_CDAB);
  //
  // zmm[8 ] = _mm512_mask_swizzle_ps(tmp[8 ], 0b1010101010101010, tmp[9 ],
  // _MM_SWIZ_REG_CDAB); zmm[9 ] = _mm512_mask_swizzle_ps(tmp[9 ],
  // 0b0101010101010101, tmp[8 ], _MM_SWIZ_REG_CDAB);
  //
  // zmm[10] = _mm512_mask_swizzle_ps(tmp[10], 0b1010101010101010, tmp[11],
  // _MM_SWIZ_REG_CDAB); zmm[11] = _mm512_mask_swizzle_ps(tmp[11],
  // 0b0101010101010101, tmp[10], _MM_SWIZ_REG_CDAB);
  //
  // zmm[12] = _mm512_mask_swizzle_ps(tmp[12], 0b1010101010101010, tmp[13],
  // _MM_SWIZ_REG_CDAB); zmm[13] = _mm512_mask_swizzle_ps(tmp[13],
  // 0b0101010101010101, tmp[12], _MM_SWIZ_REG_CDAB);
  //
  // zmm[14] = _mm512_mask_swizzle_ps(tmp[14], 0b1010101010101010, tmp[15],
  // _MM_SWIZ_REG_CDAB); zmm[15] = _mm512_mask_swizzle_ps(tmp[15],
  // 0b0101010101010101, tmp[14], _MM_SWIZ_REG_CDAB);
}

__attribute__((target(mic)))
// void transpose8x8_intrinsic( __m512d *zmm ) {
inline void
transpose8x8_intrinsic(__m512d __restrict__ zmm[8]) {
  __m512d tmp[8];

// Transpose 2x2 blocks (block size is 4x4) within 8x8 matrix
#    pragma unroll(8 / 2)
  for (int i = 0; i < 8 / 2; ++i) {
    tmp[+i] = (__m512d)_mm512_mask_alignr_epi32(
        (__m512i)zmm[+i], 0b1111111100000000, (__m512i)zmm[(8 / 2) + i],
        (__m512i)zmm[(8 / 2) + i], 24);
    tmp[(8 / 2) + i] = (__m512d)_mm512_mask_alignr_epi32(
        (__m512i)zmm[(8 / 2) + i], 0b0000000011111111, (__m512i)zmm[i],
        (__m512i)zmm[i], 8);
  }

//// Transpose 2x2 blocks (block size is 4x4) within 8x8 matrix
// tmp[0] = (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[0],
// 0b1111111100000000, (__m512i) zmm[4], (__m512i) zmm[4], 24 ); tmp[1] =
// (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[1], 0b1111111100000000,
// (__m512i) zmm[5], (__m512i) zmm[5], 24 ); tmp[2] =
// (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[2], 0b1111111100000000,
// (__m512i) zmm[6], (__m512i) zmm[6], 24 ); tmp[3] =
// (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[3], 0b1111111100000000,
// (__m512i) zmm[7], (__m512i) zmm[7], 24 );
//
// tmp[4] = (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[4],
// 0b0000000011111111, (__m512i) zmm[0], (__m512i) zmm[0], 8 ); tmp[5] =
// (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[5], 0b0000000011111111,
// (__m512i) zmm[1], (__m512i) zmm[1], 8 ); tmp[6] =
// (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[6], 0b0000000011111111,
// (__m512i) zmm[2], (__m512i) zmm[2], 8 ); tmp[7] =
// (__m512d)_mm512_mask_alignr_epi32( (__m512i) zmm[7], 0b0000000011111111,
// (__m512i) zmm[3], (__m512i) zmm[3], 8 );

// Transpose 4x4 blocks (block size is 2x2) within 8x8 matrix
#    pragma unroll(8 / 4)
  for (int j = 0; j < 8 / 4; ++j) {
#    pragma unroll(8 / 4)
    for (int i = 0; i < 8 / 4; ++i) {
      zmm[j * (8 / 2) + 0 + i] =
          _mm512_mask_swizzle_pd(tmp[j * (8 / 2) + 0 + i], 0b11001100,
                                 tmp[j * (8 / 2) + 2 + i], _MM_SWIZ_REG_BADC);
      zmm[j * (8 / 2) + 2 + i] =
          _mm512_mask_swizzle_pd(tmp[j * (8 / 2) + 2 + i], 0b00110011,
                                 tmp[j * (8 / 2) + 0 + i], _MM_SWIZ_REG_BADC);
    }
  }

//// Transpose 4x4 blocks (block size is 2x2) within 8x8 matrix
// zmm[0 ] = _mm512_mask_swizzle_pd(tmp[0 ], 0b11001100, tmp[2 ],
// _MM_SWIZ_REG_BADC); zmm[1 ] = _mm512_mask_swizzle_pd(tmp[1 ], 0b11001100,
// tmp[3 ], _MM_SWIZ_REG_BADC);
//
// zmm[2 ] = _mm512_mask_swizzle_pd(tmp[2 ], 0b00110011, tmp[0 ],
// _MM_SWIZ_REG_BADC); zmm[3 ] = _mm512_mask_swizzle_pd(tmp[3 ], 0b00110011,
// tmp[1 ], _MM_SWIZ_REG_BADC);
//
// zmm[4 ] = _mm512_mask_swizzle_pd(tmp[4 ], 0b11001100, tmp[6 ],
// _MM_SWIZ_REG_BADC); zmm[5 ] = _mm512_mask_swizzle_pd(tmp[5 ], 0b11001100,
// tmp[7 ], _MM_SWIZ_REG_BADC);
//
// zmm[6 ] = _mm512_mask_swizzle_pd(tmp[6 ], 0b00110011, tmp[4 ],
// _MM_SWIZ_REG_BADC); zmm[7 ] = _mm512_mask_swizzle_pd(tmp[7 ], 0b00110011,
// tmp[5 ], _MM_SWIZ_REG_BADC);

// Transpose 8x8 blocks (block size is 1x1) within 8x8 matrix
#    pragma unroll(8 / 2)
  for (int j = 0; j < 8 / 2; ++j) {
    tmp[j * (8 / 4) + 0] =
        _mm512_mask_swizzle_pd(zmm[j * (8 / 4) + 0], 0b10101010,
                               zmm[j * (8 / 4) + 1], _MM_SWIZ_REG_CDAB);
    tmp[j * (8 / 4) + 1] =
        _mm512_mask_swizzle_pd(zmm[j * (8 / 4) + 1], 0b01010101,
                               zmm[j * (8 / 4) + 0], _MM_SWIZ_REG_CDAB);
  }

  //// Transpose 8x8 blocks (block size is 1x1) within 8x8 matrix
  // tmp[0 ] = _mm512_mask_swizzle_pd(zmm[0 ], 0b10101010, zmm[1 ],
  // _MM_SWIZ_REG_CDAB); tmp[1 ] = _mm512_mask_swizzle_pd(zmm[1 ], 0b01010101,
  // zmm[0 ], _MM_SWIZ_REG_CDAB);

  // tmp[2 ] = _mm512_mask_swizzle_pd(zmm[2 ], 0b10101010, zmm[3 ],
  // _MM_SWIZ_REG_CDAB); tmp[3 ] = _mm512_mask_swizzle_pd(zmm[3 ], 0b01010101,
  // zmm[2 ], _MM_SWIZ_REG_CDAB);

  // tmp[4 ] = _mm512_mask_swizzle_pd(zmm[4 ], 0b10101010, zmm[5 ],
  // _MM_SWIZ_REG_CDAB); tmp[5 ] = _mm512_mask_swizzle_pd(zmm[5 ], 0b01010101,
  // zmm[4 ], _MM_SWIZ_REG_CDAB);

  // tmp[6 ] = _mm512_mask_swizzle_pd(zmm[6 ], 0b10101010, zmm[7 ],
  // _MM_SWIZ_REG_CDAB); tmp[7 ] = _mm512_mask_swizzle_pd(zmm[7 ], 0b01010101,
  // zmm[6 ], _MM_SWIZ_REG_CDAB);

#    pragma unroll(8)
  for (int j = 0; j < 8; ++j) {
    zmm[j] = tmp[j];
  }
  // zmm[0 ] = tmp[0 ];
  // zmm[1 ] = tmp[1 ];
  // zmm[2 ] = tmp[2 ];
  // zmm[3 ] = tmp[3 ];
  // zmm[4 ] = tmp[4 ];
  // zmm[5 ] = tmp[5 ];
  // zmm[6 ] = tmp[6 ];
  // zmm[7 ] = tmp[7 ];
}
#  endif // __MIC__
#endif

#endif // __TRANSPOSE_H
