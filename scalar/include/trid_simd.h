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

#ifndef TRID_SIMD_H_INCLUDED
#define TRID_SIMD_H_INCLUDED

#include <type_traits>

#include "x86intrin.h"
#ifndef __INTEL_COMPILER
#  define __assume_aligned __builtin_assume_aligned
#endif

#ifdef __AVX512F__
#  define SIMD_WIDTH              (64) // Width of SIMD vector unit in bytes
#  define SIMD_REG_D              __m512d
#  define SIMD_REG_S              __m512
#  define SIMD_MASK_D             __mmask8
#  define SIMD_MASK_S             __mmask16
#  define SIMD_LOAD_P_D           _mm512_loadu_pd
#  define SIMD_LOAD_P_S           _mm512_loadu_ps
#  define SIMD_LOAD_P_M_D(src, m) _mm512_maskz_loadu_pd(m, src)
#  define SIMD_LOAD_P_M_S(src, m) _mm512_maskz_loadu_ps(m, src)
#  define SIMD_STORE_P_D          _mm512_storeu_pd
#  define SIMD_STORE_P_S          _mm512_storeu_ps
#  define SIMD_STORE_P_M_D        _mm512_mask_storeu_pd
#  define SIMD_STORE_P_M_S        _mm512_mask_storeu_ps
#  define MASK_FIRST_D            _cvtu32_mask8(0b11111110)
#  define MASK_FIRST_S            _cvtu32_mask16(0b1111111111111110)
#  define CREATE_MASK_D           _cvtu32_mask8
#  define CREATE_MASK_S           _cvtu32_mask16
#  define SIMD_AND_MASK_D         _kand_mask8
#  define SIMD_AND_MASK_S         _kand_mask16
#  define SIMD_SET1_P_D           _mm512_set1_pd
#  define SIMD_SET1_P_S           _mm512_set1_ps
#elif defined(__AVX__)
#  define SIMD_WIDTH       (32) // Width of SIMD vector unit in bytes
#  define SIMD_REG_D       __m256d
#  define SIMD_REG_S       __m256
#  define SIMD_MASK_D      __m256i
#  define SIMD_MASK_S      __m256i
#  define SIMD_LOAD_P_D    _mm256_loadu_pd
#  define SIMD_LOAD_P_S    _mm256_loadu_ps
#  define SIMD_LOAD_P_M_D  _mm256_maskload_pd
#  define SIMD_LOAD_P_M_S  _mm256_maskload_ps
#  define SIMD_STORE_P_D   _mm256_storeu_pd
#  define SIMD_STORE_P_S   _mm256_storeu_ps
#  define SIMD_STORE_P_M_D _mm256_maskstore_pd
#  define SIMD_STORE_P_M_S _mm256_maskstore_ps
#  define MASK_FIRST_D     _mm256_setr_epi64x(1, -1, -1, -1)
#  define MASK_FIRST_S     _mm256_setr_epi32(1, -1, -1, -1, -1, -1, -1, -1)
#  define SIMD_AND_MASK_D  _mm256_and_si256
#  define SIMD_AND_MASK_S  _mm256_and_si256
#  define CREATE_MASK_D(mask)                                                  \
    _mm256_setr_epi64x(mask[0], mask[1], mask[2], mask[3]);
#  define CREATE_MASK_S(mask)                                                  \
    _mm256_setr_epi32(mask[0], mask[1], mask[2], mask[3], mask[4], mask[5],    \
                      mask[6], mask[7]);
#  define SIMD_SET1_P_D           _mm256_set1_pd
#  define SIMD_SET1_P_S           _mm256_set1_ps
#else
#  error "No vector ISA intrinsics are defined. "
#endif

#define ROUND_DOWN(N, step) (((N) / (step)) * step)

namespace {
template <typename REAL> constexpr int simd_vec_l = SIMD_WIDTH / sizeof(REAL);
template <typename REAL>
using simd_reg_t = typename std::conditional_t<std::is_same<REAL, double>::value,
                                          SIMD_REG_D, SIMD_REG_S>;
template <typename REAL>
using simd_mask_t = typename std::conditional_t<std::is_same<REAL, double>::value,
                                          SIMD_MASK_D, SIMD_MASK_S>;

// Reg Creation

template <typename REAL>
simd_reg_t<REAL> simd_set1_p(const REAL &val) {
  if constexpr (std::is_same_v<REAL, double>) {
    return SIMD_SET1_P_D(val);
  } else {
    return SIMD_SET1_P_S(val);
  }
}

// Load operations
template <typename REAL>
simd_reg_t<REAL> simd_load_p(const REAL *__restrict__ src) {
  if constexpr (std::is_same_v<REAL, double>) {
    return SIMD_LOAD_P_D(src);
  } else {
    return SIMD_LOAD_P_S(src);
  }
}

template <typename REAL>
simd_reg_t<REAL> simd_load_p_m(const REAL *__restrict__ src,
                               const simd_mask_t<REAL> &m) {
  if constexpr (std::is_same_v<REAL, double>) {
    return SIMD_LOAD_P_M_D(src, m);
  } else {
    return SIMD_LOAD_P_M_S(src, m);
  }
}

// Store operations
template <typename REAL>
void simd_store_p(REAL *__restrict__ dst, const simd_reg_t<REAL> &src) {
  if constexpr (std::is_same_v<REAL, double>) {
    return SIMD_STORE_P_D(dst, src);
  } else {
    return SIMD_STORE_P_S(dst, src);
  }
}

template <typename REAL>
void simd_store_p_m(REAL *__restrict__ dst, const simd_mask_t<REAL> &m,
                  const simd_reg_t<REAL> &src) {
  if constexpr (std::is_same_v<REAL, double>) {
    return SIMD_STORE_P_M_D(dst, m, src);
  } else {
    return SIMD_STORE_P_M_S(dst, m, src);
  }
}
// Masking functions
template <typename REAL>
simd_mask_t<REAL> simd_mask_and(const simd_mask_t<REAL> &m,
                                const simd_mask_t<REAL> &m2) {
  if constexpr (std::is_same_v<REAL, double>) {
    return SIMD_AND_MASK_D(m, m2);
  } else {         
    return SIMD_AND_MASK_S(m, m2);
  }
}

template <typename REAL>
constexpr simd_mask_t<REAL> mask_first() {
  if constexpr (std::is_same_v<REAL, double>) {
    return MASK_FIRST_D;
  } else {         
    return MASK_FIRST_S;
  }
}

#ifdef __AVX512F__
template <typename REAL> simd_mask_t<REAL> create_endmask(int sys_size) {
  unsigned mask = 0;
  for (int i = 0; i < simd_vec_l<REAL>; ++i) {
    mask += (ROUND_DOWN(sys_size - 1, simd_vec_l<REAL>) + i) < sys_size ? 1 << i
                                                                        : 0;
  }

  if constexpr (std::is_same_v<REAL, double>) {
    return CREATE_MASK_D(mask);
  } else {
    return CREATE_MASK_S(mask);
  }
}
template <typename REAL> simd_mask_t<REAL> create_cmask(int sys_size) {
  unsigned mask = 0;
  for (int i = 0; i < simd_vec_l<REAL>; ++i) {
    mask += (ROUND_DOWN(sys_size - 1, simd_vec_l<REAL>) + i) < sys_size - 1
                ? 1 << i
                : 0;
  }

  if constexpr (std::is_same_v<REAL, double>) {
    return CREATE_MASK_D(mask);
  } else {
    return CREATE_MASK_S(mask);
  }
}
#else
template <typename REAL> simd_mask_t<REAL> create_endmask(int sys_size) {
  int mask[simd_vec_l<REAL>] = {};
  for (int i = 0; i < simd_vec_l<REAL>; ++i) {
    mask[i] =
        (ROUND_DOWN(sys_size - 1, simd_vec_l<REAL>) + i) < sys_size ? -1 : 0;
  }
  if constexpr (std::is_same_v<REAL, double>) {
    return CREATE_MASK_D(mask);
  } else {
    return CREATE_MASK_S(mask);
  }
}
template <typename REAL> simd_mask_t<REAL> create_cmask(int sys_size) {
  int mask[simd_vec_l<REAL>] = {};
  for (int i = 0; i < simd_vec_l<REAL>; ++i) {
    mask[i] = (ROUND_DOWN(sys_size - 1, simd_vec_l<REAL>) + i) < sys_size - 1
                  ? -1
                  : 0;
  }
  if constexpr (std::is_same_v<REAL, double>) {
    return CREATE_MASK_D(mask);
  } else {
    return CREATE_MASK_S(mask);
  }
}
#endif

} // namespace

#endif
