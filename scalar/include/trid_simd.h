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

#ifdef __AVX512F__
#  if FPPREC == 0
// Xeon Phi float
#    define FBYTE      4
#    define SIMD_WIDTH (64) // Width of SIMD vector unit in bytes
#    define SIMD_VEC                                                           \
      (SIMD_WIDTH / FBYTE) // Number of 4 byte floats in a SIMD vector unit
#  elif FPPREC == 1
// Xeon Phi double
#    define FBYTE      8
#    define SIMD_WIDTH (64) // Width of SIMD vector unit in bytes
#    define SIMD_VEC                                                           \
      (SIMD_WIDTH / FBYTE) // Number of 4 byte floats in a SIMD vector unit
#  endif
#else
#  if FPPREC == 0
// AVX float
#    define FBYTE      4
#    define SIMD_WIDTH (32) // Width of SIMD vector unit in bytes
#    define SIMD_VEC                                                           \
      (SIMD_WIDTH / FBYTE) // Number of 4 byte floats in a SIMD vector unit
#  elif FPPREC == 1
// AVX double
#    define FBYTE      8
#    define SIMD_WIDTH (32) // Width of SIMD vector unit in bytes
#    define SIMD_VEC                                                           \
      (SIMD_WIDTH / FBYTE) // Number of 8 byte floats in a SIMD vector unit
#  endif
#endif

#include "x86intrin.h"
#ifndef __INTEL_COMPILER
#  define __assume_aligned __builtin_assume_aligned
#endif

#ifdef __INTEL_COMPILER
#include "dvec.h"
#endif

#ifdef __AVX512F__
#  if FPPREC == 0
// Xeon Phi float
#    ifdef __INTEL_COMPILER
#      define VECTOR F32vec16
#    else // if defined(__GNUC__)
#      define VECTOR __m512
#    endif
#    define SIMD_REG    __m512          // Name of Packed Register
#    define SIMD_REGI   __m512i         // Name of Packed integer Register
#    define SIMD_LOAD_P _mm512_loadu_ps // Unaligned load for packed registers
#    define SIMD_STORE_P                                                       \
      _mm512_storeu_ps // Unaligned store for packed registers
#    define SIMD_PACKSTORELO_P _mm512_packstorelo_ps
#    define SIMD_SET1_P        _mm512_set1_ps
#    define SIMD_SET_EPI       _mm512_set_epi32
#    define SIMD_SET1_EPI      _mm512_set1_epi32
#    define SIMD_I32GATHER_P   _mm512_i32gather_ps
#    define SIMD_I32SCATTER_P  _mm512_i32scatter_ps
#    define SIMD_ADD_P         _mm512_add_ps
#    define SIMD_FMADD_P       _mm512_fmadd_ps
#    define SIMD_FNMADD_P      _mm512_fnmadd_ps
#    define SIMD_ADD_EPI       _mm512_add_epi32
#    define SIMD_SUB_P         _mm512_sub_ps
#    define SIMD_SUB_EPI       _mm512_sub_epi32
#    define SIMD_MUL_P         _mm512_mul_ps
#    define SIMD_DIV_P         _mm512_div_ps
#    define SIMD_RCP_P         _mm512_rcp14_ps
#  elif FPPREC == 1
// Xeon Phi double
#    ifdef __INTEL_COMPILER
#      define VECTOR F64vec8
#    else // if defined(__GNUC__)
#      define VECTOR __m512d
#    endif
#    define SIMD_REG    __m512d         // Name of Packed Register
#    define SIMD_REGI   __m512i         // Name of Packed integer Register
#    define SIMD_LOAD_P _mm512_loadu_pd // Unaligned load for packed registers
#    define SIMD_STORE_P                                                       \
      _mm512_storeu_pd // Unaligned store for packed registers
#    define SIMD_PACKSTORELO_P _mm512_packstorelo_pd
#    define SIMD_SET1_P        _mm512_set1_pd
#    define SIMD_SET_EPI       _mm512_set_epi32
#    define SIMD_SET1_EPI      _mm512_set1_epi32
#    define SIMD_I32GATHER_P   _mm512_i32logather_pd
#    define SIMD_I32SCATTER_P  _mm512_i32loscatter_pd
#    define SIMD_ADD_P         _mm512_add_pd
#    define SIMD_FMADD_P       _mm512_fmadd_pd
#    define SIMD_FNMADD_P      _mm512_fnmadd_pd
#    define SIMD_ADD_EPI       _mm512_add_epi32
#    define SIMD_SUB_P         _mm512_sub_pd
#    define SIMD_SUB_EPI       _mm512_sub_epi32
#    define SIMD_MUL_P         _mm512_mul_pd
#    define SIMD_DIV_P         _mm512_div_pd
#    define SIMD_RCP_P         _mm512_rcp14_pd
#  else
#    error "Macro definition FPPREC unrecognized for Xeon/Xeon Phi processors"
#  endif
#elif defined(__AVX__)
#  if FPPREC == 0
// AVX float
#    ifdef __INTEL_COMPILER
#      define VECTOR F32vec8
#    else // if defined(__GNUC__)
#      define VECTOR __m256
#    endif
#    define SIMD_REG    __m256          // Name of Packed Register
#    define SIMD_LOAD_P _mm256_loadu_ps // Unaligned load for packed registers
#    define SIMD_STREAM_P                                                      \
      _mm256_stream_ps // Aligned stream store for packed registers
#    define SIMD_STORE_P                                                       \
      _mm256_storeu_ps                 // Unaligned store for packed registers
#    define SIMD_SET1_P _mm256_set1_ps // Set Packed register
#    define SIMD_ADD_P  _mm256_add_ps
#    define SIMD_SUB_P  _mm256_sub_ps
#    define SIMD_MUL_P  _mm256_mul_ps
#    define SIMD_DIV_P  _mm256_div_ps
#    define SIMD_RCP_P  _mm256_rcp_ps
#  elif FPPREC == 1
// AVX double
#    ifdef __INTEL_COMPILER
#      define VECTOR F64vec4
#    else // if defined(__GNUC__)
#      define VECTOR __m256d
#    endif
#    define SIMD_REG    __m256d         // Name of Packed Register
#    define SIMD_LOAD_P _mm256_loadu_pd // Unaligned load for packed registers
#    define SIMD_STORE_P                                                       \
      _mm256_storeu_pd                 // Unaligned store for packed registers
#    define SIMD_SET1_P _mm256_set1_pd // Set Packed register
#    define SIMD_ADD_P  _mm256_add_pd
#    define SIMD_SUB_P  _mm256_sub_pd
#    define SIMD_MUL_P  _mm256_mul_pd
#    define SIMD_DIV_P  _mm256_div_pd
#  else
#    error "Macro definition FPPREC unrecognized for AVX-based processor"
#  endif
#else
#  error "No vector ISA intrinsics are defined. "
#endif

#endif
