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

#ifndef TRID_COMMON_H__
#define TRID_COMMON_H__

#if FPPREC == 0
#  define FP float
#  define F  f
#elif FPPREC == 1
#  define FP double
#  define F
#else
#  error "Macro definition FPPREC unrecognized for CUDA"
#endif

#define WARP_SIZE    32
#define ALIGN        32          // 32 byte alignment is required
#define ALIGN_FLOAT  (ALIGN / 4) // 32 byte/ 4bytes/float = 8
#define ALIGN_DOUBLE (ALIGN / 8) // 32 byte/ 8bytes/float = 4
// Maximal dimension that can be used in the library. Defines static arrays
#define MAXDIM          3
#define CUDA_ALIGN_BYTE 32 // 32 byte alignment is used on CUDA-enabled GPUs

#ifdef __cplusplus
#  define EXTERN_C extern "C"
#else
#  define EXTERN_C
#endif

/* This is just a copy of CUSPARSE enums */
typedef enum {
  TRID_STATUS_SUCCESS                   = 0,
  TRID_STATUS_NOT_INITIALIZED           = 1,
  TRID_STATUS_ALLOC_FAILED              = 2,
  TRID_STATUS_INVALID_VALUE             = 3,
  TRID_STATUS_ARCH_MISMATCH             = 4,
  TRID_STATUS_MAPPING_ERROR             = 5,
  TRID_STATUS_EXECUTION_FAILED          = 6,
  TRID_STATUS_INTERNAL_ERROR            = 7,
  TRID_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  TRID_STATUS_ZERO_PIVOT                = 9
} tridStatus_t;

#endif
