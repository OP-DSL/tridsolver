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

#ifndef __TRID_CPU_H
#define __TRID_CPU_H

#ifdef __cplusplus
   #define EXTERN_C  extern "C"
#else
   #define EXTERN_C
#endif


/* This is just a copy of CUSPARSE enums */
typedef enum {
  TRID_STATUS_SUCCESS = 0,
  TRID_STATUS_NOT_INITIALIZED = 1,
  TRID_STATUS_ALLOC_FAILED = 2,
  TRID_STATUS_INVALID_VALUE = 3,
  TRID_STATUS_ARCH_MISMATCH = 4,
  TRID_STATUS_MAPPING_ERROR = 5,
  TRID_STATUS_EXECUTION_FAILED = 6,
  TRID_STATUS_INTERNAL_ERROR = 7,
  TRID_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  TRID_STATUS_ZERO_PIVOT = 9
} tridStatus_t;

//
// Solve a batch of linear equation systems along a specified axis.
//
// `a`, `b`, `c`, `d` are the parameters of the systems. These must be arrays
// with shape `dims`, the number of dimensions is `ndims`. These are padded
// along each axis with pads of size `pads`.
//
// Solution will be along the axis `solvedim`, i.e. values along that axis will
// form a system of equations to be solved. Separate systems will be solved
// independently. The result is written to `d`.
//
// `u` is unused.
EXTERN_C
tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b,
                                   const float *c, float *d, float *u, int ndim,
                                   int solvedim, int *dims, int *pads);
EXTERN_C
tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b,
                                   const double *c, double *d, double *u,
                                   int ndim, int solvedim, int *dims,
                                   int *pads);

//
// Solve a batch of linear equation systems along a specified axis.
//
// `a`, `b`, `c`, `d` are the parameters of the systems, `u` is the array to be
// incremented with the results. These must be arrays with shape `dims`, the
// number of dimensions is `ndims`. These are padded along each axis with pads
// of size `pads`.
//
// Solution will be along the axis `solvedim`, i.e. values along that axis will
// form a system of equations to be solved. Separate systems will be solved
// independently. `u` is incremented with the results.
EXTERN_C
tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads);
EXTERN_C
tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads);

EXTERN_C
void trid_scalarS(float *a, float *b, float *c, float *d, float *u, int N,
                  int stride);
EXTERN_C
void trid_x_transposeS(float *a, float *b, float *c, float *d, float *u,
                       int sys_size, int sys_pad, int stride);
EXTERN_C
void trid_scalar_vecS(float *a, float *b, float *c, float *d, float *u, int N,
                      int stride);
EXTERN_C
void trid_scalar_vecSInc(float *a, float *b, float *c, float *d, float *u,
                         int N, int stride);
EXTERN_C
void trid_scalarD(double *a, double *b, double *c, double *d, double *u, int N,
                  int stride);
EXTERN_C
void trid_x_transposeD(double *a, double *b, double *c, double *d, double *u,
                       int sys_size, int sys_pad, int stride);
EXTERN_C
void trid_scalar_vecD(double *a, double *b, double *c, double *d, double *u,
                      int N, int stride);
EXTERN_C
void trid_scalar_vecDInc(double *a, double *b, double *c, double *d, double *u,
                         int N, int stride);

#endif
