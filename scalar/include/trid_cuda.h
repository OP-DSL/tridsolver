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

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#ifndef __TRID_CUDA_H
#define __TRID_CUDA_H

//#include "trid_common.h"
//#include <cuda_complex.hpp>
//typedef complex<float> complexf;
//typedef complex<double> complexd;

/* This is just a copy of CUSPARSE enums */
typedef enum{
    TRID_STATUS_SUCCESS=0,
    TRID_STATUS_NOT_INITIALIZED=1,
    TRID_STATUS_ALLOC_FAILED=2,
    TRID_STATUS_INVALID_VALUE=3,
    TRID_STATUS_ARCH_MISMATCH=4,
    TRID_STATUS_MAPPING_ERROR=5,
    TRID_STATUS_EXECUTION_FAILED=6,
    TRID_STATUS_INTERNAL_ERROR=7,
    TRID_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    TRID_STATUS_ZERO_PIVOT=9
} tridStatus_t;

void initTridMultiDimBatchSolve(int ndim, int *dims, int *pads);

//in place of a handle and customizations
int* get_opts();

//tridStatus_t tridSgtsvStridedBatch(int sys_size, const float* a, const float *b, const float *c, float *d, int num_sys, int sys_stride);
//tridStatus_t tridDgtsvStridedBatch(int sys_size, const double* a, const double *b, const double *c, double *d, int num_sys, int sys_stride);
//tridStatus_t tridCgtsvStridedBatch(int sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, int num_sys, int sys_stride);
//tridStatus_t tridZgtsvStridedBatch(int sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, int num_sys, int sys_stride);

//tridStatus_t tridSgtsvStridedBatchInc(int sys_size, const float* a, const float *b, const float *c, float *d, float *u, int num_sys, int sys_stride);
//tridStatus_t tridDgtsvStridedBatchInc(int sys_size, const double* a, const double *b, const double *c, double *d, float *u, int num_sys, int sys_stride);
//tridStatus_t tridCgtsvStridedBatchInc(int sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, complexf *u, int num_sys, int sys_stride);
//tridStatus_t tridZgtsvStridedBatchInc(int sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, complexd *u, int num_sys, int sys_stride);

tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b, const float *c, float *d, float* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync);
tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b, const double *c, double *d, double* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync);
//tridStatus_t tridSmtsvStridedBatch(int ndim, int* sys_size, const float* a, const float *b, const float *c, float *d, int *sys_stride, int solvedim);
//tridStatus_t tridDmtsvStridedBatch(int ndim, int* sys_size, const double* a, const double *b, const double *c, double *d, int *sys_stride, int solvedim);
//tridStatus_t tridCmtsvStridedBatch(int ndim, int* sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, int *sys_stride, int solvedim);
//tridStatus_t tridZmtsvStridedBatch(int ndim, int* sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, int *sys_stride, int solvedim);

tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b, const float *c, float *d, float* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync);
tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b, const double *c, double *d, double* u, int ndim, int solvedim, int *dims, int *pads, int *opts, int sync);
//tridStatus_t tridSmtsvStridedBatchInc(int ndim, int* sys_size, const float* a, const float *b, const float *c, float *d, float *u, int *sys_stride, int solvedim);
//tridStatus_t tridDmtsvStridedBatchInc(int ndim, int* sys_size, const double* a, const double *b, const double *c, double *d, double *u, int *sys_stride, int solvedim);
//tridStatus_t tridCmtsvStridedBatchInc(int ndim, int* sys_size, const complexf* a, const complexf *b, const complexf *c, complexf *d, complexf *u, int *sys_stride, int solvedim);
//tridStatus_t tridZmtsvStridedBatchInc(int ndim, int* sys_size, const complexd* a, const complexd *b, const complexd *c, complexd *d, complexd *u, int *sys_stride, int solvedim);


#endif
