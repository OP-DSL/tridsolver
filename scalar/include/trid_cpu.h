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

#ifndef TRID_CPU_H__
#define TRID_CPU_H__
#include "trid_common.h"

EXTERN_C
void trid_scalarS(const float *a, const float *b, const float *c, float *d,
                  float *u, int N, int stride);
EXTERN_C
void trid_x_transposeS(const float *a, const float *b, const float *c, float *d,
                       float *u, int sys_size, int sys_pad, int stride);
EXTERN_C
void trid_scalar_vecS(const float *a, const float *b, const float *c, float *d,
                      float *u, int N, int stride);
EXTERN_C
void trid_scalar_vecSInc(const float *a, const float *b, const float *c,
                         float *d, float *u, int N, int stride);
EXTERN_C
void trid_scalarD(const double *a, const double *b, const double *c, double *d,
                  double *u, int N, int stride);
EXTERN_C
void trid_x_transposeD(const double *a, const double *b, const double *c,
                       double *d, double *u, int sys_size, int sys_pad,
                       int stride);
EXTERN_C
void trid_scalar_vecD(const double *a, const double *b, const double *c,
                      double *d, double *u, int N, int stride);
EXTERN_C
void trid_scalar_vecDInc(const double *a, const double *b, const double *c,
                         double *d, double *u, int N, int stride);

#endif
