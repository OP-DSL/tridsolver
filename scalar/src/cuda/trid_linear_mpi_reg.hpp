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
// balogh.gabor.daniel@itk.ppke.hu, 2020

// This file contains template wrappers to ease the use of linear solver with
// register blocking
#ifndef TRID_LINEAR_MPI_REG_HPP__
#define TRID_LINEAR_MPI_REG_HPP__

#include "trid_linear_mpi_reg8_double2.hpp"
#include "trid_linear_mpi_reg16_float4.hpp"

//
// Kernel launch wrapper for forward step with register blocking
//
template <typename REAL>
void trid_linear_forward_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *a,
                             const REAL *b, const REAL *c, const REAL *d,
                             REAL *aa, REAL *cc, REAL *dd, REAL *boundaries,
                             int sys_size, int sys_pads, int sys_n, int offset,
                             cudaStream_t stream);

template <>
void trid_linear_forward_reg<double>(dim3 dimGrid_x, dim3 dimBlock_x,
                                     const double *a, const double *b,
                                     const double *c, const double *d,
                                     double *aa, double *cc, double *dd,
                                     double *boundaries, int sys_size,
                                     int sys_pads, int sys_n, int offset,
                                     cudaStream_t stream) {
  const int aligned = (sys_pads % ALIGN_DOUBLE) == 0 && (offset % ALIGN_DOUBLE) == 0;
  if(aligned) {
    trid_linear_forward_double_aligned<<<dimGrid_x, dimBlock_x, 0, stream>>>(
        a, b, c, d, aa, cc, dd, boundaries, sys_size, sys_pads, sys_n, offset);
  } else {
    trid_linear_forward_double_unaligned<<<dimGrid_x, dimBlock_x, 0, stream>>>(
        a, b, c, d, aa, cc, dd, boundaries, sys_size, sys_pads, sys_n, offset);
  }
}
template <>
void trid_linear_forward_reg<float>(dim3 dimGrid_x, dim3 dimBlock_x,
                                    const float *a, const float *b,
                                    const float *c, const float *d, float *aa,
                                    float *cc, float *dd, float *boundaries,
                                    int sys_size, int sys_pads, int sys_n,
                                    int offset, cudaStream_t stream) {
  const int aligned = (sys_pads % ALIGN_FLOAT) == 0 && (offset % ALIGN_FLOAT) == 0;
  if(aligned) {
    trid_linear_forward_float_aligned<<<dimGrid_x, dimBlock_x, 0, stream>>>(
        a, b, c, d, aa, cc, dd, boundaries, sys_size, sys_pads, sys_n, offset);
  } else {
    trid_linear_forward_float_unaligned<<<dimGrid_x, dimBlock_x, 0, stream>>>(
        a, b, c, d, aa, cc, dd, boundaries, sys_size, sys_pads, sys_n, offset);
  }
}

//
// Kernel launch wrapper for backward step with register blocking
//
template <typename REAL, int INC>
void trid_linear_backward_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *aa,
                              const REAL *cc, const REAL *dd, REAL *d, REAL *u,
                              const REAL *boundaries, int sys_size,
                              int sys_pads, int sys_n, int offset,
                              int start_sys, int y_size, int y_pads,
                              cudaStream_t stream);

template <>
void trid_linear_backward_reg<double, 0>(dim3 dimGrid_x, dim3 dimBlock_x,
                                         const double *aa, const double *cc,
                                         const double *dd, double *d, double *u,
                                         const double *boundaries, int sys_size,
                                         int sys_pads, int sys_n, int offset,
                                         int start_sys, int y_size, int y_pads,
                                         cudaStream_t stream) {
  const int aligned = (sys_pads % ALIGN_DOUBLE) == 0 && (offset % ALIGN_DOUBLE) == 0;
  if(aligned) {
    trid_linear_backward_double_aligned<0><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  } else {
    trid_linear_backward_double_unaligned<0><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  }
}
template <>
void trid_linear_backward_reg<double, 1>(dim3 dimGrid_x, dim3 dimBlock_x,
                                         const double *aa, const double *cc,
                                         const double *dd, double *d, double *u,
                                         const double *boundaries, int sys_size,
                                         int sys_pads, int sys_n, int offset,
                                         int start_sys, int y_size, int y_pads,
                                         cudaStream_t stream) {
  const int aligned = (sys_pads % ALIGN_DOUBLE) == 0 && (offset % ALIGN_DOUBLE) == 0;
  if(aligned) {
    trid_linear_backward_double_aligned<1><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  } else {
    trid_linear_backward_double_unaligned<1><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  }
}

template <>
void trid_linear_backward_reg<float, 0>(dim3 dimGrid_x, dim3 dimBlock_x,
                                        const float *aa, const float *cc,
                                        const float *dd, float *d, float *u,
                                        const float *boundaries, int sys_size,
                                        int sys_pads, int sys_n, int offset,
                                        int start_sys, int y_size, int y_pads,
                                        cudaStream_t stream) {
  const int aligned = (sys_pads % ALIGN_FLOAT) == 0 && (offset % ALIGN_FLOAT) == 0;
  if(aligned) {
    trid_linear_backward_float_aligned<0><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  } else {
    trid_linear_backward_float_unaligned<0><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  }
}

template <>
void trid_linear_backward_reg<float, 1>(dim3 dimGrid_x, dim3 dimBlock_x,
                                        const float *aa, const float *cc,
                                        const float *dd, float *d, float *u,
                                        const float *boundaries, int sys_size,
                                        int sys_pads, int sys_n, int offset,
                                        int start_sys, int y_size, int y_pads,
                                        cudaStream_t stream) {
  const int aligned = (sys_pads % ALIGN_FLOAT) == 0 && (offset % ALIGN_FLOAT) == 0;
  if(aligned) {
    trid_linear_backward_float_aligned<1><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  } else {
    trid_linear_backward_float_unaligned<1><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
        start_sys, y_size, y_pads);
  }
}


#endif /* ifndef TRID_LINEAR_MPI_REG_HPP__ */
