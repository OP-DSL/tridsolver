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

#include "cuda/trid_cuda.hpp"
#include "trid_common.h"

//#ifdef __cplusplus
extern "C"
//#endif
//void initTridMultiDimBatchSolve_wrapper(int ndim, int *dims, int *pads, int *cumdims, int *cumpads) {
//  initTridMultiDimBatchSolve(ndim, dims, pads, cumdims, cumpads);
void initTridMultiDimBatchSolve_wrapper(int ndim, int *dims, int *pads) {
  initTridMultiDimBatchSolve(ndim, dims, pads);
}

//#ifdef __cplusplus
extern "C" void tridMultiDimBatchSolve_wrapper_SNA(float* d_a, float* d_b, float* d_c, float* d_d, float* d_u, int ndim, int solvedim, int *dims, int *pads, int *opts, float** d_buffer, int sync) {
  tridMultiDimBatchSolve<float,0>(d_a, d_b, d_c, d_d, d_u, ndim, solvedim, dims, pads, opts, d_buffer, sync);
}

extern "C" void tridMultiDimBatchSolve_wrapper_DNA(double* d_a, double* d_b, double* d_c, double* d_d, double* d_u, int ndim, int solvedim, int *dims, int *pads, int *opts, double** d_buffer, int sync) {
  tridMultiDimBatchSolve<double,0>(d_a, d_b, d_c, d_d, d_u, ndim, solvedim, dims, pads, opts, d_buffer, sync);
}

extern "C" void tridMultiDimBatchSolve_wrapper_SA(float* d_a, float* d_b, float* d_c, float* d_d, float* d_u, int ndim, int solvedim, int *dims, int *pads, int *opts, float** d_buffer, int sync) {
  tridMultiDimBatchSolve<float,1>(d_a, d_b, d_c, d_d, d_u, ndim, solvedim, dims, pads, opts, d_buffer, sync);
}

extern "C" void tridMultiDimBatchSolve_wrapper_DA(double* d_a, double* d_b, double* d_c, double* d_d, double* d_u, int ndim, int solvedim, int *dims, int *pads, int *opts, double** d_buffer, int sync) {
  tridMultiDimBatchSolve<double,1>(d_a, d_b, d_c, d_d, d_u, ndim, solvedim, dims, pads, opts, d_buffer, sync);
}
//#else
//template<typename REAL, int INC>
//void tridMultiDimBatchSolve_wrapper(REAL* d_a, REAL* d_b, REAL* d_c, REAL* d_d, REAL* d_u, int ndim, int solvedim, int *dims, int *pads, int *cumdims, int *cumpads, int *opts, REAL** d_buffer) {
//  tridMultiDimBatchSolve<REAL,INC>(d_a, d_b, d_c, d_d, d_u, ndim, solvedim, dims, pads, cumdims, cumpads, opts, d_buffer);
//  //cutilSafeCall( cudaDeviceSynchronize() );
//}
//#endif


