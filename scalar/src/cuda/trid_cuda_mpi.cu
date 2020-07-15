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
// With contributions from:
// Gabor Daniel Balogh, Pazmany Peter Catholic University,
// balogh.gabor.daniel@itk.ppke.hu, 2020
// Toby Flynn, University of Warwick, T.Flynn@warwick.ac.uk, 2020

#include "trid_mpi_cuda.hpp"

#include "trid_linear_mpi_reg8_double2.hpp"
#include "trid_linear_mpi_reg16_float4.hpp"
#include "trid_linear_mpi.hpp"
#include "trid_strided_multidim_mpi.hpp"
#include "trid_cuda_mpi_pcr.hpp"

#include "cutil_inline.h"

#include "timing.h"

#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <cmath>

#include <iostream>

//
// Kernel launch wrapper for forward
//
template <typename REAL>
void trid_linear_forward_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *a,
                             const REAL *b, const REAL *c, const REAL *d,
                             REAL *aa, REAL *cc, REAL *dd, REAL *boundaries,
                             int sys_size, int sys_pads, int sys_n);

template <>
void trid_linear_forward_reg<double>(dim3 dimGrid_x, dim3 dimBlock_x,
                                     const double *a, const double *b,
                                     const double *c, const double *d,
                                     double *aa, double *cc, double *dd,
                                     double *boundaries, int sys_size,
                                     int sys_pads, int sys_n) {
  trid_linear_forward_double<<<dimGrid_x, dimBlock_x>>>(
      a, b, c, d, aa, cc, dd, boundaries, sys_size, sys_pads, sys_n);
}
template <>
void trid_linear_forward_reg<float>(dim3 dimGrid_x, dim3 dimBlock_x,
                                    const float *a, const float *b,
                                    const float *c, const float *d, float *aa,
                                    float *cc, float *dd, float *boundaries,
                                    int sys_size, int sys_pads, int sys_n) {
  trid_linear_forward_float<<<dimGrid_x, dimBlock_x>>>(
      a, b, c, d, aa, cc, dd, boundaries, sys_size, sys_pads, sys_n);
}

//
// Kernel launch wrapper for backward
//
template <typename REAL, int INC>
void trid_linear_backward_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *aa,
                              const REAL *cc, const REAL *dd, REAL *d, REAL *u,
                              const REAL *boundaries, int sys_size,
                              int sys_pads, int sys_n);

template <>
void trid_linear_backward_reg<double, 0>(dim3 dimGrid_x, dim3 dimBlock_x,
                                         const double *aa, const double *cc,
                                         const double *dd, double *d, double *u,
                                         const double *boundaries, int sys_size,
                                         int sys_pads, int sys_n) {
  trid_linear_backward_double<0><<<dimGrid_x, dimBlock_x>>>(
      aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n);
}
template <>
void trid_linear_backward_reg<double, 1>(dim3 dimGrid_x, dim3 dimBlock_x,
                                         const double *aa, const double *cc,
                                         const double *dd, double *d, double *u,
                                         const double *boundaries, int sys_size,
                                         int sys_pads, int sys_n) {
  trid_linear_backward_double<1><<<dimGrid_x, dimBlock_x>>>(
      aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n);
}

template <>
void trid_linear_backward_reg<float, 0>(dim3 dimGrid_x, dim3 dimBlock_x,
                                        const float *aa, const float *cc,
                                        const float *dd, float *d, float *u,
                                        const float *boundaries, int sys_size,
                                        int sys_pads, int sys_n) {
  trid_linear_backward_float<0><<<dimGrid_x, dimBlock_x>>>(
      aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n);
}

template <>
void trid_linear_backward_reg<float, 1>(dim3 dimGrid_x, dim3 dimBlock_x,
                                        const float *aa, const float *cc,
                                        const float *dd, float *d, float *u,
                                        const float *boundaries, int sys_size,
                                        int sys_pads, int sys_n) {
  trid_linear_backward_float<1><<<dimGrid_x, dimBlock_x>>>(
      aa, cc, dd, d, u, boundaries, sys_size, sys_pads, sys_n);
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               int *a_pads, const REAL *b, int *b_pads,
                               const REAL *c, int *c_pads, REAL *d, int *d_pads,
                               REAL *u, int *u_pads, int ndim, int solvedim,
                               int *dims) {
  // TODO paddings!!
  assert(solvedim < ndim);
  assert((
      (std::is_same<REAL, float>::value || std::is_same<REAL, double>::value) &&
      "trid_solve_mpi: only double or float values are supported"));

  // The size of the equations / our domain
  const size_t local_eq_size = dims[solvedim];
  assert(local_eq_size > 2 &&
         "One of the processes has fewer than 2 equations, this is not "
         "supported\n");
  const int eq_stride =
      std::accumulate(dims, dims + solvedim, 1, std::multiplies<int>{});

  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  const int outer_size = std::accumulate(dims + solvedim + 1, dims + ndim, 1,
                                         std::multiplies<int>{});

  // The number of systems to solve
  const int sys_n = eq_stride * outer_size;

  const MPI_Datatype real_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // The local and global lengths of reduced systems
  int reduced_len_g;
  int reduced_len_l;

  reduced_len_g = 2 * params.num_mpi_procs[solvedim];
  reduced_len_l = 2;

  // Allocate memory used during the solve
  const int local_helper_size = outer_size * eq_stride * local_eq_size;
  REAL *aa, *cc, *dd, *boundaries;
  cudaSafeCall(cudaMalloc(&aa, local_helper_size * sizeof(REAL)));
  cudaSafeCall(cudaMalloc(&cc, local_helper_size * sizeof(REAL)));
  cudaSafeCall(cudaMalloc(&dd, local_helper_size * sizeof(REAL)));
  cudaSafeCall(
      cudaMalloc(&boundaries, sys_n * 3 * reduced_len_l * sizeof(REAL)));

  const int batch_size     = std::min(params.mpi_batch_size, sys_n);
  const int num_batches    = 1 + (sys_n - 1) / batch_size;
  const int sys_bound_size = 3 * reduced_len_l;
  // Calculate required number of CUDA threads and blocksS
  int blockdimx = 128;
  int blockdimy = 1;
  int dimgrid   = 1 + (batch_size - 1) / blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  // Allocate receive buffer for MPI allgather of reduced system
  REAL *recv_buf;
  cudaSafeCall(cudaMalloc(&recv_buf, reduced_len_g * 3 * sys_n * sizeof(REAL)));
#ifndef TRID_CUDA_AWARE_MPI
  // MPI buffers on host
  std::vector<REAL> send_buf(batch_size * 3 * reduced_len_l),
      receive_buf(batch_size * 3 * reduced_len_g);
#endif
  MPI_Request request;
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
    // Do modified thomas forward pass
    // For the bidx-th batch
    BEGIN_PROFILING("thomas_forward");
    if (solvedim == 0) {
      int batch_offset = batch_start * local_eq_size;
      trid_linear_forward_reg(dimGrid_x, dimBlock_x, a + batch_offset,
                              b + batch_offset, c + batch_offset,
                              d + batch_offset, aa + batch_offset,
                              cc + batch_offset, dd + batch_offset,
                              boundaries + batch_start * sys_bound_size,
                              local_eq_size, local_eq_size, bsize);
    } else {
      DIM_V pads, dims; // TODO
      for (int i = 0; i < ndim; ++i) {
        pads.v[i] = a_pads[i];
        dims.v[i] = a_pads[i];
      }
      trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x>>>(
          a, pads, b, pads, c, pads, d, pads, aa, cc, dd, boundaries, ndim,
          solvedim, bsize, dims, batch_start);
    }
    cudaSafeCall(cudaPeekAtLastError());
    cudaSafeCall(cudaDeviceSynchronize());
    END_PROFILING("thomas_forward");
    BEGIN_PROFILING("mpi_communication");
    // wait for the previous MPI transaction to finish
    if (bidx != 0) {
      MPI_Status status;
      MPI_Wait(&request, &status);
#ifndef TRID_CUDA_AWARE_MPI
      int bsize                   = batch_size;
      int p_batch_start           = (bidx - 1) * batch_size;
      size_t recv_comm_buf_offset = 3 * reduced_len_g * p_batch_start;
      // copy the results of the reduced systems to the boundaries array
      cudaMemcpy(recv_buf + recv_comm_buf_offset, receive_buf.data(),
                 reduced_len_g * 3 * bsize * sizeof(REAL),
                 cudaMemcpyHostToDevice);
      cudaSafeCall(cudaPeekAtLastError());
      cudaSafeCall(cudaDeviceSynchronize());
#endif
    }
    // Send boundaries of the current batch
    size_t comm_buf_size   = sys_bound_size * bsize;
    size_t comm_buf_offset = sys_bound_size * batch_start;
#ifdef TRID_CUDA_AWARE_MPI
    size_t recv_comm_buf_offset = 3 * reduced_len_g * batch_start;
    // Gather the reduced system to all nodes (using CUDA aware MPI)
    MPI_Iallgather(boundaries + comm_buf_offset, comm_buf_size, real_datatype,
                   recv_buf + recv_comm_buf_offset, comm_buf_size,
                   real_datatype, params.communicators[solvedim], &request);
#else
    cudaMemcpy(send_buf.data(), boundaries + comm_buf_offset,
               sizeof(REAL) * comm_buf_size, cudaMemcpyDeviceToHost);
    // Communicate boundary results
    MPI_Iallgather(send_buf.data(), comm_buf_size, real_datatype,
                   receive_buf.data(), comm_buf_size, real_datatype,
                   params.communicators[solvedim], &request);
#endif
    END_PROFILING("mpi_communication");
    // Finish the previous batch
    if (bidx != 0) {
      BEGIN_PROFILING("pcr_on_reduced");
      int batch_start      = (bidx - 1) * batch_size;
      int bsize            = batch_size;
      int buf_offset       = 3 * reduced_len_g * batch_start;
      int bound_buf_offset = 2 * batch_start;
      thomas_on_reduced_batched<REAL>(
          recv_buf + buf_offset, boundaries + bound_buf_offset, bsize,
          params.num_mpi_procs[solvedim], params.mpi_coords[solvedim],
          reduced_len_g);
      END_PROFILING("pcr_on_reduced");
      // Perform the backward run of the modified thomas algorithm
      BEGIN_PROFILING("thomas_backward");
      if (solvedim == 0) {
        int batch_offset = batch_start * local_eq_size;
        trid_linear_backward_reg<REAL, INC>(
            dimGrid_x, dimBlock_x, aa + batch_offset, cc + batch_offset,
            dd + batch_offset, d + batch_offset, u + batch_offset,
            boundaries + batch_start * 2, local_eq_size, local_eq_size, bsize);
      } else {
        DIM_V pads, dims; // TODO
        for (int i = 0; i < ndim; ++i) {
          pads.v[i] = a_pads[i];
          dims.v[i] = a_pads[i];
        }
        trid_strided_multidim_backward<REAL, INC><<<dimGrid_x, dimBlock_x>>>(
            aa, pads, cc, pads, dd, d, pads, u, pads, boundaries, ndim,
            solvedim, bsize, dims, batch_start);
      }
      cudaSafeCall(cudaPeekAtLastError());
      cudaSafeCall(cudaDeviceSynchronize());
      END_PROFILING("thomas_backward");
    }
  } // batches
  BEGIN_PROFILING("mpi_communication");
  // Need to finish last batch: receive message, do reduced and backward
  // wait for the last MPI transaction to finish
  MPI_Status status;
  MPI_Wait(&request, &status);
  int batch_start             = (num_batches - 1) * batch_size;
  int bsize                   = sys_n - batch_start;
  size_t recv_comm_buf_offset = 3 * reduced_len_g * batch_start;
#ifndef TRID_CUDA_AWARE_MPI
  // copy the results of the reduced systems to the boundaries array
  cudaMemcpy(recv_buf + recv_comm_buf_offset, receive_buf.data(),
             reduced_len_g * 3 * bsize * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaSafeCall(cudaPeekAtLastError());
  cudaSafeCall(cudaDeviceSynchronize());
#endif
  END_PROFILING("mpi_communication");
  BEGIN_PROFILING("pcr_on_reduced");
  // Solve the reduced system
  int bound_buf_offset = 2 * batch_start;
  thomas_on_reduced_batched<REAL>(recv_buf + recv_comm_buf_offset,
                                  boundaries + bound_buf_offset, bsize,
                                  params.num_mpi_procs[solvedim],
                                  params.mpi_coords[solvedim], reduced_len_g);
  END_PROFILING("pcr_on_reduced");
  // Perform the backward run of the modified thomas algorithm
  BEGIN_PROFILING("thomas_backward");
  if (solvedim == 0) {
    int batch_offset = batch_start * local_eq_size;
    trid_linear_backward_reg<REAL, INC>(
        dimGrid_x, dimBlock_x, aa + batch_offset, cc + batch_offset,
        dd + batch_offset, d + batch_offset, u + batch_offset,
        boundaries + batch_start * 2, local_eq_size, local_eq_size, bsize);
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_backward<REAL, INC><<<dimGrid_x, dimBlock_x>>>(
        aa, pads, cc, pads, dd, d, pads, u, pads, boundaries, ndim, solvedim,
        bsize, dims, batch_start);
  }
  cudaSafeCall(cudaPeekAtLastError());
  cudaSafeCall(cudaDeviceSynchronize());
  END_PROFILING("thomas_backward");

  // Free memory used in solve
  cudaSafeCall(cudaFree(aa));
  cudaSafeCall(cudaFree(cc));
  cudaSafeCall(cudaFree(dd));
  cudaSafeCall(cudaFree(boundaries));
  cudaSafeCall(cudaFree(recv_buf));
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const REAL *b, const REAL *c, REAL *d, REAL *u,
                               int ndim, int solvedim, int *dims, int *pads) {
  tridMultiDimBatchSolveMPI<REAL, INC>(params, a, pads, b, pads, c, pads, d,
                                       pads, u, pads, ndim, solvedim, dims);
}

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must
// be stored in arrays of size 'dims' with 'ndim' dimensions. The 'pads' array
// specifies any padding used in the arrays (the total length of each dimension
// including padding).
//
// The result is written to 'd'. 'u' is unused.
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, pads);
  return TRID_STATUS_SUCCESS;
}

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must
// be stored in arrays of size 'dims' with 'ndim' dimensions. The 'pads' array
// specifies any padding used in the arrays (the total length of each dimension
// including padding).
//
// 'u' is incremented with the results.
tridStatus_t tridDmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const double *a, const double *b,
                                         const double *c, double *d, double *u,
                                         int ndim, int solvedim, int *dims,
                                         int *pads) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const float *a, const float *b,
                                         const float *c, float *d, float *u,
                                         int ndim, int solvedim, int *dims,
                                         int *pads) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, pads);
  return TRID_STATUS_SUCCESS;
}

// Same as the above functions, however the different padding for each array can
// be specified.
tridStatus_t tridDmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims);
  return TRID_STATUS_SUCCESS;
}

