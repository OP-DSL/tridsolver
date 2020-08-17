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

#include "trid_linear_mpi.hpp"
#include "trid_linear_mpi_reg.hpp"
#include "trid_strided_multidim_mpi.hpp"
#include "trid_cuda_mpi_pcr.hpp"

#include "cutil_inline.h"

#include "cuda_timing.h"

#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <cmath>

#include <iostream>

// define MPI_datatype
#if __cplusplus >= 201402L
namespace {
template <typename REAL>
const MPI_Datatype mpi_datatype =
    std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
}
#  define MPI_DATATYPE(REAL) mpi_datatype<REAL>
#else
#  define MPI_DATATYPE(REAL)                                                   \
    (std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT)
#endif

template <typename REAL>
inline void forward_batched(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *a,
                            const int *a_pads, const REAL *b, const int *b_pads,
                            const REAL *c, const int *c_pads, const REAL *d,
                            const int *d_pads, REAL *aa, REAL *cc, REAL *dd,
                            REAL *boundaries, REAL *send_buf_h, const int *dims,
                            int ndim, int solvedim, int start_sys, int bsize,
                            cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    const int batch_offset = start_sys * a_pads[solvedim]; // TODO pads
    trid_linear_forward_reg(
        dimGrid_x, dimBlock_x, a + batch_offset, b + batch_offset,
        c + batch_offset, d + batch_offset, aa + batch_offset,
        cc + batch_offset, dd + batch_offset, boundaries + start_sys * 3 * 2,
        dims[solvedim], a_pads[solvedim], bsize, stream);
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        a, pads, b, pads, c, pads, d, pads, aa, cc, dd, boundaries, ndim,
        solvedim, bsize, dims, start_sys);
  }
#if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
  size_t comm_buf_size   = 3 * 2 * bsize;
  size_t comm_buf_offset = 3 * 2 * start_sys;
  cudaMemcpyAsync(send_buf_h + comm_buf_offset, boundaries + comm_buf_offset,
                  sizeof(REAL) * comm_buf_size, cudaMemcpyDeviceToHost, stream);
#endif
}

template <typename REAL, int INC>
inline void backward_batched(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *aa,
                             const int *a_pads, const REAL *cc,
                             const int *c_pads, const REAL *dd,
                             const int *d_pads, const REAL *boundaries, REAL *d,
                             REAL *u, const int *u_pads, const int *dims,
                             int ndim, int solvedim, int start_sys, int bsize,
                             cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    const int batch_offset  = start_sys * a_pads[solvedim];
    trid_linear_backward_reg<REAL, INC>(
        dimGrid_x, dimBlock_x, aa + batch_offset, cc + batch_offset,
        dd + batch_offset, d + batch_offset, u + batch_offset,
        boundaries + start_sys * 2, dims[solvedim], a_pads[solvedim], bsize,
        stream);
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_backward<REAL, INC>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa, pads, cc, pads, dd, d, pads, u, pads, boundaries, ndim,
            solvedim, bsize, dims, start_sys);
  }
}

template <typename REAL, int INC>
void reduced_and_backward(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *aa,
                          const int *a_pads, const REAL *cc, const int *c_pads,
                          const REAL *dd, const int *d_pads, REAL *boundaries,
                          REAL *d, REAL *u, const int *u_pads,
                          const REAL *recv_buf_h, REAL *recv_buf,
                          const int *dims, int ndim, int solvedim,
                          int mpi_coord, int bidx, int batch_size,
                          int num_batches, int reduced_len_g, int sys_n,
                          cudaStream_t stream) {
  int batch_start = bidx * batch_size;
  int bsize       = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
#if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
  size_t recv_comm_buf_offset = 3 * reduced_len_g * batch_start;
  // copy the results of the reduced systems to the boundaries array
  cudaMemcpyAsync(
      recv_buf + recv_comm_buf_offset, recv_buf_h + recv_comm_buf_offset,
      reduced_len_g * 3 * bsize * sizeof(REAL), cudaMemcpyHostToDevice, stream);
#endif
  // Finish the solve for batch
  BEGIN_PROFILING_CUDA2("pcr_on_reduced",stream);
  int buf_offset       = 3 * reduced_len_g * batch_start;
  int bound_buf_offset = 2 * batch_start;
  pcr_on_reduced_batched<REAL>(recv_buf + buf_offset,
                               boundaries + bound_buf_offset, bsize, mpi_coord,
                               reduced_len_g, stream);
  END_PROFILING_CUDA2("pcr_on_reduced",stream);
  // Perform the backward run of the modified thomas algorithm
  BEGIN_PROFILING_CUDA2("thomas_backward",stream);
  backward_batched<REAL, INC>(dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, dd,
                              d_pads, boundaries, d, u, u_pads, dims, ndim,
                              solvedim, batch_start, bsize, stream);
  END_PROFILING_CUDA2("thomas_backward",stream);
}

template <typename REAL, int INC>
inline void tridMultiDimBatchSolveMPI_interleaved(
    const MpiSolverParams &params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *dd, REAL *boundaries,
    REAL *recv_buf, int sys_n, REAL *send_buf_h = nullptr,
    REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");
  // length of reduced system
  const int reduced_len_l = 2;
  const int reduced_len_g = reduced_len_l * params.num_mpi_procs[solvedim];

  const int batch_size  = std::min(params.mpi_batch_size, sys_n);
  const int num_batches = 1 + (sys_n - 1) / batch_size;
  // Calculate required number of CUDA threads and blocksS
  int blockdimx = 128;
  int blockdimy = 1;
  int dimgrid   = 1 + (batch_size - 1) / blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  std::vector<MPI_Request> requests(num_batches);
  std::vector<cudaStream_t> streams(num_batches);
#ifdef TRID_NCCL
  std::vector<cudaEvent_t> events(num_batches);
  for (int bidx = 0; bidx < num_batches; ++bidx) 
    cudaSafeCall(cudaEventCreateWithFlags(&events[bidx],cudaEventDisableTiming));
#endif
  for (int bidx = 0; bidx < num_batches; ++bidx) 
    cudaStreamCreate(&streams[bidx]);
  END_PROFILING2("host-overhead");
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
    size_t comm_buf_size   = 3 * reduced_len_l * bsize;
    size_t comm_buf_offset = 3 * reduced_len_l * batch_start;
    // Do modified thomas forward pass
    // For the bidx-th batch
    BEGIN_PROFILING_CUDA2("thomas_forward",streams[bidx]);
#ifdef TRID_NCCL
    //TODO: this actually hurts in a system where p2p is enabled between all GPUs
    // but does it help when we need to go through the network?
    if (bidx != 0 ) //for interleaved, forward should wait for completion of previous forward
      cudaSafeCall(cudaStreamWaitEvent(streams[bidx],events[bidx-1],0));
#endif
    forward_batched(dimGrid_x, dimBlock_x, a, a_pads, b, b_pads, c, c_pads, d,
                    d_pads, aa, cc, dd, boundaries, send_buf_h, dims, ndim,
                    solvedim, batch_start, bsize, streams[bidx]);
    END_PROFILING_CUDA2("thomas_forward",streams[bidx]);
    // wait for the previous MPI transaction to finish
    if (bidx != 0) {
      BEGIN_PROFILING2("mpi_wait");
#ifndef TRID_NCCL
      MPI_Status status;
      MPI_Wait(&requests[bidx - 1], &status);
#endif
      END_PROFILING2("mpi_wait");
      // Finish the previous batch
      reduced_and_backward<REAL, INC>(
          dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, dd, d_pads, boundaries,
          d, u, u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
          params.mpi_coords[solvedim], bidx - 1, batch_size, num_batches,
          reduced_len_g, sys_n, streams[bidx - 1]);
    }
#ifdef TRID_NCCL
    cudaSafeCall(cudaEventRecord(events[bidx],streams[bidx]));
#else
    cudaSafeCall(cudaStreamSynchronize(streams[bidx]));
#endif
    BEGIN_PROFILING2("MPI_Iallgather");
    // Send boundaries of the current batch
    size_t recv_comm_buf_offset = 3 * reduced_len_g * batch_start;
#ifdef TRID_CUDA_AWARE_MPI
    // Gather the reduced system to all nodes (using CUDA aware MPI)
    MPI_Iallgather(boundaries + comm_buf_offset, comm_buf_size,
                   MPI_DATATYPE(REAL), recv_buf + recv_comm_buf_offset,
                   comm_buf_size, MPI_DATATYPE(REAL),
                   params.communicators[solvedim], &requests[bidx]);
#elif defined(TRID_NCCL)
    NCCLCHECK(ncclAllGather(boundaries + comm_buf_offset,
                            recv_buf + recv_comm_buf_offset,
                            comm_buf_size*sizeof(REAL), ncclChar,
                            params.ncclComms[solvedim], streams[bidx]));
#else
    // Communicate boundary results
    MPI_Iallgather(send_buf_h + comm_buf_offset, comm_buf_size,
                   MPI_DATATYPE(REAL), recv_buf_h + recv_comm_buf_offset,
                   comm_buf_size, MPI_DATATYPE(REAL),
                   params.communicators[solvedim], &requests[bidx]);
#endif
    END_PROFILING2("MPI_Iallgather");
  } // batches
  BEGIN_PROFILING2("mpi_wait");
  // Need to finish last batch: receive message, do reduced and backward
  // wait for the last MPI transaction to finish
#ifndef TRID_NCCL
  MPI_Status status;
  MPI_Wait(&requests[num_batches - 1], &status);
#endif
  END_PROFILING2("mpi_wait");
  reduced_and_backward<REAL, INC>(
      dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, dd, d_pads, boundaries, d,
      u, u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
      params.mpi_coords[solvedim], num_batches - 1, batch_size, num_batches,
      reduced_len_g, sys_n, streams[num_batches - 1]);
  BEGIN_PROFILING2("host-overhead");
#ifdef TRID_NCCL
  for (int bidx = 0; bidx < num_batches; ++bidx)
    cudaSafeCall(cudaEventDestroy(events[bidx]));
#endif
  for (int bidx = 0; bidx < num_batches; ++bidx)
    cudaStreamDestroy(streams[bidx]);
  END_PROFILING2("host-overhead");
}

template <typename REAL, int INC>
inline void tridMultiDimBatchSolveMPI_simple(
    const MpiSolverParams &params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *dd, REAL *boundaries,
    REAL *recv_buf, int sys_n, REAL *send_buf_h = nullptr,
    REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");

  // length of reduced system
  const int reduced_len_l = 2;
  const int reduced_len_g = reduced_len_l * params.num_mpi_procs[solvedim];

  const int batch_size  = std::min(params.mpi_batch_size, sys_n);
  const int num_batches = 1 + (sys_n - 1) / batch_size;
  // Calculate required number of CUDA threads and blocksS
  int blockdimx = 128;
  int blockdimy = 1;
  int dimgrid   = 1 + (batch_size - 1) / blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  std::vector<MPI_Request> requests(num_batches);
  std::vector<cudaStream_t> streams(num_batches);
  for (int bidx = 0; bidx < num_batches; ++bidx) 
    cudaStreamCreate(&streams[bidx]);
  END_PROFILING2("host-overhead");
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
    // Do modified thomas forward pass
    // For the bidx-th batch
    BEGIN_PROFILING_CUDA2("thomas_forward",streams[bidx]);
    forward_batched(dimGrid_x, dimBlock_x, a, a_pads, b, b_pads, c, c_pads, d,
                    d_pads, aa, cc, dd, boundaries, send_buf_h, dims, ndim, solvedim,
                    batch_start, bsize, streams[bidx]);
    END_PROFILING_CUDA2("thomas_forward",streams[bidx]);
  } // batches
  int ready_batches = 0;
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
#ifndef TRID_NCCL
    while(cudaStreamQuery(streams[bidx]) != cudaSuccess && ready_batches != bidx) {
      int finished, found_finished;
      MPI_Status status;
      // up until bidx all streams communicating
      MPI_Testany(bidx, requests.data(), &finished, &found_finished, &status);
      if(found_finished && finished != MPI_UNDEFINED) {
        ready_batches++;
        reduced_and_backward<REAL, INC>(
            dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, dd, d_pads,
            boundaries, d, u, u_pads, recv_buf_h, recv_buf, dims, ndim,
            solvedim, params.mpi_coords[solvedim], finished, batch_size,
            num_batches, reduced_len_g, sys_n, streams[finished]);
      }
    }
    if(ready_batches == bidx) {
      cudaStreamSynchronize(streams[bidx]);
    }
#endif
    BEGIN_PROFILING2("MPI_Iallgather");
    // Send boundaries of the current batch
    size_t comm_buf_size        = 3 * reduced_len_l * bsize;
    size_t comm_buf_offset      = 3 * reduced_len_l * batch_start;
    size_t recv_comm_buf_offset = 3 * reduced_len_g * batch_start;
#ifdef TRID_CUDA_AWARE_MPI
    // Gather the reduced system to all nodes (using CUDA aware MPI)
    MPI_Iallgather(boundaries + comm_buf_offset, comm_buf_size,
                   MPI_DATATYPE(REAL), recv_buf + recv_comm_buf_offset,
                   comm_buf_size, MPI_DATATYPE(REAL),
                   params.communicators[solvedim], &requests[bidx]);
#elif defined(TRID_NCCL)
    NCCLCHECK(ncclAllGather(boundaries + comm_buf_offset,
                            recv_buf + recv_comm_buf_offset,
                            comm_buf_size*sizeof(REAL), ncclChar,
                            params.ncclComms[solvedim], streams[bidx]));
#else
    // Communicate boundary results
      MPI_Iallgather(send_buf_h + comm_buf_offset, comm_buf_size,
                     MPI_DATATYPE(REAL), recv_buf_h + recv_comm_buf_offset,
                     comm_buf_size, MPI_DATATYPE(REAL),
                     params.communicators[solvedim], &requests[bidx]);
#endif
    END_PROFILING2("MPI_Iallgather");
  } // batches

#ifndef TRID_NCCL
  MPI_Status status;
#endif
  for (/*ready_batches*/; ready_batches < num_batches; ++ready_batches) {
    // wait for a MPI transaction to finish
    BEGIN_PROFILING2("mpi_wait");
    int bidx;
#ifdef TRID_NCCL
    bidx = ready_batches;
#else
    int rc = MPI_Waitany(requests.size(), requests.data(), &bidx, &status);
    assert(rc == MPI_SUCCESS && "error MPI communication failed");
#endif
    END_PROFILING2("mpi_wait");
    reduced_and_backward<REAL, INC>(
        dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, dd, d_pads, boundaries,
        d, u, u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
        params.mpi_coords[solvedim], bidx, batch_size, num_batches,
        reduced_len_g, sys_n, streams[bidx]);
  }
  BEGIN_PROFILING2("host-overhead");
  for (int bidx = 0; bidx < num_batches; ++bidx)
    cudaStreamDestroy(streams[bidx]);
  END_PROFILING2("host-overhead");
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI_allgather(
    const MpiSolverParams &params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *dd, REAL *boundaries,
    REAL *recv_buf, int sys_n, REAL *send_buf_h = nullptr,
    REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");
  // length of reduced system
  const int reduced_len_l = 2;
  const int reduced_len_g = reduced_len_l * params.num_mpi_procs[solvedim];

  // Calculate required number of CUDA threads and blocksS
  int blockdimx = 128;
  int blockdimy = 1;
  int dimgrid   = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  const size_t comm_buf_size = 2 * 3 * sys_n;
  END_PROFILING2("host-overhead");

  // Do modified thomas forward pass
  BEGIN_PROFILING_CUDA2("thomas_forward",0);
  forward_batched(dimGrid_x, dimBlock_x, a, a_pads, b, b_pads, c, c_pads, d,
                  d_pads, aa, cc, dd, boundaries, send_buf_h, dims, ndim,
                  solvedim, 0, sys_n);
  cudaSafeCall(cudaDeviceSynchronize());
  END_PROFILING_CUDA2("thomas_forward",0);

  BEGIN_PROFILING2("mpi_communication");
#ifdef TRID_CUDA_AWARE_MPI
  // Gather the reduced system to all nodes (using CUDA aware MPI)
  MPI_Allgather(boundaries, comm_buf_size, MPI_DATATYPE(REAL), recv_buf,
                comm_buf_size, MPI_DATATYPE(REAL),
                params.communicators[solvedim]);
#elif defined(TRID_NCCL)
  NCCLCHECK(ncclAllGather(boundaries,
                            recv_buf,
                            comm_buf_size*sizeof(REAL), ncclChar,
                            params.ncclComms[solvedim], 0));
#else
  // Communicate boundary results
  MPI_Allgather(send_buf_h, comm_buf_size, MPI_DATATYPE(REAL), recv_buf_h,
                comm_buf_size, MPI_DATATYPE(REAL),
                params.communicators[solvedim]);
  // copy the results of the reduced systems to the beginning of the boundaries
  // array
  cudaMemcpyAsync(recv_buf, recv_buf_h,
                  reduced_len_g * 3 * sys_n * sizeof(REAL),
                  cudaMemcpyHostToDevice);
#endif
  END_PROFILING2("mpi_communication");

  // Solve the reduced system
  BEGIN_PROFILING_CUDA2("pcr_on_reduced",0);
  pcr_on_reduced_batched<REAL>(recv_buf, boundaries, sys_n,
                               params.mpi_coords[solvedim], reduced_len_g);
  END_PROFILING_CUDA2("pcr_on_reduced",0);
  // Do the backward pass to solve for remaining unknowns
  BEGIN_PROFILING_CUDA2("thomas_backward",0);
  backward_batched<REAL, INC>(dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, dd,
                              d_pads, boundaries, d, u, u_pads, dims, ndim,
                              solvedim, 0, sys_n);
  END_PROFILING_CUDA2("thomas_backward",0);
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const int *a_pads, const REAL *b,
                               const int *b_pads, const REAL *c,
                               const int *c_pads, REAL *d, const int *d_pads,
                               REAL *u, const int *u_pads, int ndim,
                               int solvedim, const int *dims) {
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

  // The local length of reduced systems
  const int loc_red_len = 2;

  // Allocate memory used during the solve
  const int local_helper_size = outer_size * eq_stride * local_eq_size;
  REAL *aa, *cc, *dd, *boundaries;
  cudaSafeCall(cudaMalloc(&aa, local_helper_size * sizeof(REAL)));
  cudaSafeCall(cudaMalloc(&cc, local_helper_size * sizeof(REAL)));
  cudaSafeCall(cudaMalloc(&dd, local_helper_size * sizeof(REAL)));
  cudaSafeCall(cudaMalloc(&boundaries, sys_n * 3 * loc_red_len * sizeof(REAL)));

  // Allocate receive buffer for MPI communication of reduced system
  const size_t reduced_len_g = 2 * params.num_mpi_procs[solvedim];
  REAL *mpi_buf;
  cudaSafeCall(cudaMalloc(&mpi_buf, reduced_len_g * 3 * sys_n * sizeof(REAL)));
  REAL *send_buf = nullptr, *receive_buf= nullptr;
#ifndef TRID_CUDA_AWARE_MPI
  const size_t comm_buf_size = 2 * 3 * sys_n;
  // MPI buffers on host              
  cudaSafeCall(cudaMallocHost(&send_buf, comm_buf_size * sizeof(REAL)));
  cudaSafeCall(cudaMallocHost(&receive_buf, comm_buf_size *
                                                params.num_mpi_procs[solvedim] *
                                                sizeof(REAL)));
#endif
#ifdef TRID_NCCL
//Dry-run, first call of this is quite expensive
    int rank;
    MPI_Comm_rank(params.communicators[solvedim], &rank);
    NCCLCHECK(ncclAllGather(mpi_buf + 1 * rank,
                            mpi_buf,
                            sizeof(REAL), ncclChar,
                            params.ncclComms[solvedim], 0));
    cudaSafeCall(cudaDeviceSynchronize());
#endif  
#if PROFILING
  MPI_Barrier(MPI_COMM_WORLD);
  BEGIN_PROFILING("tridMultiDimBatchSolveMPI");
#endif
  switch (params.strategy) {
  case MpiSolverParams::GATHER_SCATTER:
    assert(false && "GATHER_SCATTER is not implemented for CUDA");
    break;
  case MpiSolverParams::ALLGATHER:
    tridMultiDimBatchSolveMPI_allgather<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa, cc, dd, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  case MpiSolverParams::LATENCY_HIDING_INTERLEAVED:
    tridMultiDimBatchSolveMPI_interleaved<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa, cc, dd, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  case MpiSolverParams::LATENCY_HIDING_TWO_STEP:
    tridMultiDimBatchSolveMPI_simple<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa, cc, dd, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  default: assert(false && "Unknown communication strategy");
  }
#if PROFILING
  BEGIN_PROFILING2("barrier");
  cudaSafeCall(cudaPeekAtLastError());
  cudaSafeCall(cudaDeviceSynchronize());
  MPI_Barrier(params.communicators[solvedim]);
  END_PROFILING2("barrier");
  END_PROFILING("tridMultiDimBatchSolveMPI");
#endif
  // Free memory used in solve
  cudaSafeCall(cudaFree(aa));
  cudaSafeCall(cudaFree(cc));
  cudaSafeCall(cudaFree(dd));
  cudaSafeCall(cudaFree(boundaries));
  cudaSafeCall(cudaFree(mpi_buf));
#ifndef TRID_CUDA_AWARE_MPI
  cudaSafeCall(cudaFreeHost(send_buf));
  cudaSafeCall(cudaFreeHost(receive_buf));
#endif
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const REAL *b, const REAL *c, REAL *d, REAL *u,
                               int ndim, int solvedim, const int *dims,
                               const int *pads) {
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
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads) {
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
                                         int ndim, int solvedim,
                                         const int *dims, const int *pads) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const float *a, const float *b,
                                         const float *c, float *d, float *u,
                                         int ndim, int solvedim,
                                         const int *dims, const int *pads) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, pads);
  return TRID_STATUS_SUCCESS;
}

// Same as the above functions, however the different padding for each array can
// be specified.
tridStatus_t tridDmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const double *a, const int *a_pads,
    const double *b, const int *b_pads, const double *c, const int *c_pads,
    double *d, const int *d_pads, double *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const float *a, const int *a_pads,
    const float *b, const int *b_pads, const float *c, const int *c_pads,
    float *d, const int *d_pads, float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const double *a, const int *a_pads,
    const double *b, const int *b_pads, const double *c, const int *c_pads,
    double *d, const int *d_pads, double *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const float *a, const int *a_pads,
    const float *b, const int *b_pads, const float *c, const int *c_pads,
    float *d, const int *d_pads, float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims);
  return TRID_STATUS_SUCCESS;
}

