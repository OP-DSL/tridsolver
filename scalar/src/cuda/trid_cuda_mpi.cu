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

// #include "trid_mpi_cuda.hpp"
#include "tridsolver.h"
#include "trid_mpi_solver_params.hpp"
#include "trid_mpi_common.hpp"

#include "trid_linear_mpi.hpp"
#include "trid_linear_mpi_reg.hpp"
#include "trid_strided_multidim_mpi.hpp"
#include "trid_cuda_mpi_pcr.hpp"
#include "trid_iterative_mpi.hpp"

#include "cutil_inline.h"

#include "cuda_timing.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <type_traits>

#include <iostream>

namespace {

enum class memory_env { HOST, DEVICE };

template <memory_env mem_env> struct mem_buffer {
  size_t size  = 0;       /*<< size of the buffer in bytes */
  char *buffer = nullptr; /*<< pointer to memory in mem_env */

  void free() {
    if (buffer) {
      if (mem_env == memory_env::DEVICE) {
        cudaFree(buffer);
      } else {
        cudaFreeHost(buffer);
      }
      buffer = nullptr;
      size   = 0;
    }
  }

  template <typename REAL> REAL *get_bytes_as(size_t bytes) {
    if (size < bytes) {
      free();
      if (mem_env == memory_env::DEVICE) {
        cudaSafeCall(cudaMalloc(&buffer, bytes));
      } else {
        cudaSafeCall(cudaMallocHost(&buffer, bytes));
      }
      size = bytes;
    }
    return reinterpret_cast<REAL *>(buffer);
  }

  ~mem_buffer() { free(); }
  mem_buffer() noexcept          = default;
  mem_buffer(const mem_buffer &) = delete;
  mem_buffer &operator=(const mem_buffer &) = delete;
  mem_buffer(const mem_buffer &&)           = delete;
  mem_buffer &operator=(mem_buffer &&) = delete;
};

mem_buffer<memory_env::DEVICE> aa_buf, cc_buf, boundaries_buf, mpi_buffer;

#if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
mem_buffer<memory_env::HOST> send_buffer;
#endif
mem_buffer<memory_env::HOST> receive_buffer;
} // namespace

template <typename REAL>
inline void forward_batched(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *a,
                            const int *a_pads, const REAL *b, const int *b_pads,
                            const REAL *c, const int *c_pads, REAL *d,
                            const int *d_pads, REAL *aa, REAL *cc,
                            REAL *boundaries, REAL *send_buf_h, const int *dims,
                            int ndim, int solvedim, int start_sys, int bsize,
                            cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    const int batch_offset = start_sys * a_pads[solvedim]; // TODO pads
    int y_size = 1, y_pads = 1;
    if (ndim > 1) {
      y_size = dims[1];
      y_pads = a_pads[1];
    }
    trid_linear_forward_reg(
        dimGrid_x, dimBlock_x, a + batch_offset, b + batch_offset,
        c + batch_offset, d + batch_offset, aa + batch_offset,
        cc + batch_offset, boundaries + start_sys * 3 * 2, dims[solvedim],
        a_pads[solvedim], bsize, start_sys, y_size, y_pads, stream);
  } else {
    DIM_V k_pads, k_dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      k_pads.v[i] = a_pads[i];
      k_dims.v[i] = dims[i];
    }
    trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x, 0, stream>>>(
        a, k_pads, b, k_pads, c, k_pads, d, k_pads, aa, cc, boundaries, ndim,
        solvedim, bsize, k_dims, start_sys);
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
                             const int *c_pads, const int *d_pads,
                             const REAL *boundaries, REAL *d, REAL *u,
                             const int *u_pads, const int *dims, int ndim,
                             int solvedim, int start_sys, int bsize,
                             cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    const int batch_offset = start_sys * a_pads[solvedim];
    int y_size = 1, y_pads = 1;
    if (ndim > 1) {
      y_size = dims[1];
      y_pads = a_pads[1];
    }
    trid_linear_backward_reg<REAL, INC>(
        dimGrid_x, dimBlock_x, aa + batch_offset, cc + batch_offset,
        d + batch_offset, u + batch_offset, boundaries + start_sys * 2,
        dims[solvedim], a_pads[solvedim], bsize, start_sys, y_size, y_pads,
        stream);
  } else {
    DIM_V k_pads, k_dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      k_pads.v[i] = a_pads[i];
      k_dims.v[i] = dims[i];
    }
    trid_strided_multidim_backward<REAL, INC>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa, k_pads, cc, k_pads, d, k_pads, u, k_pads, boundaries, ndim,
            solvedim, bsize, k_dims, start_sys);
  }
}

template <typename REAL, int INC>
void reduced_and_backward(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *aa,
                          const int *a_pads, const REAL *cc, const int *c_pads,
                          const int *d_pads, REAL *boundaries, REAL *d, REAL *u,
                          const int *u_pads, const REAL *recv_buf_h,
                          REAL *recv_buf, const int *dims, int ndim,
                          int solvedim, int mpi_coord, int bidx, int batch_size,
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
  BEGIN_PROFILING_CUDA2("reduced", stream);
  int buf_offset       = 3 * reduced_len_g * batch_start;
  int bound_buf_offset = 2 * batch_start;
  pcr_on_reduced_batched<REAL>(recv_buf + buf_offset,
                               boundaries + bound_buf_offset, bsize, mpi_coord,
                               reduced_len_g, stream);
  END_PROFILING_CUDA2("reduced", stream);
  // Perform the backward run of the modified thomas algorithm
  BEGIN_PROFILING_CUDA2("thomas_backward", stream);
  backward_batched<REAL, INC>(dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads,
                              d_pads, boundaries, d, u, u_pads, dims, ndim,
                              solvedim, batch_start, bsize, stream);
  END_PROFILING_CUDA2("thomas_backward", stream);
}

template <typename REAL, int INC>
inline void tridMultiDimBatchSolveMPI_interleaved(
    const MpiSolverParams *params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *boundaries, REAL *recv_buf,
    int sys_n, REAL *send_buf_h = nullptr, REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");
  // length of reduced system
  const int reduced_len_l = 2;
  const int reduced_len_g = reduced_len_l * params->num_mpi_procs[solvedim];

  const int batch_size  = std::min(params->mpi_batch_size, sys_n);
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
    cudaSafeCall(
        cudaEventCreateWithFlags(&events[bidx], cudaEventDisableTiming));
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
    BEGIN_PROFILING_CUDA2("thomas_forward", streams[bidx]);
#ifdef TRID_NCCL
    // TODO: this actually hurts in a system where p2p is enabled between all
    // GPUs
    // but does it help when we need to go through the network?
    if (bidx != 0) // for interleaved, forward should wait for completion of
                   // previous forward
      cudaSafeCall(cudaStreamWaitEvent(streams[bidx], events[bidx - 1], 0));
#endif
    forward_batched(dimGrid_x, dimBlock_x, a, a_pads, b, b_pads, c, c_pads, d,
                    d_pads, aa, cc, boundaries, send_buf_h, dims, ndim,
                    solvedim, batch_start, bsize, streams[bidx]);
    END_PROFILING_CUDA2("thomas_forward", streams[bidx]);
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
          dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, d_pads, boundaries, d,
          u, u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
          params->mpi_coords[solvedim], bidx - 1, batch_size, num_batches,
          reduced_len_g, sys_n, streams[bidx - 1]);
    }
#ifdef TRID_NCCL
    cudaSafeCall(cudaEventRecord(events[bidx], streams[bidx]));
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
                   params->communicators[solvedim], &requests[bidx]);
#elif defined(TRID_NCCL)
    NCCLCHECK(ncclAllGather(boundaries + comm_buf_offset,
                            recv_buf + recv_comm_buf_offset,
                            comm_buf_size * sizeof(REAL), ncclChar,
                            params->ncclComms[solvedim], streams[bidx]));
#else
    // Communicate boundary results
    MPI_Iallgather(send_buf_h + comm_buf_offset, comm_buf_size,
                   MPI_DATATYPE(REAL), recv_buf_h + recv_comm_buf_offset,
                   comm_buf_size, MPI_DATATYPE(REAL),
                   params->communicators[solvedim], &requests[bidx]);
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
      dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, d_pads, boundaries, d, u,
      u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
      params->mpi_coords[solvedim], num_batches - 1, batch_size, num_batches,
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
    const MpiSolverParams *params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *boundaries, REAL *recv_buf,
    int sys_n, REAL *send_buf_h = nullptr, REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");

  // length of reduced system
  const int reduced_len_l = 2;
  const int reduced_len_g = reduced_len_l * params->num_mpi_procs[solvedim];

  const int batch_size  = std::min(params->mpi_batch_size, sys_n);
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
    BEGIN_PROFILING_CUDA2("thomas_forward", streams[bidx]);
    forward_batched(dimGrid_x, dimBlock_x, a, a_pads, b, b_pads, c, c_pads, d,
                    d_pads, aa, cc, boundaries, send_buf_h, dims, ndim,
                    solvedim, batch_start, bsize, streams[bidx]);
    END_PROFILING_CUDA2("thomas_forward", streams[bidx]);
  } // batches
  int ready_batches = 0;
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
#ifndef TRID_NCCL
    while (cudaStreamQuery(streams[bidx]) != cudaSuccess &&
           ready_batches != bidx) {
      int finished, found_finished;
      MPI_Status status;
      // up until bidx all streams communicating
      MPI_Testany(bidx, requests.data(), &finished, &found_finished, &status);
      if (found_finished && finished != MPI_UNDEFINED) {
        ready_batches++;
        reduced_and_backward<REAL, INC>(
            dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, d_pads, boundaries,
            d, u, u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
            params->mpi_coords[solvedim], finished, batch_size, num_batches,
            reduced_len_g, sys_n, streams[finished]);
      }
    }
    if (ready_batches == bidx) {
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
                   params->communicators[solvedim], &requests[bidx]);
#elif defined(TRID_NCCL)
    NCCLCHECK(ncclAllGather(boundaries + comm_buf_offset,
                            recv_buf + recv_comm_buf_offset,
                            comm_buf_size * sizeof(REAL), ncclChar,
                            params->ncclComms[solvedim], streams[bidx]));
#else
    // Communicate boundary results
    MPI_Iallgather(send_buf_h + comm_buf_offset, comm_buf_size,
                   MPI_DATATYPE(REAL), recv_buf_h + recv_comm_buf_offset,
                   comm_buf_size, MPI_DATATYPE(REAL),
                   params->communicators[solvedim], &requests[bidx]);
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
        dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads, d_pads, boundaries, d, u,
        u_pads, recv_buf_h, recv_buf, dims, ndim, solvedim,
        params->mpi_coords[solvedim], bidx, batch_size, num_batches,
        reduced_len_g, sys_n, streams[bidx]);
  }
  BEGIN_PROFILING2("host-overhead");
  for (int bidx = 0; bidx < num_batches; ++bidx)
    cudaStreamDestroy(streams[bidx]);
  END_PROFILING2("host-overhead");
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI_allgather(
    const MpiSolverParams *params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *boundaries, REAL *recv_buf,
    int sys_n, REAL *send_buf_h = nullptr, REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");
  // length of reduced system
  const int reduced_len_l = 2;
  const int reduced_len_g = reduced_len_l * params->num_mpi_procs[solvedim];

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
  BEGIN_PROFILING_CUDA2("thomas_forward", 0);
  forward_batched(dimGrid_x, dimBlock_x, a, a_pads, b, b_pads, c, c_pads, d,
                  d_pads, aa, cc, boundaries, send_buf_h, dims, ndim, solvedim,
                  0, sys_n);
  cudaSafeCall(cudaDeviceSynchronize());
  END_PROFILING_CUDA2("thomas_forward", 0);

  BEGIN_PROFILING2("mpi_communication");
#ifdef TRID_CUDA_AWARE_MPI
  // Gather the reduced system to all nodes (using CUDA aware MPI)
  MPI_Allgather(boundaries, comm_buf_size, MPI_DATATYPE(REAL), recv_buf,
                comm_buf_size, MPI_DATATYPE(REAL),
                params->communicators[solvedim]);
#elif defined(TRID_NCCL)
  NCCLCHECK(ncclAllGather(boundaries, recv_buf, comm_buf_size * sizeof(REAL),
                          ncclChar, params->ncclComms[solvedim], 0));
#else
  // Communicate boundary results
  MPI_Allgather(send_buf_h, comm_buf_size, MPI_DATATYPE(REAL), recv_buf_h,
                comm_buf_size, MPI_DATATYPE(REAL),
                params->communicators[solvedim]);
  // copy the results of the reduced systems to the beginning of the boundaries
  // array
  cudaMemcpyAsync(recv_buf, recv_buf_h,
                  reduced_len_g * 3 * sys_n * sizeof(REAL),
                  cudaMemcpyHostToDevice);
#endif
  END_PROFILING2("mpi_communication");

  // Solve the reduced system
  BEGIN_PROFILING_CUDA2("reduced", 0);
  pcr_on_reduced_batched<REAL>(recv_buf, boundaries, sys_n,
                               params->mpi_coords[solvedim], reduced_len_g);
  END_PROFILING_CUDA2("reduced", 0);
  // Do the backward pass to solve for remaining unknowns
  BEGIN_PROFILING_CUDA2("thomas_backward", 0);
  backward_batched<REAL, INC>(dimGrid_x, dimBlock_x, aa, a_pads, cc, c_pads,
                              d_pads, boundaries, d, u, u_pads, dims, ndim,
                              solvedim, 0, sys_n);
  END_PROFILING_CUDA2("thomas_backward", 0);
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI_pcr(
    const MpiSolverParams *params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *boundaries, REAL *recv_buf,
    int sys_n, REAL *send_buf_h = nullptr,
    REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");

  // Calculate required number of CUDA threads and blocksS
  int blockdimx = 128;
  int blockdimy = 1;
  int dimgrid   = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  END_PROFILING2("host-overhead");

  // Do modified thomas forward pass
  BEGIN_PROFILING_CUDA2("forward", 0);
  forward_batched_pass<REAL, true>(
      dimGrid_x, dimBlock_x, params, a, a_pads, b, b_pads, c, c_pads, d, d_pads,
      aa, cc, boundaries, dims, ndim, solvedim, 0, sys_n);
  cudaSafeCall(cudaDeviceSynchronize());
  END_PROFILING_CUDA2("forward", 0);

  // Solve the reduced system
  BEGIN_PROFILING2("reduced");
  iterative_pcr_on_reduced(dimGrid_x, dimBlock_x, params, boundaries, sys_n,
                           solvedim, recv_buf, recv_buf_h, send_buf_h);
  END_PROFILING2("reduced");

  // Do the backward pass to solve for remaining unknowns
  BEGIN_PROFILING_CUDA2("backward", 0);
  backward_batched_pass<REAL, INC, true>(
      dimGrid_x, dimBlock_x, params, aa, a_pads, cc, c_pads, boundaries, d,
      d_pads, u, u_pads, dims, ndim, solvedim, 0, sys_n);
  END_PROFILING_CUDA2("backward", 0);
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI_jacobi(
    const MpiSolverParams *params, const REAL *a, const int *a_pads,
    const REAL *b, const int *b_pads, const REAL *c, const int *c_pads, REAL *d,
    const int *d_pads, REAL *u, const int *u_pads, int ndim, int solvedim,
    const int *dims, REAL *aa, REAL *cc, REAL *boundaries, REAL *recv_buf,
    int sys_n, REAL *send_buf_h = nullptr,
    REAL *recv_buf_h = nullptr) {
  BEGIN_PROFILING2("host-overhead");

  // Calculate required number of CUDA threads and blocksS
  int blockdimx = 128;
  int blockdimy = 1;
  int dimgrid   = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx  = dimgrid % 65536; // can go up to max 65535 on Fermi
  int dimgridy  = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  END_PROFILING2("host-overhead");

  // Do modified thomas forward pass
  BEGIN_PROFILING_CUDA2("forward", 0);
  forward_batched_pass<REAL, true, false>(
      dimGrid_x, dimBlock_x, params, a, a_pads, b, b_pads, c, c_pads, d, d_pads,
      aa, cc, boundaries, dims, ndim, solvedim, 0, sys_n);
  cudaSafeCall(cudaDeviceSynchronize());
  END_PROFILING_CUDA2("forward", 0);

  // Solve the reduced system
  BEGIN_PROFILING2("reduced");
  iterative_jacobi_on_reduced(dimGrid_x, dimBlock_x, params, boundaries, sys_n,
                              solvedim, recv_buf, recv_buf_h, send_buf_h);
  END_PROFILING2("reduced");

  // Do the backward pass to solve for remaining unknowns
  BEGIN_PROFILING_CUDA2("backward", 0);
  backward_batched_pass<REAL, INC, true, false>(
      dimGrid_x, dimBlock_x, params, aa, a_pads, cc, c_pads, boundaries, d,
      d_pads, u, u_pads, dims, ndim, solvedim, 0, sys_n);
  END_PROFILING_CUDA2("backward", 0);
}


template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams *params, const REAL *a,
                               const int *a_pads, const REAL *b,
                               const int *b_pads, const REAL *c,
                               const int *c_pads, REAL *d, const int *d_pads,
                               REAL *u, const int *u_pads, int ndim,
                               int solvedim, const int *dims) {
  assert(solvedim < ndim);
  static_assert(
      (std::is_same<REAL, float>::value || std::is_same<REAL, double>::value),
      "trid_solve_mpi: only double or float values are supported");

  // The size of the equations / our domain
  assert(dims[solvedim] >= 2 &&
         "One of the processes has fewer than 2 equations, this is not "
         "supported\n");
  const int eq_stride =
      std::accumulate(dims, dims + solvedim, 1, std::multiplies<int>());

  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  const int outer_size = std::accumulate(dims + solvedim + 1, dims + ndim, 1,
                                         std::multiplies<int>());

  // The number of systems to solve
  // const int sys_n = eq_stride * outer_size;
  int sys_n = 1;
  if (solvedim == 0) {
    if (ndim == 2) {
      sys_n = dims[1];
    } else if (ndim > 2) {
      sys_n = dims[ndim - 1] * std::accumulate(a_pads + solvedim + 1,
                                               a_pads + ndim - 1, 1,
                                               std::multiplies<int>());
    }
  } else {
    sys_n = eq_stride * outer_size;
  }

  // The local length of reduced systems
  const int loc_red_len = 2;

  // Allocate memory used during the solve
  // const int local_helper_size = outer_size * eq_stride * local_eq_size;
  const int local_helper_size =
      std::accumulate(a_pads, a_pads + ndim, 1, std::multiplies<int>());
  REAL *aa = aa_buf.get_bytes_as<REAL>(local_helper_size * sizeof(REAL)),
       *cc = cc_buf.get_bytes_as<REAL>(local_helper_size * sizeof(REAL)),
       *boundaries = boundaries_buf.get_bytes_as<REAL>(sys_n * 3 * loc_red_len *
                                                       sizeof(REAL));

  // Allocate receive buffer for MPI communication of reduced system
  const size_t reduced_len_g = 2 * params->num_mpi_procs[solvedim];
  REAL *mpi_buf              = nullptr;
  REAL *send_buf = nullptr, *receive_buf = nullptr;
  const size_t comm_buf_size = loc_red_len * 3 * sys_n;
  switch (params->strategy) {
  case MpiSolverParams::LATENCY_HIDING_INTERLEAVED:
  case MpiSolverParams::LATENCY_HIDING_TWO_STEP:
  case MpiSolverParams::GATHER_SCATTER:
  case MpiSolverParams::ALLGATHER:
    mpi_buf =
        mpi_buffer.get_bytes_as<REAL>(reduced_len_g * 3 * sys_n * sizeof(REAL));
#if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
    // MPI buffers on host
    send_buf    = send_buffer.get_bytes_as<REAL>(comm_buf_size * sizeof(REAL));
    receive_buf = receive_buffer.get_bytes_as<REAL>(
        comm_buf_size * params->num_mpi_procs[solvedim] * sizeof(REAL));
#endif
    break;
  case MpiSolverParams::JACOBI:
    mpi_buf = mpi_buffer.get_bytes_as<REAL>(3 * sys_n * sizeof(REAL));
#if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
    // MPI buffers on host
    send_buf = send_buffer.get_bytes_as<REAL>(3 * sys_n * sizeof(REAL));
#endif
    receive_buf = receive_buffer.get_bytes_as<REAL>(3 * sys_n * sizeof(REAL));
    break;
  case MpiSolverParams::PCR:
    mpi_buf = mpi_buffer.get_bytes_as<REAL>(3 * sys_n * sizeof(REAL));
#if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
    // MPI buffers on host
    send_buf = send_buffer.get_bytes_as<REAL>(3 * sys_n * sizeof(REAL));
    receive_buf =
        receive_buffer.get_bytes_as<REAL>(2 * 3 * sys_n * sizeof(REAL));
#endif
    break;
  default: assert(false && "Unknown communication strategy");
  }
#ifdef TRID_NCCL
  // Dry-run, first call of this is quite expensive
  int rank;
  MPI_Comm_rank(params->communicators[solvedim], &rank);
  NCCLCHECK(ncclAllGather(mpi_buf + 1 * rank, mpi_buf, sizeof(REAL), ncclChar,
                          params->ncclComms[solvedim], 0));
  cudaSafeCall(cudaDeviceSynchronize());
#endif
#if PROFILING
  MPI_Barrier(MPI_COMM_WORLD);
  BEGIN_PROFILING("tridMultiDimBatchSolveMPI");
#endif
const size_t offset = ((size_t)d / sizeof(REAL)) % align<REAL>;
  switch (params->strategy) {
  case MpiSolverParams::GATHER_SCATTER:
    assert(false && "GATHER_SCATTER is not implemented for CUDA");
    // break; Release mode falls back to ALLGATHER
  case MpiSolverParams::ALLGATHER:
    tridMultiDimBatchSolveMPI_allgather<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa + offset, cc + offset, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  case MpiSolverParams::JACOBI:
    tridMultiDimBatchSolveMPI_jacobi<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa + offset, cc + offset, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  case MpiSolverParams::PCR:
    tridMultiDimBatchSolveMPI_pcr<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa + offset, cc + offset, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  case MpiSolverParams::LATENCY_HIDING_INTERLEAVED:
    tridMultiDimBatchSolveMPI_interleaved<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa + offset, cc + offset, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  case MpiSolverParams::LATENCY_HIDING_TWO_STEP:
    tridMultiDimBatchSolveMPI_simple<REAL, INC>(
        params, a, a_pads, b, b_pads, c, c_pads, d, d_pads, u, u_pads, ndim,
        solvedim, dims, aa + offset, cc + offset, boundaries, mpi_buf, sys_n, send_buf,
        receive_buf);
    break;
  default: assert(false && "Unknown communication strategy");
  }
  cudaSafeCall(cudaDeviceSynchronize());
#if PROFILING
  BEGIN_PROFILING2("barrier");
  cudaSafeCall(cudaPeekAtLastError());
  cudaSafeCall(cudaDeviceSynchronize());
  MPI_Barrier(params->communicators[solvedim]);
  END_PROFILING2("barrier");
  END_PROFILING("tridMultiDimBatchSolveMPI");
#endif
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams *params, const REAL *a,
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
tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b,
                                   const double *c, double *d, double *u,
                                   int ndim, int solvedim, const int *dims,
                                   const int *pads, const TridParams *ctx) {
  tridMultiDimBatchSolveMPI<double, 0>((MpiSolverParams *)ctx->mpi_params, a, b,
                                       c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b,
                                   const float *c, float *d, float *u, int ndim,
                                   int solvedim, const int *dims,
                                   const int *pads, const TridParams *ctx) {
  tridMultiDimBatchSolveMPI<float, 0>((MpiSolverParams *)ctx->mpi_params, a, b,
                                      c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must
// be stored in arrays of size 'dims' with 'ndim' dimensions. The 'pads' array
// specifies any padding used in the arrays (the total length of each dimension
// including padding).
//
// 'u' is incremented with the results.
tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads, const TridParams *ctx) {
  tridMultiDimBatchSolveMPI<double, 1>((MpiSolverParams *)ctx->mpi_params, a, b,
                                       c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads, const TridParams *ctx) {
  tridMultiDimBatchSolveMPI<float, 1>((MpiSolverParams *)ctx->mpi_params, a, b,
                                      c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
