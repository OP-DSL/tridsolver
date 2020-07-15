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

// Written by Toby Flynn, University of Warwick, T.Flynn@warwick.ac.uk, 2020

#include "trid_mpi_cpu.h"

#include "trid_common.h"
#include "trid_mpi_cpu.hpp"
#include "trid_simd.h"
#include "math.h"
#include "omp.h"

#include <type_traits>
#include <sys/time.h>

#include "timing.h"

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

#define Z_BATCH 56

template <typename REAL>
inline void copy_boundaries_strided(const REAL *aa, const REAL *cc,
                                    const REAL *dd, REAL *sndbuf,
                                    const int *dims, const int *pads,
                                    int solvedim, int start_sys, int end_sys) {
  int result_stride = dims[solvedim] - 1;
  for (int i = 0; i < solvedim; ++i) {
    result_stride *= pads[i];
  }
  int start_pad = pads[0];
  if (solvedim == 1) start_pad *= pads[1];

#pragma omp parallel for
  for (int id = start_sys; id < end_sys; id++) {
    int start;
    if (solvedim != 0) {
      start = (id / dims[0]) * start_pad + (id % dims[0]);
    } else {
      start = id * pads[0];
    }
    int end             = start + result_stride;
    int buf_ind         = id * 6;
    sndbuf[buf_ind]     = aa[start];
    sndbuf[buf_ind + 1] = aa[end];
    sndbuf[buf_ind + 2] = cc[start];
    sndbuf[buf_ind + 3] = cc[end];
    sndbuf[buf_ind + 4] = dd[start];
    sndbuf[buf_ind + 5] = dd[end];
  }
}

template <typename REAL>
inline void forward_batched(const REAL *a, const REAL *b, const REAL *c, const REAL *d,
                    REAL *aa, REAL *cc, REAL *dd, REAL *sndbuf, const int *dims,
                    const int *pads, int ndim, int solvedim, 
                    int start_sys, int end_sys) {
  int n_sys = end_sys - start_sys;
  if (solvedim == 0) {
/*********************
 *
 * X Dimension Solve
 *
 *********************/

// Do modified thomas forward pass
#pragma omp parallel for
    for (int id = start_sys; id < end_sys; id++) {
      int ind = id * pads[0];
      thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &aa[ind],
                           &cc[ind], &dd[ind], dims[0], 1);
    }
  } else if (solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/

    // Check if 2D solve
    if (ndim == 2) {
      // Do modified thomas forward pass
      int batch_size =
          std::max(1, ROUND_DOWN(n_sys / omp_get_max_threads(), SIMD_VEC));
#pragma omp parallel for
      for (int ind = start_sys; ind < start_sys + ROUND_DOWN(n_sys, batch_size);
           ind += batch_size) {
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                       &aa[ind], &cc[ind], &dd[ind], dims[1],
                                       pads[0], batch_size);
      }
      // Do final strip if number of systems isn't a multiple of batch_size
      if (n_sys != ROUND_DOWN(n_sys, batch_size)) {
        int ind    = start_sys + ROUND_DOWN(n_sys, batch_size);
        int length = end_sys - ind;
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                       &aa[ind], &cc[ind], &dd[ind], dims[1],
                                       pads[0], length);
      }
    } else {
      // Assume 3D solve
      // Do modified thomas forward pass
      // Iterate over each z 'layer'
      int start_z = start_sys / dims[0];
      int end_z = end_sys / dims[0];
#pragma omp parallel for
      for (int z = start_z; z < end_z; z++) {
        int ind = z * pads[0] * pads[1];
        // offset in the first z 'layer'
        int strip_start = z == start_z ? start_sys - start_z * dims[0] : 0;
        int strip_len   = z == end_z ? end_sys - end_z * dims[0] : pads[0];
        strip_len -= strip_start;
        ind += strip_start * pads[0];
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                       &aa[ind], &cc[ind], &dd[ind], dims[1],
                                       pads[0], strip_len);
      }
    }
  } else if (solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/

    // Do modified thomas forward pass
    int batch_size =
        std::max(1, ROUND_DOWN(n_sys / omp_get_max_threads(), SIMD_VEC));
#pragma omp parallel for
    for (int ind = start_sys; ind < start_sys + ROUND_DOWN(n_sys, batch_size);
         ind += batch_size) {
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], batch_size);
    }

    // Do final strip if number of systems isn't a multiple of batch_size
    if (n_sys != ROUND_DOWN(n_sys, batch_size)) {
      int ind    = start_sys + ROUND_DOWN(n_sys, batch_size);
      int length = end_sys - ind;
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], length);
    }
  }
  // Pack reduced systems (boundaries of each tridiagonal system)
  copy_boundaries_strided(aa, cc, dd, sndbuf, dims, pads, solvedim, start_sys,
                          end_sys);
}
template <typename REAL>
inline void forward(const REAL *a, const REAL *b, const REAL *c, const REAL *d,
                    REAL *aa, REAL *cc, REAL *dd, REAL *sndbuf, const int *dims,
                    const int *pads, int ndim, int solvedim, 
                    int n_sys) {
  if (solvedim == 0) {
/*********************
 *
 * X Dimension Solve
 *
 *********************/

// Do modified thomas forward pass
#pragma omp parallel for
    for (int id = 0; id < n_sys; id++) {
      int ind = id * pads[0];
      thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &aa[ind],
                           &cc[ind], &dd[ind], dims[0], 1);
    }
  } else if (solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/

    // Check if 2D solve
    if (ndim == 2) {
      // Do modified thomas forward pass
      int batch_size =
          std::max(1, ROUND_DOWN(n_sys / omp_get_max_threads(), SIMD_VEC));
#pragma omp parallel for
      for (int ind = 0; ind < ROUND_DOWN(n_sys, batch_size);
           ind += batch_size) {
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                       &aa[ind], &cc[ind], &dd[ind], dims[1],
                                       pads[0], batch_size);
      }
      // Do final strip if number of systems isn't a multiple of Z_BATCH
      if (n_sys != ROUND_DOWN(n_sys, batch_size)) {
        int ind    = ROUND_DOWN(n_sys, batch_size);
        int length = n_sys - ind;
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                       &aa[ind], &cc[ind], &dd[ind], dims[1],
                                       pads[0], length);
      }
    } else {
// Assume 3D solve
// Do modified thomas forward pass
// Iterate over each z 'layer'
#pragma omp parallel for
      for (int z = 0; z < dims[2]; z++) {
        int ind = z * pads[0] * pads[1];
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                       &aa[ind], &cc[ind], &dd[ind], dims[1],
                                       pads[0], pads[0]);
      }
    }
  } else if (solvedim == 2) {
/*********************
 *
 * Z Dimension Solve
 *
 *********************/

// Do modified thomas forward pass
// Iterate over strips of length Z_BATCH
#pragma omp parallel for
    for (int ind = 0; ind < ROUND_DOWN(dims[1] * pads[0], Z_BATCH);
         ind += Z_BATCH) {
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], Z_BATCH);
    }

    // Do final strip if number of systems isn't a multiple of Z_BATCH
    if (dims[1] * pads[0] != ROUND_DOWN(dims[1] * pads[0], Z_BATCH)) {
      int ind    = ROUND_DOWN(dims[1] * pads[0], Z_BATCH);
      int length = (dims[1] * pads[0]) - ind;
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], length);
    }
  }
  // Pack reduced systems (boundaries of each tridiagonal system)
  copy_boundaries_strided(aa, cc, dd, sndbuf, dims, pads, solvedim, 0,
                          n_sys);
}

template <typename REAL>
inline void solve_reduced_batched(const MpiSolverParams &params,
                                  const REAL *rcvbuf, REAL *aa_r, REAL *cc_r,
                                  REAL *dd_r, REAL *dd, const int *dims,
                                  const int *pads, int solvedim, int sys_len_r,
                                  int start_sys, int end_sys) {
  int n_sys = end_sys - start_sys;
  int result_stride = dims[solvedim] - 1;
  for (int i = 0; i < solvedim; ++i) {
    result_stride *= pads[i];
  }
  int start_pad = pads[0];
  if (solvedim == 1) start_pad *= pads[1];
  // shift buffer 
  rcvbuf += start_sys * params.num_mpi_procs[solvedim] * 6;

// Iterate over each reduced system
#pragma omp parallel for
  for (int id = 0; id < n_sys; id++) {
    // Unpack this reduced system from receive buffer
    for (int p = 0; p < params.num_mpi_procs[solvedim]; p++) {
      int buf_ind                      = p * n_sys * 2 * 3;
      aa_r[id * sys_len_r + p * 2]     = rcvbuf[buf_ind + id * 6];
      aa_r[id * sys_len_r + p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 1];
      cc_r[id * sys_len_r + p * 2]     = rcvbuf[buf_ind + id * 6 + 2];
      cc_r[id * sys_len_r + p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 3];
      dd_r[id * sys_len_r + p * 2]     = rcvbuf[buf_ind + id * 6 + 4];
      dd_r[id * sys_len_r + p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 5];
    }

    // Solve reduced system
    thomas_on_reduced<REAL>(aa_r + id * sys_len_r, cc_r + id * sys_len_r,
                            dd_r + id * sys_len_r, sys_len_r, 1);

    // Write result back
    int p = params.mpi_coords[solvedim];
    int start;
    int g_id = id + start_sys;
    if (solvedim != 0) {
      start = (g_id / dims[0]) * start_pad + (g_id % dims[0]);
    } else {
      start = g_id * pads[0];
    }
    dd[start]                 = dd_r[id * sys_len_r + p * 2];
    dd[start + result_stride] = dd_r[id * sys_len_r + p * 2 + 1];
  }
}

template <typename REAL>
inline void solve_reduced(const MpiSolverParams &params, const REAL *rcvbuf,
                          REAL *aa_r, REAL *cc_r, REAL *dd_r, REAL *dd,
                          const int *dims, const int *pads, int solvedim,
                          int sys_len_r, int n_sys) {
  int result_stride = dims[solvedim] - 1;
  for (int i = 0; i < solvedim; ++i) {
    result_stride *= pads[i];
  }
  int start_pad = pads[0];
  if (solvedim == 1) start_pad *= pads[1];

// Iterate over each reduced system
#pragma omp parallel for
  for (int id = 0; id < n_sys; id++) {
    // Unpack this reduced system from receive buffer
    for (int p = 0; p < params.num_mpi_procs[solvedim]; p++) {
      int buf_ind                      = p * n_sys * 2 * 3;
      aa_r[id * sys_len_r + p * 2]     = rcvbuf[buf_ind + id * 6];
      aa_r[id * sys_len_r + p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 1];
      cc_r[id * sys_len_r + p * 2]     = rcvbuf[buf_ind + id * 6 + 2];
      cc_r[id * sys_len_r + p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 3];
      dd_r[id * sys_len_r + p * 2]     = rcvbuf[buf_ind + id * 6 + 4];
      dd_r[id * sys_len_r + p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 5];
    }

    // Solve reduced system
    thomas_on_reduced<REAL>(aa_r + id * sys_len_r, cc_r + id * sys_len_r,
                            dd_r + id * sys_len_r, sys_len_r, 1);

    // Write result back
    int p = params.mpi_coords[solvedim];
    int start;
    if (solvedim != 0) {
      start = (id / dims[0]) * start_pad + (id % dims[0]);
    } else {
      start = id * pads[0];
    }
    dd[start]                 = dd_r[id * sys_len_r + p * 2];
    dd[start + result_stride] = dd_r[id * sys_len_r + p * 2 + 1];
  }
}


template <typename REAL, int INC>
inline void backward(const REAL *aa, const REAL *cc,
                     const REAL *dd, REAL *d, REAL *u, const int *dims,
                     const int *pads, int ndim, int solvedim, int n_sys) {
  if(solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/

    // Do the backward pass to solve for remaining unknowns
    #pragma omp parallel for
    for (int id = 0; id < n_sys; id++) {
      int ind = id * pads[0];
      thomas_backward<REAL, INC>(&aa[ind], &cc[ind], &dd[ind], &d[ind], &u[ind],
                                 dims[0], 1);
    }
  } else if(solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/

    // Check if 2D solve
    if(ndim == 2) {
      // Do the backward pass to solve for remaining unknowns
      thomas_backward_vec_strip<REAL, INC>(aa, cc, dd, d, u, dims[1], pads[0],
                                           pads[0]);
    } else {
      // Assume 3D solve
      // Do the backward pass to solve for remaining unknowns
      #pragma omp parallel for
      for (int z = 0; z < dims[2]; z++) {
        int ind = z * pads[0] * pads[1];
        thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                             &d[ind], &u[ind], dims[1], pads[0],
                                             pads[0]);
      }
    }
  } else if(solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/

    // Do the backward pass to solve for remaining unknowns
    #pragma omp parallel for
    for (int ind = 0; ind < ROUND_DOWN(dims[1] * pads[0], Z_BATCH);
         ind += Z_BATCH) {
      thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                           &d[ind], &u[ind], dims[2],
                                           pads[0] * pads[1], Z_BATCH);
    }

    if (dims[1] * pads[0] != ROUND_DOWN(dims[1] * pads[0], Z_BATCH)) {
      int ind    = ROUND_DOWN(dims[1] * pads[0], Z_BATCH);
      int length = (dims[1] * pads[0]) - ind;
      thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                           &d[ind], &u[ind], dims[2],
                                           pads[0] * pads[1], length);
    }
  }
}

template <typename REAL, int INC>
inline void backward_batched(const REAL *aa, const REAL *cc, const REAL *dd,
                             REAL *d, REAL *u, const int *dims, const int *pads,
                             int ndim, int solvedim, int start_sys, int end_sys) {
  int n_sys = end_sys - start_sys;
  if(solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/

    // Do the backward pass to solve for remaining unknowns
    #pragma omp parallel for
    for (int id = start_sys; id < end_sys; id++) {
      int ind = id * pads[0];
      thomas_backward<REAL, INC>(&aa[ind], &cc[ind], &dd[ind], &d[ind], &u[ind],
                                 dims[0], 1);
    }
  } else if(solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/

    // Check if 2D solve
    if(ndim == 2) {
      // Do the backward pass to solve for remaining unknowns
      int batch_size =
          std::max(1, ROUND_DOWN(n_sys / omp_get_max_threads(), SIMD_VEC));
#pragma omp parallel for
      for (int ind = start_sys; ind < start_sys + ROUND_DOWN(n_sys, batch_size);
           ind += batch_size) {
        thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                             &d[ind], &u[ind], dims[1], pads[0],
                                             batch_size);
      }
      // Do final strip if number of systems isn't a multiple of batch_size
      if (n_sys != ROUND_DOWN(n_sys, batch_size)) {
        int ind    = start_sys + ROUND_DOWN(n_sys, batch_size);
        int length = end_sys - ind;
        thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                             &d[ind], &u[ind], dims[1], pads[0],
                                             length);
      }
    } else {
      // Assume 3D solve
      // Do the backward pass to solve for remaining unknowns
      // Iterate over each z 'layer'
      int start_z = start_sys / dims[0];
      int end_z = end_sys / dims[0];
      #pragma omp parallel for
      for (int z = start_z; z < end_z; z++) {
        int ind = z * pads[0] * pads[1];
        // offset in the first z 'layer'
        int strip_start = z == start_z ? start_sys - start_z * dims[0] : 0;
        int strip_len   = z == end_z ? end_sys - end_z * dims[0] : pads[0];
        strip_len -= strip_start;
        ind += strip_start * pads[0];
        thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                             &d[ind], &u[ind], dims[1], pads[0],
                                             strip_len);
      }
    }
  } else if(solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/

    // Do the backward pass to solve for remaining unknowns
    int batch_size =
        std::max(1, ROUND_DOWN(n_sys / omp_get_max_threads(), SIMD_VEC));
    #pragma omp parallel for
    for (int ind = start_sys; ind < start_sys + ROUND_DOWN(n_sys, batch_size);
         ind += batch_size) {
      thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                           &d[ind], &u[ind], dims[2],
                                           pads[0] * pads[1], batch_size);
    }

    if (n_sys != ROUND_DOWN(n_sys, batch_size)) {
      int ind    = start_sys + ROUND_DOWN(n_sys, batch_size);
      int length = end_sys - ind;
      thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                           &d[ind], &u[ind], dims[2],
                                           pads[0] * pads[1], length);
    }
  }
}


// LH
template <typename REAL, int INC>
void tridMultiDimBatchSolve(const MpiSolverParams &params, const REAL *a,
                            const REAL *b, const REAL *c, REAL *d, REAL *u,
                            int ndim, int solvedim, const int *dims,
                            const int *pads) {
  PROFILE_FUNCTION();
  BEGIN_PROFILING("memalloc");
  // Calculate number of systems that will be solved in this dimension
  int n_sys = 1;
  // Calculate size needed for aa, cc and dd arrays
  int mem_size = 1;
  for (int i = 0; i < ndim; i++) {
    if (i != solvedim) {
      n_sys *= dims[i];
    }
    mem_size *= pads[i];
  }

  // Allocate memory for aa, cc and dd arrays
  REAL *aa = (REAL *)_mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);
  REAL *cc = (REAL *)_mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);
  REAL *dd = (REAL *)_mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);

  // Length of a reduced system
  int len_r_local = 2 * 3;
  int sys_len_r   = 2 * params.num_mpi_procs[solvedim];

  // Allocate memory for send and receive buffers
  REAL *sndbuf =
      (REAL *)_mm_malloc(n_sys * len_r_local * sizeof(REAL), SIMD_WIDTH);
  REAL *rcvbuf =
      (REAL *)_mm_malloc(n_sys * sys_len_r * 3 * sizeof(REAL), SIMD_WIDTH);

  // Get MPI datatype
  const MPI_Datatype mpi_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // Allocate memory for reduced solve
  REAL *aa_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);
  REAL *cc_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);
  REAL *dd_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);

  END_PROFILING("memalloc");
  int batch_size  = std::min(n_sys, params.mpi_batch_size);
  int num_batches = 1 + (n_sys - 1) / batch_size;
  MPI_Request request;
  
  for(int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? n_sys - batch_start : batch_size;
    BEGIN_PROFILING("forward");
    forward_batched(a, b, c, d, aa, cc, dd, sndbuf, dims, pads, ndim, solvedim,
            batch_start, batch_start + bsize);
    END_PROFILING("forward");
    BEGIN_PROFILING("mpi_communication");
    // wait for the previous MPI transaction to finish
    if (bidx != 0) {
      MPI_Status status;
      MPI_Wait(&request, &status);
    }
    // Send boundaries of the current batch
    size_t comm_size = len_r_local * bsize;
    size_t buffer_offset = len_r_local * batch_start;
    size_t recv_buffer_offset = 3 * sys_len_r * batch_start;
    // Communicate reduced systems
    MPI_Iallgather(sndbuf + buffer_offset, comm_size, mpi_datatype,
                   rcvbuf + recv_buffer_offset, comm_size, mpi_datatype,
                   params.communicators[solvedim], &request);
    END_PROFILING("mpi_communication");
    // Finish the previous batch
    if (bidx != 0) {
      int batch_start      = (bidx - 1) * batch_size;
      int bsize            = batch_size;
      BEGIN_PROFILING("pcr_on_reduced");
      // Solve reduced systems on each node
      solve_reduced_batched(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads,
                            solvedim, sys_len_r, batch_start,
                            batch_start + bsize);
      END_PROFILING("pcr_on_reduced");
      BEGIN_PROFILING("backward");
      backward_batched<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim,
                                  batch_start, batch_start + bsize);
      END_PROFILING("backward");
    }
  }
  // wait for last message and finish last batch
  BEGIN_PROFILING("mpi_communication");
  MPI_Status status;
  MPI_Wait(&request, &status);
  END_PROFILING("mpi_communication");
  // Finish the previous batch
  int batch_start = (num_batches - 1) * batch_size;
  int bsize       = n_sys - batch_start;
  BEGIN_PROFILING("pcr_on_reduced");
  // Solve reduced systems on each node
  solve_reduced_batched(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads,
                        solvedim, sys_len_r, batch_start, batch_start + bsize);
  END_PROFILING("pcr_on_reduced");
  BEGIN_PROFILING("backward");
  backward_batched<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim,
                              batch_start, batch_start + bsize);
  END_PROFILING("backward");

  // Free memory used in solve
  BEGIN_PROFILING("memfree");
  _mm_free(aa);
  _mm_free(cc);
  _mm_free(dd);
  _mm_free(sndbuf);
  _mm_free(rcvbuf);
  _mm_free(aa_r);
  _mm_free(cc_r);
  _mm_free(dd_r);
  END_PROFILING("memfree");
}


template <typename REAL, int INC>
void tridMultiDimBatchSolve_orig(const MpiSolverParams &params, const REAL *a,
                                 const REAL *b, const REAL *c, REAL *d, REAL *u,
                                 int ndim, int solvedim, const int *dims,
                                 const int *pads) {
  PROFILE_FUNCTION();
  BEGIN_PROFILING("memalloc");
  // Calculate number of systems that will be solved in this dimension
  int n_sys = 1;
  // Calculate size needed for aa, cc and dd arrays
  int mem_size = 1;
  for (int i = 0; i < ndim; i++) {
    if (i != solvedim) {
      n_sys *= dims[i];
    }
    mem_size *= pads[i];
  }

  // Allocate memory for aa, cc and dd arrays
  REAL *aa = (REAL *)_mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);
  REAL *cc = (REAL *)_mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);
  REAL *dd = (REAL *)_mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);

  // Length of a reduced system
  int len_r_local = 2 * 3;
  int sys_len_r   = 2 * params.num_mpi_procs[solvedim];

  // Allocate memory for send and receive buffers
  REAL *sndbuf =
      (REAL *)_mm_malloc(n_sys * len_r_local * sizeof(REAL), SIMD_WIDTH);
  REAL *rcvbuf =
      (REAL *)_mm_malloc(n_sys * sys_len_r * 3 * sizeof(REAL), SIMD_WIDTH);

  // Get MPI datatype
  const MPI_Datatype mpi_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // Allocate memory for reduced solve
  REAL *aa_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);
  REAL *cc_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);
  REAL *dd_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);

  END_PROFILING("memalloc");
  BEGIN_PROFILING("forward");
  forward(a, b, c, d, aa, cc, dd, sndbuf, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("forward");
  BEGIN_PROFILING("mpi_communication");
  // Communicate reduced systems
  MPI_Allgather(sndbuf, n_sys * len_r_local, mpi_datatype, rcvbuf,
                n_sys * len_r_local, mpi_datatype,
                params.communicators[solvedim]);

  END_PROFILING("mpi_communication");
  BEGIN_PROFILING("pcr_on_reduced");
  // Solve reduced systems on each node
  solve_reduced(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads, solvedim,
                sys_len_r, n_sys);
  END_PROFILING("pcr_on_reduced");
  BEGIN_PROFILING("backward");
  backward<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("backward");

  // Free memory used in solve
  BEGIN_PROFILING("memfree");
  _mm_free(aa);
  _mm_free(cc);
  _mm_free(dd);
  _mm_free(sndbuf);
  _mm_free(rcvbuf);
  _mm_free(aa_r);
  _mm_free(cc_r);
  _mm_free(dd_r);
  END_PROFILING("memfree");
}

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must
// be stored in arrays of size 'dims' with 'ndim' dimensions. The 'pads' array
// specifies any padding used in the arrays (the total length of each dimension
// including padding).
//
// The result is written to 'd'. 'u' is unused.

#if FPPREC == 1
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads) {
  tridMultiDimBatchSolve<double, 0>(params, a, b, c, d, u, ndim, solvedim, dims,
                                    pads);
  return TRID_STATUS_SUCCESS;
}
#else
tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, const int *dims,
                                      const int *pads) {
  tridMultiDimBatchSolve<float, 0>(params, a, b, c, d, u, ndim, solvedim, dims,
                                   pads);
  return TRID_STATUS_SUCCESS;
}
#endif

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must
// be stored in arrays of size 'dims' with 'ndim' dimensions. The 'pads' array
// specifies any padding used in the arrays (the total length of each dimension
// including padding).
//
// 'u' is incremented with the results.
#if FPPREC == 1
tridStatus_t tridDmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const double *a, const double *b,
                                         const double *c, double *d, double *u,
                                         int ndim, int solvedim,
                                         const int *dims, const int *pads) {
  tridMultiDimBatchSolve<double, 1>(params, a, b, c, d, u, ndim, solvedim, dims,
                                    pads);
  return TRID_STATUS_SUCCESS;
}
#else
tridStatus_t tridSmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const float *a, const float *b,
                                         const float *c, float *d, float *u,
                                         int ndim, int solvedim,
                                         const int *dims, const int *pads) {
  tridMultiDimBatchSolve<float, 1>(params, a, b, c, d, u, ndim, solvedim, dims,
                                   pads);
  return TRID_STATUS_SUCCESS;
}
#endif
