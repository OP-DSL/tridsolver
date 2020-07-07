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

#include "trid_mpi_cpu_lh.hpp"

#define USE_TIMER_MACRO
#include "timer.hpp"

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

#define Z_BATCH 56

template<typename REAL, int INC>
void tridMultiDimBatchSolve(const MpiSolverParams &params, const REAL *a, const REAL *b,
                            const REAL *c, REAL *d, REAL *u, int ndim, int solvedim,
                            int *dims, int *pads) {
  // Declare timers
  TIMER_DECL(forward);
  TIMER_DECL(backward);
  TIMER_DECL(pcr_on_reduced);
  TIMER_DECL(mpi_communication);
  // Calculate number of systems that will be solved in this dimension
  int n_sys = 1;
  // Calculate size needed for aa, cc and dd arrays
  int mem_size = 1;
  for(int i = 0; i < ndim; i++) {
    if(i != solvedim) {
      n_sys *= dims[i];
    }
    mem_size *= pads[i];
  }

  // Allocate memory for aa, cc and dd arrays
  REAL *aa  = (REAL *) _mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);
  REAL *cc  = (REAL *) _mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);
  REAL *dd  = (REAL *) _mm_malloc(mem_size * sizeof(REAL), SIMD_WIDTH);

  // Length of a reduced system
  int sys_len_r = 2 * params.num_mpi_procs[solvedim];

  // Allocate memory for send and receive buffers
  REAL *sndbuf = (REAL *) _mm_malloc(n_sys * sys_len_r * 3 * sizeof(REAL), SIMD_WIDTH);
  REAL *rcvbuf = (REAL *) _mm_malloc(n_sys * sys_len_r * 3 * sizeof(REAL), SIMD_WIDTH);

  // Get MPI datatype
  const MPI_Datatype mpi_datatype = std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // Allocate memory for reduced solve
  REAL *aa_r = (REAL *) _mm_malloc(sizeof(REAL) * sys_len_r, SIMD_WIDTH);
  REAL *cc_r = (REAL *) _mm_malloc(sizeof(REAL) * sys_len_r, SIMD_WIDTH);
  REAL *dd_r = (REAL *) _mm_malloc(sizeof(REAL) * sys_len_r, SIMD_WIDTH);

  TIMER_TOGGLE(forward);
  if(solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/

    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < n_sys; id++) {
      int ind = id * pads[0];
      thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                           &aa[ind], &cc[ind], &dd[ind], dims[0], 1);
    }

    // Pack reduced systems (boundaries of each tridiagonal system)
    #pragma omp parallel for
    for(int id = 0; id < n_sys; id++) {
      int buf_ind = id * 6;
      int data_ind = id * pads[0];
      sndbuf[buf_ind]     = aa[data_ind];
      sndbuf[buf_ind + 1] = aa[data_ind + dims[0] - 1];
      sndbuf[buf_ind + 2] = cc[data_ind];
      sndbuf[buf_ind + 3] = cc[data_ind + dims[0] - 1];
      sndbuf[buf_ind + 4] = dd[data_ind];
      sndbuf[buf_ind + 5] = dd[data_ind + dims[0] - 1];
    }
  } else if(solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/

    // Check if 2D solve
    if(ndim == 2) {
      // Do modified thomas forward pass
      thomas_forward_vec_strip<REAL>(a, b, c, d, u, aa, cc, dd,
                                     dims[1], pads[0], pads[0]);

      // Pack reduced systems (boundaries of each tridiagonal system)
      #pragma omp parallel for
      for(int id = 0; id < n_sys; id++) {
        int start = id;
        int end = start + (pads[0] * (dims[1] - 1));
        int buf_ind = id * 6;
        sndbuf[buf_ind]     = aa[start];
        sndbuf[buf_ind + 1] = aa[end];
        sndbuf[buf_ind + 2] = cc[start];
        sndbuf[buf_ind + 3] = cc[end];
        sndbuf[buf_ind + 4] = dd[start];
        sndbuf[buf_ind + 5] = dd[end];
      }
    } else {
      // Assume 3D solve

      // Do modified thomas forward pass
      // Iterate over each z 'layer'
      #pragma omp parallel for
      for(int z = 0; z < dims[2]; z++) {
        int ind = z * pads[0] * pads[1];
        thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                      &u[ind], &aa[ind], &cc[ind], &dd[ind],
                                      dims[1], pads[0], pads[0]);
      }

      // Pack reduced systems (boundaries of each tridiagonal system)
      #pragma omp parallel for
      for(int id = 0; id < n_sys; id++) {
        int start = (id/dims[0]) * pads[0] * pads[1] + (id % dims[0]);
        int end = start + (pads[0] * (dims[1] - 1));
        int buf_ind = id * 6;
        sndbuf[buf_ind]     = aa[start];
        sndbuf[buf_ind + 1] = aa[end];
        sndbuf[buf_ind + 2] = cc[start];
        sndbuf[buf_ind + 3] = cc[end];
        sndbuf[buf_ind + 4] = dd[start];
        sndbuf[buf_ind + 5] = dd[end];
      }
    }
  } else if(solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/

    // Do modified thomas forward pass
    // Iterate over strips of length Z_BATCH
    #pragma omp parallel for
    for(int ind = 0; ind < ROUND_DOWN(dims[1] * pads[0], Z_BATCH); ind += Z_BATCH) {
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], Z_BATCH);
    }

    // Do final strip if number of systems isn't a multiple of Z_BATCH
    if(dims[1] * pads[0] != ROUND_DOWN(dims[1] * pads[0], Z_BATCH)) {
      int ind = ROUND_DOWN(dims[1] * pads[0], Z_BATCH);
      int length = (dims[1] * pads[0]) - ind;
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &u[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], length);
    }

    // Pack reduced systems (boundaries of each tridiagonal system)
    #pragma omp parallel for
    for(int id = 0; id < n_sys; id++) {
      int start = (id/dims[0]) * pads[0] + (id % dims[0]);
      int end = start + (pads[0] * pads[1] * (dims[2] - 1));
      int buf_ind = id * 6;
      sndbuf[buf_ind]     = aa[start];
      sndbuf[buf_ind + 1] = aa[end];
      sndbuf[buf_ind + 2] = cc[start];
      sndbuf[buf_ind + 3] = cc[end];
      sndbuf[buf_ind + 4] = dd[start];
      sndbuf[buf_ind + 5] = dd[end];
    }
  }
  TIMER_TOGGLE(forward);

  TIMER_TOGGLE(mpi_communication);
  // Communicate reduced systems
  MPI_Gather(sndbuf, n_sys*3*2, mpi_datatype, rcvbuf,
             n_sys*3*2, mpi_datatype, 0, params.communicators[solvedim]);

  TIMER_TOGGLE(mpi_communication);
  TIMER_TOGGLE(pcr_on_reduced);
  // Solve reduced system on root nodes of this dimension
  if(params.mpi_coords[solvedim] == 0) {
    // Iterate over each reduced system
    for(int id = 0; id < n_sys; id++) {
      // Unpack this reduced system from receive buffer
      for(int p = 0; p < params.num_mpi_procs[solvedim]; p++) {
        int buf_ind = p * n_sys * 2 * 3;
        aa_r[p * 2]     = rcvbuf[buf_ind + id * 6];
        aa_r[p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 1];
        cc_r[p * 2]     = rcvbuf[buf_ind + id * 6 + 2];
        cc_r[p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 3];
        dd_r[p * 2]     = rcvbuf[buf_ind + id * 6 + 4];
        dd_r[p * 2 + 1] = rcvbuf[buf_ind + id * 6 + 5];
      }

      // Solve reduced system
      thomas_on_reduced<REAL>(aa_r, cc_r, dd_r, sys_len_r, 1);

      // Pack result into send buffer
      for(int p = 0; p < params.num_mpi_procs[solvedim]; p++) {
        int buf_ind = p * n_sys * 2;
        sndbuf[buf_ind + id * 2]     = dd_r[p * 2];
        sndbuf[buf_ind + id * 2 + 1] = dd_r[p * 2 + 1];
      }
    }
  }

  TIMER_TOGGLE(pcr_on_reduced);
  TIMER_TOGGLE(mpi_communication);
  // Send back new values from reduced solve
  MPI_Scatter(sndbuf, n_sys * 2, mpi_datatype, rcvbuf,
               n_sys * 2, mpi_datatype, 0, params.communicators[solvedim]);
  TIMER_TOGGLE(mpi_communication);
  TIMER_TOGGLE(backward);

  if(solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/

    // Unpack reduced solution
    #pragma omp parallel for
    for(int id = 0; id < n_sys; id++) {
      // Gather coefficients of a,c,d
      int data_ind = id * pads[0];
      int buf_ind = id * 2;
      dd[data_ind]               = rcvbuf[buf_ind];
      dd[data_ind + dims[0] - 1] = rcvbuf[buf_ind + 1];
    }

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
      // Unpack reduced solution
      #pragma omp parallel for
      for(int id = 0; id < n_sys; id++) {
        int start = id;
        int end = start + (pads[0] * (dims[1] - 1));
        int buf_ind = id * 2;
        dd[start] = rcvbuf[buf_ind];
        dd[end]   = rcvbuf[buf_ind + 1];
      }

      // Do the backward pass to solve for remaining unknowns
      thomas_backward_vec_strip<REAL, INC>(aa, cc, dd, d, u, dims[1], pads[0],
                                           pads[0]);
    } else {
      // Assume 3D solve

      // Unpack reduced solution
      #pragma omp parallel for
      for(int id = 0; id < n_sys; id++) {
        int start = (id/dims[0]) * pads[0] * pads[1] + (id % dims[0]);
        int end = start + (pads[0] * (dims[1] - 1));
        int buf_ind = id * 2;
        dd[start] = rcvbuf[buf_ind];
        dd[end]   = rcvbuf[buf_ind + 1];
      }

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

    // Unpack reduced solution
    #pragma omp parallel for
    for(int id = 0; id < n_sys; id++) {
      int start = (id/dims[0]) * pads[0] + (id % dims[0]);
      int end = start + (pads[0] * pads[1] * (dims[2] - 1));
      int buf_ind = id * 2;
      dd[start] = rcvbuf[buf_ind];
      dd[end]   = rcvbuf[buf_ind + 1];
    }

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
  TIMER_TOGGLE(backward);


  TIMER_PRINT(forward);
  TIMER_PRINT(mpi_communication);
  TIMER_PRINT(pcr_on_reduced);
  TIMER_PRINT(backward);

  // Free memory used in solve
  _mm_free(aa);
  _mm_free(cc);
  _mm_free(dd);
  _mm_free(sndbuf);
  _mm_free(rcvbuf);
  _mm_free(aa_r);
  _mm_free(cc_r);
  _mm_free(dd_r);
}

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must be stored in
// arrays of size 'dims' with 'ndim' dimensions. The 'pads' array specifies any padding used in
// the arrays (the total length of each dimension including padding).
//
// The result is written to 'd'. 'u' is unused.

#if FPPREC == 1
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolveLH<double, 0>(params, a, b, c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#else
tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<float, 0>(params, a, b, c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#endif

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must be stored in
// arrays of size 'dims' with 'ndim' dimensions. The 'pads' array specifies any padding used in
// the arrays (the total length of each dimension including padding).
//
// 'u' is incremented with the results.
#if FPPREC == 1
tridStatus_t tridDmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const double *a, const double *b,
                                         const double *c, double *d, double *u, int ndim,
                                         int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<double, 1>(params, a, b, c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#else
tridStatus_t tridSmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                         const float *a, const float *b,
                                         const float *c, float *d, float *u, int ndim,
                                         int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<float, 1>(params, a, b, c, d, u, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#endif
