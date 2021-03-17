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
#include <cassert>
#include <cmath>

#include "timing.h"

#define ROUND_DOWN(N, step) (((N) / (step)) * step)
#define MIN(X, Y)           ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)           ((X) > (Y) ? (X) : (Y))

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

// retruns the index of the first element of the sys_idx-th system
inline int get_sys_start_idx(int sys_idx, int solvedim, const int *dims,
                             const int *pads, int ndim) {
  static_assert(MAXDIM == 3,
                "Index calculation onlz implemented for at most 3D problems");
  assert(solvedim < ndim && ndim <= MAXDIM);
  int start_pad = pads[0];
  if (solvedim == 1) start_pad *= pads[1];
  int start;
  if (solvedim == 0) {
    if (ndim == 1) {
      start = sys_idx * pads[0];
    } else {
      start = (sys_idx / dims[1]) * pads[1] * pads[0] +
              (sys_idx % dims[1]) * pads[0];
    }
  } else {
    start = (sys_idx / dims[0]) * start_pad + (sys_idx % dims[0]);
  }
  return start;
}


// return the difference between the index of the first and the index of the
// last element of a system s.t. get_sys_start_idx + get_sys_span gives the
// index of the last element of the system
inline int get_sys_span(int solvedim, const int *dims, const int *pads) {
  int result_stride = dims[solvedim] - 1;
  for (int i = 0; i < solvedim; ++i) {
    result_stride *= pads[i];
  }
  return result_stride;
}

// Transform the reduced system arise from forward pass s.t. the start row for
// each node gives a tridiagonal system. Basically performs half of the first
// iteration of the PCR.
// \begin{equation}
//   \left[\begin{array}{cccccc|c}
//      1   & c_0 &     &        &        &        &  d_0   \\
//      a_1 &  1  & c_1 &        &        &        &  d_1   \\\hline
//          & a_2 &  1  & c_2    &        &        &  d_2   \\
//          &     & a_3 &  1     & c_3    &        &  d_3   \\\hline
//          &     &     & \ddots & \ddots & \ddots & \vdots \\
//   \end{array}\right]
//   \Rightarrow
//   \left[\begin{array}{cccccc|c}
//      1     &   & c_0^* &        &        &        &  d_0^* \\
//      a_1   & 1 & c_1   &        &        &        &  d_1   \\\hline
//      a_2^* &   &  1    &        & c_2^*  &        &  d_2^* \\
//            &   & a_3   &  1     & c_3    &        &  d_3   \\\hline
//            &   &       & \ddots & \ddots & \ddots & \vdots \\
//   \end{array}\right]
// \end{equation}
template <typename REAL>
void eliminate_row_from_reduced(const MpiSolverParams &params, REAL *aa,
                                REAL *cc, REAL *dd, REAL *sndbuf, REAL *rcvbuf,
                                const int *dims, const int *pads, int ndim,
                                int solvedim, int start_sys, int end_sys,
                                int result_stride = -1, int tag_ = -1) {
  assert(start_sys < end_sys);
  MPI_Request rcv_request, snd_request;
  const int rank             = params.mpi_coords[solvedim];
  const int nproc            = params.num_mpi_procs[solvedim];
  const int n_sys            = end_sys - start_sys;
  constexpr int nvar_per_sys = 3; // a, c, d
  const int tag              = tag_ == -1 ? 1242 : tag_;
  if (result_stride < 0) {
    result_stride = get_sys_span(solvedim, dims, pads);
  }
  if (rank) {
    MPI_Irecv(rcvbuf + start_sys * nvar_per_sys, n_sys * nvar_per_sys,
              MPI_DATATYPE(REAL), rank - 1, tag, params.communicators[solvedim],
              &rcv_request);
  }
  for (int id = start_sys; id < end_sys; ++id) {
    int start           = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    int idx             = start + result_stride;
    int buf_idx         = id * nvar_per_sys;
    sndbuf[buf_idx]     = aa[idx];
    sndbuf[buf_idx + 1] = cc[idx];
    sndbuf[buf_idx + 2] = dd[idx];
  }
  if (rank != nproc - 1) {
    MPI_Isend(sndbuf + start_sys * nvar_per_sys, n_sys * nvar_per_sys,
              MPI_DATATYPE(REAL), rank + 1, tag, params.communicators[solvedim],
              &snd_request);
  }
  // clang-format off
  // 1st - c_1 * 2nd:
  // \begin{equation}
  //   \left[\begin{array}{cccc|c}
  //     a_1 &  1  & c_1 &     & d_1 \\
  //         & a_2 &  1  & c_2 & d_2 \\
  //   \end{array}\right] =
  //   \left[\begin{array}{cccc|c}
  //     \frac{a_1}{1 - a_2c_1} &  1  &   & \frac{-c_2c_1}{1 - a_2c_1} & \frac{d_1-d_2c_1}{1 - a_2c_1} \\
  //                            & a_2 & 1 &            c_2             & d_2 \\
  // \end{array}\right]
  // \end{equation}
  // clang-format on
  for (int id = start_sys; id < end_sys; ++id) {
    int top    = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    int bottom = top + result_stride;
    REAL bbi   = 1 / (1 - aa[bottom] * cc[top]);
    dd[top]    = (dd[top] - dd[bottom] * cc[top]) * bbi;
    if (rank) {
      aa[top] = aa[top] * bbi;
    }
    if (rank != nproc - 1) {
      cc[top] = -cc[bottom] * cc[top] * bbi;
    } else {
      cc[top] = 0.0;
    }
  }
  if (rank) {
    MPI_Wait(&rcv_request, MPI_STATUS_IGNORE);
    // clang-format off
    // 1st - a_1 * row_from_prev_rank:
    // \begin{equation}
    //   \left[\begin{array}{ccccc|c}
    //      a_0 &  1  & c_0 &     &     & d_0 \\
    //          & a_1 &  1  &     & c_1 & d_1 \\
    //          &     & a_2 &  1  & c_2 & d_2 \\
    //   \end{array}\right] =
    //   \left[\begin{array}{ccccc|c}
    //                  a_0         &  1  & c_0 &     &                      & d_0 \\
    //     \frac{-a_0a_1}{1-c_0a_1} &     &  1  &     & \frac{c_1}{1-c_0a_1} & \frac{d_1-d_0a_1}{1-c_0a_1} \\
    //                              &     & a_2 &  1  &             c_2      & d_2 \
    //   \end{array}\right]
    // \end{equation}
    // clang-format on
    for (int id = start_sys; id < end_sys; ++id) {
      int top     = get_sys_start_idx(id, solvedim, dims, pads, ndim);
      int buf_idx = id * nvar_per_sys; // order: +0:a, +1:c, +2:d
      REAL am1    = rcvbuf[buf_idx];
      REAL cm1    = rcvbuf[buf_idx + 1];
      REAL dm1    = rcvbuf[buf_idx + 2];
      REAL bbi    = 1 / (1 - cm1 * aa[top]);
      dd[top]     = (dd[top] - dm1 * aa[top]) * bbi;
      aa[top]     = -aa[top] * am1 * bbi;
      cc[top]     = cc[top] * bbi;
    }
  }
  if (rank != nproc - 1) {
    MPI_Wait(&snd_request, MPI_STATUS_IGNORE);
  }
}

// Variant for eliminate_row_from_reduced. This function saves the state of the
// first row on the first rank and eliminates a_0 on the second rank. The new
// tridiagonal system is on rank 1 - n.
template <typename REAL>
void eliminate_row_from_reduced_for_jacobi(
    const MpiSolverParams &params, REAL *aa, REAL *cc, REAL *dd, REAL *sndbuf,
    REAL *rcvbuf, const int *dims, const int *pads, int ndim, int solvedim,
    int start_sys, int end_sys, int result_stride = -1) {
  const int rank  = params.mpi_coords[solvedim];
  const int nproc = params.num_mpi_procs[solvedim];
  // we can reduce the size of the whole system by one since a_0 on rank = 1 can
  // be 0 after elimination which reduce the number of required jacobi
  // iterations.
  constexpr int tag          = 1242;
  constexpr int nvar_per_sys = 3;
  if (result_stride < 0) {
    result_stride = get_sys_span(solvedim, dims, pads);
  }
  if (!rank && nproc > 1) {
#pragma omp parallel for
    for (int id = start_sys; id < end_sys; ++id) {
      int start           = get_sys_start_idx(id, solvedim, dims, pads, ndim);
      int last            = start + result_stride;
      REAL bbi            = 1 / (1 - aa[last] * cc[start]);
      cc[last]            = cc[last] * bbi;
      dd[last]            = (dd[last] - aa[last] * dd[start]) * bbi;
      aa[last]            = 0;
      int buf_idx         = id * nvar_per_sys;
      sndbuf[buf_idx]     = aa[last];
      sndbuf[buf_idx + 1] = cc[last];
      sndbuf[buf_idx + 2] = dd[last];
    }
    MPI_Request snd_request;
    int n_sys = end_sys - start_sys;
    MPI_Isend(sndbuf + start_sys * nvar_per_sys, n_sys * nvar_per_sys,
              MPI_DATATYPE(REAL), rank + 1, tag, params.communicators[solvedim],
              &snd_request);
    MPI_Wait(&snd_request, MPI_STATUS_IGNORE);

  } else {
    eliminate_row_from_reduced(params, aa, cc, dd, sndbuf, rcvbuf, dims, pads,
                               ndim, solvedim, start_sys, end_sys,
                               result_stride, tag);
  }
}

// Get the result for the last row on each node after reduced solve with
// eliminate_row_from_reduced and reduced solver. The solution for the first row
// is in dd_r, the solution for the first and last row is stored in dd on exit.
// Each node receive one value from the next node to calculate last row.
// iteration of the PCR. Note the new zeros in a, and c are not written to the
// arrays. \begin{equation}
//   \left[\begin{array}{cccccc|c}
//      1   &     &     &        &        &        &  d_0   \\
//      a_1 &  1  & c_1 &        &        &        &  d_1   \\\hline
//          &     &  1  &        &        &        &  d_2   \\
//          &     & a_3 &  1     & c_3    &        &  d_3   \\\hline
//          &     &     & \ddots & \ddots & \ddots & \vdots \\
//   \end{array}\right]
//   \Rightarrow
//   \left[\begin{array}{cccccc|c}
//     1 &   &   &   &        & &  d_0   \\
//       & 1 &   &   &        & &  d_1^* \\\hline
//       &   & 1 &   &        & &  d_2   \\
//       &   &   & 1 &        & &  d_3^* \\\hline
//       &   &   &   & \ddots & & \vdots \\
//   \end{array}\right]
// \end{equation}
template <typename REAL>
void compute_last_for_reduced(const MpiSolverParams &params, const REAL *aa,
                              const REAL *cc, REAL *dd, const REAL *dd_r,
                              REAL *rcvbuf, const int *dims, const int *pads,
                              int ndim, int solvedim, int start_sys,
                              int end_sys, int result_stride = -1,
                              int tag_ = -1) {
  assert(start_sys < end_sys);
  MPI_Request rcv_request, snd_request;
  const int rank  = params.mpi_coords[solvedim];
  const int nproc = params.num_mpi_procs[solvedim];
  const int n_sys = end_sys - start_sys;
  const int tag   = tag_ == -1 ? 1342 : tag_;
  if (result_stride < 0) {
    result_stride = get_sys_span(solvedim, dims, pads);
  }
  if (rank != nproc - 1) {
    MPI_Irecv(rcvbuf + start_sys, n_sys, MPI_DATATYPE(REAL), rank + 1, tag,
              params.communicators[solvedim], &rcv_request);
  }
  if (rank) {
    MPI_Isend(dd_r + start_sys, n_sys, MPI_DATATYPE(REAL), rank - 1, tag,
              params.communicators[solvedim], &snd_request);
  }
  if (rank != nproc - 1) {
    MPI_Wait(&rcv_request, MPI_STATUS_IGNORE);
  }
  // clang-format off
  // 2nd - a_2 * 1st - c_2 * row_from_next_node:
  // \begin{equation}
  //   \left[\begin{array}{cccc|c}
  //         &  1  &   &     & d_1 \\
  //         & a_2 & 1 & c_2 & d_2 \\\hline
  //         &     &   &  1  & d_3 \\
  //   \end{array}\right] =
  //   \left[\begin{array}{cccc|c}
  //         &  1  &   &     & d_1 \\
  //         &     & 1 &     & d_2 - a_2d_1 - c_2d_3 \\\hline
  //         &     &   &  1  & d_3 \\
  // \end{array}\right]
  // \end{equation}
  // clang-format on
  for (int id = start_sys; id < end_sys; ++id) {
    int top    = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    int bottom = top + result_stride;
    dd[top]    = dd_r[id];
    dd[bottom] = (dd[bottom] - aa[bottom] * dd[top]);
    if (rank != nproc - 1) {
      dd[bottom] = dd[bottom] - cc[bottom] * rcvbuf[id];
    }
  }
  if (rank) {
    MPI_Wait(&snd_request, MPI_STATUS_IGNORE);
  }
}

// Variant for compute_last_for_reduced, this function handles rank 0 as if it
// was not part of the reduced solve.
template <typename REAL>
void compute_last_for_reduced_jacobi(const MpiSolverParams &params,
                                     const REAL *aa, const REAL *cc, REAL *dd,
                                     const REAL *dd_r, REAL *rcvbuf,
                                     const int *dims, const int *pads, int ndim,
                                     int solvedim, int start_sys, int end_sys,
                                     int result_stride = -1) {
  assert(start_sys < end_sys);
  const int rank    = params.mpi_coords[solvedim];
  const int nproc   = params.num_mpi_procs[solvedim];
  constexpr int tag = 1342;
  if (result_stride < 0) {
    result_stride = get_sys_span(solvedim, dims, pads);
  }
  if (!rank && nproc > 1) {
    MPI_Request rcv_request;
    const int n_sys = end_sys - start_sys;
    MPI_Irecv(rcvbuf + start_sys, n_sys, MPI_DATATYPE(REAL), rank + 1, tag,
              params.communicators[solvedim], &rcv_request);
    MPI_Wait(&rcv_request, MPI_STATUS_IGNORE);
#pragma omp parallel for
    for (int id = start_sys; id < end_sys; ++id) {
      int top    = get_sys_start_idx(id, solvedim, dims, pads, ndim);
      int bottom = top + result_stride;
      // dd[top]    = dd_r[id]; rank 0 does not compute solution
      dd[bottom] = dd[bottom] - cc[bottom] * rcvbuf[id];
      dd[top]    = dd[top] - cc[top] * dd[bottom];
    }
  } else {
    compute_last_for_reduced(params, aa, cc, dd, dd_r, rcvbuf, dims, pads, ndim,
                             solvedim, start_sys, end_sys, result_stride, tag);
  }
}

// Version that accounts for positive and negative padding
template <typename REAL>
inline void copy_boundaries_strided(const REAL *aa, const REAL *cc,
                                    const REAL *dd, REAL *sndbuf,
                                    const int *dims, const int *pads, int ndim,
                                    int solvedim, int start_sys, int end_sys) {
  int result_stride = get_sys_span(solvedim, dims, pads);

#pragma omp parallel for
  for (int id = start_sys; id < end_sys; id++) {
    int start           = get_sys_start_idx(id, solvedim, dims, pads, ndim);
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
inline void forward_batched(const REAL *a, const REAL *b, const REAL *c,
                            const REAL *d, REAL *aa, REAL *cc, REAL *dd,
                            const int *dims, const int *pads, int ndim,
                            int solvedim, int start_sys, int end_sys) {
  int n_sys = end_sys - start_sys;
  if (solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/
    if (ndim == 1) {
      // Do modified thomas forward pass
#pragma omp parallel for
      for (int id = start_sys; id < end_sys; id++) {
        int ind = id * pads[0];
        thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &aa[ind],
                             &cc[ind], &dd[ind], dims[0], 1);
      }
    } else {
      // Do modified thomas forward pass
#pragma omp parallel for
      for (int id = start_sys; id < end_sys; id++) {
        int ind = (id / dims[1]) * pads[1] * pads[0] + (id % dims[1]) * pads[0];
        thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &aa[ind],
                             &cc[ind], &dd[ind], dims[0], 1);
      }
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
      int end_z   = end_sys / dims[0];
#pragma omp parallel for
      for (int z = start_z; z <= end_z; z++) {
        int ind = z * pads[0] * pads[1];
        // offset in the first z 'layer'
        int strip_start = z == start_z ? start_sys - start_z * dims[0] : 0;
        int strip_len   = z == end_z ? end_sys - end_z * dims[0] : dims[0];
        strip_len -= strip_start;
        // ind += strip_start * pads[0];
        ind += strip_start;
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

    int start_y = start_sys / dims[0];
    int end_y   = end_sys / dims[0];
#pragma omp parallel for
    for (int y = start_y; y <= end_y; y++) {
      int ind         = y * pads[0];
      int strip_start = y == start_y ? start_sys - start_y * dims[0] : 0;
      int strip_len   = y == end_y ? end_sys - end_y * dims[0] : dims[0];
      strip_len -= strip_start;
      // ind += strip_start * pads[0];
      ind += strip_start;
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], strip_len);
    }
  }
}

// Positive and negative padding version
// TODO generalize padding caclulations (not just hardcode the 3D case)
template <typename REAL>
inline void forward(const REAL *a, const REAL *b, const REAL *c, const REAL *d,
                    REAL *aa, REAL *cc, REAL *dd, const int *dims,
                    const int *pads, int ndim, int solvedim, int n_sys) {

  if (solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/
    if (ndim == 1) {
      // Do modified thomas forward pass
#pragma omp parallel for
      for (int id = 0; id < n_sys; id++) {
        int ind = id * pads[0];
        thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &aa[ind],
                             &cc[ind], &dd[ind], dims[0], 1);
      }
    } else {
      // Do modified thomas forward pass
#pragma omp parallel for
      for (int id = 0; id < n_sys; id++) {
        int ind = (id / dims[1]) * pads[1] * pads[0] + (id % dims[1]) * pads[0];
        thomas_forward<REAL>(&a[ind], &b[ind], &c[ind], &d[ind], &aa[ind],
                             &cc[ind], &dd[ind], dims[0], 1);
      }
    }
  } else if (solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/
    // 2D should not need to be altered for positive and negative padding
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
      // Do final strip if number of systems isn't a multiple of batch_size
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
                                       pads[0], dims[0]);
      }
    }
  } else if (solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/
    // Do modified thomas forward pass

#pragma omp parallel for
    for (int ind = 0; ind < dims[1] * pads[0]; ind += pads[0]) {
      thomas_forward_vec_strip<REAL>(&a[ind], &b[ind], &c[ind], &d[ind],
                                     &aa[ind], &cc[ind], &dd[ind], dims[2],
                                     pads[0] * pads[1], dims[0]);
    }
  }
}

template <typename REAL>
inline void solve_reduced_batched(const MpiSolverParams &params,
                                  const REAL *rcvbuf, REAL *aa_r, REAL *cc_r,
                                  REAL *dd_r, REAL *dd, const int *dims,
                                  const int *pads, int ndim, int solvedim,
                                  int sys_len_r, int start_sys, int end_sys) {
  int n_sys         = end_sys - start_sys;
  int result_stride = get_sys_span(solvedim, dims, pads);

  // shift buffer
  rcvbuf += start_sys * sys_len_r * 3;

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
    int p     = params.mpi_coords[solvedim];
    int g_id  = id + start_sys;
    int start = get_sys_start_idx(g_id, solvedim, dims, pads, ndim);
    dd[start] = dd_r[id * sys_len_r + p * 2];
    dd[start + result_stride] = dd_r[id * sys_len_r + p * 2 + 1];
  }
}

template <typename REAL>
inline void solve_reduced(const MpiSolverParams &params, const REAL *rcvbuf,
                          REAL *aa_r, REAL *cc_r, REAL *dd_r, REAL *dd,
                          const int *dims, const int *pads, int ndim,
                          int solvedim, int sys_len_r, int n_sys) {
  int result_stride = get_sys_span(solvedim, dims, pads);

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
    int p     = params.mpi_coords[solvedim];
    int start = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    dd[start] = dd_r[id * sys_len_r + p * 2];
    dd[start + result_stride] = dd_r[id * sys_len_r + p * 2 + 1];
  }
}

// Solve reduced system with jacobi iterations.
// aa, cc, dd stores the result of the forward run. For each node for each
// system the first and last row of aa, cc, dd will create the reduced system,
// b_i = 1 everywhere. dd_r is an array with size at least n_sys will store
// the intermediate result of the jacobi iteration for the node. sndbuf and
// rcvbuf must be an array with size at least n_sys. the solution of the
// reduced system will be stored in dd
template <typename REAL>
inline void solve_reduced_jacobi(const MpiSolverParams &params, REAL *aa,
                                 REAL *cc, REAL *dd, REAL *dd_r, REAL *rcvbuf,
                                 REAL *sndbuf, const int *dims, const int *pads,
                                 int ndim, int solvedim, int n_sys) {
  assert(params.strategy == MpiSolverParams::JACOBI);
  int rank      = params.mpi_coords[solvedim];
  int nproc     = params.num_mpi_procs[solvedim];
  REAL *rcvbufL = rcvbuf;
  REAL *rcvbufR = rcvbuf + n_sys;

  // norm comp
  int global_sys_len = 0;
  double global_norm = 0.0;
  double norm0       = -1.0;
  BEGIN_PROFILING("mpi_communication");
  MPI_Allreduce(&dims[solvedim], &global_sys_len, 1, MPI_INT, MPI_SUM,
                params.communicators[solvedim]);
  END_PROFILING("mpi_communication");

  int result_stride = get_sys_span(solvedim, dims, pads);


  eliminate_row_from_reduced_for_jacobi(params, aa, cc, dd, rcvbuf, sndbuf,
                                        dims, pads, ndim, solvedim, 0, n_sys,
                                        result_stride);
  // at this point rank 1 - nproc give a tridiagonal system. We will solve this
  // with the jacobi iteration, rank 0 does not do anything but communication

// write initial guess to dd_r, note: b == 1
#pragma omp parallel for
  for (int id = 0; id < n_sys; ++id) {
    int start = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    if (rank)
      dd_r[id] = dd[start];
    else {
      dd_r[id] = 0;
    }
  }

  // TODO add iter parameter to params
  int iter    = 0;
  int maxiter = 10;
  do {
    BEGIN_PROFILING("mpi_communication");
#pragma omp parallel for
    for (int id = 0; id < n_sys; ++id) {
      rcvbufL[id] = 0;
      rcvbufR[id] = 0;
    }
    MPI_Request req[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                          MPI_REQUEST_NULL};
    if (rank)
      MPI_Irecv(rcvbufL, n_sys, MPI_DATATYPE(REAL), rank - 1, 2,
                params.communicators[solvedim], &req[0]);
    if (rank != nproc - 1)
      MPI_Irecv(rcvbufR, n_sys, MPI_DATATYPE(REAL), rank + 1, 3,
                params.communicators[solvedim], &req[1]);
      // copy boundaries and send to negihbours
#pragma omp parallel for
    for (int id = 0; id < n_sys; ++id) {
      sndbuf[id] = dd_r[id];
    }
    if (rank)
      MPI_Isend(sndbuf, n_sys, MPI_DATATYPE(REAL), rank - 1, 3,
                params.communicators[solvedim], &req[2]);
    if (rank != nproc - 1)
      MPI_Isend(sndbuf, n_sys, MPI_DATATYPE(REAL), rank + 1, 2,
                params.communicators[solvedim], &req[3]);
    MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    END_PROFILING("mpi_communication");
    /* calculate error norm - boundary */
    double local_norm = 0;
    if (rank) {
      for (int id = 0; id < n_sys; id++) {
        int start = get_sys_start_idx(id, solvedim, dims, pads, ndim);
        REAL dm1  = rcvbufL[id];
        REAL dp1  = rcvbufR[id];
        // norm += (a_{0} * x_{-1}  + b_0 * x_0 + c_{0} * x_{1} - d_0)^2
        double diff = dd_r[id] - dd[start];
        if (rank) diff += aa[start] * dm1;
        if (rank != nproc - 1) diff += cc[start] * dp1;

        double sys_norm = diff * diff;
        local_norm      = std::max(local_norm, sys_norm);
      }
    }
    /* sum over all processes */
    BEGIN_PROFILING("mpi_communication");
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM,
                  params.communicators[solvedim]);
    global_norm = sqrt(global_norm / global_sys_len);
    if (norm0 < 0) norm0 = global_norm;
    END_PROFILING("mpi_communication");
    /* correct the solution for this iteration */
    if (rank) {
#pragma omp parallel for
      for (int id = 0; id < n_sys; ++id) {
        int start = get_sys_start_idx(id, solvedim, dims, pads, ndim);
        // x_0 = (d_0 - a_{0} * x_{-1} - c_{0} * x_{1}) / b_0
        dd_r[id] = dd[start];
        if (rank != nproc - 1) dd_r[id] -= cc[start] * rcvbufR[id];
        if (rank) dd_r[id] -= aa[start] * rcvbufL[id];
      }
    }
    if (!rank) std::cout << "iter: " << iter << " " << global_norm << "\n";
    iter++;
    // TODO
  } while (iter < maxiter);
  /*} while ((params.jacobi_atol < global_norm &&
            params.jacobi_rtol < global_norm / norm0));*/

  compute_last_for_reduced_jacobi(params, aa, cc, dd, dd_r, rcvbuf, dims, pads,
                                  ndim, solvedim, 0, n_sys, result_stride);
}

// Solve reduced system with PCR algorithm.
// aa, cc, dd stores the result of the forward run. For each node for each
// system the first and last row of aa, cc, dd will create the reduced system,
// b_i = 1 everywhere.
// TODO
// dd_r is an array with size at least n_sys will store
// the intermediate result of the jacobi iteration for the node.
// sndbuf and
// rcvbuf must be an array with size at least n_sys. the solution of the
// reduced system will be stored in dd
template <typename REAL>
inline void solve_reduced_pcr(const MpiSolverParams &params, REAL *aa, REAL *cc,
                              REAL *dd, REAL *aa_r, REAL *cc_r, REAL *dd_r,
                              REAL *rcvbuf, REAL *sndbuf, const int *dims,
                              const int *pads, int ndim, int solvedim,
                              int n_sys) {
  int rank                   = params.mpi_coords[solvedim];
  int nproc                  = params.num_mpi_procs[solvedim];
  constexpr int tag          = 1242;
  constexpr int nvar_per_sys = 3;
  REAL *rcvbufL              = rcvbuf;
  REAL *rcvbufR              = rcvbuf + n_sys * nvar_per_sys;
  int result_stride          = get_sys_span(solvedim, dims, pads);

  // First step eliminate one row on each node
  eliminate_row_from_reduced(params, aa, cc, dd, rcvbuf, sndbuf, dims, pads,
                             ndim, solvedim, 0, n_sys, result_stride);
// Iterate over each reduced system
#pragma omp parallel for
  for (int id = 0; id < n_sys; id++) {
    // Unpack this reduced system from receive buffer
    int begin = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    if (rank) {
      aa_r[id] = aa[begin];
    } else {
      aa_r[id] = 0.0;
    }
    cc_r[id]                      = cc[begin];
    dd_r[id]                      = dd[begin];
    sndbuf[id * nvar_per_sys + 0] = aa_r[id];
    sndbuf[id * nvar_per_sys + 1] = cc_r[id];
    sndbuf[id * nvar_per_sys + 2] = dd_r[id];
  }
  int P = ceil(log2((double)params.num_mpi_procs[solvedim]));
  int s = 1;
  // Second step perform pcr
  // loop for for comm
  for (int p = 0; p < P; ++p) {
    // rank diff to communicate with
    int leftrank        = rank - s;
    int rightrank       = rank + s;
    MPI_Request reqs[4] = {
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
    };
    // Get the minus elements
    if (leftrank >= 0) {
      // send recv
      MPI_Isend(sndbuf, n_sys * nvar_per_sys, MPI_DATATYPE(REAL), leftrank, tag,
                params.communicators[solvedim], &reqs[0]);
      MPI_Irecv(rcvbufL, n_sys * nvar_per_sys, MPI_DATATYPE(REAL), leftrank,
                tag, params.communicators[solvedim], &reqs[1]);
    }

    // Get the plus elements
    if (rightrank < nproc) {
      // send recv
      MPI_Isend(sndbuf, n_sys * nvar_per_sys, MPI_DATATYPE(REAL), rightrank,
                tag, params.communicators[solvedim], &reqs[2]);
      MPI_Irecv(rcvbufR, n_sys * nvar_per_sys, MPI_DATATYPE(REAL), rightrank,
                tag, params.communicators[solvedim], &reqs[3]);
    }

    // Wait for communication to finish
    MPI_Waitall(4, reqs, MPI_STATUS_IGNORE);

    // PCR algorithm
#pragma omp parallel for
    for (int id = 0; id < n_sys; id++) {
      // \begin{equation}
      //   \left[\begin{array}{cccccc|c}
      //      a_{-s} &  1  & c_{-s} &        &     &        &  d_{-s} \\\hline
      //             & a_0 &  1     & c_0    &     &        &  d_0    \\\hline
      //             &     & a_s    &  1     & c_s &        &  d_s
      //   \end{array}\right]
      //   \Rightarrow
      //   \left[\begin{array}{cccccc|c}
      //      a_{-s} &  1  & c_{-s} &        &       &        &  d_{-s} \\\hline
      //       a_0^* &     &  1     &        & c_0^* &        &  d_0^*  \\\hline
      //             &     & a_s    &  1     &  c_s  &        &  d_s
      //   \end{array}\right]
      // \end{equation}
      REAL am1 = 0.0;
      REAL cm1 = 0.0;
      REAL dm1 = 0.0;
      REAL ap1 = 0.0;
      REAL cp1 = 0.0;
      REAL dp1 = 0.0;
      if (leftrank >= 0) {
        am1 = rcvbufL[id * nvar_per_sys];
        cm1 = rcvbufL[id * nvar_per_sys + 1];
        dm1 = rcvbufL[id * nvar_per_sys + 2];
      }
      if (rightrank < nproc) {
        ap1 = rcvbufR[id * nvar_per_sys];
        cp1 = rcvbufR[id * nvar_per_sys + 1];
        dp1 = rcvbufR[id * nvar_per_sys + 2];
      }
      REAL bbi = 1 / (1 - aa_r[id] * cm1 - cc_r[id] * ap1);
      dd_r[id] = (dd_r[id] - dm1 * aa_r[id] - dp1 * cc_r[id]) * bbi;
      aa_r[id] = -am1 * aa_r[id] * bbi;
      cc_r[id] = -cp1 * cc_r[id] * bbi;
      sndbuf[id * nvar_per_sys + 0] = aa_r[id];
      sndbuf[id * nvar_per_sys + 1] = cc_r[id];
      sndbuf[id * nvar_per_sys + 2] = dd_r[id];
    }

    // done
    s = s << 1;
  }

  // Last step: substitute back the solution for both row
  compute_last_for_reduced(params, aa, cc, dd, dd_r, rcvbuf, dims, pads, ndim,
                           solvedim, 0, n_sys, result_stride);
}

// Positive and negative padding version
// TODO generalize padding caclulations (not just hardcode the 3D case)
template <typename REAL, int INC>
inline void backward(const REAL *aa, const REAL *cc, const REAL *dd, REAL *d,
                     REAL *u, const int *dims, const int *pads, int ndim,
                     int solvedim, int n_sys) {

  if (solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/

    if (ndim == 1) {
      // Do the backward pass to solve for remaining unknowns

#pragma omp parallel for
      for (int id = 0; id < n_sys; id++) {
        int ind = id * pads[0];
        // int ind = id * pads[0];
        thomas_backward<REAL, INC>(&aa[ind], &cc[ind], &dd[ind], &d[ind],
                                   &u[ind], dims[0], 1);
      }
    } else {
      // Do the backward pass to solve for remaining unknowns

#pragma omp parallel for
      for (int id = 0; id < n_sys; id++) {
        int ind = (id / dims[1]) * pads[1] * pads[0] + (id % dims[1]) * pads[0];
        // int ind = id * pads[0];
        thomas_backward<REAL, INC>(&aa[ind], &cc[ind], &dd[ind], &d[ind],
                                   &u[ind], dims[0], 1);
      }
    }
  } else if (solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/
    // This dimension should not need to be altered for positive and negative
    // padding

    // Check if 2D solve
    if (ndim == 2) {
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
                                             dims[0]);
      }
    }
  } else if (solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/

#pragma omp parallel for
    for (int ind = 0; ind < dims[1] * pads[0]; ind += pads[0]) {
      thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                           &d[ind], &u[ind], dims[2],
                                           pads[0] * pads[1], dims[0]);
    }
  }
}

template <typename REAL, int INC>
inline void backward_batched(const REAL *aa, const REAL *cc, const REAL *dd,
                             REAL *d, REAL *u, const int *dims, const int *pads,
                             int ndim, int solvedim, int start_sys,
                             int end_sys) {
  int n_sys = end_sys - start_sys;
  if (solvedim == 0) {
    /*********************
     *
     * X Dimension Solve
     *
     *********************/

    if (ndim == 1) {
      // Do the backward pass to solve for remaining unknowns

#pragma omp parallel for
      for (int id = start_sys; id < end_sys; id++) {
        int ind = id * pads[0];
        // int ind = id * pads[0];
        thomas_backward<REAL, INC>(&aa[ind], &cc[ind], &dd[ind], &d[ind],
                                   &u[ind], dims[0], 1);
      }
    } else {
      // Do the backward pass to solve for remaining unknowns

#pragma omp parallel for
      for (int id = start_sys; id < end_sys; id++) {
        int ind = (id / dims[1]) * pads[1] * pads[0] + (id % dims[1]) * pads[0];
        // int ind = id * pads[0];
        thomas_backward<REAL, INC>(&aa[ind], &cc[ind], &dd[ind], &d[ind],
                                   &u[ind], dims[0], 1);
      }
    }
  } else if (solvedim == 1) {
    /*********************
     *
     * Y Dimension Solve
     *
     *********************/

    // Check if 2D solve
    if (ndim == 2) {
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
      int end_z   = end_sys / dims[0];
#pragma omp parallel for
      for (int z = start_z; z <= end_z; z++) {
        int ind = z * pads[0] * pads[1];
        // offset in the first z 'layer'
        int strip_start = z == start_z ? start_sys - start_z * dims[0] : 0;
        int strip_len   = z == end_z ? end_sys - end_z * dims[0] : dims[0];
        strip_len -= strip_start;
        // ind += strip_start * pads[0];
        ind += strip_start;
        thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                             &d[ind], &u[ind], dims[1], pads[0],
                                             strip_len);
      }
    }
  } else if (solvedim == 2) {
    /*********************
     *
     * Z Dimension Solve
     *
     *********************/

    int start_y = start_sys / dims[0];
    int end_y   = end_sys / dims[0];
#pragma omp parallel for
    for (int y = start_y; y <= end_y; y++) {
      int ind         = y * pads[0];
      int strip_start = y == start_y ? start_sys - start_y * dims[0] : 0;
      int strip_len   = y == end_y ? end_sys - end_y * dims[0] : dims[0];
      strip_len -= strip_start;
      // ind += strip_start * pads[0];
      ind += strip_start;
      thomas_backward_vec_strip<REAL, INC>(&aa[ind], &cc[ind], &dd[ind],
                                           &d[ind], &u[ind], dims[2],
                                           pads[0] * pads[1], strip_len);
    }
  }
}

// LH2
template <typename REAL, int INC>
inline void tridMultiDimBatchSolve_simple(
    const MpiSolverParams &params, const REAL *a, const REAL *b, const REAL *c,
    REAL *d, REAL *u, REAL *aa, REAL *cc, REAL *dd, int ndim, int solvedim,
    const int *dims, const int *pads, REAL *sndbuf, REAL *rcvbuf, REAL *aa_r,
    REAL *cc_r, REAL *dd_r, int len_r_local, int sys_len_r, int n_sys) {
  int batch_size  = std::min(n_sys, params.mpi_batch_size);
  int num_batches = 1 + (n_sys - 1) / batch_size;
  std::vector<MPI_Request> requests(num_batches);

  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? n_sys - batch_start : batch_size;
    BEGIN_PROFILING("forward");
    forward_batched(a, b, c, d, aa, cc, dd, dims, pads, ndim, solvedim,
                    batch_start, batch_start + bsize);
    // Pack reduced systems (boundaries of each tridiagonal system)
    copy_boundaries_strided(aa, cc, dd, sndbuf, dims, pads, ndim, solvedim,
                            batch_start, batch_start + bsize);
    END_PROFILING("forward");
    BEGIN_PROFILING("mpi_communication");
    // Send boundaries of the current batch
    size_t comm_size          = len_r_local * bsize;
    size_t buffer_offset      = len_r_local * batch_start;
    size_t recv_buffer_offset = 3 * sys_len_r * batch_start;
    // Communicate reduced systems
    MPI_Iallgather(sndbuf + buffer_offset, comm_size, MPI_DATATYPE(REAL),
                   rcvbuf + recv_buffer_offset, comm_size, MPI_DATATYPE(REAL),
                   params.communicators[solvedim], &requests[bidx]);
    END_PROFILING("mpi_communication");
  }
  // start processing messages
  for (int finished_batches = 0; finished_batches < num_batches;
       ++finished_batches) {
    // wait for a MPI transaction to finish
    int bidx;
    BEGIN_PROFILING("mpi_communication");
    int rc =
        MPI_Waitany(requests.size(), requests.data(), &bidx, MPI_STATUS_IGNORE);
    assert(rc == MPI_SUCCESS && "error MPI communication failed");
    END_PROFILING("mpi_communication");

    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? n_sys - batch_start : batch_size;
    // Finish the solve for batch
    BEGIN_PROFILING("reduced");
    // Solve reduced systems on each node
    solve_reduced_batched(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads,
                          ndim, solvedim, sys_len_r, batch_start,
                          batch_start + bsize);
    END_PROFILING("reduced");
    // Perform the backward run of the modified thomas algorithm
    BEGIN_PROFILING("backward");
    backward_batched<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim,
                                batch_start, batch_start + bsize);
    END_PROFILING("backward");
  }
}
// LH
template <typename REAL, int INC>
inline void tridMultiDimBatchSolve_LH(
    const MpiSolverParams &params, const REAL *a, const REAL *b, const REAL *c,
    REAL *d, REAL *u, REAL *aa, REAL *cc, REAL *dd, int ndim, int solvedim,
    const int *dims, const int *pads, REAL *sndbuf, REAL *rcvbuf, REAL *aa_r,
    REAL *cc_r, REAL *dd_r, int len_r_local, int sys_len_r, int n_sys) {
  int batch_size  = std::min(n_sys, params.mpi_batch_size);
  int num_batches = 1 + (n_sys - 1) / batch_size;
  MPI_Request request;

  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? n_sys - batch_start : batch_size;
    BEGIN_PROFILING("forward");
    forward_batched(a, b, c, d, aa, cc, dd, dims, pads, ndim, solvedim,
                    batch_start, batch_start + bsize);
    // Pack reduced systems (boundaries of each tridiagonal system)
    copy_boundaries_strided(aa, cc, dd, sndbuf, dims, pads, ndim, solvedim,
                            batch_start, batch_start + bsize);
    END_PROFILING("forward");
    BEGIN_PROFILING("mpi_communication");
    // wait for the previous MPI transaction to finish
    if (bidx != 0) {
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    // Send boundaries of the current batch
    size_t comm_size          = len_r_local * bsize;
    size_t buffer_offset      = len_r_local * batch_start;
    size_t recv_buffer_offset = 3 * sys_len_r * batch_start;
    // Communicate reduced systems
    MPI_Iallgather(sndbuf + buffer_offset, comm_size, MPI_DATATYPE(REAL),
                   rcvbuf + recv_buffer_offset, comm_size, MPI_DATATYPE(REAL),
                   params.communicators[solvedim], &request);
    END_PROFILING("mpi_communication");
    // Finish the previous batch
    if (bidx != 0) {
      int batch_start = (bidx - 1) * batch_size;
      int bsize       = batch_size;
      BEGIN_PROFILING("reduced");
      // Solve reduced systems on each node
      solve_reduced_batched(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads,
                            ndim, solvedim, sys_len_r, batch_start,
                            batch_start + bsize);
      END_PROFILING("reduced");
      BEGIN_PROFILING("backward");
      backward_batched<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim,
                                  batch_start, batch_start + bsize);
      END_PROFILING("backward");
    }
  }
  // wait for last message and finish last batch
  BEGIN_PROFILING("mpi_communication");
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  END_PROFILING("mpi_communication");
  // Finish the previous batch
  int batch_start = (num_batches - 1) * batch_size;
  int bsize       = n_sys - batch_start;
  BEGIN_PROFILING("reduced");
  // Solve reduced systems on each node
  solve_reduced_batched(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads, ndim,
                        solvedim, sys_len_r, batch_start, batch_start + bsize);
  END_PROFILING("reduced");
  BEGIN_PROFILING("backward");
  backward_batched<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim,
                              batch_start, batch_start + bsize);
  END_PROFILING("backward");
}


template <typename REAL, int INC>
inline void tridMultiDimBatchSolve_allgather(
    const MpiSolverParams &params, const REAL *a, const REAL *b, const REAL *c,
    REAL *d, REAL *u, REAL *aa, REAL *cc, REAL *dd, int ndim, int solvedim,
    const int *dims, const int *pads, REAL *sndbuf, REAL *rcvbuf, REAL *aa_r,
    REAL *cc_r, REAL *dd_r, int len_r_local, int sys_len_r, int n_sys) {
  BEGIN_PROFILING("forward");
  forward(a, b, c, d, aa, cc, dd, dims, pads, ndim, solvedim, n_sys);
  // Pack reduced systems (boundaries of each tridiagonal system)
  copy_boundaries_strided(aa, cc, dd, sndbuf, dims, pads, ndim, solvedim, 0,
                          n_sys);
  END_PROFILING("forward");
  BEGIN_PROFILING("mpi_communication");
  // Communicate reduced systems
  MPI_Allgather(sndbuf, n_sys * len_r_local, MPI_DATATYPE(REAL), rcvbuf,
                n_sys * len_r_local, MPI_DATATYPE(REAL),
                params.communicators[solvedim]);

  END_PROFILING("mpi_communication");
  BEGIN_PROFILING("reduced");
  // Solve reduced systems on each node
  solve_reduced(params, rcvbuf, aa_r, cc_r, dd_r, dd, dims, pads, ndim,
                solvedim, sys_len_r, n_sys);
  END_PROFILING("reduced");
  BEGIN_PROFILING("backward");
  backward<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("backward");
}

template <typename REAL, int INC>
inline void tridMultiDimBatchSolve_gather_scatter(
    const MpiSolverParams &params, const REAL *a, const REAL *b, const REAL *c,
    REAL *d, REAL *u, REAL *aa, REAL *cc, REAL *dd, int ndim, int solvedim,
    const int *dims, const int *pads, REAL *sndbuf, REAL *rcvbuf, REAL *aa_r,
    REAL *cc_r, REAL *dd_r, int len_r_local, int sys_len_r, int n_sys) {

  BEGIN_PROFILING("forward");
  forward(a, b, c, d, aa, cc, dd, dims, pads, ndim, solvedim, n_sys);
  // Pack reduced systems (boundaries of each tridiagonal system)
  copy_boundaries_strided(aa, cc, dd, sndbuf, dims, pads, ndim, solvedim, 0,
                          n_sys);
  END_PROFILING("forward");
  BEGIN_PROFILING("mpi_communication");

  // Communicate reduced systems
  MPI_Gather(sndbuf, n_sys * len_r_local, MPI_DATATYPE(REAL), rcvbuf,
             n_sys * len_r_local, MPI_DATATYPE(REAL), 0,
             params.communicators[solvedim]);

  END_PROFILING("mpi_communication");
  BEGIN_PROFILING("reduced");

  // Solve reduced system on root nodes of this dimension
  if (params.mpi_coords[solvedim] == 0) {
    // Iterate over each reduced system
    for (int id = 0; id < n_sys; id++) {
      // Unpack this reduced system from receive buffer
      for (int p = 0; p < params.num_mpi_procs[solvedim]; p++) {
        int buf_ind     = p * n_sys * 2 * 3;
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
      for (int p = 0; p < params.num_mpi_procs[solvedim]; p++) {
        int buf_ind                  = p * n_sys * 2;
        sndbuf[buf_ind + id * 2]     = dd_r[p * 2];
        sndbuf[buf_ind + id * 2 + 1] = dd_r[p * 2 + 1];
      }
    }
  }

  END_PROFILING("reduced");
  BEGIN_PROFILING("mpi_communication");

  // Send back new values from reduced solve
  MPI_Scatter(sndbuf, n_sys * 2, MPI_DATATYPE(REAL), rcvbuf, n_sys * 2,
              MPI_DATATYPE(REAL), 0, params.communicators[solvedim]);

  END_PROFILING("mpi_communication");
  BEGIN_PROFILING("backward");
  // Unpack reduced solution

  int result_stride = get_sys_span(solvedim, dims, pads);


#pragma omp parallel for
  for (int id = 0; id < n_sys; id++) {
    // Gather coefficients of d
    int data_ind = get_sys_start_idx(id, solvedim, dims, pads, ndim);
    int buf_ind  = id * 2;
    dd[data_ind] = rcvbuf[buf_ind];
    dd[data_ind + result_stride] = rcvbuf[buf_ind + 1];
  }

  backward<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim, n_sys);

  END_PROFILING("backward");
}

template <typename REAL, int INC>
inline void tridMultiDimBatchSolve_jacobi(
    const MpiSolverParams &params, const REAL *a, const REAL *b, const REAL *c,
    REAL *d, REAL *u, REAL *aa, REAL *cc, REAL *dd, int ndim, int solvedim,
    const int *dims, const int *pads, REAL *sndbuf, REAL *rcvbuf, REAL *aa_r,
    REAL *cc_r, REAL *dd_r, int len_r_local, int sys_len_r, int n_sys) {
  (void)aa_r;
  (void)cc_r;
  (void)len_r_local;
  (void)sys_len_r;
  BEGIN_PROFILING("forward");
  forward(a, b, c, d, aa, cc, dd, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("forward");
  BEGIN_PROFILING("reduced");
  // Solve reduced systems on each node
  solve_reduced_jacobi(params, aa, cc, dd, dd_r, rcvbuf, sndbuf, dims, pads,
                       ndim, solvedim, n_sys);
  END_PROFILING("reduced");
  BEGIN_PROFILING("backward");
  backward<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("backward");
}

template <typename REAL, int INC>
inline void tridMultiDimBatchSolve_pcr(
    const MpiSolverParams &params, const REAL *a, const REAL *b, const REAL *c,
    REAL *d, REAL *u, REAL *aa, REAL *cc, REAL *dd, int ndim, int solvedim,
    const int *dims, const int *pads, REAL *sndbuf, REAL *rcvbuf, REAL *aa_r,
    REAL *cc_r, REAL *dd_r, int len_r_local, int sys_len_r, int n_sys) {
  (void)len_r_local;
  (void)sys_len_r;
  BEGIN_PROFILING("forward");
  forward(a, b, c, d, aa, cc, dd, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("forward");
  BEGIN_PROFILING("reduced");
  // Solve reduced systems on each node
  solve_reduced_pcr(params, aa, cc, dd, aa_r, cc_r, dd_r, rcvbuf, sndbuf, dims,
                    pads, ndim, solvedim, n_sys);
  END_PROFILING("reduced");
  BEGIN_PROFILING("backward");
  backward<REAL, INC>(aa, cc, dd, d, u, dims, pads, ndim, solvedim, n_sys);
  END_PROFILING("backward");
}

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
  REAL *sndbuf = (REAL *)_mm_malloc(
      MAX(n_sys * len_r_local, n_sys * sys_len_r) * sizeof(REAL), SIMD_WIDTH);
  REAL *rcvbuf =
      (REAL *)_mm_malloc(n_sys * sys_len_r * 3 * sizeof(REAL), SIMD_WIDTH);

  // Allocate memory for reduced solve
  REAL *aa_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);
  REAL *cc_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);
  REAL *dd_r = (REAL *)_mm_malloc(sizeof(REAL) * sys_len_r * n_sys, SIMD_WIDTH);

  END_PROFILING("memalloc");

  switch (params.strategy) {
  case MpiSolverParams::GATHER_SCATTER:
    tridMultiDimBatchSolve_gather_scatter<REAL, INC>(
        params, a, b, c, d, u, aa, cc, dd, ndim, solvedim, dims, pads, sndbuf,
        rcvbuf, aa_r, cc_r, dd_r, len_r_local, sys_len_r, n_sys);
    break;
  case MpiSolverParams::ALLGATHER:
    tridMultiDimBatchSolve_allgather<REAL, INC>(
        params, a, b, c, d, u, aa, cc, dd, ndim, solvedim, dims, pads, sndbuf,
        rcvbuf, aa_r, cc_r, dd_r, len_r_local, sys_len_r, n_sys);
    break;
  case MpiSolverParams::JACOBI:
    tridMultiDimBatchSolve_jacobi<REAL, INC>(
        params, a, b, c, d, u, aa, cc, dd, ndim, solvedim, dims, pads, sndbuf,
        rcvbuf, aa_r, cc_r, dd_r, len_r_local, sys_len_r, n_sys);
    break;
  case MpiSolverParams::PCR:
    tridMultiDimBatchSolve_pcr<REAL, INC>(
        params, a, b, c, d, u, aa, cc, dd, ndim, solvedim, dims, pads, sndbuf,
        rcvbuf, aa_r, cc_r, dd_r, len_r_local, sys_len_r, n_sys);
    break;
  case MpiSolverParams::LATENCY_HIDING_INTERLEAVED:
    tridMultiDimBatchSolve_LH<REAL, INC>(
        params, a, b, c, d, u, aa, cc, dd, ndim, solvedim, dims, pads, sndbuf,
        rcvbuf, aa_r, cc_r, dd_r, len_r_local, sys_len_r, n_sys);
    break;
  case MpiSolverParams::LATENCY_HIDING_TWO_STEP:
    tridMultiDimBatchSolve_simple<REAL, INC>(
        params, a, b, c, d, u, aa, cc, dd, ndim, solvedim, dims, pads, sndbuf,
        rcvbuf, aa_r, cc_r, dd_r, len_r_local, sys_len_r, n_sys);
    break;
  default: assert(false && "Unknown communication strategy");
  }

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
// specifies any padding used in the arrays (the total length of each
// dimension including padding).
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
