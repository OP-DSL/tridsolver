#ifndef TRID_ITERATIVE_MPI_HPP_INCLUDED
#define TRID_ITERATIVE_MPI_HPP_INCLUDED
#include "trid_mpi_solver_params.hpp"
#include "cuda_timing.h"
#include <cassert>

#include "trid_strided_multidim_mpi.hpp"
#include "trid_mpi_helper.hpp"
#include "trid_mpi_common.hpp"

/*
 * Modified forwards pass in x direction
 * Each array should have a size of sys_size*sys_n, although the first element
 * of a (a[0]) in the first process and the last element of c in the last
 * process will not be used eventually
 */
template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
__global__ void
trid_linear_forward_pass(const REAL *__restrict__ a, const REAL *__restrict__ b,
                         const REAL *__restrict__ c, REAL *__restrict__ d,
                         REAL *__restrict__ aa, REAL *__restrict__ cc,
                         REAL *__restrict__ boundaries, int sys_size,
                         int sys_pads, int sys_n, int rank, int nproc) {

  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // forward pass
    //
    // i = 0 save and perform at last step
    REAL b_0 = b[ind];
    REAL c_0 = c[ind];
    REAL d_0 = d[ind];
    { // i = 1
      if (rank == 0) {
        REAL factor = a[ind + 1] / b_0;
        REAL bb     = 1 / (b[ind + 1] - factor * c_0);
        d[ind + 1]  = bb * (d[ind + 1] - factor * d_0);
        cc[ind + 1] = bb * c[ind + 1];
        aa[ind + 1] = 0;
        if (shift_c0_on_rank0) {
          d_0 = d_0 - c_0 * d[ind + 1];
          c_0 = -c_0 * cc[ind + 1];
        }
      } else {
        d[ind + 1]  = d[ind + 1] / b[ind + 1];
        cc[ind + 1] = c[ind + 1] / b[ind + 1];
        aa[ind + 1] = a[ind + 1] / b[ind + 1];
        b_0         = b_0 - c_0 * aa[ind + 1];
        d_0         = d_0 - c_0 * d[ind + 1];
        if (rank == nproc - 1 && sys_size == 2) {
          c_0 = 0;
        } else {
          c_0 = -c_0 * cc[ind + 1];
        }
      }
    }

    // eliminate lower off-diagonal
    for (int i = 2; i < sys_size; i++) {
      int loc_ind = ind + i;
      REAL bb =
          static_cast<REAL>(1.0) / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
      d[loc_ind] = (d[loc_ind] - a[loc_ind] * d[loc_ind - 1]) * bb;
      if (i == sys_size - 1 && rank == nproc - 1) {
        cc[loc_ind] = 0;
      } else {
        cc[loc_ind] = c[loc_ind] * bb;
      }
      if (rank == 0) {
        aa[loc_ind] = 0;
        if (shift_c0_on_rank0) {
          d_0 = d_0 - c_0 * d[loc_ind];
          c_0 = -c_0 * cc[loc_ind];
        }
      } else {
        aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
        b_0         = b_0 - c_0 * aa[loc_ind];
        d_0         = d_0 - c_0 * d[loc_ind];
        c_0         = -c_0 * cc[loc_ind];
      }
    }
    // i = 0
    if (0 == rank) {
      aa[ind] = 0;
    } else {
      aa[ind] = a[ind] / b_0;
    }
    cc[ind] = c_0 / b_0;
    d[ind]  = d_0 / b_0;
    // prepare boundaries for communication
    copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid, ind,
                                               sys_size, sys_n);
  }
}

/*
 * Modified forward pass in y or higher dimensions.
 * Each array should have a size of sys_n*sys_size.
 * The layout and indexing of aa, cc, and dd are the same as of a, c, d
 * respectively
 * The boundaries array has a size of sys_n*6 and will hold the first and last
 * elements of aa, cc, and dd for each system
 *
 */
template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
__device__ void trid_strided_multidim_forward_pass_kernel(
    const REAL *__restrict__ a, int ind_a, int stride_a,
    const REAL *__restrict__ b, int ind_b, int stride_b,
    const REAL *__restrict__ c, int ind_c, int stride_c, REAL *__restrict__ d,
    int ind_d, int stride_d, REAL *__restrict__ aa, REAL *__restrict__ cc,
    REAL *__restrict__ boundaries, int tid, int sys_size, int sys_n, int rank,
    int nproc) {
  //
  // forward pass
  //
  // i = 0
  REAL b_0 = b[ind_b];
  REAL c_0 = c[ind_c];
  REAL d_0 = d[ind_d];
  // i = 1
  {
    if (rank == 0) {
      REAL factor          = a[ind_a + stride_a] / b_0;
      REAL bb              = 1 / (b[ind_b + stride_b] - factor * c_0);
      d[ind_d + stride_d]  = bb * (d[ind_d + stride_d] - factor * d_0);
      cc[ind_c + stride_c] = bb * c[ind_c + stride_c];
      aa[ind_a + stride_a] = 0;
      if (shift_c0_on_rank0) {
        d_0 = d_0 - c_0 * d[ind_d + stride_d];
        c_0 = -c_0 * cc[ind_c + stride_c];
      }
    } else {
      d[ind_d + stride_d]  = d[ind_d + stride_d] / b[ind_b + stride_b];
      cc[ind_c + stride_c] = c[ind_c + stride_c] / b[ind_b + stride_b];
      aa[ind_a + stride_a] = a[ind_a + stride_a] / b[ind_b + stride_b];
      b_0                  = b_0 - c_0 * aa[ind_a + stride_a];
      d_0                  = d_0 - c_0 * d[ind_d + stride_d];
      if (rank == nproc - 1 && sys_size == 2) {
        c_0 = 0;
      } else {
        c_0 = -c_0 * cc[ind_c + stride_c];
      }
    }
  }

  // Eliminate lower off-diagonal
  for (int i = 2; i < sys_size; ++i) {
    REAL bb = static_cast<REAL>(1.0) /
              (b[ind_b + i * stride_b] -
               a[ind_a + i * stride_a] * cc[ind_c + (i - 1) * stride_c]);
    d[ind_d + i * stride_d] =
        (d[ind_d + i * stride_d] -
         a[ind_a + i * stride_a] * d[ind_d + (i - 1) * stride_d]) *
        bb;
    if (i == sys_size - 1 && rank == nproc - 1) {
      cc[ind_c + i * stride_c] = 0.0;
    } else {
      cc[ind_c + i * stride_c] = c[ind_c + i * stride_c] * bb;
    }
    if (rank == 0) {
      aa[ind_a + i * stride_a] = 0;
      if (shift_c0_on_rank0) {
        d_0 = d_0 - c_0 * d[ind_d + i * stride_d];
        c_0 = -c_0 * cc[ind_c + i * stride_c];
      }
    } else {
      aa[ind_a + i * stride_a] =
          (-a[ind_a + i * stride_a] * aa[ind_a + (i - 1) * stride_a]) * bb;
      b_0 = b_0 - c_0 * aa[ind_a + i * stride_a];
      d_0 = d_0 - c_0 * d[ind_d + i * stride_d];
      c_0 = -c_0 * cc[ind_c + i * stride_c];
    }
  }
  // i = 0
  if (0 == rank) {
    aa[ind_a] = 0;
  } else {
    aa[ind_a] = a[ind_a] / b_0;
  }
  cc[ind_c] = c_0 / b_0;
  d[ind_d]  = d_0 / b_0;

  // prepare boundaries for communication
  copy_boundaries_strided<REAL, boundary_SOA>(aa, ind_a, stride_a, cc, ind_c,
                                              stride_c, d, ind_d, stride_d,
                                              boundaries, tid, sys_size, sys_n);
}

template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
__global__ void trid_strided_multidim_forward_pass(
    const REAL *__restrict__ a, const DIM_V a_pads, const REAL *__restrict__ b,
    const DIM_V b_pads, const REAL *__restrict__ c, const DIM_V c_pads,
    REAL *__restrict__ d, const DIM_V d_pads, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ boundaries, int ndim,
    int solvedim, int sys_n, const DIM_V dims, int rank, int nproc,
    int sys_offset = 0) {
  // thread ID in block
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  if (solvedim < 1 || solvedim > ndim) return; /* Just hints to the compiler */

  int __shared__ d_cumdims[MAXDIM + 1];
  int __shared__ d_cumpads[4][MAXDIM + 1];

  /* Build up d_cumpads and d_cumdims */
  if (tid < 5) {
    int *tgt       = (tid == 0) ? d_cumdims : d_cumpads[tid - 1];
    const int *src = NULL;
    switch (tid) {
    case 0: src = dims.v; break;
    case 1: src = a_pads.v; break;
    case 2: src = b_pads.v; break;
    case 3: src = c_pads.v; break;
    case 4: src = d_pads.v; break;
    }

    tgt[0] = 1;
    for (int i = 0; i < ndim; i++) {
      tgt[i + 1] = tgt[i] * src[i];
    }
  }
  __syncthreads();
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  tid = sys_offset + threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  int ind_a = 0;
  int ind_b = 0;
  int ind_c = 0;
  int ind_d = 0;

  for (int j = 0; j < solvedim; j++) {
    ind_a += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[0][j];
    ind_b += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[1][j];
    ind_c += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[2][j];
    ind_d += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[3][j];
  }
  for (int j = solvedim + 1; j < ndim; j++) {
    ind_a += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[0][j];
    ind_b += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[1][j];
    ind_c += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[2][j];
    ind_d += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[3][j];
  }
  int stride_a = d_cumpads[0][solvedim];
  int stride_b = d_cumpads[1][solvedim];
  int stride_c = d_cumpads[2][solvedim];
  int stride_d = d_cumpads[3][solvedim];
  int sys_size = dims.v[solvedim];

  if (tid < sys_offset + sys_n) {
    trid_strided_multidim_forward_pass_kernel<REAL, boundary_SOA,
                                              shift_c0_on_rank0>(
        a, ind_a, stride_a, b, ind_b, stride_b, c, ind_c, stride_c, d, ind_d,
        stride_d, aa, cc, boundaries, tid, sys_size, sys_n, rank, nproc);
  }
}

//
// Modified backward pass
//
template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
__global__ void
trid_linear_backward_pass(const REAL *__restrict__ aa,
                          const REAL *__restrict__ cc, REAL *__restrict__ d,
                          REAL *__restrict__ u,
                          const REAL *__restrict__ boundaries, int sys_size,
                          int sys_pads, int sys_n, int rank, int nproc) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // reverse pass
    //
    REAL dd0, dd_p1;
    load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, dd_p1, tid,
                                                    sys_n);

    // i = n-1
    int loc_ind = ind + sys_size - 1;
    if (rank != nproc - 1)
      dd_p1 = d[loc_ind] - dd_p1 * cc[loc_ind];
    else
      dd_p1 = d[loc_ind];
    if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
    if (INC) {
      u[loc_ind] += dd_p1;
    } else {
      d[loc_ind] = dd_p1;
    }
    // i = n-2 - 1
    for (int i = sys_size - 2; i > 0; --i) {
      loc_ind = ind + i;
      dd_p1   = d[loc_ind] - cc[loc_ind] * dd_p1;
      if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
      if (INC)
        u[ind + i] += dd_p1;
      else
        d[ind + i] = dd_p1;
    }
    // i = 0
    if (0 == rank && !is_c0_cleared_on_rank0) {
      dd_p1 = dd0 - cc[ind] * dd_p1;
      if (INC)
        u[ind] += dd_p1;
      else
        d[ind] = dd_p1;
    } else {
      if (INC)
        u[ind] += dd0;
      else
        d[ind] = dd0;
    }
  }
}

/*
 * Modified backward pass in y or higher dimensions.
 * Each array should have a size of sys_n*sys_size.
 * The layout and indexing of aa, cc, and dd are the same as of a, c, d
 * respectively
 * The boundaries array has a size of sys_n*2 and hold the first element from
 * this process and from the next process of d for each system
 * The layout of boundaries is as if it holds the a and c values of the lines as
 * well.
 *
 */
template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
__device__ void trid_strided_multidim_backward_pass_kernel(
    const REAL *__restrict__ aa, int ind_a, int stride_a,
    const REAL *__restrict__ cc, int ind_c, int stride_c, REAL *__restrict__ d,
    int ind_d, int stride_d, REAL *__restrict__ u, int ind_u, int stride_u,
    const REAL *__restrict__ boundaries, int tid, int sys_size, int sys_n,
    int rank, int nproc) {
  //
  // reverse pass
  //
  REAL dd0, dd_p1;
  load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, dd_p1, tid,
                                                  sys_n);
  // i = n-1
  if (rank != nproc - 1)
    dd_p1 = d[ind_d + (sys_size - 1) * stride_d] -
            dd_p1 * cc[ind_c + (sys_size - 1) * stride_c];
  else
    dd_p1 = d[ind_d + (sys_size - 1) * stride_d];
  if (rank != 0) dd_p1 += -aa[ind_a + (sys_size - 1) * stride_a] * dd0;
  if (INC) {
    u[ind_u + (sys_size - 1) * stride_u] += dd_p1;
  } else {
    d[ind_d + (sys_size - 1) * stride_d] = dd_p1;
  }
  // i = n-2 - 1
  for (int i = sys_size - 2; i > 0; --i) {
    dd_p1 = d[ind_d + i * stride_d] - cc[ind_c + i * stride_c] * dd_p1;
    if (rank != 0) dd_p1 += -aa[ind_a + i * stride_a] * dd0;
    if (INC)
      u[ind_u + i * stride_u] += dd_p1;
    else
      d[ind_d + i * stride_d] = dd_p1;
  }
  // i = 0
  if (0 == rank && !is_c0_cleared_on_rank0) {
    dd_p1 = dd0 - cc[ind_c] * dd_p1;
    if (INC)
      u[ind_u] += dd_p1;
    else
      d[ind_d] = dd_p1;
  } else {
    if (INC)
      u[ind_u] += dd0;
    else
      d[ind_d] = dd0;
  }
}

template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
__global__ void trid_strided_multidim_backward_pass(
    const REAL *__restrict__ aa, const DIM_V a_pads,
    const REAL *__restrict__ cc, const DIM_V c_pads, REAL *__restrict__ d,
    const DIM_V d_pads, REAL *__restrict__ u, const DIM_V u_pads,
    const REAL *__restrict__ boundaries, int ndim, int solvedim, int sys_n,
    const DIM_V dims, int rank, int nproc, int sys_offset = 0) {
  // thread ID in block
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  if (solvedim < 1 || solvedim > ndim) return; /* Just hints to the compiler */

  int __shared__ d_cumdims[MAXDIM + 1];
  int __shared__ d_cumpads[4][MAXDIM + 1];

  /* Build up d_cumpads and d_cumdims */
  if (tid < 5) {
    int *tgt       = (tid == 0) ? d_cumdims : d_cumpads[tid - 1];
    const int *src = NULL;
    switch (tid) {
    case 0: src = dims.v; break;
    case 1: src = a_pads.v; break;
    case 2: src = c_pads.v; break;
    case 3: src = d_pads.v; break;
    case 4: src = u_pads.v; break;
    }

    tgt[0] = 1;
    for (int i = 0; i < ndim; i++) {
      tgt[i + 1] = tgt[i] * src[i];
    }
  }
  __syncthreads();
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  tid = sys_offset + threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  int ind_a = 0;
  int ind_c = 0;
  int ind_d = 0;
  int ind_u = 0;

  for (int j = 0; j < solvedim; j++) {
    ind_a += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[0][j];
    ind_c += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[1][j];
    ind_d += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[2][j];
    if (INC) ind_u += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[3][j];
  }
  for (int j = solvedim + 1; j < ndim; j++) {
    ind_a += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[0][j];
    ind_c += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[1][j];
    ind_d += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[2][j];
    if (INC)
      ind_u += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
               d_cumpads[3][j];
  }
  int stride_a = d_cumpads[0][solvedim];
  int stride_c = d_cumpads[1][solvedim];
  int stride_d = d_cumpads[2][solvedim];
  int stride_u = d_cumpads[3][solvedim];
  int sys_size = dims.v[solvedim];

  if (tid < sys_offset + sys_n) {
    trid_strided_multidim_backward_pass_kernel<REAL, INC, boundary_SOA,
                                               is_c0_cleared_on_rank0>(
        aa, ind_a, stride_a, cc, ind_c, stride_c, d, ind_d, stride_d, u, ind_u,
        stride_u, boundaries, tid, sys_size, sys_n, rank, nproc);
  }
}

//
//  Single PCR iteration for reduced
//  assumption: we only have one row per system in boundaries
//  and 1-1 prev and next row in recv_buf
//
template <typename REAL, bool assume_nonzeros = false>
__global__ void trid_PCR_iteration(REAL *__restrict__ boundaries,
                                   const REAL *__restrict__ recv_m1,
                                   const REAL *__restrict__ recv_p1, int sys_n,
                                   bool m1, bool p1) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = tid;

  if (tid < sys_n) {
    REAL am1 = 0.0;
    REAL cm1 = 0.0;
    REAL dm1 = 0.0;
    REAL ap1 = 0.0;
    REAL cp1 = 0.0;
    REAL dp1 = 0.0;
    if (m1) {
      am1 = recv_m1[0 * sys_n + ind];
      cm1 = recv_m1[1 * sys_n + ind];
      dm1 = recv_m1[2 * sys_n + ind];
    }
    if (p1) {
      ap1 = recv_p1[0 * sys_n + ind];
      cp1 = recv_p1[1 * sys_n + ind];
      dp1 = recv_p1[2 * sys_n + ind];
    }
    REAL bbi = 1.0 / (1 - boundaries[0 * sys_n + ind] * cm1 -
                      boundaries[1 * sys_n + ind] * ap1);
    // d
    boundaries[2 * sys_n + ind] =
        bbi * (boundaries[2 * sys_n + ind] - boundaries[0 * sys_n + ind] * dm1 -
               boundaries[1 * sys_n + ind] * dp1);
    // a
    if (m1)
      boundaries[0 * sys_n + ind] = bbi * -boundaries[0 * sys_n + ind] * am1;
    else if (assume_nonzeros)
      boundaries[0 * sys_n + ind] *= bbi;
    // c
    if (p1)
      boundaries[1 * sys_n + ind] = bbi * -boundaries[1 * sys_n + ind] * cp1;
    else if (assume_nonzeros)
      boundaries[1 * sys_n + ind] *= bbi;
  }
}

////////////////////////////////////////
//        Host functions
////////////////////////////////////////

template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
inline void forward_batched_pass(dim3 dimGrid_x, dim3 dimBlock_x,
                                 const MpiSolverParams &params, const REAL *a,
                                 const int *a_pads, const REAL *b,
                                 const int *b_pads, const REAL *c,
                                 const int *c_pads, REAL *d, const int *d_pads,
                                 REAL *aa, REAL *cc, REAL *boundaries,
                                 REAL *send_buf_h, const int *dims, int ndim,
                                 int solvedim, int start_sys, int bsize,
                                 int offset, cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    assert(offset == 0 && "I think we do not need the offset");
    assert(a_pads[0] == b_pads[0] && a_pads[0] == c_pads[0] &&
           a_pads[0] == d_pads[0] && "different paddings are not supported");
    if (ndim > 1) {
      assert(a_pads[1] == dims[1] && b_pads[1] == dims[1] &&
             c_pads[1] == dims[1] && d_pads[1] == dims[1] &&
             " ONLLY X paddings are supported");
    }
    const int batch_offset = start_sys * a_pads[solvedim];
    trid_linear_forward_pass<REAL, boundary_SOA, shift_c0_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            a + batch_offset, b + batch_offset, c + batch_offset,
            d + batch_offset, aa + batch_offset, cc + batch_offset,
            boundaries + start_sys * 3 * 2, dims[solvedim], a_pads[solvedim],
            bsize, params.mpi_coords[solvedim], params.num_mpi_procs[solvedim]);
  } else {
    DIM_V k_pads, k_dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      k_pads.v[i] = a_pads[i];
      k_dims.v[i] = dims[i];
    }
    trid_strided_multidim_forward_pass<REAL, boundary_SOA, shift_c0_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            a, k_pads, b, k_pads, c, k_pads, d, k_pads, aa, cc, boundaries,
            ndim, solvedim, bsize, k_dims, params.mpi_coords[solvedim],
            params.num_mpi_procs[solvedim], start_sys);
  }
}

template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
inline void backward_batched_pass(dim3 dimGrid_x, dim3 dimBlock_x,
                                  const MpiSolverParams &params, const REAL *aa,
                                  const int *a_pads, const REAL *cc,
                                  const int *c_pads, const REAL *boundaries,
                                  REAL *d, const int *d_pads, REAL *u,
                                  const int *u_pads, const int *dims, int ndim,
                                  int solvedim, int start_sys, int bsize,
                                  int offset, cudaStream_t stream = nullptr) {
  assert(start_sys == 0 &&
         "check the whole process for boundaries if indexing is correct for "
         "batches then remove this");
  if (solvedim == 0) {
    // assert(offset == 0 && "I think we do not need the offset");
    assert(a_pads[0] == c_pads[0] && a_pads[0] == d_pads[0] &&
           "different paddings are not supported");
    if (ndim > 1) {
      assert(a_pads[1] == dims[1] && c_pads[1] == dims[1] &&
             d_pads[1] == dims[1] && " ONLLY X paddings are supported");
    }
    const int batch_offset = start_sys * a_pads[solvedim];
    // int y_size = 1, y_pads = 1;
    // if (ndim > 1) {
    //   y_size = dims[1];
    //   y_pads = a_pads[1];
    // }
    trid_linear_backward_pass<REAL, INC, boundary_SOA, is_c0_cleared_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa + batch_offset, cc + batch_offset, d + batch_offset,
            u + batch_offset, boundaries + start_sys * 2 * 3, dims[solvedim],
            a_pads[solvedim], bsize, params.mpi_coords[solvedim],
            params.num_mpi_procs[solvedim]);
  } else {
    DIM_V k_pads, k_dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      k_pads.v[i] = a_pads[i];
      k_dims.v[i] = dims[i];
    }
    trid_strided_multidim_backward_pass<REAL, INC, boundary_SOA,
                                        is_c0_cleared_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa, k_pads, cc, k_pads, d, k_pads, u, k_pads, boundaries, ndim,
            solvedim, bsize, k_dims, params.mpi_coords[solvedim],
            params.num_mpi_procs[solvedim], start_sys);
  }
}

template <typename REAL, bool snd_down = true, bool snd_up = true>
void trid_cuda_pcr_exchange_line(REAL *snd_d, REAL *snd_h, REAL *rcv_m1_d,
                                 REAL *rcv_m1_h, REAL *rcv_p1_d, REAL *rcv_p1_h,
                                 int line_size, int nproc, int rank_m1,
                                 int rank_p1, const MpiSolverParams &params,
                                 int solvedim, MPI_Request *rcv_requests,
                                 MPI_Request *snd_requests) {
  BEGIN_PROFILING2("mpi_communication");
  static_assert(snd_down || snd_up);
  constexpr int tag = 42;
#ifdef TRID_CUDA_AWARE_MPI
  // Exchange line with upper process
  if (rank_m1 >= 0 && rank_m1 < nproc) {
    if (snd_down)
      MPI_Irecv(rcv_m1_d, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
                params.communicators[solvedim], &rcv_requests[0]);
    if (snd_up)
      MPI_Isend(snd_d, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
                params.communicators[solvedim], &snd_requests[0]);
  }
  // Exchange line with lower process
  if (rank_p1 < nproc && rank_p1 >= 0) {
    if (snd_up)
      MPI_Irecv(rcv_p1_d, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
                params.communicators[solvedim], &rcv_requests[1]);
    if (snd_down)
      MPI_Isend(snd_d, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
                params.communicators[solvedim], &snd_requests[1]);
  }
  MPI_Waitall(2, rcv_requests, MPI_STATUS_IGNORE);
#elif defined(TRID_NCCL)
  NCCLCHECK(ncclGroupStart());
  // Exchange line with upper process
  if (rank_m1 >= 0 && rank_m1 < nproc) {
    if (snd_down)
      NCCLCHECK(ncclRecv(rcv_m1_d, line_size * sizeof(REAL), ncclChar, rank_m1,
                         params.ncclComms[solvedim], 0));
    if (snd_up)
      NCCLCHECK(ncclSend(snd_d, line_size * sizeof(REAL), ncclChar, rank_m1,
                         params.ncclComms[solvedim], 0));
  }
  // Exchange line with lower process
  if (rank_p1 < nproc && rank_p1 >= 0) {
    if (snd_up)
      NCCLCHECK(ncclRecv(rcv_p1_d, line_size * sizeof(REAL), ncclChar, rank_p1,
                         params.ncclComms[solvedim], 0));
    if (snd_down)
      NCCLCHECK(ncclSend(snd_d, line_size * sizeof(REAL), ncclChar, rank_p1,
                         params.ncclComms[solvedim], 0));
  }
  NCCLCHECK(ncclGroupEnd());
  if ((rank_m1 >= 0 && rank_m1 < nproc) || (rank_p1 < nproc && rank_p1 >= 0)) {
    cudaSafeCall(cudaDeviceSynchronize());
  }
#else
  // Exchange line with upper process
  if (snd_down && rank_m1 >= 0 && rank_m1 < nproc) {
    MPI_Irecv(rcv_m1_h, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
              params.communicators[solvedim], &rcv_requests[0]);
  }
  // Exchange line with lower process
  if (snd_up && rank_p1 >= 0 && rank_p1 < nproc) {
    MPI_Irecv(rcv_p1_h, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
              params.communicators[solvedim], &rcv_requests[0]);
  }

  // copy send buffer to host
  if ((snd_up && rank_m1 >= 0 && rank_m1 < nproc) ||
      (snd_down && rank_p1 < nproc && rank_p1 >= 0)) {
    cudaMemcpy(snd_h, snd_d, line_size * sizeof(REAL), cudaMemcpyDeviceToHost);
  }

  // Exchange line with upper process
  if (snd_up && rank_m1 >= 0 && rank_m1 < nproc) {
    MPI_Isend(snd_h, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
              params.communicators[solvedim], &snd_requests[0]);
  }
  // Exchange line with lower process
  if (snd_down && rank_p1 >= 0 && rank_p1 < nproc) {
    MPI_Isend(snd_h, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
              params.communicators[solvedim], &snd_requests[1]);
  }
  MPI_Waitall(2, rcv_requests, MPI_STATUS_IGNORE);
  // Exchange line with upper process
  if (snd_down && rank_m1 >= 0 && rank_m1 < nproc) {
    // copy the received line to device
    cudaMemcpyAsync(rcv_m1_d, rcv_m1_h, line_size * sizeof(REAL),
                    cudaMemcpyHostToDevice);
  }
  // Exchange line with lower process
  if (snd_up && rank_p1 >= 0 && rank_p1 < nproc) {
    // copy the received line to device
    cudaMemcpyAsync(rcv_p1_d, rcv_p1_h, line_size * sizeof(REAL),
                    cudaMemcpyHostToDevice);
  }
#endif
  END_PROFILING2("mpi_communication");
}

// template <typename REAL>
// void print_array(std::string promt, REAL *arr, size_t size,
//                  std::ostream &o = std::cout) {
//   o << promt << ": [";
//   for (size_t i = 0; i < size; ++i) {
//     if (i) o << ", ";
//     o << arr[i];
//   }
//   o << "]\n";
// }

template <typename REAL>
inline void iterative_pcr_on_reduced(dim3 dimGrid_x, dim3 dimBlock_x,
                                     const MpiSolverParams &params,
                                     REAL *boundaries, int sys_n, int solvedim,
                                     REAL *recv_buf_d, REAL *recv_buf_h,
                                     REAL *send_buf_h) {

  MPI_Request rcv_requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  MPI_Request snd_requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  const int rank              = params.mpi_coords[solvedim];
  const int nproc             = params.num_mpi_procs[solvedim];
  constexpr int nvar_per_sys  = 3; // a, c, d
  const int line_size         = sys_n * nvar_per_sys;


  // send bottom line down
  trid_cuda_pcr_exchange_line<REAL, true, false>(
      boundaries + line_size, send_buf_h, recv_buf_d, recv_buf_h, nullptr,
      nullptr, line_size, nproc, rank - 1, rank + 1, params, solvedim,
      rcv_requests, snd_requests);

  // substract received line
  if (rank)
    trid_PCR_iteration<REAL, true><<<dimGrid_x, dimBlock_x>>>(
        boundaries, recv_buf_d, nullptr, sys_n, true, false);
#ifndef TRID_NCCL
  MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
#endif

  // PCR iterations
  int P = (int)ceil(log2((REAL)params.num_mpi_procs[solvedim]));

  for (int p = 0, s = 1; p < P; p++, s <<= 1) {
    int rank_ms = rank - s;
    int rank_ps = rank + s;
    if (rank_ms >= 0 || rank_ps < nproc) {
      // send & recv
      trid_cuda_pcr_exchange_line<REAL>(
          boundaries, send_buf_h, recv_buf_d, recv_buf_h,
          boundaries + line_size, recv_buf_h + line_size, line_size, nproc,
          rank_ms, rank_ps, params, solvedim, rcv_requests, snd_requests);

      // iteration
      trid_PCR_iteration<<<dimGrid_x, dimBlock_x>>>(
          boundaries, recv_buf_d, boundaries + line_size, sys_n, rank_ms >= 0,
          rank_ps < nproc);
#ifndef TRID_NCCL
      MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
#endif
    }
  }
  // send solution up for backward
  trid_cuda_pcr_exchange_line<REAL, false, true>(
      boundaries + 2 * sys_n, send_buf_h, nullptr, nullptr,
      boundaries + 5 * sys_n, recv_buf_h, sys_n, nproc, rank - 1, rank + 1,
      params, solvedim, rcv_requests, snd_requests);
#ifndef TRID_NCCL
  MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
#endif
}

#endif /* ifndef TRID_ITERATIVE_MPI_HPP_INCLUDED */
