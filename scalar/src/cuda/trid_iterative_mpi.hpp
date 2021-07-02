#ifndef TRID_ITERATIVE_MPI_HPP_INCLUDED
#define TRID_ITERATIVE_MPI_HPP_INCLUDED
#include "trid_mpi_solver_params.hpp"
#include "trid_linear_reg_common.hpp"
#include "cuda_timing.h"
#include <cassert>

#include "trid_strided_multidim_mpi.hpp"
#include "trid_mpi_helper.hpp"
#include "trid_mpi_common.hpp"

template <typename REAL, bool shift_c0_on_rank0>
inline __device__ void
forward_linear_process_line(const REAL a, REAL &a_m1, const REAL b,
                            const REAL c, REAL &c_m1, const REAL d, REAL &d_m1,
                            REAL &b_0, REAL &c_0, REAL &d_0, int sys_size,
                            int row_idx, int rank, int nproc) {
  REAL bb = 1.0 / (b - a * c_m1);
  d_m1    = (d - a * d_m1) * bb;
  if (row_idx == sys_size - 1 && rank == nproc - 1) {
    c_m1 = 0;
  } else {
    c_m1 = c * bb;
  }
  if (rank == 0) {
    a_m1 = 0;
    if (shift_c0_on_rank0) {
      d_0 = d_0 - c_0 * d_m1;
      c_0 = -c_0 * c_m1;
    }
  } else {
    a_m1 = (-a * a_m1) * bb;
    b_0  = b_0 - c_0 * a_m1;
    d_0  = d_0 - c_0 * d_m1;
    c_0  = -c_0 * c_m1;
  }
}

template <typename REAL, bool shift_c0_on_rank0>
inline __device__ void
forward_linear_process_row1(const REAL a, REAL &a_m1, const REAL b,
                            const REAL c, REAL &c_m1, const REAL d, REAL &d_m1,
                            REAL &b_0, REAL &c_0, REAL &d_0, int sys_size,
                            int rank, int nproc) {
  if (rank == 0) {
    REAL factor = a / b_0;
    REAL bb     = 1 / (b - factor * c_0);
    d_m1        = bb * (d - factor * d_0);
    c_m1        = bb * c;
    a_m1        = 0;
    if (shift_c0_on_rank0) {
      d_0 = d_0 - c_0 * d_m1;
      c_0 = -c_0 * c_m1;
    }
  } else {
    d_m1 = d / b;
    c_m1 = c / b;
    a_m1 = a / b;
    b_0  = b_0 - c_0 * a_m1;
    d_0  = d_0 - c_0 * d_m1;
    if (rank == nproc - 1 && sys_size == 2) {
      c_0 = 0;
    } else {
      c_0 = -c_0 * c_m1;
    }
  }
}

template <typename REAL, bool shift_c0_on_rank0 = true>
inline __device__ void trid_linear_forward_single_system(
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, int sys_size, int sys_start, int rank, int nproc) {
  REAL a_m1 = 0.0;
  REAL c_m1 = 0.0;
  REAL d_m1 = 0.0;
  //
  // forward pass
  //
  // i = 0 save and perform at last step
  REAL b_0 = b[sys_start];
  REAL c_0 = c[sys_start];
  REAL d_0 = d[sys_start];
  // i = 1
  forward_linear_process_row1<REAL, shift_c0_on_rank0>(
      a[sys_start + 1], a_m1, b[sys_start + 1], c[sys_start + 1], c_m1,
      d[sys_start + 1], d_m1, b_0, c_0, d_0, sys_size, rank, nproc);
  aa[sys_start + 1] = a_m1;
  cc[sys_start + 1] = c_m1;
  d[sys_start + 1]  = d_m1;

  // eliminate lower off-diagonal
  for (int i = 2; i < sys_size; i++) {
    int loc_ind = sys_start + i;
    forward_linear_process_line<REAL, shift_c0_on_rank0>(
        a[loc_ind], a_m1, b[loc_ind], c[loc_ind], c_m1, d[loc_ind], d_m1, b_0,
        c_0, d_0, sys_size, i, rank, nproc);
    aa[loc_ind] = a_m1;
    cc[loc_ind] = c_m1;
    d[loc_ind]  = d_m1;
  }
  // i = 0
  if (0 == rank) {
    aa[sys_start] = 0;
  } else {
    aa[sys_start] = a[sys_start] / b_0;
  }
  cc[sys_start] = c_0 / b_0;
  d[sys_start]  = d_0 / b_0;
}


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
    trid_linear_forward_single_system<REAL, shift_c0_on_rank0>(
        a, b, c, d, aa, cc, sys_size, ind, rank, nproc);
    // prepare boundaries for communication
    copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid, ind,
                                               sys_size, sys_n);
  }
}

// Modified forward pass for X dimension.
// Uses register shuffle optimization, can handle both aligned and unaligned
// memory
template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
__global__ void trid_linear_forward_pass_aligned(
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ boundaries, int sys_size,
    int sys_pads, int sys_n, int rank, int nproc) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // Warp ID in global scope - the ID wich the thread belongs to
  const int wid = tid / WARP_SIZE;
  // Global memory offset: unique to a warp;
  // every thread in a warp calculates the same woffset, which is the "begining"
  // of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid / 4) * 4 + 4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global
  // memory read/write
  const int boundary_solve = !optimized_solve && (tid < (sys_n));
  // A thread is active only if it works on valid memory
  const int active_thread = optimized_solve || boundary_solve;

  // Start index for this tridiagonal system
  int ind = sys_pads * tid;


  // Check that this is an active thread
  if (active_thread) {
    // Check that this thread can perform an optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL)) {
      // Local arrays used in the register shuffle
      vec_line_t<REAL> l_a, l_b, l_c, l_d, l_aa, l_cc;
      REAL a_m1, c_m1, d_m1;
      int n = 0;
      // Process first vector separately
      load_array_reg(a, &l_a, n, woffset, sys_pads);
      load_array_reg(b, &l_b, n, woffset, sys_pads);
      load_array_reg(c, &l_c, n, woffset, sys_pads);
      load_array_reg(d, &l_d, n, woffset, sys_pads);

      // i = 0 save and perform as last step
      REAL b_0 = l_b.f[0];
      REAL c_0 = l_c.f[0];
      REAL d_0 = l_d.f[0];
      // i = 1
      forward_linear_process_row1<REAL, shift_c0_on_rank0>(
          l_a.f[1], a_m1, l_b.f[1], l_c.f[1], c_m1, l_d.f[1], d_m1, b_0, c_0,
          d_0, sys_size, rank, nproc);
      l_d.f[1]  = d_m1;
      l_aa.f[1] = a_m1;
      l_cc.f[1] = c_m1;

      for (int i = 2; i < vec_length<REAL>; i++) {
        forward_linear_process_line<REAL, shift_c0_on_rank0>(
            l_a.f[i], a_m1, l_b.f[i], l_c.f[i], c_m1, l_d.f[i], d_m1, b_0, c_0,
            d_0, sys_size, n + i, rank, nproc);
        l_d.f[i]  = d_m1;
        l_aa.f[i] = a_m1;
        l_cc.f[i] = c_m1;
      }

      store_array_reg(d, &l_d, n, woffset, sys_pads);
      store_array_reg(cc, &l_cc, n, woffset, sys_pads);
      store_array_reg(aa, &l_aa, n, woffset, sys_pads);

      // Forward pass
      for (n = vec_length<REAL>; n < sys_size - vec_length<REAL>;
           n += vec_length<REAL>) {
        load_array_reg(a, &l_a, n, woffset, sys_pads);
        load_array_reg(b, &l_b, n, woffset, sys_pads);
        load_array_reg(c, &l_c, n, woffset, sys_pads);
        load_array_reg(d, &l_d, n, woffset, sys_pads);
#pragma unroll
        for (int i = 0; i < vec_length<REAL>; i++) {
          forward_linear_process_line<REAL, shift_c0_on_rank0>(
              l_a.f[i], a_m1, l_b.f[i], l_c.f[i], c_m1, l_d.f[i], d_m1, b_0,
              c_0, d_0, sys_size, n + i, rank, nproc);
          l_d.f[i]  = d_m1;
          l_aa.f[i] = a_m1;
          l_cc.f[i] = c_m1;
        }
        store_array_reg(d, &l_d, n, woffset, sys_pads);
        store_array_reg(cc, &l_cc, n, woffset, sys_pads);
        store_array_reg(aa, &l_aa, n, woffset, sys_pads);
      }

      // Finish off last part that may not fill an entire vector
      for (int i = n; i < sys_size; i++) {
        int loc_ind = ind + i;
        forward_linear_process_line<REAL, shift_c0_on_rank0>(
            a[loc_ind], a_m1, b[loc_ind], c[loc_ind], c_m1, d[loc_ind], d_m1,
            b_0, c_0, d_0, sys_size, i, rank, nproc);
        aa[loc_ind] = a_m1;
        cc[loc_ind] = c_m1;
        d[loc_ind]  = d_m1;
      }
      // i = 0
      if (0 == rank) {
        aa[ind] = 0;
      } else {
        aa[ind] = a[ind] / b_0;
      }
      cc[ind] = c_0 / b_0;
      d[ind]  = d_0 / b_0;
    } else {
      trid_linear_forward_single_system<REAL, shift_c0_on_rank0>(
          a, b, c, d, aa, cc, sys_size, ind, rank, nproc);
    }
    // Store boundary values for communication
    copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid, ind,
                                               sys_size, sys_n);
  }
}

template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
__global__ void trid_linear_forward_pass_unaligned(
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ boundaries, int sys_size,
    int sys_pads, int sys_n, int rank, int nproc, int offset) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid / 4) * 4 + 4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global
  // memory read/write
  const int boundary_solve = !optimized_solve && (tid < (sys_n));
  // A thread is active only if it works on valid memory
  const int active_thread = optimized_solve || boundary_solve;

  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Check that this is an active thread
  if (active_thread) {
    // Check that this thread can perform an optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL)) {
      // Local arrays used in the register shuffle
      vec_line_t<REAL> l_a, l_b, l_c, l_d, l_aa, l_cc;
      REAL a_m1, c_m1, d_m1;

      // Memory is unaligned
      int ind_floor = ((ind + offset) / align<REAL>)*align<REAL> - offset;
      int sys_off   = ind - ind_floor;

      // Handle start of unaligned memory
      // i = 0 : idx = sys_off, save and perform as last step
      //
      REAL b_0 = b[ind];
      REAL c_0 = c[ind];
      REAL d_0 = d[ind];
      // i = 1 : idx = sys_off +1
      if (sys_off + 1 < vec_length<REAL>) {
        forward_linear_process_row1<REAL, shift_c0_on_rank0>(
            a[ind + 1], a_m1, b[ind + 1], c[ind + 1], c_m1, d[ind + 1], d_m1,
            b_0, c_0, d_0, sys_size, rank, nproc);
        aa[ind + 1] = a_m1;
        cc[ind + 1] = c_m1;
        d[ind + 1]  = d_m1;

        for (int i = 2; i + sys_off < vec_length<REAL>; i++) {
          forward_linear_process_line<REAL, shift_c0_on_rank0>(
              a[ind + i], a_m1, b[ind + i], c[ind + i], c_m1, d[ind + i], d_m1,
              b_0, c_0, d_0, sys_size, i - sys_off, rank, nproc);
          d[ind + i]  = d_m1;
          aa[ind + i] = a_m1;
          cc[ind + i] = c_m1;
        }
      }

      int n = vec_length<REAL>;
      // Back to normal
      for (; n < sys_size - vec_length<REAL>; n += vec_length<REAL>) {
        load_array_reg_unaligned(a, &l_a, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(b, &l_b, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(c, &l_c, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(d, &l_d, n, tid, sys_pads, sys_size, offset);
#pragma unroll
        for (int i = 0; i < vec_length<REAL>; i++) {
          if (i == 0 && n + i - sys_off == 1) {
            // i = 1 iteration
            forward_linear_process_row1<REAL, shift_c0_on_rank0>(
                l_a.f[i], a_m1, l_b.f[i], l_c.f[i], c_m1, l_d.f[i], d_m1, b_0,
                c_0, d_0, sys_size, rank, nproc);
          } else {
            forward_linear_process_line<REAL, shift_c0_on_rank0>(
                l_a.f[i], a_m1, l_b.f[i], l_c.f[i], c_m1, l_d.f[i], d_m1, b_0,
                c_0, d_0, sys_size, n + i - sys_off, rank, nproc);
          }
          l_d.f[i]  = d_m1;
          l_aa.f[i] = a_m1;
          l_cc.f[i] = c_m1;
        }
        store_array_reg_unaligned(d, &l_d, n, tid, sys_pads, sys_size, offset);
        store_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size,
                                  offset);
        store_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size,
                                  offset);
      }

      // Handle end of unaligned memory
      for (int i = n; i < sys_size + sys_off; i++) {
        int loc_ind = ind_floor + i;
        forward_linear_process_line<REAL, shift_c0_on_rank0>(
            a[loc_ind], a_m1, b[loc_ind], c[loc_ind], c_m1, d[loc_ind], d_m1,
            b_0, c_0, d_0, sys_size, i - sys_off, rank, nproc);
        d[loc_ind]  = d_m1;
        aa[loc_ind] = a_m1;
        cc[loc_ind] = c_m1;
      }
      // i = 0
      if (0 == rank) {
        aa[ind] = 0;
      } else {
        aa[ind] = a[ind] / b_0;
      }
      cc[ind] = c_0 / b_0;
      d[ind]  = d_0 / b_0;
    } else {
      trid_linear_forward_single_system<REAL, shift_c0_on_rank0>(
          a, b, c, d, aa, cc, sys_size, ind, rank, nproc);
    }
    // Store boundary values for communication
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
template <typename REAL, int INC, bool is_c0_cleared_on_rank0>
inline __device__ void trid_linear_backward_pass_single_system(
    const REAL *__restrict__ aa, const REAL *__restrict__ cc,
    REAL *__restrict__ d, REAL *__restrict__ u, REAL dd0, REAL dd_p1,
    int sys_size, int ind, int rank, int nproc) {
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
    trid_linear_backward_pass_single_system<REAL, INC, is_c0_cleared_on_rank0>(
        aa, cc, d, u, dd0, dd_p1, sys_size, ind, rank, nproc);
  }
}

// Modified backwards pass for X dimension.
// Uses register shuffle optimization, can handle both aligned and unaligned
// memory
template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
__global__ void trid_linear_backward_pass_aligned(
    const REAL *__restrict__ aa, const REAL *__restrict__ cc,
    REAL *__restrict__ d, REAL *__restrict__ u,
    const REAL *__restrict__ boundaries, int sys_size, int sys_pads, int sys_n,
    int rank, int nproc) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // Warp ID in global scope - the ID wich the thread belongs to
  const int wid = tid / WARP_SIZE;
  // Global memory offset: unique to a warp;
  // every thread in a warp calculates the same woffset, which is the "begining"
  // of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid / 4) * 4 + 4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global
  // memory read/write
  const int boundary_solve = !optimized_solve && (tid < (sys_n));
  // A thread is active only if it works on valid memory
  const int active_thread = optimized_solve || boundary_solve;

  // Start index for this tridiagonal system
  int ind = sys_pads * tid;


  // Check if active thread
  if (active_thread) {
    // Set start and end dd values
    REAL dd0;
    REAL dd_p1;
    load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, dd_p1, tid,
                                                    sys_n);
    // Check if optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL)) {
      int n = 0;
      // Local arrays used in register shuffle
      vec_line_t<REAL> l_aa, l_cc, l_d, l_u;
      // Start with last vector
      int end_remainder = sys_size / vec_length<REAL> * vec_length<REAL>;
      if (end_remainder < sys_size) {
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
        for (int i = sys_size - 2; i >= end_remainder; --i) {
          loc_ind = ind + i;
          dd_p1   = d[loc_ind] - cc[loc_ind] * dd_p1;
          if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
          if (INC)
            u[ind + i] += dd_p1;
          else
            d[ind + i] = dd_p1;
        }
      }
      for (n = end_remainder - vec_length<REAL>; n > 0; n -= vec_length<REAL>) {
        load_array_reg(aa, &l_aa, n, woffset, sys_pads);
        load_array_reg(cc, &l_cc, n, woffset, sys_pads);
        load_array_reg(d, &l_d, n, woffset, sys_pads);
        if (INC) load_array_reg(u, &l_u, n, woffset, sys_pads);

        for (int i = vec_length<REAL> - 1; i >= 0; i--) {
          if (n + i == sys_size - 1) {
            if (rank != nproc - 1)
              dd_p1 = l_d.f[i] - dd_p1 * l_cc.f[i];
            else
              dd_p1 = l_d.f[i];
            if (rank != 0) dd_p1 += -l_aa.f[i] * dd0;
          } else {
            dd_p1 = l_d.f[i] - l_cc.f[i] * dd_p1;
            if (rank != 0) dd_p1 += -l_aa.f[i] * dd0;
          }
          if (INC)
            l_u.f[i] += dd_p1;
          else
            l_d.f[i] = dd_p1;
        }
        if (INC)
          store_array_reg(u, &l_u, n, woffset, sys_pads);
        else
          store_array_reg(d, &l_d, n, woffset, sys_pads);
      }
      // Handle first vector
      load_array_reg(aa, &l_aa, 0, woffset, sys_pads);
      load_array_reg(cc, &l_cc, 0, woffset, sys_pads);
      load_array_reg(d, &l_d, 0, woffset, sys_pads);
      if (INC) load_array_reg(u, &l_u, 0, woffset, sys_pads);

      for (int i = vec_length<REAL> - 1; i >= 1; i--) {
        if (i == sys_size - 1) {
          if (rank != nproc - 1)
            dd_p1 = l_d.f[i] - dd_p1 * l_cc.f[i];
          else
            dd_p1 = l_d.f[i];
          if (rank != 0) dd_p1 += -l_aa.f[i] * dd0;
        } else {
          dd_p1 = l_d.f[i] - l_cc.f[i] * dd_p1;
          if (rank != 0) dd_p1 += -l_aa.f[i] * dd0;
        }
        if (INC)
          l_u.f[i] += dd_p1;
        else
          l_d.f[i] = dd_p1;
      }
      // i = 0
      if (0 == rank && !is_c0_cleared_on_rank0) {
        dd_p1 = dd0 - l_cc.f[0] * dd_p1;
        if (rank != 0) dd_p1 += -l_aa.f[0] * dd0;
      } else {
        dd_p1 = dd0;
      }
      if (INC) {
        l_u.f[0] += dd_p1;
        store_array_reg(u, &l_u, n, woffset, sys_pads);
      } else {
        l_d.f[0] = dd_p1;
        store_array_reg(d, &l_d, n, woffset, sys_pads);
      }
    } else {
      // Normal modified backwards if not optimized solve
      trid_linear_backward_pass_single_system<REAL, INC,
                                              is_c0_cleared_on_rank0>(
          aa, cc, d, u, dd0, dd_p1, sys_size, ind, rank, nproc);
    }
  }
}

template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
__global__ void trid_linear_backward_pass_unaligned(
    const REAL *__restrict__ aa, const REAL *__restrict__ cc,
    REAL *__restrict__ d, REAL *__restrict__ u,
    const REAL *__restrict__ boundaries, int sys_size, int sys_pads, int sys_n,
    int offset, int rank, int nproc) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid / 4) * 4 + 4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global
  // memory read/write
  const int boundary_solve = !optimized_solve && (tid < (sys_n));
  // A thread is active only if it works on valid memory
  const int active_thread = optimized_solve || boundary_solve;

  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Check if active thread
  if (active_thread) {
    // Set start and end dd values
    REAL dd0;
    REAL dd_p1;
    load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, dd_p1, tid,
                                                    sys_n);
    // Check if optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL)) {
      int n = 0;
      // Local arrays used in register shuffle
      vec_line_t<REAL> l_aa, l_cc, l_d, l_u;
      // Unaligned memory

      int ind_floor = ((ind + offset) / align<REAL>)*align<REAL> - offset;
      int sys_off   = ind - ind_floor;
      int end_remainder =
          sys_size / vec_length<REAL> * vec_length<REAL> - sys_off;

      if (end_remainder < sys_size) {
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
        for (int i = sys_size - 2; i >= end_remainder; --i) {
          loc_ind = ind + i;
          dd_p1   = d[loc_ind] - cc[loc_ind] * dd_p1;
          if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
          if (INC)
            u[ind + i] += dd_p1;
          else
            d[ind + i] = dd_p1;
        }
      }

      for (n = end_remainder + sys_off - vec_length<REAL>; n > 0;
           n -= vec_length<REAL>) {
        load_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(d, &l_d, n, tid, sys_pads, sys_size, offset);
        if (INC)
          load_array_reg_unaligned(u, &l_u, n, tid, sys_pads, sys_size, offset);

        for (int i = vec_length<REAL> - 1; i >= 0; i--) {
          if (n + i == sys_size - 1) {
            if (rank != nproc - 1)
              dd_p1 = l_d.f[i] - dd_p1 * l_cc.f[i];
            else
              dd_p1 = l_d.f[i];
            if (rank != 0) dd_p1 += -l_aa.f[i] * dd0;
          } else {
            dd_p1 = l_d.f[i] - l_cc.f[i] * dd_p1;
            if (rank != 0) dd_p1 += -l_aa.f[i] * dd0;
          }
          if (INC)
            l_u.f[i] += dd_p1;
          else
            l_d.f[i] = dd_p1;
        }
        if (INC)
          store_array_reg_unaligned(u, &l_u, n, tid, sys_pads, sys_size,
                                    offset);
        else
          store_array_reg_unaligned(d, &l_d, n, tid, sys_pads, sys_size,
                                    offset);
      }
      // Handle first unaligned vector
      for (int i = vec_length<REAL> - 1 - sys_off; i >= 1; i--) {
        int loc_ind = ind + i;
        if (i == sys_size - 1) {
          if (rank != nproc - 1)
            dd_p1 = d[loc_ind] - dd_p1 * cc[loc_ind];
          else
            dd_p1 = d[loc_ind];
          if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
        } else {
          dd_p1 = d[loc_ind] - cc[loc_ind] * dd_p1;
          if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
        }
        if (INC)
          u[loc_ind] += dd_p1;
        else
          d[loc_ind] = dd_p1;
      }
      if (0 == rank && !is_c0_cleared_on_rank0) {
        dd_p1 = dd0 - cc[ind] * dd_p1;
        if (rank != 0) dd_p1 += -aa[ind] * dd0;
      } else {
        dd_p1 = dd0;
      }
      if (INC) {
        u[ind] += dd_p1;
      } else {
        d[ind] = dd_p1;
      }
    } else {
      // Normal modified backwards if not optimized solve
      trid_linear_backward_pass_single_system<REAL, INC,
                                              is_c0_cleared_on_rank0>(
          aa, cc, d, u, dd0, dd_p1, sys_size, ind, rank, nproc);
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
 * The layout of boundaries is as if it holds the a and c values of the lines
 * as well.
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

//
//  Single jacobi iteration for reduced
//  assumption: we only have one row per system in boundaries
//  and 1-1 prev and next row in recv_buf
//
template <typename REAL>
__global__ void trid_jacobi_iteration(REAL *__restrict__ boundaries,
                                      const REAL *__restrict__ recv_m1,
                                      const REAL *__restrict__ recv_p1,
                                      int sys_n, bool m1, bool p1) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  extern __shared__ char temp2[];
  REAL *temp = (REAL *)temp2;

  int ind       = tid;
  REAL sys_norm = 0.0;
  if (tid < sys_n) {
    REAL dm1  = 0.0;
    REAL dp1  = 0.0;
    REAL aa   = boundaries[0 * sys_n + tid];
    REAL cc   = boundaries[1 * sys_n + tid];
    REAL dd   = boundaries[3 * sys_n + tid];
    REAL dd_r = boundaries[2 * sys_n + tid];
    if (m1) {
      dm1 = recv_m1[ind];
    }
    if (p1) {
      dp1 = recv_p1[ind];
    }

    REAL diff                   = dd_r + aa * dm1 + cc * dp1 - dd;
    sys_norm                    = diff * diff;
    boundaries[2 * sys_n + tid] = dd - cc * dp1 - aa * dm1;
  }
  // perform reduction over elements
  __syncthreads();
  int bid   = blockIdx.x + blockIdx.y * gridDim.x;
  tid       = threadIdx.x + threadIdx.y * blockDim.x;
  temp[tid] = sys_norm;
  // first, cope with blockDim.x perhaps not being a power of 2
  int d = 1 << (31 - __clz(((int)(blockDim.x * blockDim.y * blockDim.z) - 1)));
  __syncthreads();
  // d = blockDim.x/2 rounded up to nearest power of 2
  if (tid + d < blockDim.x * blockDim.y * blockDim.z) {
    REAL dat_t = temp[tid + d];
    if (dat_t > sys_norm) sys_norm = dat_t;
    temp[tid] = sys_norm;
  }

  // second, do reductions
  for (d >>= 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      REAL dat_t = temp[tid + d];
      if (dat_t > sys_norm) sys_norm = dat_t;
      temp[tid] = sys_norm;
    }
  }
  if (bid < sys_n && tid == 0) boundaries[4 * sys_n + bid] = sys_norm;
}

////////////////////////////////////////
//        Host functions
////////////////////////////////////////

//
// Kernel launch wrapper for forward step with register blocking
//
template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0>
void trid_linear_forward_pass_reg(dim3 dimGrid_x, dim3 dimBlock_x,
                                  const REAL *a, const REAL *b, const REAL *c,
                                  REAL *d, REAL *aa, REAL *cc, REAL *boundaries,
                                  int sys_size, int sys_pads, int sys_n,
                                  int rank, int nproc, cudaStream_t stream) {
  const size_t offset = ((size_t)d / sizeof(REAL)) % align<REAL>;

  const int aligned =
      (sys_pads % align<REAL>) == 0 && (offset % align<REAL>) == 0;
  if (aligned) {
    trid_linear_forward_pass_aligned<REAL, boundary_SOA, shift_c0_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(a, b, c, d, aa, cc, boundaries,
                                               sys_size, sys_pads, sys_n, rank,
                                               nproc);
  } else {
    trid_linear_forward_pass_unaligned<REAL, boundary_SOA, shift_c0_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(a, b, c, d, aa, cc, boundaries,
                                               sys_size, sys_pads, sys_n, rank,
                                               nproc, offset);
  }
}

//
// Kernel launch wrapper for backward step with register blocking
//
template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0>
void trid_linear_backward_pass_reg(dim3 dimGrid_x, dim3 dimBlock_x,
                                   const REAL *aa, const REAL *cc, REAL *d,
                                   REAL *u, const REAL *boundaries,
                                   int sys_size, int sys_pads, int sys_n,
                                   int rank, int nproc, cudaStream_t stream) {
  const size_t offset = ((size_t)d / sizeof(REAL)) % align<REAL>;

  const int aligned =
      (sys_pads % align<REAL>) == 0 && (offset % align<REAL>) == 0;
  if (aligned) {
    trid_linear_backward_pass_aligned<REAL, INC, boundary_SOA,
                                      is_c0_cleared_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa, cc, d, u, boundaries, sys_size, sys_pads, sys_n, rank, nproc);
  } else {
    trid_linear_backward_pass_unaligned<REAL, INC, boundary_SOA,
                                        is_c0_cleared_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(aa, cc, d, u, boundaries,
                                               sys_size, sys_pads, sys_n,
                                               offset, rank, nproc);
  }
}

template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
inline void forward_batched_pass(
    dim3 dimGrid_x, dim3 dimBlock_x, const MpiSolverParams &params,
    const REAL *a, const int *a_pads, const REAL *b, const int *b_pads,
    const REAL *c, const int *c_pads, REAL *d, const int *d_pads, REAL *aa,
    REAL *cc, REAL *boundaries, const int *dims, int ndim, int solvedim,
    int start_sys, int bsize, cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    assert(a_pads[0] == b_pads[0] && a_pads[0] == c_pads[0] &&
           a_pads[0] == d_pads[0] && "different paddings are not supported");
    if (ndim > 1) {
      assert(a_pads[1] == dims[1] && b_pads[1] == dims[1] &&
             c_pads[1] == dims[1] && d_pads[1] == dims[1] &&
             " ONLLY X paddings are supported");
    }
    const int batch_offset = start_sys * a_pads[solvedim];
    // trid_linear_forward_pass<REAL, boundary_SOA, shift_c0_on_rank0>
    //     <<<dimGrid_x, dimBlock_x, 0, stream>>>(
    //         a + batch_offset, b + batch_offset, c + batch_offset,
    //         d + batch_offset, aa + batch_offset, cc + batch_offset,
    //         boundaries + start_sys * 3 * 2, dims[solvedim], a_pads[solvedim],
    //         bsize, params.mpi_coords[solvedim],
    //         params.num_mpi_procs[solvedim]);
    trid_linear_forward_pass_reg<REAL, boundary_SOA, shift_c0_on_rank0>(
        dimGrid_x, dimBlock_x, a + batch_offset, b + batch_offset,
        c + batch_offset, d + batch_offset, aa + batch_offset,
        cc + batch_offset, boundaries + start_sys * 3 * 2, dims[solvedim],
        a_pads[solvedim], bsize, params.mpi_coords[solvedim],
        params.num_mpi_procs[solvedim], stream);
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
                                  cudaStream_t stream = nullptr) {
  assert(start_sys == 0 &&
         "check the whole process for boundaries if indexing is correct for "
         "batches then remove this");
  if (solvedim == 0) {
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
    // trid_linear_backward_pass<REAL, INC, boundary_SOA,
    // is_c0_cleared_on_rank0>
    //     <<<dimGrid_x, dimBlock_x, 0, stream>>>(
    //         aa + batch_offset, cc + batch_offset, d + batch_offset,
    //         u + batch_offset, boundaries + start_sys * 2 * 3, dims[solvedim],
    //         a_pads[solvedim], bsize, params.mpi_coords[solvedim],
    //         params.num_mpi_procs[solvedim]);
    trid_linear_backward_pass_reg<REAL, INC, boundary_SOA,
                                  is_c0_cleared_on_rank0>(
        dimGrid_x, dimBlock_x, aa + batch_offset, cc + batch_offset,
        d + batch_offset, u + batch_offset, boundaries + start_sys * 2 * 3,
        dims[solvedim], a_pads[solvedim], bsize, params.mpi_coords[solvedim],
        params.num_mpi_procs[solvedim], stream);
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
              params.communicators[solvedim], &rcv_requests[1]);
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

template <typename REAL>
inline void iterative_jacobi_on_reduced(dim3 dimGrid_x, dim3 dimBlock_x,
                                        const MpiSolverParams &params,
                                        REAL *boundaries, int sys_n,
                                        int solvedim, REAL *recv_buf_d,
                                        REAL *recv_buf_h, REAL *send_buf_h) {

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
  if (rank) {
    trid_PCR_iteration<REAL, true><<<dimGrid_x, dimBlock_x>>>(
        boundaries, recv_buf_d, nullptr, sys_n, true, false);
    // initial guess for jacobi iterations
    cudaMemcpy(boundaries + 3 * sys_n, boundaries + 2 * sys_n,
               sys_n * sizeof(REAL), cudaMemcpyDeviceToDevice);
  }
#ifndef TRID_NCCL
  BEGIN_PROFILING("mpi_communication");
  MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
  END_PROFILING("mpi_communication");
#endif

  // norm comp
  int global_sys_len    = params.num_mpi_procs[solvedim];
  REAL local_norm_send  = -1.0;
  REAL global_norm_recv = -1.0;
  REAL global_norm      = -1.0;
  REAL norm0            = -1.0;
  bool need_iter        = true;

  MPI_Request norm_req = MPI_REQUEST_NULL;
  int iter             = 0;
  while ((params.jacobi_maxiter < 0 || iter < params.jacobi_maxiter) &&
         need_iter) {
    REAL local_norm = 0.0;
    // send res to neighbours
    if (rank) {
      // rank 0 does not do jacobi iters, only the allreduce -> rank 1 shouldn't
      // send upwards.
      trid_cuda_pcr_exchange_line<REAL>(
          boundaries + 2 * sys_n, send_buf_h, recv_buf_d, recv_buf_h,
          boundaries + 5 * sys_n, recv_buf_h + sys_n, sys_n, nproc,
          rank == 1 ? -1 : rank - 1, rank + 1, params, solvedim, rcv_requests,
          snd_requests);

#ifndef TRID_NCCL
      BEGIN_PROFILING("mpi_communication");
      MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
      END_PROFILING("mpi_communication");
#endif
      // do jacobi iter and compute norm
      // using boundaries for scratch mem.
      // boundaries store: aas, ccs, current guess for sysmtems, original dds,
      // result of cuda reduction, message from next process
      int numblocks = dimGrid_x.x * dimGrid_x.y * dimGrid_x.z;
      int nthreads  = dimBlock_x.x * dimBlock_x.y * dimBlock_x.z;
      trid_jacobi_iteration<<<dimGrid_x, dimBlock_x, nthreads * sizeof(REAL)>>>(
          boundaries, recv_buf_d, boundaries + 5 * sys_n, sys_n, rank - 1 > 0,
          rank + 1 < nproc);
      cudaMemcpy(recv_buf_h, boundaries + 4 * sys_n, numblocks * sizeof(REAL),
                 cudaMemcpyDeviceToHost);
      for (int i = 0; i < numblocks; ++i) {
        if (local_norm < recv_buf_h[i]) local_norm = recv_buf_h[i];
      }
    }
    // allgather norm
    BEGIN_PROFILING("mpi_communication");
    MPI_Wait(&norm_req, MPI_STATUS_IGNORE);
    norm_req = MPI_REQUEST_NULL;
    if (global_norm_recv >= 0) // skip until the first sum is ready
      global_norm = sqrt(global_norm_recv / global_sys_len);
    if (norm0 < 0) norm0 = global_norm;
    local_norm_send = local_norm;
    iter++;
    need_iter = global_norm < 0.0 || (params.jacobi_atol < global_norm &&
                                      params.jacobi_rtol < global_norm / norm0);
    if ((params.jacobi_maxiter < 0 || iter + 1 < params.jacobi_maxiter) &&
        need_iter) { // if norm is not enough and next is not last iteration
      MPI_Iallreduce(&local_norm_send, &global_norm_recv, 1, MPI_DATATYPE(REAL),
                     MPI_SUM, params.communicators[solvedim], &norm_req);
    }
    END_PROFILING("mpi_communication");
  }

  // send solution up for backward
  trid_cuda_pcr_exchange_line<REAL, false, true>(
      boundaries + 2 * sys_n, send_buf_h, nullptr, nullptr,
      boundaries + 5 * sys_n, recv_buf_h, sys_n, nproc, rank - 1, rank + 1,
      params, solvedim, rcv_requests, snd_requests);
#ifndef TRID_NCCL
  BEGIN_PROFILING("mpi_communication");
  MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
  END_PROFILING("mpi_communication");
#endif
}

#endif /* ifndef TRID_ITERATIVE_MPI_HPP_INCLUDED */
