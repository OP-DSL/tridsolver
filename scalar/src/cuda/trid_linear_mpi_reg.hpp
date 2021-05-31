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
// balogh.gabor.daniel@itk.ppke.hu, 2021
// Based on previous version by Toby Flynn, University of Warwick,
// T.Flynn@warwick.ac.uk, 2020

// This file contains template wrappers to ease the use of linear solver with
// register blocking
#ifndef TRID_LINEAR_MPI_REG_HPP__
#define TRID_LINEAR_MPI_REG_HPP__

#include "trid_linear_reg_common.hpp"
#include "trid_mpi_helper.hpp"

// Modified Thomas forward pass for X dimension.
// Uses register shuffle optimization, can handle both aligned and unaligned
// memory
template <typename REAL, bool boundary_SOA = false>
__global__ void trid_linear_forward_aligned(
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ boundaries, int sys_size,
    int sys_pads, int sys_n, int offset, int start_sys, int y_size,
    int y_pads) {
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
  // Check if in y padding
  const int padded_sys = ((start_sys + tid) % y_pads) >= y_size;

  int n = 0;
  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Local arrays used in the register shuffle
  vec_line_t<REAL> l_a, l_b, l_c, l_d, l_aa, l_cc, l_dd;
  REAL bb, a2, c2, d2;

  // Check that this is an active thread
  if (active_thread) {
    // Check that this thread can perform an optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL)) {
      // Process first vector separately
      load_array_reg(a, &l_a, n, woffset, sys_pads);
      load_array_reg(b, &l_b, n, woffset, sys_pads);
      load_array_reg(c, &l_c, n, woffset, sys_pads);
      load_array_reg(d, &l_d, n, woffset, sys_pads);

      for (int i = 0; i < 2; i++) {
        bb        = 1.0 / l_b.f[i];
        d2        = bb * l_d.f[i];
        a2        = bb * l_a.f[i];
        c2        = bb * l_c.f[i];
        l_dd.f[i] = d2;
        l_aa.f[i] = a2;
        l_cc.f[i] = c2;
      }

      for (int i = 2; i < vec_length<REAL>; i++) {
        bb        = 1.0 / (l_b.f[i] - l_a.f[i] * c2);
        d2        = (l_d.f[i] - l_a.f[i] * d2) * bb;
        a2        = (-l_a.f[i] * a2) * bb;
        c2        = l_c.f[i] * bb;
        l_dd.f[i] = d2;
        l_aa.f[i] = a2;
        l_cc.f[i] = c2;
      }

      store_array_reg(d, &l_dd, n, woffset, sys_pads);
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
          bb        = 1.0 / (l_b.f[i] - l_a.f[i] * c2);
          d2        = (l_d.f[i] - l_a.f[i] * d2) * bb;
          a2        = (-l_a.f[i] * a2) * bb;
          c2        = l_c.f[i] * bb;
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }
        store_array_reg(d, &l_dd, n, woffset, sys_pads);
        store_array_reg(cc, &l_cc, n, woffset, sys_pads);
        store_array_reg(aa, &l_aa, n, woffset, sys_pads);
      }

      // Finish off last part that may not fill an entire vector
      for (int i = n; i < sys_size; i++) {
        int loc_ind = ind + i;
        bb          = 1.0 / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
        d[loc_ind]  = (d[loc_ind] - a[loc_ind] * d[loc_ind - 1]) * bb;
        aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
        cc[loc_ind] = c[loc_ind] * bb;
      }

      // Backwards pass
      n -= vec_length<REAL>;

      a2 = aa[ind + sys_size - 2];
      c2 = cc[ind + sys_size - 2];
      d2 = d[ind + sys_size - 2];

      // Do part that may not fit in vector
      for (int i = sys_size - 3; i >= n + vec_length<REAL>; i--) {
        int loc_ind = ind + i;
        d2          = d[loc_ind] - cc[loc_ind] * d2;
        a2          = aa[loc_ind] - cc[loc_ind] * a2;
        c2          = -cc[loc_ind] * c2;
        d[loc_ind]  = d2;
        aa[loc_ind] = a2;
        cc[loc_ind] = c2;
      }

      // Backwards pass using vectors
      for (; n > 0; n -= vec_length<REAL>) {
        load_array_reg(aa, &l_aa, n, woffset, sys_pads);
        load_array_reg(cc, &l_cc, n, woffset, sys_pads);
        load_array_reg(d, &l_dd, n, woffset, sys_pads);

        for (int i = vec_length<REAL> - 1; i >= 0; i--) {
          d2        = l_dd.f[i] - l_cc.f[i] * d2;
          a2        = l_aa.f[i] - l_cc.f[i] * a2;
          c2        = -l_cc.f[i] * c2;
          l_dd.f[i] = d2;
          l_cc.f[i] = c2;
          l_aa.f[i] = a2;
        }

        store_array_reg(d, &l_dd, n, woffset, sys_pads);
        store_array_reg(cc, &l_cc, n, woffset, sys_pads);
        store_array_reg(aa, &l_aa, n, woffset, sys_pads);
      }

      // Final vector processed separately so that element 0 can be handled
      n = 0;

      load_array_reg(aa, &l_aa, n, woffset, sys_pads);
      load_array_reg(cc, &l_cc, n, woffset, sys_pads);
      load_array_reg(d, &l_dd, n, woffset, sys_pads);

      for (int i = vec_length<REAL> - 1; i > 0; i--) {
        d2        = l_dd.f[i] - l_cc.f[i] * d2;
        a2        = l_aa.f[i] - l_cc.f[i] * a2;
        c2        = -l_cc.f[i] * c2;
        l_dd.f[i] = d2;
        l_cc.f[i] = c2;
        l_aa.f[i] = a2;
      }

      bb        = 1.0 / (1.0 - l_cc.f[0] * a2);
      l_dd.f[0] = bb * (l_dd.f[0] - l_cc.f[0] * d2);
      l_aa.f[0] = bb * l_aa.f[0];
      l_cc.f[0] = bb * (-l_cc.f[0] * c2);

      store_array_reg(d, &l_dd, n, woffset, sys_pads);
      store_array_reg(cc, &l_cc, n, woffset, sys_pads);
      store_array_reg(aa, &l_aa, n, woffset, sys_pads);

      // Store boundary values for communication
      copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid,
                                                 ind, sys_size, sys_n);
    } else if(!padded_sys) {
      // Normal modified Thomas if not optimized solve

      for (int i = 0; i < 2; ++i) {
        bb          = 1.0 / b[ind + i];
        d[ind + i] = bb * d[ind + i];
        aa[ind + i] = bb * a[ind + i];
        cc[ind + i] = bb * c[ind + i];
      }

      if (sys_size >= 3) {
        // eliminate lower off-diagonal
        for (int i = 2; i < sys_size; i++) {
          int loc_ind = ind + i;
          bb          = 1.0 / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
          d[loc_ind]  = (d[loc_ind] - a[loc_ind] * d[loc_ind - 1]) * bb;
          aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
          cc[loc_ind] = c[loc_ind] * bb;
        }
        // Eliminate upper off-diagonal
        for (int i = sys_size - 3; i > 0; --i) {
          int loc_ind = ind + i;
          d[loc_ind]  = d[loc_ind] - cc[loc_ind] * d[loc_ind + 1];
          aa[loc_ind] = aa[loc_ind] - cc[loc_ind] * aa[loc_ind + 1];
          cc[loc_ind] = -cc[loc_ind] * cc[loc_ind + 1];
        }
        bb      = 1.0 / (1.0 - cc[ind] * aa[ind + 1]);
        d[ind]  = bb * (d[ind] - cc[ind] * d[ind + 1]);
        aa[ind] = bb * aa[ind];
        cc[ind] = bb * (-cc[ind] * cc[ind + 1]);
      }

      // Store boundary values for communication
      copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid,
                                                 ind, sys_size, sys_n);
    }
  }
}

template <typename REAL, bool boundary_SOA>
__global__ void trid_linear_forward_unaligned(
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ boundaries, int sys_size,
    int sys_pads, int sys_n, int offset, int start_sys, int y_size,
    int y_pads) {
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
  // Check if in y padding
  const int padded_sys = ((start_sys + tid) % y_pads) >= y_size;

  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Local arrays used in the register shuffle
  vec_line_t<REAL> l_a, l_b, l_c, l_d, l_aa, l_cc, l_dd;
  REAL bb, a2, c2, d2;

  // Check that this is an active thread
  if (active_thread) {
    // Check that this thread can perform an optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL) && false) {
      // Memory is unaligned
      int ind_floor = ((ind + offset) / align<REAL>)*align<REAL> - offset;
      int sys_off   = ind - ind_floor;

      // Handle start of unaligned memory
      for (int i = 0; i < vec_length<REAL>; i++) {
        if (i >= sys_off) {
          int loc_ind = ind_floor + i;
          if (i - sys_off < 2) {
            bb          = 1.0 / b[loc_ind];
            d2          = bb * d[loc_ind];
            a2          = bb * a[loc_ind];
            c2          = bb * c[loc_ind];
            d[loc_ind]  = d2;
            aa[loc_ind] = a2;
            cc[loc_ind] = c2;
          } else {
            bb          = 1.0 / (b[loc_ind] - a[loc_ind] * c2);
            d2          = (d[loc_ind] - a[loc_ind] * d2) * bb;
            a2          = (-a[loc_ind] * a2) * bb;
            c2          = c[loc_ind] * bb;
            d[loc_ind]  = d2;
            aa[loc_ind] = a2;
            cc[loc_ind] = c2;
          }
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
          bb        = 1.0 / (l_b.f[i] - l_a.f[i] * c2);
          d2        = (l_d.f[i] - l_a.f[i] * d2) * bb;
          a2        = (-l_a.f[i] * a2) * bb;
          c2        = l_c.f[i] * bb;
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }
        store_array_reg_unaligned(d, &l_dd, n, tid, sys_pads, sys_size,
                                  offset);
        store_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size,
                                  offset);
        store_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size,
                                  offset);
      }

      // Handle end of unaligned memory
      for (int i = n; i < sys_size + sys_off; i++) {
        int loc_ind = ind_floor + i;
        bb          = 1.0 / (b[loc_ind] - a[loc_ind] * c2);
        d2          = (d[loc_ind] - a[loc_ind] * d2) * bb;
        a2          = (-a[loc_ind] * a2) * bb;
        c2          = c[loc_ind] * bb;
        d[loc_ind]  = d2;
        aa[loc_ind] = a2;
        cc[loc_ind] = c2;
      }

      // Backwards pass
      d2 = d[ind_floor + sys_size + sys_off - 2];
      a2 = aa[ind_floor + sys_size + sys_off - 2];
      c2 = cc[ind_floor + sys_size + sys_off - 2];

      n -= vec_length<REAL>;

      // Start with end of unaligned memory
      for (int i = sys_size + sys_off - 3; i >= n; i--) {
        int loc_ind = ind_floor + i;
        d2          = d[loc_ind] - cc[loc_ind] * d2;
        a2          = aa[loc_ind] - cc[loc_ind] * a2;
        c2          = -cc[loc_ind] * c2;
        d[loc_ind]  = d2;
        aa[loc_ind] = a2;
        cc[loc_ind] = c2;
      }

      n -= vec_length<REAL>;

      // Back to normal
      for (; n > 0; n -= vec_length<REAL>) {
        load_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size, offset);
        load_array_reg_unaligned(d, &l_dd, n, tid, sys_pads, sys_size, offset);

        for (int i = vec_length<REAL> - 1; i >= 0; i--) {
          d2        = l_dd.f[i] - l_cc.f[i] * d2;
          a2        = l_aa.f[i] - l_cc.f[i] * a2;
          c2        = -l_cc.f[i] * c2;
          l_dd.f[i] = d2;
          l_cc.f[i] = c2;
          l_aa.f[i] = a2;
        }

        store_array_reg_unaligned(d, &l_dd, n, tid, sys_pads, sys_size,
                                  offset);
        store_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size,
                                  offset);
        store_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size,
                                  offset);
      }

      for (int i = n + vec_length<REAL> - 1; i > sys_off; i--) {
        int loc_ind = ind_floor + i;
        d2          = d[loc_ind] - cc[loc_ind] * d2;
        a2          = aa[loc_ind] - cc[loc_ind] * a2;
        c2          = -cc[loc_ind] * c2;
        d[loc_ind]  = d2;
        aa[loc_ind] = a2;
        cc[loc_ind] = c2;
      }

      bb      = 1.0 / (1.0 - cc[ind] * a2);
      d[ind]  = bb * (d[ind] - cc[ind] * d2);
      aa[ind] = bb * aa[ind];
      cc[ind] = bb * (-cc[ind] * c2);

      // Store boundary values for communication
      copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid,
                                                 ind, sys_size, sys_n);
    } else if(!padded_sys) {
      // Normal modified Thomas if not optimized solve

      for (int i = 0; i < 2; ++i) {
        bb          = 1.0 / b[ind + i];
        d[ind + i]  = bb * d[ind + i];
        aa[ind + i] = bb * a[ind + i];
        cc[ind + i] = bb * c[ind + i];
      }

      if (sys_size >= 3) {
        // eliminate lower off-diagonal
        for (int i = 2; i < sys_size; i++) {
          int loc_ind = ind + i;
          bb          = 1.0 / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
          d[loc_ind]  = (d[loc_ind] - a[loc_ind] * d[loc_ind - 1]) * bb;
          aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
          cc[loc_ind] = c[loc_ind] * bb;
        }
        // Eliminate upper off-diagonal
        for (int i = sys_size - 3; i > 0; --i) {
          int loc_ind = ind + i;
          d[loc_ind]  = d[loc_ind] - cc[loc_ind] * d[loc_ind + 1];
          aa[loc_ind] = aa[loc_ind] - cc[loc_ind] * aa[loc_ind + 1];
          cc[loc_ind] = -cc[loc_ind] * cc[loc_ind + 1];
        }
        bb      = 1.0 / (1.0 - cc[ind] * aa[ind + 1]);
        d[ind]  = bb * (d[ind] - cc[ind] * d[ind + 1]);
        aa[ind] = bb * aa[ind];
        cc[ind] = bb * (-cc[ind] * cc[ind + 1]);
      }

      // Store boundary values for communication
      copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid,
                                                 ind, sys_size, sys_n);
    }
  }
}

// Modified Thomas backwards pass for X dimension.
// Uses register shuffle optimization, can handle both aligned and unaligned
// memory
template <typename REAL, int INC, bool boundary_SOA>
__global__ void trid_linear_backward_aligned(
    const REAL *__restrict__ aa, const REAL *__restrict__ cc,
    REAL *__restrict__ d, REAL *__restrict__ u,
    const REAL *__restrict__ boundaries, int sys_size, int sys_pads, int sys_n,
    int offset, int start_sys, int y_size, int y_pads) {
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
  // Check if in y padding
  const int padded_sys = ((start_sys + tid) % y_pads) >= y_size;

  int n = 0;
  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Local arrays used in register shuffle
  vec_line_t<REAL> l_aa, l_cc, l_dd, l_d, l_u;

  // Check if active thread
  if (active_thread) {
    // Set start and end dd values
    REAL dd0;
    REAL ddn;
    load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, ddn, tid,
                                                    sys_n);
    // Check if optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL)) {
      // If in padding, do dummy loads and stores without changing values in
      // padding so the register shuffle optimization can continue for other
      // threads
      if (INC) {
        // Handle first vector
        load_array_reg(aa, &l_aa, n, woffset, sys_pads);
        load_array_reg(cc, &l_cc, n, woffset, sys_pads);
        load_array_reg(d, &l_dd, n, woffset, sys_pads);
        load_array_reg(u, &l_u, n, woffset, sys_pads);

        if (!padded_sys) {
          l_u.f[0] += dd0;

          for (int i = 1; i < vec_length<REAL>; i++) {
            l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
        }

        store_array_reg(u, &l_u, n, woffset, sys_pads);

        // Iterate over remaining vectors
        for (n = vec_length<REAL>; n < sys_size - vec_length<REAL>;
             n += vec_length<REAL>) {
          load_array_reg(aa, &l_aa, n, woffset, sys_pads);
          load_array_reg(cc, &l_cc, n, woffset, sys_pads);
          load_array_reg(d, &l_dd, n, woffset, sys_pads);
          load_array_reg(u, &l_u, n, woffset, sys_pads);
          if (!padded_sys) {
            for (int i = 0; i < vec_length<REAL>; i++) {
              l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
          }
          store_array_reg(u, &l_u, n, woffset, sys_pads);
        }

        if (!padded_sys) {
          // Handle last section separately as might not completely fit into a
          // vector
          for (int i = n; i < sys_size - 1; i++) {
            u[ind + i] += d[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
          }

          u[ind + sys_size - 1] += ddn;
        }
      } else {
        // Handle first vector
        load_array_reg(aa, &l_aa, n, woffset, sys_pads);
        load_array_reg(cc, &l_cc, n, woffset, sys_pads);
        load_array_reg(d, &l_dd, n, woffset, sys_pads);

        if (!padded_sys) {
          l_d.f[0] = dd0;

          for (int i = 1; i < vec_length<REAL>; i++) {
            l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
        }

        store_array_reg(d, &l_d, n, woffset, sys_pads);

        // Iterate over all remaining vectors
        for (n = vec_length<REAL>; n < sys_size - vec_length<REAL>;
             n += vec_length<REAL>) {
          load_array_reg(aa, &l_aa, n, woffset, sys_pads);
          load_array_reg(cc, &l_cc, n, woffset, sys_pads);
          load_array_reg(d, &l_dd, n, woffset, sys_pads);
          if (!padded_sys) {
            for (int i = 0; i < vec_length<REAL>; i++) {
              l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
          }
          store_array_reg(d, &l_d, n, woffset, sys_pads);
        }

        if (!padded_sys) {
          // Handle last section separately as might not completely fit into a
          // vector
          for (int i = n; i < sys_size - 1; i++) {
            d[ind + i] = d[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
          }

          d[ind + sys_size - 1] = ddn;
        }
      }
    } else if (!padded_sys) {
      // Normal modified Thomas backwards pass if not optimized solve
      if (INC) {
        u[ind] += dd0;

        for (int i = 1; i < sys_size - 1; i++) {
          u[ind + i] += d[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }

        u[ind + sys_size - 1] += ddn;
      } else {
        d[ind] = dd0;

        for (int i = 1; i < sys_size - 1; i++) {
          d[ind + i] = d[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }

        d[ind + sys_size - 1] = ddn;
      }
    }
  }
}

template <typename REAL, int INC, bool boundary_SOA>
__global__ void trid_linear_backward_unaligned(
    const REAL *__restrict__ aa, const REAL *__restrict__ cc,
    REAL *__restrict__ d, REAL *__restrict__ u,
    const REAL *__restrict__ boundaries, int sys_size, int sys_pads, int sys_n,
    int offset, int start_sys, int y_size, int y_pads) {
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
  // Check if in y padding
  const int padded_sys = ((start_sys + tid) % y_pads) >= y_size;

  int n = 0;
  // Start index for this tridiagonal system
  int ind = sys_pads * tid;

  // Local arrays used in register shuffle
  vec_line_t<REAL> l_aa, l_cc, l_dd, l_d, l_u;

  // Check if active thread
  if (active_thread) {
    // Set start and end dd values
    REAL dd0;
    REAL ddn;
    load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, ddn, tid,
                                                    sys_n);
    // Check if optimized solve
    if (optimized_solve && sys_size >= 192 / sizeof(REAL) && false) {
      // Unaligned memory

      // If in padding, do dummy loads and stores without changing values in
      // padding so the register shuffle optimization can continue for other
      // threads
      if (INC) {
        int ind_floor = ((ind + offset) / align<REAL>)*align<REAL> - offset;
        int sys_off   = ind - ind_floor;

        if (!padded_sys) {
          // Handle start of unaligned memory
          for (int i = 0; i < vec_length<REAL>; i++) {
            if (i >= sys_off) {
              int loc_ind = ind_floor + i;
              if (i == sys_off) {
                u[loc_ind] += dd0;
              } else {
                u[loc_ind] +=
                    d[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
              }
            }
          }
        }

        n = vec_length<REAL>;
        // Back to normal
        for (; n < sys_size - vec_length<REAL>; n += vec_length<REAL>) {
          load_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size,
                                   offset);
          load_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size,
                                   offset);
          load_array_reg_unaligned(d, &l_dd, n, tid, sys_pads, sys_size,
                                   offset);
          load_array_reg_unaligned(u, &l_u, n, tid, sys_pads, sys_size, offset);
          if (!padded_sys) {
#pragma unroll
            for (int i = 0; i < vec_length<REAL>; i++) {
              l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
          }
          store_array_reg_unaligned(u, &l_u, n, tid, sys_pads, sys_size,
                                    offset);
        }

        if (!padded_sys) {
          // Handle end of unaligned memory
          for (int i = n; i < sys_size + sys_off - 1; i++) {
            int loc_ind = ind_floor + i;
            u[loc_ind] += d[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
          }

          u[ind + sys_size - 1] += ddn;
        }
      } else {
        int ind_floor = ((ind + offset) / align<REAL>)*align<REAL> - offset;
        int sys_off   = ind - ind_floor;

        if (!padded_sys) {
          // Handle start of unaligned memory
          for (int i = 0; i < vec_length<REAL>; i++) {
            if (i >= sys_off) {
              int loc_ind = ind_floor + i;
              if (i == sys_off) {
                d[loc_ind] = dd0;
              } else {
                d[loc_ind] =
                    d[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
              }
            }
          }
        }

        n = vec_length<REAL>;
        // Back to normal
        for (; n < sys_size - vec_length<REAL>; n += vec_length<REAL>) {
          load_array_reg_unaligned(aa, &l_aa, n, tid, sys_pads, sys_size,
                                   offset);
          load_array_reg_unaligned(cc, &l_cc, n, tid, sys_pads, sys_size,
                                   offset);
          load_array_reg_unaligned(d, &l_dd, n, tid, sys_pads, sys_size,
                                   offset);
          load_array_reg_unaligned(d, &l_d, n, tid, sys_pads, sys_size, offset);
          if (!padded_sys) {
#pragma unroll
            for (int i = 0; i < vec_length<REAL>; i++) {
              l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
          }
          store_array_reg_unaligned(d, &l_d, n, tid, sys_pads, sys_size,
                                    offset);
        }

        if (!padded_sys) {
          // Handle end of unaligned memory
          for (int i = n; i < sys_size + sys_off - 1; i++) {
            int loc_ind = ind_floor + i;
            d[loc_ind]  = d[loc_ind] - aa[loc_ind] * dd0 - cc[loc_ind] * ddn;
          }

          d[ind + sys_size - 1] = ddn;
        }
      }
    } else if (!padded_sys) {
      // Normal modified Thomas backwards pass if not optimized solve
      if (INC) {
        u[ind] += dd0;

        for (int i = 1; i < sys_size - 1; i++) {
          u[ind + i] += d[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }

        u[ind + sys_size - 1] += ddn;
      } else {
        d[ind] = dd0;

        for (int i = 1; i < sys_size - 1; i++) {
          d[ind + i] = d[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }

        d[ind + sys_size - 1] = ddn;
      }
    }
  }
}


//
// Kernel launch wrapper for forward step with register blocking
//
template <typename REAL, bool boundary_SOA = false>
void trid_linear_forward_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *a,
                             const REAL *b, const REAL *c, REAL *d, REAL *aa,
                             REAL *cc, REAL *boundaries, int sys_size,
                             int sys_pads, int sys_n, int offset, int start_sys,
                             int y_size, int y_pads, cudaStream_t stream) {
  const int aligned =
      (sys_pads % align<REAL>) == 0 && (offset % align<REAL>) == 0;
  if (aligned) {
    trid_linear_forward_aligned<REAL, boundary_SOA>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(a, b, c, d, aa, cc,
                                               boundaries, sys_size, sys_pads,
                                               sys_n, offset, start_sys, y_size,
                                               y_pads);
  } else {
    trid_linear_forward_unaligned<REAL, boundary_SOA>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(a, b, c, d, aa, cc,
                                               boundaries, sys_size, sys_pads,
                                               sys_n, offset, start_sys, y_size,
                                               y_pads);
  }
}

//
// Kernel launch wrapper for backward step with register blocking
//
template <typename REAL, int INC, bool boundary_SOA = false>
void trid_linear_backward_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *aa,
                              const REAL *cc, REAL *d, REAL *u,
                              const REAL *boundaries, int sys_size,
                              int sys_pads, int sys_n, int offset,
                              int start_sys, int y_size, int y_pads,
                              cudaStream_t stream) {
  const int aligned =
      (sys_pads % align<REAL>) == 0 && (offset % align<REAL>) == 0;
  if (aligned) {
    trid_linear_backward_aligned<REAL, INC, boundary_SOA>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa, cc, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
            start_sys, y_size, y_pads);
  } else {
    trid_linear_backward_unaligned<REAL, INC, boundary_SOA>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa, cc, d, u, boundaries, sys_size, sys_pads, sys_n, offset,
            start_sys, y_size, y_pads);
  }
}

#endif /* ifndef TRID_LINEAR_MPI_REG_HPP__ */
