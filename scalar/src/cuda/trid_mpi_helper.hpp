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

// This file contains helper functions for mpi solver routines
#ifndef TRID_MPI_HELPER_HPP__
#define TRID_MPI_HELPER_HPP__

/*
 * Copy first and last row of each system to boundaries
 */
template <typename REAL, bool boundary_SOA = false>
__device__ inline void
copy_boundaries_linear(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                       const REAL *__restrict__ dd,
                       REAL *__restrict__ boundaries, int tid, int sys_start,
                       int sys_size, int sys_n) {
  if (!boundary_SOA) {
    int i             = tid * 6;
    boundaries[i + 0] = aa[sys_start];
    boundaries[i + 1] = aa[sys_start + sys_size - 1];
    boundaries[i + 2] = cc[sys_start];
    boundaries[i + 3] = cc[sys_start + sys_size - 1];
    boundaries[i + 4] = dd[sys_start];
    boundaries[i + 5] = dd[sys_start + sys_size - 1];
  } else {
    boundaries[tid]             = aa[sys_start];
    boundaries[sys_n * 1 + tid] = cc[sys_start];
    boundaries[sys_n * 2 + tid] = dd[sys_start];
    boundaries[sys_n * 3 + tid] = aa[sys_start + sys_size - 1];
    boundaries[sys_n * 4 + tid] = cc[sys_start + sys_size - 1];
    boundaries[sys_n * 5 + tid] = dd[sys_start + sys_size - 1];
  }
}

template <typename REAL, bool boundary_SOA = false>
__device__ inline void
copy_boundaries_strided(const REAL *__restrict__ aa, int ind_a, int stride_a,
                        const REAL *__restrict__ cc, int ind_c, int stride_c,
                        const REAL *__restrict__ dd, int ind_d, int stride_d,
                        REAL *__restrict__ boundaries, int tid, int sys_size,
                        int sys_n) {
  if (!boundary_SOA) {
    int i             = tid * 6;
    boundaries[i + 0] = aa[ind_a];
    boundaries[i + 1] = aa[ind_a + (sys_size - 1) * stride_a];
    boundaries[i + 2] = cc[ind_c];
    boundaries[i + 3] = cc[ind_c + (sys_size - 1) * stride_c];
    boundaries[i + 4] = dd[ind_d];
    boundaries[i + 5] = dd[ind_d + (sys_size - 1) * stride_d];
  } else {
    boundaries[tid]             = aa[ind_a];
    boundaries[sys_n * 1 + tid] = cc[ind_c];
    boundaries[sys_n * 2 + tid] = dd[ind_d];
    boundaries[sys_n * 3 + tid] = aa[ind_a + (sys_size - 1) * stride_a];
    boundaries[sys_n * 4 + tid] = cc[ind_c + (sys_size - 1) * stride_c];
    boundaries[sys_n * 5 + tid] = dd[ind_d + (sys_size - 1) * stride_d];
  }
}

/*
 * Copy first and last  d value of each system to registers from boundary
 */
template <typename REAL, bool boundary_SOA = false>
__device__ inline void
load_d_from_boundary_linear(const REAL *__restrict__ boundaries, REAL &dd0,
                            REAL &ddn, int tid, int sys_n) {
  if (!boundary_SOA) {
    dd0 = boundaries[2 * tid], ddn = boundaries[2 * tid + 1];
  } else {
    dd0 = boundaries[2 * sys_n + tid], ddn = boundaries[5 * sys_n + tid];
  }
}
#endif
