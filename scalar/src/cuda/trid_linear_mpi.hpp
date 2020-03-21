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
// balogh.gabor.daniel@itk.ppke.hu, 2020
// Simple implementations of MPI solver along x dimension.

#ifndef TRID_LINEAR_GPU_MPI__
#define TRID_LINEAR_GPU_MPI__

/*
 * Modified Thomas forwards pass in x direction
 * Each array should have a size of sys_size*sys_n, although the first element
 * of a (a[0]) in the first process and the last element of c in the last
 * process will not be used eventually
 */
template <typename REAL>
__global__ void
trid_linear_forward(const REAL *__restrict__ a, const REAL *__restrict__ b,
                    const REAL *__restrict__ c, const REAL *__restrict__ d,
                    REAL *__restrict__ aa, REAL *__restrict__ cc,
                    REAL *__restrict__ dd, REAL *__restrict__ boundaries,
                    int sys_size, int sys_pads, int sys_n) {

  REAL bb;
  int i;

  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // forward pass
    //
    for (i = 0; i < 2; ++i) {
      bb = static_cast<REAL>(1.0) / b[ind + i];
      dd[ind + i] = bb * d[ind + i];
      aa[ind + i] = bb * a[ind + i];
      cc[ind + i] = bb * c[ind + i];
    }

    if (sys_size >= 3) {
      // eliminate lower off-diagonal
      for (i = 2; i < sys_size; i++) {
        int loc_ind = ind + i;
        bb = static_cast<REAL>(1.0) /
             (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
        dd[loc_ind] = (d[loc_ind] - a[loc_ind] * dd[loc_ind - 1]) * bb;
        aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
        cc[loc_ind] = c[loc_ind] * bb;
      }
      // Eliminate upper off-diagonal
      for (i = sys_size - 3; i > 0; --i) {
        int loc_ind = ind + i;
        dd[loc_ind] = dd[loc_ind] - cc[loc_ind] * dd[loc_ind + 1];
        aa[loc_ind] = aa[loc_ind] - cc[loc_ind] * aa[loc_ind + 1];
        cc[loc_ind] = -cc[loc_ind] * cc[loc_ind + 1];
      }
      bb = static_cast<REAL>(1.0) /
           (static_cast<REAL>(1.0) - cc[ind] * aa[ind + 1]);
      dd[ind] = bb * (dd[ind] - cc[ind] * dd[ind + 1]);
      aa[ind] = bb * aa[ind];
      cc[ind] = bb * (-cc[ind] * cc[ind + 1]);
    }
    // prepare boundaries for communication
    i = tid * 6;
    boundaries[i + 0] = aa[ind];
    boundaries[i + 1] = aa[ind + sys_size - 1];
    boundaries[i + 2] = cc[ind];
    boundaries[i + 3] = cc[ind + sys_size - 1];
    boundaries[i + 4] = dd[ind];
    boundaries[i + 5] = dd[ind + sys_size - 1];
  }
}

//
// Modified Thomas backward pass
//
template <typename REAL, int INC>
__global__ void
trid_linear_backward(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                     const REAL *__restrict__ dd, REAL *__restrict__ d,
                     REAL *__restrict__ u, const REAL *__restrict__ boundaries,
                     int sys_size, int sys_pads, int sys_n) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // reverse pass
    //
    REAL dd0 = boundaries[2 * tid], dd_last = boundaries[2 * tid + 1];
    if (INC)
      u[ind] += dd0;
    else
      d[ind] = dd0;

    for (int i = 1; i < sys_size - 1; i++) {
      REAL res = dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * dd_last;
      if (INC)
        u[ind + i] += res;
      else
        d[ind + i] = res;
    }
    if (INC)
      u[ind + sys_size - 1] += dd_last;
    else
      d[ind + sys_size - 1] = dd_last;
  }
}

#endif /* ifndef TRID_LINEAR_GPU_MPI__ */
