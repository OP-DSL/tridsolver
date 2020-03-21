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

#ifndef __TRID_CUDA_MPI_PCR_HPP
#define __TRID_CUDA_MPI_PCR_HPP

// An implementation of the PCR algorithm used to solve the reduced system.
// Requires that threads operating on one reduced system are within the same block.
// 'input' is the reduced systems gathered from the allgather
// 'results' is where the results of the reduced solve relevant to this mpi_coord are stored
// 'mpi_coord' is the index of the current MPI process along the solving dimension
// 'n' is the length of each reduced system
// 'P' is the number of PCR iterations
// 'sys_n' is the number of reduced systems in total
template<typename REAL>
__global__ void pcr_on_reduced_kernel(REAL *input, REAL *results, 
                                      const int mpi_coord, const int n, const int P, 
                                      const int sys_n) {
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  
  int tridNum = tid / n;
  int i = tid % n;
  
  
  // Indices of each coefficient in the 'input' array
  int a_ind = (6 * sys_n * (i / 2)) + (tridNum * 6) + (i % 2);
  int c_ind = (6 * sys_n * (i / 2)) + (tridNum * 6) + 2 + (i % 2);
  int d_ind = (6 * sys_n * (i / 2)) + (tridNum * 6) + 4 + (i % 2);

  // Check if thread is within bounds
  if(tridNum >= sys_n) {

    for(int p = 0; p < P; p++) {
      __syncthreads();
      __syncthreads();
    }
    
    return;
  }
  
  REAL a_m, a_p, c_m, c_p, d_m, d_p;

  int s = 1;
  
  // PCR iterations
  for(int p = 0; p < P; p++) {
    // Get the minus elements
    if(i - s < 0) {
      a_m = (REAL) 0.0;
      c_m = (REAL) 0.0;
      d_m = (REAL) 0.0;
    } else {
      int a_m_ind = (6 * sys_n * ((i - s) / 2)) + (tridNum * 6) + ((i - s) % 2);
      a_m = input[a_m_ind];
      int c_m_ind = (6 * sys_n * ((i - s) / 2)) + (tridNum * 6) + 2 + ((i - s) % 2);
      c_m = input[c_m_ind];
      int d_m_ind = (6 * sys_n * ((i - s) / 2)) + (tridNum * 6) + 4 + ((i - s) % 2);
      d_m = input[d_m_ind];
    }
    
    // Get the plus elements
    if(i + s >= n) {
      a_p = (REAL) 0.0;
      c_p = (REAL) 0.0;
      d_p = (REAL) 0.0;
    } else {
      int a_p_ind = (6 * sys_n * ((i + s) / 2)) + (tridNum * 6) + ((i + s) % 2);
      a_p = input[a_p_ind];
      int c_p_ind = (6 * sys_n * ((i + s) / 2)) + (tridNum * 6) + 2 + ((i + s) % 2);
      c_p = input[c_p_ind];
      int d_p_ind = (6 * sys_n * ((i + s) / 2)) + (tridNum * 6) + 4 + ((i + s) % 2);
      d_p = input[d_p_ind];
    }
    
    // Synchronise threads within block
    __syncthreads();
    
    // PCR algorithm
    REAL r = 1.0 - input[a_ind] * c_m - input[c_ind] * a_p;
    r = 1.0 / r;
    input[d_ind] = r * (input[d_ind] - input[a_ind] * d_m - input[c_ind] * d_p);
    input[a_ind] = -r * input[a_ind] * a_m;
    input[c_ind] = -r * input[c_ind] * c_p;
    
     // Synchronise threads within block
    __syncthreads();
    
    s = s << 1;
  }
  
  // Store results of this reduced solve that are relevant to this MPI process
  if(i >= 2 * mpi_coord && i < 2 * (mpi_coord + 1)) {
    int reduced_ind_l = i - (2 * mpi_coord);
    results[2 * tridNum + reduced_ind_l] = input[d_ind];
  }
}
#endif
