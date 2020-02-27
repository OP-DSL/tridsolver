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

#include "trid_mpi_cpu.hpp"
#include "trid_simd.h"
#include "math.h"
#include "omp.h"

#include <type_traits>
#include <sys/time.h>

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

#define Z_BATCH 56

inline double elapsed_time(double *et) {
  struct timeval t;
  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
}

inline void timing_start(double *timer) {
  elapsed_time(timer);
}

inline void timing_end(double *timer, double *elapsed_accumulate) {
  double elapsed = elapsed_time(timer);
  *elapsed_accumulate += elapsed;
}

void setStartEnd(int *start, int *end, int coord, int numProcs, int numElements) {
  int tmp = numElements / numProcs;
  int remainder = numElements % numProcs;
  int total = 0;
  for(int i = 0; i < coord; i++) {
    if(i < remainder) {
      total += tmp + 1;
    } else {
      total += tmp;
    }
  }
  *start = total;
  if(coord < remainder) {
    *end = *start + tmp;
  } else {
    *end = *start + tmp -1;
  }
}

template<typename REAL>
void tridInit(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, int ndim, int *size) {
  // Get number of mpi procs and the rank of this mpi proc
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_handle.procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_handle.rank);
  
  // Split into multi dim arrangement of mpi procs
  handle.ndim = ndim;
  mpi_handle.pdims    = (int *) calloc(handle.ndim, sizeof(int));
  mpi_handle.periodic = (int *) calloc(handle.ndim, sizeof(int)); //false
  mpi_handle.coords   = (int *) calloc(handle.ndim, sizeof(int));
  MPI_Dims_create(mpi_handle.procs, handle.ndim, mpi_handle.pdims);
  
  // Reorder dims
  int tmp = mpi_handle.pdims[1];
  mpi_handle.pdims[1] = mpi_handle.pdims[2];
  mpi_handle.pdims[2] = tmp;
  
  // Create cartecian mpi comm
  MPI_Cart_create(MPI_COMM_WORLD, handle.ndim, mpi_handle.pdims, mpi_handle.periodic, 0,  &mpi_handle.comm);
  
  // Get rand and coord of current mpi proc
  MPI_Comm_rank(mpi_handle.comm, &mpi_handle.my_cart_rank);
  MPI_Cart_coords(mpi_handle.comm, mpi_handle.my_cart_rank, handle.ndim, mpi_handle.coords);
  
  // TODO extend to other dimensions
  // Create separate comms for x, y and z dimensions
  int free_coords[3];
  free_coords[0] = 1;
  free_coords[1] = 0;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.x_comm);
  MPI_Comm y_comm;
  free_coords[0] = 0;
  free_coords[1] = 1;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.y_comm);
  MPI_Comm z_comm;
  free_coords[0] = 0;
  free_coords[1] = 0;
  free_coords[2] = 1;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.z_comm);
  
  // Store the global problem sizes
  handle.size_g = (int *) calloc(handle.ndim, sizeof(int));
  for(int i = 0; i < handle.ndim; i++) {
    handle.size_g[i] = size[i];
  }
  
  // Calculate size, padding, start and end for each dimension
  handle.size    = (int *) calloc(handle.ndim, sizeof(int));
  handle.pads    = (int *) calloc(handle.ndim, sizeof(int));
  handle.start_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.end_g   = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    setStartEnd(&handle.start_g[i], &handle.end_g[i], mpi_handle.coords[i], mpi_handle.pdims[i], 
                handle.size_g[i]);

    handle.size[i]    = handle.end_g[i] - handle.start_g[i] + 1;
    
    // Only pad the x dimension
    if(i == 0) {
      handle.pads[i] = (1 + ((handle.size[i] - 1) / SIMD_VEC)) * SIMD_VEC;
    } else {
      handle.pads[i] = handle.size[i];
    }
  }
  
  // Allocate memory for arrays
  int mem_size = sizeof(REAL);
  for(int i = 0; i < handle.ndim; i++) {
    mem_size *= handle.pads[i];
  }
  
  handle.a   = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.b   = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.c   = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.du  = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.h_u = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.aa  = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.cc  = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.dd  = (REAL *) _mm_malloc(mem_size, SIMD_WIDTH);
  
  // Calculate reduced system sizes for each dimension
  handle.sys_len_r = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    handle.sys_len_r[i] = mpi_handle.pdims[i] * 2;
    handle.n_sys[i] = 1;
    for(int j = 0; j < handle.ndim; j++) {
      if(j != i) {
        handle.n_sys[i] *= handle.size[j];
      }
    }
  }
  
  // Allocate memory for container used to communicate reduced system
  int max = 0;
  for(int i = 0; i < handle.ndim; i++) {
    if(handle.n_sys[i] * handle.sys_len_r[i] > max) {
      max = handle.n_sys[i] * handle.sys_len_r[i];
    }
  }
  
  handle.halo_sndbuf = (REAL *) _mm_malloc(max * 3 * sizeof(REAL), SIMD_WIDTH);
  handle.halo_rcvbuf = (REAL *) _mm_malloc(max * 3 * sizeof(REAL), SIMD_WIDTH);
  
  // Allocate memory for reduced system arrays
  handle.aa_r = (REAL *) _mm_malloc(sizeof(REAL) * max, SIMD_WIDTH);
  handle.cc_r = (REAL *) _mm_malloc(sizeof(REAL) * max, SIMD_WIDTH);
  handle.dd_r = (REAL *) _mm_malloc(sizeof(REAL) * max, SIMD_WIDTH);
}

template<typename REAL>
void tridClean(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle) {
  free(mpi_handle.pdims);
  free(mpi_handle.periodic);
  free(mpi_handle.coords);
  free(handle.size_g);
  free(handle.size);
  free(handle.start_g);
  free(handle.end_g);
  _mm_free(handle.a);
  _mm_free(handle.b);
  _mm_free(handle.c);
  _mm_free(handle.du);
  _mm_free(handle.h_u);
  _mm_free(handle.aa);
  _mm_free(handle.cc);
  _mm_free(handle.dd);
  free(handle.sys_len_r);
  free(handle.n_sys);
  _mm_free(handle.halo_sndbuf);
  _mm_free(handle.halo_rcvbuf);
  _mm_free(handle.aa_r);
  _mm_free(handle.cc_r);
  _mm_free(handle.dd_r);
}

template<typename REAL, int INC>
void tridBatch(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, int solveDim) {
  if(solveDim == 0) {
    /*********************
     * 
     * X Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[0]; id++) {
      int base = id * handle.pads[0];
      thomas_forward<REAL>(&handle.a[base], &handle.b[base], &handle.c[base], &handle.du[base],
                     &handle.h_u[base], &handle.aa[base], &handle.cc[base],
                     &handle.dd[base], handle.size[0], 1);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[0]; id++) {
      // Gather coefficients of a,c,d
      int halo_base = id * 6;
      int data_base = id * handle.pads[0];
      handle.halo_sndbuf[halo_base]     = handle.aa[data_base];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[data_base];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[data_base];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[data_base + handle.size[0]-1];
    }
    
    // Communicate boundary values
    if(std::is_same<REAL, float>::value) {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[0]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys[0]*3*2, MPI_FLOAT, 0, mpi_handle.x_comm);
    } else {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[0]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys[0]*3*2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    }
    
    // Unpack boundary values
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys[0]; id++) {
          int halo_base = p * handle.n_sys[0] * 6 + id * 6;
          int data_base = id * handle.sys_len_r[0] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[0]; id++) {
        int base = id * handle.sys_len_r[0];
        thomas_on_reduced<REAL>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                          handle.sys_len_r[0], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys[0]; id++) {
          int halo_base = p * handle.n_sys[0] * 2 + id * 2;
          int data_base = id * handle.sys_len_r[0] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    if(std::is_same<REAL, float>::value) {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[0] * 2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys[0] * 2, MPI_FLOAT, 0, mpi_handle.x_comm);
    } else {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[0] * 2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys[0] * 2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    }
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[0]; id++) {
      // Gather coefficients of a,c,d
      int data_base = id * handle.pads[0];
      int halo_base = id * 2;
      handle.dd[data_base]                    = handle.halo_sndbuf[halo_base];
      handle.dd[data_base + handle.size[0]-1] = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    if(INC) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[0]; id++) {
        int ind = id * handle.pads[0];
        thomas_backwardInc<REAL>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                        &handle.h_u[ind], handle.size[0], 1);
      }
    } else {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[0]; id++) {
        int ind = id * handle.pads[0];
        thomas_backward<REAL>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                        &handle.du[ind], handle.size[0], 1);
      }
    }
  } else if(solveDim == 1) {
    /*********************
     * 
     * Y Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_forward_vec_strip<REAL>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[1],
                               handle.pads[0], /*handle.size[0]*/ handle.pads[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    if(std::is_same<REAL, float>::value) {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[1]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys[1]*3*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    } else {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[1]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys[1]*3*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    }
    
    // Unpack boundary values
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys[1]; id++) {
          int halo_base = p * handle.n_sys[1] * 6 + id * 6;
          int data_base = id * handle.sys_len_r[1] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[1]; id++) {
        int base = id * handle.sys_len_r[1];
        thomas_on_reduced<REAL>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_r[1], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys[1]; id++) {
          int halo_base = p * handle.n_sys[1] * 2 + id * 2;
          int data_base = id * handle.sys_len_r[1] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    if(std::is_same<REAL, float>::value) {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[1]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys[1]*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    } else {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[1]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys[1]*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    }
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    if(INC) {
      #pragma omp parallel for
      for(int z = 0; z < handle.size[2]; z++) {
        int base = z * handle.pads[0] * handle.pads[1];
        thomas_backwardInc_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.h_u[base], handle.size[1], handle.pads[0],
                                        /*handle.size[0]*/ handle.pads[0]);
      }
    } else {
      #pragma omp parallel for
      for(int z = 0; z < handle.size[2]; z++) {
        int base = z * handle.pads[0] * handle.pads[1];
        thomas_backward_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.du[base], handle.size[1], handle.pads[0],
                                        /*handle.size[0]*/ handle.pads[0]);
      }
    }
  } else {
    /*********************
     * 
     * Z Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int base = 0; base < ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH); base += Z_BATCH) {
      thomas_forward_vec_strip<REAL>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], Z_BATCH);
    }
    
    if(handle.size[1] * handle.pads[0] != ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH)) {
      int base = ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH);
      int length = (handle.size[1] * handle.pads[0]) - base;
      thomas_forward_vec_strip<REAL>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], length);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    if(std::is_same<REAL, float>::value) {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[2]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys[2]*3*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    } else {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[2]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys[2]*3*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    }
    
    // Unpack boundary data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys[2]; id++) {
          int data_base = id * handle.sys_len_r[2] + p * 2;
          int halo_base = p * handle.n_sys[2] * 6 + id * 6;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[2]; id++) {
        int base = id * handle.sys_len_r[2];
        thomas_on_reduced<REAL>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_r[2], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys[2]; id++) {
          int halo_base = p * handle.n_sys[2] * 2 + id * 2;
          int data_base = id * handle.sys_len_r[2] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    if(std::is_same<REAL, float>::value) {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[2]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys[2]*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    } else {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[2]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys[2]*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    }
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    if(INC) {
      #pragma omp parallel for
      for(int base = 0; base < ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH); base += Z_BATCH) {
        thomas_backwardInc_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.h_u[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], Z_BATCH);
      }
      
      if(handle.size[1] * handle.pads[0] != ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH)) {
        int base = ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH);
        int length = (handle.size[1] * handle.pads[0]) - base;
        thomas_backwardInc_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.h_u[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], length);
      }
    } else {
      #pragma omp parallel for
      for(int base = 0; base < ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH); base += Z_BATCH) {
        thomas_backward_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.du[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], Z_BATCH);
      }
      
      if(handle.size[1] * handle.pads[0] != ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH)) {
        int base = ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH);
        int length = (handle.size[1] * handle.pads[0]) - base;
        thomas_backward_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.du[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], length);
      }
    }
  }
}

template<typename REAL, int INC>
void tridBatchTimed(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, 
                    trid_timer &timer_handle, int solveDim) {
  if(solveDim == 0) {
    /*********************
     * 
     * X Dimension Solve
     * 
     *********************/
    
    timing_start(&timer_handle.timer);
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[0]; id++) {
      int base = id * handle.pads[0];
      thomas_forward<REAL>(&handle.a[base], &handle.b[base], &handle.c[base], &handle.du[base],
                     &handle.h_u[base], &handle.aa[base], &handle.cc[base],
                     &handle.dd[base], handle.size[0], 1);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[0]);
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[0]; id++) {
      // Gather coefficients of a,c,d
      int halo_base = id * 6;
      int data_base = id * handle.pads[0];
      handle.halo_sndbuf[halo_base]     = handle.aa[data_base];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[data_base];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[data_base];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[data_base + handle.size[0]-1];
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[1]);
    
    // Communicate boundary values
    if(std::is_same<REAL, float>::value) {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[0]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys[0]*3*2, MPI_FLOAT, 0, mpi_handle.x_comm);
    } else {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[0]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys[0]*3*2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[2]);
    
    // Unpack boundary values
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys[0]; id++) {
          int halo_base = p * handle.n_sys[0] * 6 + id * 6;
          int data_base = id * handle.sys_len_r[0] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[3]);
    
    // Compute reduced system
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[0]; id++) {
        int base = id * handle.sys_len_r[0];
        thomas_on_reduced<REAL>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                          handle.sys_len_r[0], 1);
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[4]);
    
    // Pack boundary solution data
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys[0]; id++) {
          int halo_base = p * handle.n_sys[0] * 2 + id * 2;
          int data_base = id * handle.sys_len_r[0] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[5]);
    
    // Send back new values
    if(std::is_same<REAL, float>::value) {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[0] * 2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys[0] * 2, MPI_FLOAT, 0, mpi_handle.x_comm);
    } else {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[0] * 2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys[0] * 2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[6]);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[0]; id++) {
      // Gather coefficients of a,c,d
      int data_base = id * handle.pads[0];
      int halo_base = id * 2;
      handle.dd[data_base]                    = handle.halo_sndbuf[halo_base];
      handle.dd[data_base + handle.size[0]-1] = handle.halo_sndbuf[halo_base + 1];
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[7]);
    
    // Do the backward pass of modified Thomas
    if(INC) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[0]; id++) {
        int ind = id * handle.pads[0];
        thomas_backwardInc<REAL>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                        &handle.h_u[ind], handle.size[0], 1);
      }
    } else {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[0]; id++) {
        int ind = id * handle.pads[0];
        thomas_backward<REAL>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                        &handle.du[ind], handle.size[0], 1);
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_x[8]);
    
  } else if(solveDim == 1) {
    /*********************
     * 
     * Y Dimension Solve
     * 
     *********************/
    
    timing_start(&timer_handle.timer);
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_forward_vec_strip<REAL>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[1],
                               handle.pads[0], /*handle.size[0]*/ handle.pads[0]);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[0]);
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[1]);
    
    // Communicate boundary values
    if(std::is_same<REAL, float>::value) {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[1]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys[1]*3*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    } else {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[1]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys[1]*3*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[2]);
    
    // Unpack boundary values
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys[1]; id++) {
          int halo_base = p * handle.n_sys[1] * 6 + id * 6;
          int data_base = id * handle.sys_len_r[1] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[3]);
    
    // Compute reduced system
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[1]; id++) {
        int base = id * handle.sys_len_r[1];
        thomas_on_reduced<REAL>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_r[1], 1);
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[4]);
    
    // Pack boundary solution data
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys[1]; id++) {
          int halo_base = p * handle.n_sys[1] * 2 + id * 2;
          int data_base = id * handle.sys_len_r[1] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[5]);
    
    // Send back new values
    if(std::is_same<REAL, float>::value) {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[1]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys[1]*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    } else {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[1]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys[1]*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[6]);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[7]);
    
    // Do the backward pass of modified Thomas
    if(INC) {
      #pragma omp parallel for
      for(int z = 0; z < handle.size[2]; z++) {
        int base = z * handle.pads[0] * handle.pads[1];
        thomas_backwardInc_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.h_u[base], handle.size[1], handle.pads[0],
                                        /*handle.size[0]*/ handle.pads[0]);
      }
    } else {
      #pragma omp parallel for
      for(int z = 0; z < handle.size[2]; z++) {
        int base = z * handle.pads[0] * handle.pads[1];
        thomas_backward_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.du[base], handle.size[1], handle.pads[0],
                                        /*handle.size[0]*/ handle.pads[0]);
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_y[8]);
    
  } else {
    /*********************
     * 
     * Z Dimension Solve
     * 
     *********************/
    
    timing_start(&timer_handle.timer);
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int base = 0; base < ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH); base += Z_BATCH) {
      thomas_forward_vec_strip<REAL>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], Z_BATCH);
    }
    
    if(handle.size[1] * handle.pads[0] != ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH)) {
      int base = ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH);
      int length = (handle.size[1] * handle.pads[0]) - base;
      thomas_forward_vec_strip<REAL>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], length);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[0]);
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[1]);
    
    // Communicate boundary values
    if(std::is_same<REAL, float>::value) {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[2]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys[2]*3*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    } else {
      MPI_Gather(handle.halo_sndbuf, handle.n_sys[2]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys[2]*3*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[2]);
    
    // Unpack boundary data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys[2]; id++) {
          int data_base = id * handle.sys_len_r[2] + p * 2;
          int halo_base = p * handle.n_sys[2] * 6 + id * 6;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[3]);
    
    // Compute reduced system
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys[2]; id++) {
        int base = id * handle.sys_len_r[2];
        thomas_on_reduced<REAL>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_r[2], 1);
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[4]);
    
    // Pack boundary solution data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys[2]; id++) {
          int halo_base = p * handle.n_sys[2] * 2 + id * 2;
          int data_base = id * handle.sys_len_r[2] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[5]);
    
    // Send back new values
    if(std::is_same<REAL, float>::value) {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[2]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys[2]*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    } else {
      MPI_Scatter(handle.halo_rcvbuf, handle.n_sys[2]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys[2]*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[6]);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[7]);
    
    // Do the backward pass of modified Thomas
    if(INC) {
      #pragma omp parallel for
      for(int base = 0; base < ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH); base += Z_BATCH) {
        thomas_backwardInc_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.h_u[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], Z_BATCH);
      }
      
      if(handle.size[1] * handle.pads[0] != ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH)) {
        int base = ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH);
        int length = (handle.size[1] * handle.pads[0]) - base;
        thomas_backwardInc_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.h_u[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], length);
      }
    } else {
      #pragma omp parallel for
      for(int base = 0; base < ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH); base += Z_BATCH) {
        thomas_backward_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.du[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], Z_BATCH);
      }
      
      if(handle.size[1] * handle.pads[0] != ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH)) {
        int base = ROUND_DOWN(handle.size[1] * handle.pads[0], Z_BATCH);
        int length = (handle.size[1] * handle.pads[0]) - base;
        thomas_backward_vec_strip<REAL>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                        &handle.du[base], handle.size[2], 
                                        handle.pads[0] * handle.pads[1], length);
      }
    }
    
    timing_end(&timer_handle.timer, &timer_handle.elapsed_time_z[8]);
  }
}

// Template instantiations
template void tridInit<float>(trid_handle<float> &handle, trid_mpi_handle &mpi_handle, 
                              int ndim, int *size);
template void tridInit<double>(trid_handle<double> &handle, trid_mpi_handle &mpi_handle, 
                               int ndim, int *size);
template void tridClean<float>(trid_handle<float> &handle, trid_mpi_handle &mpi_handle);
template void tridClean<double>(trid_handle<double> &handle, trid_mpi_handle &mpi_handle);
template void tridBatch<float, 0>(trid_handle<float> &handle, trid_mpi_handle &mpi_handle, 
                                  int solveDim);
template void tridBatch<float, 1>(trid_handle<float> &handle, trid_mpi_handle &mpi_handle, 
                                  int solveDim);
template void tridBatch<double, 0>(trid_handle<double> &handle, trid_mpi_handle &mpi_handle, 
                                   int solveDim);
template void tridBatch<double, 1>(trid_handle<double> &handle, trid_mpi_handle &mpi_handle, 
                                   int solveDim);
template void tridBatchTimed<float, 0>(trid_handle<float> &handle, trid_mpi_handle &mpi_handle,
                                       trid_timer &timer_handle, int solveDim);
template void tridBatchTimed<float, 1>(trid_handle<float> &handle, trid_mpi_handle &mpi_handle,
                                       trid_timer &timer_handle, int solveDim);
template void tridBatchTimed<double, 0>(trid_handle<double> &handle, trid_mpi_handle &mpi_handle,
                                       trid_timer &timer_handle, int solveDim);
template void tridBatchTimed<double, 1>(trid_handle<double> &handle, trid_mpi_handle &mpi_handle,
                                       trid_timer &timer_handle, int solveDim);
