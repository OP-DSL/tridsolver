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

#ifndef __TRID_MPI_CPU_H
#define __TRID_MPI_CPU_H

#include "mpi.h"

template<typename REAL>
struct trid_handle {
  REAL *a;
  REAL *b;
  REAL *c;
  REAL *du;
  REAL *h_u;
  
  REAL *aa;
  REAL *cc;
  REAL *dd;
  
  REAL *aa_r;
  REAL *cc_r;
  REAL *dd_r;
  
  REAL *halo_sndbuf;
  REAL *halo_rcvbuf;
  
  int ndim;
  int *size;
  int *size_g;
  int *start_g;
  int *end_g;
  int *pads;
  int *sys_len_r;
  int *n_sys;
};

struct trid_mpi_handle {
  int procs;
  int rank;
  MPI_Comm comm;
  int ndim;
  int *pdims;
  int my_cart_rank;
  int *coords;
  int *periodic;
  
  MPI_Comm x_comm;
  MPI_Comm y_comm;
  MPI_Comm z_comm;
};

struct trid_timer {
  double timer;
  double elapsed_time_x[11];
  double elapsed_time_y[11];
  double elapsed_time_z[11];
};

template<typename REAL>
void tridInit(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, int ndim, int *size);

template<typename REAL>
void tridClean(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle);

template<typename REAL, int INC>
void tridBatch(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, int solveDim);

template<typename REAL, int INC>
void tridBatchTimed(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, 
                    trid_timer &timer_handle, int solveDim);

#endif
