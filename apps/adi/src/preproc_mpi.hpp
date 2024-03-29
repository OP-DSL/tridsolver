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

#include "mpi.h"
#include "trid_common.h"
#include "adi_mpi.h"

template<typename REAL>
struct preproc_handle {
  REAL *halo_snd_x;
  REAL *halo_rcv_x;
  REAL *halo_snd_y;
  REAL *halo_rcv_y;
  REAL *halo_snd_z;
  REAL *halo_rcv_z;

  REAL lambda;
};

template<typename REAL>
inline void preproc_mpi(preproc_handle<REAL> &pre_handle, app_handle &app) {
  int   i, j, k, ind;
  double elapsed, timer = 0.0;

  int nx = app.size[0];
  int ny = app.size[1];
  int nz = app.size[2];

  int padx = app.pads[0];
  int pady = app.pads[1];
  int padz = app.pads[2];

  const MPI_Datatype mpi_datatype = std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // Gather halo
  // X boundary
  for(k = 0; k < nz; k++) {
    for(j = 0; j < ny; j++) {
      pre_handle.halo_snd_x[0*nz*ny + k*ny + j] = app.u[k*pady*padx + j*padx + 0];
      pre_handle.halo_snd_x[1*nz*ny + k*ny + j] = app.u[k*pady*padx + j*padx + nx - 1];
    }
  }

  // Y boundary
  for(k = 0; k < nz; k++) {
    for(i = 0; i < nx; i++) {
      pre_handle.halo_snd_y[0*nz*nx + k*nx + i] = app.u[k*pady*padx + i + 0];
      pre_handle.halo_snd_y[1*nz*nx + k*nx + i] = app.u[k*pady*padx + i + (padx)*(ny-1)];
    }
  }

  // Z boundary
  for(j = 0; j < ny; j++) {
    for(i = 0; i < nx; i++) {
      pre_handle.halo_snd_z[0*ny*nx + j*nx + i] = app.u[j*padx + i + 0];
      pre_handle.halo_snd_z[1*ny*nx + j*nx + i] = app.u[j*padx + i + padx*pady*(nz-1)];
    }
  }

  // Send and receive halo
  // Send X Left
  if(app.coords[0] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = app.coords[0] - 1;
      dest_coords[1] = app.coords[1];
      dest_coords[2] = app.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(app.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_x[0*nz*ny], nz*ny, mpi_datatype, destination_rank,
               0, app.comm);
  }
  // Receive
  if(app.coords[0] < app.pdims[0] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = app.coords[0] + 1;
      source_coords[1] = app.coords[1];
      source_coords[2] = app.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(app.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_x[0*nz*ny], nz*ny, mpi_datatype, source_rank, 0,
               app.comm, MPI_STATUS_IGNORE);
  }

  // Send X Right
  if(app.coords[0] < app.pdims[0] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = app.coords[0] + 1;
      dest_coords[1] = app.coords[1];
      dest_coords[2] = app.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(app.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_x[1*nz*ny], nz*ny, mpi_datatype, destination_rank,
               0, app.comm);
  }
  // Receive
  if(app.coords[0] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = app.coords[0] - 1;
      source_coords[1] = app.coords[1];
      source_coords[2] = app.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(app.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_x[1*nz*ny], nz*ny, mpi_datatype, source_rank, 0,
               app.comm, MPI_STATUS_IGNORE);
  }

  // Send Y Backwards
  if(app.coords[1] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = app.coords[0];
      dest_coords[1] = app.coords[1] - 1;
      dest_coords[2] = app.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(app.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_y[0*nz*nx], nz*nx, mpi_datatype, destination_rank,
               0, app.comm);
  }
  // Receive
  if(app.coords[1] < app.pdims[1] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = app.coords[0];
      source_coords[1] = app.coords[1] + 1;
      source_coords[2] = app.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(app.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_y[0*nz*nx], nz*nx, mpi_datatype, source_rank, 0,
               app.comm, MPI_STATUS_IGNORE);
  }

  // Send Y Forwards
  if(app.coords[1] < app.pdims[1] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = app.coords[0];
      dest_coords[1] = app.coords[1] + 1;
      dest_coords[2] = app.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(app.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_y[1*nz*nx], nz*nx, mpi_datatype, destination_rank,
               0, app.comm);
  }
  // Receive
  if(app.coords[1] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = app.coords[0];
      source_coords[1] = app.coords[1] - 1;
      source_coords[2] = app.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(app.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_y[1*nz*nx], nz*nx, mpi_datatype, source_rank, 0,
               app.comm, MPI_STATUS_IGNORE);
  }

  // Send Z Below
  if(app.coords[2] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = app.coords[0];
      dest_coords[1] = app.coords[1];
      dest_coords[2] = app.coords[2] - 1;
      int destination_rank = 0;
      MPI_Cart_rank(app.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_z[0*ny*nx], nx*ny, mpi_datatype, destination_rank,
               0, app.comm);
  }
  // Receive
  if(app.coords[2] < app.pdims[2] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = app.coords[0];
      source_coords[1] = app.coords[1];
      source_coords[2] = app.coords[2] + 1;
      int source_rank = 0;
      MPI_Cart_rank(app.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_z[0*ny*nz], nx*ny, mpi_datatype, source_rank, 0,
               app.comm, MPI_STATUS_IGNORE);
  }

  // Send Z Above
  if(app.coords[2] < app.pdims[2] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = app.coords[0];
      dest_coords[1] = app.coords[1];
      dest_coords[2] = app.coords[2] + 1;
      int destination_rank = 0;
      MPI_Cart_rank(app.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_z[1*ny*nx], nx*ny, mpi_datatype, destination_rank,
               0, app.comm);
  }
  // Receive
  if(app.coords[2] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = app.coords[0];
      source_coords[1] = app.coords[1];
      source_coords[2] = app.coords[2] - 1;
      int source_rank = 0;
      MPI_Cart_rank(app.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_z[1*ny*nx], nx*ny, mpi_datatype, source_rank, 0,
               app.comm, MPI_STATUS_IGNORE);
  }

  REAL tmp, ux_1, ux_2, uy_1, uy_2, uz_1, uz_2;

  // Iterate over local arrays
  for(k = 0; k < nz; k++) {
    for(j = 0; j < ny; j++) {
      for(i = 0; i < nx; i++) {
        ind = k*padx*pady + j*padx + i;
        // Check if on global boundary
        if( (app.start_g[0]==0 && i==0) ||
            (app.end_g[0]==app.size_g[0]-1 && i==app.size[0]-1) ||
            (app.start_g[1]==0 && j==0) ||
            (app.end_g[1]==app.size_g[1]-1 && j==app.size[1]-1) ||
            (app.start_g[2]==0 && k==0) ||
            (app.end_g[2]==app.size_g[2]-1 && k==app.size[2]-1)) {

          app.d[ind] = 0.0f; // Dirichlet b.c.'s
          app.a[ind] = 0.0f;
          app.b[ind] = 1.0f;
          app.c[ind] = 0.0f;
        }
        else {

          if(i == 0) {
            ux_1 = pre_handle.halo_rcv_x[1*nz*ny + k*ny + j];
          } else {
            ux_1 = app.u[ind - 1];
          }

          if(i == nx - 1) {
            ux_2 = pre_handle.halo_rcv_x[0*nz*ny + k*ny + j];
          } else {
            ux_2 = app.u[ind + 1];
          }

          if(j == 0) {
            uy_1 = pre_handle.halo_rcv_y[1*nz*nx + k*nx + i];
          } else {
            uy_1 = app.u[ind - padx];
          }

          if(j == ny - 1) {
            uy_2 = pre_handle.halo_rcv_y[0*nz*nx + k*nx + i];
          } else {
            uy_2 = app.u[ind + padx];
          }

          if(k == 0) {
            uz_1 = pre_handle.halo_rcv_z[1*ny*nx + j*nx + i];
          } else {
            uz_1 = app.u[ind - padx*pady];
          }

          if(k == nz - 1) {
            uz_2 = pre_handle.halo_rcv_z[0*ny*nx + j*nx + i];
          } else {
            uz_2 = app.u[ind + padx*pady];
          }

          app.d[ind] = pre_handle.lambda*( ux_1 + ux_2
                                + uy_1 + uy_2
                                + uz_1 + uz_2
                                - 6.0f * app.u[ind]);

          app.a[ind] = -0.5f * pre_handle.lambda;
          app.b[ind] =  1.0f + pre_handle.lambda;
          app.c[ind] = -0.5f * pre_handle.lambda;
        }
      }
    }
  }
}
