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

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014

//#include"adi_simd.h"
#include"trid_simd.h"

#include "adi_mpi.h"
#include "mpi.h"

template<typename REAL>
//inline void preproc_mpi(REAL lambda, REAL* __restrict u, REAL* __restrict du, REAL* __restrict ax, REAL* __restrict bx, REAL* __restrict cx, REAL* __restrict ay, REAL* __restrict by, REAL* __restrict cy, REAL* __restrict az, REAL* __restrict bz, REAL* __restrict cz, int nx, int nx_pad, int ny, int nz, int ny_g, int nz_g, int nx_g, int x_start_g, int x_end_g) {
inline void preproc_mpi(REAL lambda, REAL* __restrict u, REAL* __restrict du, REAL* __restrict ax, REAL* __restrict bx, REAL* __restrict cx, REAL* __restrict ay, REAL* __restrict by, REAL* __restrict cy, REAL* __restrict az, REAL* __restrict bz, REAL* __restrict cz, app_handle &app, mpi_handle &mpi) {
  int   i, j, k, ind;
  REAL a, b, c, d;
  double elapsed, timer = 0.0;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //
  //#ifndef VALID
  //  #pragma omp parallel for collapse(2) private(i,ind,a,b,c,d)
  //#endif

  //  REAL *halo_sndbuf = (REAL*) _mm_malloc(2*app.ny*app.nz*sizeof(REAL),SIMD_WIDTH); // Send Buffer
  //  REAL *halo_rcvbuf = (REAL*) _mm_malloc(2*app.ny*app.nz*sizeof(REAL),SIMD_WIDTH); // Receive Buffer

  //timing_start(app.prof, &timer);
  // Gather halo
  // X boundary
  for(k=0; k<app.nz; k++) {
    for(j=0; j<app.ny; j++) {
      mpi.halo_snd_x[0*app.nz*app.ny + k*app.ny + j] = u[k*app.ny*app.nx_pad + j*app.nx_pad +        0];
      mpi.halo_snd_x[1*app.nz*app.ny + k*app.ny + j] = u[k*app.ny*app.nx_pad + j*app.nx_pad + app.nx-1];
    }
  }
  
  // Y boundary
  for(k=0; k<app.nz; k++) {
    for(i=0; i<app.nx; i++) {
      mpi.halo_snd_y[0*app.nz*app.nx + k*app.nx + i] = u[k*app.ny*app.nx_pad + i +        0];
      mpi.halo_snd_y[1*app.nz*app.nx + k*app.nx + i] = u[k*app.ny*app.nx_pad + i + (app.nx_pad)*(app.ny-1)];
    }
  }
  
  // Z boundary
  for(j=0; j<app.ny; j++) {
    for(i=0; i<app.nx; i++) {
      mpi.halo_snd_z[0*app.ny*app.nx + j*app.nx + i] = u[j*app.nx_pad + i +        0];
      mpi.halo_snd_z[1*app.ny*app.nx + j*app.nx + i] = u[j*app.nx_pad + i + (app.nx_pad)*(app.ny)*(app.nz-1)];
    }
  }
  
  // Send and receive halo
  // Send X Left
  if(mpi.coords[0] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi.coords[0] - 1;
      dest_coords[1] = mpi.coords[1];
      dest_coords[2] = mpi.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&mpi.halo_snd_x[0*app.nz*app.ny], app.nz*app.ny, MPI_DOUBLE, destination_rank, 0, mpi.comm);
  }
  // Receive
  if(mpi.coords[0] < mpi.pdims[0] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi.coords[0] + 1;
      source_coords[1] = mpi.coords[1];
      source_coords[2] = mpi.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi.comm, source_coords, &source_rank);
      MPI_Recv(&mpi.halo_rcv_x[0*app.nz*app.ny], app.nz*app.ny, MPI_DOUBLE, source_rank, 0, mpi.comm, mpi.stat);
  }

  // Send X Right
  if(mpi.coords[0] < mpi.pdims[0] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi.coords[0] + 1;
      dest_coords[1] = mpi.coords[1];
      dest_coords[2] = mpi.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&mpi.halo_snd_x[1*app.nz*app.ny], app.nz*app.ny, MPI_DOUBLE, destination_rank, 0, mpi.comm);
  }
  // Receive
  if(mpi.coords[0] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi.coords[0] - 1;
      source_coords[1] = mpi.coords[1];
      source_coords[2] = mpi.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi.comm, source_coords, &source_rank);
      MPI_Recv(&mpi.halo_rcv_x[1*app.nz*app.ny], app.nz*app.ny, MPI_DOUBLE, source_rank, 0, mpi.comm, mpi.stat);
  }
  
  // Send Y Backwards
  if(mpi.coords[1] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi.coords[0];
      dest_coords[1] = mpi.coords[1] - 1;
      dest_coords[2] = mpi.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&mpi.halo_snd_y[0*app.nz*app.nx], app.nz*app.nx, MPI_DOUBLE, destination_rank, 0, mpi.comm);
  }
  // Receive
  if(mpi.coords[1] < mpi.pdims[1] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi.coords[0];
      source_coords[1] = mpi.coords[1] + 1;
      source_coords[2] = mpi.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi.comm, source_coords, &source_rank);
      MPI_Recv(&mpi.halo_rcv_y[0*app.nz*app.nx], app.nz*app.nx, MPI_DOUBLE, source_rank, 0, mpi.comm, mpi.stat);
  }
  
  // Send Y Forwards
  if(mpi.coords[1] < mpi.pdims[1] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi.coords[0];
      dest_coords[1] = mpi.coords[1] + 1;
      dest_coords[2] = mpi.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&mpi.halo_snd_y[1*app.nz*app.nx], app.nz*app.nx, MPI_DOUBLE, destination_rank, 0, mpi.comm);
  }
  // Receive
  if(mpi.coords[1] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi.coords[0];
      source_coords[1] = mpi.coords[1] - 1;
      source_coords[2] = mpi.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi.comm, source_coords, &source_rank);
      MPI_Recv(&mpi.halo_rcv_y[1*app.nz*app.nx], app.nz*app.nx, MPI_DOUBLE, source_rank, 0, mpi.comm, mpi.stat);
  }
  
  // Send Z Below
  if(mpi.coords[2] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi.coords[0];
      dest_coords[1] = mpi.coords[1];
      dest_coords[2] = mpi.coords[2] - 1;
      int destination_rank = 0;
      MPI_Cart_rank(mpi.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&mpi.halo_snd_z[0*app.ny*app.nx], app.nx*app.ny, MPI_DOUBLE, destination_rank, 0, mpi.comm);
  }
  // Receive
  if(mpi.coords[2] < mpi.pdims[2] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi.coords[0];
      source_coords[1] = mpi.coords[1];
      source_coords[2] = mpi.coords[2] + 1;
      int source_rank = 0;
      MPI_Cart_rank(mpi.comm, source_coords, &source_rank);
      MPI_Recv(&mpi.halo_rcv_z[0*app.ny*app.nz], app.nx*app.ny, MPI_DOUBLE, source_rank, 0, mpi.comm, mpi.stat);
  }
  
  // Send Z Above
  if(mpi.coords[2] < mpi.pdims[2] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi.coords[0];
      dest_coords[1] = mpi.coords[1];
      dest_coords[2] = mpi.coords[2] + 1;
      int destination_rank = 0;
      MPI_Cart_rank(mpi.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&mpi.halo_snd_z[1*app.ny*app.nx], app.nx*app.ny, MPI_DOUBLE, destination_rank, 0, mpi.comm);
  }
  // Receive
  if(mpi.coords[2] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi.coords[0];
      source_coords[1] = mpi.coords[1];
      source_coords[2] = mpi.coords[2] - 1;
      int source_rank = 0;
      MPI_Cart_rank(mpi.comm, source_coords, &source_rank);
      MPI_Recv(&mpi.halo_rcv_z[1*app.ny*app.nx], app.nx*app.ny, MPI_DOUBLE, source_rank, 0, mpi.comm, mpi.stat);
  }
  
  /*if(mpi.rank > 0) {
    //printf("SENDING mpirank = %d  left buffer\n",mpi.rank);
    MPI_Isend(&mpi.halo_sndbuf2[0*app.nz*app.ny], app.nz*app.ny, MPI_FLOAT, mpi.rank-1, 0, MPI_COMM_WORLD, mpi.req);
    //printf("Done\n");
  }
  if(mpi.rank < mpi.procs-1) {
    //printf("SENDING mpirank = %d  right buffer\n",mpi.rank);
    MPI_Isend(&mpi.halo_sndbuf2[1*app.nz*app.ny], app.nz*app.ny, MPI_FLOAT, mpi.rank+1, 1, MPI_COMM_WORLD, mpi.req);
    //printf("Done\n");
  }
  // Receive halo
  if(mpi.rank < mpi.procs-1)
    MPI_Recv(&mpi.halo_rcvbuf2[1*app.nz*app.ny], app.nz*app.ny, MPI_FLOAT, mpi.rank+1, 0, MPI_COMM_WORLD, mpi.stat);
  if(mpi.rank > 0)
    MPI_Recv(&mpi.halo_rcvbuf2[0*app.nz*app.ny], app.nz*app.ny, MPI_FLOAT, mpi.rank-1, 1, MPI_COMM_WORLD, mpi.stat);*/
  
  //timing_end(app.prof, &timer, &app.elapsed_time[9], app.elapsed_name[9]);

  REAL tmp, ux_1, ux_2, uy_1, uy_2, uz_1, uz_2;

  //timing_start(app.prof, &timer);
  for(k=0; k<app.nz; k++) {
    for(j=0; j<app.ny; j++) {
      for(i=0; i<app.nx; i++) {   // i loop innermost for sequential memory access
        ind = k*app.nx_pad*app.ny + j*app.nx_pad + i;
        if( (app.x_start_g==0 && i==0) || (app.x_end_g==app.nx_g-1 && i==app.nx-1) ||
            (app.y_start_g==0 && j==0) || (app.y_end_g==app.ny_g-1 && j==app.ny-1) ||
            (app.z_start_g==0 && k==0) || (app.z_end_g==app.nz_g-1 && k==app.nz-1) ) {

          d = 0.0f; // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          
          if(i == 0) {
            ux_1 = mpi.halo_rcv_x[1*app.nz*app.ny + k*app.ny + j];
          } else {
            ux_1 = u[ind - 1];
          }
          
          if(i == app.nx - 1) {
            ux_2 = mpi.halo_rcv_x[0*app.nz*app.ny + k*app.ny + j];
          } else {
            ux_2 = u[ind + 1];
          }
          
          if(j == 0) {
            uy_1 = mpi.halo_rcv_y[1*app.nz*app.nx + k*app.nx + i];
          } else {
            uy_1 = u[ind - app.nx_pad];
          }
          
          if(j == app.ny - 1) {
            uy_2 = mpi.halo_rcv_y[0*app.nz*app.nx + k*app.nx + i];
          } else {
            uy_2 = u[ind + app.nx_pad];
          }
          
          if(k == 0) {
            uz_1 = mpi.halo_rcv_z[1*app.ny*app.nx + j*app.nx + i];
          } else {
            uz_1 = u[ind - app.nx_pad*app.ny];
          }
          
          if(k == app.nz - 1) {
            uz_2 = mpi.halo_rcv_z[0*app.ny*app.nx + j*app.nx + i];
          } else {
            uz_2 = u[ind + app.nx_pad*app.ny];
          }
          
          /*if(i==0 && mpi.coords[0]>0) {
            tmp = mpi.halo_rcvbuf2[0*app.nz*app.ny + k*app.ny + j];
            d = lambda*(  tmp                        + u[ind+1                  ]
                        + u[ind-app.nx_pad         ] + u[ind+app.nx_pad         ]
                        + u[ind-app.nx_pad*app.ny] + u[ind+app.nx_pad*app.ny]
                        - 6.0f*u[ind]);
          } else if(i==app.nx-1 && mpi.coords[0]<mpi.pdims[0]-1) {
            tmp = mpi.halo_rcvbuf2[1*app.nz*app.ny + k*app.ny + j];
            d = lambda*(  u[ind-1]                   + tmp
                        + u[ind-app.nx_pad         ] + u[ind+app.nx_pad         ]
                        + u[ind-app.nx_pad*app.ny] + u[ind+app.nx_pad*app.ny]
                        - 6.0f*u[ind]);
          } else {
            d = lambda*(  u[ind-1                  ] + u[ind+1                  ]
                        + u[ind-app.nx_pad         ] + u[ind+app.nx_pad         ]
                        + u[ind-app.nx_pad*app.ny] + u[ind+app.nx_pad*app.ny]
                        - 6.0f*u[ind]);
          }*/
          d = lambda*(  ux_1 + ux_2
                      + uy_1 + uy_2
                      + uz_1 + uz_2
                      - 6.0f*u[ind]);
          

          a = -0.5f * lambda;
          b =  1.0f + lambda;
          c = -0.5f * lambda;
        }
        du[ind] = d;
        ax[ind] = a;
        bx[ind] = b;
        cx[ind] = c;
        ay[ind] = a;
        by[ind] = b;
        cy[ind] = c;
        az[ind] = a;
        bz[ind] = b;
        cz[ind] = c;
      }
    }
  }
  //timing_end(app.prof, &timer, &app.elapsed_time[10], app.elapsed_name[10]);
}
