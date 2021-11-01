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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <float.h>
#include <sys/time.h>

#include "tridsolver.h"
#include "trid_mpi_solver_params.hpp"
#include "trid_common.h"
#include "cutil_inline.h"

#include "omp.h"
//#include "offload.h"
#include "mpi.h"

#ifdef __MKL__
  //#include "lapacke.h"
  #include "mkl_lapacke.h"
  //#include "mkl.h"
#endif

#include "adi_mpi.h"
#include "preproc_mpi_cuda.hpp"

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"devid", required_argument, 0, 0   },
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"opt",  required_argument, 0,  0   },
  {"prof", required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {"b",   required_argument, 0,  0   },
  {"m", required_argument, 0,  0   },
  {0,      0,                 0,  0   }
};

// Function for calculating local problem size for a MPI process, as well as its
// global start and end indices.
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
    *end = *start + tmp - 1;
  }
}

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER [-opt CUDAOPT] -prof PROF\n");
  exit(0);
}

// Timing functions
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

// Function to add up a distributed array and print the result
void rms(const char* name, FP* array, app_handle &handle) {
  //Sum the square of values in app.h_u
  double sum = 0.0;
  for(int k = 0; k < handle.size[2]; k++) {
    for(int j = 0; j < handle.size[1]; j++) {
      for(int i = 0; i < handle.size[0]; i++) {
        int ind = k * handle.size[0] * handle.size[1] + j * handle.size[0] + i;
        sum += array[ind];
      }
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&sum, &global_sum,1, MPI_DOUBLE,MPI_SUM, handle.comm);

  if(handle.coords[0] == 0 && handle.coords[1] == 0 && handle.coords[2] == 0) {
    printf("%s sum = %.15g\n", name, global_sum);
  }

}

// Initialize the ADI application
int init(app_handle &app, preproc_handle<FP> &pre_handle, int &iter, int argc, char* argv[]) {
  if( MPI_Init(&argc,&argv) != MPI_SUCCESS) { printf("MPI Couldn't initialize. Exiting"); exit(-1);}

  int devid = 0;
  int nx_g = 256;
  int ny_g = 256;
  int nz_g = 256;
  iter = 10;
  int opt  = 0;
  int prof = 1;
  int batchSize = 16384;
  int m = 1;

  pre_handle.lambda = 1.0f;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"devid") == 0) devid = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"prof") == 0) prof = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
    if(strcmp((char*)options[opt_index].name,"b" ) == 0) batchSize = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"m" ) == 0) m      = atoi(optarg);
  }

  // Allocate memory to store problem characteristics
  app.size_g = (int *) calloc(3, sizeof(int));
  app.size = (int *) calloc(3, sizeof(int));
  app.start_g = (int *) calloc(3, sizeof(int));
  app.end_g = (int *) calloc(3, sizeof(int));

  app.size_g[0] = nx_g;
  app.size_g[1] = ny_g;
  app.size_g[2] = nz_g;

  // Set up MPI for tridiagonal solver
  int procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create 3D Cartesian MPI topology
  app.pdims    = (int *) calloc(3, sizeof(int));
  int *periodic = (int *) calloc(3, sizeof(int)); //false
  app.coords   = (int *) calloc(3, sizeof(int));
  MPI_Dims_create(procs, 3, app.pdims);

  // Setup up which GPU this MPI process is using
  // Currently set for 4 GPUs per node, with 1 MPI process per GPU
  //devid = rank % 4;
  cudaSafeCall( cudaSetDevice(devid) );
  cutilDeviceInit(argc, argv);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  // Create 3D Cartesian MPI communicator
  MPI_Cart_create(MPI_COMM_WORLD, 3, app.pdims, periodic, 0,  &app.comm);

  int my_cart_rank;

  MPI_Comm_rank(app.comm, &my_cart_rank);
  MPI_Cart_coords(app.comm, my_cart_rank, 3, app.coords);

  // Create MPI handle used by tridiagonal solver
  switch(m) {
    case 1:
      app.params = new MpiSolverParams(app.comm, 3, app.pdims, batchSize, MpiSolverParams::ALLGATHER);
      break;
    case 2:
      app.params = new MpiSolverParams(app.comm, 3, app.pdims, batchSize, MpiSolverParams::LATENCY_HIDING_TWO_STEP);
      break;
    case 3:
      app.params = new MpiSolverParams(app.comm, 3, app.pdims, batchSize, MpiSolverParams::LATENCY_HIDING_INTERLEAVED);
      break;
    case 4:
      app.params = new MpiSolverParams(app.comm, 3, app.pdims, batchSize, MpiSolverParams::JACOBI);
      break;
    case 5:
      app.params = new MpiSolverParams(app.comm, 3, app.pdims, batchSize, MpiSolverParams::PCR);
      break;
    default:
      exit(-1);
  }

  // Calculate local problem size for this MPI process
  for(int i = 0; i < 3; i++) {
    setStartEnd(&app.start_g[i], &app.end_g[i], app.coords[i], app.pdims[i], app.size_g[i]);
    app.size[i] = app.end_g[i] - app.start_g[i] + 1;
  }

  free(periodic);

  if(rank==0) {
    printf("\nGlobal grid dimensions: %d x %d x %d\n",
           app.size_g[0], app.size_g[1], app.size_g[2]);

    printf("\nNumber of MPI procs in each dimenstion %d, %d, %d\n",
           app.pdims[0], app.pdims[1], app.pdims[2]);
  }

  // Allocate memory for local section of problem
  int size = app.size[0] * app.size[1] * app.size[2];

  cudaSafeCall( cudaMalloc((void **)&app.a, size * sizeof(FP)) );
  cudaSafeCall( cudaMalloc((void **)&app.b, size * sizeof(FP)) );
  cudaSafeCall( cudaMalloc((void **)&app.c, size * sizeof(FP)) );
  cudaSafeCall( cudaMalloc((void **)&app.d, size * sizeof(FP)) );
  cudaSafeCall( cudaMalloc((void **)&app.u, size * sizeof(FP)) );

  FP *h_u = (FP *) malloc(sizeof(FP) * size);

  // Initialize
  for(int k = 0; k < app.size[2]; k++) {
    for(int j = 0; j < app.size[1]; j++) {
      for(int i = 0; i < app.size[0]; i++) {
        int ind = k * app.size[0] * app.size[1] + j*app.size[0] + i;
        if( (app.start_g[0]==0 && i==0) ||
            (app.end_g[0]==app.size_g[0]-1 && i==app.size[0]-1) ||
            (app.start_g[1]==0 && j==0) ||
            (app.end_g[1]==app.size_g[1]-1 && j==app.size[1]-1) ||
            (app.start_g[2]==0 && k==0) ||
            (app.end_g[2]==app.size_g[2]-1 && k==app.size[2]-1)) {
          h_u[ind] = 1.0f;
        } else {
          h_u[ind] = 0.0f;
        }
      }
    }
  }

  // Copy initial values to GPU memory
  cudaSafeCall( cudaMemcpy(app.u, h_u, sizeof(FP) * size, cudaMemcpyHostToDevice) );

  free(h_u);

  // Allocate memory used in each iteration's preprocessing
  pre_handle.rcv_size_x = 2 * app.size[1] * app.size[2];
  pre_handle.rcv_size_y = 2 * app.size[0] * app.size[2];
  pre_handle.rcv_size_z = 2 * app.size[1] * app.size[0];

  pre_handle.halo_snd_x = (FP*) malloc(pre_handle.rcv_size_x * sizeof(FP));
  pre_handle.halo_rcv_x = (FP*) malloc(pre_handle.rcv_size_x * sizeof(FP));
  pre_handle.halo_snd_y = (FP*) malloc(pre_handle.rcv_size_y * sizeof(FP));
  pre_handle.halo_rcv_y = (FP*) malloc(pre_handle.rcv_size_y * sizeof(FP));
  pre_handle.halo_snd_z = (FP*) malloc(pre_handle.rcv_size_z * sizeof(FP));
  pre_handle.halo_rcv_z = (FP*) malloc(pre_handle.rcv_size_z * sizeof(FP));

  cudaSafeCall( cudaMalloc((void **)&pre_handle.rcv_x, pre_handle.rcv_size_x * sizeof(FP)) );
  cudaSafeCall( cudaMalloc((void **)&pre_handle.rcv_y, pre_handle.rcv_size_y * sizeof(FP)) );
  cudaSafeCall( cudaMalloc((void **)&pre_handle.rcv_z, pre_handle.rcv_size_z * sizeof(FP)) );

  return 0;

}

// Free memory used
void finalize(app_handle &app, preproc_handle<FP> &pre_handle) {
  free(pre_handle.halo_snd_x);
  free(pre_handle.halo_rcv_x);
  free(pre_handle.halo_snd_y);
  free(pre_handle.halo_rcv_y);
  free(pre_handle.halo_snd_z);
  free(pre_handle.halo_rcv_z);
  cudaSafeCall( cudaFree(pre_handle.rcv_x) );
  cudaSafeCall( cudaFree(pre_handle.rcv_y) );
  cudaSafeCall( cudaFree(pre_handle.rcv_z) );

  cudaSafeCall( cudaFree(app.a) );
  cudaSafeCall( cudaFree(app.b) );
  cudaSafeCall( cudaFree(app.c) );
  cudaSafeCall( cudaFree(app.d) );
  cudaSafeCall( cudaFree(app.u) );

  free(app.size_g);
  free(app.size);
  free(app.start_g);
  free(app.end_g);
  free(app.pdims);
  free(app.coords);

  delete app.params;
}

int main(int argc, char* argv[]) {
  app_handle app;
  preproc_handle<FP> pre_handle;
  int iter;
  // Initialize
  init(app, pre_handle, iter, argc, argv);

  // Declare and reset elapsed time counters
  double timer           = 0.0;
  double timer1          = 0.0;
  double elapsed_total   = 0.0;
  double elapsed_preproc = 0.0;
  double elapsed_trid_x  = 0.0;
  double elapsed_trid_y  = 0.0;
  double elapsed_trid_z  = 0.0;

  timing_start(&timer1);

  // Allocate memory used in sums of distributed arrays
  FP *h_u = (FP *) malloc(sizeof(FP) * app.size[0] * app.size[1] * app.size[2]);
  FP *du = (FP *) malloc(sizeof(FP) * app.size[0] * app.size[1] * app.size[2]);

  TridParams trid_params;
  trid_params.mpi_params = (void *)app.params;

  // Iterate over specified number of time steps
  for(int it = 0; it < iter; it++) {
    // Preprocess
    timing_start(&timer);

    preproc_mpi_cuda<FP>(pre_handle, app);

    timing_end(&timer, &elapsed_preproc);

    cudaSafeCall( cudaDeviceSynchronize() );

    //
    // perform tri-diagonal solves in x-direction
    //
    timing_start(&timer);
#if FPPREC == 0
    tridSmtsvStridedBatch(&trid_params, app.a, app.b, app.c, app.d, 3, 0, app.size, app.size);
#else
    tridDmtsvStridedBatch(&trid_params, app.a, app.b, app.c, app.d, 3, 0, app.size, app.size);
#endif

    timing_end(&timer, &elapsed_trid_x);

    //
    // perform tri-diagonal solves in y-direction
    //
    timing_start(&timer);
#if FPPREC == 0
    tridSmtsvStridedBatch(&trid_params, app.a, app.b, app.c, app.d, 3, 1, app.size, app.size);
#else
    tridDmtsvStridedBatch(&trid_params, app.a, app.b, app.c, app.d, 3, 1, app.size, app.size);
#endif
    timing_end(&timer, &elapsed_trid_y);

    //
    // perform tri-diagonal solves in z-direction
    //
    timing_start(&timer);
#if FPPREC == 0
    tridSmtsvStridedBatchInc(&trid_params, app.a, app.b, app.c, app.d, app.u, 3, 2, app.size, app.size);
#else
    tridDmtsvStridedBatchInc(&trid_params, app.a, app.b, app.c, app.d, app.u, 3, 2, app.size, app.size);
#endif
    timing_end(&timer, &elapsed_trid_z);
  }

  timing_end(&timer1, &elapsed_total);

  // Print sum of these arrays (basic error checking)
  cudaSafeCall( cudaMemcpy(h_u, app.u, sizeof(FP) * app.size[0] * app.size[1] * app.size[2], cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaMemcpy(du, app.d, sizeof(FP) * app.size[0] * app.size[1] * app.size[2], cudaMemcpyDeviceToHost) );

  rms("end h", h_u, app);
  rms("end d", du, app);

  MPI_Barrier(MPI_COMM_WORLD);

  free(h_u);
  free(du);

  MPI_Barrier(MPI_COMM_WORLD);

  // Print out timings of each section
  if(app.coords[0] == 0 && app.coords[1] == 0 && app.coords[2] == 0) {
    // Print execution times
    printf("Time per section: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n");
    printf("%e \t%e \t%e \t%e \t%e\n",
        elapsed_total,
        elapsed_preproc,
        elapsed_trid_x,
        elapsed_trid_y,
        elapsed_trid_z);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Free memory
  finalize(app, pre_handle);

  MPI_Finalize();
  cudaDeviceReset();
  return 0;

}
