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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <float.h>
#include <sys/time.h>

#define FP double

#include "preproc_mpi.hpp"
#include "trid_mpi_cpu.h"

#include "trid_common.h"

#include "omp.h"
//#include "offload.h"
#include "mpi.h"

#ifdef __MKL__
  //#include "lapacke.h"
  #include "mkl_lapacke.h"
  //#include "mkl.h"
#endif

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"opt",  required_argument, 0,  0   },
  {"prof", required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER [-opt CUDAOPT] -prof PROF\n");
  exit(0);
}

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

void rms(char* name, FP* array, trid_handle<FP> &handle, trid_mpi_handle &mpi_handle) {
  //Sum the square of values in app.h_u
  double sum = 0.0;
  for(int k = 0; k < handle.size[2]; k++) {
    for(int j = 0; j < handle.size[1]; j++) {
      for(int i = 0; i < handle.size[0]; i++) {
        int ind = k * handle.pads[0] * handle.pads[1] + j * handle.pads[0] + i;
        //sum += array[ind]*array[ind];
        sum += array[ind];
      }
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&sum, &global_sum,1, MPI_DOUBLE,MPI_SUM, mpi_handle.comm);

  if(mpi_handle.rank ==0) {
    printf("%s sum = %lg\n", name, global_sum);
    //printf("%s rms = %2.15lg\n",name, sqrt(global_sum)/((double)(app.nx_g*app.ny_g*app.nz_g)));
  }

}
/*
void print_array_onrank(int rank, FP* array, app_handle &app, mpi_handle &mpi) {
  if(mpi.rank == rank) {
    printf("On mpi rank %d\n",rank);
    for(int k=0; k<2; k++) {
        printf("k = %d\n",k);
        for(int j=0; j<MIN(app.ny,17); j++) {
          printf(" %d   ", j);
          for(int i=0; i<MIN(app.nx,17); i++) {
            int ind = k*app.nx_pad*app.ny + j*app.nx_pad + i;
            printf(" %5.5g ", array[ind]);
          }
          printf("\n");
        }
        printf("\n");
      }
  }
}*/

int init(trid_handle<FP> &trid_handle, trid_mpi_handle &mpi_handle, preproc_handle<FP> &pre_handle, int &iter, int argc, char* argv[]) {
  if( MPI_Init(&argc,&argv) != MPI_SUCCESS) { printf("MPI Couldn't initialize. Exiting"); exit(-1);}

  //int nx, ny, nz, iter, opt, prof;
  int nx_g = 256;
  int ny_g = 256;
  int nz_g = 256;
  iter = 10;
  int opt  = 0;
  int prof = 1;

  pre_handle.lambda = 1.0f;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"prof") == 0) prof = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
  }
  
  int size[3] = {nx_g, ny_g, nz_g};
  
  tridInit<FP>(trid_handle, mpi_handle, 3, size);

  if(mpi_handle.rank==0) {
    printf("\nGlobal grid dimensions: %d x %d x %d\n", 
           trid_handle.size_g[0], trid_handle.size_g[1], trid_handle.size_g[2]);

    printf("\nNumber of MPI procs in each dimenstion %d, %d, %d\n",
           mpi_handle.pdims[0], mpi_handle.pdims[1], mpi_handle.pdims[2]);
  }

  printf("Check parameters: SIMD_WIDTH = %d, sizeof(FP) = %d\n", SIMD_WIDTH, sizeof(FP));
  printf("Check parameters: nx_pad (padded) = %d\n", trid_handle.pads[0]);
  printf("Check parameters: nx = %d, x_start_g = %d, x_end_g = %d \n", 
         trid_handle.size[0], trid_handle.start_g[0], trid_handle.end_g[0]);
  printf("Check parameters: ny = %d, y_start_g = %d, y_end_g = %d \n", 
         trid_handle.size[1], trid_handle.start_g[1], trid_handle.end_g[1]);
  printf("Check parameters: nz = %d, z_start_g = %d, z_end_g = %d \n",
         trid_handle.size[2], trid_handle.start_g[2], trid_handle.end_g[2]);
  
  // Initialize
  for(int k = 0; k < trid_handle.size[2]; k++) {
    for(int j = 0; j < trid_handle.size[1]; j++) {
      for(int i = 0; i < trid_handle.size[0]; i++) {
        int ind = k * trid_handle.pads[0] * trid_handle.pads[1] + j*trid_handle.pads[0] + i;
        if( (trid_handle.start_g[0]==0 && i==0) || 
            (trid_handle.end_g[0]==trid_handle.size_g[0]-1 && i==trid_handle.size[0]-1) ||
            (trid_handle.start_g[1]==0 && j==0) || 
            (trid_handle.end_g[1]==trid_handle.size_g[1]-1 && j==trid_handle.size[1]-1) ||
            (trid_handle.start_g[2]==0 && k==0) || 
            (trid_handle.end_g[2]==trid_handle.size_g[2]-1 && k==trid_handle.size[2]-1)) {
          trid_handle.h_u[ind] = 1.0f;
        } else {
          trid_handle.h_u[ind] = 0.0f;
        }
      }
    }
  }
  
  pre_handle.halo_snd_x = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_x = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_snd_y = (FP*) _mm_malloc(2 * trid_handle.size[0] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_y = (FP*) _mm_malloc(2 * trid_handle.size[0] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_snd_z = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[0] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_z = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[0] * sizeof(FP), SIMD_WIDTH);

  return 0;

}

void finalize(trid_handle<FP> &trid_handle, trid_mpi_handle &mpi_handle, preproc_handle<FP> &pre_handle) {
  tridClean<FP>(trid_handle, mpi_handle);
  _mm_free(pre_handle.halo_snd_x);
  _mm_free(pre_handle.halo_rcv_x);
  _mm_free(pre_handle.halo_snd_y);
  _mm_free(pre_handle.halo_rcv_y);
  _mm_free(pre_handle.halo_snd_z);
  _mm_free(pre_handle.halo_rcv_z);
}

int main(int argc, char* argv[]) {
  trid_mpi_handle mpi_handle;
  trid_handle<FP> trid_handle;
  preproc_handle<FP> pre_handle;
  trid_timer trid_timing;
  int iter;
  const int INC = 1;
  init(trid_handle, mpi_handle, pre_handle, iter, argc, argv);

  // Declare and reset elapsed time counters
  double timer           = 0.0;
  double timer1          = 0.0;
  double timer2          = 0.0;
  double elapsed         = 0.0;
  double elapsed_total   = 0.0;
  double elapsed_preproc = 0.0;
  double elapsed_trid_x  = 0.0;
  double elapsed_trid_y  = 0.0;
  double elapsed_trid_z  = 0.0;

  char elapsed_name[11][256] = {"forward","halo1","gather","halo2","reduced","halo3","scatter","halo4","backward","pre_mpi","pre_comp"};
  
  double timers_avg[11];

  for(int i = 0; i < 11; i++) {
    trid_timing.elapsed_time_x[i] = 0.0;
    trid_timing.elapsed_time_y[i] = 0.0;
    trid_timing.elapsed_time_z[i] = 0.0;
    timers_avg[i] = 0;
  }
  
#define TIMED

  timing_start(&timer1);
  
  for(int it = 0; it < iter; it++) {
    
    timing_start(&timer);
    
    preproc_mpi<FP>(pre_handle, trid_handle.h_u, trid_handle.du, trid_handle.a,
                    trid_handle.b, trid_handle.c, trid_handle, mpi_handle);
    
    timing_end(&timer, &elapsed_preproc);

    //
    // perform tri-diagonal solves in x-direction
    //
    timing_start(&timer);
    
#ifdef TIMED
    tridBatchTimed<FP, INC>(trid_handle, mpi_handle, trid_timing, 0);
#else
    tridBatch<FP, INC>(trid_handle, mpi_handle, 0);
#endif
    
    timing_end(&timer, &elapsed_trid_x);

    //
    // perform tri-diagonal solves in y-direction
    //
    timing_start(&timer);

#ifdef TIMED
    tridBatchTimed<FP, INC>(trid_handle, mpi_handle, trid_timing, 1);
#else
    tridBatch<FP, INC>(trid_handle, mpi_handle, 1);
#endif
    
    timing_end(&timer, &elapsed_trid_y);
    
    //
    // perform tri-diagonal solves in z-direction
    //
    timing_start(&timer);
    
#ifdef TIMED
    tridBatchTimed<FP, INC>(trid_handle, mpi_handle, trid_timing, 2);
#else
    tridBatch<FP, INC>(trid_handle, mpi_handle, 2);
#endif
    
    timing_end(&timer, &elapsed_trid_z);
  }
  
  timing_end(&timer1, &elapsed_total);
  
  rms("end h_u", trid_handle.h_u, trid_handle, mpi_handle);
  rms("end du", trid_handle.du, trid_handle, mpi_handle);

  MPI_Barrier(MPI_COMM_WORLD);
  
  double avg_total = 0.0;

  MPI_Reduce(trid_timing.elapsed_time_x, timers_avg, 11, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&elapsed_trid_x, &avg_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(mpi_handle.rank == 0) {
    for(int i=0; i<11; i++)
        timers_avg[i] /= mpi_handle.procs;
    
    avg_total /= mpi_handle.procs;
  }
  
  /*for(int i=0; i<mpi_handle.procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    //sleep(0.2);
    if(i==mpi_handle.rank) {
      if(mpi_handle.rank==0) {
        printf("Time in trid-x segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[checksum]\n",
            elapsed_name[0], elapsed_name[1], elapsed_name[2], elapsed_name[3], elapsed_name[4], elapsed_name[5], elapsed_name[6], elapsed_name[7], elapsed_name[8]);
      }
      printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
      1000.0*elapsed_trid_x ,
      1000.0*trid_timing.elapsed_time_x[0],
      1000.0*trid_timing.elapsed_time_x[1],
      1000.0*trid_timing.elapsed_time_x[2],
      1000.0*trid_timing.elapsed_time_x[3],
      1000.0*trid_timing.elapsed_time_x[4],
      1000.0*trid_timing.elapsed_time_x[5],
      1000.0*trid_timing.elapsed_time_x[6],
      1000.0*trid_timing.elapsed_time_x[7],
      1000.0*trid_timing.elapsed_time_x[8],
      1000.0*(trid_timing.elapsed_time_x[0] + trid_timing.elapsed_time_x[1] + trid_timing.elapsed_time_x[2] + trid_timing.elapsed_time_x[3] 
              + trid_timing.elapsed_time_x[4] + trid_timing.elapsed_time_x[5] + trid_timing.elapsed_time_x[6] + trid_timing.elapsed_time_x[7]
              + trid_timing.elapsed_time_x[8]));
    }
  }*/
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi_handle.rank == 0) {
    printf("Average time in trid-x segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s]\n",
            elapsed_name[0], elapsed_name[1], elapsed_name[2], elapsed_name[3], elapsed_name[4], elapsed_name[5], elapsed_name[6], elapsed_name[7], elapsed_name[8]);
    printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
        1000.0*avg_total ,
        1000.0*timers_avg[0],
        1000.0*timers_avg[1],
        1000.0*timers_avg[2],
        1000.0*timers_avg[3],
        1000.0*timers_avg[4],
        1000.0*timers_avg[5],
        1000.0*timers_avg[6],
        1000.0*timers_avg[7],
        1000.0*timers_avg[8]);
  }
  
  for(int i = 0; i < 11; i++) {
    timers_avg[i] = 0;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(trid_timing.elapsed_time_y, timers_avg, 11, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&elapsed_trid_y, &avg_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(mpi_handle.rank == 0) {
    for(int i=0; i<11; i++)
        timers_avg[i] /= mpi_handle.procs;
    
    avg_total /= mpi_handle.procs;
  }
  
  /*for(int i=0; i<mpi_handle.procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    //sleep(0.2);
    if(i==mpi_handle.rank) {
      if(mpi_handle.rank==0) {
        printf("Time in trid-y segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[checksum]\n",
            elapsed_name[0], elapsed_name[1], elapsed_name[2], elapsed_name[3], elapsed_name[4], elapsed_name[5], elapsed_name[6], elapsed_name[7], elapsed_name[8]);
      }
      printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
      1000.0*elapsed_trid_y ,
      1000.0*trid_timing.elapsed_time_y[0],
      1000.0*trid_timing.elapsed_time_y[1],
      1000.0*trid_timing.elapsed_time_y[2],
      1000.0*trid_timing.elapsed_time_y[3],
      1000.0*trid_timing.elapsed_time_y[4],
      1000.0*trid_timing.elapsed_time_y[5],
      1000.0*trid_timing.elapsed_time_y[6],
      1000.0*trid_timing.elapsed_time_y[7],
      1000.0*trid_timing.elapsed_time_y[8],
      1000.0*(trid_timing.elapsed_time_y[0] + trid_timing.elapsed_time_y[1] + trid_timing.elapsed_time_y[2] + trid_timing.elapsed_time_y[3] 
              + trid_timing.elapsed_time_y[4] + trid_timing.elapsed_time_y[5] + trid_timing.elapsed_time_y[6] + trid_timing.elapsed_time_y[7] 
              + trid_timing.elapsed_time_y[8]));
    }
  }*/
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi_handle.rank == 0) {
    printf("Average time in trid-y segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s]\n",
            elapsed_name[0], elapsed_name[1], elapsed_name[2], elapsed_name[3], elapsed_name[4], elapsed_name[5], elapsed_name[6], elapsed_name[7], elapsed_name[8]);
    printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
        1000.0*avg_total ,
        1000.0*timers_avg[0],
        1000.0*timers_avg[1],
        1000.0*timers_avg[2],
        1000.0*timers_avg[3],
        1000.0*timers_avg[4],
        1000.0*timers_avg[5],
        1000.0*timers_avg[6],
        1000.0*timers_avg[7],
        1000.0*timers_avg[8]);
  }
  
  for(int i = 0; i < 11; i++) {
    timers_avg[i] = 0;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(trid_timing.elapsed_time_z, timers_avg, 11, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&elapsed_trid_z, &avg_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(mpi_handle.rank == 0) {
    for(int i=0; i<11; i++)
        timers_avg[i] /= mpi_handle.procs;
    
    avg_total /= mpi_handle.procs;
  }
  
  /*for(int i=0; i<mpi_handle.procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    //sleep(0.2);
    if(i==mpi_handle.rank) {
      if(mpi_handle.rank==0) {
        printf("Time in trid-z segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[checksum]\n",
            elapsed_name[0], elapsed_name[1], elapsed_name[2], elapsed_name[3], elapsed_name[4], elapsed_name[5], elapsed_name[6], elapsed_name[7], elapsed_name[8]);
      }
      printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
      1000.0*elapsed_trid_z ,
      1000.0*trid_timing.elapsed_time_z[0],
      1000.0*trid_timing.elapsed_time_z[1],
      1000.0*trid_timing.elapsed_time_z[2],
      1000.0*trid_timing.elapsed_time_z[3],
      1000.0*trid_timing.elapsed_time_z[4],
      1000.0*trid_timing.elapsed_time_z[5],
      1000.0*trid_timing.elapsed_time_z[6],
      1000.0*trid_timing.elapsed_time_z[7],
      1000.0*trid_timing.elapsed_time_z[8],
      1000.0*(trid_timing.elapsed_time_z[0] + trid_timing.elapsed_time_z[1] + trid_timing.elapsed_time_z[2] + trid_timing.elapsed_time_z[3] 
              + trid_timing.elapsed_time_z[4] + trid_timing.elapsed_time_z[5] + trid_timing.elapsed_time_z[6] + trid_timing.elapsed_time_z[7] 
              + trid_timing.elapsed_time_z[8]));
    }
  }*/
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi_handle.rank == 0) {
    printf("Average time in trid-z segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s]\n",
            elapsed_name[0], elapsed_name[1], elapsed_name[2], elapsed_name[3], elapsed_name[4], elapsed_name[5], elapsed_name[6], elapsed_name[7], elapsed_name[8]);
    printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
        1000.0*avg_total ,
        1000.0*timers_avg[0],
        1000.0*timers_avg[1],
        1000.0*timers_avg[2],
        1000.0*timers_avg[3],
        1000.0*timers_avg[4],
        1000.0*timers_avg[5],
        1000.0*timers_avg[6],
        1000.0*timers_avg[7],
        1000.0*timers_avg[8]);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi_handle.rank == 0) {
    // Print execution times
    printf("Time per section: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n");
    printf("%e \t%e \t%e \t%e \t%e\n",
        elapsed_total,
        elapsed_preproc,
        elapsed_trid_x,
        elapsed_trid_y,
        elapsed_trid_z);
    printf("Time per element averaged on %d iterations: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n", iter);
    printf("%e \t%e \t%e \t%e \t%e\n",
        (elapsed_total/iter  ) / (trid_handle.size_g[0] * trid_handle.size_g[1] * trid_handle.size_g[2]),
        (elapsed_preproc/iter) / (trid_handle.size_g[0] * trid_handle.size_g[1] * trid_handle.size_g[2]),
        (elapsed_trid_x/iter ) / (trid_handle.size_g[0] * trid_handle.size_g[1] * trid_handle.size_g[2]),
        (elapsed_trid_y/iter ) / (trid_handle.size_g[0] * trid_handle.size_g[1] * trid_handle.size_g[2]),
        (elapsed_trid_z/iter ) / (trid_handle.size_g[0] * trid_handle.size_g[1] * trid_handle.size_g[2]));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  finalize(trid_handle, mpi_handle, pre_handle);
  
  MPI_Finalize();
  return 0;

}
