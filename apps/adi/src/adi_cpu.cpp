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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>

#include "preproc.hpp"
#include "trid_cpu.h"

#include "omp.h"
#include "offload.h"

#ifdef __MKL__
  #include "mkl.h"
#endif

#ifdef __MIC__ // Or #ifdef __KNC__ - more general option, future proof, __INTEL_OFFLOAD is another option
  __attribute__((target(mic)))
  inline double elapsed_time(double *et);

  __attribute__((target(mic)))
  inline void timing_start(int prof, double *timer);

  __attribute__((target(mic)))
  inline void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str);
#endif

#define ROUND_DOWN(N,step) (((N)/(step))*step)

//
// linux timing routine
//
#include <sys/time.h>

inline double elapsed_time(double *et) {
  struct timeval t;
  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
}

inline void timing_start(int prof, double *timer) {
  if(prof==1) elapsed_time(timer);
}

inline void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
  double elapsed;
  if(prof==1) {
    elapsed = elapsed_time(timer);
    *elapsed_accumulate += elapsed;
    printf("\n elapsed %s (sec): %1.10f (s) \n", str,elapsed);
  }
}

void rms(char* name, FP* array, int nx_pad, int nx, int ny, int nz) {
  //Sum the square of values in app.h_u
  double sum = 0.0;
  for(int k=0; k<nz; k++) {
    for(int j=0; j<ny; j++) {
      for(int i=0; i<nx; i++) {
        int ind = k*nx_pad*ny + j*nx_pad + i;
        //sum += array[ind]*array[ind];
        sum += array[ind];
      }
    }
  }

  printf("%s sum = %lg\n", name, sum);
}

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
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER -opt OPTIMIZATION -prof PROF\n");
  exit(0);
}

int main(int argc, char* argv[]) {
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_trid_x, elapsed_trid_y, elapsed_trid_z;

  int i, j, k, ind, it;
  int nx, nx_pad, ny, nz, iter, opt, prof;

  // 'h_' prefix - CPU (host) memory space
  FP  *__restrict__ h_u, *__restrict__ h_tmp, *__restrict__ h_du,
      *__restrict__ h_ax, *__restrict__ h_bx, *__restrict__ h_cx,
      *__restrict__ h_ay, *__restrict__ h_by, *__restrict__ h_cy,
      *__restrict__ h_az, *__restrict__ h_bz, *__restrict__ h_cz,
      *__restrict__ tmp,
      err, lambda=1.0f; // lam = dt/dx^2

  // Set defaults options
  nx   = 256;
  ny   = 256;
  nz   = 256;
  iter = 10;
  opt  = 0;
  prof = 1;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx   = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny   = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz   = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"prof") == 0) prof = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
  }

  if( nx>N_MAX || ny>N_MAX || nz>N_MAX ) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    exit(1);
  }

  // allocate memory for arrays
  nx_pad = (1+((nx-1)/SIMD_VEC))*SIMD_VEC; // Compute padding for vecotrization
  h_u  = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_tmp= (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_du = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_ax = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_bx = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_cx = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_ay = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_by = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_cy = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_az = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_bz = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);
  h_cz = (FP *)_mm_malloc(sizeof(FP)*nx_pad*ny*nz,SIMD_WIDTH);

  printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);
  printf("Check parameters: SIMD_WIDTH = %d, sizeof(FP) = %d, nx_pad = %d \n",SIMD_WIDTH,sizeof(FP),nx_pad);

  // Initialize
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {
        ind = k*nx_pad*ny + j*nx_pad + i;
        if(i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
          h_u[ind] = 1.0f;
        } else {
          h_u[ind] = 0.0f;
        }
      }
    }
  }


  // reset elapsed time counters
  elapsed_total   = 0.0;
  elapsed_preproc = 0.0;
  elapsed_trid_x  = 0.0;
  elapsed_trid_y  = 0.0;
  elapsed_trid_z  = 0.0;


  elapsed_time(&timer2);
  for(it = 0; it<iter; it++) {

    //
    // calculate r.h.s. and set tri-diagonal coefficients
    //
    timing_start(prof, &timer);
      preproc<FP>(lambda, h_u, h_du, h_ax, h_bx, h_cx, h_ay, h_by, h_cy, h_az, h_bz, h_cz, nx, nx_pad, ny, nz);
    timing_end(prof, &timer, &elapsed_preproc, "preproc");
    rms("h_u",h_u, nx_pad, nx, ny, nz);
    //
    // perform tri-diagonal solves in x-direction
    //
    timing_start(prof, &timer);

    // Tridiagonal solver option arguemnt's setup
    int ndim = 3;  // Number of dimensions of the (hyper)cubic data structure.
    int dims[3];   // Array containing the sizes of each ndim dimensions. size(dims) == ndim <=MAXDIM
    int pads[3];   // Padded sizes along each ndim number of dimensions
    dims[0] = nx;
    dims[1] = ny;
    dims[2] = nz;
    pads[0] = dims[0];
    pads[1] = dims[1];
    pads[2] = dims[2];

    int solvedim = 0;   // user chosen dimension for which the solution is performed
    #if FPPREC == 0
      tridSmtsvStridedBatch(h_ax, h_bx, h_cx, h_du, h_u, ndim, solvedim, dims, pads);
    #elif FPPREC == 1
      tridDmtsvStridedBatch(h_ax, h_bx, h_cx, h_du, h_u, ndim, solvedim, dims, pads);
    #endif

    timing_end(prof, &timer, &elapsed_trid_x, "trid_x");

    if(it == 0) {
      rms("ax", h_ax, nx_pad, nx, ny, nz);
      rms("bx", h_bx, nx_pad, nx, ny, nz);
      rms("cx", h_cx, nx_pad, nx, ny, nz);
      rms("du", h_du, nx_pad, nx, ny, nz);
      rms("h_u",h_u, nx_pad, nx, ny, nz);
      exit(-2);
    }

    //
    // perform tri-diagonal solves in y-direction
    //
    timing_start(prof, &timer);

    solvedim = 1;   // user chosen dimension for which the solution is performed
    #if FPPREC == 0
      tridSmtsvStridedBatch(h_ay, h_by, h_cy, h_du, h_u, ndim, solvedim, dims, pads);
    #elif FPPREC == 1
      tridDmtsvStridedBatch(h_ay, h_by, h_cy, h_du, h_u, ndim, solvedim, dims, pads);
    #endif

    /*#pragma omp parallel for collapse(2) private(k,i,ind) //schedule(guided)
    for(k=0; k<nz; k++) {
      for(i=0; i<ROUND_DOWN(nx,SIMD_VEC); i+=SIMD_VEC) {
        ind = k*nx_pad*ny + i;
        //trid_scalar_vec<FP,VECTOR,0>(&h_ay[ind], &h_by[ind], &h_cy[ind], &h_du[ind], &h_u[ind], ny, nx_pad/SIMD_VEC);
        #if FPPREC == 0
          trid_scalar_vecS(&h_ay[ind], &h_by[ind], &h_cy[ind], &h_du[ind], &h_u[ind], ny, nx_pad/SIMD_VEC);
        #elif FPPREC == 1
          trid_scalar_vecD(&h_ay[ind], &h_by[ind], &h_cy[ind], &h_du[ind], &h_u[ind], ny, nx_pad/SIMD_VEC);
        #endif
      }
    }
    if(ROUND_DOWN(nx,SIMD_VEC) < nx) { // If there is leftover, fork threads an compute it
      #pragma omp parallel for collapse(2) private(k,i,ind)
      for(k=0; k<nz; k++) {
        for(i=ROUND_DOWN(nx,SIMD_VEC); i<nx; i++) {
          ind = k*nx_pad*ny + i;
          #if FPPREC == 0
            trid_scalarS(&h_ay[ind], &h_by[ind], &h_cy[ind], &h_du[ind], &h_u[ind], ny, nx_pad);
          #elif FPPREC == 1
            trid_scalarD(&h_ay[ind], &h_by[ind], &h_cy[ind], &h_du[ind], &h_u[ind], ny, nx_pad);
          #endif
        }
      }
    }*/

    timing_end(prof, &timer, &elapsed_trid_y, "trid_y");

    //
    // perform tridiagonal solves in z-direction
    //
    timing_start(prof, &timer);

    solvedim = 2;   // user chosen dimension for which the solution is performed
    #if FPPREC == 0
      tridSmtsvStridedBatch(h_az, h_bz, h_cz, h_du, h_u, ndim, solvedim, dims, pads);
    #elif FPPREC == 1
      tridDmtsvStridedBatch(h_az, h_bz, h_cz, h_du, h_u, ndim, solvedim, dims, pads);
    #endif

    /*#pragma omp parallel for collapse(2) private(j,i,k,ind) schedule(static,1) // Interleaved scheduling for better data locality and thus lower TLB miss rate
    for(j=0; j<ny; j++) {
      for(i=0; i<ROUND_DOWN(nx,SIMD_VEC); i+=SIMD_VEC) {
        ind = j*nx_pad + i;
        //trid_scalar_vec<FP,VECTOR,1>(&h_az[ind], &h_bz[ind], &h_cz[ind], &h_du[ind], &h_u[ind], nz, (nx_pad/SIMD_VEC)*ny);
        #if FPPREC == 0
          trid_scalar_vecSInc(&h_az[ind], &h_bz[ind], &h_cz[ind], &h_du[ind], &h_u[ind], nz, (nx_pad/SIMD_VEC)*ny);
        #elif FPPREC == 1
          trid_scalar_vecDInc(&h_az[ind], &h_bz[ind], &h_cz[ind], &h_du[ind], &h_u[ind], nz, (nx_pad/SIMD_VEC)*ny);
        #endif
      }
    }
    if(ROUND_DOWN(nx,SIMD_VEC) < nx) { // If there is leftover, fork threads an compute it
      #pragma omp parallel for collapse(2) private(j,i,k,ind) schedule(static,1)
      for(j=0; j<ny; j++) {
        for(i=ROUND_DOWN(nx,SIMD_VEC); i<nx; i++) {
          ind = j*nx_pad + i;
          #if FPPREC == 0
            trid_scalarS(&h_az[ind], &h_bz[ind], &h_cz[ind], &h_du[ind], &h_u[ind], nz, nx_pad*ny);
          #elif FPPREC == 1
            trid_scalarD(&h_az[ind], &h_bz[ind], &h_cz[ind], &h_du[ind], &h_u[ind], nz, nx_pad*ny);
          #endif
          for(k=0; k<nz; k++) {
            h_u[ind + k*nx_pad*ny] += h_du[ind + k*nx_pad*ny];
          }
        }
      }
    }*/


    timing_end(prof, &timer, &elapsed_trid_z, "trid_z");
  }
  elapsed = elapsed_time(&timer2);
  elapsed_total = elapsed;
  printf("\nADI total execution time for %d iterations (sec): %f (s) \n", iter, elapsed);
  fflush(0);

  int ldim=nx_pad;
  #include "print_array.c"

  _mm_free(h_u);
  _mm_free(h_du);
  _mm_free(h_ax);
  _mm_free(h_bx);
  _mm_free(h_cx);
  _mm_free(h_ay);
  _mm_free(h_by);
  _mm_free(h_cy);
  _mm_free(h_az);
  _mm_free(h_bz);
  _mm_free(h_cz);

  printf("Done.\n");

  // Print execution times
  if(prof == 0) {
    printf("Avg(per iter) \n[total]\n");
    printf("%f\n", elapsed_total/iter);
  }
  else if(prof == 1) {
  printf("Time per element averaged on %d iterations: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n", iter);
  printf("%e \t%e \t%e \t%e \t%e\n",
      (elapsed_total/iter)/(nx*ny*nz),
      (elapsed_preproc/iter)/(nx*ny*nz),
      (elapsed_trid_x/iter)/(nx*ny*nz),
      (elapsed_trid_y/iter)/(nx*ny*nz),
      (elapsed_trid_z/iter)/(nx*ny*nz));
  }
  exit(0);

}
