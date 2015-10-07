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
//#include <cstring>
#include <sys/time.h>
#include <getopt.h>
#include <openacc.h>

#include "acc/trid_x_acc.hpp"
#include "acc/trid_y_acc.hpp"
#include "acc/trid_z_acc.hpp"

//#include "cuda/trid_cuda.hpp"
#include "trid_common.h"
//#include <cusparse_v2.h>

//void initTridMultiDimBatchSolve_wrapper(int ndim, int *dims, int *pads, int *cumdims, int *cumpads);
//void tridMultiDimBatchSolve_wrapper(float* d_a, float* d_b, float* d_c, float* d_d, float* d_u, int ndim, int solvedim, int *dims, int *pads, int *cumdims, int *cumpads, int *opts, float** d_buffer);

#if FPPREC == 0
#  define FP float
#elif FPPREC == 1
#  define FP double
#else
#  error "Macro definition FPPREC unrecognized for CUDA"
#endif

//
// linux timing routine
//

extern char *optarg;
extern int  optind, opterr, optopt; 
static struct option options[] = {
  {"nx",        required_argument, 0,  0   },
  {"ny",        required_argument, 0,  0   },
  {"nz",        required_argument, 0,  0   },
  {"iter",      required_argument, 0,  0   },
  {"optlinear", required_argument, 0,  0   },
  {"optstride", required_argument, 0,  0   },
  {"prof",      required_argument, 0,  0   },
  {"help",      no_argument,       0,  'h' },
  {0,           0,                 0,  0   }
};

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("\nPlease specify the ADI configuration "
    "e.g. ./adi_* -nx NX -ny NY -nz NZ -iter ITER -opt CUDAOPT -prof PROF\n");
}

#define N ((NX)*(NY)*(NZ))

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
    //cutilSafeCall( cudaDeviceSynchronize() ); 
    elapsed = elapsed_time(timer); 
    *elapsed_accumulate += elapsed; 
    printf("\n elapsed %s (sec): %1.10f (s) \n", str,elapsed); 
  }
}

inline void adi_acc(float lambda, float *u, float *du, float *ax, float *bx, float *cx, float *ay, float *by, float *cy, float *az, float *bz, float *cz, float *buffer, int nx, int ny, int nz, double *elapsed_preproc, double *elapsed_trid_x, double *elapsed_trid_y, double *elapsed_trid_z, int prof, int *opts, int ndim, int *dims, int *pads) {
  int    i, j, k;//, ind, ind2, base;
  double timer, elapsed;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //

//  acc_set_device_num(0,acc_device_nvidia);
  int n = nx*ny*nz;

  timing_start(prof,&timer);
#pragma acc data deviceptr(u,du,ax,bx,cx,ay,by,cy,az,bz,cz,buffer)
//#pragma acc data present(u[0:nx*ny*nz]) pcreate(du[0:nx*ny*nz],ax[0:nx*ny*nz],bx[0:nx*ny*nz],cx[0:nx*ny*nz],ay[0:nx*ny*nz],by[0:nx*ny*nz],cy[0:nx*ny*nz],az[0:nx*ny*nz],bz[0:nx*ny*nz],cz[0:nx*ny*nz],buffer[0:nx*ny*nz])
{
  #pragma acc kernels loop collapse(3) independent //async
  for(k=0; k<NZ; k++) {
    for(j=0; j<NY; j++) {
      for(i=0; i<NX; i++) {   // i loop innermost for sequential memory access
        int ind;
        float a,b,c,d;

        ind = k*NX*NY + j*NX + i;
        if(i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
          d = 0.0f; // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          d = lambda*(  u[ind-1    ] + u[ind+1]
                      + u[ind-NX   ] + u[ind+NX]
                      + u[ind-NX*NY] + u[ind+NX*NY]
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
}
  timing_end(prof,&timer,elapsed_preproc,"preproc");

  //
  // perform tri-diagonal solves in x-direction
  //

//  int sys_stride; // Stride between the consecutive elements of a system
//  int sys_size;   // Size (length) of a system
//  int sys_pads;   // Padded sizes along each ndim number of dimensions
//  int sys_n;      // Number of systems to be solved.
  int solvedim;   // user chosen dimension for which the solution is performed
  int sync = 1; // Host-synchronous execution

  timing_start(prof,&timer);
    //trid_x_acc(ax, bx, cx, du, u, NX, NY, NZ);
    solvedim   = 0;
    //tridBatchSolve(&d_ax, &d_bx, &d_cx, &d_du, &d_u, sys_stride, sys_size, sys_pads, sys_n, nx, ny, nz, optx, opty, optz, &handle_sp, &d_buffer);
    //#pragma acc host_data use_device(ax, bx, cx, du, u, buffer)
    #pragma acc deviceptr(ax, bx, cx, du, u, buffer)
    tridMultiDimBatchSolve_wrapper_SNA(ax, bx, cx, du, u, ndim, solvedim, dims, pads, opts, &buffer, sync);
  timing_end(prof,&timer,elapsed_trid_x,"trid_x");

  //
  // perform tri-diagonal solves in y-direction
  //
  timing_start(prof,&timer);
    //trid_y_acc(ay, by, cy, du, u, NX, NY, NZ);
    solvedim = 1;
    //#pragma acc host_data use_device(ay, by, cy, du, u, buffer)
    #pragma acc deviceptr(ay, by, cy, du, u, buffer)
    tridMultiDimBatchSolve_wrapper_SNA(ay, by, cy, du, u, ndim, solvedim, dims, pads, opts, &buffer, sync);
  timing_end(prof,&timer,elapsed_trid_y,"trid_y");
  //
  // perform tri-diagonal solves in z-direction
  //
  timing_start(prof,&timer);
    //trid_z_acc(az, bz, cz, du, u, NX, NY, NZ);
    solvedim = 2;
    //#pragma acc host_data use_device(az, bz, cz, du, u, buffer)
    #pragma acc deviceptr(az, bz, cz, du, u, buffer)
    tridMultiDimBatchSolve_wrapper_SNA(az, bz, cz, du, u, ndim, solvedim, dims, pads, opts, &buffer, sync);
  timing_end(prof,&timer,elapsed_trid_z,"trid_z");
// }
}

int main(int argc, char* argv[]) { 
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_trid_x, elapsed_trid_y, elapsed_trid_z;

  // 'h_' prefix - CPU (host) memory space

  int   i, j, k, ind, it;
  int   nx, ny, nz, iter, optlinear, optstride, prof;
  float * h_u, * h_du,
        * h_ax, * h_bx, * h_cx,
        * h_ay, * h_by, * h_cy,
        * h_az, * h_bz, * h_cz,
        * h_buffer,
        * tmp,
        err, lambda=1.0f; // lam = dt/dx^2

//  float * d_u, * d_du,
//        * d_ax, * d_bx, * d_cx,
//        * d_ay, * d_by, * d_cy,
//        * d_az, * d_bz, * d_cz,
//        * d_buffer;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp(options[opt_index].name,"nx"        ) == 0) nx         = atoi(optarg); //printf("nx   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"ny"        ) == 0) ny         = atoi(optarg); //printf("ny   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"nz"        ) == 0) nz         = atoi(optarg); //printf("nz   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"iter"      ) == 0) iter       = atoi(optarg); //printf("iter ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"optlinear" ) == 0) optlinear  = atoi(optarg); //printf("opt  ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"optstride" ) == 0) optstride  = atoi(optarg); //printf("opt  ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"prof"      ) == 0) prof       = atoi(optarg); //printf("prof ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"help"      ) == 0) print_help();
  }

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  if( NX>N_MAX || NY>N_MAX || NZ>N_MAX ) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    return -1;
  }
  // allocate memory for arrays

  h_u  = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_du = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_ax = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_bx = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_cx = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_ay = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_by = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_cy = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_az = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_bz = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_cz = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);
  h_buffer = (float *)acc_malloc(sizeof(float)*NX*NY*NZ);

  // Initialize
  for(k=0; k<NZ; k++) {
    for(j=0; j<NY; j++) {
      for(i=0; i<NX; i++) {
        ind = k*NX*NY + j*NX + i;
        if(i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
          h_u[ind] = 1.0f;
        } else {
          h_u[ind] = 0.0f;
        }
      }
    }
  }

  nx=NX;
  ny=NY;
  nz=NZ;
//  iter=ITER;
//  nx=256;
//  ny=256;
//  nz=256;
//  iter=1;

  acc_init(acc_device_nvidia);
  acc_set_device_num(0,acc_device_nvidia);

  // reset elapsed time counters
  elapsed_total   = 0.0;
  elapsed_preproc = 0.0;  
  elapsed_trid_x  = 0.0;
  elapsed_trid_y  = 0.0;
  elapsed_trid_z  = 0.0;

  // Tridiagonal solver option arguemnt's setup
  int ndim = 3;          // Number of dimensions of the (hyper)cubic data structure.
  int dims[MAXDIM];      // Array containing the sizes of each ndim dimensions. size(dims) == ndim <=MAXDIM
  int pads[MAXDIM];      // Padded sizes along each ndim number of dimensions
  dims[0] = nx;
  dims[1] = ny;
  dims[2] = nz;
  pads[0] = dims[0];
  pads[1] = dims[1];
  pads[2] = dims[2];

  initTridMultiDimBatchSolve_wrapper(ndim, dims, pads);

  int opts[MAXDIM];// = {nx, ny, nz,...};
  opts[0] = optlinear;
  opts[1] = optstride;
  opts[2] = optstride;

  int n = nx*ny*nz;

  // Compute sequentially
//#pragma acc data pcopy(h_u[n]) present(h_du[n],h_ax[n],h_bx[n],h_cx[n],h_ay[n],h_by[n],h_cy[n],h_az[n],h_bz[n],h_cz[n],h_buffer[n])
#pragma acc data pcopy(h_u[n]) deviceptr(h_du,h_ax,h_bx,h_cx,h_ay,h_by,h_cy,h_az,h_bz,h_cz,h_buffer)
//#pragma acc data deviceptr(h_u[n],h_du[n],h_ax[n],h_bx[n],h_cx[n],h_ay[n],h_by[n],h_cy[n],h_az[n],h_bz[n],h_cz[n],h_buffer[n])
{
#pragma acc host_data use_device(h_u)
{
  elapsed_time(&timer2);
  for(it=0; it<iter; it++) {
    adi_acc(lambda, h_u, h_du, h_ax, h_bx, h_cx, h_ay, h_by, h_cy, h_az, h_bz, h_cz, h_buffer, nx, ny, nz, &elapsed_preproc, &elapsed_trid_x, &elapsed_trid_y, &elapsed_trid_z, prof, opts, ndim, dims, pads);
  }
  elapsed_total = elapsed_time(&timer2);
}
}
  printf("\nComputing ADI on GPU with OpenACC: %f (s) \n", elapsed_total);

  #include "print_array.c"

  free(h_u);
  acc_free(h_du);
  acc_free(h_ax);
  acc_free(h_bx);
  acc_free(h_cx);
  acc_free(h_ay);
  acc_free(h_by);
  acc_free(h_cy);
  acc_free(h_az);
  acc_free(h_bz);
  acc_free(h_cz);
  acc_free(h_buffer);
  
  printf("Done.\n");

  // Print execution times
  if(prof == 0) { 
    printf("Avg(per iter) \n[total]\n");
    printf("%f\n", elapsed_total/iter);
  }
  else if(prof == 1) {
    printf("Avg(per iter) \n[total]  [prepro] [trid_x] [trid_y] [trid_z]\n");
    printf("%f %f %f %f %f\n", elapsed_total/iter, elapsed_preproc/iter, elapsed_trid_x/iter, elapsed_trid_y/iter, elapsed_trid_z/iter);
  }

  acc_shutdown(acc_device_nvidia);
  exit(0);
}
