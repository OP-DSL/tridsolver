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
#include <sys/time.h>
#include <getopt.h>
#include <openacc.h>

#if FPPREC == 0
#  define FP float
#elif FPPREC == 1
#  define FP double
#else
#  error "Macro definition FPPREC unrecognized for CUDA"
#endif

#include "acc/trid_x_acc.hpp"
#include "acc/trid_y_acc.hpp"
#include "acc/trid_z_acc.hpp"

//
// linux timing routine
//

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
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER -opt CUDAOPT -prof PROF\n");
  exit(0);
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

//
// tridiagonal solver
//
//void trid_cpu(float* a, float* b, float* c, float* d, int N){
//  a[0] = 1.0f;
//inline void trid_acc(float* const __restrict__ a, float* const __restrict__ b, float* const  __restrict__ c, float* const __restrict__ d, int N, int stride) {
////inline void trid_acc(float a, float b, float c, float d, int N, int stride, int base) {
//  int   i, ind = 0;
//  float aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
//  //
//  // forward pass
//  //
//  bb    = 1.0f/b[ind];
//  cc    = bb*c[ind];
//  dd    = bb*d[ind];
//  c2[0] = cc;
//  d2[0] = dd;
//  //#pragma acc loop seq
//  for(i=1; i<N; i++) {
//    //ind   = ind + stride;
//    aa    = a[ind+1];
//    bb    = b[ind+1] - aa*cc;
//    dd    = d[ind+1] - aa*dd;
//    bb    = 1.0f/bb;
//    cc    = bb*c[ind+1];
//    dd    = bb*dd;
//    c2[i] = cc;
//    d2[i] = dd;
//  }
//  //
//  // reverse pass
//  //
//  d[ind] = dd;
////  #pragma acc loop seq
//  //for(i=N-2; i>=0; i--) {
//  for(i=1; i<=N-1; i++) {
//    //ind    = ind - stride;
//    dd     = d2[i] - c2[i]*dd;
//    d[ind-i] = dd;
//  }
//}
//
//inline void foo(float* b, float *a) {
//  b[0] = a[0];
//  }




inline void adi_acc(float lambda, float* restrict u, float* restrict du, float* restrict ax, float* restrict bx, float* restrict cx, float* restrict ay, float* restrict by, float* restrict cy, float* restrict az, float* restrict bz, float* restrict cz, int nx, int ny, int nz, double *elapsed_preproc, double *elapsed_trid_x, double *elapsed_trid_y, double *elapsed_trid_z, int prof) {
  int    i, j, k;//, ind, ind2, base;
  double timer, elapsed;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //
  timing_start(prof,&timer);
  #pragma acc data present(u[0:N],du[0:N],ax[0:N],bx[0:N],cx[0:N],ay[0:N],by[0:N],cy[0:N],az[0:N],bz[0:N],cz[0:N])
  #pragma acc parallel loop collapse(3) //async
  //#pragma acc kernels loop independent private(ind) async
  //#pragma acc kernels loop independent async
  for(int k=0; k<NZ; k++) {
    //#pragma acc loop independent
    //#pragma acc loop independent private(ind)
     for(int j=0; j<NY; j++) {
      //#pragma acc loop independent
      //#pragma acc loop independent private(ind)
      for(int i=0; i<NX; i++) {   // i loop innermost for sequential memory access
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
  timing_end(prof,&timer,elapsed_preproc,"preproc");

  //
  // perform tri-diagonal solves in x-direction
  //
//#pragma acc data present(u[0:N],du[0:N],ax[0:N],bx[0:N],cx[0:N]) 
//{
  //#pragma acc parallel loop private(ind,c2,d2) pcreate(c2,d2)
  //#pragma acc kernels loop collapse(2) independent private(base,ind,c2,d2) // pcreate(c2,d2) //private(ind,c2,d2) //pcreate(c2,d2)
  //#pragma acc kernels loop collapse(2) independent private(aa,bb,cc,dd,base,ind,c2,d2) async // pcreate(c2,d2) //private(ind,c2,d2) //pcreate(c2,d2)
  //#pragma acc kernels loop independent vector(16) async  
  //#pragma acc kernels loop independent vector(16) async  

    timing_start(prof,&timer);
      trid_x_acc(ax, bx, cx, du, u, NX, NY, NZ);
    timing_end(prof,&timer,elapsed_trid_x,"trid_x");

  //
  // perform tri-diagonal solves in y-direction
  //
//#pragma acc data present(u[0:N],du[0:N],ay[0:N],by[0:N],cy[0:N]) 
//{
//#pragma omp parallel for shared(ay,by,cy,du)
  //#pragma acc kernels loop independent private(base,ind) 
  //#pragma acc kernels loop independent vector(32) private(aa,bb,cc,dd,base,ind,c2,d2) async
  //#pragma acc kernels loop independent async
    timing_start(prof,&timer);
      trid_y_acc(ay, by, cy, du, u, NX, NY, NZ);
    timing_end(prof,&timer,elapsed_trid_y,"trid_y");
  //
  // perform tri-diagonal solves in z-direction
  //
  //#pragma acc cache(c2,d2)
  //#pragma acc kernels loop collapse(2) independent async
    timing_start(prof,&timer);
      trid_z_acc(az, bz, cz, du, u, NX, NY, NZ);
    timing_end(prof,&timer,elapsed_trid_z,"trid_z");

}

//inline void timing_start(int prof, double *timer) {
//  if(prof==1) elapsed_time(timer); 
//}
//
//inline void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
//  double elapsed;
//  if(prof==1) {
//    cutilSafeCall( cudaDeviceSynchronize() ); 
//    elapsed = elapsed_time(timer); 
//    *elapsed_accumulate += elapsed; 
//    printf("\n elapsed %s (sec): %1.10f (s) \n", str,elapsed); 
//  }
//}


int main(int argc, char* argv[]) { 
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_trid_x, elapsed_trid_y, elapsed_trid_z;

  // 'h_' prefix - CPU (host) memory space

  int   i, j, k, ind, it;
  int   nx, ny, nz, iter, opt, prof;
  float *restrict h_u, *restrict h_du,
        *restrict h_ax, *restrict h_bx, *restrict h_cx, 
        *restrict h_ay, *restrict h_by, *restrict h_cy, 
        *restrict h_az, *restrict h_bz, *restrict h_cz, 
        *restrict tmp, 
        err, lambda=1.0f; // lam = dt/dx^2

  nx=NX;
  ny=NY;
  nz=NZ;
  iter=ITER;
  prof = 1;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp(options[opt_index].name,"nx"  ) == 0) nx   = atoi(optarg); //printf("nx   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"ny"  ) == 0) ny   = atoi(optarg); //printf("ny   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"nz"  ) == 0) nz   = atoi(optarg); //printf("nz   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"iter") == 0) iter = atoi(optarg); //printf("iter ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg); //printf("opt  ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"prof") == 0) prof = atoi(optarg); //printf("prof ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"help") == 0) print_help();
  }

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  if( NX>N_MAX || NY>N_MAX || NZ>N_MAX ) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    return -1;
  }
  // allocate memory for arrays

  h_u  = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_du = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_ax = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_bx = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_cx = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_ay = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_by = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_cy = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_az = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_bz = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_cz = (float *)malloc(sizeof(float)*NX*NY*NZ);

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



  acc_init(acc_device_nvidia);
  acc_set_device_num(1,acc_device_nvidia);

  // reset elapsed time counters
  elapsed_total   = 0.0;
  elapsed_preproc = 0.0;  
  elapsed_trid_x  = 0.0;
  elapsed_trid_y  = 0.0;
  elapsed_trid_z  = 0.0;


  // Compute sequentially
#pragma acc data pcopy(h_u[0:N]) create(h_du[0:N],h_ax[0:N],h_bx[0:N],h_cx[0:N],h_ay[0:N],h_by[0:N],h_cy[0:N],h_az[0:N],h_bz[0:N],h_cz[0:N]) 
{
  elapsed_time(&timer2);
  for(it =0; it<iter; it++) {
    adi_acc(lambda, h_u, h_du, h_ax, h_bx, h_cx, h_ay, h_by, h_cy, h_az, h_bz, h_cz, nx, ny, nz, &elapsed_preproc, &elapsed_trid_x, &elapsed_trid_y, &elapsed_trid_z, prof); 
  }  
  elapsed_total = elapsed_time(&timer2);
}
  printf("\nComputing ADI on GPU with OpenACC: %f (s) \n", elapsed_total);

  int ldim = nx;
  #include "print_array.c"

  free(h_u);
  free(h_du);
  free(h_ax);
  free(h_bx);
  free(h_cx);
  free(h_ay);
  free(h_by);
  free(h_cy);
  free(h_az);
  free(h_bz);
  free(h_cz);
  
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

  exit(0);
}
