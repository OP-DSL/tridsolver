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
#include <getopt.h>

//
// linux timing routine
//

#include <sys/time.h>

#if FPPREC == 0
#  define FP float
#elif FPPREC == 1
#  define FP double
#else
#  error "Macro definition FPPREC unrecognized for CUDA"
#endif

inline double elapsed_time(double *et) {
  struct timeval t;

  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
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
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER -prof PROF\n");
  exit(0);
}

//__attribute__((target(mic)))
void timing_start(int prof, double *timer) {
  if(prof==1) elapsed_time(timer); 
}

//__attribute__((target(mic)))
void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
  double elapsed;
  if(prof==1) {
    elapsed = elapsed_time(timer); 
    *elapsed_accumulate += elapsed; 
    printf("\n elapsed %s (sec): %1.10f (s) \n", str,elapsed); 
  }
}

//
// tridiagonal solver
//
void trid_cpu(FP* __restrict a, FP* __restrict b, FP* __restrict c, FP* __restrict d, FP* __restrict u, int N, int stride) {
  int   i, ind = 0;
  FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = 1.0f/b[0];
  cc    = bb*c[0];
  dd    = bb*d[0];
  c2[0] = cc;
  d2[0] = dd;

  //u[0] = dd;//a[0];
  //*((int*)&u[ind]) = (int)a[0];

  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = a[ind];
    bb    = b[ind] - aa*cc;
    dd    = d[ind] - aa*dd;
    bb    = 1.0f/bb;
    cc    = bb*c[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;

    //u[ind] = dd;//a[ind];
    //*((int*)&u[ind]) = (int)a[ind];

  }
  //
  // reverse pass
  //
  d[ind] = dd;
  //u[ind] = dd;//ind;//N-1;//dd;
  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
    d[ind] = dd;

    //u[ind] = dd;//ind;//i;//dd;//d2[i];//dd;
  }
}

void adi_cpu(FP lambda, FP* __restrict u, FP* __restrict du, FP* __restrict ax, FP* __restrict bx, FP* __restrict cx, FP* __restrict ay, FP* __restrict by, FP* __restrict cy, FP* __restrict az, FP* __restrict bz, FP* __restrict cz, int nx, int ny, int nz, double *elapsed_preproc, double *elapsed_trid_x, double *elapsed_trid_y, double *elapsed_trid_z, int prof) {
  int   i, j, k, ind;
  FP a, b, c, d;
  double elapsed, timer = 0.0;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //
  timing_start(prof,&timer);
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {   // i loop innermost for sequential memory access
        ind = k*nx*ny + j*nx + i;
        if(i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
          d = 0.0f; // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          d = lambda*(  u[ind-1    ] + u[ind+1]
                      + u[ind-nx   ] + u[ind+nx]
                      + u[ind-nx*ny] + u[ind+nx*ny] 
                      - 6.0f*u[ind]);
          a = -0.5f * lambda;
          b =  1.0f + lambda;
          c = -0.5f * lambda;
        }
        du[ind] = d;
        //*((int*)&ax[ind]) = ind;//a;
        ax[ind] = a;
        //ax[ind] = ind;//a;
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
  timing_start(prof,&timer);
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      ind = k*nx*ny + j*nx;
      trid_cpu(&ax[ind], &bx[ind], &cx[ind], &du[ind], &u[ind], nx, 1);
    }
  }
  timing_end(prof,&timer,elapsed_trid_x,"trid_x");

  //
  // perform tri-diagonal solves in y-direction
  //
  timing_start(prof,&timer);
  for(k=0; k<nz; k++) {
    for(i=0; i<nx; i++) {
      ind = k*nx*ny + i;
      trid_cpu(&ay[ind], &by[ind], &cy[ind], &du[ind], &u[ind], ny, nx);
    }
  }
  timing_end(prof,&timer,elapsed_trid_y,"trid_y");

  //
  // perform tri-diagonal solves in z-direction
  //
  timing_start(prof,&timer);
  for(j=0; j<ny; j++) {
    for(i=0; i<nx; i++) {
      ind = j*nx + i;
      trid_cpu(&az[ind], &bz[ind], &cz[ind], &du[ind], &u[ind], nz, nx*ny);
      //#pragma ivdep
      //for(k=0; k<NZ; k++) {
      //  u[ind] += du[ind];
      //  //u[ind] = du[ind];
      //  ind    += NX*NY;
      //}
    }
  }

  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {
        ind = k*nx*ny + j*nx + i;
        u[ind] += du[ind];
        //u[ind] = du[ind];
      }
    }
  }
  timing_end(prof,&timer,elapsed_trid_z,"trid_z");
}

int main(int argc, char* argv[]) { 
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_trid_x, elapsed_trid_y, elapsed_trid_z;

  // 'h_' prefix - CPU (host) memory space

  int i, j, k, ind, it;
  int nx, ny, nz, iter, opt, prof;
  FP  *__restrict__ h_u, *__restrict__ h_du,
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
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx   = atoi(optarg); //printf("nx   ===== %d\n",atoi(optarg));
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny   = atoi(optarg); //printf("ny   ===== %d\n",atoi(optarg));
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz   = atoi(optarg); //printf("nz   ===== %d\n",atoi(optarg));
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg); //printf("iter ===== %d\n",atoi(optarg));
    if(strcmp((char*)options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg); //printf("opt  ===== %d\n",atoi(optarg));
    if(strcmp((char*)options[opt_index].name,"prof") == 0) prof = atoi(optarg); //printf("prof ===== %d\n",atoi(optarg));
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
  }

  printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);

  if( nx>N_MAX || ny>N_MAX || nz>N_MAX ) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    return -1;
  }
  // allocate memory for arrays

  h_u  = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_du = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_ax = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_bx = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_cx = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_ay = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_by = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_cy = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_az = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_bz = (FP *)malloc(sizeof(FP)*nx*ny*nz);
  h_cz = (FP *)malloc(sizeof(FP)*nx*ny*nz);

  // Initialize
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {
        ind = k*nx*ny + j*nx + i;
        if(i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
          h_u[ind] = 1.0f;
        } else {
          h_u[ind] = 0.0f;
        }
      }
    }
  }

  // Compute sequentially
  elapsed_time(&timer);
  for(it =0; it<iter; it++) {
    adi_cpu(lambda, h_u, h_du, h_ax, h_bx, h_cx, h_ay, h_by, h_cy, h_az, h_bz, h_cz, nx, ny, nz, &elapsed_preproc, &elapsed_trid_x, &elapsed_trid_y, &elapsed_trid_z, prof); 
  }
  elapsed_total = elapsed_time(&timer);
  printf("\nComputing ADI on CPU(seq): %f (s) \n", elapsed_total);

  int ldim=nx; // Lead dimension
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
