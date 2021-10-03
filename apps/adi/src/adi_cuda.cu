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
#include <string>

#include "trid_common.h"
#include "trid_util.h"
#include "trid_cuda.h"
#include "cutil_inline.h"

#define ROUND_DOWN(N,step) (((N)/(step))*step)

__global__ void preproc(FP lambda, FP* u, FP* du, FP* ax, FP* bx, FP* cx, FP* ay, FP* by, FP* cy, FP* az, FP* bz, FP* cz, int nx, int ny, int nz);

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"devid",required_argument, 0,  0   },
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"optx", required_argument, 0,  0   },
  {"opty", required_argument, 0,  0   },
  {"optz", required_argument, 0,  0   },
  {"prof", required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

/*
 * Print essential information on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -devid DEVICEID -nx NX -ny NY -nz NZ -iter ITER -optx OPTX -opty=OPTY -optz=OPTZ -prof PROF\n");
  exit(0);
}

inline void timing_start(int prof, double *timer) {
  if(prof==1) elapsed_time(timer);
}

//inline void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
inline void timing_end(int prof, double *timer, double *elapsed_accumulate, std::string str) {
  double elapsed;
  if(prof==1) {
    cudaSafeCall( cudaDeviceSynchronize() );
    elapsed = elapsed_time(timer);
    *elapsed_accumulate += elapsed;
    //printf("\n elapsed %s (sec): %1.10f (s) \n", str.c_str() ,elapsed);
  }
}

void rms(char* name, FP* array, int nx, int ny, int nz, int padx) {
  //Sum the square of values in app.h_u
  double sum = 0.0;
  for(int k = 0; k < nz; k++) {
    for(int j = 0; j < ny; j++) {
      for(int i = 0; i < nx; i++) {
        int ind = k * padx * ny + j * padx + i;
        //sum += array[ind]*array[ind];
        sum += array[ind];
      }
    }
  }

  printf("%s sum = %lg\n", name, sum);
}

int main(int argc, char* argv[]) {
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_trid_x, elapsed_trid_y, elapsed_trid_z;
  int    i, j, k, ind, it;
  int    devid, nx, ny, nz, iter, optx, opty, optz, prof;

  // 'h_' prefix - CPU (host) memory space
  FP  *h_u, *h_du,
      *h_ax, *h_bx, *h_cx,
      *h_ay, *h_by, *h_cy,
      *h_az, *h_bz, *h_cz,
      lambda=1.0f; // lam = dt/dx^2

  // 'd_' prefix - GPU (device) memory space
  FP  *d_u,  *d_du, *d_du2, *d_du3,
      *d_ax, *d_bx, *d_cx,
      *d_ay, *d_by, *d_cy,
      *d_az, *d_bz, *d_cz,
      *d_buffer;

  // Process arguments
  int opt_index = 0;

  // Set defaults options
  devid= 0;
  nx   = 256;
  ny   = 256;
  nz   = 256;
  iter = 10;
  optx = 0;
  opty = 0;
  optz = 0;
  prof = 1;

  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp(options[opt_index].name,"devid") == 0) devid   = atoi(optarg);
    if(strcmp(options[opt_index].name,"nx"  ) == 0) nx   = atoi(optarg);
    if(strcmp(options[opt_index].name,"ny"  ) == 0) ny   = atoi(optarg);
    if(strcmp(options[opt_index].name,"nz"  ) == 0) nz   = atoi(optarg);
    if(strcmp(options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp(options[opt_index].name,"optx" ) == 0) optx  = atoi(optarg);
    if(strcmp(options[opt_index].name,"opty" ) == 0) opty  = atoi(optarg);
    if(strcmp(options[opt_index].name,"optz" ) == 0) optz  = atoi(optarg);
    if(strcmp(options[opt_index].name,"prof") == 0) prof = atoi(optarg);
    if(strcmp(options[opt_index].name,"help") == 0) print_help();
  }

  printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);

  if( nx>N_MAX || ny>N_MAX || nz>N_MAX ) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    exit(1);
  }

  // Initialise GPU
  cudaSafeCall( cudaDeviceReset() );
  cutilDeviceInit(argc, argv);
  cudaSafeCall( cudaSetDevice(devid) );

  // Allocate memory for arrays
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

  cudaSafeCall(cudaMalloc((void **)&d_u,      sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_du,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_du2,    sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_du3,    sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_ax,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_bx,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_cx,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_ay,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_by,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_cy,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_az,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_bz,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_cz,     sizeof(FP)*(nx*ny*nz + 16)) );
  cudaSafeCall(cudaMalloc((void **)&d_buffer, sizeof(FP)*(nx*ny*nz + 16)) );

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

  size_t limit = 0;

  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
  printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

  // Copy u and constants to device
  elapsed_time(&timer);
    cudaSafeCall(cudaMemcpy(d_u, h_u, sizeof(FP)*nx*ny*nz, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaDeviceSynchronize());
  elapsed = elapsed_time(&timer);
  printf("\nCopy u to device: %f (s) \n", elapsed);

  // Set up the execution configuration
  // If double precion is used, switch to larger Shared memory bank size
  #if FPPREC == 0
    cudaSafeCall( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );
  #elif FPPREC == 1
    cudaSafeCall( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
  #endif

  // Set L1 cache or Shared memory preference for kernels
  //cudaSafeCall( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );

  //  cudaSafeCall( cudaFuncSetCacheConfig(preproc, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_linear, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_linear_shared, cudaFuncCachePreferShared) );
  //
  //  //cudaSafeCall( cudaFuncSetCacheConfig(trid_x_pcrgj<FP, WARP_SIZE>, cudaFuncCachePreferShared) );
  //
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_strided<FP,FP>, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_strided_add<FP,FP>, cudaFuncCachePreferL1) );
  //#if FPPREC == 0
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_linear_reg16_float4, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_x_reg8_float4, cudaFuncCachePreferL1) );
  //  //cudaSafeCall( cudaFuncSetCacheConfig(trid_y_float4, cudaFuncCachePreferL1) );
  //  //cudaSafeCall( cudaFuncSetCacheConfig(trid_strided<float,float4>, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_strided_add<float,float4>, cudaFuncCachePreferL1) );
  //#elif FPPREC == 1
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_linear_reg8_double2, cudaFuncCachePreferL1) );
  //  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_x_shared_float4, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_strided<double,double2>, cudaFuncCachePreferL1) );
  //  cudaSafeCall( cudaFuncSetCacheConfig(trid_strided_add<double,double2>, cudaFuncCachePreferL1) );
  //#endif


  // Timing variables
  // Reset elapsed time counters
  elapsed_total   = 0.0;
  elapsed_preproc = 0.0;
  elapsed_trid_x  = 0.0;
  elapsed_trid_y  = 0.0;
  elapsed_trid_z  = 0.0;

  // Tridiagonal solver option arguemnt's setup
  int ndim = 3; // Number of dimensions of the (hyper)cubic data structure.
  int dims[3];  // Array containing the sizes of each ndim dimensions. size(dims) == ndim <=MAXDIM
  int pads[3];  // Padded sizes along each ndim number of dimensions
  dims[0] = nx;
  dims[1] = ny;
  dims[2] = nz;
  pads[0] = dims[0];
  pads[1] = dims[1];
  pads[2] = dims[2];

  int solvedim;   // user chosen dimension for which the solution is performed

  int opts[MAXDIM];// = {nx, ny, nz,...};
  //int *opts = get_opts();
  opts[0] = optx;
  opts[1] = opty;
  opts[2] = optz;

  int sync = 1; // Host-synchronous kernel execution

  elapsed_time(&timer2);
  for(it = 0; it<iter; it++) {
    timing_start(prof,&timer);
      dim3 dimGrid1(1+(nx-1)/32, 1+(ny-1)/4);
      dim3 dimBlock1(32,4);
      preproc<<<dimGrid1, dimBlock1>>>(lambda,  d_u,  d_du, d_ax, d_bx, d_cx, d_ay, d_by, d_cy, d_az, d_bz, d_cz, nx, ny, nz);
      cudaCheckMsg("preproc execution failed\n");
    timing_end(prof,&timer,&elapsed_preproc,"preproc");

    timing_start(prof,&timer);
      solvedim = 0;
      //tridMultiDimBatchSolve<FP,0>(d_ax, d_bx, d_cx, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #if FPPREC==0
        tridSmtsvStridedBatch(d_ax, d_bx, d_cx, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #elif FPPREC==1
        tridDmtsvStridedBatch(d_ax, d_bx, d_cx, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #endif
    timing_end(prof,&timer,&elapsed_trid_x,"trid_x");

    timing_start(prof,&timer);
      solvedim = 1;
      //if(opts[1]==4) trid_y_cusparse<FP>(&handle_sp, &d_ay, &d_by, &d_cy, &d_du, &d_u, nx, ny, nz, &d_buffer);
      //else           tridMultiDimBatchSolve<FP,0>(d_ay, d_by, d_cy, d_du, d_u, ndim, solvedim, dims, pads, opts, &d_buffer, sync);
      //tridMultiDimBatchSolve<FP,0>(d_ay, d_by, d_cy, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #if FPPREC==0
        tridSmtsvStridedBatch(d_ay, d_by, d_cy, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #elif FPPREC==1
        tridDmtsvStridedBatch(d_ay, d_by, d_cy, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #endif
    timing_end(prof,&timer,&elapsed_trid_y,"trid_y");

    timing_start(prof,&timer);
      solvedim = 2;
      //if(opts[2]==4) {
      //  trid_z_cusparse<FP>(&handle_sp, &d_az, &d_bz, &d_cz, &d_du, &d_u, nx, ny, nz, &d_buffer);
      //  add_contrib(d_du, d_u, nx, ny, nz);
      //}
      //else           tridMultiDimBatchSolve<FP,1>(d_az, d_bz, d_cz, d_du, d_u, ndim, solvedim, dims, pads, opts, &d_buffer, sync);
      //tridMultiDimBatchSolve<FP,1>(d_az, d_bz, d_cz, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #if FPPREC==0
        tridSmtsvStridedBatchInc(d_az, d_bz, d_cz, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #elif FPPREC==1
        tridDmtsvStridedBatchInc(d_az, d_bz, d_cz, d_du, d_u, ndim, solvedim, dims, pads, opts, sync);
      #endif
    timing_end(prof,&timer,&elapsed_trid_z,"trid_z");
  }

  cudaSafeCall( cudaDeviceSynchronize() );
  elapsed_total = elapsed_time(&timer2);
  printf("\nComputing ADI on GPU: %f (s) \n", elapsed_total);

  // Get GPU results
  elapsed_time(&timer);
    //cudaSafeCall(cudaMemcpy(h_u, d_du, sizeof(FP)*nx*ny*nz, cudaMemcpyDeviceToHost) );
    cudaSafeCall(cudaMemcpy(h_u, d_u, sizeof(FP)*nx*ny*nz, cudaMemcpyDeviceToHost) );
  elapsed = elapsed_time(&timer);
  printf("\nCopy u to host: %f (s) \n", elapsed);

  rms("end h_u", h_u, nx, ny, nz, dims[0]);

  // Release GPU and CPU memory
  cudaSafeCall(cudaFree(d_u) );
  cudaSafeCall(cudaFree(d_du));
  cudaSafeCall(cudaFree(d_du2));
  cudaSafeCall(cudaFree(d_du3));
  cudaSafeCall(cudaFree(d_ax));
  cudaSafeCall(cudaFree(d_bx));
  cudaSafeCall(cudaFree(d_cx));
  cudaSafeCall(cudaFree(d_ay));
  cudaSafeCall(cudaFree(d_by));
  cudaSafeCall(cudaFree(d_cy));
  cudaSafeCall(cudaFree(d_az));
  cudaSafeCall(cudaFree(d_bz));
  cudaSafeCall(cudaFree(d_cz));
  cudaSafeCall(cudaFree(d_buffer));
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

  cudaDeviceReset();

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

  // Make sure the device is left in a consistent state
  cudaSafeCall( cudaDeviceReset() );
  exit(0);
}
