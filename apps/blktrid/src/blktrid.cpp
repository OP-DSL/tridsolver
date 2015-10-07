/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the block-tridiagonal solver distribution.
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

// Block tridiagonal solver headers
#ifdef __GPU__
  #define CUDA_DEVICE 0
  #include "cutil_inline.h"
  #include "blktrid_gpu.h" // Place blktrid_gpu.h above cutil_inline.h because CUDA_DEVICE defined here is used in cutil
#endif
#include "blktrid_cpu.h"
#include "blktrid_util.h"

// MKL headers
#ifdef __MKL__
  #include "mkl.h"
#endif
#include <mkl_vsl.h>

// System headers
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#define KU (2*(blkdim)-1) // Number of upper off-diagonals
#define KL (KU)            // Number of lower off-diagonals
#define NUM_DIAGS ((2*(KL))+(KU)+1) // Number of diagonals with blkdim sized blocks in a block-tridiagonal system
#define LEN_DIAGS ((N)*(blkdim))   // Length of diagonals

// Argument handling structures
extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"devid",  required_argument, 0,  0   },
  {"n",      required_argument, 0,  0   },
  {"p",      required_argument, 0,  0   },
  {"blkdim", required_argument, 0,  0   },
  {"solver", required_argument, 0,  0   },
  {"iter",   required_argument, 0,  0   },
  {"help",   no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

/*
 * Print essential information on the use of the program
 */
void print_help() {
  printf("\nPlease specify arguments "
    "e.g. ./blktrid* -devid=DEVICEID -n=SYSTEMLENGTH -p=PROBLEMSIZE -blkdim=BLOCKDIM -solver={0-CPU,1-GPU,2-MKL,-1-ALL(Default)} -iter ITERATIONS \n");
  exit(0);
}

////////////////////////////////////////////////////////////////////////
// Main program  
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  // Process arguments
  int opt_index = 0;
  // Set defaults
  int devid  = 0;
  int N      = NLEN;
  int P      = PROB;
  int blkdim = 3;
  int solver = -1; // default value - every solver does
  int iter   = 1;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp(options[opt_index].name,"devid") == 0)  devid  = atoi(optarg);
    if(strcmp(options[opt_index].name,"n") == 0)      N      = atoi(optarg);
    if(strcmp(options[opt_index].name,"p") == 0)      P      = atoi(optarg);
    if(strcmp(options[opt_index].name,"blkdim") == 0) blkdim = atoi(optarg);
    if(strcmp(options[opt_index].name,"solver") == 0) solver = atoi(optarg);
    if(strcmp(options[opt_index].name,"iter") == 0)   iter   = atoi(optarg);
    if(strcmp(options[opt_index].name,"help") == 0)   print_help();
  }
  int blkels = blkdim*blkdim;
  printf("\nProblem Parameters:\nN:\t%d\nP:\t%d\nblkdim:\t%d\n", N, P, blkdim);

  // host variables
  FP *h_ACPU, *h_BCPU, *h_CCPU, *h_dCPU, *h_CAdj_CPU, *h_dAdj_CPU, *h_uCPU,
     *h_AGPU, *h_BGPU, *h_CGPU, *h_dGPU, *h_CAdj_GPU, *h_dAdj_GPU, *h_uGPU,
     *h_uLAPACKE, *h_uDifference, *h_bandMatrices;

  // device variables
  FP *d_A, *d_B, *d_C, *d_d, *d_u, *d_dstar, *d_Cstar;

  // timer variable and elapsed time
  double timer, elapsed, t, t_lapacke; 

  // FLOP counters
  long int FLOPcount;
  long int LUfactorization, backsubstitution, LAPACKE_FLOPcount; // LAPACKE

  /////////////////////
  // ALLOCATE memory //
  /////////////////////
  double dataCPU = sizeof(FP)*blkdim*N*P*6 + sizeof(FP)*blkels*N*P*5;
  printf("Allocating %.1f MB data on the CPU...", (dataCPU/1024)/1024);
  h_uGPU        = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_uCPU        = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_uLAPACKE    = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_bandMatrices= (FP *) malloc(sizeof(FP) * NUM_DIAGS * LEN_DIAGS * P);
  h_uDifference = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_dCPU        = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_dGPU        = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_ACPU        = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_BCPU        = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_CCPU        = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_AGPU        = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_BGPU        = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_CGPU        = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_CAdj_CPU    = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_CAdj_GPU    = (FP *) malloc(sizeof(FP) * blkels * N * P);
  h_dAdj_CPU    = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  h_dAdj_GPU    = (FP *) malloc(sizeof(FP) * blkdim * N * P);
  printf("done\n");

  
  /////////////////////
  // INITIALIZE data //
  /////////////////////

  /* initialize random seed: */
  srand (time(NULL));
  int errcode = 0;
  VSLStreamStatePtr stream;
  errcode = vslNewStream( &stream, VSL_BRNG_SOBOL, 10 );
  #if FPPREC == 0
    errcode = vsRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream, N*P*blkdim*blkdim, h_BCPU, 4.0, 4.01 );
  #elif FPPREC == 1
    errcode = vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream, N*P*blkdim*blkdim, h_BCPU, 4.0, 4.01 );
  #endif

  /* Deleting the stream */        
     vslDeleteStream( &stream );

  // Set up block-tridiagonal matrix for CPU
  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; n++) {
    for (int p = 0; p < P; p++) {
      for (int i = 0; i < blkdim; i++) {
        for (int j = 0; j < blkdim; ++j) {
          int idx = i * blkdim + j;
          if (i == j) {
            // Block tridiagonal storage
            h_ACPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.01);
            //h_BCPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(4.0 + (0.01*rand()/RAND_MAX)); // Perturbe matrix to have better validation
            h_CCPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.01);
          } else if (i==(j+1) || i==(j-1)) {
            // Block tridiagonal storage
            h_ACPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.0);
            h_BCPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.01);
            h_CCPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.0);
          } else {
            // Block tridiagonal storage
            h_ACPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.0);
            h_BCPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.0);
            h_CCPU[p * blkels * N + n * blkels + idx] = static_cast<FP>(0.0);
          }
        }
      }
    }
  }

#ifdef __GPU__
  // Set up block-tridiagonal matrix for GPU
  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; n++) {
    for (int p = 0; p < P; p++) {
      for (int i = 0; i < blkdim; i++) {
        for (int j = 0; j < blkdim; ++j) {
          int idx = i * blkdim + j;
          if (i == j) {
            // Block tridiagonal storage
            h_AGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.01);
            h_BGPU[n * blkels * P + p * blkels + idx] = h_BCPU[p * blkels * N + n * blkels + idx]; //static_cast<FP>(4.0);//4.0f+(0.01*rand()/RAND_MAX); // Perturbe matrix to have better validation
            h_CGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.01);
          } else if (i==(j+1) || i==(j-1)) {
            // Block tridiagonal storage
            h_AGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.0);
            h_BGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.01);
            h_CGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.0);
          } else {
            // Block tridiagonal storage
            h_AGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.0);
            h_BGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.0);
            h_CGPU[n * blkels * P + p * blkels + idx] = static_cast<FP>(0.0);
          }
        }
      }
    }
  }
#endif


#ifdef __MKL__
  // Set up coefficients of banded matrix
  #pragma omp parallel for collapse(3)
  for (int p = 0; p < P; p++) {
    for (int d = 0; d < NUM_DIAGS; d++) { //diagonals
      for (int n = 0; n < LEN_DIAGS; n++) { // elements in diagonals
        // Index mapping between standard matrix (i,j) indexing and banded storage indexing (d,n) with extra space for fill in
        // which is needed in dgbsv()
        //   d = KL + KU + i - j
        //   n = j
        // Inverse indexing
        int i = d-KL-KU+n;
        int j = n;
        if(i>=0 && j>=0 && i<=LEN_DIAGS-1 && j<=LEN_DIAGS-1) {
          //if(i==j)                 h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = static_cast<FP>(4.0);// + (0.01*rand()/RAND_MAX)); // Perturbe matrix to have better validation;
          //if(i==j)                 h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = h_B[(n/blkdim) * P * blkels + p * blkels + (n%blkdim)*blkdim+(n%blkdim)]; // Perturbe matrix to have better validation;
          //h_BCPU[p * blkels * N + n * blkels + idx]
          if(i==j)                 h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = h_BCPU[p * N * blkels + (n/blkdim) * blkels + (n%blkdim)*blkdim+(n%blkdim)]; // Perturbe matrix to have better validation;
          else if(abs(i-j)==1)     h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = static_cast<FP>(0.01);
          else if(j == (i-blkdim) || j == (i+blkdim) ) h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = static_cast<FP>(0.01);
          else                     h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = static_cast<FP>(0.0);
        } else {
          h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d] = static_cast<FP>(0.0);
        }
      }
    }
  }
#endif

  // Set up RHS
  #pragma omp parallel for
  for (int i = 0; i < N * P * blkdim; ++i) {
    h_dCPU[i] = static_cast<FP>(1.0);
    h_dGPU[i] = static_cast<FP>(1.0);
    h_uCPU[i] = static_cast<FP>(-999.0);
    h_uGPU[i] = static_cast<FP>(-999.0);
    h_uLAPACKE[i] = static_cast<FP>(1.0);
  }

/////////////
// RUN CPU //
/////////////
#ifdef __CPU__
  //if (DO_CPU) {
  printf("Running CPU GOLD code... ");
  elapsed_time(&timer); // initialise timer
  //const int iter=1;
  for(int it=0; it<iter; it++) {
    //blkThomas_GOLD(h_ACPU, h_BCPU, h_CCPU, h_CAdj_CPU, h_dAdj_CPU, h_dCPU, h_uCPU, N, P);
    //blkThomas_GOLD(h_ACPU, h_BCPU, h_CCPU, h_CCPU, h_dCPU, h_dCPU, h_uCPU, N, P);
    #if FPPREC==0
      sbtsv_cpu(h_ACPU, h_BCPU, h_CCPU, h_CCPU, h_dCPU, h_dCPU, h_uCPU, N, P, blkdim);
    #elif FPPREC==1
      dbtsv_cpu(h_ACPU, h_BCPU, h_CCPU, h_CCPU, h_dCPU, h_dCPU, h_uCPU, N, P, blkdim);
    #endif
  }
  //cudaDeviceSynchronize();
  elapsed = elapsed_time(&timer)/iter;
  printf("done\n");
  //}
  printf("CPU Block tridiagonal solver results: [first 3 subvectors]\n");
  for (int p = 0; p < 3/*P*/; ++p) {
    printf("Problem p=%d \n", p);
    for (int n = 0; n < 3/*N*/; ++n) {
      printf("  n=%d ", n);
      for (int i = 0; i < blkdim; ++i) {
        printf(" %f ", h_uCPU[p*N*blkdim + n*blkdim + i]);
      }
      printf(" | ");
    }
    printf("\n");
  }
  printf("\n");
#endif

/////////////
// RUN GPU //
/////////////
#ifdef __GPU__
  if(solver == 1 || solver == -1) {
    //cudaDeviceReset();
    cutilDeviceInit(argc, argv);
    cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
    // If double precion is used, switch to larger Shared memory bank size
    #if FPPREC == 0
      cudaSafeCall( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );
    #elif FPPREC == 1
      cudaSafeCall( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
    #endif

    double dataGPU = sizeof(FP)*blkdim*N*P*3 + sizeof(FP)*blkels*N*P*4;
    printf("Allocating %.1f MB data on the GPU...", (dataGPU/1024)/1024);
    cudaSafeCall(cudaMalloc((void **)&d_u,     sizeof(FP)*blkdim*N*P));
    cudaSafeCall(cudaMalloc((void **)&d_d,     sizeof(FP)*blkdim*N*P));
    cudaSafeCall(cudaMalloc((void **)&d_dstar, sizeof(FP)*blkdim*N*P));
    cudaSafeCall(cudaMalloc((void **)&d_A,     sizeof(FP)*blkels*N*P));
    cudaSafeCall(cudaMalloc((void **)&d_B,     sizeof(FP)*blkels*N*P));
    cudaSafeCall(cudaMalloc((void **)&d_C,     sizeof(FP)*blkels*N*P));
    cudaSafeCall(cudaMalloc((void **)&d_Cstar, sizeof(FP)*blkels*N*P));
    printf("done\n");

    // COPY data to device
    cudaSafeCall(cudaMemcpy(d_u, h_uGPU, sizeof(FP)*N*P*blkdim, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_d, h_dGPU, sizeof(FP)*N*P*blkdim, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_A, h_AGPU, sizeof(FP)*N*P*blkels, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_B, h_BGPU, sizeof(FP)*N*P*blkels, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_C, h_CGPU, sizeof(FP)*N*P*blkels, cudaMemcpyHostToDevice));

    printf("Running GPU code...");
    elapsed_time(&timer); // initialise timer
    for(int it=0; it<iter; it++) {
      #if FPPREC == 0
        sbtsv_gpu(d_A, d_B, d_C, d_d, d_Cstar, d_dstar, d_u, N, P, blkdim);
      #elif FPPREC == 1
        dbtsv_gpu(d_A, d_B, d_C, d_d, d_Cstar, d_dstar, d_u, N, P, blkdim);
      #endif
      //FP* tmp = d_u;
      //d_u = d_d;
      //d_d = tmp;
    }
    cudaCheckMsg("kernel failed to launch:\n");
    cudaSafeCall(cudaDeviceSynchronize());
    t = elapsed_time(&timer)/(double)iter;
    printf("done\n");

    // COPY back to host
    cudaSafeCall(cudaMemcpy(h_uGPU,         d_u, sizeof(FP)*N*P*blkdim, cudaMemcpyDeviceToHost));
    //cudaSafeCall(cudaMemcpy(h_uGPU,         d_d, sizeof(FP)*N*P*blkdim, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_CAdj_GPU, d_Cstar, sizeof(FP)*N*P*blkels, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_dAdj_GPU, d_dstar, sizeof(FP)*N*P*blkdim, cudaMemcpyDeviceToHost));

    printf("GPU Block tridiagonal solver results: [first 3 subvectors]\n");
    for (int p = 0; p < 3/*P*/; ++p) {
      printf("Problem p=%d \n", p);
      for (int n = 0; n < 3/*N*/; ++n) {
        printf("  n=%d ", n);
        for (int i = 0; i < blkdim; ++i) {
          printf(" %f ", h_uGPU[n*P*blkdim + p*blkdim + i]);
        }
        printf(" | ");
      }
      printf("\n");
    }
    printf("\n");

    cudaSafeCall(cudaFree(d_u));
    cudaSafeCall(cudaFree(d_d));
    cudaSafeCall(cudaFree(d_A));
    cudaSafeCall(cudaFree(d_B));
    cudaSafeCall(cudaFree(d_C));
    cudaSafeCall(cudaFree(d_Cstar));
    cudaSafeCall(cudaFree(d_dstar));
    cudaDeviceReset();
  }
#endif

////////////////
// RUN MKL    //
////////////////
#ifdef __MKL__
  if(solver == 2 || solver == -1) {
    //lapack_int LAPACKE_dgbsv( int matrix_order, lapack_int n, lapack_int kl,
    //                          lapack_int ku, lapack_int nrhs, double* ab,
    //                          lapack_int ldab, lapack_int* ipiv, double* b,
    //                          lapack_int ldb );
    int matrix_order = LAPACK_COL_MAJOR; // Matrix order
    lapack_int  n    = LEN_DIAGS; // Length/rank of the system, ie. number of equations within a system
    lapack_int  kl   = KL;//NUM_DIAGS/2; // Lower off-diagonals
    lapack_int  ku   = KU;//kl;          // Upper off-diagonals
    lapack_int  nrhs = 1;           // Number of RHS
    FP         *ab   = h_bandMatrices;
    lapack_int  ldab = NUM_DIAGS; // Leading dimension of AB matrix
    lapack_int *ipiv = (lapack_int*) malloc(LEN_DIAGS*P*sizeof(lapack_int)); // The pivot indices that define the permutation matrix P; row i of the matrix was interchanged with row IPIV(i).
    FP         *b    = h_uLAPACKE;//(double*) malloc(n*nrhs*P*sizeof(double)); // N-by-NRHS right hand side matrix B
    lapack_int  ldb  = LEN_DIAGS;
    lapack_int  info;

    // Print banded coefficient matrix
    //  printf("Banded coefficient matrix: KL=%d, KU=%d, LEN_DIAGS=%d\n",KL,KU,LEN_DIAGS);
    //  for (int p = 0; p < 3; p++) {
    //    printf("Problem p=%d \n", p);
    //    for (int d = 0; d < NUM_DIAGS; d++) { //diagonals
    //      printf("  d=%d ", d);
    //      for (int n = 0; n < 5; n++) { // elements in diagonals
    //        //printf(" %0.2f ", h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + d*LEN_DIAGS + n]);
    //        printf(" %0.2f ", h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d]);
    //      }
    //      printf(" ... ", d);
    //      for (int n = LEN_DIAGS-5; n < LEN_DIAGS; n++) { // elements in diagonals
    //        //printf(" %0.2f ", h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + d*LEN_DIAGS + n]);
    //        printf(" %0.2f ", h_bandMatrices[p * NUM_DIAGS * LEN_DIAGS + n*NUM_DIAGS + d]);
    //      }
    //      printf("\n");
    //    }
    //  }

    // Run banded solver LAPACKE_dgbsv()
    printf("Running CPU MKL code...\n");
    elapsed_time(&timer); // initialise timer
    for(int it=0; it<iter; it++) {
      #pragma omp parallel for
      for(int p=0; p<P; p++) {
        //printf("p=%d : Running CPU LAPACKE code...\n",p);
        #if FPPREC == 0
          //sgbsv(&n, &kl, &ku, &nrhs, &ab[p*NUM_DIAGS*LEN_DIAGS], &ldab, &ipiv[p*LEN_DIAGS], &b[p*LEN_DIAGS], &ldb, &info);
          info = LAPACKE_sgbsv_work(matrix_order, n, kl, ku, nrhs, &ab[p*NUM_DIAGS*LEN_DIAGS], ldab, &ipiv[p*LEN_DIAGS], &b[p*LEN_DIAGS], ldb);
        #elif FPPREC == 1
          //dgbsv(&n, &kl, &ku, &nrhs, &ab[p*NUM_DIAGS*LEN_DIAGS], &ldab, &ipiv[p*LEN_DIAGS], &b[p*LEN_DIAGS], &ldb, &info);
          info = LAPACKE_dgbsv_work(matrix_order, n, kl, ku, nrhs, &ab[p*NUM_DIAGS*LEN_DIAGS], ldab, &ipiv[p*LEN_DIAGS], &b[p*LEN_DIAGS], ldb);
        #endif
        if(info!=0) printf("CPU LAPACKE_dgbsv() info = %d \n",info);
        // Check permutation vector for reordering due to pivoting
        for(int n=0; n<LEN_DIAGS; n++) {
          if(n != (ipiv[p*LEN_DIAGS + n] - 1)) printf("CPU LAPACKE_dgbsv() ipiv shows reordering: p=%d  ipiv[%d] -> %d \n",p,n,ipiv[p*LEN_DIAGS+n]);
        }
      }
    }
    t_lapacke = elapsed_time(&timer) / (double) iter;
    printf("done\n");

    printf("LAPACKE Banded solver results: \n");
    for (int p = 0; p < 3; p++) {
      printf("Problem p=%d \n", p);
      for (int n = 0; n < 8; n++) { // elements in diagonals
        printf(" %f ", b[p * LEN_DIAGS + n]);
      }
      printf("\n");
    }
    printf("\nipiv= \n");
    for (int j = 0; j < 16; ++j) { // elements in diagonals
      printf(" %d ", ipiv[j]);
    }
    printf("\n");

    free(ipiv);
    //Operation count - from the algorithm in Golub and van Loan book
    LUfactorization  = 2*N*kl*ku;
    backsubstitution = 2*N*ku;
    LAPACKE_FLOPcount = P*(LUfactorization + backsubstitution);
  }


#endif

  
  //////////////////////////////
  // PERFORMANCE calculations //
  //////////////////////////////

  //calculate floating-point operations required
  long int MMmul_FMul = blkdim * blkdim * blkdim; 
  long int MMmul_FAdd = blkdim * blkdim * blkdim;
  long int MVmul_FMul = blkdim * blkdim;
  long int MVmul_FAdd = blkdim * blkdim;
  long int Madd_FMul  = 2 * blkdim * blkdim;
  long int Madd_FAdd  = blkdim * blkdim;
  long int Vadd_FMul  = 0;
  long int Vadd_FAdd  = blkdim;
  long int GJ_FMul    = 2 * (blkdim - 1) * blkdim * blkdim;
  long int GJ_FAdd    = 2 * (blkdim - 1) * blkdim * blkdim;
  long int GJ_FDiv    = 2 * blkdim * blkdim;
  // Numbers are per warp
  //long int MMmul_FMul = (N-1) * blkdim * blkdim * ACTIVE_T_PER_WARP;
  //long int MMmul_FAdd = (N-1) * blkdim * blkdim * ACTIVE_T_PER_WARP; 
  //long int MVmul_FMul = (N-1) * blkdim * ACTIVE_T_PER_WARP; 
  //long int MVmul_FAdd = (N-1) * blkdim * ACTIVE_T_PER_WARP; 
  //long int Madd_FMul  = 2 * ((N-1) * blkdim * ACTIVE_T_PER_WARP); 
  //long int Madd_FAdd  = 2 * ((N-1) * blkdim * ACTIVE_T_PER_WARP); 
  //long int Vadd_FMul  = 0; 
  //long int Vadd_FAdd  = 2 * ( (N-1) * ACTIVE_T_PER_WARP); 
  //long int GJ_FMul    = N * (2 * blkdim * ACTIVE_T_PER_WARP + PROBS_PER_WARP + blkdim * blkdim * 2 * ACTIVE_T_PER_WARP + PROBS_PER_WARP * blkdim);
  //long int GJ_FAdd    = N * (blkdim * blkdim * 2 * ACTIVE_T_PER_WARP + PROBS_PER_WARP * blkdim);
  //long int GJ_FDiv    = N * (blkdim * ACTIVE_T_PER_WARP);

  //calculate integer operations required (approximate)
  long int MMmul_IMul = 2 * blkdim * blkdim * blkdim + blkdim * blkdim;
  long int MMmul_IAdd = 3 * blkdim * blkdim * blkdim + 2 * blkdim * blkdim;
  long int MVmul_IMul = blkdim * blkdim;
  long int MVmul_IAdd = 2 * blkdim * blkdim;
  long int Madd_IMul  = blkdim * blkdim;
  long int Madd_IAdd  = 2 * blkdim * blkdim + blkdim;
  long int GJ_IMul    = blkdim * blkdim * (6 * blkdim - 1);
  long int GJ_IAdd    = blkdim * blkdim * (9 * blkdim - 3);

  //total number of operations of each type (per problem)
  long int nMMmul = N - 1;
  long int nMVmul = 2 * (N - 1);
  long int nMadd  = (N - 1);
  long int nVadd  = 2 * (N - 1);
  long int nGJ    = N;

  //total floating-point and integer operations of each type
  long int total_FMul = P * (MMmul_FMul * nMMmul + MVmul_FMul * nMVmul + Madd_FMul * nMadd + Vadd_FMul * nVadd + GJ_FMul * nGJ);
  long int total_FAdd = P * (MMmul_FAdd * nMMmul + MVmul_FAdd * nMVmul + Madd_FAdd * nMadd + Vadd_FAdd * nVadd + GJ_FAdd * nGJ);
  long int total_FDiv = P * GJ_FDiv * nGJ;

  long int total_IMul = P * (MMmul_IMul * nMMmul + MVmul_IMul * nMVmul + Madd_IMul * nMadd + GJ_IMul * nGJ);
  long int total_IAdd = P * (MMmul_IAdd * nMMmul + MVmul_IAdd * nMVmul + Madd_IAdd * nMadd + GJ_IAdd * nGJ);

  long int FMul_weight = 1, FAdd_weight = 1, FDiv_weight = 1, IMul_weight = 1, IAdd_weight = 1;
  //long int FMul_weight = 1, FAdd_weight = 1, FDiv_weight = 8, IMul_weight = 2, IAdd_weight = 1;
  //if(sizeof(FP) == 8){
  //  FMul_weight = 2;
  //  FAdd_weight = 2;
  //}

  //weighted total FLOPs and IOPs
  long int totalFLOPs = total_FMul * FMul_weight + total_FAdd * FAdd_weight + total_FDiv * FDiv_weight;
  //long int totalFLOPs = P/WARP_SIZE * (MMmul_FMul+MMmul_FAdd+MVmul_FMul+MVmul_FAdd+Madd_FMul+Madd_FAdd+Vadd_FMul+Vadd_FAdd+GJ_FMul+GJ_FAdd+GJ_FDiv);
  long int totalIOPs = total_IMul * IMul_weight + total_IAdd * IAdd_weight;
  
  //long int matLoads = 6 * N - 3, vecLoads = 4 * N - 1;
  //long int totalData = (matLoads * blkels + vecLoads * blkdim) * P * sizeof(FP);
  //long int matLoadStore = 6 * N - 3;
  //long int vecLoadStore = 6 * N - 1;
  long int matLoadStore = 5 * N - 3;
  long int vecLoadStore = 5 * N - 1;
  long int totalLoadStore = (matLoadStore * blkels + vecLoadStore * blkdim) * P * sizeof(FP);
  
  ///////////////////
  // PRINT options //
  ///////////////////

  printf("\n");
  printf("CPU \n  duration [ms]: %lf\n  GFLOPS:        %lf\n  GB/sec:        %lf\n", elapsed * 1000, (totalFLOPs / (double)elapsed) * 1e-9, (totalLoadStore/(double) elapsed)*1e-9);
  //printf("GPU \n  duration [ms]: %lf\n  GFLOPS:         %f\n  GB/sec:         %lf\n", t * 1000, (totalFLOPs / t) / (1e9), (totalData / t) / (1e9));
  printf("GPU \n  duration [ms]: %lf\n  GFLOPS:        %lf\n  GB/sec:        %lf\n", t * 1000, (totalFLOPs / (double)t) * 1e-9, (totalLoadStore /(double) t) * 1e-9);
  printf("LAPACKE \n  duration [ms]: %lf\n  GFLOPS:        X\n  GB/sec:        X\n", t_lapacke * 1000);
  printf("\n");

//  //for (int i = 0; i < blkdim * N * P; ++i) {
//  for (int p = 0; p < 3/*P*/; ++p) {
//    printf("Problem p=%d \n", p);
//    for (int n = 0; n < 3/*N*/; ++n) {
//      printf("  n=%d ", n);
//      for (int i = 0; i < blkdim; ++i) {
//        printf(" %f ", h_uGPU[p*N*blkdim + n*blkdim + i*blkdim]);
//      }
//      printf("\n");
//    }
//  }
//  printf("\n");

// Validate if compiled with proper options and solver is selected
#if defined (__GPU__) && defined (__CPU__)
  if(solver==-1 || solver == 1) {
    FILE *fileGPU;
    fileGPU = fopen("results_binary", "w");
    fwrite(h_uGPU, blkdim * N * P, sizeof(FP), fileGPU);
    fclose(fileGPU);

    printf("\nDiscrepancies CPU - GPU:\n");
    int nDiscrep_GPU = 0;
    int nDiscrepFPE_GPU = 0;
    for (int p = 0; p < P; ++p) {
      for (int n = 0; n < N; ++n) {
        for (int i = 0; i < blkdim; ++i) {
          int idx_GPU = n * P * blkdim + p * blkdim + i;
          int idx_CPU = p * N * blkdim + n * blkdim + i;
          // CPU - GPU discrepancies
          if(h_uCPU[idx_CPU] != h_uGPU[idx_GPU]) {
            ++nDiscrep_GPU;
            //if (abs(h_uCPU[idx_CPU] - h_uGPU[idx_GPU]) < abs(h_uCPU[idx_CPU] / FPE_CUTTOFF) ) {
            if( (fabs(h_uCPU[idx_CPU] - h_uGPU[idx_GPU]) / fabs(h_uCPU[idx_CPU] ) ) < 1e-6 ) {
              ++nDiscrepFPE_GPU;
            } else {
              if((nDiscrep_GPU-nDiscrepFPE_GPU)<100)
                printf("idx_CPU=%d\t p=%d\t n=%d\t i=%d\t h_uCPU=%f\t h_uGPU=%f\t diff=%.10f\t rel_diff=%.10f\n", idx_CPU, p, n, i, h_uCPU[idx_CPU], h_uGPU[idx_GPU], h_uCPU[idx_CPU] - h_uGPU[idx_GPU], (h_uCPU[idx_CPU] - h_uGPU[idx_GPU]) / h_uCPU[idx_CPU]);
            }
          }
        }
      }
    }
    printf("Round-off errors[<1e-7] %d \n", nDiscrepFPE_GPU);
    printf("NOT due to round-off:   %d \n", nDiscrep_GPU - nDiscrepFPE_GPU);
    printf("------------------------------------------------\n");
    printf("Total discrepancies:    %d of %d\n", nDiscrep_GPU, N * P * blkdim);
  }
#endif

// Validate if compiled with proper options and solver is selected
#if defined (__MKL__) && defined (__CPU__)
  if(solver==-1 || solver == 2) {
    printf("\nDiscrepancies CPU - LAPACKE:\n");
    //  int counter = 0;
    int nDiscrep_LAPACKE = 0;
    int nDiscrepFPE_LAPACKE = 0;

    for (int n = 0; n < N; ++n) {
      for (int p = 0; p < P; ++p) {
        for (int i = 0; i < blkdim; ++i) {
          int idx_CPU     = n * P * blkdim + p * blkdim + i;
          int idx_LAPACKE = p * N * blkdim + n * blkdim + i;
          // CPU - LAPACK discrepancies
          if (h_uCPU[idx_CPU] != h_uLAPACKE[idx_LAPACKE]) {
            ++nDiscrep_LAPACKE;
            //if (abs(h_uCPU[idx] - h_uLAPACKE[idx_LAPACKE]) < abs(h_uCPU[idx] / FPE_CUTTOFF)) {
            //if (fabs(h_uCPU[idx] - h_uLAPACKE[idx_LAPACKE]) < fabs(h_uCPU[idx] / 0.00001)) {
            if( (fabs(h_uCPU[idx_CPU] - h_uLAPACKE[idx_LAPACKE]) / fabs(h_uCPU[idx_CPU] ) ) < 1e-2 ) {
              ++nDiscrepFPE_LAPACKE;
            } else {
              if((nDiscrep_LAPACKE-nDiscrepFPE_LAPACKE)<100)
                printf("idx_CPU=%d\t p=%d\t n=%d\t i=%d\t h_uCPU=%f\t h_uLAPACKE=%f\t diff=%.10f\t rel_diff=%.10f\n", idx_CPU, p, n, i, h_uCPU[idx_CPU], h_uLAPACKE[idx_LAPACKE], h_uCPU[idx_CPU] - h_uLAPACKE[idx_LAPACKE], (h_uCPU[idx_CPU] - h_uLAPACKE[idx_LAPACKE]) / h_uCPU[idx_CPU]);
            }
          }
          //if(idx<1024 && counter < 100) {
          //        if(idx<1024) {
          //          printf("idx=%d\t p=%d\t n=%d\t i=%d\t h_uCPU=%f\t h_uLAPACKE=%f\t diff=%.10f\t rel_diff=%.10f\n", idx, p, n, i, h_uCPU[idx], h_uLAPACKE[idx_LAPACKE], h_uCPU[idx] - h_uLAPACKE[idx_LAPACKE], (h_uCPU[idx] - h_uLAPACKE[idx_LAPACKE]) / h_uCPU[idx]);
          //          counter++;
          //        }
        }
      }
    }
    printf("Round-off errors[<1e-2]: %d \n", nDiscrepFPE_LAPACKE);
    printf("NOT due to round-off:    %d \n", nDiscrep_LAPACKE - nDiscrepFPE_LAPACKE);
    printf("------------------------------------------------\n");
    printf("Total discrepancies:     %d of %d\n", nDiscrep_LAPACKE, N * P * blkdim);
  }
#endif

  /////////////////
  // FREE memory //
  /////////////////
  free(h_ACPU);
  free(h_BCPU);
  free(h_CCPU);
  free(h_dCPU);
  free(h_uCPU);
  free(h_CAdj_CPU);
  free(h_dAdj_CPU);

  free(h_AGPU);
  free(h_BGPU);
  free(h_CGPU);
  free(h_dGPU);
  free(h_uGPU);
  free(h_CAdj_GPU);
  free(h_dAdj_GPU);

  free(h_bandMatrices);
  free(h_uLAPACKE);

  free(h_uDifference);

  printf("\n");
  printf("Program End\n");
  printf("\n");

  // Print execution times for benchmarking
  printf("Profiling output\n");

#ifdef __CPU__
  if(solver == 0 || solver==-1)
    printf(" |                          [CPU]                         | ");
#endif
#ifdef __GPU__
  if(solver == 1 || solver==-1)
    printf("\t |                        [GPU]                           | ");
#endif
#ifdef __MKL__
  if(solver == 2 || solver==-1)
    printf("\t |                      [LAPACKE]                         | ");
#endif
  printf("\n");

#ifdef __CPU__
  if(solver == 0 || solver==-1)
    printf(" | Time\t\t Time/block\t GFLOPS\t\t GBs\t  | ");
#endif
#ifdef __GPU__
  if(solver == 1 || solver==-1)
    printf("\t | Time\t\t Time/block\t GFLOPS\t\t GBs\t  | ");
#endif
#ifdef __MKL__
  if(solver == 2 || solver==-1)
    printf("\t | Time\t\t Time/block\t GFLOPS\t\t GBs\t  | ");
#endif
  printf("\n");


#ifdef __CPU__
  if(solver == 0 || solver==-1)
    printf("   %e %e \t%lf \t%lf ", elapsed, elapsed/(double)(N*P), (totalFLOPs/(double)elapsed)*1e-9, (totalLoadStore/(double) elapsed)*1e-9);
#endif
#ifdef __GPU__
  if(solver == 1 || solver==-1) {
    printf("\t   %e %e \t%lf \t%lf ", t, t/(double)(N*P), (totalFLOPs/(double)t)*1e-9, (totalLoadStore/(double) t)*1e-9);
  }
#endif
#ifdef __MKL__
  if(solver == 2 || solver==-1) {
    printf("\t   %e %e \t%lf \t%lf ", t_lapacke, t_lapacke/(double)(N*P), (LAPACKE_FLOPcount/(double)t_lapacke)*1e-9, (totalLoadStore/(double) t_lapacke)*1e-9);
  }
#endif
  printf("\n");

  fflush(0);

  return 0;
}
