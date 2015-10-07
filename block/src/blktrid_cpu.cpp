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
 
#include <stdlib.h>
#include <stdio.h>
#include "blktrid_common.h"
#include "math.h" // Needed for isnan(), isinf(), NAN macro and fabs()

#include <omp.h>

//#define ROUND_DOWN(num,step) ((num) & ((step)-1))
#define ROUND_DOWN(num,step) (((num)/(step))*step)

/////////////////////
// GOLD functions //
/////////////////////
template<typename REAL>
inline void gold_invert2x2(REAL*  M, REAL*  ans){
  REAL a = M[0], b = M[1], c = M[2], d = M[3];
  REAL det = a*d-b*c;

  //if(det == 0 ) throw 0;

  ans[0] = d/det;
  ans[1] = (-1)*b / det;
  ans[2] = (-1)*c / det;
  ans[3] = a/det;
}

template<typename REAL>
inline void gold_invert3x3(REAL*  M, REAL*  ans){
  REAL a = M[0], b = M[1], c = M[2],
     d = M[3], e = M[4], f = M[5],
     g = M[6], h = M[7], i = M[8];

  REAL A = e*i - f*h, D = -(b*i-c*h), G = (b*f-c*e),
     B = -(d*i-f*g), E = (a*i-c*g), H = -(a*f-c*d),
     C = (d*h-e*g), F = -(a*h-b*g), I = (a*e-b*d);

  REAL det = a*A+b*B+c*C;

  //if(det ==0) throw 0;

  ans[0] = A/det; ans[1] = D/det; ans[2] = G/det;
  ans[3] = B/det; ans[4] = E/det; ans[5] = H/det;
  ans[6] = C/det; ans[7] = F/det; ans[8] = I/det;

}

// computes the inverse of matrix a and stores it in matrix I
// this code is destructive. a is transformed into the identity matrix
template<typename REAL, int DIM, int UNROLL_STEP>
inline void gold_invert(REAL*  a, REAL*  I) {//, const int matrixSide ){

  if(DIM == 2) {
    gold_invert2x2(a,I);
    return;
  }

  if(DIM == 3){
    gold_invert3x3(a,I);
    return;
  }

  // Note that all matrices in this function are stored row-wise.
  // I'm also using a[j][i] = a[DIM*j + i], cause this guy sucks.

  REAL temp_0=0,temp_1=0,temp_2=0,temp_3=0,temp_4=0,temp_5=0;
  int i=0,j=0,p=0,q=0;

  // Initialize our identity matrix
//  for(i=0; i<DIM; i++) {
//    for(j=0;j<DIM;j++) {
//      if(i==j) I[DIM*j + i] = 1;
//      else I[DIM*j + i] = 0;
//    }
//  }
  for(i=0; i<DIM; i++) {
    #pragma simd
    for(j=0;j<DIM; j++) {
      I[DIM*i + j] = static_cast<REAL>(0.0);
    }
    I[DIM*i + i] = static_cast<REAL>(1.0);
  }

  for(i=0; i<DIM; i++) {
    temp_1 = fabs( a[i + DIM*i] );
    //if(temp_1<0) temp_1 = temp_1*(-1);
    //if(temp_1<0) temp_1 = -temp_1;
    //temp_1 = fabs(temp);
    p=i;
    for(j=i+1;j<DIM;j++){
      temp_0 = fabs( a[DIM*j + i] );
//      if(a[DIM*j + i]<0) {
//        temp_0 = -a[DIM*j + i];
//      } else {
//        temp_0 =  a[DIM*j + i];
//      }
      //if(temp_1<0) temp_1=temp_1*(-1);
      //if(temp_1<0) temp_1 = -temp_1;
      //temp_1 = fabs(temp_1);
      if(temp_0>temp_1){
        p=j;
        //temp_1=a[DIM*j + i];
        temp_1 = fabs(a[DIM*j + i]);
      }
    }

    //row exchange in both the matrices
    #pragma ivdep
    for(j=0; j<DIM; j++){
      temp_2=a[DIM*i + j];
      a[DIM*i + j] = a[DIM*p + j];
      //a[DIM*i + j] = temp_2;
      a[DIM*p + j] = temp_2;
      temp_3 = I[DIM*i + j];
      I[DIM*i + j] = I[DIM*p + j];
      I[DIM*p + j] = temp_3;
    }

    //dividing the row by a[i][i]
    temp_4 = a[DIM*i + i];
//    if (temp_4 == 0.0) {
//      throw 0;
//    }
    REAL itemp_4 = static_cast<REAL>(1.0) / temp_4;
    #pragma ivdep
    for(j=0; j<DIM; j++){
      int ind = DIM*i + j;
      a[ind] = a[ind] * itemp_4;

      I[ind] = I[ind] * itemp_4;

    }

    //making other elements 0 in order to make the matrix a an identity matrix and obtaining a inverse I matrix
    const int idim = i*DIM;
    for(q=0; q<DIM; q++){
      const int qdim = q*DIM;
      if(q==i)
        continue;
      temp_5 = a[qdim + i];
//      int j;
//      #pragma ivdep
//      for(j=0; j<ROUND_DOWN(DIM,UNROLL_STEP); j+=UNROLL_STEP){
//        const int indq = qdim + j;
//        const int indi = idim + j;
//        a[indq  ] = a[indq  ] - (temp_5*a[indi  ]);
//        a[indq+1] = a[indq+1] - (temp_5*a[indi+1]);
//        a[indq+2] = a[indq+2] - (temp_5*a[indi+2]);
//        a[indq+3] = a[indq+3] - (temp_5*a[indi+3]);
//
//        I[indq  ] = I[indq  ] - (temp_5*I[indi  ]);
//        I[indq+1] = I[indq+1] - (temp_5*I[indi+1]);
//        I[indq+2] = I[indq+2] - (temp_5*I[indi+2]);
//        I[indq+3] = I[indq+3] - (temp_5*I[indi+3]);
//      }
//      for(j=0; j<DIM; j++){
//        const int indq = qdim + j;
//        const int indi = idim + j;
//        a[qdim+j  ] = a[qdim+j  ] - (temp_5*a[indi  ]);
//        I[qdim+j  ] = I[qdim+j  ] - (temp_5*I[indi  ]);
//      }

      #pragma ivdep
      for(int j=0;j<DIM;j++) {
        const int indq = qdim + j;
        const int indi = idim + j;
        a[indq] = a[indq] - (temp_5*a[indi]);
        I[indq] = I[indq] - (temp_5*I[indi]);
      }
    }
  }
}

//computes A*M1 + B*M2 = Mans where A, B are scalars, M1, M2, Mans are matrices with dimensions dimX, dimY
template<typename REAL, int DIM, int UNROLL_STEP>
inline void gold_matAdd(const REAL*  M1, const REAL*  M2, REAL*  Mans, const REAL A, const REAL B) {
//  for(int i = 0; i<DIM; i++){
//    const int idim = i*DIM;
//    int j;
//    for(j=0; j<ROUND_DOWN(DIM,UNROLL_STEP); j+=UNROLL_STEP){
//      //if(UNROLL_STEP==2) {
//      Mans[idim+j  ] = A*M1[idim+j  ] + B*M2[idim+j  ];
//      Mans[idim+j+1] = A*M1[idim+j+1] + B*M2[idim+j+1];
//      Mans[idim+j+2] = A*M1[idim+j+2] + B*M2[idim+j+2];
//      Mans[idim+j+3] = A*M1[idim+j+3] + B*M2[idim+j+3];
//    }
//    for(; j<DIM; j++){
//      Mans[idim+j] = A*M1[idim+j] + B*M2[idim+j];
//    }
//  }

  int idx;
  for(int i = 0; i<DIM; i++){
    #pragma simd
    for(int j=0; j<DIM; j++){
      idx = i*DIM + j;
      Mans[idx] = A*M1[idx] + B*M2[idx];
    }
  }
}

//a*v1 + b*v2 = vAns, a,b scalars, v1, v2, vans vectors of length len
template<typename REAL, int DIM>
inline void gold_vecAdd(const REAL*  v1, const REAL*  v2, REAL*  vAns, const REAL a, const REAL b) {
  for(int i=0; i<DIM; i++) vAns[i] = a*v1[i] + b*v2[i];
}

// multiply square matrices M1, M2 of dimension dim, store in Mans
template<typename REAL, int DIM, int UNROLL_STEP>
inline void gold_MMmultiply(const REAL*  M1, const REAL*  M2, REAL*  Mans) {//, const int dim){
//  REAL temp;
//  for(int i=0; i<DIM; i++){
//    for(int j=0; j<DIM; j++){
//      temp = 0;
//      for(int k=0; k<DIM; k++) temp += M1[i*DIM+k]*M2[k*DIM+j];
//      Mans[i*DIM + j] = temp;
//    }
//  }

//  //const int step = UNROLL_STEP;//4;
//  // Initialize result block
//  for(int i=0; i<DIM; i++){
//    //for(int j=0; j<DIM; j++){
//    int idim = i*DIM;
//    #pragma ivdep
//    for(int j=0; j<DIM; j++) {
//      Mans[idim+j] = static_cast<REAL>(0.0);
//    }
//  }
//  // Multiply blocks
//  //#pragma unroll
//  for(int i=0; i<DIM; i++){
//    int idim = i*DIM;
//    for(int k=0; k<DIM; k++){
//      const REAL tmp = M1[i*DIM+k];
//      #pragma ivdep
//      for(int j=0; j<DIM; j++){
//        //int indij = i*DIM + j;
//        //Mans[indij] += M1[i*DIM+k] * M2[indij];
//        Mans[idim+j] += tmp * M2[idim+j];
//      }
//    }
//  }

  // Initialize result block
  for(int i=0; i<DIM; i++){
    //for(int j=0; j<DIM; j++){
    int idim = i*DIM;
    int j;
    for(j=0; j<ROUND_DOWN(DIM,UNROLL_STEP); j+=UNROLL_STEP){
      if(UNROLL_STEP==2) {
        Mans[idim+j  ] = static_cast<REAL>(0.0);
        Mans[idim+j+1] = static_cast<REAL>(0.0);
      }
      if(UNROLL_STEP==4) {
        Mans[idim+j  ] = static_cast<REAL>(0.0);
        Mans[idim+j+1] = static_cast<REAL>(0.0);
        Mans[idim+j+2] = static_cast<REAL>(0.0);
        Mans[idim+j+3] = static_cast<REAL>(0.0);
      }
    }
    for(; j<DIM; j++) {
      Mans[idim+j] = static_cast<REAL>(0.0);
    }
  }
  // Multiply blocks
  //#pragma unroll
  for(int i=0; i<DIM; i++){
    int idim = i*DIM;
    for(int k=0; k<DIM; k++){
      const REAL tmp = M1[i*DIM+k];
      int j;
      for(j=0; j<ROUND_DOWN(DIM,UNROLL_STEP); j+=UNROLL_STEP){
        //int indij = i*DIM + j;
        //Mans[indij] += M1[i*DIM+k] * M2[indij];
        if(UNROLL_STEP==2) {
          Mans[idim+j  ] += tmp * M2[idim+j  ];
          Mans[idim+j+1] += tmp * M2[idim+j+1];
        }
        if(UNROLL_STEP==4) {
          Mans[idim+j  ] += tmp * M2[idim+j  ];
          Mans[idim+j+1] += tmp * M2[idim+j+1];
          Mans[idim+j+2] += tmp * M2[idim+j+2];
          Mans[idim+j+3] += tmp * M2[idim+j+3];
        }
      }
      for(; j<DIM; j++){
        //int indij = i*DIM + j;
        //Mans[indij] += M1[i*DIM+k] * M2[indij];
        Mans[idim+j] += tmp * M2[idim+j];
      }
    }
  }
}

//multiply square matrix mat, of dimension dim, with the vector vec. store in vAns
template<typename REAL, int DIM, int UNROLL_STEP>
inline void gold_MVmultiply(const REAL*  mat,  const REAL*  vec, REAL*  vAns) { //, const int dim){
  REAL temp[DIM];
  for(int i=0; i<DIM; i++ ){
    // Create a element-by-element product
    int j;
    const int idim = i*DIM;
    for(j=0; j<ROUND_DOWN(DIM,UNROLL_STEP); j+=UNROLL_STEP){
      if(UNROLL_STEP==2) {
        temp[j  ] = mat[idim+j  ] * vec[j  ];
        temp[j+1] = mat[idim+j+1] * vec[j+1];
      }
      if(UNROLL_STEP==4) {
        temp[j  ] = mat[idim+j  ] * vec[j  ];
        temp[j+1] = mat[idim+j+1] * vec[j+1];
        temp[j+2] = mat[idim+j+2] * vec[j+2];
        temp[j+3] = mat[idim+j+3] * vec[j+3];
      }
    }
    for(; j<DIM; j++ ){
      temp[j] = mat[idim + j] * vec[j];
      //temp[j] = mat[i*DIM + j] * vec[j];
    }
    // Reduce product vector to get dot product
    vAns[i] = static_cast<REAL>(0.0);
    for(int j=0; j<DIM; j++ )
      vAns[i] += temp[j];
  }
//  REAL temp;
//  for(int i=0; i<DIM; i++ ){
//    temp = 0;
//    for(int j=0; j<DIM; j++ ){
//      temp += mat[i*DIM + j]*vec[j];
//    }
//    vAns[i] = temp;
//  }
}

template<typename REAL, int DIM>
inline void gold_matCopy(const REAL*  in, REAL*  out) {//, const int matrixSide){
  for(int i=0; i<DIM; i++){
    for(int j=0; j<DIM; j++){
      out[i*DIM + j] = in[i*DIM + j];
    }
  }
}

template<typename REAL, int DIM>
inline void gold_vecCopy(const REAL*  in, REAL*  out) {//, const int vectorSide){
  for(int i=0; i<DIM; i++){
    out[i] = in[i];
  }
}

//
// p - problem to be solved = 0..n_sys
// N - number of blocks in a system
// P - number of systems/problems
template<typename REAL, int DIM>
void gold_blkThomas_solve(const REAL*  A, const REAL*  B, const REAL*  C, REAL*  CAdj, REAL*  dAdj, const REAL*  d, REAL*  u, const int p, const int N, const int P) {
  int matOffset;
  int vecOffset;
  int vecOffsetPlusOne;
  int matOffsetMinusOne;
  int vecOffsetMinusOne;

  // FORWARD PASS

  const int ELEMS = DIM*DIM;

  REAL Binv[ELEMS], tempB[ELEMS], tempC[ELEMS], tempd[DIM];
  REAL tempX[ELEMS];
  REAL tempXinv[ELEMS];
  REAL tempX2[ELEMS];
  REAL tempw[DIM];
  REAL tempv[DIM];

  // Define C*[0] and d*[0]
//  matOffset = p*ELEMS;
//  vecOffset = p*DIM;
  matOffset = p*N*ELEMS;
  vecOffset = p*N*DIM;

  // Binv[0] <= B[0]^-1
  gold_matCopy<REAL,DIM>(B + matOffset, tempB);
  gold_invert<REAL,DIM,4>(tempB, Binv);

  // Cadj[0] <= Binv * C[0]
  // dAdj[0] <= Binv * d[0]
  gold_MMmultiply<REAL,DIM,4>(Binv, C + matOffset, tempC);
  gold_MVmultiply<REAL,DIM,4>(Binv, d + vecOffset, tempd);
  gold_matCopy<REAL,DIM>(tempC, CAdj + matOffset);
  gold_vecCopy<REAL,DIM>(tempd, dAdj + vecOffset);

  // Begin the forward sweep through the sub-block lines
  for(int n=1; n<N; n++){

//    matOffset = n*sys_n*ELEMS + p*ELEMS;
//    matOffsetMinusOne = (n-1)*sys_n*ELEMS + p*ELEMS;
//    vecOffset = n*sys_n*DIM + p*DIM;
//    vecOffsetMinusOne = (n-1)*sys_n*DIM + p*DIM;

    matOffset = p*N*ELEMS + n*ELEMS;
    matOffsetMinusOne = p*N*ELEMS + (n-1)*ELEMS;
    vecOffset = p*N*DIM + n*DIM;
    vecOffsetMinusOne = p*N*DIM + (n-1)*DIM;

    // A[i]*C[i-1] --> X
    gold_MMmultiply<REAL,DIM,4>(A + matOffset, CAdj + matOffsetMinusOne, tempX);

    // B[i] - X --> X
    gold_matAdd<REAL,DIM,4>(&B[matOffset], tempX, tempX, static_cast<REAL>(1.0), static_cast<REAL>(-1.0));

    // X^(-1) --> X
    gold_invert<REAL,DIM,4>(tempX, tempXinv);

    // X*C[i] --> C[i]
    gold_MMmultiply<REAL,DIM,4>(tempXinv, &C[matOffset], tempX2);
    gold_matCopy<REAL,DIM>(tempX2, &CAdj[matOffset]);

    // A[i]*d[i-1] --> v
    gold_MVmultiply<REAL,DIM,4>(&A[matOffset], &dAdj[vecOffsetMinusOne], tempv);

    // d[i] - v --> v
    gold_vecAdd<REAL,DIM>(&d[vecOffset], tempv, tempv, static_cast<REAL>(1.0), static_cast<REAL>(-1.0));

    // X*v --> d[i]
    gold_MVmultiply<REAL,DIM,4>(tempXinv, tempv, &dAdj[vecOffset]);
  }
  //  if(COMPARE_INTERIM){
  //    for(int i=0; i<DIM * N * P; ++i) dAdj[i] = d[i];
  //    for(int i=0; i<ELEMS * N * P; ++i) CAdj[i] = C[i];
  //  }
  //
  //  if(PRINT_INTERM){
  //    printf("C'(CPU):\n");
  //    printVec(C, ELEMS*N*P);
  //    printf("d'(CPU):\n");
  //    printVec(d, DIM*N*P);
  //  }

  // BACKWARD PASS

  // Define u[N-1]
  //matOffsetMinusOne = (sys_size-1)*sys_n*DIM + p*DIM;
  matOffsetMinusOne = p*N*DIM + (N-1)*DIM;
  for(int i=0; i<DIM; i++){
    u[matOffsetMinusOne + i] = dAdj[matOffsetMinusOne + i];
  }

  // Begin the backward sweep through the sub-block lines
  for(int n=N-2; n>=0; n--){
    //int tempwOffset = p*DIM;
//    matOffset = n*sys_n*ELEMS + p*ELEMS;
//    vecOffset = n*P*DIM + p*DIM;
//    vecOffsetPlusOne = (n+1)*P*DIM + p*DIM;
    matOffset = p*N*ELEMS + n*ELEMS;
    vecOffset = p*N*DIM + n*DIM;
    vecOffsetPlusOne = p*N*DIM + (n+1)*DIM;

    // C[i]*u[i+1] --> w
    gold_MVmultiply<REAL,DIM,4>(&CAdj[matOffset], &u[vecOffsetPlusOne], tempw);
    // d[i] - w --> u[i]
    gold_vecAdd<REAL,DIM>(&dAdj[vecOffset], tempw, &u[vecOffset], static_cast<REAL>(1.0), static_cast<REAL>(-1.0));
  }
}

template<typename REAL, int DIM>
void blkThomas_GOLD(REAL*  A, REAL*  B, REAL*  C, REAL*  CAdj, REAL*  dAdj, REAL*  d, REAL*  u, const int N, const int P){
  #pragma omp parallel for schedule(guided) //num_threads(12)
  for(int p=0; p<P; p++) {
    gold_blkThomas_solve<REAL,DIM>(A, B, C, CAdj, dAdj, d, u, p, N, P);
  }
}

// s - single
// d - double
// b - block
// t - tridiagonal
// sv- solver
void sbtsv_cpu(float*  A, float*  B, float*  C, float*  CAdj, float*  dAdj, float*  d, float*  u, const int N, const int P, const int blkdim) {
  switch(blkdim) {
    //case 1:
    //  blkThomas_GOLD<float,1>(A, B, C, CAdj, dAdj, d, u, N, P);
    //  break;
    case 2:
      blkThomas_GOLD<float,2>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 3:
      blkThomas_GOLD<float,3>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 4:
      blkThomas_GOLD<float,4>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 5:
      blkThomas_GOLD<float,5>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 6:
      blkThomas_GOLD<float,6>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 7:
      blkThomas_GOLD<float,7>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 8:
      blkThomas_GOLD<float,8>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 9:
      blkThomas_GOLD<float,9>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 10:
      blkThomas_GOLD<float,10>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    default:
      printf("Only BLK_DIM block dimension <= 10 is supported!");
      break;
  }
}

void dbtsv_cpu(double*  A, double*  B, double*  C, double*  CAdj, double*  dAdj, double*  d, double*  u, const int N, const int P, const int blkdim) {
  switch(blkdim) {
    //case 1:
    //  blkThomas_GOLD<double,1>(A, B, C, CAdj, dAdj, d, u, N, P);
    //  break;
    case 2:
      blkThomas_GOLD<double,2>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 3:
      blkThomas_GOLD<double,3>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 4:
      blkThomas_GOLD<double,4>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 5:
      blkThomas_GOLD<double,5>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 6:
      blkThomas_GOLD<double,6>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 7:
      blkThomas_GOLD<double,7>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 8:
      blkThomas_GOLD<double,8>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 9:
      blkThomas_GOLD<double,9>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    case 10:
      blkThomas_GOLD<double,10>(A, B, C, CAdj, dAdj, d, u, N, P);
      break;
    default:
      printf("Only BLK_DIM block dimension <= 10 is supported!");
      break;
  }
}
