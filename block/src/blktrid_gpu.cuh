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

// Written by Endre Laszlo, James Whittle and Catherine Hastings, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

template<typename REAL, int DIM>
__device__ inline void loadblk(REAL *regBlk, const REAL * d_array, const int offset) {
  //load B(0), C(0) and d(0) into registers
  for(int i=0; i<DIM; i++) {
    regBlk[i] = d_array[i*DIM + offset];
  }
}

//template<typename REAL, int PROBS_PER_TBLK, int DIM>
//__device__ inline void loadblk(REAL *regBlk, const REAL * d_array, const int localpid, const int offset) {
//  //load B(0), C(0) and d(0) into registers
//  REAL __shared__ tmp[PROBS_PER_TBLK*DIM*DIM];
//  for(int i=0; i<DIM; i++) {
//    tmp[localpid] = d_array[i*DIM + threadIdx.x];
//  }
//  for(int i=0; i<DIM; i++) {
//    regBlk[i] = d_array[i*DIM + offset];
//  }
//}

template<typename REAL, int DIM>
__device__ inline void storeblk(const REAL *regBlk, REAL *d_array, const int offset) {
  for(int i=0; i<DIM; i++) {
    d_array[i*DIM + offset] = regBlk[i];
  }
}

template<typename REAL, int DIM>
__device__ inline void MVmult(REAL* sharedArray, const REAL *regBlk, REAL *d, const int offset, const int colid) {
  // C'(n) * u(n+1) -> tmp_vec
  sharedArray[offset + colid] = static_cast<REAL>(0.0);
  //__syncthreads(); // no need for sync, because every thread stores 0.0
  for (int j=0; j<DIM; j++){
    // Strided access is less efficient due to the lack of register allocation
    // sharedArray[poffset*DIM + (i+colid)%DIM] += regBlk1[(i+colid)%DIM] * d;
    if(colid==j) {
      for(int i=0;i<DIM; i++)
        sharedArray[offset + i] += regBlk[i] * (*d);
    }
    __syncthreads();
  }
}

template<typename REAL, int DIM>
__device__ inline void MVmult_shfl(REAL* vec, const REAL *regBlk, REAL *d, const int offset, const int colid) {
  // C'(n) * u(n+1) -> tmp_vec
  REAL tmp;
  REAL tmp2;
  for(int i=0; i<DIM; i++) {
    //sharedArray[offset + i] += regBlk[i] * (*d);
    tmp = regBlk[i] * (*d);
    for (int j=1; j<DIM; j++){
      tmp2 = __shfl(tmp,offset+((colid+j)%DIM));
      //tmp2 = __shfl(tmp,((threadIdx.x%32)/DIM)*DIM+(colid+j)%DIM);
      if(colid==i) tmp += tmp2;
    }
    if(colid==i) *vec = tmp;
  }
}

template<typename REAL, int DIM>
//__device__ inline void MVmult_mod_shfl(REAL* vec, const REAL *regBlk, REAL *d, const int offset, const int colid) {
__device__ inline void MVmult_mod_shfl(REAL* vec, REAL *regBlk, REAL *d, const int offset, const int colid) {
  // C'(n) * u(n+1) -> tmp_vec
  REAL tmp = static_cast<REAL>(0.0);
  REAL tmp2; //= static_cast<REAL>(0.0);
  *vec = static_cast<REAL>(0.0);
    // Strided access is less efficient due to the lack of register allocation
    // sharedArray[poffset*DIM + (i+colid)%DIM] += regBlk1[(i+colid)%DIM] * d;
    //if(colid==j) {
  for(int i=0; i<DIM; i++) {
    //sharedArray[offset + i] += regBlk[i] * (*d);
    regBlk[i] *= (*d);
  }

  // Transpose matrix to have rows of a block in a thread rather than columns
  for (int j=0; j<DIM; j++){
    for (int i=0; i<DIM; i++){
      tmp = __shfl(regBlk[j],offset+(j+i)%DIM);
      if(colid==j) *vec += tmp;
        //*vec += __shfl(regBlk[j],offset+(colid+i)%DIM);
    //tmp2 = __shfl(regBlk[i],offset+i);
    //if(colid==i) tmp += tmp2;
    }
  }
  //if(colid==i) *sharedArray = tmp;
}

template<typename REAL, int DIM>
__device__ inline void MMmult(REAL *regBlk3, REAL *regBlk2, REAL *regBlk1, REAL *s_temp, int colid, int poffset) {
  REAL pivot;
  // Perform Matrix-Matrix multiplication
  // Initialize the output matrix
  for(int j=0; j<DIM; ++j) regBlk3[j] = 0.0f;
  // perform the multiplication
  for(int i=0; i<DIM; ++i){
    for(int j=0; j<DIM; ++j){
      if(colid==i) s_temp[poffset] = regBlk1[j];
      __syncthreads();
      pivot = s_temp[poffset];
      regBlk3[j] += pivot*regBlk2[i];
    }
  }
}

template<typename REAL, int DIM>
__device__ inline void MMmult_shfl(REAL *regBlk3, REAL *regBlk2, REAL *regBlk1, int offset) {
  REAL pivot;
  // Perform Matrix-Matrix multiplication
  // Initialize the output matrix
  for(int j=0; j<DIM; ++j) regBlk3[j] = 0.0f;
  // perform the multiplication
  for(int i=0; i<DIM; ++i){
    for(int j=0; j<DIM; ++j){
      //if(colid==i) s_temp[poffset] = regBlk1[j];
      pivot = __shfl(regBlk1[j],offset+i); //if(colid==i) s_temp[poffset] = regBlk1[j];
      //__syncthreads();
      //pivot = s_temp[poffset];
      regBlk3[j] += pivot*regBlk2[i];
    }
  }
}

template<typename REAL, int DIM>
__device__ void gaussjordan(REAL *regBlk1, REAL *regBlk2, REAL *d, REAL *s_temp, int colid, int poffset) {
  REAL pivot, di;
  for(int i=0; i<DIM; i++){
    // load a single element of B(0) into shared memory
    // this is the single element B(0)_{i,i} and this is the `pivot'
    if(colid == i) {
      s_temp[poffset] = regBlk1[i]; // This value is shared among DIM threads
      //if(regBlk1[i] == 0) printf("\nPivot for i=%d is 0.\n", i);
    }
    __syncthreads();

    // give the pivot to all threads
    pivot = 1.0f / s_temp[poffset];

    // multiply the whole row by the pivot
    // this ensures that the leading entry of the pivot row is now 1
    regBlk1[i] *= pivot; // Mul = 2 * DIM
    regBlk2[i] *= pivot; // 

    // multiply the ith element of d by the pivot
    // then store the new ith elemtent of d into shared
    // and pass this value to all threads
    if (colid == i) {
      (*d) *= pivot;  // Mul = PROBS_PER_WARP
      s_temp[poffset] = *d;
    }
    __syncthreads();
    di = s_temp[poffset];
   
    for(int j=0; j< DIM; j++){
      // Now load the B(0)_{j,i}th element into shared
      // This is the leading value of each row, in the same column as the pivot
      if (colid == i) s_temp[poffset] = regBlk1[j];
      __syncthreads();
      if(j != i){
        // Give the element in shared to each thread.
        // We will reuse the variable-name pivot, to save register space
        // THIS VALUE IS NOT THE PIVOT
        pivot = s_temp[poffset];

        // Now subtract a multiple of the row containing the pivot off each row
        // The purpose of this is to ensure that all the values above and below the pivot are 0
        regBlk1[j] -= pivot*regBlk1[i];
        regBlk2[j] -= pivot*regBlk2[i];
        if (colid == j) (*d) -= pivot*di;
      }
    }
  }
}

template<typename REAL, int DIM>
__device__ void gaussjordan_shfl(REAL *regBlk1, REAL *regBlk2, REAL *d, int colid, int offset) {
  REAL pivot, di;
  for(int i=0; i<DIM; i++){
    // load a single element of B(0) into shared memory
    // this is the single element B(0)_{i,i} and this is the `pivot'
    //if(colid == i) {
    //  s_temp[poffset] = regBlk1[i]; // This value is shared among DIM threads
    //  //if(regBlk1[i] == 0) printf("\nPivot for i=%d is 0.\n", i);
    //}
    //__syncthreads();
    pivot = __shfl(regBlk1[i],offset+i); //if(colid==i) s_temp[poffset] = regBlk1[j];

    // give the pivot to all threads
    //pivot = 1.0f / s_temp[poffset];
    pivot = 1.0f / pivot;

    // multiply the whole row by the pivot
    // this ensures that the leading entry of the pivot row is now 1
    regBlk1[i] *= pivot; // Mul = 2 * DIM
    regBlk2[i] *= pivot; //

    // multiply the ith element of d by the pivot
    // then store the new ith elemtent of d into shared
    // and pass this value to all threads
    if (colid == i) {
      (*d) *= pivot;  // Mul = PROBS_PER_WARP
      //s_temp[poffset] = *d;
    }
    //__syncthreads();
    //di = s_temp[poffset];
    di = __shfl(*d,offset+i);

    for(int j=0; j< DIM; j++){
      // Now load the B(0)_{j,i}th element into shared
      // This is the leading value of each row, in the same column as the pivot
      //if (colid == i) s_temp[poffset] = regBlk1[j];
      //__syncthreads();
      REAL tmp = __shfl(regBlk1[j],offset+i);
      if(j != i){
        // Give the element in shared to each thread.
        // We will reuse the variable-name pivot, to save register space
        // THIS VALUE IS NOT THE PIVOT
        //pivot = s_temp[poffset];
        pivot = tmp;

        // Now subtract a multiple of the row containing the pivot off each row
        // The purpose of this is to ensure that all the values above and below the pivot are 0
        regBlk1[j] -= pivot*regBlk1[i];
        regBlk2[j] -= pivot*regBlk2[i];
        if (colid == j) (*d) -= pivot*di;
      }
    }
  }
}

template<typename REAL, int DIM, int SHFL, int WARP_SIZE, int THREADBLOCK>
__global__ void
__launch_bounds__(128,4)
blk_thomas_gpu(const REAL * __restrict__ d_A,
               const REAL * __restrict__ d_B,
               const REAL * __restrict__ d_C,
               const REAL * __restrict__ d_d,
                     REAL * __restrict__ d_Cstar,
                     REAL * __restrict__ d_dstar,
                     REAL * __restrict__ d_u,
               const int               N,
               const int               P) {

  const int ELS = DIM*DIM;
  const int MAT_ELEMS = ((ELS) * (P));
  const int VEC_ELEMS = ((DIM) * (P));

  const int PROBS_PER_WARP = (WARP_SIZE/DIM);
  const int PROBS_PER_TBLK = ( (1+((THREADBLOCK-1)/WARP_SIZE)) * PROBS_PER_WARP );

  const int tid        = threadIdx.x + blockIdx.x * blockDim.x;
  const int warpid     = tid / WARP_SIZE;                    // Global Warp ID
  //const int laneid   = tid % WARP_SIZE;
  const int laneid     = threadIdx.x % WARP_SIZE;            // Lane ID within a thread warp
  const int localpid   = laneid / DIM;                       // Local Problem ID within a warp
  const int pid        = warpid * PROBS_PER_WARP + localpid; // Global Problem ID

  const int colid      = laneid % DIM;
  //const int colid      = threadIdx.x % DIM;              // Column index of a thread within a block

  const int poffset    = pid - blockIdx.x * PROBS_PER_TBLK;  // Problem number within a thread block 
  const int mat_offset = pid * ELS;
  const int rhs_offset = pid * DIM;

  // Often used index components precalculated
  const int poffsetTIMESDIM = poffset * DIM;
  const int colidPLUSmat_offset = colid + mat_offset;
  const int rhs_offsetPLUScolid = rhs_offset + colid;

  //one column of each matrix or one element of each vector
  REAL regBlk1[DIM], regBlk2[DIM], regBlk3[DIM];

  REAL __shared__ s_temp[PROBS_PER_TBLK]; // Every problem in a thread block needs one shared variable for intra matrix block communication
  REAL __shared__ sharedArray[PROBS_PER_TBLK*DIM]; // In mat-vec product the acummulating vector is shared
  //volatile REAL __shared__ s_temp[PROBS_PER_TBLK], sharedArray[PROBS_PER_TBLK*DIM];

  //register REAL pivot, d, tmp_vec, di;
  register REAL d, tmp_vec;

  if (localpid >= PROBS_PER_WARP || pid >= P) return; // Deactivate threads that don't have job to done

  //////////////////////////////////////////////////////////////////////////////////
  //                                FORWARD PASS                                  //
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////
  // sub-matrix row 0 //
  //////////////////////

  //load B(0), C(0) and d(0) into registers
  loadblk<REAL,DIM>(regBlk1, d_B, colidPLUSmat_offset);
  loadblk<REAL,DIM>(regBlk2, d_C, colidPLUSmat_offset);
  d = d_d[rhs_offsetPLUScolid];

  // perform B(0)*C'(0) = C(0) using Gauss-Jordan [so we output C'(0)]
  // and simultaneously perform B(0)*d'(0) = d(0) using Gauss-Jordan [so we output d'(0)]
  //-------------------------------------------------------------------------------------------
  if(SHFL==0) gaussjordan<REAL,DIM>(regBlk1, regBlk2, &d, s_temp, colid, poffset);
  else        gaussjordan_shfl<REAL,DIM>(regBlk1, regBlk2, &d, colid, localpid*DIM);
//  gaussjordan(regBlk1, regBlk2, &d, s_temp, colid, poffset);

  // now store back C[0] and d[0] to global memory
  storeblk<REAL,DIM>(regBlk2, d_Cstar, colid + mat_offset);
  d_dstar[rhs_offsetPLUScolid] = d;
  //-------------------------------------------------------------------------------------------
  // end of Gauss-Jordan

  //////////////////////////////
  // sub-matrix rows 1 to N-1 //
  //////////////////////////////
  for (int n = 1; n < N; ++n) {
    // PREPARE [ B(n) - A(n)*C'(n-1) ]
    //--------------------------------------
    // load A(n) and C'(n-1) into registers
    loadblk<REAL,DIM>(regBlk1,     d_A, colidPLUSmat_offset +  n    * MAT_ELEMS);
    loadblk<REAL,DIM>(regBlk2, d_Cstar, colidPLUSmat_offset + (n-1) * MAT_ELEMS);

    // A(n) * C'(n-1) -> tempM
    if(SHFL==0) MMmult<REAL,DIM>(regBlk3, regBlk2, regBlk1, s_temp, colid, poffset);
    else        MMmult_shfl<REAL,DIM>(regBlk3, regBlk2, regBlk1, localpid*DIM);

    // load B(n) into registers
    // and perform B(n) - tempM -> tempM
    loadblk<REAL,DIM>(regBlk2, d_B, colidPLUSmat_offset + n*MAT_ELEMS);
    for (int i=0; i<DIM; i++){
      //regBlk2[i] = d_B[i*DIM + colidPLUSmat_offset + n*MAT_ELEMS];
      regBlk3[i] = regBlk2[i] - regBlk3[i];
    }
    //--------------------------------------

    // PREPARE [ d(n) - A(n)*d'(n-1) ]
    //--------------------------------------
    // load d'(n-1) into registers
    d = d_dstar[rhs_offsetPLUScolid + (n-1)*VEC_ELEMS];

    //// A(n) * d'(n-1) -> tmp_vec
    if(SHFL==0) {
      MVmult<REAL,DIM>(sharedArray, regBlk1, &d, poffsetTIMESDIM, colid);
      tmp_vec = sharedArray[poffsetTIMESDIM + colid];
    } else {
      MVmult_shfl<REAL,DIM>(&tmp_vec, regBlk1, &d, localpid*DIM, colid);
    }

    // d(n) - tmp_vec -> tmp_vec
    tmp_vec = d_d[rhs_offsetPLUScolid + n*VEC_ELEMS] - tmp_vec;
    //--------------------------------------
    
    // load C(n) into registers
    loadblk<REAL,DIM>(regBlk2, d_C, colidPLUSmat_offset + n*MAT_ELEMS);
    
    // perform regBlk3*C'(n) = C(n) using Gauss-Jordan [so we output C'(n)]
    // and simultaneously perform regBlk3*d'(n) = tmp_vec using Gauss-Jordan [so we output d'(n)]
    // here regBlk3  = [ B(n) - A(n)*C'(n-1) ]
    // and tmp_vec  = [ d(n) - A(n)*d'(n-1) ]
    //-------------------------------------------------------------------------------------------
    if(SHFL==0) gaussjordan<REAL,DIM>(regBlk3, regBlk2, &tmp_vec, s_temp, colid, poffset);
    else        gaussjordan_shfl<REAL,DIM>(regBlk3, regBlk2, &tmp_vec, colid, localpid*DIM);
//    gaussjordan(regBlk3, regBlk2, &tmp_vec, s_temp, colid, poffset);

    // store back to global memory
    d_dstar[rhs_offsetPLUScolid + n*VEC_ELEMS] = tmp_vec;
    storeblk<REAL,DIM>(regBlk2, d_Cstar, colidPLUSmat_offset + n*MAT_ELEMS);
    //-------------------------------------------------------------------------------------------
    // end of Gauss-Jordan
  }

  //////////////////////////////////////////////////////////////////////////////////
  //                                 BACKWARD PASS                                //
  //////////////////////////////////////////////////////////////////////////////////
  ////////////////////////
  // sub-matrix row N-1 //
  ////////////////////////

  // d'(N-1) -> u(N-1)
  d_u[rhs_offsetPLUScolid + (N-1) * VEC_ELEMS] = d_dstar[rhs_offsetPLUScolid + (N-1)*VEC_ELEMS];

  //////////////////////////////
  // sub-matrix rows N-2 to 0 //
  //////////////////////////////
  // d'(n) -> u(n)
  for (int n = N - 2; n >= 0; --n) {
    // load C'(n) and u(n+1) into registers
    loadblk<REAL,DIM>(regBlk1, d_Cstar, colidPLUSmat_offset + n*MAT_ELEMS);
    d = d_u[rhs_offsetPLUScolid + (n+1)*VEC_ELEMS];
    
    // C'(n) * u(n+1) -> tmp_vec
    if(SHFL==0) {
      MVmult<REAL,DIM>(sharedArray, regBlk1, &d, poffsetTIMESDIM, colid);
      tmp_vec = sharedArray[poffsetTIMESDIM + colid];
    }
    else {
      MVmult_shfl<REAL,DIM>(&tmp_vec, regBlk1, &d, localpid*DIM, colid);
    }

    // d'(n)-tmp_vec -> u(n)
    d_u[rhs_offsetPLUScolid + n*VEC_ELEMS] = d_dstar[rhs_offsetPLUScolid + n*VEC_ELEMS] - tmp_vec;
  }
}
