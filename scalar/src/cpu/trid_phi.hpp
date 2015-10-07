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
 
#ifndef __TRID_PHI_HPP
#define __TRID_PHI_HPP

//#pragma offload_attribute(push,target(mic))

#include "trid_simd.h"
#include <assert.h>
#include "transpose.hpp"

__attribute__((target(mic)))
inline void load(SIMD_REG * __restrict__ dst, FP * __restrict__ src, int n, int pad) {
#ifdef __MIC__
  __assume_aligned(src,SIMD_WIDTH);
  __assume_aligned(dst,SIMD_WIDTH);
  //  *(SIMD_REG*)&(u[i*N]) = *(SIMD_REG*)&(a[i*N]);
  for(int i=0; i<SIMD_VEC; i++) {
    //assert( ((long long)&(src[i*pad+n]) % 64) == 0);
    dst[i] = *(SIMD_REG*)&(src[i*pad+n]);
  }
#endif
}

__attribute__((target(mic)))
inline void store(FP * __restrict__ dst, SIMD_REG * __restrict__ src, int n, int pad) {
#ifdef __MIC__
  __assume_aligned(src,SIMD_WIDTH);
  __assume_aligned(dst,SIMD_WIDTH);
  //  *(SIMD_REG*)&(u[i*N]) = *(SIMD_REG*)&(a[i*N]);
  for(int i=0; i<SIMD_VEC; i++) {
    //assert( ((long long)&(dst[i*pad+n]) % 64) == 0);
    *(SIMD_REG*)&(dst[i*pad+n]) = src[i];
  }
#endif
}

//
// tridiagonal-x solver
//
//__attribute__((vector(linear(a),linear(b),linear(c),linear(d),linear(u))))
__attribute__((target(mic)))
inline void trid_x_phi(FP* __restrict a, FP* __restrict b, FP* __restrict c, FP* __restrict d, FP* __restrict u, int N, int stride) {
//inline void trid_x_phi(FP* a, FP* b, FP* c, FP* d, FP* u, int N, int stride) {

  #ifdef __MIC__ // Or #ifdef __KNC__ - more general option, future proof, __INTEL_OFFLOAD is another option
  //SIMD_REG *a = (SIMD_REG*) h_a;
  //SIMD_REG *b = (SIMD_REG*) h_b;
  //SIMD_REG *c = (SIMD_REG*) h_c;
  //SIMD_REG *d = (SIMD_REG*) h_d;
  //SIMD_REG *u = (SIMD_REG*) h_u;
  __assume_aligned(a,SIMD_WIDTH);
  __assume_aligned(b,SIMD_WIDTH);
  __assume_aligned(c,SIMD_WIDTH);
  __assume_aligned(d,SIMD_WIDTH);

  //printf("trid_x_phi() sizeof(SIMD_REG)=%i\n", sizeof(SIMD_REG));

  //assert( (((long long)a)%64) == 0);

  int   i, ind = 0;
  SIMD_REG aa;  
  SIMD_REG bb;
  SIMD_REG cc;
  SIMD_REG dd;
  SIMD_REG a_reg[SIMD_VEC];  
  SIMD_REG b_reg[SIMD_VEC];
  SIMD_REG c_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  SIMD_REG c2[N_MAX];
  SIMD_REG d2[N_MAX];

  //
  // forward pass
  //
  int   n = 0;
  SIMD_REG ones = SIMD_SET1_P(1.0F);

  #if FPPREC == 0
  load(a_reg,a,n,N); transpose16x16_intrinsic(a_reg);
  load(b_reg,b,n,N); transpose16x16_intrinsic(b_reg);
  load(c_reg,c,n,N); transpose16x16_intrinsic(c_reg);
  load(d_reg,d,n,N); transpose16x16_intrinsic(d_reg);
  #elif FPPREC == 1
  load(a_reg,a,n,N); transpose8x8_intrinsic(a_reg);
  load(b_reg,b,n,N); transpose8x8_intrinsic(b_reg);
  load(c_reg,c,n,N); transpose8x8_intrinsic(c_reg);
  load(d_reg,d,n,N); transpose8x8_intrinsic(d_reg);
  #endif

  bb = b_reg[0];
  #if FPPREC == 0
    bb = SIMD_RCP_P(bb);
  #elif FPPREC == 1
    bb = SIMD_DIV_P(ones,bb);
  #endif
  cc = c_reg[0];
  cc = SIMD_MUL_P(bb,cc);
  dd = d_reg[0];
  dd = SIMD_MUL_P(bb,dd);
  c2[0] = cc;
  d2[0] = dd;
  
  //d_reg[0] = dd;

  for(i=1; i<SIMD_VEC; i++) {
    aa    = a_reg[i];
    #ifdef __MIC__
      bb    = SIMD_FNMADD_P(aa,cc,b_reg[i]);
      dd    = SIMD_FNMADD_P(aa,dd,d_reg[i]);
    #else
      bb    = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa,cc) );
      dd    = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa,dd) );
    #endif
    #if FPPREC == 0
      bb    = SIMD_RCP_P(bb);
    #elif FPPREC == 1
      bb    = SIMD_DIV_P(ones,bb);
    #endif
    cc    = SIMD_MUL_P(bb,c_reg[i]);
    dd    = SIMD_MUL_P(bb,dd);
    c2[n+i] = cc;
    d2[n+i] = dd;

    //d_reg[i] = dd;
  }
  //transpose16x16_intrinsic(d_reg); store(u,d_reg,n,N);

  for(n=SIMD_VEC; n<N; n+=SIMD_VEC) {
    #if FPPREC == 0
    load(a_reg,a,n,N); transpose16x16_intrinsic(a_reg);
    load(b_reg,b,n,N); transpose16x16_intrinsic(b_reg);
    load(c_reg,c,n,N); transpose16x16_intrinsic(c_reg);
    load(d_reg,d,n,N); transpose16x16_intrinsic(d_reg);
    #elif FPPREC == 1
    load(a_reg,a,n,N); transpose8x8_intrinsic(a_reg);
    load(b_reg,b,n,N); transpose8x8_intrinsic(b_reg);
    load(c_reg,c,n,N); transpose8x8_intrinsic(c_reg);
    load(d_reg,d,n,N); transpose8x8_intrinsic(d_reg);
    #endif
    for(i=0; i<SIMD_VEC; i++) {
      aa    = a_reg[i];
      #ifdef __MIC__
        bb    = SIMD_FNMADD_P(aa,cc,b_reg[i]);
        dd    = SIMD_FNMADD_P(aa,dd,d_reg[i]);
      #else
        bb    = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa,cc) );
        dd    = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa,dd) );
      #endif
      #if FPPREC == 0
        bb    = SIMD_RCP_P(bb);
      #elif FPPREC == 1
        bb    = SIMD_DIV_P(ones,bb);
      #endif
      cc    = SIMD_MUL_P(bb,c_reg[i]);
      dd    = SIMD_MUL_P(bb,dd);
      c2[n+i] = cc;
      d2[n+i] = dd;
      
      d_reg[i] = dd;
    }
    //transpose16x16_intrinsic(d_reg); store(u,d_reg,n,N);
    //load(a_reg,a,n,N);
    //transpose16x16_intrinsic(a_reg);
    //store(u,a_reg,n,N);
  }

  //
  // reverse pass
  //
  d_reg[SIMD_VEC-1] = dd;
  n -= SIMD_VEC;
        //printf("n = %d\n",n);
        //for(i=VEC-sys_off-2; i>=0; i--) {
  for(i=SIMD_VEC-2; i>=0; i--) {
          //for(i=sys_off-2; i>=0; i--) {
          //if(i==VEC-sys_off-1) l_d[i] = dd;
          //if(i<VEC-sys_off-1) {
    dd     = SIMD_SUB_P(d2[n+i], SIMD_MUL_P(c2[n+i],dd) );
          //  dd      = d2[n+i] - c2[n+i]*dd;
    d_reg[i] = dd;
            //l_d[i]  = n+i;
          //}
  }
  //transpose16x16_intrinsic(d_reg); store(u,d_reg,n,N);
  #if FPPREC == 0
  transpose16x16_intrinsic(d_reg); store(d,d_reg,n,N);
  #elif FPPREC == 1
  transpose8x8_intrinsic(d_reg); store(d,d_reg,n,N);
  #endif
        //store_array_shared<REAL>(d,l_d,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
        //store_array_shared<REAL>(u,l_d,n,shared,goffset,soffset,trow,tcol,sys_size);

        //for(n=sys_size-2*VEC; n>=0; n-=VEC) {
  for(n=n-SIMD_VEC; n>=0; n-=SIMD_VEC) {
    for(i=(SIMD_VEC-1); i>=0; i--) {
      dd     = SIMD_SUB_P(d2[n+i], SIMD_MUL_P(c2[n+i],dd) );
      //dd     = d2[n+i] - c2[n+i]*dd;
      d_reg[i] = dd;
            //l_d[i] = n+i;
    }
    //transpose16x16_intrinsic(d_reg); store(u,d_reg,n,N);
    #if FPPREC == 0
    transpose16x16_intrinsic(d_reg); store(d,d_reg,n,N);
    #elif FPPREC == 1
    transpose8x8_intrinsic(d_reg); store(d,d_reg,n,N);
    #endif
          //store_array_shared<REAL>(d,l_d,n,shared,goffset,soffset,trow,tcol,sys_size,sys_pads);
          //store_array_shared<REAL>(u,l_d,n,shared,goffset,soffset,trow,tcol,sys_size);
  }

//  d_reg[SIMD_VEC-1] = dd;
//  //u[ind] = dd;
//  for(n=N-2; n>=0; n--) {
//    ind    = ind - stride;
//    dd     = SIMD_SUB_P(d2[n], SIMD_MUL_P(c2[n],dd) );
//    d[ind] = dd;
//    //SIMD_STORE_P(&d[ind], dd);
//    //u[ind] = dd;
//  }

//  d[ind] = dd;
//
//  //u[ind] = dd;
//
//  for(i=N-2; i>=0; i--) {
//    ind    = ind - stride;
//    dd     = d2[i] - c2[i]*dd;
//    d[ind] = dd;
//
//    //u[ind] = dd;
//    
//  }
  #endif
}
//
// tridiagonal solver
//
template<typename REAL, typename VECTOR, int INC>
//template<typename REAL>
__attribute__((target(mic)))
inline void trid_scalar_vec(REAL* __restrict h_a, REAL* __restrict h_b, REAL* __restrict h_c, REAL* __restrict h_d, REAL* __restrict h_u, int N, int stride) {

//#ifdef __MIC__
// F32vec16 a;
//#endif
  int i, ind = 0;
  VECTOR aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];

  VECTOR* __restrict a = (VECTOR*) h_a;
  VECTOR* __restrict b = (VECTOR*) h_b;
  VECTOR* __restrict c = (VECTOR*) h_c;
  VECTOR* __restrict d = (VECTOR*) h_d;
  VECTOR* __restrict u = (VECTOR*) h_u;

//    b[0] = a[0];

  VECTOR ones(1.0f);

  //
  // forward pass
  //
  //bb    = 1.0f / b[0];
  bb    = ones / b[0];
  cc    = bb*c[0];
  dd    = bb*d[0];
  c2[0] = cc;
  d2[0] = dd;

  //u[0] = a[0];
  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = a[ind];
    bb    = b[ind] - aa*cc;
    dd    = d[ind] - aa*dd;
    //bb    = 1.0f/bb;
    bb    = ones / bb;
    cc    = bb*c[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;
//    u[ind] = a[ind];
  }
  //
  // reverse pass
  //
  if(INC) u[ind] += dd;
  else    d[ind]  = dd;
//  u[ind] = dd;
  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
    if(INC) u[ind] += dd;
    else    d[ind]  = dd;
//    u[ind] = ones;
  }
}

//
// tridiagonal solver
//
//__attribute__((vector(linear(a),linear(b),linear(c),linear(d),linear(u))))
__attribute__((target(mic)))
inline void trid_scalar(FP* __restrict a, FP* __restrict b, FP* __restrict c, FP* __restrict d, FP* __restrict u, int N, int stride) {
        
  //printf("trid_scalar()\n");

  int   i, ind = 0;
  FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = 1.0F/b[0];
  cc    = bb*c[0];
  dd    = bb*d[0];
  c2[0] = cc;
  d2[0] = dd;

  //u[0] = a[0];

  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = a[ind];
    bb    = b[ind] - aa*cc;
    dd    = d[ind] - aa*dd;
    bb    = 1.0F/bb;
    cc    = bb*c[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;

    //u[ind] = a[ind];

  }
  //
  // reverse pass
  //
  d[ind] = dd;

  //u[ind] = dd;

  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
    d[ind] = dd;

    //u[ind] = dd;
    
  }
}

////
//// tridiagonal solver
////
////__attribute__((target(mic)))
////__attribute__((vector))
////__attribute__((vector(linear(a),linear(b),linear(c),linear(d),linear(u))))
////inline void trid_cpu(FP* restrict a, FP* restrict b, FP* restrict c, FP* restrict d, FP* restrict u){//, int N, int stride) {
//////void trid_cpu(FP* __restrict a, FP* __restrict b, FP* __restrict c, FP* __restrict d, FP* __restrict u, int N, int stride) {
////  int   i, ind = 0;
////  FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
////  //
////  // forward pass
////  //
////  bb    = 1.0F/b[0];
////  cc    = bb*c[0];
////  dd    = bb*d[0];
////  c2[0] = cc;
////  d2[0] = dd;
////
////  //u[0] = a[0];
//////#pragma novector
////  for(i=1; i<256; i++) {
////    ind   = ind + 256;
////    aa    = a[ind];
////    bb    = b[ind] - aa*cc;
////    dd    = d[ind] - aa*dd;
////    bb    = 1.0F/bb;
////    cc    = bb*c[ind];
////    dd    = bb*dd;
////    c2[i] = cc;
////    d2[i] = dd;
////
//////    u[ind] = a[ind];
////
////  }
////  //
////  // reverse pass
////  //
////  d[ind] = dd;
//////  u[ind] = dd;
//////#pragma novector
////  for(i=256-2; i>=0; i--) {
////    ind    = ind - 256;
////    dd     = d2[i] - c2[i]*dd;
////    d[ind] = dd;
////
//////    u[ind] = dd;
////  }
////}
//
//////
////// tridiagonal-x solver
//////
//////#ifdef FPISA>1
////B
////inline void trid_gather(FP* __restrict ax, FP* __restrict bx, FP* __restrict cx, FP* __restrict du, FP* __restrict u, int N, int stride) {
////      int  n;
////
////      __declspec(align(SIMD_WIDTH)) SIMD_REG *a, *b, *c, *d;
////      a = (SIMD_REG*) ax;
////      b = (SIMD_REG*) bx;
////      c = (SIMD_REG*) cx;
////      d = (SIMD_REG*) du;
////      __declspec(align(SIMD_WIDTH)) SIMD_REG *ux = (SIMD_REG*) u;
////      //register FP aa, bb, cc, dd;
////      register SIMD_REG aa, bb, cc, dd, dd2;
////      __declspec(align(SIMD_WIDTH)) SIMD_REG c2[N_MAX], d2[N_MAX];
////
////      //__m512i ind = _mm512_set_epi32(15*NX,14*NX,13*NX,12*NX,11*NX,10*NX,9*NX,8*NX,7*NX,6*NX,5*NX,4*NX,3*NX,2*NX,1*NX,0*NX);
//////      #if FPISA==2
////      SIMD_REGI ind = SIMD_SET_EPI(15*N,14*N,13*N,12*N,11*N,10*N,9*N,8*N,7*N,6*N,5*N,4*N,3*N,2*N,1*N,0*N);
//////      #elif FPISA==3
//////        SIMD_REGI ind = SIMD_SET_EPI32(7*NX,6*NX,5*NX,4*NX,3*NX,2*NX,1*NX,0*NX);
//////      #endif
////      //__m512i ind = _mm512_set_epi32(0,1*NX,2*NX,3*NX,4*NX,5*NX,6*NX,7*NX,8*NX,9*NX,10*NX,11*NX,12*NX,13*NX,14*NX,15*NX);
////      //bb = _mm512_i32gather_ps(ind, (void*)b, FBYTE);//b[0];
////      bb = SIMD_I32GATHER_P(ind, (void*)b, FBYTE);//b[0];
////      //bb = SIMD_DIV_P(_mm512_set1_ps(1.0F),bb);
////      bb = SIMD_DIV_P(SIMD_SET1_P(1.0F),bb);
////      //cc = c[0];
////      //cc = _mm512_i32gather_ps(ind, (void*)c, FBYTE);
////      cc = SIMD_I32GATHER_P(ind, (void*)c, FBYTE);
////      cc = SIMD_MUL_P(bb,cc);
////      //dd = d[0];
////      //dd = _mm512_i32gather_ps(ind, (void*)d, FBYTE);
////      dd = SIMD_I32GATHER_P(ind, (void*)d, FBYTE);
////      dd = SIMD_MUL_P(bb,dd);
////      c2[0] = cc;
////      d2[0] = dd;
////
////      for(n=1; n<N; n++) {
////        ind   = SIMD_ADD_EPI(ind, SIMD_SET1_EPI(stride));
////        aa    = SIMD_I32GATHER_P(ind, (void*)a, FBYTE); //a[ind];
////        bb    = SIMD_I32GATHER_P(ind, (void*)b, FBYTE);
////        bb    = SIMD_SUB_P(bb, SIMD_MUL_P(aa,cc) );
////        dd2   = SIMD_I32GATHER_P(ind, (void*)d, FBYTE);
////        dd    = SIMD_SUB_P(dd2, SIMD_MUL_P(aa,dd) );
////        bb    = SIMD_DIV_P( SIMD_SET1_P(1.0F), bb);
////        cc    = SIMD_I32GATHER_P(ind, (void*)c, FBYTE);
////        cc    = SIMD_MUL_P(bb,cc);
////        dd    = SIMD_MUL_P(bb,dd);
////        c2[n] = cc;
////        d2[n] = dd;
////      }
////      //
////      // reverse pass
////      //
////      SIMD_I32SCATTER_P((void*)d, ind, dd, FBYTE);
//////      #pragma unroll(8)
////      for(n=N-2; n>=0; n--) {
////        ind    = SIMD_SUB_EPI(ind, SIMD_SET1_EPI(stride));
////        dd     = SIMD_SUB_P(d2[n], SIMD_MUL_P(c2[n],dd) );
////        SIMD_I32SCATTER_P((void*)d, ind, dd, FBYTE);
////      }
////}
//////#endif
//
////inline void load(float *ga, __m512 *la, const int gind, const int stride) {
////  //a = (SIMD_REG*) &ga[gind];
////  //__assume_aligned(a,SIMD_WIDTH);
////
////  for(int i=0; i<SIMD_VEC; i++) {
////    la[i] = *((SIMD_REG*)&ga[gind+i*stride]);
////  }
////}
//
//
////
//// tridiagonal-x solver
////
//__attribute__((target(mic)))
//inline void trid_x_cpu(FP* __restrict az, FP* __restrict bz, FP* __restrict cz, FP* __restrict du, FP* __restrict u, int N, int stride) {
//  int  n, ind=0;
//  __declspec(align(SIMD_WIDTH)) SIMD_REG *a, *b, *c, *d;
//  a = (SIMD_REG*) az;
//  b = (SIMD_REG*) bz;
//  c = (SIMD_REG*) cz;
//  d = (SIMD_REG*) du;
//
//  __assume_aligned(a,SIMD_WIDTH);
//  __assume_aligned(b,SIMD_WIDTH);
//  __assume_aligned(c,SIMD_WIDTH);
//  __assume_aligned(d,SIMD_WIDTH);
//  //__assume_aligned(u,SIMD_WIDTH);
//
//  register SIMD_REG aa, bb, cc, dd;
//  __declspec(align(SIMD_WIDTH)) SIMD_REG c2[N_MAX], d2[N_MAX];
//
//  SIMD_REG ones = SIMD_SET1_P(1.0F);
//  bb = SIMD_LOAD_P(&b[0]);
//  #if FPPREC == 0
//    bb = SIMD_RCP_P(bb);
//  #elif FPPREC == 1
//    bb = SIMD_DIV_P(ones,bb);
//  #endif
//  cc = SIMD_LOAD_P(&c[0]);
//  cc = SIMD_MUL_P(bb,cc);
//  dd = SIMD_LOAD_P(&d[0]);
//  dd = SIMD_MUL_P(bb,dd);
//  c2[0] = cc;
//  d2[0] = dd;
//
//  for(n=1; n<N; n++) {
//    ind   = ind + stride;
//    aa    = SIMD_LOAD_P(&a[ind]);
//    #ifdef __MIC__
//      bb    = SIMD_FNMADD_P(aa,cc,SIMD_LOAD_P(&b[ind]));
//      dd    = SIMD_FNMADD_P(aa,dd,SIMD_LOAD_P(&d[ind]));
//    #else
//      bb    = SIMD_SUB_P(SIMD_LOAD_P(&b[ind]), SIMD_MUL_P(aa,cc) );
//      dd    = SIMD_SUB_P(SIMD_LOAD_P(&d[ind]), SIMD_MUL_P(aa,dd) );
//    #endif
//    #if FPPREC == 0
//      bb    = SIMD_RCP_P(bb);
//    #elif FPPREC == 1
//      bb    = SIMD_DIV_P(ones,bb);
//    #endif
//    cc    = SIMD_MUL_P(bb,SIMD_LOAD_P(&c[ind]));
//    dd    = SIMD_MUL_P(bb,dd);
//    c2[n] = cc;
//    d2[n] = dd;
//  }
//  //
//  // reverse pass
//  //
//  d[ind] = dd;
//  //u[ind] = dd;
//  for(n=N-2; n>=0; n--) {
//    ind    = ind - stride;
//    dd     = SIMD_SUB_P(SIMD_LOAD_P(&d2[n]), SIMD_MUL_P(SIMD_LOAD_P(&c2[n]),dd) );
//    d[ind] = dd;
//    //SIMD_STORE_P(&d[ind], dd);
//    //u[ind] = dd;
//  }
//}

//
// tridiagonal-y solver
//
//__attribute__((target(mic)))
inline void trid_y_cpu(FP* __restrict az, FP* __restrict bz, FP* __restrict cz, FP* __restrict du, FP* __restrict u, int N, int stride) {
//  #pragma offload_attribute(push,target(mic))
//  #ifdef __MIC__ // Or #ifdef __KNC__ - more general option, future proof, __INTEL_OFFLOAD is another option
  int  n, ind=0;
  __declspec(align(SIMD_WIDTH)) SIMD_REG *a, *b, *c, *d;
  a = (SIMD_REG*) az;
  b = (SIMD_REG*) bz;
  c = (SIMD_REG*) cz;
  d = (SIMD_REG*) du;

  __assume_aligned(a,SIMD_WIDTH);
  __assume_aligned(b,SIMD_WIDTH);
  __assume_aligned(c,SIMD_WIDTH);
  __assume_aligned(d,SIMD_WIDTH);
  //__assume_aligned(u,SIMD_WIDTH);

  register SIMD_REG aa, bb, cc, dd;
  __declspec(align(SIMD_WIDTH)) SIMD_REG c2[N_MAX], d2[N_MAX];

  SIMD_REG ones = SIMD_SET1_P(1.0F);
  bb = b[0];
  #if FPPREC == 0
    bb = SIMD_RCP_P(bb);
  #elif FPPREC == 1
    bb = SIMD_DIV_P(ones,bb);
  #endif
  cc = c[0];
  cc = SIMD_MUL_P(bb,cc);
  dd = d[0];
  dd = SIMD_MUL_P(bb,dd);
  c2[0] = cc;
  d2[0] = dd;

  for(n=1; n<N; n++) {
    ind   = ind + stride;
    aa    = a[ind];
    #ifdef __MIC__
      bb    = SIMD_FNMADD_P(aa,cc,b[ind]);
      dd    = SIMD_FNMADD_P(aa,dd,d[ind]);
    #else
      bb    = SIMD_SUB_P(b[ind], SIMD_MUL_P(aa,cc) );
      dd    = SIMD_SUB_P(d[ind], SIMD_MUL_P(aa,dd) );
    #endif
    #if FPPREC == 0
      bb    = SIMD_RCP_P(bb);
    #elif FPPREC == 1
      bb    = SIMD_DIV_P(ones,bb);
    #endif
    cc    = SIMD_MUL_P(bb,c[ind]);
    dd    = SIMD_MUL_P(bb,dd);
    c2[n] = cc;
    d2[n] = dd;
  }
  //
  // reverse pass
  //
  d[ind] = dd;
  //u[ind] = dd;
  for(n=N-2; n>=0; n--) {
    ind    = ind - stride;
    dd     = SIMD_SUB_P(d2[n], SIMD_MUL_P(c2[n],dd) );
    d[ind] = dd;
    //SIMD_STORE_P(&d[ind], dd);
    //u[ind] = dd;
  }
//  #endif
//  #pragma offload_attribute(pop)
}

//
// tridiagonal-z solver
//
//__attribute__((target(mic)))
inline void trid_z_cpu(FP* __restrict az, FP* __restrict bz, FP* __restrict cz, FP* __restrict du, FP* __restrict u, int N, int stride) {
  int  n, ind=0;
  __declspec(align(SIMD_WIDTH)) SIMD_REG *a, *b, *c, *d, *u2;
  a  = (SIMD_REG*) az;
  b  = (SIMD_REG*) bz;
  c  = (SIMD_REG*) cz;
  d  = (SIMD_REG*) du;
  u2 = (SIMD_REG*)  u;

  __assume_aligned(a,SIMD_WIDTH);
  __assume_aligned(b,SIMD_WIDTH);
  __assume_aligned(c,SIMD_WIDTH);
  __assume_aligned(d,SIMD_WIDTH);
  __assume_aligned(u2,SIMD_WIDTH);

  register SIMD_REG aa, bb, cc, dd;
  __declspec(align(SIMD_WIDTH)) SIMD_REG c2[N_MAX], d2[N_MAX];

  SIMD_REG ones = SIMD_SET1_P(1.0F);
  bb = b[0];
  #if FPPREC == 0
    bb    = SIMD_RCP_P(bb);
  #elif FPPREC == 1
    bb    = SIMD_DIV_P(ones,bb);
  #endif
  cc = c[0];
  cc = SIMD_MUL_P(bb,cc);
  dd = d[0];
  dd = SIMD_MUL_P(bb,dd);
  c2[0] = cc;
  d2[0] = dd;

  for(n=1; n<N; n++) {
    ind   = ind + stride;
    aa    = a[ind];
    #ifdef __MIC__
      bb    = SIMD_FNMADD_P(aa,cc,SIMD_LOAD_P(&b[ind]));
      dd    = SIMD_FNMADD_P(aa,dd,SIMD_LOAD_P(&d[ind]));
    #else
      bb    = SIMD_SUB_P(SIMD_LOAD_P((FP*)&b[ind]), SIMD_MUL_P(aa,cc) );
      dd    = SIMD_SUB_P(SIMD_LOAD_P((FP*)&d[ind]), SIMD_MUL_P(aa,dd) );
    #endif
    //bb    = SIMD_SUB_P(b[ind], SIMD_MUL_P(aa,cc) );
    //dd    = SIMD_SUB_P(d[ind], SIMD_MUL_P(aa,dd) );
    #if FPPREC == 0
      bb    = SIMD_RCP_P(bb);
    #elif FPPREC == 1
      bb    = SIMD_DIV_P(ones,bb);
    #endif
    cc    = SIMD_MUL_P(bb,c[ind]);
    dd    = SIMD_MUL_P(bb,dd);
    c2[n] = cc;
    d2[n] = dd;
  }
  //
  // reverse pass
  //
  u2[ind] = SIMD_ADD_P(u2[ind], dd);
  for(n=N-2; n>=0; n--) {
    ind    = ind - stride;
    //dd     = SIMD_SUB_P(d2[n], SIMD_MUL_P(c2[n],dd) );
    #ifdef __MIC__
      dd    = SIMD_FNMADD_P(c2[n],dd,SIMD_LOAD_P(&d2[n]));
      SIMD_PACKSTORELO_P(&u2[ind],SIMD_ADD_P(u2[ind], dd));
    #else
      dd    = SIMD_SUB_P(d2[n], SIMD_MUL_P(c2[n],dd) );
      SIMD_STORE_P((FP*)&u2[ind], SIMD_ADD_P(u2[ind], dd));
    #endif

    //u2[ind] = SIMD_ADD_P(u2[ind], dd);
  }
}

//#pragma offload_attribute(pop)
#endif
