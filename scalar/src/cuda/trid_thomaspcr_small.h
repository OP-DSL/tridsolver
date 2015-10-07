/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Jeremy Appleyard and others. Please see the AUTHORS file in
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
 *     * The name of Jeremy Appleyard may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Jeremy Appleyard ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Jeremy Appleyard BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//////////////////////////////////////////////////////////////////////
//
// This header file has various tridiagonal solution routines
// written by Mike Giles, Sept/Oct 2013, and updated Jan 2014
// with contributions from Jeremy Appleyard and Julien Demouth
//
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//
// 64bit shuffle commands, based on Julien Demouth's GTC 2013 talk:
// on-demand.gputechconf.com/gtc/2013/
//                  presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf

/*
__forceinline__ __device__ double __shfl_up(double x, int offset) {
  return __hiloint2double( __shfl_up(__double2hiint(x), offset),
                           __shfl_up(__double2loint(x), offset) );
}

__forceinline__ __device__ double __shfl_down(double x, int offset) {
  return __hiloint2double( __shfl_down(__double2hiint(x), offset),
                           __shfl_down(__double2loint(x), offset) );
}
*/

// __forceinline__ __device__ double __shfl_up(double x, uint s) {
 // int lo, hi;
 // asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
 // lo = __shfl_up(lo,s);
 // hi = __shfl_up(hi,s);
 // asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
 // return x;
// }

// __forceinline__ __device__ double __shfl_down(double x, uint s) {
 // int lo, hi;
 // asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
 // lo = __shfl_down(lo,s);
 // hi = __shfl_down(hi,s);
 // asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
 // return x;
// }

// __forceinline__ __device__ double __shfl_xor(double x, int s) {
 // int lo, hi;
 // asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
 // lo = __shfl_xor(lo,s);
 // hi = __shfl_xor(hi,s);
 // asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
 // return x;
// }


//////////////////////////////////////////////////////////////////////////
// define reciprocals
//////////////////////////////////////////////////////////////////////////

static __forceinline__ __device__ float __rcp(float a) {
  return 1.0f / a;
}

static __forceinline__ __device__ double __rcp(double a) {

#if __CUDA_ARCH__ >= 300
  double e, y; 
  asm ("rcp.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(a)); 
  e = __fma_rn (-a, y, 1.0); 
  e = __fma_rn ( e, e,   e); 
  y = __fma_rn ( e, y,   y); 
  return y; 
#else
  return 1.0 / a;
#endif
}


//////////////////////////////////////////////////////////////////////////
//
// This function solves a separate tridiagonal system for each warp.
// It requires blockDim.x to be a multiple of 32.
//
// The tridiagonal system is of the form Ax=d where
//
//     ( 1.0  c[0]                            )
//     ( a[1] 1.0  c[1]                       )
//     (      a[2] 1.0  c[2]                  )
// A = (            .    .    .               )
//     (                 .    .   .           )
//     (                                      )
//     (                            a[31] 1.0 )
//
// Here a[n] means register a on thread/lane n within the warp
//
// It is essential for the user to set a[0] = c[31] = 0
//
// Note this works only on Kepler because of the shuffle instructions.
// A Fermi version would need to exchange data via shared memory.
//

template <typename REAL>
__forceinline__ __device__ 
REAL trid1_warp(REAL a, REAL c, REAL d){

  REAL b;
  uint s=1;

#pragma unroll
  for (int n=0; n<5; n++) {
    b = __rcp( 1.0f - a*__shfl_up(c,s)
                    - c*__shfl_down(a,s) );
    d = ( d  - a*__shfl_up(d,s)
             - c*__shfl_down(d,s) ) * b; 
    a =      - a*__shfl_up(a,s)     * b;
    c =      - c*__shfl_down(c,s)   * b; 

    s = s<<1;
  }

  return d;
}


// an alternative version using a volatile shared memory
// array of size at least equal to the blocksize

template <typename REAL>
__forceinline__ __device__ 
REAL trid1_warp_shared(REAL a, REAL c, REAL d,
                       volatile REAL *shared){

  REAL b, a0;
  uint s=1;
  int  t=threadIdx.x, tm, tp;

#pragma unroll
  for (int n=0; n<5; n++) {
    b  = 1.0;
    a0 = a;
    //    tm = 32*(t/32) + (32+t-s)%32;
    //    tp = 32*(t/32) + (32+t+s)%32;
    tm = (t&~31) + ((t-s)&31);
    tp = (t&~31) + ((t+s)&31);

    shared[t] = d;
    d = d - a*shared[tm];
    d = d - c*shared[tp];
    shared[t] = a;
    b = b - c*shared[tp];
    a =   - a*shared[tm];
    shared[t] = c;
    b = b -a0*shared[tm];
    c =   - c*shared[tp];

    b = __rcp(b);
    d = d * b;
    a = a * b;
    c = c * b; 

    s = s<<1;
  }

  return d;
}



//
// This function solves a separate tridiagonal system for each warp,
// with 2 rows per thread.
//
// The tridiagonal system is of the form Ax=d where
//
//     (  1.0  cm[0]                              )      ( dm[0]  )
//     ( ap[1]  1.0  cp[1]                        )      ( dp[0]  )
//     (       am[2]  1.0  cm[2]                  )      ( dm[1]  )
// A = (               .     .    .               ), d = (        )
//     (                     .    .    .          )      (   .    )
//     (                                          )      (   .    )
//     (                               ap[31] 1.0 )      ( dp[31] )
//
// It is essential for the user to set am[0] = cp[31] = 0
//
// Note this works only on Kepler because of the shuffle instructions.
// A Fermi version would need to exchange data via shared memory.
//
// This code uses optimisations due to Jeremy Appleyard (NVIDIA)

template <typename REAL>
__forceinline__ __device__ 
void trid2_warp(REAL &am, REAL &cm, REAL &dm,
                REAL &ap, REAL &cp, REAL &dp){

  REAL b;
  uint s=1;

  b   = 1.0f - ap*cm - cp*__shfl_down(am,1);
  b   = __rcp(b);
  // for some odd reason, the next line is best there not higher
  dp  =  dp  - ap*dm - cp*__shfl_down(dm,1); 
  dp  =   b*dp;
  ap  = - b*ap*am;
  cp  = - b*cp*__shfl_down(cm,1); 

#pragma unroll
  for (int n=0; n<5; n++) {
    dp  =   dp - ap*__shfl_up(dp,s) - cp*__shfl_down(dp,s);
    b   = 1.0f - ap*__shfl_up(cp,s) - cp*__shfl_down(ap,s);
    b   = __rcp(b);
    dp  =   b*dp;
    ap  = - b*ap*__shfl_up(ap,s);
    cp  = - b*cp*__shfl_down(cp,s);

    s = s<<1;
  }
  
  dm = dm - am*__shfl_up(dp,1) - cm*dp;
  
  return;
}

template <typename REAL, int tridSolveSize, int blockSize>
__forceinline__ __device__ 
void trid2_warp_large(REAL &am, REAL &cm, REAL &dm,
                      REAL &ap, REAL &cp, REAL &dp,
                      volatile REAL *shared){

  REAL b, a0;
  uint s=1;
  int  t=threadIdx.x, tm, tp;
  
  int numCycles;
  
  switch (tridSolveSize) {
    case 64: numCycles = 6; break;
    case 128: numCycles = 7; break;
    case 256: numCycles = 8; break;
    case 512: numCycles = 9; break;
    case 1024: numCycles = 10; break;
  }
  
  int numDecoupledSystems = 1 << (numCycles - 5);

  if (numCycles > 5) {
    tp = (t+1)%blockSize;

    shared[t] = dm;
    shared[blockSize + t] = am;
    shared[2* blockSize + t] = cm;
    __syncthreads();
    
    dp = dp   - ap*dm - cp*shared[tp];
    b  = 1.0f - ap*cm - cp*shared[blockSize + tp];
    b  = __rcp(b); 
    dp =   dp*b;
    ap = - ap*am*b;
    cp = - cp*shared[2 * blockSize + tp]*b;   
    
    __syncthreads();
    
#pragma unroll
    for (int n=0; n < numCycles - 4; n++) {
      b  = 1.0f;
      a0 = ap;
      tm = (blockSize + t-s)%blockSize;
      tp = (            t+s)%blockSize;
      // tm = (t&~31) + ((t-s)&31);
      // tp = (t&~31) + ((t+s)&31);

      shared[t] = dp;
      shared[blockSize + t] = ap;
      shared[2 * blockSize + t] = cp;
      
      __syncthreads();
      
      dp = dp - ap*shared[tm];
      dp = dp - cp*shared[tp];
      
      b  = b - cp*shared[blockSize + tp];
      ap =   - ap*shared[blockSize + tm];
      
      b  = b - a0*shared[2 * blockSize + tm];
      cp =   - cp*shared[2 * blockSize + tp];

      __syncthreads();
      
      b  = __rcp(b);
      dp = dp*b;
      ap = ap*b;
      cp = cp*b; 

      s = s<<1;
    }

    // We have decoupled into N systems, but need to rearrange through smem for warp-wide PCR.
    shared[t] = dp;
    shared[blockSize + t] = ap;
    shared[2 * blockSize + t] = cp;
    
    __syncthreads();
    
    int packedThread = (t / numDecoupledSystems) + (t % numDecoupledSystems) * (blockSize / numDecoupledSystems);
    
    dp = shared[packedThread];
    ap = shared[blockSize + packedThread];
    cp = shared[2 * blockSize + packedThread];
    
    __syncthreads();
  }
  else {
    b   = 1.0f - ap*cm - cp*__shfl_down(am,1);
    b   = __rcp(b);
    // for some odd reason, the next line is best there not higher
    dp  =  dp  - ap*dm - cp*__shfl_down(dm,1); 
    dp  =   b*dp;
    ap  = - b*ap*am;
    cp  = - b*cp*__shfl_down(cm,1); 
  }
 

  s = 1;
#pragma unroll
  for (int n=0; n < 5; n++) {
    dp  =   dp - ap*__shfl_up(dp,s) - cp*__shfl_down(dp,s);
    b   = 1.0f - ap*__shfl_up(cp,s) - cp*__shfl_down(ap,s);
    b   = __rcp(b);
    dp  =   b*dp;
    ap  = - b*ap*__shfl_up(ap,s);
    cp  = - b*cp*__shfl_down(cp,s);    

    s = s<<1;
  }
  

  if (numCycles > 5) {
    int packedThread = (t / numDecoupledSystems) + (t % numDecoupledSystems) * (blockSize / numDecoupledSystems);

    shared[packedThread] = dp;
    shared[blockSize + packedThread] = ap;
    shared[2 * blockSize + packedThread] = cp;
    
    __syncthreads();
    
    dp = shared[t];
    ap = shared[blockSize + t];
    cp = shared[2 * blockSize + t];
    
    __syncthreads();  
  
    //shared[t] = dp;
    //__syncthreads();
    tm = (blockSize + t-1)%blockSize;
    dm = dm - am*shared[tm] - cm*dp;

  }
  else {
    dm = dm - am*__shfl_up(dp,1) - cm*dp;
  }
  
  return;
}


// an alternative version using a volatile shared memory
// array of size at least equal to the blocksize

template <typename REAL>
__forceinline__ __device__ 
void trid2_warp_s(REAL &am, REAL &cm, REAL &dm,
                  REAL &ap, REAL &cp, REAL &dp,
                  volatile REAL *shared){

  REAL b, a0;
  uint s=1;
  int  t=threadIdx.x, tm, tp;

  tp = (t&~31) + ((t+1)&31);

  shared[t] = dm;
  dp = dp   - ap*dm - cp*shared[tp];
  shared[t] = am;
  b  = 1.0f - ap*cm - cp*shared[tp];
  b  = __rcp(b); 
  dp =   dp*b;
  ap = - ap*am*b;
  shared[t] = cm;
  cp = - cp*shared[tp]*b; 

#pragma unroll
  for (int n=0; n<5; n++) {
    b  = 1.0f;
    a0 = ap;
    //    tm = 32*(t/32) + (32+t-s)%32;
    //    tp = 32*(t/32) + (32+t+s)%32;
    tm = (t&~31) + ((t-s)&31);
    tp = (t&~31) + ((t+s)&31);

    shared[t] = dp;
    dp = dp - ap*shared[tm];
    dp = dp - cp*shared[tp];
    shared[t] = ap;
    b  = b - cp*shared[tp];
    ap =   - ap*shared[tm];
    shared[t] = cp;
    b  = b - a0*shared[tm];
    cp =   - cp*shared[tp];

    b  = __rcp(b);
    dp = dp*b;
    ap = ap*b;
    cp = cp*b; 

    s = s<<1;
  }

  shared[t] = dp;
  tm = (t&~31) + ((t-1)&31);
  dm = dm - am*shared[tm] - cm*dp;

  return;
}

//
// this function solves tridiagonal systems spread across the
// the threads of each warp, with SIZE elements per thread
//

template <int SIZE, typename REAL>
__forceinline__ __device__ 
void trid_warp(REAL *a, REAL *b, REAL *c, REAL *d){

  REAL bbi;

  for (int i=0; i<2; i++) {
    bbi  = __rcp(b[i]);
    d[i] = bbi * d[i];
    a[i] = bbi * a[i];
    c[i] = bbi * c[i];
  }

  for (int i=2; i<SIZE; i++) {
    bbi  =   __rcp( b[i] - a[i]*c [i-1] );
    d[i] =  bbi * ( d[i] - a[i]*d [i-1] );
    a[i] =  bbi * (      - a[i]*a [i-1] );
    c[i] =  bbi *   c[i];
  }

  for (int i=SIZE-3; i>0; i--) {
    d[i] =  d[i] - c[i]*d[i+1];
    a[i] =  a[i] - c[i]*a[i+1];
    c[i] =       - c[i]*c[i+1];
  }

  bbi  = __rcp( 1.0f - c[0]*a[1] );
  d[0] =  bbi * ( d[0] - c[0]*d[1] );
  a[0] =  bbi *   a[0];
  c[0] =  bbi * (      - c[0]*c[1] );

  trid2_warp(a[0],c[0],d[0],a[SIZE-1],c[SIZE-1],d[SIZE-1]);

  for (int i=1; i<SIZE-1; i++) {
    d[i] = d[i] - a[i]*d[0] - c[i]*d[SIZE-1];
  }
}


//
// this pair of functions is based on Julien Demouth's implementation
// for cases in which the same tridiagonal matrix is used repeatedly
//
// In a cleaned-up version of these routines, the additional arrays
// and variables (atmp, ap, bb, cp, bbi, bb0, a0, c0) could all be
// packed into a single array to simplify the API

template <int SIZE, typename REAL>
__forceinline__ __device__ 
void trid_warp_setup(REAL *a,  REAL *b,  REAL *c, REAL *atmp,
                     REAL *ap, REAL *bb, REAL *cp,
                     REAL &bbi, REAL &bb0, REAL &a0, REAL &c0){
  uint s=1;

  for(int i=0; i<2; i++) {
    b[i] = __rcp(b[i]);
    a[i] = a[i]*b[i];
    c[i] = c[i]*b[i];
    atmp[i] = a[i];
  }

  for(int i=2; i<8; i++) {
    b[i] = __rcp(b[i] - a[i]*c[i-1]);
    a[i] = a[i]*b[i];
    c[i] = c[i]*b[i];
    atmp[i] = -a[i]*atmp[i-1];
  }

  REAL c1 = c[6];
  for(int i=5; i>0 ; i--) {
    atmp[i] = atmp[i] - c[i]*atmp[i+1];
    c1      =         - c[i]*c1;
  }

  bbi = __rcp(1.0f - c[0]*atmp[1]);
  a0  =  bbi * atmp[0];
  c0  = -bbi * c[0]*c1;

  c[0] *= bbi;

  bb0 = __rcp(1.0f - atmp[7]*c0 - c[7]*__shfl_down(a0, 1));
  ap[0] = - atmp[7]*a0 * bb0;
  cp[0] = - c[7]*__shfl_down(c0, 1) * bb0;  

  #pragma unroll
  for(int i=0; i<4 ; i++) {
    bb[i]   = __rcp((1.0f - ap[i]*__shfl_up(cp[i], s)
                          - cp[i]*__shfl_down(ap[i], s)));
    ap[i+1] = -ap[i]*__shfl_up  (ap[i], s) * bb[i];
    cp[i+1] = -cp[i]*__shfl_down(cp[i], s) * bb[i]; 
    s = s<<1;
  }
  bb[4] = __rcp((1.0f - ap[4]*__shfl_up(cp[4], 16)
                      - cp[4]*__shfl_down(ap[4], 16)));
}


template <int SIZE, typename REAL>
__forceinline__ __device__ 
void trid_warp_solve(REAL *a,  REAL *b,  REAL *c, REAL *atmp,
                     REAL *ap, REAL *bb, REAL *cp, REAL *f,
                     REAL &bbi, REAL &bb0, REAL &a0, REAL &c0){
  uint s=1;

  f[0] = b[0]*f[0];
  f[1] = b[1]*f[1];
  f[2] = b[2]*f[2] - a[2]*f[1];
  f[3] = b[3]*f[3] - a[3]*f[2];
  f[4] = b[4]*f[4] - a[4]*f[3];
  f[5] = b[5]*f[5] - a[5]*f[4];
  f[6] = b[6]*f[6] - a[6]*f[5];
  f[7] = b[7]*f[7] - a[7]*f[6];
    
  f[5] =      f[5] - c[5]*f[6];
  f[4] =      f[4] - c[4]*f[5];
  f[3] =      f[3] - c[3]*f[4];
  f[2] =      f[2] - c[2]*f[3];
  f[1] =      f[1] - c[1]*f[2];
  f[0] =  bbi*f[0] - c[0]*f[1];

  // The core of the tridiagonal solver

  f[7] = (f[7] - atmp[7]*f[0] - c[7]*__shfl_down(f[0], 1)) * bb0; 

  #pragma unroll
  for(int i=0; i<5; i++) {
    f[7] = (f[7] - ap[i]*__shfl_up(f[7], s)
                 - cp[i]*__shfl_down(f[7], s)) * bb[i]; 
    s = s<<1;
  }
  f[0] =  f[0] - a0*__shfl_up(f[7], 1) - c0*f[7];
    
  REAL ci = c[6];
  f[6] = f[6] - atmp[6]*f[0] - ci*f[7];
  ci = -c[5]*ci;
  f[5] = f[5] - atmp[5]*f[0] - ci*f[7];
  ci = -c[4]*ci;
  f[4] = f[4] - atmp[4]*f[0] - ci*f[7];
  ci = -c[3]*ci;
  f[3] = f[3] - atmp[3]*f[0] - ci*f[7];
  ci = -c[2]*ci;
  f[2] = f[2] - atmp[2]*f[0] - ci*f[7];
  ci = -c[1]*ci;
  f[1] = f[1] - atmp[1]*f[0] - ci*f[7];
}
