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

// Written by Gabor Daniel Balogh, Pazmany Peter Catholic University,
// balogh.gabor.daniel@itk.ppke.hu, 2021
// Based on old trid_linear_reg_float16 and trid_linear_reg_double8 by
// Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014

#ifndef TRID_LINEAR_REG_HPP__
#define TRID_LINEAR_REG_HPP__

#include "trid_linear_reg_common.hpp"
#include "cutil_inline.h"

// All threads iterate along dimension X by step size 16
// 32 thread (a warp) that is originally mapped to dimension Y now loads data
// along dimension X
// Warps within a block wrap into a 4x8(double) or 4x16(float) tile where
// indices are (y,x)=(threadIdx.x/8,threadIdx.y%8)
// Such a 4x8 tile iterates 8 times along the dimension Y to read the data for
// the 32 rows.
template <typename REAL>
__global__ void trid_linear_reg(const REAL *a, const REAL *b, const REAL *c,
                                REAL *d, REAL *u, int sys_size, int sys_pads,
                                int sys_n) {
  int i;
  vec_line_t<REAL> l_a, l_b, l_c, l_d;
  REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];

  //
  // set up indices for main block
  //
  const int tid =
      threadIdx.x + threadIdx.y * blockDim.x +
      blockIdx.x * blockDim.y * blockDim.x +
      blockIdx.y * gridDim.x * blockDim.y *
          blockDim
              .x; // Thread ID in global scope - every thread solves one system,
  const int wid =
      tid /
      WARP_SIZE; // Warp ID in global scope - the ID wich the thread belongs to
  // const int woffset = wid * WARP_SIZE * nx; // Global memory offset: unique
  // to a warp; every thread in a warp calculates the same woffset, which is the
  // "begining" of 3D tile
  const int woffset =
      wid * WARP_SIZE * sys_pads; // Global memory offset: unique to a warp;
                                  // every thread in a warp calculates the same
                                  // woffset, which is the "begining" of 3D tile
  // const int stride  = nx;                   // Stride between systems
  //      int ind     = tid * stride;         // First index a system being
  //      solved by thread tid.
  int ind = tid * sys_pads; // First index a system being solved by thread tid.
  int n   = 0;              // Iterator for every 16th element along dimension X
  // const int optimized_solve = ((tid/4)*4+4 <= ny*nz);                // These
  // 4-threads do the regular memory read/write and data transpose
  const int optimized_solve =
      ((tid / 4) * 4 + 4 <= sys_n); // These 4-threads do the regular memory
                                    // read/write and data transpose
  // const int boundary_solve  = !optimized_solve && ( tid < (ny*nz) ); // Among
  // these 4-threads are some that have to be deactivated from global memory
  // read/write
  const int boundary_solve =
      !optimized_solve &&
      (tid < (sys_n)); // Among these 4-threads are some that have to be
                       // deactivated from global memory read/write
  const int active_thread =
      optimized_solve ||
      boundary_solve; // A thread is active only if it works on valid memory
  // const int aligned         = !(nx % align<REAL>);
  const int aligned = !(sys_pads % align<REAL>);

  //
  // forward pass
  //

  if (active_thread) {
    if (optimized_solve) {
      if (aligned) { // If the data is aligned do an aligned, fully optimized
                     // solve
        // The 0th element is treated differently, so do the first VECtor
        // separately
        load_array_reg(a, &l_a, n, woffset, sys_size);
        load_array_reg(b, &l_b, n, woffset, sys_size);
        load_array_reg(c, &l_c, n, woffset, sys_size);
        load_array_reg(d, &l_d, n, woffset, sys_size);
        bb    = 1.0f / l_b.f[0];
        cc    = bb * l_c.f[0];
        dd    = bb * l_d.f[0];
        c2[0] = cc;
        d2[0] = dd;

#pragma unroll
        for (i = 1; i < vec_length<REAL>; i++) {
          aa    = l_a.f[i];
          bb    = l_b.f[i] - aa * cc;
          dd    = l_d.f[i] - aa * dd;
          bb    = 1.0f / bb;
          cc    = bb * l_c.f[i];
          dd    = bb * dd;
          c2[i] = cc;
          d2[i] = dd;
        }
        // Process the rest of the VECtors
        for (n = vec_length<REAL>; n < sys_size; n += vec_length<REAL>) {
          load_array_reg(a, &l_a, n, woffset, sys_size);
          load_array_reg(b, &l_b, n, woffset, sys_size);
          load_array_reg(c, &l_c, n, woffset, sys_size);
          load_array_reg(d, &l_d, n, woffset, sys_size);
#pragma unroll
          for (i = 0; i < vec_length<REAL>; i++) {
            aa        = l_a.f[i];
            bb        = l_b.f[i] - aa * cc;
            dd        = l_d.f[i] - aa * dd;
            bb        = 1.0f / bb;
            cc        = bb * l_c.f[i];
            dd        = bb * dd;
            c2[n + i] = cc;
            d2[n + i] = dd;
          }
        }
        //
        // reverse pass
        //
        // The last element is treated differently, so do the the last VECtor
        // separately
        l_d.f[vec_length<REAL> - 1] = dd;
        n                           = sys_size - vec_length<REAL>;
        for (i = vec_length<REAL> - 2; i >= 0; i--) {
          dd       = d2[n + i] - c2[n + i] * dd;
          l_d.f[i] = dd;
        }
        store_array_reg(d, &l_d, n, woffset, sys_size);

        // Process the rest of the VECtors
        for (n = sys_size - 2 * vec_length<REAL>; n >= 0;
             n -= vec_length<REAL>) {
          for (i = vec_length<REAL> - 1; i >= 0; i--) {
            dd       = d2[n + i] - c2[n + i] * dd;
            l_d.f[i] = dd;
          }
          store_array_reg(d, &l_d, n, woffset, sys_size);
        }
      } else { // Else if the data is not aligned do a naive solve on the first
               // and last elements
        //
        // forward pass
        //
        int aligned;
        // int bound     = stride*ny*nz;
        int ind_floor = (ind / align<REAL>)*align<REAL>;
        int sys_off =
            ind -
            ind_floor; // Offset index from ind_floor, 0 for aligned systems

        for (i = 0; i < align<REAL>; i++) {
          if (i < sys_off) {
            c2[i] = 0.0;
            d2[i] = 0.0;
          }
          if (i == sys_off) {
            // bb    = 1.0f/__ldg( &b[ind] );
            // cc    = bb*__ldg( &c[ind] );
            // dd    = bb*__ldg( &d[ind] );
            bb    = 1.0f / b[ind];
            cc    = bb * c[ind];
            dd    = bb * d[ind];
            c2[i] = cc;
            d2[i] = dd;
            // u[ind_floor+i] =
            // dd;//a[ind_floor+i];//i;//dd;//a[ind];//aa;//i;//dd;//i;//dd;
          }
          if (i > sys_off) {
            aa    = __ldg(&a[ind_floor + i]);
            bb    = __ldg(&b[ind_floor + i]) - aa * cc;
            dd    = __ldg(&d[ind_floor + i]) - aa * dd;
            bb    = 1.0f / bb;
            cc    = bb * __ldg(&c[ind_floor + i]);
            dd    = bb * dd;
            c2[i] = cc;
            d2[i] = dd; // i;//dd;
            // u[ind_floor+i] = dd;//aa;//i;//dd;//aa;//i;//dd;//i;//dd;
          }
        }
        // Process the rest of the VECtors
        // for(aligned=align<REAL>; aligned<nx-2*align<REAL>;
        // aligned+=vec_length<REAL>) { for(aligned=align<REAL>;
        // aligned<nx-align<REAL>; aligned+=vec_length<REAL>) {
        for (aligned = align<REAL>; aligned < sys_size - vec_length<REAL>;
             aligned += vec_length<REAL>) {
          // aligned=align<REAL>;
          //{
          load_array_reg_unaligned(a, &l_a, aligned, tid, sys_size, sys_size,
                                   0);
          load_array_reg_unaligned(b, &l_b, aligned, tid, sys_size, sys_size,
                                   0);
          load_array_reg_unaligned(c, &l_c, aligned, tid, sys_size, sys_size,
                                   0);
          load_array_reg_unaligned(d, &l_d, aligned, tid, sys_size, sys_size,
                                   0);
#pragma unroll
          for (i = 0; i < vec_length<REAL>; i++) {
            aa              = l_a.f[i];
            bb              = l_b.f[i] - aa * cc;
            dd              = l_d.f[i] - aa * dd;
            bb              = 1.0f / bb;
            cc              = bb * l_c.f[i];
            dd              = bb * dd;
            c2[aligned + i] = cc;
            d2[aligned + i] = dd;
            // u[ind_floor+aligned+i] =
            // dd;//aa;//ind_floor+aligned+i;//aa;//ind_floor+aligned+i;
            // dd;//aa;//aligned+i;//dd;//aligned+i;//i;//dd;
          }
        }
        n                = aligned;
        int last_aligned = aligned;
        // int j;
        for (; n + pack_len<REAL> - 1 <
               ((sys_size + sys_off) / pack_len<REAL>)*pack_len<REAL>;
             n += pack_len<REAL>) {
          // l_a.vec[0] = *(vec_t<REAL>*) &a[ind_floor+n];
          // l_b.vec[0] = *(vec_t<REAL>*) &b[ind_floor+n];
          // l_c.vec[0] = *(vec_t<REAL>*) &c[ind_floor+n];
          // l_d.vec[0] = *(vec_t<REAL>*) &d[ind_floor+n];
          l_a.vec[0] = __ldg((vec_t<REAL> *)&a[ind_floor + n]);
          l_b.vec[0] = __ldg((vec_t<REAL> *)&b[ind_floor + n]);
          l_c.vec[0] = __ldg((vec_t<REAL> *)&c[ind_floor + n]);
          l_d.vec[0] = __ldg((vec_t<REAL> *)&d[ind_floor + n]);
          for (i = 0; i < pack_len<REAL>; i++) {
            aa       = l_a.f[i];
            bb       = l_b.f[i] - aa * cc;
            dd       = l_d.f[i] - aa * dd;
            bb       = 1.0f / bb;
            cc       = bb * l_c.f[i];
            dd       = bb * dd;
            l_c.f[i] = cc;
            l_d.f[i] = dd;
            // u[ind_floor+n+i] = dd;
          }
          *((vec_t<REAL> *)&c2[n]) = l_c.vec[0];
          *((vec_t<REAL> *)&d2[n]) = l_d.vec[0];
          //*((vec_t<REAL>*)& u[ind_floor+n]) =
          //(vec_t<REAL>){ind_floor+n, ind_floor+n+1,
          //              ind_floor+n+2, ind_floor+n+3};
          // l_d.vec[0];
          //*((vec_t<REAL>*)& u[ind_floor+n]) =
          // l_d.vec[0];//l_a.vec[0];//(vec_t<REAL>){n, n+1, n+2,
          // n+3};//l_d.vec[0];
        }

        if (n == ((sys_size + sys_off) / pack_len<REAL>)*pack_len<REAL>) {
          i = n;
        } else {
          i = n - pack_len<REAL>;
        }

        for (; i < sys_size + sys_off; i++) {
          aa    = a[ind_floor + i];
          bb    = b[ind_floor + i] - aa * cc;
          dd    = d[ind_floor + i] - aa * dd;
          bb    = 1.0f / bb;
          cc    = bb * c[ind_floor + i];
          dd    = bb * dd;
          c2[i] = cc;
          d2[i] = dd;
          // aa    = __ldg( &a[ind_floor+i] );
          // bb    = __ldg( &b[ind_floor+i] ) - aa*cc;
          // dd    = __ldg( &d[ind_floor+i] ) - aa*dd;
          // bb    = 1.0f/bb;
          // cc    = bb*__ldg( &c[ind_floor+i] );
          // dd    = bb*dd;
          // c2[i] = cc;
          // d2[i] = dd;
          // u[ind_floor+i] = dd;
        }
        //
        // reverse pass
        //

        // d[ind+nx-1] = dd;
        // u[ind+nx-1] = nx-1;//dd;//nx-1;//dd;
        for (i = sys_size + sys_off - 1; i >= last_aligned; i--) {
          dd               = d2[i] - c2[i] * dd;
          d[ind_floor + i] = dd;
          // u[ind_floor+i] = dd;//i-sys_off;//dd;//d2[sys_off+i];//dd;
        }
        // for(n=align<REAL>; n<nx-2*align<REAL>; n+=vec_length<REAL>) {
        // for(aligned=nx-vec_length<REAL>-1; aligned>=align<REAL>; aligned--) {
        //  dd       = d2[aligned] - c2[aligned]*dd;
        //  d[ind_floor+aligned] = dd;
        //  //u[ind_floor+aligned] = dd;//n;//dd;//d2[sys_off+i];//dd;
        //  //u[ind_floor+aligned] = dd;//aligned;//n;//dd;//d2[sys_off+i];//dd;
        //}
        for (aligned = last_aligned - vec_length<REAL>; aligned >= align<REAL>;
             aligned -= vec_length<REAL>) {
          for (i = vec_length<REAL> - 1; i >= 0; i--) {
            dd       = d2[aligned + i] - c2[aligned + i] * dd;
            l_d.f[i] = dd; // aligned+i;//dd;//aligned+i;//dd;
            // u[ind_floor+aligned+i] =
            // dd;//i-sys_off;//dd;//d2[sys_off+i];//dd;
          }
          // store_array_reg_unaligned(u,&l_d,aligned, tid, nx, nx, 0);
          store_array_reg_unaligned(d, &l_d, aligned, tid, sys_size, sys_size,
                                    0);
        }
        for (i = align<REAL> - 1; i >= sys_off; i--) {
          // for(i=align<REAL>-1; i>=0; i--) {
          dd               = d2[i] - c2[i] * dd;
          d[ind_floor + i] = dd;
          // u[ind_floor+i] = dd;//n;//dd;//d2[sys_off+i];//dd;
        }
      }
      // Else if the system is a boundary one, use naive algorithm to solve it
    } else {
      //
      // forward pass
      //
      // bb    = 1.0f/__ldg( &b[ind] );
      // cc    = bb*__ldg( &c[ind] );
      // dd    = bb*__ldg( &d[ind] );
      bb    = 1.0f / b[ind];
      cc    = bb * c[ind];
      dd    = bb * d[ind];
      c2[0] = cc;
      d2[0] = dd;

      // u[ind] = 0;//dd;

      for (i = 1; i < sys_size; i++) {
        ind   = ind + 1;
        aa    = __ldg(&a[ind]);
        bb    = __ldg(&b[ind]) - aa * cc;
        dd    = __ldg(&d[ind]) - aa * dd;
        bb    = 1.0f / bb;
        cc    = bb * __ldg(&c[ind]);
        dd    = bb * dd;
        c2[i] = cc;
        d2[i] = dd;

        // u[ind] = ind;//dd;
      }
      //
      // reverse pass
      //
      d[ind] = dd;
      // u[ind] = dd;
      for (i = sys_size - 2; i >= 0; i--) {
        ind    = ind - 1;
        dd     = d2[i] - c2[i] * dd;
        d[ind] = dd;
        // u[ind] = dd;
      }
    }
  }
}

template <typename REAL>
void trid_linear_reg(dim3 dimGrid_x, dim3 dimBlock_x, const REAL *d_ax,
                     const REAL *d_bx, const REAL *d_cx, REAL *d_du, REAL *d_u,
                     int sys_size, int sys_pads, int sys_n) {
  trid_linear_reg<<<dimGrid_x, dimBlock_x>>>(d_ax, d_bx, d_cx, d_du, d_u,
                                             sys_size, sys_pads, sys_n);
  cudaCheckMsg("trid_linear_reg execution failed\n");
}


#endif /* ifndef TRID_LINEAR_REG_HPP__ */

