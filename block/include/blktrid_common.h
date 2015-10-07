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

#ifndef __BLKTRID_COMMON_H
#define __BLKTRID_COMMON_H

////////////////////////////////////////////////////////////////////////
// Define problem parameters
////////////////////////////////////////////////////////////////////////

//#define N 128
//#define P 16384
#define NLEN 128
#define PROB 16384 //(N*N)

//#define BLK_DIM 2
//#define FPPREC 1
//#define CUDA_DEVICE 1
// Define the type of inter-thread communication: 0 - shared, 1 - register shuffle
#define COMMUNICATION 1

#if FPPREC == 0
# define FP float
#elif FPPREC == 1
# define FP double
#endif

////////////////////////////////////////////////////////////////////////
// Define program parameters
////////////////////////////////////////////////////////////////////////

#define WARP_SIZE 32
#define TBLOCK (128)

#define FPE_CUTTOFF 10000
  
#endif
