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

// This takes a matrix which is stored with each problem stored in one block,
// to a matrix where the first sub-matrix row is stored in one large block, followed by the second etc.
// This means that sub-matrices from different problems are interleaved.
__global__ void transposeMatrix( FP* d_out, FP* d_in ){

	int tid		= threadIdx.x + blockDim.x*blockIdx.x;	// the thread ID
	int sbEl	= tid % BLK_ELS;						// the sub-block element
	int Nval	= (tid % (BLK_ELS*N)) / BLK_ELS;		// the N value
	int Pval	= tid / (BLK_ELS*N);					// the P value

	if(tid < BLK_ELS*N*P) {
		// SIMULTANEOUSLY PERFORM the transposition and STORE the data back to global memory
		d_out[BLK_ELS*P*Nval + BLK_ELS*Pval + sbEl] = d_in[tid];
	}
}

// This performs the same function as the kernel above, but for vectors rather than matrices.
__global__ void transposeVector( FP* d_out, FP* d_in ){

	int tid		= threadIdx.x + blockDim.x*blockIdx.x;	// the thread ID
	int sbEl	= tid % BLK_DIM;						// the sub-block element
	int Nval	= (tid % (BLK_DIM*N)) / BLK_DIM;		// the N value
	int Pval	= tid / (BLK_DIM*N);					// the P value

	if(tid < BLK_DIM*N*P){
		// SIMULTANEOUSLY PERFORM the transposition and STORE the data back to global memory
		d_out[BLK_DIM*P*Nval + BLK_DIM*Pval + sbEl] = d_in[tid];
	}
}

// This undoes the above transposition.
// That is, if the data is stored with the sub-matrices from different problems interleaved,
// this kernel transposes the data so that all of the sub-matrices from a single problem are then
// stored in one large block, followed by the entirety of the data for the next problem.
__global__ void UNtransposeVector( FP* d_out, FP* d_in ){

	int tid		= threadIdx.x + blockDim.x*blockIdx.x;	// the thread ID
	int sbEl	= tid % BLK_DIM;						// the sub-block element
	int Nval	= (tid % (BLK_DIM*N)) / BLK_DIM;		// the N value
	int Pval	= tid / (BLK_DIM*N);					// the P value

	if(tid < BLK_DIM*N*P){
		// SIMULTANEOUSLY PERFORM the transposition and STORE the data back to global memory
		d_out[BLK_DIM*N*Pval + BLK_DIM*Nval + sbEl] = d_in[tid];
	}
}
