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

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

// print out left corner of array
for(int k=0; k<2; k++) {
//for (k=nz-4; k<nz; k++) {
  printf("k = %i\n",k);
  for(int j=0; j<MIN(ny,17); j++) {
  //for (j=ny-9; j<ny; j++) {
    printf(" %d   ", j);
    for(int i=0; i<MIN(nx,17); i++) {
      //ind = i + j*(nx+STRIDE) + k*(nx+STRIDE)*ny;
      int ind = i + j*ldim + k*ldim*ny;
      printf(" %5.5g ", h_u[ind]);
      //printf(" %d ", (int) h_u[ind]);
    }
    printf("\n");
  }
  printf("\n");
}

// print out right corner of array
for(int k=0; k<2; k++) {
//for (k=nz-4; k<nz; k++) {
  printf("k = %i\n",k);
  for(int j=0; j<MIN(ny,17); j++) {
  //for (j=ny-9; j<ny; j++) {
    printf(" %d   ", j);
    for(int i=MAX(0,nx-17); i<nx; i++) {
      //ind = i + j*(nx+STRIDE) + k*(nx+STRIDE)*ny;
      int ind = i + j*ldim + k*ldim*ny;
      printf(" %5.5g ", h_u[ind]);
      //printf(" %d ", (int) h_u[ind]);
    }
    printf("\n");
  }
  printf("\n");
}

// Set output filname to binary executable.dat
char out_filename[256];
strcpy(out_filename,argv[0]);
strcat(out_filename,".dat");
// print h_u to file
FILE *fout;
fout = fopen(out_filename,"w");
if(fout == NULL) {
  printf("ERROR: File stream could not be opened. Data will not be written to file!\n");
}else{
  //fwrite(h_u,sizeof(float),(nx+STRIDE)*ny*nz,fout);
  for(int k=0; k<nz; k++) {
    for(int j=0; j<ny; j++) {
      for(int i=0; i<nx; i++) {
        int ind = i + j*ldim + k*ldim*ny;
        //h_u[ind] = i + j*nx + k*nx*ny;
        fwrite(&h_u[ind],sizeof(FP),1,fout);
      }
    }
  }
  //fwrite(h_u,sizeof(float),nx*ny*nz,fout);
  fclose(fout);
}
