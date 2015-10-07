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
#include "trid_common.h"

#ifndef FPPREC
# error "Error: define FPPREC!"
#endif

//#if FPPREC == 0
//#  define FP float
//#elif FPPREC == 1
//#  define FP double
//#else
//#  error "Macro definition FPPREC unrecognized for CUDA"
//#endif

extern char *optarg;
extern int  optind, opterr, optopt; 
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ e.g. ./compare file1.dat file2.dat -nx NX -ny NY -nz NZ \n");
  exit(0);
}


int main(int argc, char** argv) {
  // Process arguments
  int  nx=256;
  int  ny=256;
  int  nz=256;
  char filename1[256];
  char filename2[256];

  // Get program arguments
  strcpy(filename1, argv[1]);
  strcpy(filename2, argv[2]);

  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp(options[opt_index].name,"nx"  ) == 0) nx   = atoi(optarg); //printf("nx   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"ny"  ) == 0) ny   = atoi(optarg); //printf("ny   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"nz"  ) == 0) nz   = atoi(optarg); //printf("nz   ===== %d\n",atoi(optarg));
    if(strcmp(options[opt_index].name,"help") == 0) print_help();
  }

  printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);

  // Declare stuff
  FILE *fin1, *fin2;
  FP   *h_u1, *h_u2, *diff, *rel_diff;
  int  size = nx*ny*nz;
  h_u1     = (FP*) calloc(size, sizeof(FP));
  h_u2     = (FP*) calloc(size, sizeof(FP));
  diff     = (FP*) calloc(size, sizeof(FP));
  rel_diff = (FP*) calloc(size, sizeof(FP));

  // Open files
  printf("Opening file: %s \n", filename1);
  fin1 = fopen(filename1,"r");
  printf("Opening file: %s \n", filename2);
  fin2 = fopen(filename2,"r");

  // Read files
  if(size != fread(h_u2,sizeof(FP),size,fin2)) 
    printf("There was an error while reading the file %s!\n", filename1);
  if(size != fread(h_u1,sizeof(FP),size,fin1)) 
    printf("There was an error while reading the file %s!\n", filename2);
//  for (k=0; k<nz; k++) {
//    for (j=0; j<ny; j++) {
//      for (i=0; i<nx; i++) {
//        ind = i + j*nx + k*nx*ny;
//        fprintf(fout, " %5.20e ", h_u[ind]);
//      }
//    }
//  }
  FP sum = 0.0f, rel_diff_reg;
  int i,j,k,ind;
  int count = 0;
  for (k=0; k<nz; k++) {
    for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
        ind = i + j*nx + k*nx*ny;
        diff[ind] = h_u2[ind]-h_u1[ind];
        //if(diff[ind] != 0.0) {
        //  printf(" %e %e \n", h_u1[ind], h_u2[ind]);
        //}
        if(isnan(diff[ind])) {
          printf("%d %d %d\n",i,j,k);
        }
        rel_diff_reg = diff[ind] / h_u1[ind];
        rel_diff[ind] = isnan(rel_diff_reg) ? 0 : rel_diff_reg;   
        //if(rel_diff[ind] > 1e-5) {
        //  count++;
        //  printf("\nRelative error %g exceeded error tolerance (1e-6) %d times! \n", rel_diff[ind], count);
        //}
        sum += diff[ind];
      }
      //printf("\n");
    }
    //printf("\n");
  }
  printf("\nSumOfDiff = %e; Normalized SumOfDiff = %e \n", sum, sum/(FP)(nx*ny*nz));
  fclose(fin1);
  fclose(fin2);
  free(h_u1);
  free(h_u2);
  free(diff);
  free(rel_diff);
}
