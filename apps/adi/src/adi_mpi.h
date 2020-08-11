#ifndef __ADI_MPI_CUDA_H
#define __ADI_MPI_CUDA_H

#include "mpi.h"
#include "trid_mpi_solver_params.hpp"

// ADI handle for MPI versions of the application
struct app_handle {
  FP *a;
  FP *b;
  FP *c;
  FP *d;
  FP *u;
  
  int *size_g;
  int *size;
  int *start_g;
  int *end_g;
  int *pads;
  
  int *pdims;
  int *coords;
  
  MPI_Comm comm;
  MpiSolverParams *params;
};

#endif
