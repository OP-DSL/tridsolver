#ifndef __ADI_MPI_H
#define __ADI_MPI_H

#include "mpi.h"

struct mpi_handle {
  int          procs;
  int          rank;
  MPI_Status  *stat;
  MPI_Request *req;
  FP          *halo_sndbuf;  // Send Buffer
  FP          *halo_rcvbuf;  // Receive Buffer
  FP          *halo_sndbuf2; // Send Buffer
  FP          *halo_rcvbuf2; // Receive Buffer
};

struct app_handle {
  // 'h_' prefix - CPU (host) memory space
  FP *__restrict__ h_u;
  FP *__restrict__ du;
  FP *__restrict__ ax;
  FP *__restrict__ bx;
  FP *__restrict__ cx;
  FP *__restrict__ ay;
  FP *__restrict__ by;
  FP *__restrict__ cy;
  FP *__restrict__ az;
  FP *__restrict__ bz;
  FP *__restrict__ cz;
  FP *__restrict__ aa;
  FP *__restrict__ cc;
  FP *__restrict__ dd;
  FP *__restrict__ tmp;
  FP err;
  FP lambda;

  FP *__restrict__ aa_r;
  FP *__restrict__ cc_r;
  FP *__restrict__ dd_r;
  int nx_g;      // Global size in X dim
  int ny_g;      // Global size in Y dim
  int nz_g;      // Global size in Z dim
  int iter;      // Number of iterations
  int opt;       // Optimization
  int prof;      // Profiling
  int nx_pad;    // Local padded size in the X dim

  int x_start_g; // Global start index of partition in the X dim
  int x_end_g;   // Global end index of partition in the X dim

  int y_start_g; // Global start index of partition in the Y dim
  int y_end_g;   // Global end index of partition in the Y dim

  int z_start_g; // Global start index of partition in the Z dim
  int z_end_g;   // Global end index of partition in the Z dim

  int nx;        // Local size in X dim
  int ny;        // Local size in Y dim
  int nz;        // Local size in Z dim
  int sys_len_l; // Reduced system size in X dim
  int n_sys_g;   // Number of reduced systems on global scale
  int n_sys_l;   // Number of reduced systems solved by the current process

#define TIMERS 11
  double elapsed_time[TIMERS];
  double   timers_min[TIMERS];
  double   timers_max[TIMERS];
  double   timers_avg[TIMERS];
  char   elapsed_name[TIMERS][256];
};

#endif
