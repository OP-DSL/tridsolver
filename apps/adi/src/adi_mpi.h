#ifndef __ADI_MPI_H
#define __ADI_MPI_H

#include "mpi.h"

struct mpi_handle {
  int          procs;
  int          rank;
  MPI_Status  *stat;
  MPI_Request *req;
  FP          *halo_sndbuf_x;  // Send Buffer
  FP          *halo_rcvbuf_x;  // Receive Buffer
  FP          *halo_sndbuf_y;  // Send Buffer
  FP          *halo_rcvbuf_y;  // Receive Buffer
  FP          *halo_sndbuf_z;  // Send Buffer
  FP          *halo_rcvbuf_z;  // Receive Buffer
  // Sending and receiving buffers for x, y and z boundaries
  FP          *halo_snd_x;
  FP          *halo_rcv_x;
  FP          *halo_snd_y;
  FP          *halo_rcv_y;
  FP          *halo_snd_z;
  FP          *halo_rcv_z;

  MPI_Comm comm;

  int ndim;
  int* pdims; //number of mpi procs in each dimension
  int* periodic;
  int my_cart_rank; // new rank within a cartecian communicator;
  int* coords; //rank in each dimension - i.e. coordinates of a mpi proc

  MPI_Comm x_comm;
  MPI_Comm y_comm;
  MPI_Comm z_comm;

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

  FP *__restrict__ aa_rx;
  FP *__restrict__ cc_rx;
  FP *__restrict__ dd_rx;
  
  FP *__restrict__ aa_ry;
  FP *__restrict__ cc_ry;
  FP *__restrict__ dd_ry;
  
  FP *__restrict__ aa_rz;
  FP *__restrict__ cc_rz;
  FP *__restrict__ dd_rz;
  
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
  
  int sys_len_lx; // Reduced system size in X dim
  int n_sys_gx;   // Number of reduced systems on global scale for X dimension
  int n_sys_lx;   // Number of reduced systems solved by the current process for X dimension
  
  int sys_len_ly; // Reduced system size in Y dim
  int n_sys_gy;   // Number of reduced systems on global scale for Y dimension
  int n_sys_ly;   // Number of reduced systems solved by the current process for Y dimension
  
  int sys_len_lz; // Reduced system size in Z dim
  int n_sys_gz;   // Number of reduced systems on global scale for Z dimension
  int n_sys_lz;   // Number of reduced systems solved by the current process for Z dimension

#define TIMERS 11
  double elapsed_time[TIMERS];
  double   timers_min[TIMERS];
  double   timers_max[TIMERS];
  double   timers_avg[TIMERS];
  char   elapsed_name[TIMERS][256];
};

#endif
