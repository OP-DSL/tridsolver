#ifndef TRID_ITERATIVE_MPI_HPP_INCLUDED
#define TRID_ITERATIVE_MPI_HPP_INCLUDED
#include "trid_mpi_solver_params.hpp"
#include "cuda_timing.h"
#include <cassert>

#include <iostream>
#include <string>
#include <vector>

#include "trid_mpi_helper.hpp"
#include "trid_mpi_common.hpp"

/*
 * Modified forwards pass in x direction
 * Each array should have a size of sys_size*sys_n, although the first element
 * of a (a[0]) in the first process and the last element of c in the last
 * process will not be used eventually
 */
template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
__global__ void
trid_linear_forward_pass(const REAL *__restrict__ a, const REAL *__restrict__ b,
                         const REAL *__restrict__ c, REAL *__restrict__ d,
                         REAL *__restrict__ aa, REAL *__restrict__ cc,
                         REAL *__restrict__ boundaries, int sys_size,
                         int sys_pads, int sys_n, int rank, int nproc) {

  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // forward pass
    //
    // i = 0 save and perform at last step
    REAL b_0 = b[ind];
    REAL c_0 = c[ind];
    REAL d_0 = d[ind];
    { // i = 1
      if (rank == 0) {
        REAL factor = a[ind + 1] / b_0;
        REAL bb     = 1 / (b[ind + 1] - factor * c_0);
        d[ind + 1]  = bb * (d[ind + 1] - factor * d_0);
        cc[ind + 1] = bb * c[ind + 1];
        aa[ind + 1] = 0;
        if (shift_c0_on_rank0) {
          d_0 = d_0 - c_0 * d[ind + 1];
          c_0 = -c_0 * cc[ind + 1];
        }
      } else {
        d[ind + 1]  = d[ind + 1] / b[ind + 1];
        cc[ind + 1] = c[ind + 1] / b[ind + 1];
        aa[ind + 1] = a[ind + 1] / b[ind + 1];
        b_0         = b_0 - c_0 * aa[ind + 1];
        d_0         = d_0 - c_0 * d[ind + 1];
        if (rank == nproc - 1 && sys_size == 2) {
          c_0 = 0;
        } else {
          c_0 = -c_0 * cc[ind + 1];
        }
      }
    }

    // eliminate lower off-diagonal
    for (int i = 2; i < sys_size; i++) {
      int loc_ind = ind + i;
      REAL bb =
          static_cast<REAL>(1.0) / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
      d[loc_ind] = (d[loc_ind] - a[loc_ind] * d[loc_ind - 1]) * bb;
      if (i == sys_size - 1 && rank == nproc - 1) {
        cc[loc_ind] = 0;
      } else {
        cc[loc_ind] = c[loc_ind] * bb;
      }
      if (rank == 0) {
        aa[loc_ind] = 0;
        if (shift_c0_on_rank0) {
          d_0 = d_0 - c_0 * d[loc_ind];
          c_0 = -c_0 * cc[loc_ind];
        }
      } else {
        aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
        b_0         = b_0 - c_0 * aa[loc_ind];
        d_0         = d_0 - c_0 * d[loc_ind];
        c_0         = -c_0 * cc[loc_ind];
      }
    }
    // i = 0
    if (0 == rank) {
      aa[ind] = 0;
    } else {
      aa[ind] = a[ind] / b_0;
    }
    cc[ind] = c_0 / b_0;
    d[ind]  = d_0 / b_0;
    // prepare boundaries for communication
    copy_boundaries_linear<REAL, boundary_SOA>(aa, cc, d, boundaries, tid, ind,
                                               sys_size, sys_n);
  }
}

//
// Modified Thomas backward pass
//
template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
__global__ void
trid_linear_backward_pass(const REAL *__restrict__ aa,
                          const REAL *__restrict__ cc, REAL *__restrict__ d,
                          REAL *__restrict__ u,
                          const REAL *__restrict__ boundaries, int sys_size,
                          int sys_pads, int sys_n, int rank, int nproc) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // reverse pass
    //
    REAL dd0, dd_p1;
    load_d_from_boundary_linear<REAL, boundary_SOA>(boundaries, dd0, dd_p1, tid,
                                                    sys_n);

    // i = n-1
    int loc_ind = ind + sys_size - 1;
    if (rank != nproc - 1)
      dd_p1 = d[loc_ind] - dd_p1 * cc[loc_ind];
    else
      dd_p1 = d[loc_ind];
    if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
    if (INC) {
      u[loc_ind] += dd_p1;
    } else {
      d[loc_ind] = dd_p1;
    }
    // i = n-2 - 1
    for (int i = sys_size - 2; i > 0; --i) {
      loc_ind = ind + i;
      dd_p1   = d[loc_ind] - cc[loc_ind] * dd_p1;
      if (rank != 0) dd_p1 += -aa[loc_ind] * dd0;
      if (INC)
        u[ind + i] += dd_p1;
      else
        d[ind + i] = dd_p1;
    }
    // i = 0
    if (0 == rank && !is_c0_cleared_on_rank0) {
      dd_p1 = dd0 - cc[ind] * dd_p1;
      if (INC)
        u[ind] += dd_p1;
      else
        d[ind] = dd_p1;
    } else {
      if (INC)
        u[ind] += dd0;
      else
        d[ind] = dd0;
    }
  }
}

//
//  Single PCR iteration for reduced
//  assumption: we only have one row per system in boundaries
//  and 1-1 prev and next row in recv_buf
//
template <typename REAL, bool assume_nonzeros = false>
__global__ void trid_PCR_iteration(REAL *__restrict__ boundaries,
                                   const REAL *__restrict__ recv_m1,
                                   const REAL *__restrict__ recv_p1, int sys_n,
                                   bool m1, bool p1) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = tid;

  if (tid < sys_n) {
    REAL am1 = 0.0;
    REAL cm1 = 0.0;
    REAL dm1 = 0.0;
    REAL ap1 = 0.0;
    REAL cp1 = 0.0;
    REAL dp1 = 0.0;
    if (m1) {
      am1 = recv_m1[0 * sys_n + ind];
      cm1 = recv_m1[1 * sys_n + ind];
      dm1 = recv_m1[2 * sys_n + ind];
    }
    if (p1) {
      ap1 = recv_p1[0 * sys_n + ind];
      cp1 = recv_p1[1 * sys_n + ind];
      dp1 = recv_p1[2 * sys_n + ind];
    }
    REAL bbi = 1.0 / (1 - boundaries[0 * sys_n + ind] * cm1 -
                      boundaries[1 * sys_n + ind] * ap1);
    // d
    boundaries[2 * sys_n + ind] =
        bbi * (boundaries[2 * sys_n + ind] - boundaries[0 * sys_n + ind] * dm1 -
               boundaries[1 * sys_n + ind] * dp1);
    // a
    if (m1)
      boundaries[0 * sys_n + ind] = bbi * -boundaries[0 * sys_n + ind] * am1;
    else if (assume_nonzeros)
      boundaries[0 * sys_n + ind] *= bbi;
    // c
    if (p1)
      boundaries[1 * sys_n + ind] = bbi * -boundaries[1 * sys_n + ind] * cp1;
    else if (assume_nonzeros)
      boundaries[1 * sys_n + ind] *= bbi;
  }
}

////////////////////////////////////////
//        Host functions
////////////////////////////////////////

template <typename REAL, bool boundary_SOA, bool shift_c0_on_rank0 = true>
inline void forward_batched_pass(dim3 dimGrid_x, dim3 dimBlock_x,
                                 const MpiSolverParams &params, const REAL *a,
                                 const int *a_pads, const REAL *b,
                                 const int *b_pads, const REAL *c,
                                 const int *c_pads, REAL *d, const int *d_pads,
                                 REAL *aa, REAL *cc, REAL *boundaries,
                                 REAL *send_buf_h, const int *dims, int ndim,
                                 int solvedim, int start_sys, int bsize,
                                 int offset, cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    assert(a_pads[solvedim] == b_pads[solvedim] &&
           a_pads[solvedim] == c_pads[solvedim] &&
           a_pads[solvedim] == d_pads[solvedim] &&
           "different paddings are not supported");
    assert(a_pads[1] == dims[1] && b_pads[1] == dims[1] &&
           c_pads[1] == dims[1] && d_pads[1] == dims[1] &&
           " ONLLY X paddings are supported");
    assert(offset == 0 && "I think we do not need the offset");
    const int batch_offset = start_sys * a_pads[solvedim];
    trid_linear_forward_pass<REAL, boundary_SOA, shift_c0_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            a + batch_offset, b + batch_offset, c + batch_offset,
            d + batch_offset, aa + batch_offset, cc + batch_offset,
            boundaries + start_sys * 3 * 2, dims[solvedim], a_pads[solvedim],
            bsize, params.mpi_coords[solvedim], params.num_mpi_procs[solvedim]);
  } else {
    assert(false);
    // DIM_V k_pads, k_dims; // TODO
    // for (int i = 0; i < ndim; ++i) {
    //   k_pads.v[i] = a_pads[i];
    //   k_dims.v[i] = dims[i];
    // }
    // trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x, 0,
    // stream>>>(
    //     a, k_pads, b, k_pads, c, k_pads, d, k_pads, aa, cc, dd, boundaries,
    //     ndim, solvedim, bsize, k_dims, start_sys);
  }
  // #if !(defined(TRID_CUDA_AWARE_MPI) || defined(TRID_NCCL))
  //   size_t comm_buf_size   = 3 * 2 * bsize;
  //   size_t comm_buf_offset = 3 * 2 * start_sys;
  //   cudaMemcpyAsync(send_buf_h + comm_buf_offset, boundaries +
  //   comm_buf_offset,
  //                   sizeof(REAL) * comm_buf_size, cudaMemcpyDeviceToHost,
  //                   stream);
  // #endif
}

template <typename REAL, int INC, bool boundary_SOA,
          bool is_c0_cleared_on_rank0 = true>
inline void backward_batched_pass(dim3 dimGrid_x, dim3 dimBlock_x,
                                  const MpiSolverParams &params, const REAL *aa,
                                  const int *a_pads, const REAL *cc,
                                  const int *c_pads, const REAL *boundaries,
                                  REAL *d, const int *d_pads, REAL *u,
                                  const int *u_pads, const int *dims, int ndim,
                                  int solvedim, int start_sys, int bsize,
                                  int offset, cudaStream_t stream = nullptr) {
  if (solvedim == 0) {
    assert(a_pads[solvedim] == c_pads[solvedim] &&
           a_pads[solvedim] == d_pads[solvedim] &&
           "different paddings are not supported");
    assert(a_pads[1] == dims[1] && c_pads[1] == dims[1] &&
           d_pads[1] == dims[1] && " ONLLY X paddings are supported");
    assert(offset == 0 && "I think we do not need the offset");
    assert(start_sys == 0 &&
           "check the whole process for boundaries if indexing is correct for "
           "batches then remove this");
    const int batch_offset = start_sys * a_pads[solvedim];
    // int y_size = 1, y_pads = 1;
    // if (ndim > 1) {
    //   y_size = dims[1];
    //   y_pads = a_pads[1];
    // }
    trid_linear_backward_pass<REAL, INC, boundary_SOA, is_c0_cleared_on_rank0>
        <<<dimGrid_x, dimBlock_x, 0, stream>>>(
            aa + batch_offset, cc + batch_offset, d + batch_offset,
            u + batch_offset, boundaries + start_sys * 2 * 3, dims[solvedim],
            a_pads[solvedim], bsize, params.mpi_coords[solvedim],
            params.num_mpi_procs[solvedim]);
  } else {
    assert(false);
    // DIM_V k_pads, k_dims; // TODO
    // for (int i = 0; i < ndim; ++i) {
    //   k_pads.v[i] = a_pads[i];
    //   k_dims.v[i] = dims[i];
    // }
    // trid_strided_multidim_backward<REAL, INC>
    //     <<<dimGrid_x, dimBlock_x, 0, stream>>>(
    //         aa, k_pads, cc, k_pads, dd, d, k_pads, u, k_pads, boundaries,
    //         ndim, solvedim, bsize, k_dims, start_sys);
  }
}

template <typename REAL, bool snd_down = true, bool snd_up = true>
void trid_cuda_pcr_exchange_line(REAL *snd_d, REAL *snd_h, REAL *rcv_m1_d,
                                 REAL *rcv_m1_h, REAL *rcv_p1_d, REAL *rcv_p1_h,
                                 int line_size, int nproc, int rank_m1,
                                 int rank_p1, const MpiSolverParams &params,
                                 int solvedim, MPI_Request *rcv_requests,
                                 MPI_Request *snd_requests) {
  BEGIN_PROFILING2("mpi_communication");
  static_assert(snd_down || snd_up);
  constexpr int tag = 42;
#ifdef TRID_CUDA_AWARE_MPI
  // Exchange line with upper process
  if (rank_m1 >= 0 && rank_m1 < nproc) {
    if (snd_down)
      MPI_Irecv(rcv_m1_d, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
                params.communicators[solvedim], &rcv_requests[0]);
    if (snd_up)
      MPI_Isend(snd_d, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
                params.communicators[solvedim], &snd_requests[0]);
  }
  // Exchange line with lower process
  if (rank_p1 < nproc && rank_p1 >= 0) {
    if (snd_up)
      MPI_Irecv(rcv_p1_d, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
                params.communicators[solvedim], &rcv_requests[1]);
    if (snd_down)
      MPI_Isend(snd_d, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
                params.communicators[solvedim], &snd_requests[1]);
  }
  MPI_Waitall(2, rcv_requests, MPI_STATUS_IGNORE);
#elif defined(TRID_NCCL)
  NCCLCHECK(ncclGroupStart());
  // Exchange line with upper process
  if (rank_m1 >= 0 && rank_m1 < nproc) {
    if (snd_down)
      NCCLCHECK(ncclRecv(rcv_m1_d, line_size * sizeof(REAL), ncclChar, rank_m1,
                         params.ncclComms[solvedim], 0));
    if (snd_up)
      NCCLCHECK(ncclSend(snd_d, line_size * sizeof(REAL), ncclChar, rank_m1,
                         params.ncclComms[solvedim], 0));
  }
  // Exchange line with lower process
  if (rank_p1 < nproc && rank_p1 >= 0) {
    if (snd_up)
      NCCLCHECK(ncclRecv(rcv_p1_d, line_size * sizeof(REAL), ncclChar, rank_p1,
                         params.ncclComms[solvedim], 0));
    if (snd_down)
      NCCLCHECK(ncclSend(snd_d, line_size * sizeof(REAL), ncclChar, rank_p1,
                         params.ncclComms[solvedim], 0));
  }
  NCCLCHECK(ncclGroupEnd());
  if ((rank_m1 >= 0 && rank_m1 < nproc) || (rank_p1 < nproc && rank_p1 >= 0)) {
    cudaSafeCall(cudaDeviceSynchronize());
  }
#else
  // Exchange line with upper process
  if (snd_down && rank_m1 >= 0 && rank_m1 < nproc) {
    MPI_Irecv(rcv_m1_h, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
              params.communicators[solvedim], &rcv_requests[0]);
  }
  // Exchange line with lower process
  if (snd_up && rank_p1 >= 0 && rank_p1 < nproc) {
    MPI_Irecv(rcv_p1_h, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
              params.communicators[solvedim], &rcv_requests[0]);
  }

  // copy send buffer to host
  if ((snd_up && rank_m1 >= 0 && rank_m1 < nproc) ||
      (snd_down && rank_p1 < nproc && rank_p1 >= 0)) {
    cudaMemcpy(snd_h, snd_d, line_size * sizeof(REAL), cudaMemcpyDeviceToHost);
  }

  // Exchange line with upper process
  if (snd_up && rank_m1 >= 0 && rank_m1 < nproc) {
    MPI_Isend(snd_h, line_size, MPI_DATATYPE(REAL), rank_m1, tag,
              params.communicators[solvedim], &snd_requests[0]);
  }
  // Exchange line with lower process
  if (snd_down && rank_p1 >= 0 && rank_p1 < nproc) {
    MPI_Isend(snd_h, line_size, MPI_DATATYPE(REAL), rank_p1, tag,
              params.communicators[solvedim], &snd_requests[1]);
  }
  MPI_Waitall(2, rcv_requests, MPI_STATUS_IGNORE);
  // Exchange line with upper process
  if (snd_down && rank_m1 >= 0 && rank_m1 < nproc) {
    // copy the received line to device
    cudaMemcpyAsync(rcv_m1_d, rcv_m1_h, line_size * sizeof(REAL),
                    cudaMemcpyHostToDevice);
  }
  // Exchange line with lower process
  if (snd_up && rank_p1 >= 0 && rank_p1 < nproc) {
    // copy the received line to device
    cudaMemcpyAsync(rcv_p1_d, rcv_p1_h, line_size * sizeof(REAL),
                    cudaMemcpyHostToDevice);
  }
#endif
  END_PROFILING2("mpi_communication");
}

// template <typename REAL>
// void print_array(std::string promt, REAL *arr, size_t size,
//                  std::ostream &o = std::cout) {
//   o << promt << ": [";
//   for (size_t i = 0; i < size; ++i) {
//     if (i) o << ", ";
//     o << arr[i];
//   }
//   o << "]\n";
// }

template <typename REAL>
inline void iterative_pcr_on_reduced(dim3 dimGrid_x, dim3 dimBlock_x,
                                     const MpiSolverParams &params,
                                     REAL *boundaries, int sys_n, int solvedim,
                                     REAL *recv_buf_d, REAL *recv_buf_h,
                                     REAL *send_buf_h) {

  MPI_Request rcv_requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  MPI_Request snd_requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  const int rank              = params.mpi_coords[solvedim];
  const int nproc             = params.num_mpi_procs[solvedim];
  constexpr int nvar_per_sys  = 3; // a, c, d
  const int line_size         = sys_n * nvar_per_sys;


  // send bottom line down
  trid_cuda_pcr_exchange_line<REAL, true, false>(
      boundaries + line_size, send_buf_h, recv_buf_d, recv_buf_h, nullptr,
      nullptr, line_size, nproc, rank - 1, rank + 1, params, solvedim,
      rcv_requests, snd_requests);

  // substract received line
  if (rank)
    trid_PCR_iteration<REAL, true><<<dimGrid_x, dimBlock_x>>>(
        boundaries, recv_buf_d, nullptr, sys_n, true, false);
#ifndef TRID_NCCL
  MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
#endif

  // PCR iterations
  int P = (int)ceil(log2((REAL)params.num_mpi_procs[solvedim]));

  for (int p = 0, s = 1; p < P; p++, s <<= 1) {
    int rank_ms = rank - s;
    int rank_ps = rank + s;
    if (rank_ms >= 0 || rank_ps < nproc) {
      // send & recv
      trid_cuda_pcr_exchange_line<REAL>(
          boundaries, send_buf_h, recv_buf_d, recv_buf_h,
          boundaries + line_size, recv_buf_h + line_size, line_size, nproc,
          rank_ms, rank_ps, params, solvedim, rcv_requests, snd_requests);

      // iteration
      trid_PCR_iteration<<<dimGrid_x, dimBlock_x>>>(
          boundaries, recv_buf_d, boundaries + line_size, sys_n, rank_ms >= 0,
          rank_ps < nproc);
#ifndef TRID_NCCL
      MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
#endif
    }
  }
  // send solution up for backward
  trid_cuda_pcr_exchange_line<REAL, false, true>(
      boundaries + 2 * sys_n, send_buf_h, nullptr, nullptr,
      boundaries + 5 * sys_n, recv_buf_h, sys_n, nproc, rank - 1, rank + 1,
      params, solvedim, rcv_requests, snd_requests);
#ifndef TRID_NCCL
  MPI_Waitall(2, snd_requests, MPI_STATUS_IGNORE);
#endif
}

#endif /* ifndef TRID_ITERATIVE_MPI_HPP_INCLUDED */
