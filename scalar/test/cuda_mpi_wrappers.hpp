#ifndef TRID_CUDA_MPI_WRAPPERS_HPP
#define TRID_CUDA_MPI_WRAPPERS_HPP

#include <tridsolver.h>
#include <trid_mpi_solver_params.hpp>

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(const MpiSolverParams *params,
                                            const Float *a, const Float *b,
                                            const Float *c, Float *d, Float *u,
                                            int ndim, int solvedim,
                                            const int *dims, const int *pads);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams *params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, const int *dims,
    const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridSmtsvStridedBatch(&trid_params, a, b, c, d, ndim, solvedim,
                               dims, pads);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams *params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim,
    const int *dims, const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridDmtsvStridedBatch(&trid_params, a, b, c, d, ndim, solvedim,
                               dims, pads);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float, true>(
    const MpiSolverParams *params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, const int *dims,
    const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridSmtsvStridedBatchInc(&trid_params, a, b, c, d, u, ndim, solvedim,
                                  dims, pads);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double, true>(
    const MpiSolverParams *params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim,
    const int *dims, const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridDmtsvStridedBatchInc(&trid_params, a, b, c, d, u, ndim, solvedim,
                                  dims, pads);
}

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(
    const MpiSolverParams *params, const Float *a, const int *a_pads,
    const Float *b, const int *b_pads, const Float *c, const int *c_pads,
    Float *d, const int *d_pads, Float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims);

#endif /* ifndef TRID_CUDA_MPI_WRAPPERS_HPP */
