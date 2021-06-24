#ifndef TRID_CUDA_MPI_WRAPPERS_HPP
#define TRID_CUDA_MPI_WRAPPERS_HPP
#include "trid_mpi_cuda.hpp"

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(const MpiSolverParams &params,
                                            const Float *a, const Float *b,
                                            const Float *c, Float *d, Float *u,
                                            int ndim, int solvedim,
                                            const int *dims, const int *pads,
                                            const int offset = 0);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, const int *dims,
    const int *pads, const int offset) {
  return tridSmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads, offset);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim,
    const int *dims, const int *pads, const int offset) {
  return tridDmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads, offset);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float, true>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, const int *dims,
    const int *pads, const int offset) {
  return tridSmtsvStridedBatchIncMPI(params, a, b, c, d, u, ndim, solvedim,
                                     dims, pads, offset);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double, true>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim,
    const int *dims, const int *pads, const int offset) {
  return tridDmtsvStridedBatchIncMPI(params, a, b, c, d, u, ndim, solvedim,
                                     dims, pads, offset);
}

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(
    const MpiSolverParams &params, const Float *a, const int *a_pads,
    const Float *b, const int *b_pads, const Float *c, const int *c_pads,
    Float *d, const int *d_pads, Float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims, const int offset = 0);

#endif /* ifndef TRID_CUDA_MPI_WRAPPERS_HPP */
