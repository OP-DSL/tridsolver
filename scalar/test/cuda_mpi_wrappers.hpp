#ifndef TRID_CUDA_MPI_WRAPPERS_HPP
#define TRID_CUDA_MPI_WRAPPERS_HPP
#include "trid_mpi_cuda.hpp"

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(const MpiSolverParams &params,
                                            const Float *a, const Float *b,
                                            const Float *c, Float *d, Float *u,
                                            int ndim, int solvedim, int *dims,
                                            int *pads);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, int *dims,
    int *pads) {
  return tridSmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads, nullptr);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim, int *dims,
    int *pads) {
  return tridDmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads, nullptr);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float, true>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, int *dims,
    int *pads) {
  return tridSmtsvStridedBatchIncMPI(params, a, b, c, d, u, ndim, solvedim,
                                     dims, pads, nullptr);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double, true>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim, int *dims,
    int *pads) {
  return tridDmtsvStridedBatchIncMPI(params, a, b, c, d, u, ndim, solvedim,
                                     dims, pads, nullptr);
}

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(
    const MpiSolverParams &params, const Float *a, int *a_pads, const Float *b,
    int *b_pads, const Float *c, int *c_pads, Float *d, int *d_pads, Float *u,
    int *u_pads, int ndim, int solvedim, int *dims);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims) {
  return tridSmtsvStridedBatchPaddedMPI(params, a, a_pads, b, b_pads, c, c_pads, d,
                                  d_pads, u, u_pads, ndim, solvedim, dims, nullptr);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims) {
  return tridDmtsvStridedBatchPaddedMPI(params, a, a_pads, b, b_pads, c, c_pads, d,
                                  d_pads, u, u_pads, ndim, solvedim, dims, nullptr);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float, true>(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims) {
  return tridSmtsvStridedBatchPaddedIncMPI(params, a, a_pads, b, b_pads, c, c_pads, d,
                                     d_pads, u, u_pads, ndim, solvedim, dims, nullptr);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double, true>(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims) {
  return tridDmtsvStridedBatchPaddedIncMPI(params, a, a_pads, b, b_pads, c, c_pads, d,
                                     d_pads, u, u_pads, ndim, solvedim, dims, nullptr);
}

#endif /* ifndef TRID_CUDA_MPI_WRAPPERS_HPP */
