#ifndef TRID_CPU_MPI_WRAPPERS_HPP
#define TRID_CPU_MPI_WRAPPERS_HPP

#include <tridsolver.h>
#include <trid_mpi_solver_params.hpp>

template <typename Float>
tridStatus_t tridStridedBatchWrapper(const MpiSolverParams *params,
                                     const Float *a, const Float *b,
                                     const Float *c, Float *d, Float *u,
                                     int ndim, int solvedim, const int *dims,
                                     const int *pads);

template <>
tridStatus_t tridStridedBatchWrapper<float>(const MpiSolverParams *params,
                                            const float *a, const float *b,
                                            const float *c, float *d, float *u,
                                            int ndim, int solvedim, const int *dims,
                                            const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridSmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads,
                               &trid_params);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const MpiSolverParams *params,
                                             const double *a, const double *b,
                                             const double *c, double *d,
                                             double *u, int ndim, int solvedim,
                                             const int *dims, const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridDmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads,
                               &trid_params);
}


#endif /* ifndef TRID_CPU_MPI_WRAPPERS_HPP */
