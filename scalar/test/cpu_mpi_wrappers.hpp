#ifndef TRID_CPU_MPI_WRAPPERS_HPP
#define TRID_CPU_MPI_WRAPPERS_HPP

#include <tridsolver.h>
#include <trid_mpi_solver_params.hpp>

template <typename Float>
tridStatus_t tridStridedBatchWrapper(const MpiSolverParams *params,
                                     const Float *a, const Float *b,
                                     const Float *c, Float *d, int ndim,
                                     int solvedim, const int *dims,
                                     const int *pads);

template <>
tridStatus_t tridStridedBatchWrapper<float>(const MpiSolverParams *params,
                                            const float *a, const float *b,
                                            const float *c, float *d, int ndim,
                                            int solvedim, const int *dims,
                                            const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridSmtsvStridedBatch(&trid_params, a, b, c, d, ndim, solvedim, dims,
                               pads);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const MpiSolverParams *params,
                                             const double *a, const double *b,
                                             const double *c, double *d,
                                             int ndim, int solvedim,
                                             const int *dims, const int *pads) {
  TridParams trid_params;
  trid_params.mpi_params = (void *)params;
  return tridDmtsvStridedBatch(&trid_params, a, b, c, d, ndim, solvedim, dims,
                               pads);
}


#endif /* ifndef TRID_CPU_MPI_WRAPPERS_HPP */
