#ifndef TRID_CPU_MPI_WRAPPERS_HPP
#define TRID_CPU_MPI_WRAPPERS_HPP
#include <trid_mpi_cpu.h>

template <typename Float>
tridStatus_t tridStridedBatchWrapper(const MpiSolverParams &params,
                                     const Float *a, const Float *b,
                                     const Float *c, Float *d, Float *u,
                                     int ndim, int solvedim, const int *dims,
                                     const int *pads_m, const int *pads_p);

template <>
tridStatus_t tridStridedBatchWrapper<float>(const MpiSolverParams &params,
                                            const float *a, const float *b,
                                            const float *c, float *d, float *u,
                                            int ndim, int solvedim, const int *dims,
                                            const int *pads_m, const int *pads_p) {
  return tridSmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads_m, pads_p);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const MpiSolverParams &params,
                                             const double *a, const double *b,
                                             const double *c, double *d,
                                             double *u, int ndim, int solvedim,
                                             const int *dims, const int *pads_m,
                                             const int *pads_p) {
  return tridDmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads_m, pads_p);
}


#endif /* ifndef TRID_CPU_MPI_WRAPPERS_HPP */
