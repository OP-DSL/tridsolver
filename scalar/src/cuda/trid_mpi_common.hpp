#ifndef TRID_MPI_COMMON_DEFS_HPP_INCLUDED
#define TRID_MPI_COMMON_DEFS_HPP_INCLUDED
#include <mpi.h>
// define MPI_datatype
#if __cplusplus >= 201402L
namespace {
template <typename REAL>
const MPI_Datatype mpi_datatype =
    std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
}

#  ifdef TRID_NCCL
namespace {
template <typename REAL>
constexpr ncclDataType_t nccl_datatype =
    std::is_same<REAL, double>::value ? ncclDouble : ncclFloat;
}

#  endif
#  define MPI_DATATYPE(REAL) mpi_datatype<REAL>
#else
#  define MPI_DATATYPE(REAL)                                                   \
    (std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT)
#endif

#endif /* ifndef TRID_MPI_COMMON_DEFS_HPP_INCLUDED */
