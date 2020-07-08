#ifndef CATCH_MPI_OUTPUTS_HPP_MV2GTGJR
#define CATCH_MPI_OUTPUTS_HPP_MV2GTGJR

#include <sstream>

// New cerr/cout/clog to separate the outputs of the different processes
// Use Catch::cout(), etc. in the code
namespace catch_mpi_outputs {
extern std::stringstream cout, cerr, clog;
}

namespace Catch {
inline std::ostream &cout() { return catch_mpi_outputs::cout; }
inline std::ostream &cerr() { return catch_mpi_outputs::cerr; }
inline std::ostream &clog() { return catch_mpi_outputs::clog; }
} // namespace Catch

#endif /* end of include guard: CATCH_MPI_OUTPUTS_HPP_MV2GTGJR */
