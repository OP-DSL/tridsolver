#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_NOSTDOUT
#include "catch.hpp"
#include "catch_mpi_outputs.hpp"

#include <cstdlib>
#include <mpi.h>

// New cerr/cout/clog to separate the outputs of the different processes
namespace catch_mpi_outputs {
std::stringstream cout, cerr, clog;
}

int main(int argc, char *argv[]) {
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  // For the debug prints
  int rank, num_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  std::srand(rank);

  int result = Catch::Session().run(argc, argv);

  // Print the outputs
  for (int i = 0; i < num_proc; ++i) {
    if (i == rank) {
      if (catch_mpi_outputs::cout.str().size() > 0) {
        std::cout << "##########################\n"
                  << "Rank " << i << " stdout:\n"
                  << "##########################\n"
                  << catch_mpi_outputs::cout.str();
      }
      if (catch_mpi_outputs::cerr.str().size() > 0) {
        std::cerr << "##########################\n"
                  << "Rank " << i << " stderr:\n"
                  << "##########################\n"
                  << catch_mpi_outputs::cerr.str();
      }
      if (catch_mpi_outputs::clog.str().size() > 0) {
        std::clog << "##########################\n"
                  << "Rank " << i << " stdlog:\n"
                  << "##########################\n"
                  << catch_mpi_outputs::clog.str();
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return result;
}
