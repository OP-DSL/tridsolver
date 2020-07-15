#include <iostream>
#include <map>
#include <vector>
#include <mpi.h>
#include <unistd.h>

#include "cuda_utils.hpp"
#include "timing.h"
#include <trid_mpi_cuda.hpp>
#include "cuda_mpi_wrappers.hpp"

template <typename Float>
void test_solver_with_generated(const std::vector<int> global_dims,
                                int solvedim, bool is_global_size) {
  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(global_dims.size()), periods(global_dims.size(), 0);
  MPI_Dims_create(num_proc, global_dims.size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, global_dims.size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, global_dims.size(), mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(global_dims.size());
  // The starting indices of the local domain in each dimension.
  if (is_global_size) {
    for (size_t i = 0; i < local_sizes.size(); ++i) {
      const int global_dim = global_dims[i];
      size_t domain_offset = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
      local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                           ? global_dim - domain_offset
                           : global_dim / mpi_dims[i];
    }
  } else {
    local_sizes = global_dims;
  }

  RandomMesh<Float> mesh(local_sizes, solvedim);
  GPUMesh<Float> mesh_d(mesh.a(), mesh.b(), mesh.c(), mesh.d(), local_sizes);

  // Solve the equations
  tridmtsvStridedBatchMPIWrapper<Float>(
      params, mesh_d.a().data(), mesh_d.b().data(), mesh_d.c().data(),
      mesh_d.d().data(), nullptr, mesh_d.dims().size(), mesh.solve_dim(),
      local_sizes.data(), local_sizes.data());
}

void usage(const char *name) {
  std::cerr << "Usage:\n";
  std::cerr << "\t" << name << " [-x nx -d ndims -s solvedim -l]" << std::endl;
}

int main(int argc, char *argv[]) {
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int opt;
  int size            = 256;
  int ndims           = 2;
  int solvedim        = 0;
  bool is_global_size = true;
  while ((opt = getopt(argc, argv, "x:s:d:l")) != -1) {
    switch (opt) {
    case 'x': size = atoi(optarg); break;
    case 'd': ndims = atoi(optarg); break;
    case 's': solvedim = atoi(optarg); break;
    case 'l': is_global_size = false; break;
    default:
      if (rank == 0) usage(argv[0]);
      return 2;
      break;
    }
  }

  std::vector<int> dims;
  for (int i = 0; i < ndims; ++i) {
    dims.push_back(size);
  }
  if (rank == 0) {
    std::cout << argv[0] << " {" << size;
    for (size_t i = 1; i < dims.size(); ++i)
      std::cout << "x" << dims[i];
    std::cout << "}";
    if (!is_global_size) std::cout << "/node";
    std::cout << " solvedim" << solvedim << "\n";
  }
  if (solvedim >= (int)dims.size()) {
    if (rank == 0) {
      std::cerr << "Solvedim must be smaller than number of dimenstions!\n";
    }
    return 2;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  test_solver_with_generated<double>(dims, solvedim, is_global_size);

  PROFILE_REPORT();
  MPI_Finalize();
  return 0;
}
