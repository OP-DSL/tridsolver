#include <cctype>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <mpi.h>
#include <thread>
#include <unistd.h>


#ifndef TRID_PERF_CUDA
#  include "timing.h"
#  include <trid_mpi_cpu.h>
#  include "utils.hpp"
#  include "cpu_mpi_wrappers.hpp"

template <typename Float>
void run_tridsolver(const MpiSolverParams &params, RandomMesh<Float> mesh,
                    int num_iters) {
  AlignedArray<Float, 1> d(mesh.d());

  // Solve the equations
  while (num_iters--) {
    tridStridedBatchWrapper<Float>(params, mesh.a().data(), mesh.b().data(),
                                   mesh.c().data(), d.data(), nullptr,
                                   mesh.dims().size(), mesh.solve_dim(),
                                   mesh.dims().data(), mesh.dims().data());
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
#else
#  include "cuda_timing.h"
#  include <trid_mpi_cuda.hpp>
#  include "cuda_utils.hpp"
#  include "cuda_mpi_wrappers.hpp"

template <typename Float>
void run_tridsolver(const MpiSolverParams &params, RandomMesh<Float> mesh,
                    int num_iters) {
  GPUMesh<Float> mesh_d(mesh.a(), mesh.b(), mesh.c(), mesh.d(), mesh.dims());

  // Solve the equations
  while (num_iters--) {
    tridmtsvStridedBatchMPIWrapper<Float>(
        params, mesh_d.a().data(), mesh_d.b().data(), mesh_d.c().data(),
        mesh_d.d().data(), nullptr, mesh_d.dims().size(), mesh.solve_dim(),
        mesh_d.dims().data(), mesh_d.dims().data());
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
#endif

void print_local_sizes(int rank, int num_proc, const int *mpi_dims,
                       const std::vector<int> &mpi_coords,
                       const std::vector<int> &local_sizes) {
  for (int i = 0; i < num_proc; ++i) {
    // Print the outputs
    if (i == rank) {
      std::string idx    = std::to_string(mpi_coords[0]),
                  dims   = std::to_string(local_sizes[0]),
                  m_dims = std::to_string(mpi_dims[0]);
      for (size_t j = 1; j < local_sizes.size(); ++j) {
        idx += "," + std::to_string(mpi_coords[j]);
        dims += "x" + std::to_string(local_sizes[j]);
        m_dims += "x" + std::to_string(mpi_dims[j]);
      }
      if (rank == 0) {
        std::cout << "########## Local decomp sizes {" + m_dims +
                         "} ##########\n";
      }
      std::cout << "# Rank " << i << "(" + idx + "){" + dims + "}\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

template <typename Float>
void test_solver_with_generated(const std::vector<int> global_dims,
                                int solvedim,
                                MpiSolverParams::MPICommStrategy strategy,
                                int batch_size, int mpi_parts_in_s,
                                int num_iters) {
  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(global_dims.size(), 0),
      periods(global_dims.size(), 0);
  mpi_dims[solvedim] = std::min(num_proc, mpi_parts_in_s);
  MPI_Dims_create(num_proc, global_dims.size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, global_dims.size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, global_dims.size(), mpi_dims.data(),
                         batch_size, strategy);
  params.jacobi_atol = abs_tolerance<Float>;
  params.jacobi_rtol = rel_tolerance<Float>;

  // The size of the local domain.
  std::vector<int> local_sizes(global_dims.size());
  // The starting indices of the local domain in each dimension.
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = global_dims[i];
    size_t domain_offset = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offset
                               : global_dim / mpi_dims[i];
  }

  print_local_sizes(rank, num_proc, params.num_mpi_procs, params.mpi_coords,
                    local_sizes);

  RandomMesh<Float> mesh(local_sizes, solvedim);
  run_tridsolver(params, mesh, num_iters);
}

std::ostream &operator<<(std::ostream &o,
                         const MpiSolverParams::MPICommStrategy &s) {
  const char *labels[] = {"Gather-scatter",
                          "Allgather",
                          "Jacobi",
                          "PCR",
                          "LatencyHiding-interleaved",
                          "LatencyHiding-two-step"};
  return o << labels[s];
}

void usage(const char *name) {
  std::cerr << "Usage:\n";
  std::cerr
      << "\t" << name
      << " [-x nx -y ny -z nz -d ndims -s solvedim -b batch_size -m "
         "mpir_strat_idx -p num_partitions_along_solvedim] -n num_iterations"
      << std::endl;
}

int main(int argc, char *argv[]) {
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  int num_proc, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  int opt;
  int size[]         = {256, 256, 256};
  int ndims          = 2;
  int solvedim       = 0;
  int batch_size     = 32;
  int mpi_strat_idx  = 1;
  int num_iters      = 1;
  int mpi_parts_in_s = 0; // 0 means automatic
  while ((opt = getopt(argc, argv, "x:y:z:s:d:b:m:p:n:")) != -1) {
    switch (opt) {
    case 'x': size[0] = atoi(optarg); break;
    case 'y': size[1] = atoi(optarg); break;
    case 'z': size[2] = atoi(optarg); break;
    case 'd': ndims = atoi(optarg); break;
    case 's': solvedim = atoi(optarg); break;
    case 'b': batch_size = atoi(optarg); break;
    case 'm': mpi_strat_idx = atoi(optarg); break;
    case 'n': num_iters = atoi(optarg); break;
    case 'p': mpi_parts_in_s = atoi(optarg); break;
    default:
      if (rank == 0) usage(argv[0]);
      return 2;
      break;
    }
  }
  assert(ndims < 4 && "ndims must be smaller or equal than MAXDIM");
  assert(mpi_strat_idx < 6 && mpi_strat_idx > 0 &&
         "No such communication strategy");
  MpiSolverParams::MPICommStrategy strategy =
      MpiSolverParams::MPICommStrategy(mpi_strat_idx);
  std::vector<int> dims;
  for (int i = 0; i < ndims; ++i) {
    dims.push_back(size[i]);
  }
  if (rank == 0) {
    std::string fname = argv[0];
    fname             = fname.substr(fname.rfind("/") + 1);
    std::cout << fname << " " << ndims << "DS" << solvedim << "NP" << num_proc
              << "BS" << batch_size << " " << strategy;
    std::cout << " {" << dims[0];
    for (size_t i = 1; i < dims.size(); ++i)
      std::cout << "x" << dims[i];
    std::cout << "}";
    std::cout << " solvedim" << solvedim << "\n";
  }
  if (solvedim >= (int)dims.size()) {
    if (rank == 0) {
      std::cerr << "Solvedim must be smaller than number of dimenstions!\n";
    }
    return 2;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  test_solver_with_generated<double>(dims, solvedim, strategy, batch_size,
                                     mpi_parts_in_s, num_iters);

  PROFILE_REPORT();
  MPI_Finalize();
  return 0;
}
