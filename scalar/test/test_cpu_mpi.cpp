#define CATCH_CONFIG_NOSTDOUT
#include "catch.hpp"
#include "catch_mpi_outputs.hpp"
#include "utils.hpp"
#include "catch_utils.hpp"

#include <trid_mpi_cpu.h>
#include "cpu_mpi_wrappers.hpp"

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

template <typename Float, int INC, MpiSolverParams::MPICommStrategy strategy>
void test_solver_from_file(const std::string &file_name) {
  assert(INC == 0 && "Increment testing not implemented");
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size()), periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data(), 32,
                         strategy);
  params.jacobi_atol = abs_tolerance<Float>;
  params.jacobi_rtol = rel_tolerance<Float>;

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load our data
  AlignedArray<Float, 1> a(domain_size), b(domain_size), c(domain_size),
      u(domain_size), d(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  // Solve the equations
  tridStridedBatchWrapper<Float>(params, a.data(), b.data(), c.data(), d.data(),
                                 nullptr, mesh.dims().size(), mesh.solve_dim(),
                                 local_sizes.data(), local_sizes.data());

  // Check result
  require_allclose(u, d, domain_size, 1);
}

template <typename Float, int INC, MpiSolverParams::MPICommStrategy strategy>
void test_solver_from_file_padded(const std::string &file_name) {
  assert(INC == 0 && "Increment testing not implemented");
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size()), periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data(), 32,
                         strategy);

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load our data
  AlignedArray<Float, 1> a(domain_size), b(domain_size), c(domain_size),
      u(domain_size), d(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  std::vector<int> padded_dims = local_sizes;
  int padded_size              = 1;
  for (int i = 0; i < padded_dims.size(); i++) {
    padded_dims[i] += 2;
    padded_size *= padded_dims[i];
  }

  std::vector<Float> a_p(padded_size);
  std::vector<Float> b_p(padded_size);
  std::vector<Float> c_p(padded_size);
  std::vector<Float> d_p(padded_size);
  std::vector<Float> u_p(padded_size);

  copy_to_padded_array(a, a_p, local_sizes);
  copy_to_padded_array(b, b_p, local_sizes);
  copy_to_padded_array(c, c_p, local_sizes);
  copy_to_padded_array(d, d_p, local_sizes);
  copy_to_padded_array(u, u_p, local_sizes);

  int offset_to_first_element =
      padded_dims[1] * padded_dims[0] + padded_dims[0] + 1;

  // Solve the equations
  tridStridedBatchWrapper<Float>(params, a_p.data() + offset_to_first_element,
                                 b_p.data() + offset_to_first_element,
                                 c_p.data() + offset_to_first_element,
                                 d_p.data() + offset_to_first_element, nullptr,
                                 mesh.dims().size(), mesh.solve_dim(),
                                 local_sizes.data(), padded_dims.data());

  // Check result
  require_allclose_padded(u_p, d_p);
}

enum ResDest { assign = 0, increment };

#define PARAM_COMBOS                                                           \
  (double, assign, MpiSolverParams::GATHER_SCATTER),                           \
      (double, assign, MpiSolverParams::ALLGATHER),                            \
      (double, assign, MpiSolverParams::JACOBI),                               \
      (double, assign, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),           \
      (double, assign, MpiSolverParams::LATENCY_HIDING_TWO_STEP),              \
      (float, assign, MpiSolverParams::GATHER_SCATTER),                        \
      (float, assign, MpiSolverParams::ALLGATHER),                             \
      (float, assign, MpiSolverParams::JACOBI),                                \
      (float, assign, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),            \
      (float, assign, MpiSolverParams::LATENCY_HIDING_TWO_STEP)


TEMPLATE_TEST_CASE_SIG("mpi: solver small", "[small]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 1") {
    test_solver_from_file<TestType, INC, strategy>("files/one_dim_small");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/two_dim_small_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/two_dim_small_solve1");
    }
  }
}

TEMPLATE_TEST_CASE_SIG("mpi: solver large", "[large]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 1") {
    test_solver_from_file<TestType, INC, strategy>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/three_dim_large_solve2");
    }
  }
}

TEMPLATE_TEST_CASE_SIG("mpi: solver large padded", "[large][padded]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/three_dim_large_solve2");
    }
  }
}

#if MAXDIM > 3
TEMPLATE_TEST_CASE_SIG("mpi 4D: solver large", "[large]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 4") {
    SECTION("solvedim: 0") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/four_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/four_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/four_dim_large_solve2");
    }
    SECTION("solvedim: 3") {
      test_solver_from_file<TestType, INC, strategy>(
          "files/four_dim_large_solve3");
    }
  }
}
#endif
