#define CATCH_CONFIG_NOSTDOUT
#include "catch.hpp"
#include "catch_mpi_outputs.hpp"
#include "utils.hpp"

#include <trid_mpi_cpu.h>
#include "cpu_mpi_wrappers.hpp"

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

// Print routine for debugging
template <typename Container>
void print_array(const std::string &prompt, const Container &array) {
  Catch::cout() << prompt << ": [";
  for (size_t i = 0; i < array.size(); ++i) {
    Catch::cout() << (i == 0 ? "" : ", ") << std::setprecision(2) << array[i];
  }
  Catch::cout() << "]\n";
}

template <typename Float, unsigned Align>
void require_allclose(const AlignedArray<Float, Align> &expected,
                      const AlignedArray<Float, Align> &actual, size_t N = 0,
                      int stride = 1) {
  if (N == 0) {
    assert(expected.size() == actual.size());
    N = expected.size();
  }
  for (size_t j = 0, i = 0; j < N; ++j, i += stride) {
    CAPTURE(i);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double abs_tolerance =
        std::is_same<Float, float>::value ? ABS_TOLERANCE_FLOAT : ABS_TOLERANCE;
    const double rel_tolerance =
        std::is_same<Float, float>::value ? REL_TOLERANCE_FLOAT : REL_TOLERANCE;
    const double tolerance = abs_tolerance + rel_tolerance * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(expected[i]) - actual[i]);
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}

template <typename Float>
void require_allclose_padded(const std::vector<Float> &expected,
                           const std::vector<Float> &actual, size_t N = 0,
                           int stride = 1) {
  if (N == 0) {
    assert(expected.size() == actual.size());
    N = expected.size();
  }

  for (size_t j = 0, i = 0; j < N; ++j, i += stride) {
    CAPTURE(i);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double abs_tolerance =
        std::is_same<Float, float>::value ? ABS_TOLERANCE_FLOAT : ABS_TOLERANCE;
    const double rel_tolerance =
        std::is_same<Float, float>::value ? REL_TOLERANCE_FLOAT : REL_TOLERANCE;
    const double tolerance = abs_tolerance + rel_tolerance * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(expected[i]) - actual[i]);
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}

// Adds 1 depth of padding to all dimensions
template <typename Float, unsigned Align>
void copy_to_padded_array(const AlignedArray<Float, Align> &original,
                        std::vector<Float> &padded,
                        std::vector<int> &dims) {
  assert(dims.size() == 3);
  std::vector<int> padded_dims = dims;
  for(int i = 0; i < padded_dims.size(); i++) {
    // -1 and 1 padding
    padded_dims[i] += 2;
  }
  assert(padded.size() == padded_dims[0] * padded_dims[1] * padded_dims[2]);

  for(int z = -1; z < dims[2] + 1; z++) {
    for(int y = -1; y < dims[1] + 1; y++) {
      for(int x = -1; x < dims[0] + 1; x++) {
        int array_index = (z + 1) * padded_dims[1] * padded_dims[0]
                          + (y + 1) * padded_dims[0] + (x + 1);
        if(x == -1 || x == dims[0] || y == -1 || y == dims[1]
           || z == -1 || z == dims[2]) {
          padded[array_index] = 0.0;
        } else {
          int aligned_array_index = z * dims[1] * dims[0] + y * dims[0] + x;
          padded[array_index] = original[aligned_array_index];
        }
      }
    }
  }
}

template <typename Float> struct ToMpiDatatype {};

template <> struct ToMpiDatatype<double> {
  static const MPI_Datatype value; // = MPI_DOUBLE;
};
const MPI_Datatype ToMpiDatatype<double>::value = MPI_DOUBLE;

template <> struct ToMpiDatatype<float> {
  static const MPI_Datatype value; // = MPI_FLOAT;
};
const MPI_Datatype ToMpiDatatype<float>::value = MPI_FLOAT;

// Copies the local domain defined by `local_sizes` and `offsets` from the mesh.
//
// The 0th dimension is the contiguous one. The function is recursive; `dim` is
// current dimension, should equal one less than the number of dimensions when
// called from outside.
//
// `global_strides` is the product of the all global sizes in the lower
// dimensions (e.g. `global_strides[0] == 1`).
template <typename Float, unsigned Alignment>
void copy_strided(const AlignedArray<Float, Alignment> &src,
                  AlignedArray<Float, Alignment> &dest,
                  const std::vector<int> &local_sizes,
                  const std::vector<int> &offsets,
                  const std::vector<int> &global_strides, size_t dim,
                  int global_offset = 0) {
  if (dim == 0) {
    for (int i = 0; i < local_sizes[dim]; ++i) {
      dest.push_back(src[global_offset + offsets[dim] + i]);
    }
  } else {
    for (int i = 0; i < local_sizes[dim]; ++i) {
      const int new_global_offset =
          global_offset + (offsets[dim] + i) * global_strides[dim];
      copy_strided(src, dest, local_sizes, offsets, global_strides, dim - 1,
                   new_global_offset);
    }
  }
}

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
  int padded_size = 1;
  for(int i = 0; i < padded_dims.size(); i++) {
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

  int offset_to_first_element = padded_dims[1] * padded_dims[0]
                                + padded_dims[0] + 1;

  // Solve the equations
  tridStridedBatchWrapper<Float>(params, a_p.data() + offset_to_first_element,
                                 b_p.data() + offset_to_first_element,
                                 c_p.data() + offset_to_first_element,
                                 d_p.data() + offset_to_first_element,
                                 nullptr, mesh.dims().size(), mesh.solve_dim(),
                                 local_sizes.data(), padded_dims.data());

  // Check result
  require_allclose_padded(u_p, d_p);
}

enum ResDest { assign = 0, increment };

#define PARAM_COMBOS                                                           \
  (double, assign, MpiSolverParams::GATHER_SCATTER),                           \
      (double, assign, MpiSolverParams::ALLGATHER),                            \
      (double, assign, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),           \
      (double, assign, MpiSolverParams::LATENCY_HIDING_TWO_STEP),              \
      (float, assign, MpiSolverParams::GATHER_SCATTER),                        \
      (float, assign, MpiSolverParams::ALLGATHER),                             \
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
