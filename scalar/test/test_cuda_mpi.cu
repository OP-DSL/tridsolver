#define CATCH_CONFIG_NOSTDOUT
#include "catch.hpp"
#include "catch_mpi_outputs.hpp"
#include "cuda_utils.hpp"
#include "cuda_mpi_wrappers.hpp"

#include <trid_common.h>
#include <trid_cuda.h>
#include <trid_mpi_cuda.hpp>


#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
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
void require_allclose(const Float *expected, const Float *actual, size_t N,
                      int stride = 1, std::string value = "") {
  for (size_t j = 0, i = 0; j < N; ++j, i += stride) {
    CAPTURE(value);
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

template <typename Float, bool INC = false>
void test_solver_from_file(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size(), 0),
      periods(mesh.dims().size(), 0);
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i] = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i] = params.mpi_coords[i] == mpi_dims[i] - 1
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

  GPUMesh<Float> local_device_mesh(a, b, c, d, local_sizes);

  // Solve the equations
  std::vector<Float> host_init(domain_size, 0);
  DeviceArray<Float> u_d(host_init.data(), domain_size);
  tridmtsvStridedBatchMPIWrapper<Float, INC>(
      params, local_device_mesh.a().data(), local_device_mesh.b().data(),
      local_device_mesh.c().data(), local_device_mesh.d().data(), u_d.data(),
      mesh.dims().size(), mesh.solve_dim(), local_sizes.data(),
      local_sizes.data());
  
  if (!INC) {
    cudaMemcpy(d.data(), local_device_mesh.d().data(),
               sizeof(Float) * domain_size, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(d.data(), u_d.data(), sizeof(Float) * domain_size,
               cudaMemcpyDeviceToHost);
  }
  // Check result
  require_allclose(u, d, domain_size, 1);
}

TEMPLATE_TEST_CASE("cuda solver mpi: solveX", "[solver][NOINC][solvedim:0]",
                   double, float) {
  SECTION("ndims: 1") {
    test_solver_from_file<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    test_solver_from_file<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi: solveY", "[solver][NOINC][solvedim:1]",
                   double, float) {
  SECTION("ndims: 2") {
    test_solver_from_file<TestType>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi: solveZ", "[solver][NOINC][solvedim:2]",
                   double, float) {
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve2");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc: solveX", "[solver][INC][solvedim:0]",
                   double, float) {
  SECTION("ndims: 1") {
    test_solver_from_file<TestType, true>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    test_solver_from_file<TestType, true>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType, true>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc: solveY", "[solver][INC][solvedim:1]",
                   double, float) {
  SECTION("ndims: 2") {
    test_solver_from_file<TestType, true>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType, true>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc: solveZ", "[solver][INC][solvedim:2]",
                   double, float) {
  SECTION("ndims: 3") {
    test_solver_from_file<TestType, true>("files/three_dim_large_solve2");
  }
}

#if MAXDIM > 3
TEMPLATE_TEST_CASE("cuda solver mpi 4D: solveX", "[solver][NOINC][solvedim:0]",
                   double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType>("files/four_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi 4D: solveY", "[solver][NOINC][solvedim:1]",
                   double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType>("files/four_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi 4D: solveZ", "[solver][NOINC][solvedim:2]",
                   double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType>("files/four_dim_large_solve2");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc 4D: solveX",
                   "[solver][INC][solvedim:0]", double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, true>("files/four_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc 4D: solveY",
                   "[solver][INC][solvedim:1]", double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, true>("files/four_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc 4D: solveZ",
                   "[solver][INC][solvedim:2]", double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, true>("files/four_dim_large_solve2");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi inc: solve3", "[solver][INC][solvedim:3]",
                   double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, true>("files/four_dim_large_solve3");
  }
}

TEMPLATE_TEST_CASE("cuda solver mpi: solve3", "[solver][NOINC][solvedim:3]",
                   double, float) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType>("files/four_dim_large_solve3");
  }
}
#endif
