#include "catch.hpp"
#include "utils.hpp"

#include <trid_cpu.h>

template <typename Float>
void require_allclose(const std::vector<Float> &expected,
                      const std::vector<Float> &actual) {
  assert(expected.size() == actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
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

// TODO something like this really should be in the API
template <typename Float>
tridStatus_t tridStridedBatchWrapper(const Float *a, const Float *b,
                                     const Float *c, Float *d, Float *u,
                                     int ndim, int solvedim, int *dims,
                                     int *pads);

template <>
tridStatus_t tridStridedBatchWrapper<float>(const float *a, const float *b,
                                            const float *c, float *d, float *u,
                                            int ndim, int solvedim, int *dims,
                                            int *pads) {
  return tridSmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const double *a, const double *b,
                                             const double *c, double *d,
                                             double *u, int ndim, int solvedim,
                                             int *dims, int *pads) {
  return tridDmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads);
}

template <typename Float> void test_from_file(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<Float> d(mesh.d().begin(), mesh.d().end());
  std::vector<int> dims = mesh.dims(); // Because it isn't const in the lib
  // Fix num_dims workaround
  while (dims.size() < 3) {
    dims.push_back(1);
  }

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(mesh.a().data(),    // a
                                     mesh.b().data(),    // b
                                     mesh.c().data(),    // c
                                     d.data(),           // d
                                     nullptr,            // u
                                     mesh.dims().size(), // ndim
                                     mesh.solve_dim(),   // solvedim
                                     dims.data(),        // dims
                                     dims.data());       // pads

  CHECK(status == TRID_STATUS_SUCCESS);
  require_allclose(mesh.u(), d);
}

TEST_CASE("cpu: one dimension small") {
  SECTION("double") { test_from_file<double>("files/one_dim_small"); }
  SECTION("float") { test_from_file<float>("files/one_dim_small"); }
}

TEST_CASE("cpu: one dimension large") {
  SECTION("double") { test_from_file<double>("files/one_dim_large"); }
  SECTION("float") { test_from_file<float>("files/one_dim_large"); }
}

TEST_CASE("cpu: two dimensions small") {
  SECTION("double") {
    SECTION("solvedim=0") {
      test_from_file<double>("files/two_dim_small_solve0");
    }
    SECTION("solvedim=1") {
      test_from_file<double>("files/two_dim_small_solve1");
    }
  }
  SECTION("float") {
    SECTION("solvedim=0") {
      test_from_file<float>("files/two_dim_small_solve0");
    }
    SECTION("solvedim=1") {
      test_from_file<float>("files/two_dim_small_solve1");
    }
  }
}

TEST_CASE("cpu: two dimensions large") {
  SECTION("double") {
    SECTION("solvedim=0") {
      test_from_file<double>("files/two_dim_large_solve0");
    }
    SECTION("solvedim=1") {
      test_from_file<double>("files/two_dim_large_solve1");
    }
  }
  SECTION("float") {
    SECTION("solvedim=0") {
      test_from_file<float>("files/two_dim_large_solve0");
    }
    SECTION("solvedim=1") {
      test_from_file<float>("files/two_dim_large_solve1");
    }
  }
}
