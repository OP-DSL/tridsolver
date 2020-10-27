#include "catch.hpp"
#include "utils.hpp"

#include <trid_cpu.h>
#include <trid_simd.h>

#include <memory>

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

template <typename Float>
void trid_scalar_wrapper(const Float *a, const Float *b, const Float *c,
                         Float *d, Float *u, int N, int stride);

template <>
void trid_scalar_wrapper<float>(const float *a, const float *b, const float *c,
                                float *d, float *u, int N, int stride) {
  trid_scalarS(a, b, c, d, u, N, stride);
}

template <>
void trid_scalar_wrapper<double>(const double *a, const double *b,
                                 const double *c, double *d, double *u, int N,
                                 int stride) {
  trid_scalarD(a, b, c, d, u, N, stride);
}

template <typename Float>
void trid_scalar_vec_wrapper(const Float *a, const Float *b, const Float *c,
                             Float *d, Float *u, int N, int stride);

template <>
void trid_scalar_vec_wrapper<float>(const float *a, const float *b,
                                    const float *c, float *d, float *u, int N,
                                    int stride) {
  trid_scalar_vecS(a, b, c, d, u, N, stride);
}

template <>
void trid_scalar_vec_wrapper<double>(const double *a, const double *b,
                                     const double *c, double *d, double *u,
                                     int N, int stride) {
  trid_scalar_vecD(a, b, c, d, u, N, stride);
}

template <typename Float> void test_from_file(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  AlignedArray<Float, 1> d(mesh.d());
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

template <typename Float> void test_from_file_padded(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<int> dims = mesh.dims(); // Because it isn't const in the lib
  // Fix num_dims workaround
  while (dims.size() < 3) {
    dims.push_back(1);
  }

  std::vector<int> padded_dims = dims;
  int padded_size = 1;
  for(int i = 0; i < padded_dims.size(); i++) {
    padded_dims[i] += 2;
    padded_size *= padded_dims[i];
  }

  std::vector<Float> a(padded_size);
  std::vector<Float> b(padded_size);
  std::vector<Float> c(padded_size);
  std::vector<Float> d(padded_size);
  std::vector<Float> u(padded_size);

  copy_to_padded_array(mesh.a(), a, dims);
  copy_to_padded_array(mesh.b(), b, dims);
  copy_to_padded_array(mesh.c(), c, dims);
  copy_to_padded_array(mesh.d(), d, dims);
  copy_to_padded_array(mesh.u(), u, dims);

  int offset_to_first_element = padded_dims[1] * padded_dims[0]
                                + padded_dims[0] + 1;

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(a.data() + offset_to_first_element,    // a
                                     b.data() + offset_to_first_element,    // b
                                     c.data() + offset_to_first_element,    // c
                                     d.data() + offset_to_first_element,    // d
                                     nullptr,            // u
                                     mesh.dims().size(), // ndim
                                     mesh.solve_dim(),   // solvedim
                                     dims.data(),        // dims
                                     padded_dims.data());       // pads

  CHECK(status == TRID_STATUS_SUCCESS);
  require_allclose_padded(u, d);
}

template <typename Float>
void test_from_file_scalar(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  AlignedArray<Float, 1> d(mesh.d());

  int stride = 1;
  for (size_t i = 0; i < mesh.solve_dim(); ++i) {
    stride *= mesh.dims()[i];
  }
  const size_t N = mesh.dims()[mesh.solve_dim()];

  trid_scalar_wrapper<Float>(mesh.a().data(), // a
                             mesh.b().data(), // b
                             mesh.c().data(), // c
                             d.data(),        // d
                             nullptr,         // u
                             N,               // N
                             stride);         // stride

  require_allclose(mesh.u(), d, N, stride);
}

template <typename Float>
void test_from_file_scalar_vec(const std::string &file_name) {
  MeshLoader<Float, SIMD_WIDTH> mesh(file_name);
  AlignedArray<Float, SIMD_WIDTH> d(mesh.d());

  int stride = 1;
  for (size_t i = 0; i < mesh.solve_dim(); ++i) {
    stride *= mesh.dims()[i];
  }
  const size_t N = mesh.dims()[mesh.solve_dim()];

  trid_scalar_vec_wrapper<Float>(mesh.a().data(), // a
                                 mesh.b().data(), // b
                                 mesh.c().data(), // c
                                 d.data(),        // d
                                 nullptr,         // u
                                 N,               // N
                                 stride /
                                     (SIMD_WIDTH / sizeof(Float))); // stride

  require_allclose(mesh.u(), d, N, stride);
}

TEST_CASE("cpu: strided batch small", "[small]") {
  SECTION("double") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_small"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file<double>("files/two_dim_small_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file<double>("files/two_dim_small_solve1");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_from_file<float>("files/one_dim_small"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file<float>("files/two_dim_small_solve0");
      }
      // This won't work because the array size is 8 and we don't test with
      // padding at the end yet
      /* SECTION("solvedim: 1") { */
      /*   test_from_file<float>("files/two_dim_small_solve1"); */
      /* } */
    }
  }
}

TEMPLATE_TEST_CASE("cpu: strided batch large", "[large]", double, float) {
  SECTION("ndims: 1") { test_from_file<TestType>("files/one_dim_large"); }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file<TestType>("files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file<TestType>("files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_from_file<TestType>("files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file<TestType>("files/three_dim_large_solve2");
    }
  }
}

TEMPLATE_TEST_CASE("cpu: strided batch large padded", "[large][padded]", double, float) {
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_from_file_padded<TestType>("files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_padded<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file_padded<TestType>("files/three_dim_large_solve2");
    }
  }
}

TEMPLATE_TEST_CASE("cpu: trid_scalar small", "[small]", double, float) {
  SECTION("ndims: 1") {
    test_from_file_scalar<TestType>("files/one_dim_small");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file_scalar<TestType>("files/two_dim_small_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_scalar<TestType>("files/two_dim_small_solve1");
    }
  }
}

TEMPLATE_TEST_CASE("cpu: trid_scalar large", "[large]", double, float) {
  SECTION("ndims: 1") {
    test_from_file_scalar<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file_scalar<TestType>("files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_scalar<TestType>("files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_from_file_scalar<TestType>("files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_scalar<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file_scalar<TestType>("files/three_dim_large_solve2");
    }
  }
}

TEST_CASE("cpu: trid_scalar_vec small") {
  SECTION("double") {
    SECTION("ndims: 2") {
      SECTION("solvedim: 1") {
        test_from_file_scalar_vec<double>("files/two_dim_small_solve1");
      }
    }
  }
  // This won't work because the array size is 8 and we don't test with
  // padding at the end yet
  /* SECTION("float") { */
  /*   SECTION("ndims: 2") { */
  /*     SECTION("solvedim: 1") { */
  /*       test_from_file_scalar_vec<float>("files/two_dim_small_solve1"); */
  /*     } */
  /*   } */
  /* } */
}

TEMPLATE_TEST_CASE("cpu: trid_scalar_vec large", "[large]", double, float) {
  SECTION("ndims: 2") {
    SECTION("solvedim: 1") {
      test_from_file_scalar_vec<TestType>("files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 1") {
      test_from_file_scalar_vec<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file_scalar_vec<TestType>("files/three_dim_large_solve2");
    }
  }
}
