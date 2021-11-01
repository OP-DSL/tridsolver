#include "catch.hpp"
#include "utils.hpp"
#include "catch_utils.hpp"

#include <tridsolver.h>
#include <trid_cpu.h>
#include <trid_simd.h>

#include <memory>

template <typename Float>
tridStatus_t tridStridedBatchWrapper(const Float *a, const Float *b,
                                     const Float *c, Float *d, int ndim,
                                     int solvedim, const int *dims,
                                     const int *pads);

template <>
tridStatus_t tridStridedBatchWrapper<float>(const float *a, const float *b,
                                            const float *c, float *d, int ndim,
                                            int solvedim, const int *dims,
                                            const int *pads) {
  return tridSmtsvStridedBatch(nullptr, a, b, c, d, ndim, solvedim, dims, pads);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const double *a, const double *b,
                                             const double *c, double *d,
                                             int ndim, int solvedim,
                                             const int *dims, const int *pads) {
  return tridDmtsvStridedBatch(nullptr, a, b, c, d, ndim, solvedim, dims, pads);
}

template <typename Float>
void trid_scalar_wrapper(const Float *a, const Float *b, const Float *c,
                         Float *d, int N, int stride);

template <>
void trid_scalar_wrapper<float>(const float *a, const float *b, const float *c,
                                float *d, int N, int stride) {
  trid_scalarS(a, b, c, d, N, stride);
}

template <>
void trid_scalar_wrapper<double>(const double *a, const double *b,
                                 const double *c, double *d, int N,
                                 int stride) {
  trid_scalarD(a, b, c, d, N, stride);
}

template <typename Float>
void trid_scalar_vec_wrapper(const Float *a, const Float *b, const Float *c,
                             Float *d, int N, int stride);

template <>
void trid_scalar_vec_wrapper<float>(const float *a, const float *b,
                                    const float *c, float *d, int N,
                                    int stride) {
  trid_scalar_vecS(a, b, c, d, N, stride);
}

template <>
void trid_scalar_vec_wrapper<double>(const double *a, const double *b,
                                     const double *c, double *d, int N,
                                     int stride) {
  trid_scalar_vecD(a, b, c, d, N, stride);
}

template <typename Float> void test_from_file(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<Float> d(mesh.d());

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(mesh.a().data(),     // a
                                     mesh.b().data(),     // b
                                     mesh.c().data(),     // c
                                     d.data(),            // d
                                     mesh.dims().size(),  // ndim
                                     mesh.solve_dim(),    // solvedim
                                     mesh.dims().data(),  // dims
                                     mesh.dims().data()); // pads

  CHECK(status == TRID_STATUS_SUCCESS);
  require_allclose(mesh.u(), d);
}

template <typename Float>
void test_from_file_padded(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);

  std::vector<int> padded_dims = mesh.dims();
  int padded_size              = 1;
  for (size_t i = 0; i < padded_dims.size(); i++) {
    padded_dims[i] += 2;
    padded_size *= padded_dims[i];
  }

  std::vector<Float> a_p(padded_size);
  std::vector<Float> b_p(padded_size);
  std::vector<Float> c_p(padded_size);
  std::vector<Float> d_p(padded_size);
  std::vector<Float> u_p(padded_size);

  copy_to_padded_array(mesh.a(), a_p, mesh.dims());
  copy_to_padded_array(mesh.b(), b_p, mesh.dims());
  copy_to_padded_array(mesh.c(), c_p, mesh.dims());
  copy_to_padded_array(mesh.d(), d_p, mesh.dims());
  copy_to_padded_array(mesh.u(), u_p, mesh.dims());

  int offset_to_first_element = 1;
  for (size_t i = 0; i < padded_dims.size() - 1; ++i) {
    offset_to_first_element +=
        std::accumulate(padded_dims.begin(), padded_dims.begin() + i + 1, 1,
                        std::multiplies<int>());
  }

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(a_p.data() + offset_to_first_element, // a
                                     b_p.data() + offset_to_first_element, // b
                                     c_p.data() + offset_to_first_element, // c
                                     d_p.data() + offset_to_first_element, // d
                                     mesh.dims().size(),  // ndim
                                     mesh.solve_dim(),    // solvedim
                                     mesh.dims().data(),  // dims
                                     padded_dims.data()); // pads

  CHECK(status == TRID_STATUS_SUCCESS);
  require_allclose(u_p, d_p);
}

template <typename Float>
void test_from_file_scalar(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<Float> d(mesh.d());

  int stride = 1;
  for (size_t i = 0; i < mesh.solve_dim(); ++i) {
    stride *= mesh.dims()[i];
  }
  const size_t N = mesh.dims()[mesh.solve_dim()];

  trid_scalar_wrapper<Float>(mesh.a().data(), // a
                             mesh.b().data(), // b
                             mesh.c().data(), // c
                             d.data(),        // d
                             N,               // N
                             stride);         // stride

  require_allclose(mesh.u(), d, N, stride);
}

template <typename Float>
void test_from_file_scalar_vec(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  AlignedArray<Float, SIMD_WIDTH> a(mesh.a());
  AlignedArray<Float, SIMD_WIDTH> b(mesh.b());
  AlignedArray<Float, SIMD_WIDTH> c(mesh.c());
  AlignedArray<Float, SIMD_WIDTH> d(mesh.d());

  int stride = 1;
  for (size_t i = 0; i < mesh.solve_dim(); ++i) {
    stride *= mesh.dims()[i];
  }
  const size_t N = mesh.dims()[mesh.solve_dim()];

  trid_scalar_vec_wrapper<Float>(a.data(), // a
                                 b.data(), // b
                                 c.data(), // c
                                 d.data(), // d
                                 N,        // N
                                 stride);  // stride

  require_allclose(mesh.u().data(), d.data(), N, stride);
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

TEST_CASE("cpu: trid_scalar_vec small", "[small]") {
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

TEST_CASE("cpu: strided batch small padded", "[small][padded]") {
  SECTION("double") {
    SECTION("ndims: 1") {
      test_from_file_padded<double>("files/one_dim_small");
    }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file_padded<double>("files/two_dim_small_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file_padded<double>("files/two_dim_small_solve1");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_from_file_padded<float>("files/one_dim_small"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file_padded<float>("files/two_dim_small_solve0");
      }
      // This won't work because the array size is 8 and we don't test with
      // padding at the end yet
      /* SECTION("solvedim: 1") { */
      /*  test_from_file<float>("files/two_dim_small_solve1"); */
      /* } */
    }
  }
}

TEMPLATE_TEST_CASE("cpu: solver large padded", "[large][padded]", double,
                   float) {
  SECTION("ndims: 1") {
    test_from_file_padded<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file_padded<TestType>("files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_padded<TestType>("files/two_dim_large_solve1");
    }
  }
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
