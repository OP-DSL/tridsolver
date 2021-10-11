#include "catch.hpp"
#include "utils.hpp"
#include "catch_utils.hpp"

#include <trid_cpu.h>
#include <trid_simd.h>

#include <memory>

template <typename Float>
tridStatus_t tridStridedBatchWrapper(const Float *a, const Float *b,
                                     const Float *c, Float *d, Float *u,
                                     int ndim, int solvedim, const int *dims,
                                     const int *pads);

template <>
tridStatus_t tridStridedBatchWrapper<float>(const float *a, const float *b,
                                            const float *c, float *d, float *u,
                                            int ndim, int solvedim,
                                            const int *dims, const int *pads) {
  return tridSmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const double *a, const double *b,
                                             const double *c, double *d,
                                             double *u, int ndim, int solvedim,
                                             const int *dims, const int *pads) {
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

template <typename Float>
void test_from_file_shifted(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<Float> d(mesh.d());
  std::vector<Float> a(mesh.a().begin() + 1, mesh.a().end());
  std::vector<Float> c(mesh.c().begin(),
                       mesh.c().begin() + mesh.c().size() - 1);

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(a.data() - 1,        // a
                                     mesh.b().data(),     // b
                                     c.data(),            // c
                                     d.data(),            // d
                                     nullptr,             // u
                                     mesh.dims().size(),  // ndim
                                     mesh.solve_dim(),    // solvedim
                                     mesh.dims().data(),  // dims
                                     mesh.dims().data()); // pads

  CHECK(status == TRID_STATUS_SUCCESS);
  require_allclose(mesh.u(), d);
}

template <typename Float>
void test_from_file_padded_shifted(const std::string &file_name) {
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

  std::vector<Float> a_p_s(a_p.begin() + 1, a_p.end());
  std::vector<Float> c_p_s(c_p.begin(), c_p.begin() + c_p.size() -
                                            offset_to_first_element - 1);
  const tridStatus_t status = tridStridedBatchWrapper<Float>(
      a_p_s.data() - 1,                       // a
      b_p.data() + offset_to_first_element,   // b
      c_p_s.data() + offset_to_first_element, // c
      d_p.data() + offset_to_first_element,   // d
      nullptr,                                // u
      mesh.dims().size(),                     // ndim
      mesh.solve_dim(),                       // solvedim
      mesh.dims().data(),                     // dims
      padded_dims.data());                    // pads

  CHECK(status == TRID_STATUS_SUCCESS);
  require_allclose(u_p, d_p);
}

TEST_CASE("cpu: strided batch small", "[small]") {
  SECTION("double") {
    SECTION("ndims: 1") {
      test_from_file_shifted<double>("files/one_dim_small");
    }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file_shifted<double>("files/two_dim_small_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file_shifted<double>("files/two_dim_small_solve1");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") {
      test_from_file_shifted<float>("files/one_dim_small");
    }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file_shifted<float>("files/two_dim_small_solve0");
      }
      // This won't work because the array size is 8 and we don't test with
      // padding at the end yet
      /* SECTION("solvedim: 1") { */
      /*   test_from_file_shifted<float>("files/two_dim_small_solve1"); */
      /* } */
    }
  }
}

TEMPLATE_TEST_CASE("cpu: strided batch large", "[large]", double, float) {
  SECTION("ndims: 1") {
    test_from_file_shifted<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file_shifted<TestType>("files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_shifted<TestType>("files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_from_file_shifted<TestType>("files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_shifted<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file_shifted<TestType>("files/three_dim_large_solve2");
    }
  }
}

TEST_CASE("cpu: strided batch small padded", "[small][padded]") {
  SECTION("double") {
    SECTION("ndims: 1") {
      test_from_file_padded_shifted<double>("files/one_dim_small");
    }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file_padded_shifted<double>("files/two_dim_small_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file_padded_shifted<double>("files/two_dim_small_solve1");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") {
      test_from_file_padded_shifted<float>("files/one_dim_small");
    }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file_padded_shifted<float>("files/two_dim_small_solve0");
      }
      // This won't work because the array size is 8 and we don't test with
      // padding at the end yet
      /* SECTION("solvedim: 1") { */
      /*  test_from_file_shifted<float>("files/two_dim_small_solve1"); */
      /* } */
    }
  }
}

TEMPLATE_TEST_CASE("cpu: solver large padded", "[large][padded]", double,
                   float) {
  SECTION("ndims: 1") {
    test_from_file_padded_shifted<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_from_file_padded_shifted<TestType>("files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_padded_shifted<TestType>("files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_from_file_padded_shifted<TestType>("files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_from_file_padded_shifted<TestType>("files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_from_file_padded_shifted<TestType>("files/three_dim_large_solve2");
    }
  }
}
