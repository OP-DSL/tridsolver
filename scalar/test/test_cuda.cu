#include "catch.hpp"
#include "cuda_utils.hpp"
#include "catch_utils.hpp"

#include <trid_cuda.h>

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
  int opts[] = {0, 0, 0};
  return tridSmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads, opts,
                               0);
}

template <>
tridStatus_t tridStridedBatchWrapper<double>(const double *a, const double *b,
                                             const double *c, double *d,
                                             double *u, int ndim, int solvedim,
                                             int *dims, int *pads) {
  int opts[] = {0, 0, 0};
  return tridDmtsvStridedBatch(a, b, c, d, u, ndim, solvedim, dims, pads, opts,
                               0);
}

template <typename Float> void test_from_file(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<int> dims = mesh.dims(); // Because it isn't const in the lib
  while (dims.size() < 3) {
    dims.push_back(1);
  }
  GPUMesh<Float> device_mesh(mesh);

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(device_mesh.a().data(), // a
                                     device_mesh.b().data(), // b
                                     device_mesh.c().data(), // c
                                     device_mesh.d().data(), // d
                                     nullptr,                // u
                                     mesh.dims().size(),     // ndim
                                     mesh.solve_dim(),       // solvedim
                                     dims.data(),            // dims
                                     dims.data());           // pads

  CHECK(status == TRID_STATUS_SUCCESS);

  AlignedArray<Float, 1> d(mesh.d());
  cudaMemcpy(d.data(), device_mesh.d().data(), d.size() * sizeof(Float),
             cudaMemcpyDeviceToHost);
  require_allclose(mesh.u(), d);
}

template <typename Float>
void test_from_file_padded(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  std::vector<int> dims = mesh.dims(); // Because it isn't const in the lib
  while (dims.size() < 3) {
    dims.push_back(1);
  }

  std::vector<int> padded_dims = dims;
  int padded_size              = 1;
  for (int i = 0; i < padded_dims.size(); i++) {
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

  Float *a_d, *b_d, *c_d, *d_d;
  cudaMalloc((void **)&a_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&b_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&c_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&d_d, padded_size * sizeof(Float));

  cudaMemcpy(a_d, a.data(), a.size() * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b.data(), b.size() * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c.data(), c.size() * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d.data(), d.size() * sizeof(Float), cudaMemcpyHostToDevice);

  int offset_to_first_element =
      padded_dims[1] * padded_dims[0] + padded_dims[0] + 1;

  const tridStatus_t status =
      tridStridedBatchWrapper<Float>(a_d + offset_to_first_element, // a
                                     b_d + offset_to_first_element, // b
                                     c_d + offset_to_first_element, // c
                                     d_d + offset_to_first_element, // d
                                     nullptr,                       // u
                                     mesh.dims().size(),            // ndim
                                     mesh.solve_dim(),              // solvedim
                                     dims.data(),                   // dims
                                     padded_dims.data());           // pads

  CHECK(status == TRID_STATUS_SUCCESS);

  cudaMemcpy(d.data(), d_d, d.size() * sizeof(Float), cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  require_allclose_padded(u, d);
}

TEMPLATE_TEST_CASE("cuda: solveX", "[solvedim:0]", double, float) {
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("cuda: solveY", "[solvedim:1]", double, float) {
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("cuda: solveZ", "[solvedim:2]", double, float) {
  SECTION("ndims: 3") {
    SECTION("solvedim: 2") {
      test_from_file<TestType>("files/three_dim_large_solve2");
    }
  }
}

TEMPLATE_TEST_CASE("cuda: padded", "[padded]", double, float) {
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

#if MAXDIM > 3
TEMPLATE_TEST_CASE("cuda 4D: solveX", "[solvedim:0]", double, float) {
  SECTION("ndims: 4") {
    test_from_file<TestType>("files/four_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("cuda 4D: solveY", "[solvedim:1]", double, float) {
  SECTION("ndims: 4") {
    test_from_file<TestType>("files/four_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("cuda 4D: solveZ", "[solvedim:2]", double, float) {
  SECTION("ndims: 4") {
    SECTION("solvedim: 2") {
      test_from_file<TestType>("files/four_dim_large_solve2");
    }
  }
}

TEMPLATE_TEST_CASE("cuda: solve3", "[solvedim:3]", double, float) {
  SECTION("ndims: 4") {
    SECTION("solvedim: 3") {
      test_from_file<TestType>("files/four_dim_large_solve3");
    }
  }
}
#endif
