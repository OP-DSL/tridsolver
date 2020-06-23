#include "catch.hpp"
#include "cuda_utils.hpp"

#include <trid_cuda.h>

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

TEST_CASE("cuda: solveX", "[solvedim:0]") {
  SECTION("double") {
    SECTION("ndims: 2") {
      test_from_file<double>("files/two_dim_large_solve0");
    }
    SECTION("ndims: 3") {
      test_from_file<double>("files/three_dim_large_solve0");
    }
  }
  SECTION("float") {
    SECTION("ndims: 2") { test_from_file<float>("files/two_dim_large_solve0"); }
    SECTION("ndims: 3") {
      test_from_file<float>("files/three_dim_large_solve0");
    }
  }
}

TEST_CASE("cuda: solveY", "[solvedim:1]") {
  SECTION("double") {
    SECTION("ndims: 2") {
      test_from_file<double>("files/two_dim_large_solve1");
    }
    SECTION("ndims: 3") {
      test_from_file<double>("files/three_dim_large_solve1");
    }
  }
  SECTION("float") {
    SECTION("ndims: 2") { test_from_file<float>("files/two_dim_large_solve1"); }
    SECTION("ndims: 3") {
      test_from_file<float>("files/three_dim_large_solve1");
    }
  }
}

TEST_CASE("cuda: solveZ", "[solvedim:2]") {
  SECTION("double") {
    SECTION("ndims: 3") {
      SECTION("solvedim: 2") {
        test_from_file<double>("files/three_dim_large_solve2");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 3") {
      SECTION("solvedim: 2") {
        test_from_file<float>("files/three_dim_large_solve2");
      }
    }
  }
}

#if MAXDIM > 3
TEST_CASE("cuda 4D: solveX", "[solvedim:0]") {
  SECTION("double") {
    SECTION("ndims: 4") {
      test_from_file<double>("files/four_dim_large_solve0");
    }
  }
  SECTION("float") {
    SECTION("ndims: 4") {
      test_from_file<float>("files/four_dim_large_solve0");
    }
  }
}

TEST_CASE("cuda 4D: solveY", "[solvedim:1]") {
  SECTION("double") {
    SECTION("ndims: 4") {
      test_from_file<double>("files/four_dim_large_solve1");
    }
  }
  SECTION("float") {
    SECTION("ndims: 4") {
      test_from_file<float>("files/four_dim_large_solve1");
    }
  }
}

TEST_CASE("cuda 4D: solveZ", "[solvedim:2]") {
  SECTION("double") {
    SECTION("ndims: 4") {
      SECTION("solvedim: 2") {
        test_from_file<double>("files/four_dim_large_solve2");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 4") {
      SECTION("solvedim: 2") {
        test_from_file<float>("files/four_dim_large_solve2");
      }
    }
  }
}

TEST_CASE("cuda: solve3", "[solvedim:3]") {
  SECTION("double") {
    SECTION("ndims: 4") {
      SECTION("solvedim: 3") {
        test_from_file<double>("files/four_dim_large_solve3");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 4") {
      SECTION("solvedim: 3") {
        test_from_file<float>("files/four_dim_large_solve3");
      }
    }
  }
}
#endif
