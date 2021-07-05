#ifndef CATCH_UTIL_FUNCTIONS_INCLUDED
#define CATCH_UTIL_FUNCTIONS_INCLUDED
#include <functional>
#include <iomanip>

#include "catch.hpp"
#include "catch_mpi_outputs.hpp"
#include "utils.hpp"

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
    CAPTURE(N);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double tolerance =
        abs_tolerance<Float> + rel_tolerance<Float> * min_val;
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
    CAPTURE(N);
    CAPTURE(i);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double tolerance =
        abs_tolerance<Float> + rel_tolerance<Float> * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(expected[i]) - actual[i]);
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}

template <typename Float>
void require_allclose(const std::vector<Float> &expected,
                      const std::vector<Float> &actual, size_t N = 0,
                      int stride = 1) {
  if (N == 0) {
    assert(expected.size() == actual.size());
    N = expected.size();
  }

  for (size_t j = 0, i = 0; j < N; ++j, i += stride) {
    CAPTURE(i);
    CAPTURE(N);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double tolerance =
        abs_tolerance<Float> + rel_tolerance<Float> * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(expected[i]) - actual[i]);
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}

// Adds 1 depth of padding to all dimensions
template <typename Float>
void copy_to_padded_array(const std::vector<Float> &original,
                          std::vector<Float> &padded,
                          const std::vector<int> &dims) {
  assert(dims.size() <= 3);
  std::vector<int> padded_dims = dims;
  for (size_t i = 0; i < padded_dims.size(); i++) {
    // -1 and 1 padding
    padded_dims[i] += 2;
  }
  assert(padded.size() == std::accumulate(padded_dims.begin(),
                                          padded_dims.end(), 1ul,
                                          std::multiplies<size_t>()));

  if (dims.size() == 1ul) {
    for (int x = -1; x < dims[0] + 1; x++) {
      int array_index = x + 1;
      if (x == -1 || x == dims[0]) {
        padded[array_index] = 0.0;
      } else {
        int aligned_array_index = x;
        padded[array_index]     = original[aligned_array_index];
      }
    }
  } else if (dims.size() == 2ul) {
    for (int y = -1; y < dims[1] + 1; y++) {
      for (int x = -1; x < dims[0] + 1; x++) {
        int array_index = (y + 1) * padded_dims[0] + (x + 1);
        if (x == -1 || x == dims[0] || y == -1 || y == dims[1]) {
          padded[array_index] = 0.0;
        } else {
          int aligned_array_index = y * dims[0] + x;
          padded[array_index]     = original[aligned_array_index];
        }
      }
    }
  } else {
    for (int z = -1; z < dims[2] + 1; z++) {
      for (int y = -1; y < dims[1] + 1; y++) {
        for (int x = -1; x < dims[0] + 1; x++) {
          int array_index = (z + 1) * padded_dims[1] * padded_dims[0] +
                            (y + 1) * padded_dims[0] + (x + 1);
          if (x == -1 || x == dims[0] || y == -1 || y == dims[1] || z == -1 ||
              z == dims[2]) {
            padded[array_index] = 0.0;
          } else {
            int aligned_array_index = z * dims[1] * dims[0] + y * dims[0] + x;
            padded[array_index]     = original[aligned_array_index];
          }
        }
      }
    }
  }
}

// Copies the local domain defined by `local_sizes` and `offsets` from the mesh.
//
// The 0th dimension is the contiguous one. The function is recursive; `dim` is
// current dimension, should equal one less than the number of dimensions when
// called from outside.
//
// `global_strides` is the product of the all global sizes in the lower
// dimensions (e.g. `global_strides[0] == 1`).
template <typename Float, int Alignment>
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
template <typename Float>
void copy_strided(const std::vector<Float> &src, std::vector<Float> &dest,
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


template <int MAXDIM = 3>
inline int get_sys_start_idx(int sys_idx, int solvedim, const int *dims,
                             const int *pads, int ndim) {
  static_assert(MAXDIM == 3,
                "Index calculation onlz implemented for at most 3D problems");
  assert(solvedim < ndim && ndim <= MAXDIM);
  int start_pad = pads[0];
  if (solvedim == 1) start_pad *= pads[1];
  int start;
  if (solvedim == 0) {
    if (ndim == 1) {
      start = sys_idx * pads[0];
    } else {
      start = (sys_idx / dims[1]) * pads[1] * pads[0] +
              (sys_idx % dims[1]) * pads[0];
    }
  } else {
    start = (sys_idx / dims[0]) * start_pad + (sys_idx % dims[0]);
  }
  return start;
}

#endif /* ifndef CATCH_UTIL_FUNCTIONS_INCLUDED */
