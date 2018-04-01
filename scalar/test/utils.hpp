#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>

constexpr double ABS_TOLERANCE = 1e-5;
constexpr double REL_TOLERANCE = 1e-5;

template <typename Float> class MeshLoader {
  size_t _solve_dim;
  std::vector<int> _dims;
  std::vector<Float> _a, _b, _c, _d, _u;

public:
  MeshLoader(const std::string &file_name);

  size_t solve_dim() const { return _solve_dim; }
  const std::vector<int> &dims() const { return _dims; }
  const std::vector<Float> &a() const { return _a; }
  const std::vector<Float> &b() const { return _b; }
  const std::vector<Float> &c() const { return _c; }
  const std::vector<Float> &d() const { return _d; }
  const std::vector<Float> &u() const { return _u; }

private:
  void load_array(std::ifstream &f, size_t num_elements,
                  std::vector<Float> &array);
};

/**********************************************************************
*                          Implementations                           *
**********************************************************************/


template <typename Float>
MeshLoader<Float>::MeshLoader(const std::string &file_name) {
  std::ifstream f(file_name);
  size_t num_dims;
  f >> num_dims >> _solve_dim;
  // Load sizes along the different dimensions
  size_t num_elements = 1;
  for (size_t i = 0; i < num_dims; ++i) {
    size_t size;
    f >> size;
    _dims.push_back(size);
    num_elements *= size;
  }
  // Load arrays
  load_array(f, num_elements, _a);
  load_array(f, num_elements, _b);
  load_array(f, num_elements, _c);
  load_array(f, num_elements, _d);
  load_array(f, num_elements, _u);
}

template <typename Float>
void MeshLoader<Float>::load_array(std::ifstream &f, size_t num_elements,
                                   std::vector<Float> &array) {
  for (size_t i = 0; i < num_elements; ++i) {
    Float value;
    f >> value;
    array.push_back(value);
  }
}

#endif /* end of include guard: __UTILS_HPP */
