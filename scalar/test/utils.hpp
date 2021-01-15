#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

constexpr double ABS_TOLERANCE       = 1e-11;
constexpr double REL_TOLERANCE       = 1e-11;
constexpr double ABS_TOLERANCE_FLOAT = 1e-6;
constexpr double REL_TOLERANCE_FLOAT = 1e-5;

template <typename T, unsigned Align> class AlignedArray {
  char *padded_data;
  size_t _size, _capacity;
  unsigned padding;

public:
  AlignedArray(size_t capacity);
  AlignedArray();
  ~AlignedArray();
  AlignedArray(const AlignedArray &other);
  AlignedArray(const AlignedArray &other, size_t start, size_t end);
  AlignedArray(AlignedArray &&);
  AlignedArray &operator=(AlignedArray);

  void push_back(T value);
  /**
   * Only when not allocated in ctor
   */
  void allocate(size_t capacity);
  /**
   * Only when size is 0
   */
  void resize(size_t size, T default_val = T());

  T *data() { return reinterpret_cast<T *>(padded_data + padding); }

  const T *data() const {
    return reinterpret_cast<const T *>(padded_data + padding);
  }

  T &operator[](size_t index) {
    assert(index < _size);
    return data()[index];
  }

  const T &operator[](size_t index) const {
    assert(index < _size);
    return data()[index];
  }

  size_t size() const { return _size; }

  size_t capacity() const { return _capacity; }

};

template <typename Float, unsigned Align = 1> class MeshLoader {
  size_t _solve_dim;
  std::vector<int> _dims;
  AlignedArray<Float, Align> _a, _b, _c, _d, _u;

public:
  MeshLoader(const std::string &file_name);

  size_t solve_dim() const { return _solve_dim; }
  const std::vector<int> &dims() const { return _dims; }
  const AlignedArray<Float, Align> &a() const { return _a; }
  const AlignedArray<Float, Align> &b() const { return _b; }
  const AlignedArray<Float, Align> &c() const { return _c; }
  const AlignedArray<Float, Align> &d() const { return _d; }
  const AlignedArray<Float, Align> &u() const { return _u; }

private:
  void load_array(std::ifstream &f, size_t num_elements,
                  AlignedArray<Float, Align> &array);
};

template <typename Float, unsigned Align = 1> class RandomMesh {
  size_t _solve_dim;
  std::vector<int> _dims;
  AlignedArray<Float, Align> _a, _b, _c, _d;

public:
  RandomMesh(const std::vector<int> dims, size_t solvedim);

  size_t solve_dim() const { return _solve_dim; }
  const std::vector<int> &dims() const { return _dims; }
  const AlignedArray<Float, Align> &a() const { return _a; }
  const AlignedArray<Float, Align> &b() const { return _b; }
  const AlignedArray<Float, Align> &c() const { return _c; }
  const AlignedArray<Float, Align> &d() const { return _d; }
};
/**********************************************************************
 *                          Implementations                           *
 **********************************************************************/

template <typename T, unsigned Align>
AlignedArray<T, Align>::AlignedArray(size_t capacity)
    : padded_data{nullptr}, _size{0}, _capacity{0} {
  allocate(capacity);
}

template <typename T, unsigned Align>
AlignedArray<T, Align>::AlignedArray()
    : padded_data{nullptr}, _size{0}, _capacity{0}, padding{0} {}

template <typename T, unsigned Align> AlignedArray<T, Align>::~AlignedArray() {
  delete[] padded_data;
}

template <typename T, unsigned Align>
AlignedArray<T, Align>::AlignedArray(const AlignedArray &other)
    : padded_data{nullptr}, _size{0}, _capacity{0} {
  allocate(other._capacity);
  std::copy(other.data(), other.data() + other._size, this->data());
  _size = other._size;
}

template <typename T, unsigned Align>
AlignedArray<T, Align>::AlignedArray(const AlignedArray &other, size_t start,
                                     size_t end)
    : padded_data{nullptr}, _size{0}, _capacity{0} {
  allocate(end - start);
  std::copy(other.data() + start, other.data() + end, this->data());
  _size = end - start;
}

template <typename T, unsigned Align>
AlignedArray<T, Align> &AlignedArray<T, Align>::operator=(AlignedArray rhs) {
  std::swap(this->_capacity, rhs._capacity);
  std::swap(this->_size, rhs._size);
  std::swap(this->padded_data, rhs.padded_data);
  std::swap(this->padding, rhs.padding);
  return *this;
}

template <typename T, unsigned Align>
AlignedArray<T, Align>::AlignedArray(AlignedArray &&other) : AlignedArray{} {
  std::swap(this->_capacity, other._capacity);
  std::swap(this->_size, other._size);
  std::swap(this->padded_data, other.padded_data);
  std::swap(this->padding, other.padding);
}

template <typename T, unsigned Align>
void AlignedArray<T, Align>::allocate(size_t capacity) {
  assert(_capacity == 0 && "Array has already been allocated");
  assert(padded_data == nullptr && "Array has already been allocated");
  _capacity                       = capacity;
  _size                           = 0;
  constexpr unsigned ELEMENT_SIZE = sizeof(T) / sizeof(char);
  padded_data                     = new char[capacity * ELEMENT_SIZE + Align];
  const size_t ptr                = reinterpret_cast<size_t>(padded_data);
  if (ptr % Align == 0) {
    padding = 0;
  } else {
    padding = Align - (ptr % Align);
  }
}

template <typename T, unsigned Align>
void AlignedArray<T, Align>::resize(size_t size, T default_val) {
  if (_capacity == 0) allocate(size);
  assert(_size == 0 && "Array has already been initialised");
  for (int i = 0; i < size; ++i) {
    this->push_back(default_val);
  }
}

template <typename T, unsigned Align>
void AlignedArray<T, Align>::push_back(T value) {
  assert(_size < _capacity && "The array is full");
  constexpr unsigned ELEMENT_SIZE = sizeof(T) / sizeof(char);
  T *new_data_ptr =
      reinterpret_cast<T *>(padded_data + padding + ELEMENT_SIZE * _size);
  *new_data_ptr = value;
  ++_size;
}

template <typename Float, unsigned Align>
MeshLoader<Float, Align>::MeshLoader(const std::string &file_name)
    : _a{}, _b{}, _c{}, _d{}, _u{} {
  std::ifstream f(file_name);
  assert(f.good() && "Couldn't open file");
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
  if (std::is_same<Float, double>::value) {
    load_array(f, num_elements, _u);
  } else {
    std::string tmp;
    // Skip the line with the double values
    std::getline(f >> std::ws, tmp);
    load_array(f, num_elements, _u);
  }
}

template <typename Float, unsigned Align>
void MeshLoader<Float, Align>::load_array(std::ifstream &f, size_t num_elements,
                                          AlignedArray<Float, Align> &array) {
  array.allocate(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    // Load with the larger precision, then convert to the specified type
    double value;
    f >> value;
    array.push_back(value);
  }
}

template <typename Float, unsigned Align>
RandomMesh<Float, Align>::RandomMesh(const std::vector<int> dims,
                                     size_t solvedim)
    : _solve_dim(solvedim), _dims(dims), _a{}, _b{}, _c{}, _d{} {
  assert(_solve_dim < _dims.size() && "solve dim greater than number of dims");

  size_t num_elements =
      std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<int>());
  _a.resize(num_elements);
  _b.resize(num_elements);
  _c.resize(num_elements);
  _d.resize(num_elements);
#pragma omp parallel
  {
    std::mt19937 gen(omp_get_thread_num());
    std::uniform_real_distribution<Float> dist;
#pragma omp for
    for (size_t i = 0; i < num_elements; i++)
      _a[i] = -1 + 0.1 * dist(gen);
#pragma omp for
    for (size_t i = 0; i < num_elements; i++)
      _b[i] = 2 + dist(gen);
#pragma omp for
    for (size_t i = 0; i < num_elements; i++)
      _c[i] = -1 + 0.1 * dist(gen);
#pragma omp for
    for (size_t i = 0; i < num_elements; i++)
      _d[i] = dist(gen);
  }
}

#endif /* end of include guard: __UTILS_HPP */
