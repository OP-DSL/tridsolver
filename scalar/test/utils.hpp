#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

constexpr double ABS_TOLERANCE = 1e-11;
constexpr double REL_TOLERANCE = 1e-11;
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
  AlignedArray(AlignedArray &&);
  AlignedArray &operator=(AlignedArray);

  void push_back(T value);
  /**
   * Only when not allocated in ctor
   */
  void allocate(size_t capacity);

  T *data() { return reinterpret_cast<T *>(padded_data + padding); }
  const T *data() const {
    return reinterpret_cast<const T *>(padded_data + padding);
  }

  T &operator[](size_t index) { return data()[index]; }
  const T &operator[](size_t index) const { return data()[index]; }

  size_t size() const { return _size; }

  size_t capacity() const { return _capacity; }

  friend void std::swap(AlignedArray<T, Align>, AlignedArray<T, Align>);
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
AlignedArray<T, Align> &AlignedArray<T, Align>::operator=(AlignedArray rhs) {
  std::swap(*this, rhs);
  return *this;
}

template <typename T, unsigned Align>
AlignedArray<T, Align>::AlignedArray(AlignedArray &&other) : AlignedArray{} {
  std::swap(*this, other);
}

namespace std {
template <typename T, unsigned Align>
void swap(AlignedArray<T, Align> lhs, AlignedArray<T, Align> rhs) {
  std::swap(lhs._capacity, rhs._capacity);
  std::swap(lhs._size, rhs._size);
  std::swap(lhs.padded_data, rhs.padded_data);
  std::swap(lhs.padding, rhs.padding);
}
} // namespace std

template <typename T, unsigned Align>
void AlignedArray<T, Align>::allocate(size_t capacity) {
  assert(_capacity == 0 && "Array has already been allocated");
  assert(padded_data == nullptr && "Array has already been allocated");
  _capacity = capacity;
  _size = 0;
  constexpr unsigned ELEMENT_SIZE = sizeof(T) / sizeof(char);
  padded_data = new char[capacity * ELEMENT_SIZE + Align];
  const size_t ptr = reinterpret_cast<size_t>(padded_data);
  if (ptr % Align == 0) {
    padding = 0;
  } else {
    padding = Align - (ptr % Align);
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

#endif /* end of include guard: __UTILS_HPP */
