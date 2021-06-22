#ifndef CUDA_TRIDSOLV_TEST_UTILS_HPP
#define CUDA_TRIDSOLV_TEST_UTILS_HPP

#include "utils.hpp"

template <typename T> class DeviceArray {
  size_t _size;
  T *arr_d;

public:
  DeviceArray();
  DeviceArray(size_t size);
  DeviceArray(const T *host_arr, size_t size);
  DeviceArray(const std::vector<T> &host_arr);
  DeviceArray(const DeviceArray &);
  DeviceArray &operator=(DeviceArray);
  DeviceArray(DeviceArray &&);
  ~DeviceArray();

  void allocate(size_t);
  size_t size() const { return _size; }
  T *data() { return arr_d; }
  const T *data() const { return arr_d; }
};

template <typename Float> class GPUMesh {
  std::vector<int> _dims;
  DeviceArray<Float> _a, _b, _c, _d;

public:
  GPUMesh(const MeshLoader<Float> &mesh);
  GPUMesh(const std::vector<Float> &a, const std::vector<Float> &b,
          const std::vector<Float> &c, const std::vector<Float> &d,
          const std::vector<int> dims);

  const std::vector<int> &dims() const { return _dims; }
  const DeviceArray<Float> &a() const { return _a; }
  const DeviceArray<Float> &b() const { return _b; }
  const DeviceArray<Float> &c() const { return _c; }
  DeviceArray<Float> &d() { return _d; }
};

/**********************************************************************
 *                          Implementations                           *
 **********************************************************************/

template <typename T>
DeviceArray<T>::DeviceArray() : _size(0), arr_d(nullptr) {}
template <typename T>
DeviceArray<T>::DeviceArray(size_t size) : _size(size), arr_d(nullptr) {
  cudaMalloc((void **)&arr_d, sizeof(T) * _size);
}

template <typename T>
DeviceArray<T>::DeviceArray(const T *host_arr, size_t size)
    : DeviceArray(size) {
  cudaMemcpy(arr_d, host_arr, sizeof(T) * _size, cudaMemcpyHostToDevice);
}

template <typename T>
DeviceArray<T>::DeviceArray(const std::vector<T> &host_arr)
    : DeviceArray(host_arr.data(), host_arr.size()) {}

template <typename T>
DeviceArray<T>::DeviceArray(const DeviceArray &other)
    : DeviceArray(other.size()) {
  cudaMemcpy(arr_d, other.data(), sizeof(T) * _size, cudaMemcpyDeviceToDevice);
}

template <typename T>
DeviceArray<T>::DeviceArray(DeviceArray &&other) : _size(0), arr_d(nullptr) {
  std::swap(this->arr_d, other.arr_d);
  std::swap(this->_size, other._size);
}

template <typename T>
DeviceArray<T> &DeviceArray<T>::operator=(DeviceArray<T> other) {
  std::swap(this->_size, other._size);
  std::swap(this->arr_d, other.arr_d);
}
template <typename T> DeviceArray<T>::~DeviceArray() {
  if (arr_d) cudaFree(arr_d);
}

template <typename T> void DeviceArray<T>::allocate(size_t N) {
  assert(_size == 0 && "Array has been already allocated. Allocate must called "
                       "on a device array with size 0");
  assert(arr_d == nullptr && "Array has been already allocated.");
  cudaMalloc((void **)&arr_d, sizeof(T) * N);
  _size = N;
}

template <typename Float>
GPUMesh<Float>::GPUMesh(const MeshLoader<Float> &mesh)
    : GPUMesh<Float>(mesh.a(), mesh.b(), mesh.c(), mesh.d(), mesh.dims()) {}

template <typename Float>
GPUMesh<Float>::GPUMesh(const std::vector<Float> &a,
                        const std::vector<Float> &b,
                        const std::vector<Float> &c,
                        const std::vector<Float> &d,
                        const std::vector<int> dims)
    : _a(a), _b(b), _c(c), _d(d), _dims(dims) {}

#endif /* ifndef CUDA_TRIDSOLV_TEST_UTILS_HPP */
