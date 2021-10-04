#define CATCH_CONFIG_NOSTDOUT
#include "catch.hpp"
#include "catch_mpi_outputs.hpp"
#include "cuda_utils.hpp"
#include "catch_utils.hpp"
#include "cuda_mpi_wrappers.hpp"

#include "trid_cuda_mpi_pcr.hpp"
#include "trid_iterative_mpi.hpp"

#include <trid_common.h>
#include <tridsolver.h>
#include <trid_mpi_solver_params.hpp>


#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

template <typename Float, int INC, MpiSolverParams::MPICommStrategy strategy>
void test_solver_from_file(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size(), 0),
      periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data(), 256,
                         strategy);

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load our data
  std::vector<Float> a, b, c, u, d;
  a.reserve(domain_size);
  b.reserve(domain_size);
  c.reserve(domain_size);
  d.reserve(domain_size);
  u.reserve(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  GPUMesh<Float> local_device_mesh(a, b, c, d, local_sizes);

  // Solve the equations
  std::vector<Float> host_init(domain_size, 0);
  DeviceArray<Float> u_d(host_init.data(), domain_size);
  tridmtsvStridedBatchMPIWrapper<Float, INC>(
      &params, local_device_mesh.a().data(), local_device_mesh.b().data(),
      local_device_mesh.c().data(), local_device_mesh.d().data(), u_d.data(),
      mesh.dims().size(), mesh.solve_dim(), local_sizes.data(),
      local_sizes.data());

  if (!INC) {
    cudaMemcpy(d.data(), local_device_mesh.d().data(),
               sizeof(Float) * domain_size, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(d.data(), u_d.data(), sizeof(Float) * domain_size,
               cudaMemcpyDeviceToHost);
  }
  // Check result
  require_allclose(u, d, domain_size, 1);
}

template <typename Float, int INC, MpiSolverParams::MPICommStrategy strategy>
void test_solver_from_file_padded(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size(), 0),
      periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data(), 256,
                         strategy);

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load our data
  std::vector<Float> a, b, c, u, d;
  a.reserve(domain_size);
  b.reserve(domain_size);
  c.reserve(domain_size);
  d.reserve(domain_size);
  u.reserve(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  std::vector<int> padded_dims = local_sizes;
  int padded_size              = 1;
  for (int i = 0; i < padded_dims.size(); i++) {
    padded_dims[i] += 2;
    padded_size *= padded_dims[i];
  }

  std::vector<Float> a_p(padded_size);
  std::vector<Float> b_p(padded_size);
  std::vector<Float> c_p(padded_size);
  std::vector<Float> d_p(padded_size);
  std::vector<Float> u_p(padded_size);
  std::vector<Float> u_zero(padded_size, 0);

  copy_to_padded_array(a, a_p, local_sizes);
  copy_to_padded_array(b, b_p, local_sizes);
  copy_to_padded_array(c, c_p, local_sizes);
  copy_to_padded_array(d, d_p, local_sizes);
  copy_to_padded_array(u, u_p, local_sizes);

  int offset_to_first_element = 1;
  for (size_t i = 0; i < padded_dims.size() - 1; ++i) {
    offset_to_first_element +=
        std::accumulate(padded_dims.begin(), padded_dims.begin() + i + 1, 1,
                        std::multiplies<int>());
  }

  Float *a_d, *b_d, *c_d, *d_d, *u_d;
  cudaMalloc((void **)&a_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&b_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&c_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&d_d, padded_size * sizeof(Float));
  cudaMalloc((void **)&u_d, padded_size * sizeof(Float));

  cudaMemcpy(a_d, a_p.data(), a_p.size() * sizeof(Float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_p.data(), b_p.size() * sizeof(Float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_p.data(), c_p.size() * sizeof(Float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d_p.data(), d_p.size() * sizeof(Float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(u_d, u_zero.data(), d_p.size() * sizeof(Float),
             cudaMemcpyHostToDevice);

  tridmtsvStridedBatchMPIWrapper<Float, INC>(
      &params, a_d + offset_to_first_element, b_d + offset_to_first_element,
      c_d + offset_to_first_element, d_d + offset_to_first_element,
      u_d + offset_to_first_element, mesh.dims().size(), mesh.solve_dim(),
      local_sizes.data(), padded_dims.data());

  if (!INC) {
    cudaMemcpy(d_p.data(), d_d, sizeof(Float) * d_p.size(),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(d_p.data(), u_d, sizeof(Float) * d_p.size(),
               cudaMemcpyDeviceToHost);
  }
  // Check result
  require_allclose(u_p, d_p);
}

template <typename Float>
void test_PCR_on_reduced(const std::string &file_name) {
  // reduced system:
  //  b is 1 everywhere
  //  consider each 2 element as a result of the forward run for separate mpi
  //  nodes
  //  one input from merged a c and d arrays
  //    layout: [aa0 aa1 cc0 cc1 dd0 dd1 (for each system) aa1 aa2 bb1 bb2 ...]
  //    size: 6 * sys_n * mpi_process_num
  //  one output with the 2 d values per system based on mpi coord
  //    layout: [d_[2*mpi_proc_id] d_[2*mpi_proc_id + 1] ...(for every system)]
  //    size: 2 * sys_n

  MeshLoader<Float> mesh(file_name);
  const int reduced_sys_len = mesh.dims()[mesh.solve_dim()];
  const int num_mpi_procs   = reduced_sys_len / 2;
  const int mpi_coord       = num_mpi_procs / 2;
  const int sys_n =
      std::accumulate(mesh.dims().begin() + mesh.solve_dim() + 1,
                      mesh.dims().end(), 1, std::multiplies<int>());
  // buffer holding the 3 arrays (a, c, d) merged:
  std::vector<Float> buffer;
  buffer.reserve(sys_n * reduced_sys_len * 3);
  for (int mpi_coord = 0; mpi_coord < num_mpi_procs; ++mpi_coord) {
    for (int sys_idx = 0; sys_idx < sys_n; ++sys_idx) {
      buffer.push_back(mesh.a()[sys_idx * reduced_sys_len + 2 * mpi_coord]);
      buffer.push_back(mesh.a()[sys_idx * reduced_sys_len + 2 * mpi_coord + 1]);
      buffer.push_back(mesh.c()[sys_idx * reduced_sys_len + 2 * mpi_coord]);
      buffer.push_back(mesh.c()[sys_idx * reduced_sys_len + 2 * mpi_coord + 1]);
      buffer.push_back(mesh.d()[sys_idx * reduced_sys_len + 2 * mpi_coord]);
      buffer.push_back(mesh.d()[sys_idx * reduced_sys_len + 2 * mpi_coord + 1]);
    }
  }
  DeviceArray<Float> buffer_d(buffer);
  DeviceArray<Float> result_d(2 * sys_n);

  pcr_on_reduced_batched<Float>(buffer_d.data(), result_d.data(), sys_n,
                                mpi_coord, reduced_sys_len);

  std::vector<Float> result(2 * sys_n);
  cudaMemcpy(result.data(), result_d.data(), sizeof(Float) * 2 * sys_n,
             cudaMemcpyDeviceToHost);
  // BATCHING reduced calls
  const int batch_size  = 32;
  const int num_batches = 1 + (sys_n - 1) / batch_size;
  std::vector<Float> batched_buffer;
  batched_buffer.reserve(sys_n * reduced_sys_len * 3);
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
    // Solve the reduced system
    for (int mpi_coord = 0; mpi_coord < num_mpi_procs; ++mpi_coord) {
      for (int sys_idx = batch_start; sys_idx < batch_start + bsize;
           ++sys_idx) {
        batched_buffer.push_back(
            mesh.a()[sys_idx * reduced_sys_len + 2 * mpi_coord]);
        batched_buffer.push_back(
            mesh.a()[sys_idx * reduced_sys_len + 2 * mpi_coord + 1]);
        batched_buffer.push_back(
            mesh.c()[sys_idx * reduced_sys_len + 2 * mpi_coord]);
        batched_buffer.push_back(
            mesh.c()[sys_idx * reduced_sys_len + 2 * mpi_coord + 1]);
        batched_buffer.push_back(
            mesh.d()[sys_idx * reduced_sys_len + 2 * mpi_coord]);
        batched_buffer.push_back(
            mesh.d()[sys_idx * reduced_sys_len + 2 * mpi_coord + 1]);
      }
    }
  }
  DeviceArray<Float> buffer_batched_d(batched_buffer);
  DeviceArray<Float> result_batched_d(2 * sys_n);
  const int sys_bound_size = 6;
  for (int bidx = 0; bidx < num_batches; ++bidx) {
    int batch_start = bidx * batch_size;
    int bsize = bidx == num_batches - 1 ? sys_n - batch_start : batch_size;
    // Solve the reduced system
    int buf_offset       = sys_bound_size * num_mpi_procs * batch_start;
    int bound_buf_offset = 2 * batch_start;
    pcr_on_reduced_batched<Float>(buffer_batched_d.data() + buf_offset,
                                  result_batched_d.data() + bound_buf_offset,
                                  bsize, mpi_coord, reduced_sys_len);
  }
  std::vector<Float> result_batched(2 * sys_n);
  cudaMemcpy(result_batched.data(), result_batched_d.data(),
             sizeof(Float) * 2 * sys_n, cudaMemcpyDeviceToHost);
  require_allclose(result, result_batched);
}

template <typename Float>
void test_short_forward_pass(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size(), 0),
      periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data(), 256,
                         MpiSolverParams::PCR);

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  std::vector<int> local_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
    local_strides[i] = i == 0 ? 1 : local_strides[i - 1] * local_sizes[i - 1];
  }

  // Simulate distributed environment: only load our data
  std::vector<Float> a, b, c, u, d;
  a.reserve(domain_size);
  b.reserve(domain_size);
  c.reserve(domain_size);
  d.reserve(domain_size);
  u.reserve(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  GPUMesh<Float> local_device_mesh(a, b, c, d, local_sizes);
  int n_sys = domain_size / local_sizes[mesh.solve_dim()];

  // Solve the equations
  std::vector<Float> host_init(domain_size, 0);
  DeviceArray<Float> aa(host_init.data(), domain_size);
  DeviceArray<Float> cc(host_init.data(), domain_size);
  DeviceArray<Float> boundaries(host_init.data(), 2 * 3 * n_sys);
  // tridmtsvStridedBatchMPIWrapper<Float, 0>(
  //     params, local_device_mesh.a().data(), local_device_mesh.b().data(),
  //     local_device_mesh.c().data(), local_device_mesh.d().data(), u_d.data(),
  //     mesh.dims().size(), mesh.solve_dim(), local_sizes.data(),
  //     local_sizes.data());

  dim3 dimBlock_x(128, 1);
  int dimgrid = 1 + (n_sys - 1) / dimBlock_x.x; // can go up to 65535
  dim3 dimGrid_x(dimgrid % 65536, 1 + dimgrid / 65536);
  forward_batched_pass<Float, true>(
      dimGrid_x, dimBlock_x, &params, local_device_mesh.a().data(),
      local_sizes.data(), local_device_mesh.b().data(), local_sizes.data(),
      local_device_mesh.c().data(), local_sizes.data(),
      local_device_mesh.d().data(), local_sizes.data(), aa.data(), cc.data(),
      boundaries.data(), local_sizes.data(), mesh.dims().size(),
      mesh.solve_dim(), 0, n_sys, 0);
  std::vector<Float> a_loc(a.data(), a.data() + domain_size);
  std::vector<Float> c_loc(c.data(), c.data() + domain_size);
  std::vector<Float> d_loc(d.data(), d.data() + domain_size);
  // ref
  int stride = local_strides[mesh.solve_dim()];
  for (int sys_idx = 0; sys_idx < n_sys; ++sys_idx) {
    size_t sys_index =
        get_sys_start_idx(sys_idx, mesh.solve_dim(), local_sizes.data(),
                          local_sizes.data(), local_sizes.size());
    // i = 0:
    Float b_0 = b[sys_index];
    Float c_0 = c[sys_index];
    Float d_0 = d[sys_index];
    // i = 1
    if (rank == 0) {
      int ind      = sys_index + stride;
      Float factor = a[ind] / b_0;
      Float bbi    = static_cast<Float>(1.0) / (b[ind] - factor * c_0);
      d_loc[ind]   = bbi * (d[ind] - factor * d_0);
      c_loc[ind]   = bbi * c[ind];
      a_loc[ind]   = 0;
      d_0          = d_0 - c_0 * d_loc[ind];
      c_0          = -c_0 * c_loc[ind];
    } else {
      d_loc[sys_index + stride] = d[sys_index + stride] / b[sys_index + stride];
      c_loc[sys_index + stride] = c[sys_index + stride] / b[sys_index + stride];
      a_loc[sys_index + stride] = a[sys_index + stride] / b[sys_index + stride];
      b_0                       = b_0 - c_0 * a_loc[sys_index + stride];
      d_0                       = d_0 - c_0 * d_loc[sys_index + stride];
      if (rank == params.num_mpi_procs[mesh.solve_dim()] - 1 &&
          local_sizes[mesh.solve_dim()] == 2) {
        c_0 = 0;
      } else {
        c_0 = -c_0 * c_loc[sys_index + stride];
      }
    }

    // Eliminate lower off-diagonal
    for (int i = 2; i < local_sizes[mesh.solve_dim()]; i++) {
      int ind = sys_index + i * stride;
      Float bbi =
          static_cast<Float>(1.0) / (b[ind] - a[ind] * c_loc[ind - stride]);
      d_loc[ind] = bbi * (d[ind] - a[ind] * d_loc[ind - stride]);
      if (i == local_sizes[mesh.solve_dim()] - 1 &&
          rank == params.num_mpi_procs[mesh.solve_dim()] - 1) {
        c_loc[ind] = 0.0;
      } else {
        c_loc[ind] = bbi * c[ind];
      }
      if (rank == 0) {
        a_loc[ind] = 0;
      } else {
        a_loc[ind] = bbi * (-a[ind] * a_loc[ind - stride]);
        b_0        = b_0 - c_0 * a_loc[ind];
      }
      d_0 = d_0 - c_0 * d_loc[ind];
      c_0 = -c_0 * c_loc[ind];
    }
    if (rank == 0)
      a_loc[sys_index] = 0.0;
    else
      a_loc[sys_index] = a[sys_index] / b_0;
    c_loc[sys_index] = c_0 / b_0;
    d_loc[sys_index] = d_0 / b_0;
  }

  cudaMemcpy(d.data(), local_device_mesh.d().data(),
             sizeof(Float) * domain_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(a.data(), aa.data(), sizeof(Float) * domain_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(c.data(), cc.data(), sizeof(Float) * domain_size,
             cudaMemcpyDeviceToHost);
  // Check result
  // print_array("a_loc", a_loc);
  // print_array("aa   ", a);
  require_allclose(a_loc.data(), a.data(), domain_size, 1);
  // print_array("c_loc", c_loc);
  // print_array("cc   ", c);
  require_allclose(c_loc.data(), c.data(), domain_size, 1);
  // print_array("d_loc", d_loc);
  // print_array("dd   ", d);
  require_allclose(d_loc.data(), d.data(), domain_size, 1);
}

template <typename Float>
void test_iterative_pcr_on_reduced(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size(), 0),
      periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data(), 256,
                         MpiSolverParams::PCR);

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  std::vector<int> local_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
    local_strides[i] = i == 0 ? 1 : local_strides[i - 1] * local_sizes[i - 1];
  }

  // Simulate distributed environment: only load our data
  std::vector<Float> a, b, c, u, d;
  a.reserve(domain_size);
  b.reserve(domain_size);
  c.reserve(domain_size);
  d.reserve(domain_size);
  u.reserve(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  int n_sys = domain_size / local_sizes[mesh.solve_dim()];
  std::vector<Float> boundaries_h(2 * 3 * n_sys, 0);
  GPUMesh<Float> local_device_mesh(a, b, c, d, local_sizes);

  // Solve the equations
  std::vector<Float> host_init(domain_size, 0);
  DeviceArray<Float> aa(host_init.data(), domain_size);
  DeviceArray<Float> cc(host_init.data(), domain_size);
  DeviceArray<Float> boundaries(boundaries_h.data(), 2 * 3 * n_sys);

  dim3 dimBlock_x(128, 1);
  int dimgrid = 1 + (n_sys - 1) / dimBlock_x.x; // can go up to 65535
  dim3 dimGrid_x(dimgrid % 65536, 1 + dimgrid / 65536);
  forward_batched_pass<Float, true>(
      dimGrid_x, dimBlock_x, &params, local_device_mesh.a().data(),
      local_sizes.data(), local_device_mesh.b().data(), local_sizes.data(),
      local_device_mesh.c().data(), local_sizes.data(),
      local_device_mesh.d().data(), local_sizes.data(), aa.data(), cc.data(),
      boundaries.data(), local_sizes.data(), mesh.dims().size(),
      mesh.solve_dim(), 0, n_sys, 0);
  // std::vector<Float> a_loc(a.data(), a.data() + domain_size);
  // std::vector<Float> c_loc(c.data(), c.data() + domain_size);
  // std::vector<Float> d_loc(d.data(), d.data() + domain_size);
  //
  // cudaMemcpy(a_loc.data(), aa.data(), sizeof(Float) * domain_size,
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(c_loc.data(), cc.data(), sizeof(Float) * domain_size,
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(d_loc.data(), local_device_mesh.d().data(),
  //            sizeof(Float) * domain_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(boundaries_h.data(), boundaries.data(),
  //            sizeof(Float) * 2 * 3 * n_sys, cudaMemcpyDeviceToHost);
  // print_array("a    ", a_loc);
  // print_array("c    ", c_loc);
  // print_array("d    ", d_loc);
  // print_array("bound", boundaries_h);
  DeviceArray<Float> rcv_m1_d(host_init.data(), 3 * n_sys);
  Float *rcv_m1_h, *snd_h;
  cudaSafeCall(cudaMallocHost(&rcv_m1_h, 3 * n_sys * sizeof(Float)));
  cudaSafeCall(cudaMallocHost(&snd_h, 3 * n_sys * sizeof(Float)));
  // snd_buf_h, rcv_buf_h
  iterative_pcr_on_reduced<Float>(dimGrid_x, dimBlock_x, &params,
                                  boundaries.data(), n_sys, mesh.solve_dim(),
                                  rcv_m1_d.data(), rcv_m1_h, snd_h);
  // boundaries stores the result to the first row for each node at this
  // point.
  cudaMemcpy(boundaries_h.data(), boundaries.data(),
             sizeof(Float) * 2 * 3 * n_sys, cudaMemcpyDeviceToHost);
  cudaFreeHost(snd_h);
  cudaFreeHost(rcv_m1_h);
  // Check result
  // print_array("bound", boundaries_h);
  // print_array("u    ", u);
  int stride   = local_strides[mesh.solve_dim()];
  int sys_size = local_sizes[mesh.solve_dim()];
  for (int sys_idx = 0; sys_idx < n_sys; ++sys_idx) {
    size_t sys_index =
        get_sys_start_idx(sys_idx, mesh.solve_dim(), local_sizes.data(),
                          local_sizes.data(), local_sizes.size());
    CAPTURE(sys_idx);
    CAPTURE(n_sys);
    CAPTURE(boundaries_h[n_sys * 2 + sys_idx]);
    CAPTURE(u[sys_index]);
    Float exp     = u[sys_index];
    Float act     = boundaries_h[n_sys * 2 + sys_idx];
    Float min_val = std::min(std::abs(exp), std::abs(act));
    const double tolerance =
        abs_tolerance<Float> + rel_tolerance<Float> * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(exp) - act);
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}


TEMPLATE_TEST_CASE("short forward pass small X", "[forward][small][solvedim:0]",
                   double, float) {
  SECTION("ndims: 1") {
    test_short_forward_pass<TestType>("files/one_dim_small");
  }
  SECTION("ndims: 2") {
    test_short_forward_pass<TestType>("files/two_dim_small_solve0");
  }
}

TEMPLATE_TEST_CASE("short forward pass solveX", "[forward][large][solvedim:0]",
                   double, float) {
  SECTION("ndims: 1") {
    test_short_forward_pass<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    test_short_forward_pass<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_short_forward_pass<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("short forward pass small Y", "[forward][small][solvedim:1]",
                   double, float) {
  SECTION("ndims: 2") {
    test_short_forward_pass<TestType>("files/two_dim_small_solve1");
  }
}

TEMPLATE_TEST_CASE("short forward pass solveY", "[forward][large][solvedim:1]",
                   double, float) {
  SECTION("ndims: 2") {
    test_short_forward_pass<TestType>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_short_forward_pass<TestType>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("short forward pass solveZ", "[forward][large][solvedim:2]",
                   double, float) {
  SECTION("ndims: 3") {
    test_short_forward_pass<TestType>("files/three_dim_large_solve2");
  }
}

TEMPLATE_TEST_CASE("iterative reduced solve with PCR small", "[reduced][small]",
                   double, float) {
  SECTION("ndims: 1") {
    test_iterative_pcr_on_reduced<TestType>("files/one_dim_small");
  }
  SECTION("ndims: 2") {
    test_iterative_pcr_on_reduced<TestType>("files/two_dim_small_solve0");
  }
}

TEMPLATE_TEST_CASE("iterative reduced solve with PCR", "[reduced][large]",
                   double, float) {
  SECTION("ndims: 1") {
    test_iterative_pcr_on_reduced<TestType>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    test_iterative_pcr_on_reduced<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_iterative_pcr_on_reduced<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("PCR on reduced", "[reduced]", double, float) {
  test_PCR_on_reduced<TestType>("files/reduced_test_small");
}

enum ResDest { assign = 0, increment };

#define PARAM_COMBOS                                                           \
  (double, assign, MpiSolverParams::ALLGATHER),                                \
      (double, assign, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),           \
      (double, assign, MpiSolverParams::LATENCY_HIDING_TWO_STEP),              \
      (double, assign, MpiSolverParams::PCR),                                  \
      (double, assign, MpiSolverParams::JACOBI),                               \
      (float, assign, MpiSolverParams::ALLGATHER),                             \
      (float, assign, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),            \
      (float, assign, MpiSolverParams::LATENCY_HIDING_TWO_STEP),               \
      (float, assign, MpiSolverParams::PCR),                                   \
      (float, assign, MpiSolverParams::JACOBI),                                \
      (double, increment, MpiSolverParams::ALLGATHER),                         \
      (double, increment, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),        \
      (double, increment, MpiSolverParams::LATENCY_HIDING_TWO_STEP),           \
      (double, increment, MpiSolverParams::PCR),                               \
      (double, increment, MpiSolverParams::JACOBI),                            \
      (float, increment, MpiSolverParams::ALLGATHER),                          \
      (float, increment, MpiSolverParams::LATENCY_HIDING_INTERLEAVED),         \
      (float, increment, MpiSolverParams::LATENCY_HIDING_TWO_STEP),            \
      (float, increment, MpiSolverParams::PCR),                                \
      (float, increment, MpiSolverParams::JACOBI)

TEMPLATE_TEST_CASE_SIG("cuda solver mpi: solveX", "[solver][solvedim:0]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 1") {
    test_solver_from_file<TestType, INC, strategy>("files/one_dim_large");
  }
  SECTION("ndims: 2") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE_SIG("cuda solver mpi: solveY", "[solver][solvedim:1]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 2") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE_SIG("cuda solver mpi: solveZ", "[solver][solvedim:2]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 3") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/three_dim_large_solve2");
  }
}

TEMPLATE_TEST_CASE_SIG("cuda: padded", "[padded]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 1") {
    test_solver_from_file_padded<TestType, INC, strategy>(
        "files/one_dim_large");
  }
  SECTION("ndims: 2") {
    SECTION("solvedim: 0") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/two_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/two_dim_large_solve1");
    }
  }
  SECTION("ndims: 3") {
    SECTION("solvedim: 0") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/three_dim_large_solve0");
    }
    SECTION("solvedim: 1") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/three_dim_large_solve1");
    }
    SECTION("solvedim: 2") {
      test_solver_from_file_padded<TestType, INC, strategy>(
          "files/three_dim_large_solve2");
    }
  }
}

#if MAXDIM > 3
TEMPLATE_TEST_CASE_SIG("cuda solver mpi 4D: solveX", "[solver][solvedim:0]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/four_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE_SIG("cuda solver mpi 4D: solveY", "[solver][solvedim:1]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/four_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE_SIG("cuda solver mpi 4D: solveZ", "[solver][solvedim:2]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/four_dim_large_solve2");
  }
}

TEMPLATE_TEST_CASE_SIG("cuda solver mpi: solve3", "[solver][solvedim:3]",
                       ((typename TestType, ResDest INC,
                         MpiSolverParams::MPICommStrategy strategy),
                        TestType, INC, strategy),
                       PARAM_COMBOS) {
  SECTION("ndims: 4") {
    test_solver_from_file<TestType, INC, strategy>(
        "files/four_dim_large_solve3");
  }
}
#endif
