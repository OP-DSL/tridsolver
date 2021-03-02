/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Endre László and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Endre László may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 */

// Written by Gabor Daniel Balogh, Pazmany Peter Catholic University,
// balogh.gabor.daniel@itk.ppke.hu, 2020
// Implementation of a struct bounding together MPI related parameters.

#ifndef TRID_MPI_SOLVER_PARAMS_HPP
#define TRID_MPI_SOLVER_PARAMS_HPP

#include <algorithm>
#include <mpi.h>
#include <vector>
#ifdef TRID_NCCL
#  include <stdio.h>
#  include "nccl.h"
#  define NCCLCHECK(cmd)                                                       \
    do {                                                                       \
      ncclResult_t r = cmd;                                                    \
      if (r != ncclSuccess) {                                                  \
        printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,          \
               ncclGetErrorString(r));                                         \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
    } while (0)
#endif

struct MpiSolverParams {
  // MPI communication strategies
  enum MPICommStrategy {
    GATHER_SCATTER = 0, // Gather boundaries on first nodes solve reduced system
                        // and scatter results
    ALLGATHER,          // Gather boundaries and solve reduced on all nodes
    JACOBI,             // Use Jacobi iterations to solve the reduced system
    LATENCY_HIDING_INTERLEAVED, // Perform solves in mini-batches. Do forward
                                // run of the current mini-batch start
                                // communication and finish the previous
                                // mini-batch
    LATENCY_HIDING_TWO_STEP     // Perform solves in min-batches. First step:
                            // forwards and start communication, second step:
                            // wait for ready requests and finish mini-batches
  };

  // This will be an array with a communicator for each dimension. Separate
  // communicator that includes every node calculating the same set of equations
  // as the current node for each dimension.
  std::vector<MPI_Comm> communicators;
  std::vector<MPI_Group> cart_groups;
  std::vector<MPI_Group> neighbours_groups;
#ifdef TRID_NCCL
  std::vector<ncclComm_t> ncclComms;
#endif

  // The number of MPI processes in each dimension. It is `num_dims` large. It
  // won't be owned.
  const int *num_mpi_procs;

  // The coordinates of the current MPI process in the cartesian mesh.
  std::vector<int> mpi_coords;

  // The number of system in a mini-batch used for hide latency of the MPI
  // communication.
  int mpi_batch_size = 32;

  // Used MPI communication strategy
  MPICommStrategy strategy = LATENCY_HIDING_INTERLEAVED;

  // Tolerance values for jacobi iteration on reduced system
  double jacobi_rtol = 1e-12;
  double jacobi_atol = 1e-11;

  MpiSolverParams(MPI_Comm cartesian_communicator, int num_dims,
                  int *num_mpi_procs_)
      : communicators(num_dims), cart_groups(num_dims),
        neighbours_groups(num_dims), num_mpi_procs(num_mpi_procs_),
        mpi_coords(num_dims) {
    int cart_rank;
    MPI_Comm_rank(cartesian_communicator, &cart_rank);
    MPI_Cart_coords(cartesian_communicator, cart_rank, num_dims,
                    this->mpi_coords.data());
#ifdef TRID_NCCL
    ncclComms.resize(num_dims);
#endif
    for (int equation_dim = 0; equation_dim < num_dims; ++equation_dim) {
      std::vector<int> neighbours = {cart_rank};
      int mpi_coord               = this->mpi_coords[equation_dim];
      // Collect the processes in the same row/column
      for (int i = 1; i <= std::max(num_mpi_procs[equation_dim] - mpi_coord - 1,
                                    mpi_coord);
           ++i) {
        int prev, next;
        MPI_Cart_shift(cartesian_communicator, equation_dim, i, &prev, &next);
        if (i <= mpi_coord) {
          neighbours.push_back(prev);
        }
        if (i + mpi_coord < num_mpi_procs[equation_dim]) {
          neighbours.push_back(next);
        }
      }

      // This is needed, otherwise the communications hang
      std::sort(neighbours.begin(), neighbours.end());

      // Create new communicator for neighbours
      // MPI_Group cart_group;
      MPI_Comm_group(cartesian_communicator, &this->cart_groups[equation_dim]);
      // MPI_Group neighbours_group;
      MPI_Group_incl(this->cart_groups[equation_dim], neighbours.size(),
                     neighbours.data(), &this->neighbours_groups[equation_dim]);
      MPI_Comm_create(cartesian_communicator,
                      this->neighbours_groups[equation_dim],
                      &this->communicators[equation_dim]);
#ifdef TRID_NCCL
      int this_rank, this_size;
      MPI_Comm_rank(this->communicators[equation_dim], &this_rank);
      MPI_Comm_size(this->communicators[equation_dim], &this_size);
      ncclUniqueId id;
      if (this_rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
      }
      MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0,
                this->communicators[equation_dim]);
      NCCLCHECK(ncclCommInitRank(&this->ncclComms[equation_dim],
                                 neighbours.size(), id, this_rank));
#endif
    }
  }

  MpiSolverParams(MPI_Comm cartesian_communicator, int num_dims,
                  int *num_mpi_procs_, int mpi_batch_size,
                  MPICommStrategy _strategy)
      : MpiSolverParams(cartesian_communicator, num_dims, num_mpi_procs_) {
    this->mpi_batch_size = mpi_batch_size;
    this->strategy       = _strategy;
  }

  // Constructor for Jacobi iteration
  MpiSolverParams(MPI_Comm cartesian_communicator, int num_dims,
                  int *num_mpi_procs_, double jacobi_atol_, double jacobi_rtol_)
      : MpiSolverParams(cartesian_communicator, num_dims, num_mpi_procs_) {
    strategy    = JACOBI;
    jacobi_atol = jacobi_atol_;
    jacobi_rtol = jacobi_rtol_;
  }

  ~MpiSolverParams() {
    for (unsigned int equation_dim = 0;
         equation_dim < this->communicators.size(); ++equation_dim) {
      MPI_Group_free(&this->cart_groups[equation_dim]);
      MPI_Group_free(&this->neighbours_groups[equation_dim]);
      MPI_Comm_free(&this->communicators[equation_dim]);
#ifdef TRID_NCCL
      NCCLCHECK(ncclCommDestroy(this->ncclComms[equation_dim]));
#endif
    }
  }
};

#endif /* ifndef TRID_MPI_SOLVER_PARAMS_HPP */
