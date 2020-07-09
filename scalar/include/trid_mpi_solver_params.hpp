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

struct MpiSolverParams {
  // This will be an array with a communicator for each dimension. Separate
  // communicator that includes every node calculating the same set of equations
  // as the current node for each dimension.
  std::vector<MPI_Comm> communicators;

  // The number of MPI processes in each dimension. It is `num_dims` large. It
  // won't be owned.
  const int *num_mpi_procs;

  // The coordinates of the current MPI process in the cartesian mesh.
  std::vector<int> mpi_coords;

  // Assumes that the number
  MpiSolverParams(MPI_Comm cartesian_communicator, int num_dims,
                  int *num_mpi_procs_)
      : communicators(num_dims), num_mpi_procs(num_mpi_procs_),
        mpi_coords(num_dims) {
    int cart_rank;
    MPI_Comm_rank(cartesian_communicator, &cart_rank);
    MPI_Cart_coords(cartesian_communicator, cart_rank, num_dims,
                    this->mpi_coords.data());
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
      MPI_Group cart_group;
      MPI_Comm_group(cartesian_communicator, &cart_group);
      MPI_Group neighbours_group;
      MPI_Group_incl(cart_group, neighbours.size(), neighbours.data(),
                     &neighbours_group);
      MPI_Comm_create(cartesian_communicator, neighbours_group,
                      &this->communicators[equation_dim]);
    }
  }
};

#endif /* ifndef TRID_MPI_SOLVER_PARAMS_HPP */
