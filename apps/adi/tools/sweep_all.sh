#!/usr/bin/env bash
# Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

echo "Don't forget to:"
echo " 1) set the CUDA 7.0 paths, "
echo " 2) compile single precision, "
echo " 3) compile double precision "

# Single precision
./sweep2.sh adi_cuda_sp -optx=0 -opty=0 -optz=0
./sweep2.sh adi_cuda_sp -optx=1 -opty=0 -optz=0
./sweep2.sh adi_cuda_sp -optx=2 -opty=0 -optz=0
./sweep2.sh adi_cuda_sp -optx=3 -opty=0 -optz=0
./sweep2.sh adi_cuda_sp -optx=4 -opty=0 -optz=0

./sweep2_y.sh adi_cuda_sp -optx=0 -opty=0 -optz=0
./sweep2_y.sh adi_cuda_sp -optx=0 -opty=3 -optz=0

# Double precision
./sweep2.sh adi_cuda_dp -optx=0 -opty=0 -optz=0
./sweep2.sh adi_cuda_dp -optx=1 -opty=0 -optz=0
./sweep2.sh adi_cuda_dp -optx=2 -opty=0 -optz=0
./sweep2.sh adi_cuda_dp -optx=3 -opty=0 -optz=0
./sweep2.sh adi_cuda_dp -optx=4 -opty=0 -optz=0

./sweep2_y.sh adi_cuda_dp -optx=0 -opty=0 -optz=0
./sweep2_y.sh adi_cuda_dp -optx=0 -opty=3 -optz=0
