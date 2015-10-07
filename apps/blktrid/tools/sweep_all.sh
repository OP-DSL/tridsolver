#!/usr/bin/env bash
# Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

# Benchmark CPU
#rm -f blktrid_cpu
#rm -f blktrid_cpu_mkl
#make blktrid_cpu
#make blktrid_cpu_mkl
#./sweep.sh run_blktrid_cpu.sh
#./sweep.sh run_blktrid_cpu_mkl.sh

## Benchmark MIC
#rm -f blktrid_mic
#rm -f blktrid_mic_mkl
#make blktrid_mic
#make blktrid_mic_mkl
#./sweep.sh run_blktrid_mic.sh
#./sweep.sh run_blktrid_mic_mkl.sh

## Benchmark GPU
#rm -f blktrid_gpu
#make blktrid_gpu
./sweep.sh blktrid_gpu_sp
./sweep.sh blktrid_gpu_dp
