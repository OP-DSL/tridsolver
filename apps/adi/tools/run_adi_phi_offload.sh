#!/usr/bin/env bash
# Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#MIC_LD_LIBRARY_PATH=${INTEL_PATH}/compiler/lib/mic:${INTEL_PATH}/mkl/lib/mic
#MIC_ENV_PREFIX=PHI
#PHI_KMP_AFFINITY=compact
#PHI_KMP_PLACE_THREADS=59c,1t,0O
#PHI_OMP_NUM_THREADS=59

MIC_LD_LIBRARY_PATH=${INTEL_PATH}/compiler/lib/mic:${INTEL_PATH}/mkl/lib/mic
MIC_ENV_PREFIX=PHI
PHI_KMP_AFFINITY=compact
PHI_KMP_PLACE_THREADS=59c,4t,0O
PHI_OMP_NUM_THREADS=236

echo "MIC_LD_LIBRARY_PATH  = $MIC_LD_LIBRARY_PATH"
echo "MIC_ENV_PREFIX       = $MIC_ENV_PREFIX"
echo "PHI_KMP_AFFINITY     = $PHI_KMP_AFFINITY"
echo "PHI_KMP_PLACE_THREAD = $PHI_KMP_PLACE_THREADS"
echo "PHI_OMP_NUM_THREADS  = $PHI_OMP_NUM_THREADS"

export MIC_LD_LIBRARY_PATH
export MIC_ENV_PREFIX
export PHI_KMP_AFFINITY
export PHI_KMP_PLACE_THREADS
export PHI_OMP_NUM_THREADS

#unset  PHI_KMP_PLACE_THREADS
#unset  PHI_OMP_NUM_THREADS
#./adi_phi_offload -nx=256 -ny=256 -nz=256
./adi_phi_offload $@
#./adi_phi_offload -nx=16 -ny=16 -nz=16
#./adi_phi_offload -nx=4 -ny=4 -nz=4
