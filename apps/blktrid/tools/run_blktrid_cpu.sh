#!/usr/bin/env bash
# Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 
KMP_AFFINITY=compact,0
#KMP_PLACE_THREADS=59c,4t,0O
OMP_NUM_THREADS=32

echo "KMP_AFFINITY     = $KMP_AFFINITY"
#echo "KMP_PLACE_THREAD = $KMP_PLACE_THREADS"
echo "OMP_NUM_THREADS  = $OMP_NUM_THREADS"

export KMP_AFFINITY
#export KMP_PLACE_THREADS
export OMP_NUM_THREADS

./blktrid_cpu $@
