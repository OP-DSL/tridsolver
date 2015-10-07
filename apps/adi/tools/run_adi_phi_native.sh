#!/usr/bin/env bash
# Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 
ARGS=$@

#KMP_AFFINITY=compact,0
#KMP_PLACE_THREADS=59c,1t,0O
#OMP_NUM_THREADS=59

KMP_AFFINITY=compact,0
KMP_PLACE_THREADS=59c,4t,0O
OMP_NUM_THREADS=236

echo "KMP_AFFINITY         = $KMP_AFFINITY"
echo "KMP_PLACE_THREAD     = $KMP_PLACE_THREADS"
echo "OMP_NUM_THREADS      = $OMP_NUM_THREADS"

#SINK_LD_LIBRARY_PATH=/opt/intel/composer_xe_2015.2.164/compiler/lib/mic /opt/intel/mic/bin/micnativeloadex ./adi_phi_native -a "-nx=64 -ny=256 -nz=256" -e "KMP_AFFINITY=compact,0"
#SINK_LD_LIBRARY_PATH=/opt/intel/composer_xe_2015.2.164/compiler/lib/mic /opt/intel/mic/bin/micnativeloadex ./adi_phi_native -a "$@"
SINK_LD_LIBRARY_PATH=/opt/intel/composer_xe_2015.2.164/compiler/lib/mic /opt/intel/mic/bin/micnativeloadex ./adi_phi_native -a "$ARGS" -e "KMP_AFFINITY=$PHI_KMP_AFFINITY KMP_PLACE_THREADS=$KMP_PLACE_THREADS OMP_NUM_THREADS=$OMP_NUM_THREADS"
