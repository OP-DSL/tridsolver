#!/bin/bash
# Set error checking: if a statment returns other than 0, the script stops executing
set +e 
set -u

# Use this tool to sweep a problem space
#if [[ -z "$1" ]] ; then
#	echo "Specify an ADI implementation!"
#	exit 
#fi

echo "Validation started..."

#export KMP_AFFINITY=compact,0

# Create directories for validation logging
mkdir -p log_validation
#mkdir -p benchmark

# Set iteration number and precision
#export BINARY=$1
export ITER=2
#export FPPREC=$2
export OPT=3
#export NATIVELOADSCRIPT=$4

## Set sweep data file
#SWEEPDAT="benchmark/sweep_${BINARY}_OPT${OPT}_ITER${ITER}_FPPREC_${FPPREC}.dat"
#rm -f $SWEEPDAT
#echo "[N]  [TOTAL] [PREPROC] [TRIDX] [TRIDY] [TRIDZ]" >> $SWEEPDAT
#echo "[N]  [TOTAL] [PREPROC] [TRIDX] [TRIDY] [TRIDZ]"
#
## STDOUT and ERROUT log filenames
#BUILDOUTLOG=log/$BINARY\_FPPREC$2\_build_out.log
#BUILDERRLOG=log/$BINARY\_FPPREC$2\_build_err.log
VALLOG=log_validation/validation.log
OUTLOG=log_validation/out.log
ERRLOG=log_validation/err.log
echo "# VALIDATION LOG - Outputs of ./compare" > $VALLOG
echo "# STDOUT LOG - Outputs of ./adi_*" > $OUTLOG
echo "# ERROUT LOG - Outputs of ./adi_*" > $ERRLOG

#
## Erase binary to make sure it will be rebuilt
#rm -f $BINARY
#
## Build the binary
#make $BINARY > $BUILDOUTLOG 2> $BUILDERRLOG
#
## Set environment variables to use thread pinning
#export KMP_AFFINITY=compact,0
#export MIC_KMP_AFFINITY=compact,0
#module load mic-nativeloadex/2013

# Sweep through problem N^3 size
#for N in {230..270}
for N in 231 247 251 252 253 254 255 257 258 259
do
  # Set cube dimensions for build
  NX=$N 
  NY=256 
  NZ=256

  echo "##########################################" >> $VALLOG
  echo "Validating $NX $NY $NZ\n" | tee -a $VALLOG

  ./adi_cuda     -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -optx=$OPT -opty=0 -optz=0 -prof=1 >> $OUTLOG 2>> $ERRLOG
  ./adi_cpu_orig -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -prof=1 >> $OUTLOG 2>> $ERRLOG
  #./adi_cuda     -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -optx=$OPT -opty=0 -optz=0 -prof=1 
  #./adi_cpu_orig -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -prof=1
  ./compare adi_cpu_orig.dat adi_cuda.dat -nx=$NX -ny=$NY -nz=$NZ | tee -a $VALLOG 

done

echo "Validation ended.\n"

# Turn off error checking
set -e
