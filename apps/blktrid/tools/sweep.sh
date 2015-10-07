#!/bin/bash
# Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 
# Set error checking: if a statment returns other than 0, the script stops executing
set +e 
set -u

# Use this tool to sweep a problem space

if [[ -z "$1" ]] ; then
	echo "Specify an executable!"
	exit 
fi
#if [[ -z "$2" ]] ; then
#	echo "Specify the floating point precision: 0-single, 1-double!"
#	exit 
#fi

echo "Sweeping started..."

# Create directories for logging and benchmarking
mkdir -p log
mkdir -p benchmark

# Set iteration number and precision
export BINARY=$1
#export BLK_DIM=4
#export N=128 
#export P=65536
#export ITER=30
#export SOLVER=-1
#export FPPREC=0

# Set sweep data file
#SWEEPDAT="benchmark/sweep_${BINARY}_N${N}_P${P}_SOLVER${SOLVER}_ITER${ITER}_FPPREC_${FPPREC}.dat"
#SWEEPDAT="sweep_${BINARY}_$(date +"%d_%m_%Y_%H_%M_%S")_SOLVER_${SOLVER}_FPPREC_${FPPREC}.dat"
SWEEPDAT="sweep_${BINARY}_$(date +"%d_%m_%Y_%H_%M_%S").dat"
rm -f $SWEEPDAT
#echo "[N]  [P] [SOLVER] [ITER] " >> $SWEEPDAT
#echo "[N]  [TOTAL] [PREPROC] [TRIDX] [TRIDY] [TRIDZ]"

# STDOUT and ERROUT log filenames
#BUILDOUTLOG=log/${SWEEPDAT}\_build_out.log
#BUILDERRLOG=log/${SWEEPDAT}\_build_err.log
OUTLOG=log/${SWEEPDAT}_out.log
ERRLOG=log/${SWEEPDAT}_err.log

# Erase binary to make sure it will be rebuilt
#rm -f $BINARY

# Build the binary
#make $BINARY > $BUILDOUTLOG 2> $BUILDERRLOG
#make $BINARY 

# Set environment variables to use thread pinning
#export KMP_AFFINITY=compact,0
#export MIC_KMP_AFFINITY=compact,0
#module load mic-nativeloadex/2013

# Sweep through problem N^3 size
#for P in  64 128 256 512 1024 2048 4096 8092 16384 32768 65536
#for B in  2 4 6
for B in  2 3 4 5 6 7
do
  N=128 
  #P=65536
  #N=96 
  P=32768
  ITER=30
  ./$BINARY -n=$N -p=$P -blkdim=$B -iter=$ITER | tee > $OUTLOG 2> $ERRLOG
  RES=`tail -n 1 $OUTLOG`
  #RES=`tail -n 2 $OUTLOG | head -n 1`
  LINE="N=$N P=$P ITER=$ITER B=$B $RES"
  echo $LINE >> ./benchmark/$SWEEPDAT
  echo $LINE 
done

for B in 8 9 10
do
  #N=96 
  #P=32768
  N=64 
  P=16384
  ITER=30
  ./$BINARY -n=$N -p=$P -blkdim=$B -iter=$ITER | tee > $OUTLOG 2> $ERRLOG
  RES=`tail -n 1 $OUTLOG`
  #RES=`tail -n 2 $OUTLOG | head -n 1`
  LINE="N=$N P=$P ITER=$ITER B=$B $RES"
  echo $LINE >> ./benchmark/$SWEEPDAT
  echo $LINE 
done

echo "Sweeping ended.\n"




#if [ -z $BINARY ]; then
#	echo "Please specify the geometry h5 file!"
#	exit 1
#fi
#
#for i in *.h5; do
#	# If the geometry filename is different from the snapshot filename, convert the file
#	if [ $i != $BINARY ]; then
#		if [ -n `echo ${i} | grep '\.h5$'` ]; then
#			newname=`echo ${i} | sed -e 's/\.h5$/\.vtk/g'`
#       		 	echo "Converting ${i} to ${newname} ..."
#			if [ -z $2 ]; then
#				./hdf52vtk $BINARY $i 0 $newname 
#			else
#				./hdf52vtk $BINARY $i $2 $newname 
#			fi
#		fi
#	fi
#done

# Turn off error checking
set -e
