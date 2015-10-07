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
mkdir -p benchmark2

# Set iteration number and precision
BINARY=$1
FPPREC=0
ARGS=$@

# Set sweep data file
ARGS_NO_WHITESPACE="$(echo -e "${ARGS}" | tr -d '[[:space:]]')"
SWEEPDAT="sweep_${ARGS_NO_WHITESPACE}_$(date +"%d_%m_%Y_%H_%M_%S").dat"
rm -f $SWEEPDAT
#echo "[N]  [P] [SOLVER] [ITER] " >> $SWEEPDAT
#echo "[N]  [TOTAL] [PREPROC] [TRIDX] [TRIDY] [TRIDZ]"

# STDOUT and ERROUT log filenames
BUILDOUTLOG=log/${SWEEPDAT}_build_out.log
BUILDERRLOG=log/${SWEEPDAT}_build_err.log
OUTLOG=log/${SWEEPDAT}_out.log
ERRLOG=log/${SWEEPDAT}_err.log

echo "OUTLOG=$OUTLOG"
echo "ERRLOG=$ERRLOG"
echo "SWEEPDAT=$SWEEPDAT"

# Erase binary to make sure it will be rebuilt
#rm -f $BINARY
# Build the binary
#make $BINARY 

# Set environment variables to use thread pinning
#export KMP_AFFINITY=compact,0
#export MIC_KMP_AFFINITY=compact,0
#module load mic-nativeloadex/2013

# Sweep through problem N^3 size
#for NX in  64 128 256 512 1024 2048 4096 8092 16384 32768 65536
#for N in $(seq 64 4 4096)
for N in $(seq 64 4 1024)
do

  #./$BINARY -nx=$N -ny=256 -nz=256 -iter=10 -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #let 'ITER = 40 + ( 1024 - N ) / 4'
  #let 'ITER = 40 + ( 1024 - N ) / 4'
  ITER=20

  # For CPU binaries
  #KMP_AFFINITY=compact,0 ./$BINARY -nx=256 -ny=$N -nz=256 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=$N -ny=256 -nz=256 -iter=$ITER -prof=1 
  #./$BINARY -nx=256 -ny=$N -nz=256 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=$N -ny=256 -nz=256 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG

  # For GPU binaries
  ./$ARGS -nx=$N -ny=256 -nz=256 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=$N -ny=256 -nz=256 -iter=$ITER -prof=1
  #./$BINARY -nx=$N -ny=256 -nz=256 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=$N -ny=256 -nz=256 -optx=3 -opty=0 -optz=0 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=$N -ny=256 -nz=256 -optx=3 -opty=0 -optz=0 -iter=$ITER -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=256 -ny=$N -nz=256 -optx=3 -opty=3 -optz=3 -iter=30 -prof=1 | tee > $OUTLOG 2> $ERRLOG
  #./$BINARY -nx=$NX -ny=256 -nz=256 -optx=3 -opty=3 -optz=3 -iter=10 -prof=1 
  #RES=`tail -n 2 $OUTLOG | head -n 1`
  RES=`tail -n 1 $OUTLOG`
  echo "$N $RES" >> ./benchmark2/$SWEEPDAT
  echo "$N $RES" 



#  # Execute simulation to measure total execution time
#  if [ $NATIVELOADSCRIPT == "0" ]; then
#    ./$BINARY -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -opt=$OPT -prof=0 > $OUTLOG 2> $ERRLOG
#    RES1=`tail -n 1 $OUTLOG`
#  elif [ $NATIVELOADSCRIPT == "1" ]; then
#    /opt/intel/mic/bin/micnativeloadex ./$BINARY -a "-nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -prof=0" -e "KMP_AFFINITY=compact,0" > $OUTLOG 2> $ERRLOG
#    RES1=`tail -n 2 $OUTLOG`
#  else
#    echo "Invalid NATIVELOADSCRIPT option!"
#    exit -1
#  fi
#  #RUN1=`tail -n 1 $OUTLOG`
#  # Get last line
#
#  #read -ra WORDS <<< $RES1
#  # Get total execution time
#  #RUN1=${WORDS[0]}
# 
#  # Execute simulation to measure total execution time
#  if [ $NATIVELOADSCRIPT == "0" ]; then
#    ./$BINARY -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -opt=$OPT -prof=0 > $OUTLOG 2> $ERRLOG
#    RES2=`tail -n 1 $OUTLOG`
#  elif [ $NATIVELOADSCRIPT == "1" ]; then
#    /opt/intel/mic/bin/micnativeloadex ./$BINARY -a "-nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -prof=0" -e "KMP_AFFINITY=compact,0" > $OUTLOG 2> $ERRLOG
#    RES2=`tail -n 2 $OUTLOG`
#  else
#    echo "Invalid NATIVELOADSCRIPT option!"
#    exit -1
#  fi
#  #RUN2=`tail -n 1 $OUTLOG`
#  # Get last line
#  #read -ra WORDS <<< $RES2
#  # Get total execution time
#  #RUN2=${WORDS[0]} 
#
#  # Execute simulation to measure kernel execution times
#  if [ $NATIVELOADSCRIPT == "0" ]; then
#    ./$BINARY -nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -opt=$OPT -prof=1 > $OUTLOG 2> $ERRLOG
#    RES=`tail -n 1 $OUTLOG`
#  elif [ $NATIVELOADSCRIPT == "1" ]; then
#    /opt/intel/mic/bin/micnativeloadex ./$BINARY -a "-nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -prof=1" -e "KMP_AFFINITY=compact,0" > $OUTLOG 2> $ERRLOG
#    #/opt/intel/mic/bin/micnativeloadex ./$BINARY -a "-nx=$NX -ny=$NY -nz=$NZ -iter=$ITER -prof=1" -e "KMP_AFFINITY=compact,0" > $ERRLOG | tee $OUTLOG
#    RES=`tail -n 2 $OUTLOG`
#  else
#    echo "Invalid NATIVELOADSCRIPT option!"
#    exit -1
#  fi
#
#  # Get kernel execution times
#  read -ra WORDS <<< $RES
#  KERNELPROFS="${WORDS[1]} ${WORDS[2]} ${WORDS[3]} ${WORDS[4]}"
#
##  # Get kernel execution times
##  if [ $NATIVELOADSCRIPT == "0" ]; then
##    # Get last line
##    RES=`tail -n 1 $OUTLOG`
##    read -ra WORDS <<< $RES
##    KERNELPROFS="${WORDS[1]} ${WORDS[2]} ${WORDS[3]} ${WORDS[4]}"
##  elif [ $NATIVELOADSCRIPT == "1" ]; then
##    # Get line before last line
##    RES=`tail -n 2 $OUTLOG`
##    read -ra WORDS <<< $RES
##echo $RES
##    KERNELPROFS="${WORDS[1]} ${WORDS[2]} ${WORDS[3]} ${WORDS[4]}"
##  else
##    echo "Invalid NATIVELOADSCRIPT option!"
##    exit -1
##  fi
#
#  # Save the mininum execution time of the two simulations 
#  #ISMIN=`echo "$RUN1 < $RUN2" | bc`
#  ISMIN=`echo "$RES1 < $RES2" | bc`
#  if [ $ISMIN -eq 1 ]; then
#    #TOTAL=$RUN1
#    LINE=$RES1
#  else
#    #TOTAL=$RUN2
#    LINE=$RES2
#  fi
#
#  LINE="$LINE $KERNELPROFS"
  


##if grep "cuda" <<< $BINARY ; then 
#  # Profile the code to get kernel execution times
#  nvprof -u s ./$BINARY > /dev/null 2> prof.log
#
#  # Extract preproc kernel execution time
#  STR=`cat prof.log | grep "preproc"`
#  read -ra WORDS <<< $STR
#  PREPROC=${WORDS[3]} # Min exec time - 5th column
#
#  # Extract preproc kernel execution time
#  STR=`cat prof.log | grep "trid_x"`
#  read -ra WORDS <<< $STR
#  TRIDX=${WORDS[3]} # Min exec time - 5th column
#
#  # Extract preproc kernel execution time
#  STR=`cat prof.log | grep "trid_y"`
#  read -ra WORDS <<< $STR
#  TRIDY=${WORDS[3]} # Min exec time - 5th column
# 
#  # Extract preproc kernel execution time
#  STR=`cat prof.log | grep "trid_z"`
#  read -ra WORDS <<< $STR
#  TRIDZ=${WORDS[3]} # Min exec time - 5th column
#
#
#  # Write results to file: 
#  #  N       - problem size NxNxN
#  #  TOTAL   - total execution time
#  #  PREPROC - preproc kernel min exec time
#  #  TRIDX   - trid_x* kernel min exec time
#  #  TRIDY   - trid_y kernel min exec time
#  #  TRIDZ   - trid_z kernel min exec time
#  #echo "$N $TOTAL $PREPROC $TRIDX $TRIDY $TRIDZ" >> $SWEEPDAT
#  echo "$N $LINE" >> $SWEEPDAT
# 
#  # Print result on the screen
#  #echo "$NX x $NY x $NZ (prec=$2)=> $TOTAL $PREPROC $TRIDX $TRIDY $TRIDZ"
#  echo "$N $LINE"

done

#make clean
#let "NX = 2 * NX" 
#let "NY = 2 * NY"
#let "NZ = 2 * NZ"
#make $BINARY
#./$BINARY > $BINARY\_$NX\_$NY\_$NZ\_out.log 2> $BINARY\_$NX\_$NY\_$NZ\_err.log

#unset KMP_AFFINITY

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
