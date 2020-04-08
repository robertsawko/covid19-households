#!/bin/bash
# NOTE: This script is to be run from the main directory of the repo.

ml purge
ml Community 2> /dev/null
ml gcc openmpi/4.0.3
ml python hdf5

export PYTHONPATH=$(pwd):$PYTHONPATH

NUM_OF_CORES=40
CORES_PER_HOST=20

JOB_NAME=lockdown

bsub << EOF
#BSUB -J ${JOB_NAME}
#BSUB -q tuleta
#BSUB -oo ${JOB_NAME}.txt
#BSUB -n $NUM_OF_CORES
#BSUB -R "span[ptile=$CORES_PER_HOST]"
#BSUB -a "p8aff(1,1,)"
#BSUB -W 2:00
#BSUB -g /covid/households

ulimit -s 10240
mpirun \\
    python analysis/parallel/lockdown/scan.py
EOF
