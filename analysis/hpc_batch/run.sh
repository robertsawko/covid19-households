#!/bin/bash
# This script is to be run from the main directory of the repo.

ml purge
ml Community 2> /dev/null
ml anaconda

export PYTHONPATH=$(pwd):$PYTHONPATH

NO_OF_SIMS=288
FIRST=1

while getopts r:l:f:n:F OPTION
do
case "${OPTION}"
in
f) FIRST=${OPTARG};;
n) NO_OF_SIMS=${OPTARG};;
esac
done

CORES_PER_HOST=20
LAST=$(($FIRST+$NO_OF_SIMS-1))
NO_OF_BATCHES=$(($NO_OF_SIMS/$CORES_PER_HOST))
REMAINDER=$(($NO_OF_SIMS%$CORES_PER_HOST))
JOB_NAME=ensemble-${FIRST}-${LAST}
if [[ $REMAINDER -ne 0  ]]
then
    NO_OF_BATCHES=$(($NO_OF_BATCHES+1))
fi

# For outputs
mkdir -p lsf/$LEVEL 2> /dev/null
# SIMS_IDS are an array
bsub << EOF
#BSUB -J ${JOB_NAME}[1-${NO_OF_BATCHES}]
#BSUB -q tuleta
#BSUB -oo lsf/${JOB_NAME}.%I.txt
#BSUB -n $CORES_PER_HOST
#BSUB -R "span[ptile=$CORES_PER_HOST]"
#BSUB -a "p8aff(1,1,)"
#BSUB -W 8:00
#BSUB -g /covid/households

SIMS_IDS=(\$(seq $FIRST $LAST))
BATCH_FIRST_INDEX=\$(((\${LSB_JOBINDEX}-1)*$CORES_PER_HOST))
REMAINING_SIMS=\$(($LAST-\$BATCH_FIRST_INDEX))
SIMS_IN_BATCH=\$(( \\
    $CORES_PER_HOST < \$REMAINING_SIMS ? \\
    $CORES_PER_HOST : \$REMAINING_SIMS))
SIM0=\$((\${BATCH_FIRST_INDEX}+${FIRST}))
python analysis/hpc_batch/scan.py \\
    --num_of_simulations \$SIMS_IN_BATCH \\
    --sim0 \$SIM0 \\
    --pool_size ${CORES_PER_HOST}
EOF
