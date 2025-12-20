#!/bin/bash
#PBS -N rftune
#PBS -q workq
#PBS -l nodes=node04:ppn=40
#PBS -j oe
#PBS -o rftune.log

# Load Anaconda & activate environment
source /home/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate tfex

# Change to job submission directory
cd "$PBS_O_WORKDIR"

echo "Process started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"

# Run the script

scriptname="rf_analysis.py"
DATA_DIR="/NAS/protem/midhun/AI4KLIM/DATASETS/DATASET_IND32M/DATA_TRANS/numpy"
SAVE_DIR=$(dirname "$PBS_O_WORKDIR")

python ${scriptname} ${DATA_DIR} ${SAVE_DIR} 11323 true

echo "Process finished at: $(date)"