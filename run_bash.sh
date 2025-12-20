#!/bin/bash

scriptname="rf_analysis.py"
DATA_DIR="/NAS/protem/midhun/AI4KLIM/DATASETS/DATASET_IND32M/DATA_TRANS/numpy"
SAVE_DIR=$(dirname "$PWD")

python ${scriptname} ${DATA_DIR} ${SAVE_DIR} 11323 true
