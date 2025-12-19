#!/bin/bash

scriptname="rf_shap_analysis.py"
DATA_DIR="/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/DATASETS/DATASET_IND32M/DATA_TRANS/numpy"
SAVE_DIR="/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/EXPSDATA/P03B/P03B.GLM.E5AUXV.STD"

python ${scriptname} ${DATA_DIR} ${SAVE_DIR} 
