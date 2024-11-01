#!/bin/bash

PIPENN_HOME=/scistor/informatica/rhu300/pipennemb
JOB_DIR=$PIPENN_HOME/jobs

ALG_DIR=rnet-ppi
ALG_FILE=rnet-XD-ppi-keras.py
PRED_TYPE=$1

source $JOB_DIR/common-lisa-job.sh $ALG_DIR $ALG_FILE $PRED_TYPE
