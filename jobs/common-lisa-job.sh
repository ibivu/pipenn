#!/bin/bash

PIPENN_HOME=/scistor/informatica/rhu300/pipennemb/

ALG_DIR=$1
ALG_FILE=$2
ALG_PATH=$PIPENN_HOME/$ALG_DIR
PRED_TYPE=$3

mkdir $ALG_DIR
echo "starting computation at $(date) from $(pwd)"
python3 $ALG_PATH/$ALG_FILE $PIPENN_HOME $PRED_TYPE
