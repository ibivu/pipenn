#!/bin/bash

ALG_DIR=$1
ALG_FILE=$2

# Please change 'my-path' to your real path.
BASE_DIR=my-path/pipenn
MODEL_DIR=$BASE_DIR/models
LOG_DIR=$BASE_DIR/logs

rm $LOG_DIR/$ALG_DIR.log
cd $BASE_DIR/$ALG_DIR
python $ALG_FILE
cp $LOG_DIR/$ALG_DIR.log $MODEL_DIR/$ALG_DIR/.
