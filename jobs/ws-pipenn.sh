#!/bin/bash

PRED_TYPE=$1

USER_HOME=/scistor/informatica/rhu300
PIPENN_HOME=$USER_HOME/pipennemb
JOB_DIR=$PIPENN_HOME/jobs
CONDA_HOME=$USER_HOME/miniconda3

# your input file and output folder
USERDS_INPUT_DIR=.
USERDS_INPUT_FILE=$USERDS_INPUT_DIR/input.fa
USERDS_OUTPUT_FILE=$USERDS_INPUT_DIR/prepared_userds.csv

USERDS_OUTPUT_DIR=.
PIPENN_ENV=pipenn-2.0

prepare_run() {
  source $CONDA_HOME/etc/profile.d/conda.sh
  conda deactivate
  conda deactivate
}

set_pipenn_env() {
  export PYTHONPATH=$PIPENN_HOME/utils:$PIPENN_HOME/config:$PIPENN_HOME/ann-ppi:$PIPENN_HOME/rnn-ppi:$PIPENN_HOME/dnet-ppi:$PIPENN_HOME/rnet-ppi:$PIPENN_HOME/unet-ppi:$PIPENN_HOME/cnn-rnn-ppi:$PIPENN_HOME/ensnet-ppi
  conda activate $PIPENN_ENV
  echo "PIPENN_HOME is: $PIPENN_HOME and PIPENN_ENV is: $PIPENN_ENV"
}

gen_pipenn_features() {
  source $JOB_DIR/utils/fgt-lisa-job.sh $USERDS_INPUT_FILE $USERDS_OUTPUT_FILE
}

gen_pipenn_preds() {
	ALG_DIR=dnet-ppi
	ALG_JOB=$JOB_DIR/dnet-ppi/dnet-lisa-job.sh
	#ALG_DIR=ensnet-ppi
	#ALG_JOB=$JOB_DIR/ensnet-ppi/ensnet-lisa-job.sh
	#$(mkdir logs)
        echo "Prediction type is: $PRED_TYPE"
	source $ALG_JOB $PRED_TYPE 2> reza_log.txt
	$(mv $ALG_DIR results)
}


#:'##Comment##
echo "preparing computation at $(date) ..."  | tee message.php
prepare_run
echo "setting pipenn environment at $(date) ..."  | tee message.php
set_pipenn_env
echo "generating PIPENN features at $(date) ..."  | tee message.php
gen_pipenn_features
#'##Comment##

echo "executing prediction at $(date) ..."  | tee message.php
gen_pipenn_preds
wait

