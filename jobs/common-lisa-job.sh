#!/bin/bash


## As it is not possible to pass parameters to SBATCH, we create this SBATCH script dynamically. Thereafter we pass the whole generated script to the
## sbatch command. This is for HPC.

cat << EOF  | sbatch
#!/bin/bash

#SBATCH -p binf
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=$3

#SBATCH --job-name=$1
#SBATCH --output=$1_%j.out

module load cuda10.2

ALG_DIR=$1
ALG_FILE=$2
SLURM_OUT=\$SLURM_SUBMIT_DIR/$1_\$SLURM_JOB_ID.out

# asks SLURM to send the TERM signal 120 seconds before end of the time limit
#SBATCH --signal=B:TERM@120

# define the handler function; note that this is not executed here, but rather; when the associated signal is sent
save_last_model()
{
    echo "time limit expired, so save the last model at \$(date)"
}
# call your_cleanup_function once we receive TERM signal
trap 'save_last_model' TERM

# Please change 'my-path' to your real path.
do_computation() {
        BASE_DIR=my-path/pipenn
        MODEL_DIR=\$BASE_DIR/models
        LOG_DIR=\$BASE_DIR/logs
        rm -f \$LOG_DIR/$ALG_DIR.log
        cd \$BASE_DIR/$ALG_DIR
        pwd
        python $ALG_FILE
        cp \$LOG_DIR/$ALG_DIR.log \$MODEL_DIR/$ALG_DIR/.
}
echo "starting computation at \$(date)"
do_computation &
wait

EOF