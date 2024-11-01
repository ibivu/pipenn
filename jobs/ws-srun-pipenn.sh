#!/bin/bash

# This script must have an argument p, n, or s.

# remove results if it's there.
rm -fr results

if [ -z "$1" ]; then
    echo "Missing parameter. You should provide p, n, or s"
    return 1
fi

PRED_TYPE=$1   
echo "PRED_TYPE is $PRED_TYPE"

USER_HOME=/scistor/informatica/rhu300
PIPENN_HOME=$USER_HOME/pipennemb
JOB_DIR=$PIPENN_HOME/jobs

srun -N 1 -p binf --cpus-per-task 24 $JOB_DIR/ws-pipenn.sh $PRED_TYPE
