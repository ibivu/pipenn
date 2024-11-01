#!/bin/bash

PIPENN_HOME=/scistor/informatica/rhu300/pipennemb

ALG_DIR=utils
ALG_FILE=GenerateMinFeatures.py
ALG_PATH=$PIPENN_HOME/$ALG_DIR

# Use this one to generate a table from a fasta file and also to generate protbert embeddings for each protein in the fasta file.
# Fasta file must contain two lines for each protein: (1) starts with a line ">id " and (2) the sequence.
echo "starting generation of embeddings and pipenn-features at $(date) from $(pwd) with args $1 $2"
python3 $ALG_PATH/$ALG_FILE $PIPENN_HOME $1 $2

