#!/bin/bash
set -ex

MODE=full BIOTEXT=$BIOTEXT snakemake --cores $CORES

python mergeDBs.py --inDir working/ --outDB metadata.db

