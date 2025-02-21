#!/bin/bash

usage

base=$(realpath $(dirname $0)/../..)

echo "Hi"
experiments=$(awk '
    /class ExperimentType/{enum=1; next}
    enum==1&&/^$/{enum=0; exit}
    enum==1{print $1}
' $base/src/main/config.py)

for vision in 0 1
do
    for
    do
        echo "$(dirname $0)/run.sh
