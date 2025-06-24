#!/bin/bash

usage(){
    echo "Usage: $0 <seeds> [ARGS...]"
    echo "       Schedules lots of runs"
    echo "       As with run_rl.sh, seeds are used to populate a slurm array and ARGS"
    echo "        are passed directly to the executable"
}

seeds=$1
shift

base=$(realpath $(dirname $0)/../..)

export SILENT_SKIP_EXISTING=1

# First MLP-based
for vision in None 6,4
do
    vflag=""
    [ "$vision" != "None" ] && vflag="-vision"

    for EXP in $experiments
    do
        exp=$(tr '[:upper:]_' '[:lower:]-' <<< $EXP)$vflag
        duration=${durations[$EXP$vflag]}
        export SLURM_DURATION=$duration
        if [[ $(uname -n) =~ "ripper" ]]
        then
            $(dirname $0)/run.sh $exp $seeds --vision $vision --experiment $EXP \
                --generations 100 --population 100 $@
        else
            for i in 0 1
            do
                folder=tmp/run_all_test/$exp$vflag/run-$i
                [ -d $folder ] && continue
                python src/main/main.py --experiment $EXP --vision "$vision" --overwrite False --seed 0 --data-root $folder $@ || exit 42
            done
        fi
    done
done
