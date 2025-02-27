#!/bin/bash

usage(){
    echo "Usage: $0 <seeds> [ARGS...]"
    echo "       Schedules runs for all experiments defined in src/main/config.py"
    echo "       As with run.sh, seeds are used to populate a slurm array and ARGS"
    echo "        are passed directly to the executable"
}

seeds=$1
shift

base=$(realpath $(dirname $0)/../..)

experiments=$(awk '
    /class ExperimentType/{enum=1; next}
    enum==1&&/^$/{enum=0; exit}
    enum==1{print $1}
' $base/src/main/config.py)

declare -A durations=(
 [LOCOMOTION]=2:00:00
 [PUNCH_ONCE]=2:00:00
 [PUNCH_AHEAD]=2:00:00
 [PUNCH_BACK]=4:00:00
 [PUNCH_THRICE]=8:00:00
 [PUNCH_TOGETHER]=10:00:00
 [LOCOMOTION-vision]=5:00:00
 [PUNCH_ONCE-vision]=5:00:00
 [PUNCH_AHEAD-vision]=5:00:00
 [PUNCH_BACK-vision]=10:00:00
 [PUNCH_THRICE-vision]=20:00:00
 [PUNCH_TOGETHER-vision]=100:00:00
)

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
