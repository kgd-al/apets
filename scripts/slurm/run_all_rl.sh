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

prefix(){
  printf "[%s] " "$(date)"
}

(
  # First MLP-based
  for reward in distance kernels
  do
    echo $reward/mlp-0-0 $seeds --reward $reward --policy mlp --depth 0 $@

    for layers in 1 2
    do
      for width in 1 2 4 8 16 32 64 128
      do
        echo $reward/mlp-${layers}-${width} $seeds --reward $reward --policy mlp --depth $layers --width $width $@
      done
    done
  done
) | while read cmd
do
  while [ $(squeue -u kgd| wc -l) -gt 20 ]
  do
    prefix
    printf "Waiting for some room in queue\r"
    sleep 10
  done

  prefix
  $(dirname $0)/run_rl.sh $cmd
  sleep 1
done