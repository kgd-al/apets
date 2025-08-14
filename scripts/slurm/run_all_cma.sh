#!/bin/bash

usage(){
    echo "Usage: $0 <seeds> [ARGS...]"
    echo "       Schedules lots of runs"
    echo "       As with run_cma.sh, seeds are used to populate a slurm array and ARGS"
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
  for reward in distance kernels
  do
    for neighborhood in 0 2 4 6
    do
      echo $reward/cpg-$neighborhood $seeds --reward $reward --arch cpg --neighborhood $neighborhood $@
    done
  done
  for reward in distance kernels
  do
    echo $reward/mlp-0-0 $seeds --reward $reward --arch mlp --depth 0 --width 0 $@
    for depth in 1 2
    do
      for width in 1 2 4 8 16 32 64 128
      do
        echo $reward/mlp-$depth-$width $seeds --reward $reward --arch mlp --depth $depth --width $width $@
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
  $(dirname $0)/run_cma.sh $cmd
  sleep 1
done