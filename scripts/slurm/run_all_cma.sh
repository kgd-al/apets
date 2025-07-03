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
    for neighborhood in 1 2 3
    do
      echo $reward/cpg-$neighborhood $seeds --reward $reward --neighborhood $neighborhoodls $@
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