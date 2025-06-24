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
for reward in distance #kernels
do
  for layers in 0 1 2
  do
    for width in 8 16 32 64 128
    do
      $(dirname $0)/run_rl.sh ppo/spider45/mlp2 0-9 --simulation-time 30 --body spider --rotated --timesteps 200000 --reward distance --depth $layers --width $width
    done
  done
done
