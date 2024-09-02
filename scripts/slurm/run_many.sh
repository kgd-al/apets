#!/bin/bash

usage(){
  echo "Usage: $0 <name> <replicates> <generations> <populations>"
}

if [ $# -ne 4 ]
then
  usage
  exit 1
fi

name=$1
count=$2
generations=$3
population=$4

for seed in $(seq 0 $(($count-1)))
do
  $(dirname $0)/run.sh $name $seed $generations $population
done
