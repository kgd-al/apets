#!/bin/bash

echo "2"

exp="test"
if [ $# -ge 1 ]
then
  exp=$1
fi

squeue -o "%.18i %.9P %.21j %.8u %.2t %.10M %.6D %R"
echo

completed_str=""
errors_str=""
running_str=""

while read file
do
  name=$(basename $(dirname $file))

  slurm_base="$HOME/data/apets/slurm_logs/$exp/$name"

  slurm_out=$slurm_base.out
  [ ! -f $slurm_base.out ] && slurm_out=$(dirname $file)/slurm.out
  errors=$(tac "$slurm_out" | grep -m 1 "Error")
  if [ -n "$errors" ]
  then
    errors_str="$errors_str\033[31m$name\033[0m $errors\n"
#    continue
  fi

  slurm_err=$slurm_base.err
  [ ! -f $slurm_base.err ] && slurm_err=$(dirname $file)/slurm.err
  if [ $(wc -l < $slurm_err) -gt 0 ]
  then
    errors_str="$errors_str\033[31m$name\033[0m $(cat $slurm_err)\n"
    continue
  fi

  completed=$(tac "$file" | grep -m 1 'Completed' | cut -d ' ' -f-2,5-)
  if [ -n "$completed" ]
  then
    completed_str="$completed_str\033[32m$name\033[0m $completed\n"
    continue
  fi

#  gen=$(jq ._generation $(dirname $file)/evolution.json)
#  gen=$(grep -m 1 -o '"_generation": [0-9]*,' $(dirname $file)/evolution.json | cut -d' ' -f2 | tr -d ,)
#  if [ -n "$gen" ]
#  then
#    running_str="$running_str\033[33m$name\033[0m $gen\n"
#  else
#    running_str="$running_str\033[37m$name\033[0m Starting\n"
#  fi

  # get header
  if [ -z "$header" ]
  then
    header=$(grep -m 1 '[[]EVO' $(dirname $file)/log| cut -c 43-)
    running_str="${running_str}header $header\n"
  fi

  evo=$(tac $(dirname $file)/log | grep -m 1 '[[]EVO' | cut -c 43-)
  if [ -n "$evo" ]
  then
    running_str="$running_str\033[33m$name\033[0m $evo\n"
  else
    running_str="$running_str\033[37m$name\033[0m Starting\n"
  fi

done < <(ls -v ~/data/apets/$exp/*/log)

printf "$running_str" | column -t -o ' '
printf "$errors_str"
printf "$completed_str" | column -t -o ' '
