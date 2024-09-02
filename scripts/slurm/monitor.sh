#!/bin/bash

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

  errors=$(tac $(dirname $file)/slurm.out | grep -m 1 "Error")
  if [ -n "$errors" ]
  then
    errors_str="$errors_str\033[31m$name\033[0m $errors\n"
    continue
  fi

  error_log=$(dirname $file)/slurm.err
  if [ $(wc -l < $error_log) -gt 0 ]
  then
    errors_str="$errors_str\033[31m$name\033[0m $(cat $error_log)\n"
    continue
  fi

  completed=$(tac "$file" | grep -m 1 'Completed' | cut -d ' ' -f-2,5-)
  if [ -n "$completed" ]
  then
    completed_str="$completed_str\033[32m$name\033[0m $completed\n"
    continue
  fi

  gen=$(tac "$file" | grep -m 1 '[[]Gen' | cut -d ' ' -f-2,5-)
  if [ -n "$gen" ]
  then
    running_str="$running_str\033[33m$name\033[0m $gen\n"
  else
    running_str="$running_str\033[37m$name\033[0m Starting\n"
  fi
done < <(ls -v ~/data/apets/$exp/*/log)

printf "$running_str" | column -t -o ' '
printf "$errors_str"
printf "$completed_str" | column -t -o ' '
