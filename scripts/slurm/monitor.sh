#!/bin/bash

exp="test"
if [ $# -ge 1 ]
then
  exp=$1
fi

pretty_time(){
  tokens=$(awk '{ printf "%d %d %d %g", $1 / 86400, ($1 / 3600) % 24, $1 / 60 % 60, $1 % 60 }' <<< "$1")
  read D H M S <<< "$tokens"
  [[ $D > 0 ]] && printf "%d days " $D
  [[ $H > 0 ]] && printf "%d hours " $H
  [[ $M > 0 ]] && printf "%d minutes " $M
  [[ $S > 0 ]] && printf "%g seconds " $S
  printf "\n"
}

squeue -o "%.18i %.9P %.21j %.8u %.2t %.10M %.6D %R"
echo

errors_str=""
running_str=""

while read file
do
  folder=$(dirname $file)
  name=$(basename $folder)

  slurm_base="$HOME/data/apets/slurm_logs/$exp/$name"

  slurm_out=$slurm_base.out
  [ ! -f $slurm_base.out ] && slurm_out=$(dirname $file)/slurm.out
  errors=$(tac "$slurm_out" | grep -m 1 "Error")
  if [ -n "$errors" ]
  then
    errors_str="$errors_str\033[31m$name\033[0m $errors\n"
    continue
  fi

  slurm_err=$slurm_base.err
  [ ! -f $slurm_base.err ] && slurm_err=$folder/slurm.err
#   if [ $(grep -v -e "GLFWError" -e '^$' $slurm_err | wc -l) -gt 0 ]
  if [ $(cat $slurm_err | wc -l) -gt 0 ]
  then
    errors_str="$errors_str\033[31m$name\033[0m $(cat $slurm_err)\n"
    continue
  fi

  state=33
  completed=$(tac "$file" | grep -m 1 'Completed' | cut -d ' ' -f-2,5-)
  if [ -n "$completed" ]
  then
    state=32
    completed=$(sed 's/.*Completed evolution in //' <<< $completed)
  fi

  # get header
  if [ -z "$header" ]
  then
    header=$(grep -m 1 '[[]EVO' $folder/log| cut -c 43- | tr -s ' ' '|')
    running_str="${running_str}header|${header}ETA\n"
  fi

  evo=$(tac $folder/log | grep -m 1 '[[]EVO' | cut -c 43- | awk -vOFS="|" '{$1=$1;print}')
  if [ -n "$evo" ]
  then
    running_str="$running_str\033[${state}m$name\033[0m|$evo"

    if [ -n "$completed" ]
    then
      eta=$completed
    else
      elapsed=$(($(date +%s) - $(stat -c "%W" $folder/log)))
      curr_gen=$(cut -d '|' -f1 <<< $evo)
      if [ -f $folder/evolution.json ]
      then
        target_gen=$(jq .config.generations $folder/evolution.json)
        eta=$(awk '{ print $2 ? $1 * ($3 - $2)/$2 : "inf" }' <<< "$elapsed $curr_gen $target_gen")
        eta=$(pretty_time $eta)
      else
        eta="N/A"
      fi
    fi

    running_str="$running_str|\033[${state}m$eta\033[0m\n"
  else
    running_str="$running_str\033[37m$name\033[0m|Starting\n"
  fi

done < <(ls -v ~/data/apets/$exp/*/log)

printf "$running_str" | column -ts'|' -o ' '
printf "$errors_str"
