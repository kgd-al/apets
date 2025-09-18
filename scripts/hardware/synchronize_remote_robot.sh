#!/bin/bash

usage(){
  echo "Usage: $0 <local_folder> <ip> <password>"
  echo
  echo "        Schedules repo-wide update of the repository on the robot at the given address (password is needed)"
  echo "        Also sets up a sshfs filesystem"
  echo "    NB: Yes this is brittle and ugly. Deal with it."
}

if [ $# -ne 3 ]
then
  echo "Invalid number of arguments"
  usage
  exit 2
fi

local_folder=$1
ip=$2
password=$3

pkill sshfs
sleep 2

rmdir -p $local_folder
mkdir -p $local_folder
sleep 2

sshfs robo@$ip:/home/robo $local_folder || exit 1
sleep 2

echo "Connected to $local_folder:"
ls $local_folder || exit 1

sleep 5

auto_action_on_write.sh -r src/ 'clear; date; echo; rsync -avzh src/apets/ '$local_folder'/kgd_apets/src/apets/ --include="*/" --include="*.py" --exclude="*"; printf "\a\n"'
