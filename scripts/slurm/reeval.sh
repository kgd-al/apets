#!/bin/bash

usage(){
  echo "Usage: $0 <folder> [...ARGS]"
  echo
  echo "          Iterates over every run-* under the provided folder and reevaluate champions *if* no video is found"
  echo
  echo "          folder under which to look for"
  echo "          other arguments are passed through to the executable (reeval.py)"
}

if [ $# -lt 2 ]
then
  usage
  exit 1
fi

export MUJOCO_GL=egl

find "$1" -name "champion.json" | while champ
do

done
