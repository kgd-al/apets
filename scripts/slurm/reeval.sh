#!/bin/bash

usage(){
  echo "Usage: $0 <folder> [...ARGS]"
  echo
  echo "          Reevaluate every champion.json under <folder> *if* no video is found"
  echo
  echo "          folder under which to look for"
  echo "          other arguments are passed through to the executable (reeval.py)"
}

if [ $# -lt 1 ]
then
  usage
  exit 1
else
  root=$1
  shift 1
fi

export MUJOCO_GL=egl

[[ $(uname -n) =~ "ripper" ]] && module load dot
[[ $(uname -n) =~ "ripper" ]] && echo "Loading dot"

find "$root" -name "champion.json" | while read champ
do
  movie=$(dirname $champ)/$(basename $champ .json).mp4
  [ -f "$movie" ] && continue
  python src/basic_attempt/rerun.py "$champ" --movie True --headless True "$@" || exit 2
#  --render png
done
