#!/bin/bash

usage(){
  echo "Usage: $0 <folder>"
  echo
  echo "       Looks for .mp4 files under the provided folder and moves them to the ./video"
  echo "        folder. Relative paths are preserved *except* that run-N/champion.mp4 is replaced"
  echo "        with <fitness>.mp4 and the remote/ prefix is removed"
}

if [ $# -ne 1 ]
then
  usage
  exit 1
fi

mkdir -pv ./videos

files=0
duplicates=0
while read file
do
  json=$(dirname $file)/$(basename $file .mp4).json
  fitness=$(jq .fitness $json | xargs printf "%.2g")
  dst=./videos/$(sed -e 's|remote/||' -e 's|run-[0-9]*/.*.mp4|'$fitness'.mp4|' <<< $file)
  dst_folder=$(dirname $dst)
  dst_base=$(basename $dst .mp4)
  if [ -f $dst ]
  then
    n=$(ls $dst_folder/$dst_base* | wc -l)
    dst=$dst_folder/${dst_base}_$n.mp4
    duplicates=$((duplicates+1))
  fi

  mkdir -pv $dst_folder
  cp -v $file $dst

  files=$((files + 1))
done < <(find $1 -name "*.mp4")

echo "Found $duplicates individuals with overlapping filenames"
echo $files
if [ $duplicates -eq $files ]
then
  printf "\033[31mEvery processed file was a duplicate. This does not look good!\033[0m\n"
fi
