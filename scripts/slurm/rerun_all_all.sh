#!/bin/bash

#SBATCH --job-name=rerun-all-all
#SBATCH --partition=batch
#SBATCH --nodes=1
###SBATCH --exclusive
#SBATCH --time=1:00:00

usage(){
    echo "Usage: $0 FOLDER [FOLDERS...]"
    echo "       Searches folder(s) structure for runs to re-evaluate"
}

base=$(realpath $(dirname $0)/../..)

duration=15

find $@ -name "summary.csv" | sort | while read f
do
  folder=$(dirname $f)
  trainer=$(cut -d/ -f 5 <<< $f)
  reward=$(cut -d/ -f 6 <<< $f)
  arch=$(cut -d/ -f 7 <<< $f)
  nn=$(cut -d- -f 1 <<< $arch)
  seed=$(cut -d/ -f 8 <<< $f | cut -d- -f 2)

  if [ $nn == "mlp" ]
  then
    depth="--depth $(cut -d- -f2 <<< $arch)"
    width="--width $(cut -d- -f3 <<< $arch)"
    neighborhood=
  else
    neighborhood="--neighborhood $(cut -d- -f2 <<< $arch)"
    depth=
    width=
  fi

#  echo $f >&2
  if [ $trainer == "rlearn" ]
  then
    echo src/apets/hack/rl/train.py --rerun $folder/model.zip \
      --rotated --reward $reward $depth $width --seed $seed --headless -T $duration
  else
    echo src/apets/hack/cma_es/evolve.py -o $folder --arch $nn --reward $reward \
      --rerun $depth $width --rotated --headless -T $duration --budget 10000 --seed $seed
  fi
#done | xargs -t -L1 -P 0 python
done | while read cmd
do
  python $cmd
done