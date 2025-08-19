#!/bin/bash

usage(){
    echo "Usage: $0 FOLDER [FOLDERS...]"
    echo "       Searches folder(s) structure for runs to re-evaluate"
}

base=$(realpath $(dirname $0)/../..)

duration=15

#find $@ -name "summary.csv" | sort | while read f
cat ~/data/pareto | while read f
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
      --rotated --reward $reward $depth $width --seed $seed --headless -T $duration \
      --introspective
  else
    echo src/apets/hack/cma_es/evolve.py -o $folder --arch $nn --reward $reward \
      --rerun $depth $width $neighborhood --rotated --headless -T $duration --seed $seed \
      --introspective
  fi
done > .tasks

ntasks=$(wc -l .tasks | cut -d ' ' -f1)
echo "$ntasks tasks"

#
#sbatch --exclusive --nodes=1 --job-name=rerun-all-all --time=1:00:00 <<EOF
##!/bin/bash
#
##SBATCH --ntasks $ntasks
#
#W=\$SLURM_JOB_CPUS_PER_NODE
#W=4
#
#cat .tasks | while read cmd
#do
#  echo "\$(jobs -p | wc -l)" "\$W"
#  while [ "\$(jobs -p | wc -l)" -ge "\$W" ]
#  do
#    echo "\$(jobs -p | wc -l)" "\$W"
#    sleep 1
#  done
#  echo \$cmd
#  srun --ntasks=1 --cpus-per-task=1 --exact python \$cmd &
#done
#
#wait
#rm .tasks
#
#EOF

sbatch --nodes=1 --ntasks 1 --cpus-per-task 1 --job-name=rerun-all-all --time=10:00:00 <<EOF
#!/bin/bash

i=0
cat .tasks | while read cmd
do
#  printf "\n\033[32m[%6.2f%%]\033[0m " \$(( 100 * \$i / $ntasks ))
  awk -vi=\$i -vn=$ntasks 'BEGIN{printf "\n\033[32m[%6.2f%%]\033[0m ", 100 * i / n}'
  i=\$((\$i+1))
  echo \$cmd
  python \$cmd
done

printf "\n\033[32m[100.00%%] Done\033[0m\n"

rm .tasks

EOF
