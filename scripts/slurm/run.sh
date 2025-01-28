#!/bin/bash

usage(){
  echo "Usage: $0 <name> <seeds> [...ARGS]"
  echo "          name is the name of the experiment (top-level folder)"
  echo "          seeds will populate SLURM's array field"
  echo "          any other argument are passed through to the executable"
}

if [ $# -lt 3 ]
then
  usage
  exit 1
fi

name=$1
seeds=$2
shift 2

data_root=$HOME/data/apets/
mkdir -p "$data_root"

slurm_logs=$data_root/slurm_logs/$name/
mkdir -p "$slurm_logs"

job_name=apets-$name

slurm_logs_base="$slurm_logs/run-%a"

sbatch -o "$slurm_logs_base.out" -e "$slurm_logs_base.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=$seeds
#SBATCH --time=10:00:00               # Maximum runtime (D-HH:MM:SS)

source ~/code/venv/bin/activate

seed=\$SLURM_ARRAY_TASK_ID
data_folder=$data_root$name/run-\$seed

date
echo "Seed is \$seed"
echo "Saving data to \$data_folder"
echo "Additional arguments: $@"

#xvfb-run
export MUJOCO_GL=egl
python src/basic_attempt/main.py --overwrite False --seed \$seed --data-root \$data_folder $@

for ext in out err
do
  mv -v $slurm_logs/run-\$seed.\$ext \$data_folder/slurm.\$ext
done

EOF
