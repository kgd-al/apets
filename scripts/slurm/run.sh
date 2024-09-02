#!/bin/bash

set -euo pipefail

usage(){
  echo "Usage: $0 <name> <seed> <generations> <population>"
}

if [ $# -ne 4 ]
then
  usage
  exit 1
fi

name=$1
seed=$2
data_folder=$HOME/data/apets/$name/run$seed
generations=$3
population=$4
job_name=apets-$name-$seed-${population}x${generations}

if [ -d $data_folder ]
then
  echo "Target directory <$data_folder> already exists" >&2
  exit 2
fi

mkdir -p $data_folder

sbatch -o $data_folder/slurm.out -e $data_folder/slurm.err <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00               # Maximum runtime (D-HH:MM:SS)

source ~/code/venv/bin/activate

echo "Seed is $seed"
echo "Saving data to $data_folder"
echo "Evolving $population for $generations generations"
xvfb-run python src/basic_attempt/main.py --overwrite \
  --seed $seed --generations $generations --population-size $population \
   --data-root $data_folder
EOF
