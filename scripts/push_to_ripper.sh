#!/bin/bash

set -euo pipefail

user=kgd
host=hex
base=$user@$host:code

update(){
  dir=$1
  cd "../$dir"
  shift
  echo "Updating from $(pwd): $*"
  rsync -avzhP --prune-empty-dirs -f '- *.pyc' "$@" "$base/$dir"
}

line
update apets src scripts requirements.txt

line
update abrain -f '- *.so' -f '- .egg-info/' src commands.sh CMakeLists.txt setup.py pyproject.toml

line
revolve_dirs=$(ls -d ../revolve/*/ | cut -d/ -f 3)
update revolve $revolve_dirs student_install.sh requirements_editable.txt README.md

if [ $# -ge 1 ] && [ "$1" == '--compile' ]
then
  line
  echo "Compiling abrain"
  ssh $user@$host bash <<EOF
    set -euo pipefail
    cd code/abrain
    source ../venv/bin/activate
    ./commands.sh install-editable release
EOF
fi
