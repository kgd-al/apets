#!/bin/bash

set -euo pipefail

user=kgd
host=ripper1
base=$user@$host:code

update(){
  dir=$1
  cd ../$dir
  shift
  echo "Updating from $(pwd): $@"
  rsync -avzhP --prune-empty-dirs $@ $base/$dir
}

line
update apets src scripts

line
update abrain src commands.sh CMakeLists.txt setup.py pyproject.toml

line
revolve_dirs=$(ls -d ../revolve/*/ | cut -d/ -f 3)
update revolve $revolve_dirs student_install.sh requirements_editable.txt README.md

if [ $# -gt 1 ] && [ $1 == '--compile' ]
then
  line
  ssh $user@$host bash <<EOF
    set -euo pipefail
    cd code/abrain
    source ../venv/bin/activate
    ./commands.sh install-cached release
EOF
fi
