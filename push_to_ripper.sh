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

update apets src
update abrain src commands.sh CMakeLists.txt setup.py pyproject.toml

revolve_dirs=$(ls -d ../revolve/*/ | cut -d/ -f 3)
update revolve $revolve_dirs student_install.sh requirements_editable.txt README.md

if [ $# -gt 1 ] && [ $2 == '--compile' ]
then
  ssh $user@$host bash <<EOF
    set -euo pipefail
    cd code/abrain
    source ../venv/bin/activate
    ./commands.sh install-cached release
EOF
fi
