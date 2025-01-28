#!/bin/bash

set -euo pipefail
export PYTHONPATH=.

user=kgd
host=hex
base=$user@$host:data/apets/

exp='test'
[ $# -ge 1 ] && exp=$1

info=""
# info=--info=progress2
(
  set -x;
  rsync -avzh $info $base remote --prune-empty-dirs -f '+ '$exp'/' -f '+ run*/' -f '+ failures/' \
    -f '+ *.json' -f '+ *.dat' -f '+ *.csv' -f '+ log' -f '+ *.png' -f '+ *.mp4' -f '- *'
)

#[ -z ${VIRTUAL_ENV+x} ] && source ~/work/code/vu/venv/bin/activate

# for f in remote/$exp
# do
#   ./bin/tools/retina_summary.py $f
# done
#
# # groups=$(ls -d remote/collect_v3/*/ | cut -d '-' -f 2 | sort -u)
# # echo $groups
# # cd remote/collect_v3
# # sorted(){
# #   find *-$1-100K -name 'best.json' \
# #   | xargs jq -r '"\(input_filename) \(.fitnesses.collect)"' \
# #   | sort -k2gr \
# #   | awk -F. -ve=$2 '{print $1"."e}'
# # }
# # for g in $groups
# # do
# #   if ls *-$g-*/best.trajectory.png >/dev/null 2>&1
# #   then
# #     montage -geometry +0+0 -label '%d' $(sorted $g trajectory.png) $g.trajectories.png
# #   fi
# #
# #   if ls *-$g-*/best.cppn.png >/dev/null 2>&1
# #   then
# #     montage -geometry '256x256>+0+0' -label '%d' $(sorted $g cppn.png) $g.cppn.png
# #   fi
# #
# #   for t in weight leo bias
# #   do
# #     if ls *-$g-*/best.cppn.$t.png >/dev/null 2>&1
# #     then
# #       montage -geometry '+10+10' -label '%d' $(sorted $g cppn.$t.png) $g.cppn.$t.png
# #     fi
# #   done
# # done
# # cd -
#
# pgrep -a feh | grep $exp'/$' > /dev/null || feh -.Z --reload 10 remote/$exp/ 2>/dev/null &

find remote/$exp/ -name '*.mp4' | xargs vlc --no-random --no-loop 2> /dev/null
