#!/bin/bash
# Launch an experiment using the docker gpu image
# http://priss:5001
echo "Executing in the docker (gpu image) for experiment $1"

docker run --runtime=nvidia -it -d --rm \
  --mount src=$(pwd),target=/root/code/ctm2-drl,type=bind keshaviyengar/ctm2_drl\
  bash -c "cd /root/code/ctm2-drl/ && mpirun -np 19 --allow-run-as-root python3 run_experiment.py -e $1"
