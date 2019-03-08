#!/bin/bash
# Launch an experiment using the docker gpu image
# http://priss:5001

echo "Executing in the docker (gpu image):"


docker run --runtime=nvidia -it --rm \
  -p 5000:8888 -p 5001:6006\
  -e DISPLAY=$DISPLAY \
  -v $HOME/.Xauthority:/home/developer/.Xauthority \
  --mount src=$(pwd),target=/root/code/ctm2-drl,type=bind keshaviyengar/ctm2_drl\
  bash -c "cd /root/code/ctm2-drl/ && python3 run_experiments.py"
