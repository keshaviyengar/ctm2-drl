#!/bin/bash
# Launch an experiment using the docker gpu image

echo "Executing in the docker (gpu image):"


nvidia-docker run -it --rm --network host --ipc=host \
  --mount src=$(pwd),target=/root/code/ctm2-drl,type=bind keshaviyengar/ctm2_drl\
  bash -c "cd /root/code/ctm2-drl/ && python3 run_experiments.py"
