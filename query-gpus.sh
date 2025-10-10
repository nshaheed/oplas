#!/usr/bin/bash

# get a list of all of the gpus that are currently running as jobs

squeue --me -O "Nodelist" | tail -n +2 | awk '{$1=$1};1' | xargs -n1 -I{} ssh {} -t "nvidia-smi --query-gpu=name --format=csv,noheader"
