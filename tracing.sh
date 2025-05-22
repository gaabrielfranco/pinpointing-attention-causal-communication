#!/bin/bash

for model in "gpt2-small" "EleutherAI/pythia-160m" "gemma-2-2b"
do
    for task in "ioi" "gt" "gp"
    do
        if [ $task == "gp" ]
        then
            python3 tracing.py -m $model -t $task -n 100
        else
            python3 tracing.py -m $model -t $task -n 256
        fi
    done
done