#!/bin/bash

for model in "gpt2-small" "pythia-160m" "gemma-2-2b"
do
    for task in "ioi" "gt" "gp"
    do
        if [ "$task" == "gt" ] && [ "$model" == "gemma-2-2b" ];
        then
            continue
        else
            python3 graphs_unification.py -m "$model" -t "$task"
        fi
    done
done