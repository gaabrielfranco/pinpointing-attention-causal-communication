#!/bin/bash

# Intervention Experiments
for model in "gpt2-small" "EleutherAI/pythia-160m"
do
    for task in "ioi" "gt" "gp"
    do
        if [ $task == "gp" ]
        then
            python3 interventions.py -m $model -t $task -n 100
        else
            python3 interventions.py -m $model -t $task -n 256
        fi
    done
done

python3 interventions_plots.py