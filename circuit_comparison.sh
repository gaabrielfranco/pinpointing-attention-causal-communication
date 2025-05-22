

#!/bin/bash

# Circuit comparison
for model in "gpt2-small" "EleutherAI/pythia-160m"
do
    for task in "ioi" "gt" "gp"
    do
        python3 circuit_comparison.py -m $model -t $task -th 0.2 -i
    done
done

# Compute the barplot

python3 circuit_comparison_barplot.py