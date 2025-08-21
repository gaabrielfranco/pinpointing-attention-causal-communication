# Pinpointing Attention-Causal Communication in Language Models

This repository contains the code to reproduce the findings of the paper "Pinpointing Attention-Causal Communication in Language Models."

# Dependencies

```sh
pip install -r requirements.txt
```

# Reproducing The Paper Results

Follow these steps to reproduce the results from the paper.

## Step 0: Download Cached Data (Optional)

We provide files with cached data to speed up the reproduction process. Downloading these files and extracting them in the main folder is recommended if you don't want to recompute everything, which can be time-consuming.

The available files are:
- Cached SVD of $\Omega$ for Pythia-160M (used in Step 1): [Link](https://figshare.com/s/65ca64a0cce99490692b)
- Tracing results (in the `tracing_results` folder, used in Step 2): [Link](https://figshare.com/s/5f6672b23527ba99c541)
- Cached control signals (in the `control_signals_cache` folder, used in `control_signals.ipynb` for Step 5): [Link](https://figshare.com/s/48b884dbeb4efc132a34)

You can recompute all this data. However, it can take a significant amount of time.
**For a faster reproduction**, we suggest either using **GPT-2 small** only by modifying the provided `.sh` files or downloading the cache data.

The file with the cached SVD of $\Omega$ for Gemma-2 2B is too large and could no be shared. Instructions to compute that are in the Step 1.

## Step 1: Compute SVD of $\Omega$

This step calculates the Singular Value Decomposition (SVD) of the $\Omega$ matrix.

For Pythia-160M and Gemma-2B, we used `compute_all_new_omega_svd_pythia.py` and `compute_all_omega_svd_gemma.py` respectively to compute the SVD and save it to disk. For GPT-2 small, the SVD is computed on-the-fly. This computation only needs to be done once.

To run this step:
```sh
./compute_omega.sh
```
This script will generate:
- `matrices_pythia-160m.hdf5` for Pythia-160M.
- Files named `gemma-2-2b-omega/matrices_{layer}.hdf5` for each layer of Gemma-2B.

## Step 2: Run Tracing (`tracing.py`)

This step performs the tracing analysis.

To run tracing for all models and tasks:
```sh
./tracing.sh
```
This will create:
- Graphs in the `traced_graphs/` folder.
- A dictionary with traced information (e.g., singular values used) in the `tracing_results/` folder.
Both are needed for the paper's results.

**Note for Gemma-2B**: Due to memory limits, we used the `-b` option to process data in batches. You may need to adjust the batch size inside the `tracing.py` file based on your available memory.

More details about `tracing.py` are in the "Extra Documentation" section.

## Step 3: Sparse Attention Decomposition

The Jupyter notebook `sparse_attention_decomposition_plots.ipynb` uses the tracing results to plot the sparse attention decomposition phenomena. These plots are saved in `figures/sparse_attn_decomp`.

## Step 4: Build Circuits

### Step 4.1: Unify Graphs

This combines individual trace graphs.
```sh
./unify_graphs.sh
```
This command creates the `combined_graphs` folder.

### Step 4.2: Construct Intervention Graphs

This step identifies causal edges in the combined graph.
```sh
./intervention_graphs.sh
```
This script runs two main processes:
1.  **`interventions.py`**: Runs interventions in the model to measure the causal effect of each edge in the combined graph on downstream performance.
2.  **`interventions_graph_pruning.py`**: Removes edges from the graph based on their measured causal effect.

### Step 4.3: Compute Circuits

Circuits are computed based on a threshold for the causal effect of each edge. We used a threshold of 0.2 for all experiments.
```sh
./circuit_comparison.sh
```
This creates the `circuit_comparison` folder with the results. It also runs `circuit_comparison_barplot.py` to plot a comparison of metrics and saves it in the `figures/circuit_comparison` folder.

## Step 5: Run Intervention Experiments

This step runs various intervention experiments.
```sh
./intervention_experiments.sh
```
This creates the `intervention_data` folder with the results. It also runs `interventions_plots.py` to generate all intervention figures and saves them in the `figures/interventions` folder.


## Step 6: Analyze Control Signals

The Jupyter notebook `control_signals.ipynb` performs the control signals analysis and generates the corresponding plots. These plots are saved in `figures/control_signals`.


## Step 7: Export Graphs for Visualization

The example graphs used in the paper are in the `traced_graphs_with_tokens` folder. To trace with tokens, use the `-tt` option when running `tracing.py`. See the "Extra Documentation" section for more details on `tracing.py`.

After tracing with tokens, run:
```sh
python3 export_visualize_graphs.py
```
This script creates the `traced_graphs_vis_with_tokens` folder, which contains the graphs ready for visualization. We used [Cytoscape](https://cytoscape.org/) for visualization.

All the plots generated with Cytoscape are in the `graph_plots` folder.


# Extra Documentation

## More Details About `tracing.py`

The `tracing.py` script is used to trace model activations and identify attention patterns.

**Usage:**
```bash
python tracing.py -m <model_name> -t <task> -n <num_prompts> [options]
```

### Parameters

* `-m, --model_name`
    * **Description**: Name of the language model.
    * **Choices**: `gpt2-small`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-160m-deduped`, `gemma-2-2b`
    * **Required**: Yes
* `-t, --task`
    * **Description**: Specific task to perform.
    * **Choices**: `ioi` (Indirect Object Identification), `gt` (Greater Than), `gp` (Gender Pronoun)
    * **Required**: Yes
* `-n, --num_prompts`
    * **Description**: Number of prompts to process.
    * **Type**: `int`
    * **Required**: Yes
* `-d, --device`
    * **Description**: Device to run the model on.
    * **Choices**: `cpu`, `cuda`
    * **Default**: `cpu`
* `-s, --seed`
    * **Description**: Random seed for reproducible results.
    * **Type**: `int`
    * **Default**: `0`
* `-le, --lazy_eval`
    * **Description**: Use this flag for lazy evaluation of $\Omega$ matrix computation. This is helpful if the SVD of $\Omega$ is not already cached. (No value needed)
    * **Action**: `store_true`
* `-b, --batch`
    * **Description**: Batch ID for processing. Batch size is defined in the code depending on the task. Used for Gemma-2B due to memory constraints. Adjust as needed.
    * **Type**: `int`
    * **Default**: None
* `-tt, --trace_w_tokens`
    * **Description**: Use this flag to trace with actual tokens from the prompt instead of their grammatical roles. This was used for the graphs in the paper. (No value needed)
    * **Action**: `store_true`

### Examples

1.  Trace 256 IOI prompts using GPT-2 small:
    ```sh
    python tracing.py -m "gpt2-small" -t "ioi" -n 256
    ```

2.  Trace the first batch of 256 IOI prompts using Gemma-2B:
    ```sh
    python tracing.py -m "gemma-2-2b" -t "ioi" -n 256 -b 0
    ```
