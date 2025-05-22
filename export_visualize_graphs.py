from collections import defaultdict
import glob
import os
import networkx as nx
from ioi_dataset import IOIDataset
from gt_dataset import get_valid_years, YearDataset
from gp_dataset import GenderedPronoun
from utils import format_graph_cytoscape_by_token_pos
from transformer_lens import HookedTransformer


def get_tokens(model_name, task, prompt_id):
    if model_name == "gemma-2-2b":
        #tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", padding_side='right', truncation_side='right')
        model = HookedTransformer.from_pretrained("gemma-2-2b", device="cpu")
        tokenizer = model.tokenizer
        model_family = "gemma"
    elif model_name == "pythia-160m":
        model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m", device="cpu")
        tokenizer = model.tokenizer
        model_family = "pythia"
    else:
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        tokenizer = model.tokenizer
        model_family = "gpt2"
    
    if task == "ioi":
        dataset = IOIDataset(
            model_family=model_family,
            prompt_type="mixed",
            N = 256,
            tokenizer=tokenizer,
            prepend_bos=True if model_family == "gemma" else False,
            seed=0,
            device="cpu"
        )

    elif task == "gt":
        years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        dataset = YearDataset(years_to_sample_from, 
                              256, 
                              tokenizer, 
                              balanced=True, 
                              device="cpu", 
                              eos=True, # Following the original paper
                              random_seed=0)
    elif task == "gp":
        dataset = GenderedPronoun(model, model_family=model_family, device="cpu", prepend_bos=True if model_family == "gemma" else False)

    tokens = []
    count_token_dict = defaultdict(int)
    for i in range(dataset.word_idx["end"][prompt_id]+1):
        token_decoded = tokenizer.decode(dataset.toks[prompt_id, i])
        count_token = count_token_dict[token_decoded]
        if count_token > 0:
            tokens.append(f"{tokenizer.decode(dataset.toks[prompt_id, i])} ({count_token})")
        else:
            tokens.append(f"{tokenizer.decode(dataset.toks[prompt_id, i])}")
        count_token_dict[token_decoded] += 1

    return tokens


# TODO: make it more generic
root_node_task = {
    "ioi": 'IO-S direction',
    "gp": 'Correct - Incorrect pronoum',
    "gt": 'True YY - False YY',
}

model_cfg = {
    "gpt2-small": (12, 12),
    "pythia-160m": (12, 12),
    "gemma-2-2b": (26, 8)
}

files = []

for model_name in ["gpt2-small", "pythia-160m", "gemma-2-2b"]:
    for task in ["ioi", "gt", "gp"]:
        if task == "gp":
            num_prompts = 100
            prompt_id = 0
        elif task == "gt":
            num_prompts = 256
            prompt_id = 71
        else:
            num_prompts = 256
            prompt_id = 148

        files = glob.glob(f"traced_graphs_with_tokens/{model_name}/{task}/*_n{num_prompts}_{prompt_id}_0.graphml")
        if len(files) > 1:
            print(files)
            raise Exception("More than 1 graph.")
        if len(files) == 0:
            print("No file...\n")
            continue

        file = files[0]

        n_layers, n_heads = model_cfg[model_name]

        G: nx.MultiDiGraph = nx.read_graphml(file)

        new_file = file.replace("traced_graphs", "traced_graphs_vis")
        new_path = "/".join(new_file.split("/")[:-1])

        if not os.path.exists(new_path):
            os.makedirs(new_path, exist_ok=True)

        tokens = get_tokens(model_name, task, prompt_id)
        root_node = [node for node in G.nodes if root_node_task[task] in node]
        if len(root_node) != 1:
            print(root_node)
            raise Exception("Number of root nodes is not 1.")
        
        root_node = root_node[0]

        G = format_graph_cytoscape_by_token_pos(G, root_node, n_layers, n_heads, tokens)
        nx.write_graphml(G, new_file)
        print(f"Saved: {new_file}")