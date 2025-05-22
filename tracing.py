import argparse
from collections import defaultdict
from copy import deepcopy
import os
from pprint import pprint
import time
from typing import Union, Tuple
import h5py
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache
from ioi_dataset import IOIDataset
from gt_dataset import YearDataset, get_valid_years
from gp_dataset import GenderedPronoun
import torch
from utils import get_omega_decomposition_all_ahs, trace_firing_optimized, with_no_bias_models, get_omega_decomposition_ah
import numpy as np
import networkx as nx
import math

torch.set_grad_enabled(False)

def get_ah_idx_label(ah_idx: int, n_heads: int):
    if ah_idx == n_heads:
        return "MLP"
    elif ah_idx == n_heads+1:
        return "AH bias"
    elif ah_idx == n_heads+2:
        return "Embedding"
    return ah_idx

def get_upstream_contributors(contrib, frac_contrib_thresh=1.0):
    # Greedly find the set of (layer, ah_idx, token) s.t. their sum is equal to the total sum of the contrib
    sorted_contribs = np.sort(np.ravel(contrib))[::-1]
    thresh = frac_contrib_thresh * np.sum(np.ravel(contrib))
    cutoff = sorted_contribs[np.where(np.cumsum(sorted_contribs) > thresh)[0][0]]
    upstream_contributors = np.where(contrib >= cutoff)
    upstream_contributors = [(layer, ah_idx, token) for layer, ah_idx, token in zip(upstream_contributors[0], upstream_contributors[1], upstream_contributors[2])]
    return upstream_contributors

def get_upstream_contributors_seed(contrib, frac_contrib_thresh=1.0):
    # Greedly find the set of (layer, ah_idx, token) s.t. their sum is equal to the total sum of the contrib
    sorted_contribs = np.sort(np.ravel(contrib))[::-1]
    thresh = frac_contrib_thresh * np.sum(np.ravel(contrib))
    cutoff = sorted_contribs[np.where(np.cumsum(sorted_contribs) > thresh)[0][0]]
    # Reducing the cuttoff if it's greater than 50% of the logit
    if cutoff > contrib.sum() / 2:
        cutoff = contrib.sum() / 2
    upstream_contributors = np.where(contrib >= cutoff)
    upstream_contributors = [(layer, ah_idx, token) for layer, ah_idx, token in zip(upstream_contributors[0], upstream_contributors[1], upstream_contributors[2])]
    return upstream_contributors

def f_W_U(prompt_id, model, dataset, task):
    if task == "ioi":
        IO_token, S_token = dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]], dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]
        return model.W_U[:, IO_token] - model.W_U[:, S_token] # IO-S direction for the IOI dataset
    elif task == "gt":
        YY_idx = dataset.possible_targets_toks.index(dataset.YY_toks[prompt_id])
        # indices less or equal than YY are negative
        # indices greater than YY are positive
        direction_neg = None
        direction_pos = None

        for i in range(len(dataset.possible_targets_toks)):
            target_tok = dataset.possible_targets_toks[i]
            if i <= YY_idx:
                if direction_neg is None:
                    direction_neg = deepcopy(-model.W_U[:, target_tok])
                else:
                    direction_neg -= model.W_U[:, target_tok]
            else:
                if direction_pos is None:
                    direction_pos = deepcopy(model.W_U[:, target_tok])
                else:
                    direction_pos += model.W_U[:, target_tok]

        # Normalizing the directions by the number of directions
        direction_neg /= (YY_idx + 1)
        direction_pos /= (len(dataset.possible_targets_toks) - YY_idx - 1)

        return direction_pos + direction_neg # (direction negative is already with the - sign)
    elif task == "gp":
        correct_direction_idx = dataset.answers[prompt_id]
        wrong_direction_idx = dataset.wrongs[prompt_id]
        return model.W_U[:, correct_direction_idx] - model.W_U[:, wrong_direction_idx]
    
def get_seeds(model: HookedTransformer,
              model_name: str,
              prompt_id: int,
              cache: ActivationCache,
              dataset: Union[IOIDataset, GenderedPronoun, YearDataset],
              task: str) -> Tuple[dict, list]:
    if task == "ioi":
        end_token_pos = dataset.word_idx["end"][prompt_id].item()
        n_tokens = end_token_pos + 1
    elif task == "gt":
        end_token_pos = dataset.word_idx["end"][prompt_id].item()
        n_tokens = end_token_pos + 1
    elif task == "gp":
        end_token_pos = dataset.word_idx["end"][prompt_id].item()
        n_tokens = end_token_pos + 1

    # Breaking down the OV for all AHs in the model
    #upstream_output_breakdown = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, n_tokens, n_tokens, model.cfg.d_model))
    upstream_output_breakdown = torch.zeros((model.cfg.n_layers, model.cfg.n_heads+3, n_tokens, n_tokens, model.cfg.d_model))
    # Embedding
    upstream_output_breakdown[0, -1, end_token_pos, end_token_pos] = deepcopy(cache["blocks.0.hook_resid_pre"][prompt_id, end_token_pos])
    for upstream_layer in range(model.cfg.n_layers):
        for upstream_ah_idx in range(model.cfg.n_heads+3):
            if upstream_ah_idx < model.cfg.n_heads: #AHs
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :n_tokens, :n_tokens] # n_tokens x n_tokens
                if model_name == "gemma-2-2b":
                    # Gemma-2-2b uses group query attention. Then, ah_idx 0 and 1 have the same hook_v (idx=0).
                    V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :n_tokens, upstream_ah_idx//2, :] # n_tokens x d_head
                else:
                    V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :n_tokens, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown[upstream_layer, upstream_ah_idx] = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                if model_name == "gemma-2-2b":
                    # Post attention layer norm term (not folded)
                    upstream_output_breakdown[upstream_layer, upstream_ah_idx] *= model.blocks[upstream_layer].ln1_post.w.detach()
                    ln_post_term = cache[f"blocks.{upstream_layer}.ln1_post.hook_scale"][prompt_id, :n_tokens]
                    upstream_output_breakdown[upstream_layer, upstream_ah_idx] /= ln_post_term.view(ln_post_term.shape[0], 1, 1)
                
            # For all these cases, both dest and src tokens are the same (there is no concept of src tokens)
            elif upstream_ah_idx == model.cfg.n_heads: #MLP
                upstream_output_breakdown[upstream_layer, upstream_ah_idx, end_token_pos, end_token_pos] = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"][prompt_id, end_token_pos])
            elif upstream_ah_idx == model.cfg.n_heads+1: #AH bias
                upstream_output_breakdown[upstream_layer, upstream_ah_idx, end_token_pos, end_token_pos] = deepcopy(model.b_O[upstream_layer])
    
    W_U_direction = f_W_U(prompt_id, model, dataset, task)
    
    # Layer norming the upstream outputs using the LN final and taking the dot product with W_U_direction
    contrib_end_f_W_U_tensor = ((upstream_output_breakdown[:, :, end_token_pos, :, :] / cache["ln_final.hook_scale"][prompt_id, end_token_pos]) @ W_U_direction)

    # Old
    #trace_seeds = get_upstream_contributors(contrib_end_f_W_U_tensor, 1.0)

    # New: more flexible cuttoff, also considering components with contribution
    # to the logit_diff greater or equal than 50% of the logit_diff for the case in which the cuttoff is 
    # larget than 0.5 * logit_diff
    if contrib_end_f_W_U_tensor.sum() > 0:
        trace_seeds = get_upstream_contributors_seed(contrib_end_f_W_U_tensor, 1.0)
        seeds_contrib = {seed: contrib_end_f_W_U_tensor[seed].item() for seed in trace_seeds}
    else:
        # The logit diffence is coming from b_U. Let's not trace these cases
        trace_seeds = []
        seeds_contrib = {}

    return trace_seeds, seeds_contrib

def trace(model: HookedTransformer,
            cache: ActivationCache,
            all_tracing_results: dict,
            U: dict,
            S: dict,
            VT: dict,
            model_name: str,
            device: str,
            idx_to_token: dict,
            G: nx.MultiDiGraph,
            is_traced: dict, 
            prompt_id: int,
            layer: int, 
            ah_idx: int, 
            dest_token: int, 
            src_token: int,
            signal_dict: dict,
            lazy_eval: bool) -> None:
    
    is_traced[(prompt_id, layer, ah_idx, dest_token, src_token)] = 1

    if layer == 0 or dest_token == 0:
        return
    
    if src_token > dest_token:
        return
    
    # Due to the softmax property, for the contributions to be positive, A_ij >= 1/n
    attn_thresh_final = 1/(dest_token+1)
    if cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < attn_thresh_final:
        return

    # Updating the dicts if we perform lazy eval
    if lazy_eval:
        get_omega_decomposition_ah(
            model,
            model_name,
            layer,
            ah_idx,
            dest_token,
            U,
            S,
            VT,
            device
        )

    tracing_results_dict, contrib_dest_start, contrib_src_start = trace_firing_optimized(
        model,
        cache,
        prompt_id,
        layer,
        ah_idx,
        dest_token,
        src_token,
        True,
        1.0,
        U,
        S,
        VT,
        model_name,
        device
    )

    # Updating the dict tracing
    for key in tracing_results_dict:
        all_tracing_results[key][(prompt_id, layer, ah_idx, dest_token, src_token)] = tracing_results_dict[key][(prompt_id, layer, ah_idx, dest_token, src_token)]
    
    contrib_src = tracing_results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)]
    contrib_dest = tracing_results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)]
    
    # Adding the initial state of the contribution as a element of contrib_src
    new_contrib_src = torch.zeros((contrib_src.shape[0], contrib_src.shape[1]+1, contrib_src.shape[2]))
    new_contrib_src[:, :-1, :] = contrib_src
    assert torch.allclose(new_contrib_src[:, :-1, :], contrib_src)
    contrib_src = new_contrib_src
    contrib_src[0, -1, src_token] = contrib_src_start
    assert torch.allclose(contrib_src[:, -1, :].sum(), contrib_src_start)

    # Adding the initial state of the contribution as a element of contrib_dest
    new_contrib_dest = torch.zeros((contrib_dest.shape[0], contrib_dest.shape[1]+1, contrib_dest.shape[2]))
    new_contrib_dest[:, :-1, :] = contrib_dest
    assert torch.allclose(new_contrib_dest[:, :-1, :], contrib_dest)
    contrib_dest = new_contrib_dest
    contrib_dest[0, -1, dest_token] = contrib_dest_start
    assert torch.allclose(contrib_dest[:, -1, :].sum(), contrib_dest_start)

    # Using 70% rule
    # If the contrib_dest is zero, we don't trace dest
    if contrib_dest.sum() > 0:
        upstream_seeds_dest = get_upstream_contributors(contrib_dest, 0.7)
    else:
        upstream_seeds_dest = []
    
    # If the contrib_src is zero, we don't trace src
    if contrib_src.sum() > 0:
        upstream_seeds_src = get_upstream_contributors(contrib_src, 0.7)
    else:
        upstream_seeds_src = []

    ah_idx_label = get_ah_idx_label(ah_idx, model.cfg.n_heads)

    # Node downstream
    if (prompt_id, dest_token) in idx_to_token and (prompt_id, src_token) in idx_to_token:
        node_downstream = (layer, ah_idx_label, idx_to_token[prompt_id, dest_token], idx_to_token[prompt_id, src_token])
    else:
        print(f"Prompt {prompt_id} not tracing tokens that are not in the mapping: {dest_token}, {src_token}...")
        return # We do not trace tokens that are not in the mapping

    # Tracing dest
    for (upstream_layer, upstream_ah_idx, upstream_src_token) in upstream_seeds_dest:        
        upstream_ah_idx_label = get_ah_idx_label(upstream_ah_idx, model.cfg.n_heads)

        if upstream_src_token > dest_token: # src > dest upstream
            continue

        # If these tokens are in the mapping
        if (prompt_id, dest_token) in idx_to_token and (prompt_id, upstream_src_token) in idx_to_token:
            # Node upstream
            node_upstream = (upstream_layer, upstream_ah_idx_label, idx_to_token[prompt_id, dest_token], idx_to_token[prompt_id, upstream_src_token])

            G.add_edge(
                node_upstream,
                node_downstream,
                weight=contrib_dest[(upstream_layer, upstream_ah_idx, upstream_src_token)].item(),
                type="d",
            )

            # upstream_ah_idx >= model.cfg.n_heads indicates MLP, AH bias, and Embedding, and we do not trace them upstream
            if upstream_ah_idx < model.cfg.n_heads:
                if not (prompt_id, upstream_layer, upstream_ah_idx, dest_token, upstream_src_token) in is_traced.keys():
                    # Tracing upstream
                    trace(model,
                    cache,
                    all_tracing_results,
                    U,
                    S,
                    VT,
                    args.model_name,
                    args.device,
                    idx_to_token,
                    G,
                    is_traced,
                    prompt_id, 
                    upstream_layer, 
                    upstream_ah_idx, 
                    dest_token, 
                    upstream_src_token,
                    signal_dict,
                    lazy_eval)     

    # Tracing src
    for (upstream_layer, upstream_ah_idx, upstream_src_token) in upstream_seeds_src:        
        upstream_ah_idx_label = get_ah_idx_label(upstream_ah_idx, model.cfg.n_heads)

        if upstream_src_token > src_token: # src > dest upstream
            continue

        if (prompt_id, src_token) in idx_to_token and (prompt_id, upstream_src_token) in idx_to_token:
            # Node upstream
            node_upstream = (upstream_layer, upstream_ah_idx_label, idx_to_token[prompt_id, src_token], idx_to_token[prompt_id, upstream_src_token])

            G.add_edge(
                node_upstream,
                node_downstream,
                weight=contrib_src[(upstream_layer, upstream_ah_idx, upstream_src_token)].item(),
                type="s",
            )

            # upstream_ah_idx >= model.cfg.n_heads indicates MLP, AH bias, and Embedding, and we do not trace them upstream
            if upstream_ah_idx < model.cfg.n_heads:
                if not (prompt_id, upstream_layer, upstream_ah_idx, src_token, upstream_src_token) in is_traced.keys():
                    trace(model,
                    cache,
                    all_tracing_results,
                    U,
                    S,
                    VT,
                    args.model_name,
                    args.device,
                    idx_to_token,
                    G,
                    is_traced,
                    prompt_id, 
                    upstream_layer, 
                    upstream_ah_idx, 
                    src_token, 
                    upstream_src_token,
                    signal_dict,
                    lazy_eval)    

    return

def main(args, ATTRIBUTES, batch_size):
    model_family = args.model_name.split("/")[-1].split("-")[0] # For models of the form COMPANY/MODELNAME-PARAMS

    model = HookedTransformer.from_pretrained(args.model_name, device="cpu")

    if args.task == "ioi":
        dataset = IOIDataset(
            model_family=model_family,
            prompt_type="mixed",
            N = args.num_prompts,
            tokenizer=model.tokenizer,
            prepend_bos=True if model_family == "gemma" else False,
            seed=args.seed,
            device=args.device
        )
    elif args.task == "gt":
        years_to_sample_from = get_valid_years(model.tokenizer, 1000, 1900)
        dataset = YearDataset(years_to_sample_from, 
                              args.num_prompts, 
                              model.tokenizer, 
                              balanced=True, 
                              device=args.device, 
                              eos=True, # Following the original paper
                              random_seed=args.seed)
    elif args.task == "gp":
        dataset = GenderedPronoun(model, model_family=model_family, device=args.device, prepend_bos=True if model_family == "gemma" else False)
        if args.num_prompts != 100:
            raise Exception("Currently, GenderedPronoun generates only 100 examples. Please set num_prompts to 100.")
    
    if not args.lazy_eval:
        U, S, VT = get_omega_decomposition_all_ahs(model, args.model_name, args.device, True) # New Omega defn.
    elif not args.model_name in with_no_bias_models: # Models with bias will have two Omegas
        U, S, VT = {"d": {}, "s": {}}, {"d": {}, "s": {}}, {"d": {}, "s": {}}
    else:
        U, S, VT = {}, {}, {}
    
    prompt_ids_skip = []
    if args.task == "ioi":
        logits, cache = model.run_with_cache(dataset.toks)
        
        # Removing instances that the model do not predict the IO name
        for pid in range(args.num_prompts):
            end_token_id = dataset.word_idx["end"][pid].item()
            io_token = dataset.toks[pid, dataset.word_idx["IO"][pid].item()]
            model_pred = logits[pid, end_token_id, :].argmax()
            if model_pred != io_token:
                prompt_ids_skip.append(pid)
        
    elif args.task == "gt":
        logits, cache = model.run_with_cache(dataset.good_toks)
        predicted_toks = logits[:, -1, :].argmax(axis=1)

        # Removing instances that the model do not predict a number greater than YY     
        for pid in range(args.num_prompts):
            prediction_idx = dataset.possible_targets_toks.index(predicted_toks[pid])
            YY_idx = dataset.possible_targets_toks.index(dataset.YY_toks[pid])
            if prediction_idx <= YY_idx:
                prompt_ids_skip.append(pid)

    elif args.task == "gp":
        logits, cache = model.run_with_cache(dataset.tokens)

        predicted_toks = logits[:, -1, :].argmax(axis=1)
        # Removing instances that the model do not predict a number greater than YY     
        for pid in range(args.num_prompts):
            if predicted_toks[pid] != dataset.answers[pid]:
                prompt_ids_skip.append(pid)

    # Getting the map from idx to token for each prompt
    idx_to_token = {}
    if args.task == "ioi":
        if args.trace_w_tokens:
            # This will give a mapping to token.
            for prompt_id in range(args.num_prompts):
                count_token_dict = defaultdict(int)
                for i in range(dataset.word_idx["end"][prompt_id]+1):
                    token_decoded = model.tokenizer.decode(dataset.toks[prompt_id, i])
                    count_token = count_token_dict[token_decoded]
                    if count_token > 0:
                        idx_to_token[(prompt_id, i)] = f"{model.tokenizer.decode(dataset.toks[prompt_id, i])} ({count_token})"
                    else:
                        idx_to_token[(prompt_id, i)] = f"{model.tokenizer.decode(dataset.toks[prompt_id, i])}"
                    count_token_dict[token_decoded] += 1
        else:
            # This will give a mapping to gram role.
            # Tokens that do not have a grammatical role will not be traced.
            # This serves for circuit aggregation purposes. To trace everything, we can just use the previous mapping.
            # Alternatively, we can map each token it to itself.
            for gram_role in dataset.word_idx.keys():
                for prompt_id, tok_id in enumerate(dataset.word_idx[gram_role]):
                    idx_to_token[(prompt_id, tok_id.item())] = gram_role
    elif args.task == "gt":
        # For this dataset, only the tokens that are in the gram role are different. Everything else is the same
        # First, gets the actual tokens
        for prompt_id in range(args.num_prompts):
            count_token_dict = defaultdict(int)
            for i in range(dataset.word_idx["end"][prompt_id]+1):
                token_decoded = model.tokenizer.decode(dataset.good_toks[prompt_id, i])
                count_token = count_token_dict[token_decoded]
                if count_token > 0:
                    idx_to_token[(prompt_id, i)] = f"{model.tokenizer.decode(dataset.good_toks[prompt_id, i])} ({count_token})"
                else:
                    idx_to_token[(prompt_id, i)] = f"{model.tokenizer.decode(dataset.good_toks[prompt_id, i])}"
                count_token_dict[token_decoded] += 1

        if not args.trace_w_tokens:
            # Replacing the ones that have gram role
            for gram_role in dataset.word_idx.keys():
                for prompt_id, tok_id in enumerate(dataset.word_idx[gram_role]):
                    idx_to_token[(prompt_id, tok_id.item())] = gram_role
    elif args.task == "gp":
        if args.trace_w_tokens:
            # This will give a mapping to token.
            for prompt_id in range(args.num_prompts):
                count_token_dict = defaultdict(int)
                for i in range(dataset.word_idx["end"][prompt_id]+1):
                    token_decoded = model.tokenizer.decode(dataset.toks[prompt_id, i])
                    count_token = count_token_dict[token_decoded]
                    if count_token > 0:
                        idx_to_token[(prompt_id, i)] = f"{model.tokenizer.decode(dataset.toks[prompt_id, i])} ({count_token})"
                    else:
                        idx_to_token[(prompt_id, i)] = f"{model.tokenizer.decode(dataset.toks[prompt_id, i])}"
                    count_token_dict[token_decoded] += 1
        else:
            # For this dataset, every token has a gram role
            for gram_role in dataset.word_idx.keys():
                for prompt_id, tok_id in enumerate(dataset.word_idx[gram_role]):
                    idx_to_token[(prompt_id, tok_id.item())] = gram_role       
    
    all_tracing_results = {
        "contrib_sv_src": {},
        "contrib_sv_dest": {},
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {}
    }

    start_time = time.time()
    if not args.batch is None:
        loop_range = range(batch_size * args.batch, batch_size * (args.batch+1))
    else:
        loop_range = range(args.num_prompts)
    
    for prompt_id in tqdm(loop_range, desc="Tracing prompts", total=len(loop_range)):
        if prompt_id in prompt_ids_skip:
            print(f"Skipping prompt {prompt_id} for {args.task}...\n")
            continue

        if args.task == "ioi":
            end_token_pos = dataset.word_idx["end"][prompt_id].item()
        elif args.task == "gt":
            end_token_pos = dataset.word_idx["end"][prompt_id].item()
        elif args.task == "gp":
            end_token_pos = dataset.word_idx["end"][prompt_id].item()
        
        trace_seeds, seeds_contrib = get_seeds(model, args.model_name, prompt_id, cache, dataset, args.task)
        
        if len(trace_seeds) == len(seeds_contrib) == 0:
            print(f"Skipping prompt {prompt_id} for {args.task}...\n")
            continue
                
        not_ah_seeding = True
        for layer, ah_idx, _ in trace_seeds:
            if ah_idx < model.cfg.n_heads:
                not_ah_seeding = False
                break

        if not_ah_seeding:
            print(f"Skipping prompt {prompt_id} for {args.task} - No AH in the seeds...\n")
            continue
        
        G = nx.MultiDiGraph()
        is_traced = dict()
        signal_dict = dict()
        for layer, ah_idx, src_token in trace_seeds:
            if ah_idx < model.cfg.n_heads:
                print(f"Prompt {prompt_id}. Tracing from the seed: AH({layer}, {ah_idx}) in ({end_token_pos}, {src_token})...")
            elif ah_idx == model.cfg.n_heads:
                print(f"Prompt {prompt_id}. Tracing from the seed: MLP {layer} in ({end_token_pos}, {src_token})...")
                assert src_token == end_token_pos # There is no src token for components besides AHs
            elif ah_idx == model.cfg.n_heads+1:
                print(f"Prompt {prompt_id}. Tracing from the seed: AH bias {layer} in ({end_token_pos}, {src_token})...")
                assert src_token == end_token_pos # There is no src token for components besides AHs
            elif ah_idx == model.cfg.n_heads+2:
                print(f"Prompt {prompt_id}. Tracing from the seed: Embedding {layer} in ({end_token_pos}, {src_token})...")
                assert src_token == end_token_pos # There is no src token for components besides AHs

            ah_idx_label = get_ah_idx_label(ah_idx, model.cfg.n_heads)

            if args.task == "ioi":
                #root_node = f"IO-S direction\n{idx_to_token[prompt_id, end_token_pos]}"
                root_node = ("IO-S direction", idx_to_token[prompt_id, end_token_pos])
                # If these tokens are in the mapping
                if (prompt_id, end_token_pos) in idx_to_token and (prompt_id, src_token) in idx_to_token:
                    
                    G.add_edge(
                        (layer, ah_idx_label, idx_to_token[prompt_id, end_token_pos], idx_to_token[prompt_id, src_token]),
                        root_node,
                        weight=seeds_contrib[(layer, ah_idx, src_token)],
                        type="d"
                    )
            elif args.task == "gt":
                root_node = ("True YY - False YY", idx_to_token[prompt_id, end_token_pos])
                # If these tokens are in the mapping
                if (prompt_id, end_token_pos) in idx_to_token and (prompt_id, src_token) in idx_to_token:
                    G.add_edge(
                        (layer, ah_idx_label, idx_to_token[prompt_id, end_token_pos], idx_to_token[prompt_id, src_token]),
                        root_node,
                        weight=seeds_contrib[(layer, ah_idx, src_token)],
                        type="d"
                    )
            elif args.task == "gp":
                root_node = ("Correct - Incorrect pronoum", idx_to_token[prompt_id, end_token_pos])
                # If these tokens are in the mapping
                if (prompt_id, end_token_pos) in idx_to_token and (prompt_id, src_token) in idx_to_token:
                    G.add_edge(
                        (layer, ah_idx_label, idx_to_token[prompt_id, end_token_pos], idx_to_token[prompt_id, src_token]),
                        root_node,
                        weight=seeds_contrib[(layer, ah_idx, src_token)],
                        type="d"
                    )

            if ah_idx < model.cfg.n_heads: # Otherwise the seeds are MLP, embedding or bias
                trace(model,
                    cache,
                    all_tracing_results,
                    U,
                    S,
                    VT,
                    args.model_name,
                    args.device,
                    idx_to_token,
                    G,
                    is_traced,
                    prompt_id, 
                    layer, 
                    ah_idx, 
                    end_token_pos, 
                    src_token,
                    signal_dict,
                    args.lazy_eval)    
        
        # Creating the graph folder
        base_folder = "traced_graphs"
        if args.trace_w_tokens:
            base_folder += "_with_tokens"
        graph_folder = f"{base_folder}/{args.model_name.split('/')[-1]}/{args.task}"
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder, exist_ok=True)

        # Saving graph in disk
        nx.write_graphml(G, f'{graph_folder}/{args.model_name.split("/")[-1]}_{args.task}_n{args.num_prompts}_{prompt_id}_{args.seed}.graphml')
    
    end_time = time.time()

    # Creating the tracing folder
    tracing_folder = f"tracing_results"
    if not os.path.exists(tracing_folder):
        os.mkdir(tracing_folder)

    # Saving tracing dict
    with h5py.File(f"{tracing_folder}/{args.model_name.split('/')[-1]}_{args.task}_{args.num_prompts}_{args.seed}_{args.batch}.hdf5", 'w') as f:
        for dataset_name in all_tracing_results:
            for key in all_tracing_results[dataset_name]:
                f.create_dataset(f"{dataset_name}_{key}", data=all_tracing_results[dataset_name][key], compression='gzip', compression_opts=9)

    dict_args = vars(args)

    # Saving time log
    with open("tracing_results/trace_log_time.csv", "a") as f:
        for attr in ATTRIBUTES:
            f.writelines(f"{dict_args[attr]},")
        f.writelines(f"{end_time-start_time}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", choices=["gpt2-small", "EleutherAI/pythia-160m", "EleutherAI/pythia-160m-deduped", "gemma-2-2b"], required=True)
    parser.add_argument("-t", "--task", choices=["ioi", "gt", "gp"], required=True)
    parser.add_argument("-n", "--num_prompts", type=int, required=True)
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-le", "--lazy_eval", action="store_true")
    parser.add_argument("-b", "--batch", type=int)
    parser.add_argument("-tt", "--trace_w_tokens", action="store_true")
    args = parser.parse_args()

    if args.task == "ioi":
        batch_size = 8
    elif args.task == "gp":
        batch_size = 20
    else:
        batch_size = args.num_prompts

    n_batches = args.num_prompts // batch_size

    if args.batch is not None and args.batch >= n_batches:
        raise Exception(f"args.batch:{args.batch} is greater or equal than n_batches: {n_batches}")

    ATTRIBUTES = ["model_name", "task", "num_prompts", "device", "seed", "lazy_eval", "batch", "trace_w_tokens"]

    main(args, ATTRIBUTES, batch_size)

