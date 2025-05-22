from copy import deepcopy
import h5py
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
import networkx as nx
import math
from jaxtyping import Float
from torch import Tensor
from typing import Tuple
import einops

# Models that do not use RoPE
non_RoPE_models = [
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large"
]

# Models that use RoPE
RoPE_models = [
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-70m-deduped",
    "gemma-2-2b"
]

# Models with no bias in the AH
with_no_bias_models = [
    "gemma-2-2b"
]

def get_rotary_matrix(idx_rotation: int,
                      rotary_dim: int,
                      d_head: int,
                      angles: Tensor) -> Tensor:
    """
    Compute the rotary matrix for the idx_rotation-th rotation.
    """
    assert rotary_dim <= d_head

    R_m = torch.zeros((d_head, d_head))
    for i in range(d_head):
        if i < rotary_dim:
            R_m[i, i] = np.cos(angles[idx_rotation][i])
        else:
            R_m[i, i] = 1

    idx = 0
    for i, j in zip(range(0, rotary_dim // 2), range(rotary_dim // 2, rotary_dim)):
        R_m[i, j] = -np.sin(angles[idx_rotation][idx])
        idx += 1

    idx = rotary_dim // 2
    for i, j in zip(range(rotary_dim // 2, rotary_dim), range(0, rotary_dim // 2)):
        R_m[i, j] = np.sin(angles[idx_rotation][idx])
        idx += 1

    return R_m

def get_rotation_matrix_diff(model: HookedTransformer, 
                             dest_token: int, 
                             src_token: int) -> Tensor:
    """
    Compute the angles used for RoPE using TransformerLens code.
    Then, compute the rotation matrix R_{d - s} and return it.
    """
    # Compute the rotation angles
    rotary_dim = model.cfg.rotary_dim
    n_ctx = model.cfg.n_ctx
    pos = torch.arange(n_ctx)
    dim = torch.arange(rotary_dim // 2)
    freq = model.cfg.rotary_base ** (dim / (rotary_dim / 2))
    freq = einops.repeat(freq, "d -> (2 d)") # TODO: this may change depending on the model. Make this generic like TL.
    angles = pos[:, None] / freq[None, :]

    R_i = get_rotary_matrix(dest_token, rotary_dim, model.cfg.d_head, angles)
    R_j = get_rotary_matrix(src_token, rotary_dim, model.cfg.d_head, angles)
    R_diff = R_i.T @ R_j

    return R_diff


def get_omega_decomposition_ah(model: HookedTransformer, 
                               model_name: str,
                               layer: int,
                               ah_idx: int,
                               dest_token: int,
                               U: dict,
                               S: dict,
                               VT: dict,
                               device: str = "cpu") -> None:
    """
    Populate U, S, and VT dicts with the Omega matrix decomposition. When models have bias, dicts have keys "d" and "s".
    Used for the Lazy Evaluation version of the tracing.
    """
    rank = model.W_Q.shape[-1]

    if model_name in non_RoPE_models:
        omega =  {}

        # If it's already in the dict, just return
        if (layer, ah_idx, -1) in U["d"] and (layer, ah_idx, -1) in U["s"]:
            return

        # Omega_d
        omega["d"] = torch.column_stack([
            model.W_Q[layer, ah_idx, :, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
            (model.W_Q[layer, ah_idx, :, :].cpu() @ model.b_K[layer, ah_idx, :])
        ])

        # Omega_s
        omega["s"] = torch.row_stack([
            model.W_Q[layer, ah_idx, :, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
            model.b_Q[layer, ah_idx, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T
        ])

        for omega_type in ["d", "s"]:
            Ubig, Sbig, VTbig = torch.linalg.svd(omega[omega_type]) # For GPT-2, we can use the Torch SVD implementation (no ill matrices)
            U[omega_type][(layer, ah_idx, -1)], S[omega_type][(layer, ah_idx, -1)], VT[omega_type][(layer, ah_idx, -1)] = Ubig[:,:rank].to(device), Sbig[:rank].to(device), VTbig[:rank].to(device)

    elif model_name in RoPE_models and not model_name in with_no_bias_models:
        # Compute it for every possible src_token
        for src_token in range(dest_token+1):
            omega =  {}

            diff = dest_token - src_token
            
            #If it's already in the dict, skip
            if (layer, ah_idx, diff) in U["d"] and (layer, ah_idx, diff) in U["s"]:
                continue

            R_diff = get_rotation_matrix_diff(model, dest_token, src_token)

            # Omega_d
            omega["d"] = torch.column_stack([
                model.W_Q[layer, ah_idx, :, :].cpu() @ R_diff @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                (model.W_Q[layer, ah_idx, :, :].cpu() @ R_diff @ model.b_K[layer, ah_idx, :])
            ]).numpy()

            # Omega_s
            omega["s"] = torch.row_stack([
                model.W_Q[layer, ah_idx, :, :].cpu() @ R_diff @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                model.b_Q[layer, ah_idx, :].cpu() @ R_diff @ model.W_K[layer, ah_idx, :, :].cpu().T
            ]).numpy()

            for omega_type in ["d", "s"]:
                Ubig, Sbig, VTbig = np.linalg.svd(omega[omega_type])

                Ubig, Sbig, VTbig = Ubig[:,:rank], Sbig[:rank], VTbig[:rank]

                U[omega_type][(layer, ah_idx, diff)] = torch.from_numpy(Ubig).to(device)
                S[omega_type][(layer, ah_idx, diff)] = torch.from_numpy(Sbig).to(device)
                VT[omega_type][(layer, ah_idx, diff)] = torch.from_numpy(VTbig).to(device)

                # Having this check due to the ill conditioned matrices in Pythia
                try:
                    assert np.isclose(omega[omega_type], U[omega_type][(layer, ah_idx, diff)] @ np.diag(S[omega_type][(layer, ah_idx, diff)]) @ VT[omega_type][(layer, ah_idx, diff)], atol=1e-5).all()
                except:
                    raise Exception(f"Failed to reconstruct the omega matrix for {(layer, ah_idx, diff)}")
    elif model_name in with_no_bias_models:
        # Compute it for every possible src_token
        for src_token in range(dest_token+1):
            omega =  {}

            diff = dest_token - src_token

            #If it's already in the dict, skip
            if (layer, ah_idx, diff) in U:
                continue

            if diff < 22:
                with h5py.File(f'gemma-2-2b-omega/matrices_{layer}.hdf5', 'r') as f:
                    u, s, vt = f[f"U_{layer}_{ah_idx}_{diff}"][:], f[f"S_{layer}_{ah_idx}_{diff}"][:], f[f"VT_{layer}_{ah_idx}_{diff}"][:]
                    # Convert to torch tensors
                    U[(layer, ah_idx, diff)] = torch.from_numpy(u).to(device)
                    S[(layer, ah_idx, diff)] = torch.from_numpy(s).to(device)
                    VT[(layer, ah_idx, diff)] = torch.from_numpy(vt).to(device)
            else:
                R_diff = get_rotation_matrix_diff(model, dest_token, src_token)
                omega = model.W_Q[layer, ah_idx, :, :] @ R_diff @ model.W_K[layer, ah_idx, :, :].T
                Ubig, Sbig, VTbig = np.linalg.svd(omega.cpu())
                Ubig, Sbig, VTbig = Ubig[:,:rank], Sbig[:rank], VTbig[:rank]

                # Check if the SVD is correct
                try:
                    assert np.isclose(omega, Ubig @ np.diag(Sbig) @ VTbig, atol=1e-5).all()
                except:
                    print(f"Failed to reconstruct the omega matrix for {(layer, ah_idx, diff)}")

                U[(layer, ah_idx, diff)] = torch.from_numpy(Ubig).to(device)
                S[(layer, ah_idx, diff)] = torch.from_numpy(Sbig).to(device)
                VT[(layer, ah_idx, diff)] = torch.from_numpy(VTbig).to(device)

def get_omega_decomposition_all_ahs(model: HookedTransformer, 
                                    model_name: str, 
                                    device: str = "cpu",
                                    new_defn_omega: bool = True) -> Tuple[dict, dict, dict]:
    """
    Get the cached Omega decomposition. Return U, S, and VT dicts.
    """
    U, S, VT, omega = {}, {}, {}, {}
    # Special case for Gemma-2-2b
    if model_name == "gemma-2-2b":
        for layer in tqdm(range(model.cfg.n_layers)):
            with h5py.File(f'gemma-2-2b-omega/matrices_{layer}.hdf5', 'r') as f:
                for ah_idx in range(model.cfg.n_heads):
                    for diff in range(22): # TODO: make it more general
                        u, s, vt = f[f"U_{layer}_{ah_idx}_{diff}"][:], f[f"S_{layer}_{ah_idx}_{diff}"][:], f[f"VT_{layer}_{ah_idx}_{diff}"][:]
                        # Convert to torch tensors
                        U[(layer, ah_idx, diff)] = torch.from_numpy(u).to(device)
                        S[(layer, ah_idx, diff)] = torch.from_numpy(s).to(device)
                        VT[(layer, ah_idx, diff)] = torch.from_numpy(vt).to(device)
    # Two Omegas, following Appendix A
    elif model_name in RoPE_models:
        U, S, VT = {"d": {}, "s": {}}, {"d": {}, "s": {}}, {"d": {}, "s": {}}
        with h5py.File(f'matrices_{model_name.split("/")[-1]}.hdf5', 'r') as f:
            for layer in tqdm(range(model.cfg.n_layers)):
                for ah_idx in range(model.cfg.n_heads):
                    for diff in range(21): # TODO: make it more general
                        for omega_type in ["d", "s"]:
                            u, s, vt = f[f"U_{omega_type}_{layer}_{ah_idx}_{diff}"][:], f[f"S_{omega_type}_{layer}_{ah_idx}_{diff}"][:], f[f"VT_{omega_type}_{layer}_{ah_idx}_{diff}"][:]
                            # Convert to torch tensors
                            U[omega_type][(layer, ah_idx, diff)] = torch.from_numpy(u).to(device)
                            S[omega_type][(layer, ah_idx, diff)] = torch.from_numpy(s).to(device)
                            VT[omega_type][(layer, ah_idx, diff)] = torch.from_numpy(vt).to(device)
    # Two Omegas, following Appendix A
    elif model_name in non_RoPE_models:
        U, S, VT = {"d": {}, "s": {}}, {"d": {}, "s": {}}, {"d": {}, "s": {}}

        ALL_AHS = [(i, j) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
        rank = model.W_Q.shape[-1]
        for (layer, ah_idx) in tqdm(ALL_AHS):
            # Current (layer, ah_idx) omega
            omega =  {}

            # Omega_d
            omega["d"] = torch.column_stack([
                model.W_Q[layer, ah_idx, :, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                (model.W_Q[layer, ah_idx, :, :].cpu() @ model.b_K[layer, ah_idx, :])
            ])

            # Omega_s
            omega["s"] = torch.row_stack([
                model.W_Q[layer, ah_idx, :, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                model.b_Q[layer, ah_idx, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T
            ])

            for omega_type in ["d", "s"]:
                Ubig, Sbig, VTbig = torch.linalg.svd(omega[omega_type]) # For GPT-2, we can use the Torch SVD implementation (no ill matrices)
                U[omega_type][(layer, ah_idx, -1)], S[omega_type][(layer, ah_idx, -1)], VT[omega_type][(layer, ah_idx, -1)] = Ubig[:,:rank].to(device), Sbig[:rank].to(device), VTbig[:rank].to(device)
    else:
        raise Exception(f"model_name: {model_name} is not implemented.")
        
    return U, S, VT

def apply_projection(p: Float[Tensor, "d_model+1 1"],
                     x: Float[Tensor, "d_model+1"]) -> Float[Tensor, "d_model+1"]:
    return ((p.T @ x)[0] * p.reshape(-1))

def get_components_used_comparative_no_bias(X: Float[Tensor, "n_tokens d_model"], 
                                        src_token: str, 
                                        dest_token: str,
                                        layer: int, 
                                        ah_idx: int, 
                                        U: dict,
                                        S: dict,
                                        VT:dict,
                                        model_name: str,
                                        device: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Comparative method to compute the components used.
    Assumes RoPE matrices (using diff index) and models with no bias.

    Return two dataframes, one for x_i and one for x_j
    '''

    if not model_name in with_no_bias_models:
        raise Exception(f"Model {model_name} has biases, but this function accept only models with no biases.")
    
    # x_i is the destination token, x_j is the source token
    x_i = X[dest_token, :]
    x_j = X[src_token, :]

    diff = -1 if model_name in non_RoPE_models else dest_token - src_token

    k = U[(layer, ah_idx, diff)].shape[1]

    sim_decomp_i = []
    sim_decomp_j = []
    for idx_k in range(k):
        u_k = U[(layer, ah_idx, diff)][:, idx_k].reshape(-1, 1)
        vT_k = VT[(layer, ah_idx, diff)][idx_k, :].reshape(-1, 1)

        slice_sum_decomp_i = 0
        slice_sum_decomp_j = 0

        if dest_token == 0:
            denom_avg_decomp_i = 0
            denom_avg_decomp_j = 0
        else:
            for src_token_prime in range(dest_token+1):
                if src_token_prime == src_token:
                    continue
                x_j_prime = X[src_token_prime, :]
                diff_prime = -1 if model_name in non_RoPE_models else dest_token - src_token_prime

                slice_sum_decomp_j += (x_i @ U[(layer, ah_idx, diff_prime)]) @ torch.diag(S[(layer, ah_idx, diff_prime)]).to(device) @ (VT[(layer, ah_idx, diff_prime)] @ apply_projection(vT_k, x_j_prime))
                slice_sum_decomp_i += (apply_projection(u_k, x_i) @ U[(layer, ah_idx, diff_prime)]) @ torch.diag(S[(layer, ah_idx, diff_prime)]).to(device) @ (VT[(layer, ah_idx, diff_prime)] @ x_j_prime)            
                    
            denom_avg_decomp_i = slice_sum_decomp_i / dest_token
            denom_avg_decomp_j = slice_sum_decomp_j / dest_token

        # compute the contribution of each slice, comparing to denom_avg
        x_i_u_inner_product = x_i @ u_k
        x_j_v_inner_product = vT_k.T @ x_j
        product =  x_i_u_inner_product * S[(layer, ah_idx, diff)][idx_k] * x_j_v_inner_product

        sim_decomp_i.append((idx_k, S[(layer, ah_idx, diff)][idx_k], x_i_u_inner_product, x_j_v_inner_product, denom_avg_decomp_i, product, product - denom_avg_decomp_i))
        sim_decomp_j.append((idx_k, S[(layer, ah_idx, diff)][idx_k], x_i_u_inner_product, x_j_v_inner_product, denom_avg_decomp_j, product, product - denom_avg_decomp_j))

    # Creating a dataframe for easier manipulation
    df_decomp_i = pd.DataFrame(sim_decomp_i, columns=["idx", "singular_value", "x_i_ip",  "x_j_ip", "denom_avg", "product", "contrib"], dtype=np.float32)
    df_decomp_j = pd.DataFrame(sim_decomp_j, columns=["idx", "singular_value", "x_i_ip",  "x_j_ip", "denom_avg", "product", "contrib"], dtype=np.float32)

    # Cumsum of the products sorted in descending order divided by the first term of the attention
    # This gives the percentage of contribution of the singular values to the attention
    # We are interested in the SMALLEST number of singular values that contribute to 99% of the attention
    # Note that this value is always 1.0 when we use all singular values
    if dest_token != 0:
        assert(np.sum(df_decomp_i['contrib']) > 0)
        assert(np.sum(df_decomp_j['contrib']) > 0)

    # Decomposition on x_i
    sv_frac_contribution = np.cumsum(np.sort(df_decomp_i['contrib'])[::-1]) / np.sum(df_decomp_i['contrib'])
    # Sort the dataframe in descending order
    df_decomp_i = df_decomp_i.sort_values(by="contrib", ascending=False)
    df_decomp_i["idx"] = df_decomp_i["idx"].astype(int)
    df_decomp_i["sv_frac_contribution"] = sv_frac_contribution

    # Decomposition on x_j
    sv_frac_contribution = np.cumsum(np.sort(df_decomp_j['contrib'])[::-1]) / np.sum(df_decomp_j['contrib'])
    # Sort the dataframe in descending order
    df_decomp_j = df_decomp_j.sort_values(by="contrib", ascending=False)
    df_decomp_j["idx"] = df_decomp_j["idx"].astype(int)
    df_decomp_j["sv_frac_contribution"] = sv_frac_contribution

    return df_decomp_i, df_decomp_j

def get_components_used_comparative_new_defn(X: Float[Tensor, "n_tokens d_model"], 
                                        src_token: str, 
                                        dest_token: str,
                                        layer: int, 
                                        ah_idx: int, 
                                        U: dict,
                                        S: dict,
                                        VT: dict,
                                        model_name: str,
                                        device: str,
                                        negative_contributions: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Comparative method to compute the components used.
    Assumes RoPE matrices (using diff index). For GPT-2, diff=-1
    Assumes the new definition for models with bias, ie, Omega_d and Omega_s

    Return two dataframes, one for x_d and one for x_s

    When negative_contributions=True, returns dataframes even when the contribution is negative.
    '''
    # x_d is the destination token, x_s is the source token
    x_d = X[dest_token, :]
    x_d_tilde = torch.cat([x_d, torch.ones(1).to(device)])
    x_s = X[src_token, :]
    x_s_tilde = torch.cat([x_s, torch.ones(1).to(device)])

    diff = -1 if model_name in non_RoPE_models else dest_token - src_token

    k = U["s"][(layer, ah_idx, diff)].shape[1] # Rank is always the same for both src and dest (by construction)

    sim_decomp_s = []
    sim_decomp_d = []
    for idx_k in range(k):
        u_k_d = U["d"][(layer, ah_idx, diff)][:, idx_k].reshape(-1, 1)
        vT_k_s = VT["s"][(layer, ah_idx, diff)][idx_k, :].reshape(-1, 1)

        slice_sum_decomp_s = 0
        slice_sum_decomp_d = 0

        if dest_token == 0:
            denom_avg_decomp_s = 0
            slice_sum_decomp_d = 0
        else:
            for src_token_prime in range(dest_token+1):
                if src_token_prime == src_token:
                    continue
                x_s_prime = X[src_token_prime, :]
                x_s_prime_tilde = torch.cat([x_s_prime, torch.ones(1).to(device)])
                diff_prime = -1 if model_name in non_RoPE_models else dest_token - src_token_prime

                # Source
                slice_sum_decomp_s += (x_d_tilde @ U["s"][(layer, ah_idx, diff_prime)]) @ torch.diag(S["s"][(layer, ah_idx, diff_prime)]) @ (VT["s"][(layer, ah_idx, diff_prime)] @ apply_projection(vT_k_s, x_s_prime))

                # Dest
                slice_sum_decomp_d += (apply_projection(u_k_d, x_d) @ U["d"][(layer, ah_idx, diff_prime)]) @ torch.diag(S["d"][(layer, ah_idx, diff_prime)]) @ (VT["d"][(layer, ah_idx, diff_prime)] @ x_s_prime_tilde)
                    
            denom_avg_decomp_s = slice_sum_decomp_s / dest_token
            denom_avg_decomp_d = slice_sum_decomp_d / dest_token

        # compute the contribution of each slice, comparing to denom_avg
        product_s = (x_d_tilde @ U["s"][(layer, ah_idx, diff)]) @ torch.diag(S["s"][(layer, ah_idx, diff)]) @ (VT["s"][(layer, ah_idx, diff)] @ apply_projection(vT_k_s, x_s))
        product_d = (apply_projection(u_k_d, x_d) @ U["d"][(layer, ah_idx, diff)]) @ torch.diag(S["d"][(layer, ah_idx, diff)]) @ (VT["d"][(layer, ah_idx, diff)] @ x_s_tilde)

        sim_decomp_s.append((idx_k, S["s"][(layer, ah_idx, diff)][idx_k], denom_avg_decomp_s, product_s, product_s - denom_avg_decomp_s))
        sim_decomp_d.append((idx_k, S["d"][(layer, ah_idx, diff)][idx_k], denom_avg_decomp_d, product_d, product_d - denom_avg_decomp_d))

    # Creating a dataframe for easier manipulation
    df_decomp_d = pd.DataFrame(sim_decomp_d, columns=["idx", "singular_value", "denom_avg", "product", "contrib"], dtype=np.float32)
    df_decomp_s = pd.DataFrame(sim_decomp_s, columns=["idx", "singular_value", "denom_avg", "product", "contrib"], dtype=np.float32)

    # Cumsum of the products sorted in descending order divided by the first term of the attention
    # This gives the percentage of contribution of the singular values to the attention
    # We are interested in the SMALLEST number of singular values that contribute to 99% of the attention
    # Note that this value is always 1.0 when we use all singular values
    trace_dest = True
    trace_src = True
    
    if not negative_contributions:
        if dest_token != 0:
            if np.sum(df_decomp_d['contrib']) < 0:
                # This means that, for the dest token, all the contribution is coming from c_1 or bias terms
                # As defined in Appendix A, we do not trace these cases
                trace_dest = False
                df_decomp_d = None
            
            if np.sum(df_decomp_s['contrib']) < 0:
                # This means that, for the dest token, all the contribution is coming from the bias term
                trace_src = False
                df_decomp_s = None

    if trace_dest:
        # Decomposition on x_d
        sv_frac_contribution = np.cumsum(np.sort(df_decomp_d['contrib'])[::-1]) / np.sum(df_decomp_d['contrib'])
        # Sort the dataframe in descending order
        df_decomp_d = df_decomp_d.sort_values(by="contrib", ascending=False)
        df_decomp_d["idx"] = df_decomp_d["idx"].astype(int)
        df_decomp_d["sv_frac_contribution"] = sv_frac_contribution        

    if trace_src:
        # Decomposition on x_s
        sv_frac_contribution = np.cumsum(np.sort(df_decomp_s['contrib'])[::-1]) / np.sum(df_decomp_s['contrib'])
        # Sort the dataframe in descending order
        df_decomp_s = df_decomp_s.sort_values(by="contrib", ascending=False)
        df_decomp_s["idx"] = df_decomp_s["idx"].astype(int)
        df_decomp_s["sv_frac_contribution"] = sv_frac_contribution

    return df_decomp_d, df_decomp_s

def trace_firing_optimized(model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> dict:
    if model_name in RoPE_models:
        if model_name in with_no_bias_models:
            return __trace_firing_optimized_rope_no_bias(model, cache, prompt_id, layer, ah_idx, dest_token, src_token, use_svs, frac_contrib_thresh, U, S, VT, model_name, device)
        else:
            return __trace_firing_optimized_rope_new_defn_omega(model, cache, prompt_id, layer, ah_idx, dest_token, src_token, use_svs, frac_contrib_thresh, U, S, VT, model_name, device)
    else:
        return __trace_firing_optimized_non_rope_new_defn_omega(model, cache, prompt_id, layer, ah_idx, dest_token, src_token, use_svs, frac_contrib_thresh, U, S, VT, model_name, device)
    
def __trace_firing_optimized_non_rope_new_defn_omega(model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> dict:
    
    """
    Tracing version for models that do not use RoPE and have bias in the AH, such as GPT-2.
    """
        
    results_dict = {
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {},
        "contrib_sv_src": {},
        "contrib_sv_dest": {}
    }

    X = deepcopy(cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id, :, :]) #Float[Tensor, 'n_tokens d_model']

    df_decomp_i, df_decomp_j = get_components_used_comparative_new_defn(X, src_token, dest_token, layer, ah_idx, U, S, VT, model_name, device)

    # We trace src only if df_decomp_i is None
    if df_decomp_i is None:
        results_dict, start_src = __trace_firing_optimized_non_rope_new_defn_omega_src_only(
            df_decomp_j,
            model,
            cache,
            prompt_id,
            layer,
            ah_idx,
            dest_token,
            src_token,
            use_svs,
            frac_contrib_thresh,
            U,
            S,
            VT,
            model_name,
            device
        )

        return results_dict, torch.zeros(1), start_src

    # Decomposing on x_i
    if use_svs:
        # We round to avoid minor selection errors
        last_sv_idx = np.where(df_decomp_i['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_i = df_decomp_i.iloc[:last_sv_idx+1].idx.astype(int).values

    # Decomposing on x_j
    if use_svs:
        # We round to avoid minor selection errors
        last_sv_idx = np.where(df_decomp_j['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_j = df_decomp_j.iloc[:last_sv_idx+1].idx.astype(int).values

    results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_i
    results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_j

    diff = -1 # Case for non-RoPE models

    contrib_src = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP
    contrib_dest = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))

    # Building all possible x_i, x_j's at once. They will be used across the tracing
    X_tilde = torch.cat([X[:dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    x_i_tilde = X_tilde[dest_token] # x_i downstream
    x_j_tilde = X_tilde[src_token] # x_j downstream
    
    # Optimization used for non-RoPE models
    x_j_prime_tilde_avg = (x_j_tilde - (X_tilde.sum(axis=0) - x_j_tilde) / dest_token)

    # Computing downstream matrices
    P_u = U["d"][(layer, ah_idx, diff)][:, svs_decomp_i] @ U["d"][(layer, ah_idx, diff)][:, svs_decomp_i].T
    P_v = VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :].T @ VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :]

    Omega_s = U["s"][(layer, ah_idx, diff)] @ torch.diag(S["s"][(layer, ah_idx, diff)]) @ VT["s"][(layer, ah_idx, diff)]
    Omega_d = U["d"][(layer, ah_idx, diff)] @ torch.diag(S["d"][(layer, ah_idx, diff)]) @ VT["d"][(layer, ah_idx, diff)]

    # Caching some downstream computations
    x_i_tilde_times_Omega_s = x_i_tilde @ Omega_s
    Omega_d_times_x_j_prime_tilde_avg = Omega_d @ x_j_prime_tilde_avg

    for upstream_layer in range(layer):
        for upstream_ah_idx in range(model.cfg.n_heads+2): #+2 is the MLP and AH bias cases
            if upstream_ah_idx == model.cfg.n_heads: # MLP case:
                upstream_output = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"])
                X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
                x_out_j = X_out[src_token]
                x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)

            elif upstream_ah_idx == model.cfg.n_heads+1: # AH bias case:
                upstream_output = deepcopy(model.b_O[upstream_layer]) # d_model
                # The AH bias adds the same constant for all tokens
                X_out = upstream_output.repeat(dest_token+1, 1) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
                x_out_j = X_out[src_token]
                x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)
            else:
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :, :] # n_tokens x n_tokens
                V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                # Limit the breakdown by dest_token+1
                upstream_output_breakdown = upstream_output_breakdown[:dest_token+1, :dest_token+1, :]

                X_out = deepcopy(upstream_output_breakdown)

                # Broadcasting the operation
                X_out /= cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].unsqueeze(1)

                #X_out = torch.cat([X_out, torch.zeros(X_out.shape[0], X_out.shape[1], 1).to(device)], dim=2)
                x_out_i = X_out[dest_token]
                x_out_j = X_out[src_token]
                
                x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)
            # MLP and AH bias cases
            if upstream_ah_idx >= model.cfg.n_heads:
                # Dest case
                contrib_dest[upstream_layer, upstream_ah_idx, dest_token] = (P_u @ x_out_i) @ Omega_d_times_x_j_prime_tilde_avg
                # Src case
                contrib_src[upstream_layer, upstream_ah_idx, src_token] = x_i_tilde_times_Omega_s @ (P_v @ x_out_j_prime_avg)
            else: # AH case
                contrib_dest[upstream_layer, upstream_ah_idx, :] = ((x_out_i @ P_u) @ Omega_d_times_x_j_prime_tilde_avg)
                contrib_src[upstream_layer, upstream_ah_idx, :] =  x_i_tilde_times_Omega_s @ (x_out_j_prime_avg @ P_v).T
        
    results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_dest
    results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_src

    # Starting state of the contribution
    upstream_output = deepcopy(cache["blocks.0.hook_resid_pre"])
    #X_out = torch.cat([upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
    x_out_i = X_out[dest_token]
    x_out_j = X_out[src_token]
    x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)

    # Dest case
    start_dest = (P_u @ x_out_i) @ Omega_d_times_x_j_prime_tilde_avg
    # Src case
    start_src = x_i_tilde_times_Omega_s @ (P_v @ x_out_j_prime_avg)

    return results_dict, start_dest, start_src

def __trace_firing_optimized_non_rope_new_defn_omega_src_only(df_decomp_j: pd.DataFrame,
                    model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> dict:
    
    results_dict = {
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {},
        "contrib_sv_src": {},
        "contrib_sv_dest": {}
    }

    X = deepcopy(cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id, :, :]) #Float[Tensor, 'n_tokens d_model']

    # Decomposing on x_j
    if use_svs:
        # We round to avoid minor selection errors
        last_sv_idx = np.where(df_decomp_j['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_j = df_decomp_j.iloc[:last_sv_idx+1].idx.astype(int).values

    results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = []
    results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_j

    diff = -1 # Case for non-RoPE models

    contrib_src = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP
    contrib_dest = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP

    # Building all possible x_i, x_j's at once. They will be used across the tracing
    X_tilde = torch.cat([X[:dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    x_i_tilde = X_tilde[dest_token] # x_i downstream
    x_j_tilde = X_tilde[src_token] # x_j downstream
    
    # Optimization used for non-RoPE models
    x_j_prime_tilde_avg = (x_j_tilde - (X_tilde.sum(axis=0) - x_j_tilde) / dest_token)

    # Computing downstream matrices
    P_v = VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :].T @ VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :]

    Omega_s = U["s"][(layer, ah_idx, diff)] @ torch.diag(S["s"][(layer, ah_idx, diff)]) @ VT["s"][(layer, ah_idx, diff)]

    # Caching some downstream computations
    x_i_tilde_times_Omega_s = x_i_tilde @ Omega_s

    for upstream_layer in range(layer):
        for upstream_ah_idx in range(model.cfg.n_heads+2): #+2 is the MLP and AH bias cases
            if upstream_ah_idx == model.cfg.n_heads: # MLP case:
                upstream_output = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"])
                #X_out = torch.cat([upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :], torch.zeros(dest_token+1).reshape(-1, 1).to(device)], dim=1)
                X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
                x_out_j = X_out[src_token]
                x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)

            elif upstream_ah_idx == model.cfg.n_heads+1: # AH bias case:
                upstream_output = deepcopy(model.b_O[upstream_layer]) # d_model
                # The AH bias adds the same constant for all tokens
                #X_out = torch.cat([upstream_output.repeat(dest_token+1, 1) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :], torch.zeros(dest_token+1).reshape(-1, 1).to(device)], dim=1)
                X_out = upstream_output.repeat(dest_token+1, 1) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
                x_out_j = X_out[src_token]
                x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)
            else:
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :, :] # n_tokens x n_tokens
                V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                # Limit the breakdown by dest_token+1
                upstream_output_breakdown = upstream_output_breakdown[:dest_token+1, :dest_token+1, :]

                X_out = deepcopy(upstream_output_breakdown)

                # Broadcasting the operation
                X_out /= cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].unsqueeze(1)

                #X_out = torch.cat([X_out, torch.zeros(X_out.shape[0], X_out.shape[1], 1).to(device)], dim=2)
                x_out_i = X_out[dest_token]
                x_out_j = X_out[src_token]
                
                x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)
            # MLP and AH bias cases
            if upstream_ah_idx >= model.cfg.n_heads:
                # Src case
                contrib_src[upstream_layer, upstream_ah_idx, src_token] = x_i_tilde_times_Omega_s @ (P_v @ x_out_j_prime_avg)
            else: # AH case
                contrib_src[upstream_layer, upstream_ah_idx, :] =  x_i_tilde_times_Omega_s @ (x_out_j_prime_avg @ P_v).T
    
    results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_src
    results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_dest

    # Starting state of the contribution
    upstream_output = deepcopy(cache["blocks.0.hook_resid_pre"])
    #X_out = torch.cat([upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
    x_out_i = X_out[dest_token]
    x_out_j = X_out[src_token]
    x_out_j_prime_avg = (x_out_j - (X_out.sum(axis=0) - x_out_j) / dest_token)

    # Src case
    start_src = x_i_tilde_times_Omega_s @ (P_v @ x_out_j_prime_avg)

    return results_dict, start_src

def __trace_firing_optimized_rope_new_defn_omega_src_only(df_decomp_j: pd.DataFrame,
                    model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> dict:
    
    results_dict = {
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {},
        "contrib_sv_src": {},
        "contrib_sv_dest": {}
    }

    X = deepcopy(cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id, :, :]) #Float[Tensor, 'n_tokens d_model'] 

    # Decomposing on x_j
    if use_svs:
        last_sv_idx = np.where(df_decomp_j['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_j = df_decomp_j.iloc[:last_sv_idx+1].idx.astype(int).values
    
    results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = []
    results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_j
    
    diff = dest_token - src_token

    contrib_src = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP
    contrib_dest = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))

    # Building all possible x_i, x_j's at once. They will be used across the tracing
    X_tilde = torch.cat([X[:dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    x_i_tilde = X_tilde[dest_token] # x_i downstream
    
    # Computing downstream matrices
    # P_v uses only the svs used when we decompose on x_j
    P_v = VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :].T @ VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :]
    
    Omega_all_s = torch.zeros(dest_token+1, model.cfg.d_model+1, model.cfg.d_model)
    for j_prime in range(dest_token+1):
        diff_prime = dest_token - j_prime
        Omega_all_s[j_prime, :, :] = U["s"][(layer, ah_idx, diff_prime)] @ torch.diag(S["s"][(layer, ah_idx, diff_prime)]) @ VT["s"][(layer, ah_idx, diff_prime)]

    # Caching some computations
    x_i_tilde_times_Omega_s = torch.bmm(x_i_tilde.repeat(dest_token+1, 1).unsqueeze(1), Omega_all_s).squeeze(1)

    for upstream_layer in range(layer):
        for upstream_ah_idx in range(model.cfg.n_heads+2): #+2 is the MLP and AH bias cases
            # Input
            if upstream_ah_idx == model.cfg.n_heads: # MLP case:
                upstream_output = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"])
                X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
            elif upstream_ah_idx == model.cfg.n_heads+1: # AH bias case:
                upstream_output = deepcopy(model.b_O[upstream_layer]) # d_model
                # The AH bias adds the same constant for all tokens
                X_out = upstream_output.repeat(dest_token+1, 1) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
            else:
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :, :] # n_tokens x n_tokens
                V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                # Limit the breakdown by dest_token+1
                upstream_output_breakdown = upstream_output_breakdown[:dest_token+1, :dest_token+1, :]

                X_out = deepcopy(upstream_output_breakdown)

                # Broadcasting the operation
                X_out /= cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].unsqueeze(1)
            # MLP and AH bias cases
            if upstream_ah_idx >= model.cfg.n_heads:
                # Src contribution
                all_terms = (x_i_tilde_times_Omega_s * (X_out @ P_v)).sum(dim=1) # shape [dest_token+1]
                first_term = all_terms[src_token] # First term: when the x_j is the src_token
                second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
                contrib_src[upstream_layer, upstream_ah_idx, src_token] = first_term - second_term
            else: # AH case
                # Computing the projections
                proj = torch.matmul(X_out, P_v)
                # Weighting them by the x_i_tilde @ Omega
                weighted_proj = x_i_tilde_times_Omega_s.unsqueeze(1) * proj  
                # Sum over the last dim to get all terms, shape [dest_tokens+1, dest_tokens + 1]
                all_terms = weighted_proj.sum(dim=-1)
                first_term = all_terms[src_token, :]
                second_term = (all_terms.sum(dim=0) - first_term) / dest_token 
                contrib_src[upstream_layer, upstream_ah_idx, :] = first_term - second_term
        
    results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_dest
    results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_src

    # Starting state of the contribution
    upstream_output = deepcopy(cache["blocks.0.hook_resid_pre"])
    X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
    
    # Src contribution
    all_terms = (x_i_tilde_times_Omega_s * (X_out @ P_v)).sum(dim=1) # shape [dest_token+1]
    first_term = all_terms[src_token] # First term: when the x_j is the src_token
    second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
    start_src = first_term - second_term

    return results_dict, start_src

def __trace_firing_optimized_rope_new_defn_omega_dest_only(df_decomp_i: pd.DataFrame,
                    model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> dict:
    
    results_dict = {
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {},
        "contrib_sv_src": {},
        "contrib_sv_dest": {}
    }

    X = deepcopy(cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id, :, :]) #Float[Tensor, 'n_tokens d_model'] 

    # Decomposing on x_i
    if use_svs:
        last_sv_idx = np.where(df_decomp_i['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_i = df_decomp_i.iloc[:last_sv_idx+1].idx.astype(int).values
    
    results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_i
    results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = []
    
    diff = dest_token - src_token

    contrib_src = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP
    contrib_dest = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))

    # Building all possible x_i, x_j's at once. They will be used across the tracing
    X_tilde = torch.cat([X[:dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    x_i_tilde = X_tilde[dest_token] # x_i downstream
    
    # Computing downstream matrices
    # P_u uses only the svs used when we decompose on x_i
    P_u = U["d"][(layer, ah_idx, diff)][:, svs_decomp_i] @ U["d"][(layer, ah_idx, diff)][:, svs_decomp_i].T
    
    Omega_all_d = torch.zeros(dest_token+1, model.cfg.d_model, model.cfg.d_model+1)
    for j_prime in range(dest_token+1):
        diff_prime = dest_token - j_prime
        Omega_all_d[j_prime, :, :] = U["d"][(layer, ah_idx, diff_prime)] @ torch.diag(S["d"][(layer, ah_idx, diff_prime)]) @ VT["d"][(layer, ah_idx, diff_prime)]

    # Caching some computations
    Omega_d_times_X_tilde = torch.bmm(Omega_all_d, X_tilde.unsqueeze(2)).squeeze(2) # [dest_token+1, d_model+1]

    for upstream_layer in range(layer):
        for upstream_ah_idx in range(model.cfg.n_heads+2): #+2 is the MLP and AH bias cases
            # Input
            if upstream_ah_idx == model.cfg.n_heads: # MLP case:
                upstream_output = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"])
                X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
            elif upstream_ah_idx == model.cfg.n_heads+1: # AH bias case:
                upstream_output = deepcopy(model.b_O[upstream_layer]) # d_model
                # The AH bias adds the same constant for all tokens
                X_out = upstream_output.repeat(dest_token+1, 1) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
            else:
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :, :] # n_tokens x n_tokens
                V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                # Limit the breakdown by dest_token+1
                upstream_output_breakdown = upstream_output_breakdown[:dest_token+1, :dest_token+1, :]

                X_out = deepcopy(upstream_output_breakdown)

                # Broadcasting the operation
                X_out /= cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].unsqueeze(1)
                x_out_i = X_out[dest_token]
            
            # MLP and AH bias cases
            if upstream_ah_idx >= model.cfg.n_heads:
                # Dest contribution
                all_terms = (P_u @ x_out_i) @ Omega_d_times_X_tilde.T # shape [dest_token+1]
                first_term = all_terms[src_token] # First term: when the x_j is the src_token
                second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
                contrib_dest[upstream_layer, upstream_ah_idx, dest_token] = first_term - second_term
            else: # AH case
                # Dest contributions                    
                all_terms = ((x_out_i @ P_u) @ Omega_d_times_X_tilde.T) # shape [dest_tokens+1, dest_tokens + 1]
                first_terms = all_terms[:, src_token]
                second_terms = (all_terms.sum(dim=1) - first_terms) / dest_token
                contrib_dest[upstream_layer, upstream_ah_idx, :] = first_terms - second_terms # shape [dest_token+1]
        
    results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_dest
    results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_src

    # Starting state of the contribution
    upstream_output = deepcopy(cache["blocks.0.hook_resid_pre"])
    X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
    x_out_i = X_out[dest_token]
    
    # Dest contribution
    all_terms = (P_u @ x_out_i) @ Omega_d_times_X_tilde.T # shape [dest_token+1]
    first_term = all_terms[src_token] # First term: when the x_j is the src_token
    second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
    start_dest = first_term - second_term

    return results_dict, start_dest


def __trace_firing_optimized_rope_new_defn_omega(model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> dict:
    
    """
    Tracing version for models that uses RoPE and have bias in the AH, such as Pythia.
    """
        
    results_dict = {
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {},
        "contrib_sv_src": {},
        "contrib_sv_dest": {}
    }

    X = deepcopy(cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id, :, :]) #Float[Tensor, 'n_tokens d_model'] 

    df_decomp_i, df_decomp_j = get_components_used_comparative_new_defn(X, src_token, dest_token, layer, ah_idx, U, S, VT, model_name, device)

    if df_decomp_i is None and df_decomp_j is None:
        results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = []
        results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = []
        results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))
        results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))

        return results_dict, torch.zeros(1), torch.zeros(1)

    # We trace src only if df_decomp_i is None
    if df_decomp_i is None:
        results_dict, start_src = __trace_firing_optimized_rope_new_defn_omega_src_only(
            df_decomp_j,
            model,
            cache,
            prompt_id,
            layer,
            ah_idx,
            dest_token,
            src_token,
            use_svs,
            frac_contrib_thresh,
            U,
            S,
            VT,
            model_name,
            device
        )

        return results_dict, torch.zeros(1), start_src
    
    # We trace dest only if df_decomp_j is None
    if df_decomp_j is None:
        results_dict, start_dest = __trace_firing_optimized_rope_new_defn_omega_dest_only(
            df_decomp_i,
            model,
            cache,
            prompt_id,
            layer,
            ah_idx,
            dest_token,
            src_token,
            use_svs,
            frac_contrib_thresh,
            U,
            S,
            VT,
            model_name,
            device
        )

        return results_dict, start_dest, torch.zeros(1)


    # Decomposing on x_i
    if use_svs:
        last_sv_idx = np.where(df_decomp_i['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_i = df_decomp_i.iloc[:last_sv_idx+1].idx.astype(int).values

    # Decomposing on x_j
    if use_svs:
        last_sv_idx = np.where(df_decomp_j['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_j = df_decomp_j.iloc[:last_sv_idx+1].idx.astype(int).values
    
    results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_i
    results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_j
    
    diff = dest_token - src_token

    contrib_src = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP
    contrib_dest = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))

    # Building all possible x_i, x_j's at once. They will be used across the tracing
    X_tilde = torch.cat([X[:dest_token+1, :], torch.ones(dest_token+1).reshape(-1, 1).to(device)], dim=1)
    x_i_tilde = X_tilde[dest_token] # x_i downstream
    
    # Computing downstream matrices
    # P_u uses only the svs used when we decompose on x_i
    P_u = U["d"][(layer, ah_idx, diff)][:, svs_decomp_i] @ U["d"][(layer, ah_idx, diff)][:, svs_decomp_i].T
    # P_v uses only the svs used when we decompose on x_j
    P_v = VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :].T @ VT["s"][(layer, ah_idx, diff)][svs_decomp_j, :]
    
    Omega_all_d = torch.zeros(dest_token+1, model.cfg.d_model, model.cfg.d_model+1)
    Omega_all_s = torch.zeros(dest_token+1, model.cfg.d_model+1, model.cfg.d_model)
    for j_prime in range(dest_token+1):
        diff_prime = dest_token - j_prime
        Omega_all_d[j_prime, :, :] = U["d"][(layer, ah_idx, diff_prime)] @ torch.diag(S["d"][(layer, ah_idx, diff_prime)]) @ VT["d"][(layer, ah_idx, diff_prime)]
        Omega_all_s[j_prime, :, :] = U["s"][(layer, ah_idx, diff_prime)] @ torch.diag(S["s"][(layer, ah_idx, diff_prime)]) @ VT["s"][(layer, ah_idx, diff_prime)]

    # Caching some computations
    Omega_d_times_X_tilde = torch.bmm(Omega_all_d, X_tilde.unsqueeze(2)).squeeze(2) # [dest_token+1, d_model+1]

    x_i_tilde_times_Omega_s = torch.bmm(x_i_tilde.repeat(dest_token+1, 1).unsqueeze(1), Omega_all_s).squeeze(1)

    for upstream_layer in range(layer):
        for upstream_ah_idx in range(model.cfg.n_heads+2): #+2 is the MLP and AH bias cases
            # Input
            if upstream_ah_idx == model.cfg.n_heads: # MLP case:
                upstream_output = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"])
                X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
            elif upstream_ah_idx == model.cfg.n_heads+1: # AH bias case:
                upstream_output = deepcopy(model.b_O[upstream_layer]) # d_model
                # The AH bias adds the same constant for all tokens
                X_out = upstream_output.repeat(dest_token+1, 1) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
                x_out_i = X_out[dest_token]
            else:
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :, :] # n_tokens x n_tokens
                V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                # Limit the breakdown by dest_token+1
                upstream_output_breakdown = upstream_output_breakdown[:dest_token+1, :dest_token+1, :]

                X_out = deepcopy(upstream_output_breakdown)

                # Broadcasting the operation
                X_out /= cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].unsqueeze(1)
                x_out_i = X_out[dest_token]
            
            # MLP and AH bias cases
            if upstream_ah_idx >= model.cfg.n_heads:
                # Dest contribution
                all_terms = (P_u @ x_out_i) @ Omega_d_times_X_tilde.T # shape [dest_token+1]
                first_term = all_terms[src_token] # First term: when the x_j is the src_token
                second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
                contrib_dest[upstream_layer, upstream_ah_idx, dest_token] = first_term - second_term

                # Src contribution
                all_terms = (x_i_tilde_times_Omega_s * (X_out @ P_v)).sum(dim=1) # shape [dest_token+1]
                first_term = all_terms[src_token] # First term: when the x_j is the src_token
                second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
                contrib_src[upstream_layer, upstream_ah_idx, src_token] = first_term - second_term
            else: # AH case
                # Dest contributions                    
                all_terms = ((x_out_i @ P_u) @ Omega_d_times_X_tilde.T) # shape [dest_tokens+1, dest_tokens + 1]
                first_terms = all_terms[:, src_token]
                second_terms = (all_terms.sum(dim=1) - first_terms) / dest_token
                contrib_dest[upstream_layer, upstream_ah_idx, :] = first_terms - second_terms # shape [dest_token+1]

                # Computing the projections
                proj = torch.matmul(X_out, P_v)
                # Weighting them by the x_i_tilde @ Omega
                weighted_proj = x_i_tilde_times_Omega_s.unsqueeze(1) * proj  
                # Sum over the last dim to get all terms, shape [dest_tokens+1, dest_tokens + 1]
                all_terms = weighted_proj.sum(dim=-1)
                first_term = all_terms[src_token, :]
                second_term = (all_terms.sum(dim=0) - first_term) / dest_token 
                contrib_src[upstream_layer, upstream_ah_idx, :] = first_term - second_term
        
    results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_dest
    results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_src

    # Starting state of the contribution
    upstream_output = deepcopy(cache["blocks.0.hook_resid_pre"])
    X_out = upstream_output[prompt_id, :dest_token+1, :] / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :]
    x_out_i = X_out[dest_token]
    
    # Dest contribution
    all_terms = (P_u @ x_out_i) @ Omega_d_times_X_tilde.T # shape [dest_token+1]
    first_term = all_terms[src_token] # First term: when the x_j is the src_token
    second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
    start_dest = first_term - second_term

    # Src contribution
    all_terms = (x_i_tilde_times_Omega_s * (X_out @ P_v)).sum(dim=1) # shape [dest_token+1]
    first_term = all_terms[src_token] # First term: when the x_j is the src_token
    second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
    start_src = first_term - second_term

    return results_dict, start_dest, start_src

def __trace_firing_optimized_rope_no_bias(model: HookedTransformer,
                    cache: ActivationCache,
                    prompt_id: int,
                    layer: int,
                    ah_idx: int,
                    dest_token: int,
                    src_token: int,
                    use_svs: bool,
                    frac_contrib_thresh: float,
                    U: dict,
                    S: dict,
                    VT: dict,
                    model_name: str,
                    device: str
                    ) -> tuple:
    """
    Tracing version for models that uses RoPE and do not have bias in the AH, such as Gemma-2.
    """

    results_dict = {
        "svs_used_decomp_i": {},
        "svs_used_decomp_j": {},
        "contrib_sv_src": {},
        "contrib_sv_dest": {}
    }

    X = deepcopy(cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id, :, :]).to(device) #Float[Tensor, 'n_tokens d_model'] 

    df_decomp_i, df_decomp_j = get_components_used_comparative_no_bias(X, src_token, dest_token, layer, ah_idx, U, S, VT, model_name, device)

    # return df_decomp_i, df_decomp_j # TODO: remove it

    # Decomposing on x_i
    if use_svs:
        last_sv_idx = np.where(df_decomp_i['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_i = df_decomp_i.iloc[:last_sv_idx+1].idx.astype(int).values

    # Decomposing on x_j
    if use_svs:
        last_sv_idx = np.where(df_decomp_j['sv_frac_contribution'].values.round(5) >= frac_contrib_thresh)[0][0]
    else:
        last_sv_idx = model.cfg.d_head-1 # all SVs
    svs_decomp_j = df_decomp_j.iloc[:last_sv_idx+1].idx.astype(int).values
    
    # For Pythia/RoPE models, the SVs can be different in the two decompositions. So we store them separately
    results_dict["svs_used_decomp_i"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_i
    results_dict["svs_used_decomp_j"][(prompt_id, layer, ah_idx, dest_token, src_token)] = svs_decomp_j
    
    diff = dest_token - src_token

    contrib_src = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1)) # All AHS + MLP
    contrib_dest = torch.zeros((layer, model.cfg.n_heads+2, dest_token+1))

    # Building all possible x_i, x_j's at once. They will be used across the tracing
    X_tilde = deepcopy(X[:dest_token+1, :]).to(device)
    x_i_tilde = X_tilde[dest_token] # x_i downstream
    
    # Computing downstream matrices
    # P_u uses only the svs used when we decompose on x_i
    P_u = U[(layer, ah_idx, diff)][:, svs_decomp_i] @ U[(layer, ah_idx, diff)][:, svs_decomp_i].T
    # P_v uses only the svs used when we decompose on x_j
    P_v = VT[(layer, ah_idx, diff)][svs_decomp_j, :].T @ VT[(layer, ah_idx, diff)][svs_decomp_j, :]
    
    Omega_all = torch.zeros(dest_token+1, model.cfg.d_model, model.cfg.d_model, device=device)
    for j_prime in range(dest_token+1):
        diff_prime = dest_token - j_prime
        Omega_all[j_prime, :, :] = U[(layer, ah_idx, diff_prime)] @ torch.diag(S[(layer, ah_idx, diff_prime)]) @ VT[(layer, ah_idx, diff_prime)]

    # Caching some computations
    Omega_times_X_tilde = torch.bmm(Omega_all, X_tilde.unsqueeze(2)).squeeze(2) # [dest_token+1, d_model+1]

    x_i_tilde_times_Omega = torch.bmm(x_i_tilde.repeat(dest_token+1, 1).unsqueeze(1), Omega_all).squeeze(1)

    for upstream_layer in range(layer):
        for upstream_ah_idx in range(model.cfg.n_heads+2): #+2 is the MLP and AH bias cases
            # Input
            if upstream_ah_idx == model.cfg.n_heads: # MLP case:
                upstream_output = deepcopy(cache[f"blocks.{upstream_layer}.hook_mlp_out"]).to(device)
                X_out = deepcopy(upstream_output[prompt_id, :dest_token+1, :]).to(device) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].to(device)
                x_out_i = X_out[dest_token]
            elif upstream_ah_idx == model.cfg.n_heads+1: # AH bias case:
                upstream_output = deepcopy(model.b_O[upstream_layer]).to(device) # d_model
                # The AH bias adds the same constant for all tokens
                X_out = deepcopy(upstream_output.repeat(dest_token+1, 1)).to(device) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].to(device)
                x_out_i = X_out[dest_token]
            else:
                # This only works for AH
                # Breaking down the upstream_output using the OV circuit linearity
                A = cache[f"blocks.{upstream_layer}.attn.hook_pattern"][prompt_id, upstream_ah_idx, :, :] # n_tokens x n_tokens
                if model_name == "gemma-2-2b":
                    # Gemma-2-2b uses group query attention. Then, ah_idx 0 and 1 have the same hook_v (idx=0).
                    V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx//2, :] # n_tokens x d_head
                else:
                    V = cache[f"blocks.{upstream_layer}.attn.hook_v"][prompt_id, :, upstream_ah_idx, :] # n_tokens x d_head
                upstream_output_breakdown = torch.einsum('ti,ij->tij', A, V) @ model.W_O[upstream_layer, upstream_ah_idx, :, :] # n_tokens x n_tokens (breakdown_token) x d_model

                if model_name == "gemma-2-2b":
                    # Post attention layer norm term (not folded)
                    upstream_output_breakdown *= model.blocks[upstream_layer].ln1_post.w.detach()
                    ln_post_term = cache[f"blocks.{upstream_layer}.ln1_post.hook_scale"][prompt_id]
                    upstream_output_breakdown /= ln_post_term.view(ln_post_term.shape[0], 1, 1)
                
                # Limit the breakdown by dest_token+1
                upstream_output_breakdown = upstream_output_breakdown[:dest_token+1, :dest_token+1, :]

                X_out = deepcopy(upstream_output_breakdown).to(device)

                # Broadcasting the operation                
                X_out /= cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].unsqueeze(1).to(device)
                x_out_i = X_out[dest_token]
            
            # MLP and AH bias cases
            if upstream_ah_idx >= model.cfg.n_heads:
                # Dest contribution
                all_terms = (P_u @ x_out_i) @ Omega_times_X_tilde.T # shape [dest_token+1]
                first_term = all_terms[src_token] # First term: when the x_j is the src_token
                second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
                contrib_dest[upstream_layer, upstream_ah_idx, dest_token] = first_term - second_term

                # Src contribution
                all_terms = (x_i_tilde_times_Omega * (X_out @ P_v)).sum(dim=1) # shape [dest_token+1]
                first_term = all_terms[src_token] # First term: when the x_j is the src_token
                second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
                contrib_src[upstream_layer, upstream_ah_idx, src_token] = first_term - second_term
            else: # AH case
                # Dest contributions                    
                all_terms = ((x_out_i @ P_u) @ Omega_times_X_tilde.T) # shape [dest_tokens+1, dest_tokens + 1]
                first_terms = all_terms[:, src_token]
                second_terms = (all_terms.sum(dim=1) - first_terms) / dest_token
                contrib_dest[upstream_layer, upstream_ah_idx, :] = first_terms - second_terms # shape [dest_token+1]

                # Computing the projections
                proj = torch.matmul(X_out, P_v)
                # Weighting them by the x_i_tilde @ Omega
                weighted_proj = x_i_tilde_times_Omega.unsqueeze(1) * proj  
                # Sum over the last dim to get all terms, shape [dest_tokens+1, dest_tokens + 1]
                all_terms = weighted_proj.sum(dim=-1)
                first_term = all_terms[src_token, :]
                second_term = (all_terms.sum(dim=0) - first_term) / dest_token 
                contrib_src[upstream_layer, upstream_ah_idx, :] = first_term - second_term
        
    results_dict["contrib_sv_dest"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_dest
    results_dict["contrib_sv_src"][(prompt_id, layer, ah_idx, dest_token, src_token)] = contrib_src

    # Starting state of the contribution
    upstream_output = deepcopy(cache["blocks.0.hook_resid_pre"]).to(device)
    X_out = deepcopy(upstream_output[prompt_id, :dest_token+1, :]).to(device) / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, :dest_token+1, :].to(device)
    x_out_i = X_out[dest_token]
    
    # Dest contribution    
    all_terms = (P_u @ x_out_i) @ Omega_times_X_tilde.T # shape [dest_token+1]
    first_term = all_terms[src_token] # First term: when the x_j is the src_token
    second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
    start_dest = first_term - second_term

    # Src contribution
    all_terms = (x_i_tilde_times_Omega * (X_out @ P_v)).sum(dim=1) # shape [dest_token+1]
    first_term = all_terms[src_token] # First term: when the x_j is the src_token
    second_term = (all_terms.sum() - first_term) / dest_token # Second term: x_j' for all j' != j
    start_src = first_term - second_term

    return results_dict, start_dest, start_src

def format_graph_cytoscape(G: nx.Graph,
                            root_node: str,
                            n_layers: int,
                            n_heads: int) -> nx.Graph:
    spacing_factor = 50    # fixed spacing factor from your original code
    node_width = 150       # Cytoscape node width
    node_height = 40       # Cytoscape node height
    intra_layer_gap = 10   # gap between rows within the same layer (in pixels)
    inter_layer_gap = 50   # gap between layers (in pixels)

    # Compute maximum nodes per row using the fixed spacing factor.
    max_nodes_per_row = int(spacing_factor**2 / node_width)
    if max_nodes_per_row < 1:
        max_nodes_per_row = 1

    pos = {}
    # Separate nodes by layer (each node is assumed to be a tuple: (layer, ah_idx, dest_token, src_token))
    layer_nodes = {layer: [] for layer in range(n_layers)}
    for node in G.nodes:
        if node != root_node:
            layer, ah_idx, dest_token, src_token = eval(node)
            layer_nodes[layer].append(node)

    # Sort nodes within each layer (using your original sorting logic)
    for layer in range(n_layers):
        layer_nodes[layer].sort(key=lambda x: int(x[1]) if x[1] not in ["MLP", "AH bias", "Embedding"]
                                 else n_heads + 2 if x[1] == "AH bias"
                                 else n_heads + 1 if x[1] == "MLP"
                                 else n_heads + 3)

    # We'll accumulate a vertical offset to ensure that the entire block of each layer (all its rows)
    # is kept separate from the next layer.
    cumulative_y_offset = 0

    for layer in range(n_layers):
        nodes = layer_nodes[layer]
        if not nodes:
            # Even if a layer is empty, add the inter-layer gap.
            cumulative_y_offset += inter_layer_gap
            continue

        n_nodes = len(nodes)
        # Determine how many rows are needed in this layer.
        n_rows = math.ceil(n_nodes / max_nodes_per_row)
        # Compute the block height for this layer.
        # (For n rows, the total height is n*node_height plus the gaps between rows.)
        block_height = n_rows * node_height + (n_rows - 1) * intra_layer_gap

        # For each row, compute positions.
        for row in range(n_rows):
            # Get the nodes for this row.
            row_nodes = nodes[row * max_nodes_per_row : (row + 1) * max_nodes_per_row]
            n_in_row = len(row_nodes)
            # Horizontally center the nodes: the total width is (n_in_row-1)*node_width.
            total_row_width = (n_in_row - 1) * node_width if n_in_row > 1 else 0
            start_x = - total_row_width / 2
            y = - (cumulative_y_offset + row * (node_height + intra_layer_gap) + node_height / 2)
            for col, node in enumerate(row_nodes):
                x = start_x + col * node_width
                pos[node] = (x, y)
        # After placing this layer, update the cumulative vertical offset.
        cumulative_y_offset += block_height + inter_layer_gap

    # Handle the special root node: place it below the last non-empty layer.
    if root_node in G.nodes:
        # Find the last non-empty layer.
        last_non_empty_layer = None
        for layer in range(n_layers - 1, -1, -1):
            if layer_nodes[layer]:
                last_non_empty_layer = layer
                break
        if last_non_empty_layer is not None:
            last_positions = [pos[node] for node in layer_nodes[last_non_empty_layer] if node in pos]
            avg_x = sum(x for x, _ in last_positions) / len(last_positions)
            # The lowest y (most negative) among the nodes in that layer.
            lowest_y = min(y for _, y in last_positions)
        else:
            avg_x = 0
            lowest_y = 0
        # Place the root node further down.
        root_y = lowest_y - inter_layer_gap - node_height / 2
        pos[root_node] = (avg_x, root_y)

    nx.set_node_attributes(G, {node: {'x': coord[0], 'y': coord[1]} for node, coord in pos.items()})

    # Changing node attributes in cytoscape
    for node in G.nodes:
        if node != root_node:
            node_tuple = eval(node)
            if node_tuple[1] == "MLP":
                G.nodes[node]["shape"] = "ellipse"
                G.nodes[node]["color"] = "#32CD32"
            elif node_tuple[1] == "AH bias":
                G.nodes[node]["shape"] = "diamond"
                G.nodes[node]["color"] = "#FFD700"
            elif node_tuple[1] == "Embedding":
                G.nodes[node]["shape"] = "hexagon"
                G.nodes[node]["color"] = "#00FFFF"

    # Last formatting:
    # All nodes now have the form (layer, ah_idx, dest_token, src_token)
    # Changes to make for better visualization
    # 1. For nodes with ["MLP", "AH bias", "Embedding"], we will remove the src_token (it is not necessary)
    # 2. Nodes with AHs will be a string of the form "AH(layer, ah_idx)\n(dest_token,src_token)"
    # 3. Nodes without AHs will be a string of the form "MLP layer\ndest_token"

    node_names_mapping = {"('IO-S direction', 'end')": "IO-S direction\nend"}
    for node in G.nodes:
        if node != root_node:
            node = eval(node)
            if node[1] in ["MLP", "AH bias", "Embedding"]:
                node_names_mapping[str(node)] = f"{node[1]} {node[0]}\n{node[2]}"
            else:
                node_names_mapping[str(node)] = f"AH({node[0]},{node[1]})\n({node[2]},{node[3]})"
    
    G = nx.relabel_nodes(G, node_names_mapping)

    return G

def format_graph_cytoscape_by_token_pos(G: nx.Graph,
                                        root_node: str,
                                        n_layers: int,
                                        n_heads: int,
                                        sentence_tokens: list) -> nx.Graph:
    # Constants for positioning and spacing.
    node_height = 40         # Height of each node (in pixels)
    intra_group_gap = 30     # Gap between nodes in the same token group (vertical gap)
    inter_layer_gap = 30     # Gap between layers
    token_spacing = 200      # Horizontal spacing for tokens

    pos = {}  # Dictionary to hold positions for each node.

    bottom_y = 0 #min_internal_y + (node_height + inter_layer_gap)

    for i, token in enumerate(sentence_tokens):
        bottom_node_id = token
        x = i * token_spacing
        pos[bottom_node_id] = (x, bottom_y)
        G.add_node(bottom_node_id,
                   x=x,
                   y=bottom_y,
                   background_color="#FFFFFF", 
                   border_color="#FFFFFF",
                   color="#FFFFFF",
                   shape="quare",
                   label=token)
    
    # --- Lay out the internal graph nodes by layer ---
    # Each node is assumed to be a string representation of a tuple:
    #   (layer, ah_idx, dest_token, src_token)
    layer_nodes = {layer: [] for layer in range(n_layers)}
    for node in G.nodes:
        # Exclude the root and any previously added bottom nodes.
        if node == root_node or node in sentence_tokens:
            continue
        # Assume the node name can be evaluated to a tuple.
        layer, ah_idx, dest_token, src_token = eval(node)
        #layer, ah_idx, dest_token = eval(node)       
        layer_nodes[layer].append(node)

    # Sort nodes in each layer by the index of dest_token in sentence_tokens.
    for layer in range(n_layers):
        layer_nodes[layer].sort(key=lambda n: sentence_tokens.index(eval(n)[2])
                                  if eval(n)[2] in sentence_tokens else -1)

    cumulative_y_offset = node_height + inter_layer_gap  # Accumulates vertical space used by layers.

    def node_sort_func(node_str):
        try:
            # Convert string to tuple
            tup = eval(node_str)
            second = tup[1]
            
            # Define custom sort priority
            if second == "Embedding":
                sort_val = -1
            elif second == "AH bias":
                sort_val = 1000000-1
            elif second == "MLP":
                sort_val = 1000000  # arbitrarily large
            elif isinstance(second, int):
                sort_val = second
            else:
                # fallback for unrecognized values
                sort_val = 999999
        except Exception:
            sort_val = 999999  # fallback for bad inputs
        
        return sort_val

    # Process each layer.
    for layer in range(n_layers):
        nodes = layer_nodes[layer]
        if not nodes:
            cumulative_y_offset += node_height + inter_layer_gap
            continue

        # Group nodes in the layer by dest_token index.
        token_groups = {}
        for node in nodes:
            token = eval(node)[2]
            try:
                token_index = sentence_tokens.index(token)
            except ValueError:
                token_index = 0  # Default if token not found.
            token_groups.setdefault(token_index, []).append(node)

        # Determine the vertical block height needed for this layer.
        layer_block_height = 0
        for group in token_groups.values():
            m = len(group)
            group_height = m * node_height + (m - 1) * intra_group_gap
            layer_block_height = max(layer_block_height, group_height)
        if layer_block_height == 0:
            layer_block_height = node_height

        # Assign positions to each node in each token group.
        for token_index, group in token_groups.items():
            base_x = token_index * token_spacing  # x is fixed by token order.
            m = len(group)
            group = sorted(group, key=node_sort_func)
            for i, node in enumerate(group):
                # Stack the nodes vertically in the group so that they’re centered.
                offset = (i - (m - 1) / 2) * (node_height + intra_group_gap)
                y = - (cumulative_y_offset + layer_block_height / 2 + offset)
                pos[node] = (base_x, y)

        cumulative_y_offset += layer_block_height + inter_layer_gap

    # Updating position of bottom nodes
    for i, token in enumerate(sentence_tokens):
        bottom_node_id = token
        x = i * token_spacing
        pos[bottom_node_id] = (x, -10.)

    # Update position of root_node
    pos[root_node] = (x, min(x[1] for x in pos.values()) - 50) # x is from the last x
        
    # --- Update positions for all nodes in G ---
    nx.set_node_attributes(G, {node: {'x': coord[0], 'y': coord[1]} for node, coord in pos.items()})

    # --- Set additional attributes for internal nodes ---
    for node in G.nodes:
        if node == root_node or node in sentence_tokens:
            continue
        node_tuple = eval(node)
        if node_tuple[1] == "MLP":
            G.nodes[node]["shape"] = "ellipse"
            G.nodes[node]["color"] = "#32CD32"
        elif node_tuple[1] == "AH bias":
            G.nodes[node]["shape"] = "diamond"
            G.nodes[node]["color"] = "#FFD700"
        elif node_tuple[1] == "Embedding":
            G.nodes[node]["shape"] = "hexagon"
            G.nodes[node]["color"] = "#00FFFF"

    # --- Optionally, update internal node labels for better visualization ---
    # We avoid relabeling bottom nodes.
    _, dest_token_root = eval(root_node)
    node_names_mapping = {root_node: f"Logit direction\n{dest_token_root}"}
    for node in list(G.nodes):
        if node == root_node or node in sentence_tokens:
            continue
        node_tuple = eval(node)
        if node_tuple[1] in ["MLP", "AH bias", "Embedding"]:
            node_names_mapping[node] = f"{node_tuple[1]} {node_tuple[0]}\n{node_tuple[2]}"            
        else:
            node_names_mapping[node] = f"AH({node_tuple[0]},{node_tuple[1]})\n({node_tuple[2]},{node_tuple[3]})"

    # Relabel only the internal nodes.
    internal_nodes = {node: new_label for node, new_label in node_names_mapping.items()}
    G = nx.relabel_nodes(G, internal_nodes)

    return G