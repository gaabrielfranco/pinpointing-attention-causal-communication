# import os
# os.environ['HF_HOME'] = "../../cache/"
# os.environ['TRANSFORMERS_CACHE'] = "../../cache/"

from tqdm import tqdm
from transformer_lens import HookedTransformer
import torch
import einops
import numpy as np
import h5py
import argparse

torch.set_grad_enabled(False)

def get_rotary_matrix(idx_rotation, rotary_dim, d_head, angles):
    """
    Compute the rotary matrix for the idx_rotation-th rotation
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

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layer", "-l", type=int)
args = parser.parse_args()
layer = args.layer

print("Layer:", layer)

# Constants
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
n_toks = 22 # Maximum number of tokens in the input sequence
model_name = "gemma-2-2b"

# Load the model
model = HookedTransformer.from_pretrained(model_name, device=device)

# Compute the rotation angles
rotary_dim = model.cfg.rotary_dim
n_ctx = model.cfg.n_ctx
pos = torch.arange(n_ctx)
dim = torch.arange(rotary_dim // 2)
freq = model.cfg.rotary_base ** (dim / (rotary_dim / 2))
freq = einops.repeat(freq, "d -> (2 d)")
angles = pos[:, None] / freq[None, :]

# Compute the omega matrix
R = [None for _ in range(n_toks)]
diffs = set()
for i in range(n_toks):
    for j in range(n_toks):
        if i < j:
            continue
        diff = i - j
        if R[diff] is None:
            R_i = get_rotary_matrix(i, rotary_dim, model.cfg.d_head, angles)
            R_j = get_rotary_matrix(j, rotary_dim, model.cfg.d_head, angles)
            R[diff] = R_i.T @ R_j

# Compute the omega matrices
rank = model.W_Q.shape[-1]
with h5py.File(f'matrices_{layer}.hdf5', 'a') as f:
    for ah_idx in tqdm(range(model.cfg.n_heads)):
        omega, U, S, VT = {}, {}, {}, {}
        for diff in tqdm(range(n_toks)):
            dataset_name = f"{layer}_{ah_idx}_{diff}"                
            omega[dataset_name] = model.W_Q[layer, ah_idx, :, :] @ R[diff] @ model.W_K[layer, ah_idx, :, :].T
            Ubig, Sbig, VTbig = np.linalg.svd(omega[dataset_name])
            U[dataset_name], S[dataset_name], VT[dataset_name] = Ubig[:,:rank], Sbig[:rank], VTbig[:rank]
            # Check if the SVD is correct
            try:
                assert np.isclose(omega[dataset_name], U[dataset_name] @ np.diag(S[dataset_name]) @ VT[dataset_name], atol=1e-5).all()
            except:
                print(f"Failed to reconstruct the omega matrix for {dataset_name}")

        # Saving the U matrices of a layer
        for dataset_name in U:
            f.create_dataset(f"U_{dataset_name}", data=U[dataset_name], compression='gzip', compression_opts=9)

        # Saving the S matrices of a layer
        for dataset_name in S:
            f.create_dataset(f"S_{dataset_name}", data=S[dataset_name], compression='gzip', compression_opts=9)

        # Saving the VT matrices of a layer
        for dataset_name in VT:
            f.create_dataset(f"VT_{dataset_name}", data=VT[dataset_name], compression='gzip', compression_opts=9)
        
        # Clearing the memory
        del omega, U, S, VT