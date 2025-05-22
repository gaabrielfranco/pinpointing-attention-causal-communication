from tqdm import tqdm
from transformer_lens import HookedTransformer
import torch
import einops
import numpy as np
import h5py
import argparse
from numpy.linalg import svd

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

# Constants
device = "cuda" if torch.cuda.is_available() else "cpu"
n_toks = 21 # Maximum number of tokens in the input sequence
model_name = "EleutherAI/pythia-160m"

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

with h5py.File(f'matrices_pythia-160m.hdf5', 'w') as f:
    U, S, VT = {"d": {}, "s": {}}, {"d": {}, "s": {}}, {"d": {}, "s": {}}
    for layer in tqdm(range(model.cfg.n_layers)):
        for ah_idx in range(model.cfg.n_heads):
            for diff in range(n_toks):
                dataset_name = f"{layer}_{ah_idx}_{diff}"                
                omega =  {}

                # Omega_d
                omega["d"] = torch.column_stack([
                    model.W_Q[layer, ah_idx, :, :].cpu() @ R[diff] @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                    (model.W_Q[layer, ah_idx, :, :].cpu() @ R[diff] @ model.b_K[layer, ah_idx, :])
                ]).numpy()

                # Omega_s
                omega["s"] = torch.row_stack([
                    model.W_Q[layer, ah_idx, :, :].cpu() @ R[diff] @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                    model.b_Q[layer, ah_idx, :].cpu() @ R[diff] @ model.W_K[layer, ah_idx, :, :].cpu().T
                ]).numpy()

                for omega_type in ["d", "s"]:
                    Ubig, Sbig, VTbig = np.linalg.svd(omega[omega_type])
                    U[omega_type][dataset_name], S[omega_type][dataset_name], VT[omega_type][dataset_name] = Ubig[:,:rank], Sbig[:rank], VTbig[:rank]                # Check if the SVD is correct
                    try:
                        assert np.isclose(omega[omega_type], U[omega_type][dataset_name] @ np.diag(S[omega_type][dataset_name]) @ VT[omega_type][dataset_name], atol=1e-5).all()
                    except:
                        print(f"Failed to reconstruct the omega matrix for {dataset_name}")
                        exit()

    # Saving the U matrices of a layer
    for omega_type in ["d", "s"]: 
        for dataset_name in U[omega_type]:
            f.create_dataset(f"U_{omega_type}_{dataset_name}", data=U[omega_type][dataset_name], compression='gzip', compression_opts=9)

        # Saving the S matrices of a layer
        for dataset_name in S[omega_type]:
            f.create_dataset(f"S_{omega_type}_{dataset_name}", data=S[omega_type][dataset_name], compression='gzip', compression_opts=9)

        # Saving the VT matrices of a layer
        for dataset_name in VT[omega_type]:
            f.create_dataset(f"VT_{omega_type}_{dataset_name}", data=VT[omega_type][dataset_name], compression='gzip', compression_opts=9)
    
    # Clearing the memory
    del omega, U, S, VT