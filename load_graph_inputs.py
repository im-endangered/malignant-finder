import json
import numpy as np
import torch
from torch_geometric.data import Data

def load_npz_as_pyg(npz_path: str, device: str = "cpu"):
    z = np.load(npz_path, allow_pickle=True)

    cell_names = z["cell_names"]               # (N,)
    edge_index = z["edge_index"]               # (2, E)
    edge_weight = z["edge_weight"]             # (E,)
    x = z["rna_feat"]                          # (N, N)
    hvg_genes = z["hvg_genes"]                 # (G_hvg,)
    params = json.loads(str(z["params"]))

    # torch tensors
    edge_index = torch.from_numpy(edge_index).long()
    edge_weight = torch.from_numpy(edge_weight).float()
    x = torch.from_numpy(x).float()

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    # move if requested
    data = data.to(device)
    return data, cell_names, hvg_genes, params
