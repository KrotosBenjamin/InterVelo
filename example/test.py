import scvelo as scv
import numpy as np
import random
import torch
import scanpy as sc
from scvelo.preprocessing.moments import get_moments

from InterVelo.train import train, Constants
from InterVelo._utils import update_dict, autoset_coeff_s
from InterVelo.data import preprocess_data

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
adata = preprocess_data(adata, layers=["Ms","Mu"], filter_on_r2=False)
#spliced = csr_matrix(adata.layers["spliced"]).astype(np.float32).A
spliced = torch.tensor(adata.layers["Ms"])
unspliced = torch.tensor(adata.layers["Mu"])
inputdata=torch.cat([spliced,unspliced],dim=1)

configs = {
        "name": "InterVelo", # name of the experiment
        "loss_pearson": {"coeff_s": autoset_coeff_s(adata)}  ,# Automatic setting of the spliced correlation objective
        "arch": {"args": {"pred_unspliced": True}},
    }
configs = update_dict(Constants.default_configs, configs)
trainer = train(adata, inputdata, configs)

adata.write(f"./ourmethod_results.h5ad")
