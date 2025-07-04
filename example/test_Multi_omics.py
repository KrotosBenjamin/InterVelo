import scvelo as scv
import numpy as np
import random
import torch
import scanpy as sc
from scvelo.preprocessing.moments import get_moments
import pandas as pd

from InterVelo.train import train, Constants
from InterVelo.utils import update_dict, autoset_coeff_s
from InterVelo.data import preprocess_data

SEED = 12
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
adata=sc.read_h5ad("greenleaf_multivelo_0525.h5ad")
#data from https://multivelo.readthedocs.io/en/latest/MultiVelo_Fig6.html has been smoothed, the detailed preprocessing steps please refer to MultiVelo tutorial
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
#scv.pp.moments(adata, n_pcs=30, n_neighbors=30) ##30
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
adata = preprocess_data(adata, layers=["Ms","Mu","Mc"], filter_on_r2=False)
spliced = torch.tensor(adata.layers["Ms"])
unspliced = torch.tensor(adata.layers["Mu"])
atac = torch.tensor(adata.layers["Mc"].todense())
inputdata=torch.cat([spliced,unspliced,atac],dim=1)

configs = {
        "name": "InterVelo", # name of the experiment
        "loss_pearson": {"coeff_s": autoset_coeff_s(adata)} ,# Automatic setting of the spliced correlation objective
        "arch": {"args": {"pred_unspliced": False}},
        "trainer": { "epochs": 100},
    }
configs = update_dict(Constants.default_configs, configs)
trainer = train(adata, inputdata, configs)
