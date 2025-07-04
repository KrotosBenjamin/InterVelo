import scvelo as scv
import numpy as np
import random
import torch
import scanpy as sc
from scvelo.preprocessing.moments import get_moments

from InterVelo.train import train, Constants
from InterVelo._utils import update_dict, autoset_coeff_s
from InterVelo.data import preprocess_data

#Setting appropriate initial values can help estimate pseudotime. 
#As the velocity direction is more stable than pseudotime, we suggest taking it as the references
#If the pseudotime is inconsistent with the velocity direction, try changing the random seed.
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
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
        "arch": {"args": {"pred_unspliced": False,
				#"scale1":1,   # Changing the sign of scale1 can help control the direction of pseudotime.
				}},
    	}
configs = update_dict(Constants.default_configs, configs)
trainer = train(adata, inputdata, configs)
