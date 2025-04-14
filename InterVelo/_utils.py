import torch
import numpy as np
import pandas as pd

from pathlib import Path

import json
from typing import Dict

from itertools import repeat
from collections import OrderedDict
from collections.abc import Mapping
from scvelo.core import sum as sum_
from anndata import AnnData


##calculate KL divergence
def normal_kl(mu1, lv1, mu2, lv2):
    """
    Calculate KL divergence
    This function is from torchdiffeq: https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1/2.
    lstd2 = lv2/2.

    kl = lstd2 - lstd1 + (v1 + (mu1-mu2)**2.)/(2.*v2) - 0.5
    return kl

## get step size
def get_step_size(step_size, t1, t2, t_size):
    """
    This function is from get_step_size: https://github.com/LiQian-XC/sctour/blob/main/sctour/_utils.py
    """
    if step_size is None:
        options = {}
    else:
        step_size = (t2 - t1)/t_size/step_size
        options = dict(step_size = step_size)
    return options

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    
def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def update_dict(d: Dict, u: Mapping, copy=False):
    """recursively updates nested dict with values from u."""
    if copy:
        d = d.copy()
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = update_dict(d.get(k, {}), v, copy)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def autoset_coeff_s(adata: AnnData, use_raw: bool = True) -> float:
    """
    Automatically set the weighting for objective term of the spliced
    read correlation. Modified from the scv.pl.proportions function.

    Args:
        adata (Anndata): Anndata object.
        use_raw (bool): use raw data or processed data.

    Returns:
        float: weighting coefficient for objective term of the unpliced read

    This function is from autoset_coeff_s: https://github.com/bowang-lab/DeepVelo/blob/main/deepvelo/utils/preprocess.py
    """
    layers = ["spliced", "unspliced", "ambigious"]
    layers_keys = [key for key in layers if key in adata.layers.keys()]
    counts_layers = [sum_(adata.layers[key], axis=1) for key in layers_keys]

    if use_raw:
        ikey, obs = "initial_size_", adata.obs
        counts_layers = [
            obs[ikey + layer_key] if ikey + layer_key in obs.keys() else c
            for layer_key, c in zip(layers_keys, counts_layers)
        ]
    counts_total = np.sum(counts_layers, 0)
    counts_total += counts_total == 0
    counts_layers = np.array([counts / counts_total for counts in counts_layers])
    counts_layers = np.mean(counts_layers, axis=1)

    spliced_counts = counts_layers[layers_keys.index("spliced")]
    ratio = spliced_counts / counts_layers.sum()

    if ratio < 0.7:
        coeff_s = 0.5
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (less than 70%). "
            f"Suggest using coeff_s {coeff_s}."
        )
    elif ratio < 0.85:
        coeff_s = 0.75
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (between 70% and 85%). "
            f"Suggest using coeff_s {coeff_s}."
        )
    else:
        coeff_s = 1.0
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (more than 85%). "
            f"Suggest using coeff_s {coeff_s}."
        )

    return coeff_s
