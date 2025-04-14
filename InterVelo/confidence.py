from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from anndata import AnnData
from scvelo import logging as logg

# Code modified from cross_boundary_correctness
# https://github.com/qiaochen/VeloAE/blob/main/veloproj/eval_util.py
def cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str, str]],
    return_raw: bool = False,
    x_emb_key: str = "umap",
    inplace: bool = True,
    output_key_prefix: str = "",
):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding. If one of the keys in adata.layers, then
            will use the layer and compute the score in the raw space. Otherwise,
            will use the embedding in adata.obsm. Default to "umap".
        inplace (bool): whether to add the score to adata.obs.
        output_key_prefix (str): prefix to the output key. Defaults to "".

    Returns:
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        and
        dict: all_scores indexed by cluster_edges if return_raw is True.
    """

    def _select_emb(adata: AnnData, k_velocity: str, x_emb_key: str):
        if x_emb_key in adata.layers.keys():
            # using embedding from raw space
            x_emb = adata.layers[x_emb_key]
            v_emb = adata.layers[k_velocity]

        else:  # embedding from visualization dimensions
            if x_emb_key.startswith("X_"):
                v_emb_key = k_velocity + x_emb_key[1:]
            else:
                v_emb_key = k_velocity + "_" + x_emb_key
                x_emb_key = "X_" + x_emb_key
            assert x_emb_key in adata.obsm.keys()
            assert v_emb_key in adata.obsm.keys()
            x_emb = adata.obsm[x_emb_key]
            v_emb = adata.obsm[v_emb_key]
        return x_emb, v_emb

    scores = {}
    all_scores = {}
    x_emb, v_emb = _select_emb(adata, k_velocity, x_emb_key)

    for u, v in cluster_edges:
        assert u in adata.obs[k_cluster].cat.categories, f"cluster {u} not found"
        assert v in adata.obs[k_cluster].cat.categories, f"cluster {v} not found"
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns["neighbors"]["indices"][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    all_cbcs_ = np.concatenate([np.array(d) for d in all_scores.values()])

    if inplace:
        adata.uns[f"{output_key_prefix}direction_scores"] = scores
        adata.uns[f"{output_key_prefix}raw_direction_scores"] = all_cbcs_
        logg.info(f"added '{output_key_prefix}direction_scores' (adata.uns)")
        logg.info(f"added '{output_key_prefix}raw_direction_scores' (adata.uns)")

    if return_raw:
        return scores, np.mean(all_cbcs_), all_scores

    return scores, np.mean(all_cbcs_)



def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type

    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe
    Returns:
        list: Selected cells.
    """
    return nodes[adata.obs[k_cluster][nodes].values == target]

# Code modified from cross_boundary_scvelo_probs
# https://github.com/qiaochen/VeloAE/blob/main/veloproj/eval_util.py

def cross_boundary_scvelo_probs(adata, k_cluster, cluster_edges, k_trans_g, return_raw=False):
    """Compute Cross-Boundary Confidence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_trans_g (str): key to the transition graph computed using velocity.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    
    scores = {}
    all_scores = {}
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        type_score = [trans_probs.toarray()[:, nodes].mean() 
                      for trans_probs, nodes in zip(adata.uns[k_trans_g][sel], boundary_nodes) 
                      if len(nodes) > 0]
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    if return_raw:
        return all_scores
    return scores, np.mean(np.concatenate([np.array(d) for d in all_scores.values()]))#, np.mean([sc for sc in scores.values()])
'''
def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: Group-wise aggregation scores.
        float: score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s }
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg
'''