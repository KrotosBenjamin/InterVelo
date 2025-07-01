from copy import deepcopy
from typing import Callable, Mapping
import numpy as np

import torch
from anndata import AnnData
import scvelo as scv
from scvelo import logging as logg
from scipy.stats import pearsonr

from InterVelo.parseconfig import ConfigParser
from InterVelo.model import Trainer
import InterVelo.data as module_data
import InterVelo.module as module_arch



# a hack to make constants, see https://stackoverflow.com/questions/3203286
class MetaConstants(type):
    @property
    def default_configs(cls):
        return deepcopy(cls._default_configs)


class Constants(object, metaclass=MetaConstants):
    _default_configs = {
        "name": "InterVelo_project",
        "n_gpu": 1,  # whether to use GPU
        "arch": {
            "type": "InterVELO",
            "args": {
            "n_ode_hidden": 25,
            "n_hidden": 128,
            "n_latent": 20,    
            "log_variational": False,
            "pred_unspliced": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "ode_method": 'euler',
            "step_size": None,
            "alpha_recon_lec": 0.5,
            "alpha_recon_lode": 0.5,
            "loss1_scale": 1.,
            "loss2_scale": 1.,
            "dropout_rate": 0.1, 
            "scale1":1,
            "scale2":1,       
            },
        },
        "data_loader": {
            "type": "VeloDataLoader",
            "args": {
                "batch_size": 1024,
                "shuffle": True,
                "validation_split": 0.1,
                "num_workers": 10,
                "velocity_genes": False,
                "use_scaled_u": False,
            },
        },
        "optimizer": {
            "type": "Adam",
            "args": {"lr": 0.01, "weight_decay": 0.01, "amsgrad": True, "eps":0.01},
        },
        "loss_pearson": {
            "coeff_u": 1.0,
            "coeff_s": 1.0,
        },
        "mask_zeros": False,
        "lr_scheduler": {"type": "StepLR", "args": {"step_size": 1, "gamma": 0.97}},
        "trainer": {
            "check_direction":True,
            "epochs": 100,
            "loss1_epochs": 0,
            "save_dir": "saved/",
            "save_period": 1000,
            "verbosity": 1,
            "early_stop": 10,
            "tensorboard": True,
            "grad_clip": True,
        },
    }


def train(
    adata: AnnData,
    inputdata: torch.Tensor,
    configs: Mapping,
    verbose: bool = False,
    return_kinetic_rates: bool = True,
    callback: Callable = None,
    **kwargs,
):
    n_cells, n_genes = adata.layers["Ms"].shape
    if configs["data_loader"]["args"]["velocity_genes"]:
        n_genes = int(np.sum(adata.var["velocity_genes"]))
    configs["arch"]["args"]["n_genes"] = n_genes 
    config = ConfigParser(configs)
    logger = config.get_logger("train")

    # setup data_loader instances, use adata as the data_source to load inmemory data
    data_loader = config.init_obj("data_loader", module_data, data_source=adata, inputdata=inputdata)
    valid_data_loader = data_loader.split_validation()
    
    model = config.init_obj("arch", module_arch, n_input=inputdata.shape[1])
    logger.info(f"Beginning training of {configs['name']} ...")
    if verbose:
        logger.info(configs)
        logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    def callback_wrapper(epoch):
        # evaluate all and return the velocity matrix (cells, features)
        config_copy = configs["data_loader"]["args"].copy()
        config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata, inputdata=inputdata)
        eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
        velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field = trainer.eval(
            eval_loader, return_kinetic_rates=return_kinetic_rates
        )

        if callback is not None:
            callback(adata, velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, epoch)
        else:
            logg.warn(
                "Set verbose to True but no callback function provided. A possible "
                "callback function accepts at least two arguments: adata, velo_mat "
            )

    if verbose:
        trainer.train_with_epoch_callback(
            callback=callback_wrapper,
            freq=kwargs.get("freq", 30),
        )
    else:
        trainer.train()

    
    config_copy = configs["data_loader"]["args"].copy()
    config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata, inputdata=inputdata)
    eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
    velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field = trainer.eval(
        eval_loader, return_kinetic_rates=return_kinetic_rates
    )

    print("velo_mat shape:", velo_mat.shape)
    # add velocity
    if configs["data_loader"]["args"]["velocity_genes"]:
        # the predictions only contain the velocity genes
        velocity_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
        idx = adata.var["velocity_genes"].values
        velocity_[:, idx] = velo_mat
        if len(velo_mat_u) > 0:
            velocity_u = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
            velocity_u[:, idx] = velo_mat_u
    else:
        velocity_ = velo_mat
        velocity_u = velo_mat_u

    assert adata.layers["Ms"].shape == velocity_.shape
    adata.layers["velocity"] = velocity_  # (cells, genes)
    adata.obs["pseudotime"] = pseudotime
    adata.obsm['X_TNODE'] = mix_z
    adata.obsm['X_VF']= vector_field
    if len(velo_mat_u) > 0:
        adata.layers["velocity_unspliced"] = velocity_u
        logg.hint(f"added 'velocity_unspliced' (adata.layers)")

    logg.hint(f"added 'velocity' (adata.layers)")
    logg.hint(f"added 'pseudotime'(adata.obs)")
    logg.hint(f"added 'X_TNODE'(adata.obsm)")
    logg.hint(f"added 'X_VF'(adata.obsm)")

    if return_kinetic_rates:
        if configs["arch"]["args"]["pred_unspliced"]:
            if configs["data_loader"]["args"]["velocity_genes"]:
                alpha_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                alpha_[:, adata.var["velocity_genes"].values] = alpha_rates
            else:
                alpha_= alpha_rates
            adata.layers['pred_alpha'] = alpha_
            logg.hint(f"added 'pred_alpha'(adata.layers)")
        for k, v in kinetic_rates.items():
            if v is not None:
                if configs["data_loader"]["args"]["velocity_genes"]:
                    v_ = np.zeros(adata.shape, dtype=v.dtype)
                    v_[adata.var["velocity_genes"].values] = v
                    v = v_
                adata.var["pred_" + k] = v
                logg.hint(f"added 'pred_{k}' (adata.var)")

    if configs["trainer"]["check_direction"]:
        scv.tl.velocity_graph(adata, n_jobs=10)
        scv.tl.velocity_pseudotime(adata)
        logg.hint("added 'velocity_pseudotime'(adata.obs)")

    
        A = adata.obs["pseudotime"]
        B = adata.obs["velocity_pseudotime"]
        correlation = np.corrcoef(A, B)[0, 1]

        if correlation < 0:
            logg.hint("Train again to correct direction of pseudotime.")
            if correlation < 0:
                trainer.model.scale1 = torch.nn.Parameter(-trainer.model.scale1)
                trainer.model.scale2 = torch.nn.Parameter(-trainer.model.scale2)
            if verbose:
                trainer.train_with_epoch_callback(
                    callback=callback_wrapper,
                    freq=kwargs.get("freq", 30),
                )
            else:
                trainer.train()

      
            config_copy = configs["data_loader"]["args"].copy()
            config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata, inputdata=inputdata)
            eval_loader = getattr(module_data, configs["data_loader"]["type"])(
                **config_copy
            )
            velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field = trainer.eval(
                eval_loader, return_kinetic_rates=return_kinetic_rates
            )

            print("velo_mat shape:", velo_mat.shape)
            # add velocity
            if configs["data_loader"]["args"]["velocity_genes"]:
                # the predictions only contain the velocity genes
                velocity_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                idx = adata.var["velocity_genes"].values
                velocity_[:, idx] = velo_mat
                if len(velo_mat_u) > 0:
                    velocity_u = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                    velocity_u[:, idx] = velo_mat_u
            else:
                velocity_ = velo_mat
                velocity_u = velo_mat_u

            assert adata.layers["Ms"].shape == velocity_.shape
            adata.layers["velocity"] = velocity_  # (cells, genes)
            adata.obs["pseudotime"] = pseudotime
            adata.obsm['X_TNODE'] = mix_z
            adata.obsm['X_VF'] = vector_field
            if len(velo_mat_u) > 0:
                adata.layers["velocity_unspliced"] = velocity_u
                logg.hint(f"added 'velocity_unspliced' (adata.layers)")
                num_columns = adata.layers["velocity"].shape[1]
                correlations = []
                for i in range(num_columns):
                    corr, _ = pearsonr(adata.layers["velocity"][:, i], adata.layers["velocity_unspliced"][:, i])
                    correlations.append(corr)
                correlation2 = np.mean(correlations)
                if correlation2 < 0:
                    logg.hint(f"the correlation of 'velocity_unspliced' and 'velocity' is negative, consider to reverse 'velocity_unspliced'")

            logg.hint(f"added 'velocity' (adata.layers)")
            logg.hint(f"added 'pseudotime'(adata.obs)")
            logg.hint(f"added 'X_TNODE'(adata.obsm)")
            logg.hint(f"added 'X_VF'(adata.obsm)")

            if return_kinetic_rates:
                if configs["arch"]["args"]["pred_unspliced"]:
                    if configs["data_loader"]["args"]["velocity_genes"]:
                        alpha_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                        alpha_[:, adata.var["velocity_genes"].values] = alpha_rates
                    else:
                        alpha_= alpha_rates
                    adata.layers['alpha'] = alpha_
                    logg.hint(f"added 'alpha'(adata.layers)")
                for k, v in kinetic_rates.items():
                    if v is not None:
                        if configs["data_loader"]["args"]["velocity_genes"]:
                            v_ = np.zeros(adata.shape, dtype=v.dtype)
                            v_[adata.var["velocity_genes"].values] = v
                            v = v_
                        adata.var["pred_" + k] = v
                        logg.hint(f"added 'pred_{k}' (adata.var)")

            scv.tl.velocity_graph(adata, n_jobs=10)
            scv.tl.velocity_pseudotime(adata)
            logg.hint(f"added 'velocity_pseudotime'(adata.obs)")

    logg.hint(f"model scale1: {trainer.model.scale1}")
    logg.hint(f"model scale2: {trainer.model.scale2}")
    return trainer