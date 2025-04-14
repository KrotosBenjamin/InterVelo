import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal
from torchdiffeq import odeint

from InterVelo._utils import get_step_size, normal_kl



class LatentODEfunc(nn.Module):
    """
    A class modelling the latent state derivatives with respect to time.

    Parameters
    ----------
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 25)
    """

    def __init__(
        self,
        n_latent: int = 5,
        n_hidden: int = 25,
    ):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        return self

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        Compute the gradient at a given time t and a given state x.

        Parameters
        ----------
        t
            A given time point.
        x
            A given latent state.

        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


def Velo_Euler_func(v, y, t):
    """
    Compute the spliced and unspliced RNA expression based on the velocity and pseudotime using Euler's method.

    Parameters
    ----------
    v
        A given velocity.
    y
        A given expression data.
    t
        A given time point.

    """
    segment_length = 50
    segments = []
    for segment_start in range(0, len(t), segment_length):
        y0 = y [segment_start]
        current_segment = [y0]
        for i in range(1, segment_length):
            if segment_start + i < len(t):
                dt = t[segment_start + i] - t[segment_start + i - 1]
                new_state = current_segment[i - 1] + v[segment_start + i - 1] * dt
                current_segment.append(new_state)
        segments.append(torch.stack(current_segment, dim=0))
    return torch.cat(segments, dim=0)


class Encoder(nn.Module):
    """
    Encoder class generating the time and latent space.

    Parameters
    ----------
    n_int
        The dimensionality of the input.
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 128)
    dropout_rate
        The dropout rate of the 'Dropout' layer.
        (default: 0.1)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    layer_norm
        Whether to include `LayerNorm` layer or not.
        (Default: `False`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 5,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        super(Encoder, self).__init__()
        self.n_latent = n_latent
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_int, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        if layer_norm:
            self.fc.add_module('N2',nn.LayerNorm(n_hidden, elementwise_affine=False))
        if dropout_rate > 0:
            self.fc.add_module('D1',nn.Dropout(p=dropout_rate))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_latent*2)
        self.fc3 = nn.Sequential()
        self.fc3.add_module('L2',nn.Linear(n_hidden, 1))
        if dropout_rate > 0:
            self.fc3.add_module('D2',nn.Dropout(p=dropout_rate))

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        return self
    
    def forward(self, x:torch.Tensor) -> tuple:
        x = self.fc(x)
        out = self.fc2(x)
        qz_mean, qz_logvar = out[:, :self.n_latent], out[:, self.n_latent:]
        t = self.fc3(x).sigmoid()
        return t, qz_mean, qz_logvar



class Decoder1(nn.Module):
    """
    Decoder class to reconstruct the original input based on its latent space.

    Parameters
    ----------

    n_int
        The dimensionality of the input.
    n_latent
        The dimensionality of the latent space.
        (Default: 20)
    n_hidden
        The dimensionality of the hidden layer for the VAE.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    layer_norm
        Whether to include `LayerNorm` layer or not.
        (Default: `False`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_hidden: int = 128,
        batch_norm: bool = False,
        layer_norm: bool = False,

    ):
        super(Decoder1, self).__init__()
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        if layer_norm:
            self.fc.add_module('N2',nn.LayerNorm(n_hidden, elementwise_affine=False))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_int)


    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        return self
    
    def forward(self, z: torch.Tensor):
        out = self.fc(z)
        recon_x = self.fc2(out)
        return recon_x

        
class TNODE(nn.Module):
    """
    Class to automatically infer cellular dynamics using VAE and neural ODE.

    Parameters
    ----------

    n_int
        The dimensionality of the input.
    n_latent
        The dimensionality of the latent space.
        (Default: 20)
    n_ode_hidden
        The dimensionality of the hidden layer for the latent ODE function.
        (Default: 25)
    n_hidden
        The dimensionality of the hidden layer for the VAE.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    layer_norm
        Whether to include `LayerNorm` layer or not.
        (Default: `False`)
    ode_method
        Solver for integration.
        (Default: `'euler'`)
    step_size
        The step size during integration.
    alpha_recon_lec
        Scaling factor for reconstruction loss from encoder-derived latent space.
        (Default: 0.5)
    alpha_recon_lode
        Scaling factor for reconstruction loss from ODE-solver latent space.
        (Default: 0.5)
    dropout_rate
        The dropout rate of the 'Dropout' layer.
        (default: 0.1)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_ode_hidden: int = 25,
        n_hidden: int = 128,
        batch_norm: bool = False,
        layer_norm: bool = False,
        ode_method: str = 'euler',
        step_size: Optional[int] = None,
        alpha_recon_lec: float = 0.5,
        alpha_recon_lode: float = 0.5,
        dropout_rate: float = 0.1,
    ):
        super(TNODE, self).__init__()
        self.n_int = n_int
        self.n_latent = n_latent
        self.n_ode_hidden = n_ode_hidden
        self.n_hidden = n_hidden
        self.ode_method = ode_method
        self.step_size = step_size
        self.alpha_recon_lec = alpha_recon_lec
        self.alpha_recon_lode = alpha_recon_lode

        self.lode_func = LatentODEfunc(n_latent, n_ode_hidden)
        self.encoder = Encoder(n_int, n_latent, n_hidden, dropout_rate, batch_norm, layer_norm)
        self.decoder = Decoder1(n_int, n_latent, n_hidden, batch_norm, layer_norm)
    
    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        self.lode_func=self.lode_func.to(device)
        self.encoder=self.encoder.to(device)
        self.decoder=self.decoder.to(device)
        return self

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Given the transcriptomes of cells, this function derives the time and latent space of the cells, as well as reconstructs the transcriptomes.

        Parameters
        ----------
        x
            The input data.
        """

        ## get the time and latent space through Encoder
        T, qz_mean, qz_logvar = self.encoder(x)
        T = T.ravel()  ## odeint requires 1-D Tensor for time
        epsilon = torch.randn(qz_mean.size()).to(T.device)
        z = epsilon * torch.exp(.5 * qz_logvar) + qz_mean

        sort_T, sort_ridx = torch.unique(T, return_inverse=True)

        index = torch.argsort(T)
        T = T[index]
        z = z[index]
        x = x[index]
#        qz_mean = qz_mean[index]
#        qz_logvar = qz_logvar[index]
        index2 = (T[:-1] != T[1:])
        index2 = torch.cat((index2, torch.tensor([True]).to(T.device))) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
        T = T[index2]
        z = z[index2]
        x = x[index2]
#        qz_mean = qz_mean[index2]
#        qz_logvar = qz_logvar[index2]

        ## infer the latent space through ODE solver based on z0, t, and LatentODEfunc
        z0 = z[0]
        options = get_step_size(self.step_size, T[0], T[-1], len(T))
        pred_z = odeint(self.lode_func, z0, T, method = self.ode_method, options = options).view(-1, self.n_latent)
        
        ## reconstruct the input through Decoder and compute reconstruction loss
        pred_x1 = self.decoder(z) ## decode through latent space returned by Encoder
        pred_x2 = self.decoder(pred_z) ## decode through latent space returned by ODE solver
        recon_loss_ec = F.mse_loss(x, pred_x1, reduction='none').sum(-1).mean()
        recon_loss_ode = F.mse_loss(x, pred_x2, reduction='none').sum(-1).mean()
        
        ## compute KL divergence and z divergence
        z_div = F.mse_loss(z, pred_z, reduction='none').sum(-1).mean()
        pz_mean = torch.zeros_like(qz_mean)
        pz_logvar = torch.zeros_like(qz_mean)
        kl_div = normal_kl(qz_mean, qz_logvar, pz_mean, pz_logvar).sum(-1).mean()

        loss1 = self.alpha_recon_lec * recon_loss_ec + self.alpha_recon_lode * recon_loss_ode + z_div

        output_dict = {
            "t": T.view(-1, 1),
            "z": z,
            "x": x,
            "pred_x1": pred_x1,
            "pred_x2": pred_x2,
            "index": index,
            "index2": index2,
            "pred_z": pred_z,
            "loss1": loss1,
            "recon_loss_ec": recon_loss_ec,
            "recon_loss_ode": recon_loss_ode, 
            "kl_div":kl_div, 
            "z_div":z_div,
            "sort_ridx":sort_ridx,
        }
        return output_dict

class VELODecoder(nn.Module):
    """Decodes velocity based on latent space.

    Parameters
    ----------
    n_gene
        The dimensionality of the gene number
    pred_unspliced
        Whether to predict the unspliced velocity
    n_latent
        The dimensionality of the latent space
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_gene: int,
        pred_unspliced:bool = True,
        n_latent: int = 20,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(VELODecoder, self).__init__()
        self.n_gene = n_gene
        self.pred_unspliced = pred_unspliced
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent, n_hidden))
    
        if dropout_rate > 0:
            self.fc.add_module('D1',nn.Dropout(p=dropout_rate))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_gene)
        

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        return self
    
    def forward(self, z: torch.Tensor, t: torch.Tensor):
        #z_in=torch.cat([z,t],dim=1)
        z_in=z

        para = self.fc(z_in)
        para = self.fc2(para)
        para = nn.Softplus()(para)

        return para


    
# VAE model
class InterVELO(nn.Module):
    """InterVelo model.

    Parameters
    ----------
    n_input
        Number of input features
    n_genes
        Number of input gene number
    n_ode_hidden
        Number of nodes per hidden layer of latent space derivative
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    pred_unspliced
        Whether to predict the unspliced velocity
    use_batch_norm
        Whether to include `BatchNorm` layer or not.
    use_layer_norm
        Whether to include `LayerNorm` layer or not.
    loss_select
        Choose the loss type. 
        'loss1_loss2' means the total loss, while 'loss1' represents the loss of unsupervised component only.
    gamma_init
        Whether to give gamma an initial value
    ode_method
        Solver for integration in the unsupervised component.
    step_size
        The step size during integration of unsupervised component.
    alpha_recon_lec
        Scaling factor for reconstruction loss from encoder-derived latent space in loss1.
    alpha_recon_lode
        Scaling factor for reconstruction loss from ODE-solver latent space in loss1.
    kl_weight
        Scaling factor for KL divergence in loss1.
    loss1_scale
        Scaling factor for loss1 (unsupervised component)
    loss2_scale
        Scaling factor for loss2 (supervised component)
    dropout_rate
        Dropout rate for neural networks
    scale1
        Initial value for scaling factor of spliced RNA velocity
    scale2
        Initial value for scaling factor of unspliced RNA velocity
    """

    def __init__(
        self,
        n_input: int,
        n_genes: int,
        n_ode_hidden: int = 25,
        n_hidden: int = 128,
        n_latent: int = 10,
        log_variational: bool = False,
        pred_unspliced: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        loss_select: Literal["loss1","loss1_loss2"] = "loss1_loss2",
        gamma_init: Optional[np.ndarray] = None,        
        ode_method: str = 'euler',
        step_size: Optional[int] = None,
        alpha_recon_lec: float = 0.5,
        alpha_recon_lode: float = 0.5,
        kl_weight: float = 0.5,
        loss1_scale: float = 1.,
        loss2_scale: float = 1.,
        dropout_rate: float = 0.1,
        scale1: float = 1,
        scale2: float = 1,
    ):
        super().__init__()

        self.ode_method = ode_method
        self.step_size = step_size
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.n_input = n_input
        self.n_genes = n_genes
        self.kl_weight = kl_weight
        self.loss1_scale = loss1_scale
        self.loss2_scale = loss2_scale
        self.pred_unspliced = pred_unspliced
        self.alpha_recon_lec = alpha_recon_lec
        self.alpha_recon_lode = alpha_recon_lode
        self.loss_select = loss_select

        # degradation
        if gamma_init is None:
            self.gamma_mean = torch.nn.Parameter(0.1 * torch.randn(n_genes))
        else:
            self.gamma_mean = torch.nn.Parameter(
                torch.from_numpy(gamma_init)
            )

        # splicing
        # first samples around 1
        self.beta_mean = torch.nn.Parameter(0.1 * torch.randn(n_genes))

        self.current_kinetic_rates = {}
     
        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        
        self.scale1 = torch.nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.scale2 = torch.nn.Parameter(torch.tensor(scale2, dtype=torch.float32))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        
        self.z_encoder = TNODE(
            n_int=n_input,
            n_latent=n_latent,
            n_ode_hidden=n_ode_hidden,
            n_hidden=n_hidden,
            batch_norm=use_batch_norm,
            layer_norm=use_layer_norm,
            ode_method=ode_method,
            step_size=step_size,
            alpha_recon_lec=alpha_recon_lec,
            alpha_recon_lode=alpha_recon_lode,
            dropout_rate=dropout_rate,
        )
        self.decoder = VELODecoder(
            n_gene=n_genes,
            pred_unspliced=pred_unspliced,
            n_latent=n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)

        self.z_encoder = self.z_encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.beta_mean = self.beta_mean.to(device)
        self.gamma_mean = self.gamma_mean.to(device)
        self.scale1 = self.scale1.to(device)
        self.scale2 = self.scale2.to(device)
        return self
    
    def forward(self, spliced, unspliced, input_data, mask_u):
        inference_output = self.inference(input_data)
        #mix_z=self.alpha_recon_lec*inference_output["z"]+self.alpha_recon_lode*inference_output["pred_z"]
        sort_ridx = inference_output["sort_ridx"]
        velo = self.generative_velo(inference_output["beta"], inference_output["gamma"],
                                    spliced, unspliced, inference_output["pred_z"][sort_ridx],inference_output["t"][sort_ridx])
        if self.pred_unspliced:
            generative_output = self.generative(inference_output, torch.cat([spliced,unspliced],dim=1), velo, mask_u)
        else:
            generative_output = self.generative(inference_output, spliced, velo, mask_u)
        loss = self.loss(inference_output,generative_output)
        
        return loss, generative_output["t"], velo

    def inference(
        self,
        inputdata,
    ):
        inputdata_ = inputdata
        if self.log_variational:
            inputdata_ = torch.log(0.01 + inputdata)

        output = self.z_encoder(inputdata_)

        pred_z = output["pred_z"]
        z = output["z"]
        loss1 = output["loss1"]
        kl_z = output["kl_div"]
        x = output["x"]
        t = output["t"]
        index = output["index"]
        index2 = output["index2"]
        sort_ridx = output["sort_ridx"]
        pred_x1 = output["pred_x1"]
        pred_x2 = output["pred_x2"]

        gamma, beta = self._get_rates()

        outputs = {
            "t": t,
            "pred_z": pred_z,
            "z": z,
            "x": x,
            "pred_x1": pred_x1,
            "pred_x2": pred_x2,
            "index": index,
            "index2": index2,
            "sort_ridx": sort_ridx,
            "loss1": loss1,
            "kl_z":kl_z,
            "gamma": gamma,
            "beta": beta,
        }
        return outputs

    def _get_rates(self):
        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean), 0, 50)
        self.current_kinetic_rates["beta"] = beta
        self.current_kinetic_rates["gamma"] = gamma

        return gamma, beta
       
    def generative_velo(self, beta, gamma, s, u, z, t):
        alpha = self.decoder(z,t)
        if self.pred_unspliced:
            pred_s = (beta * u - gamma * s)*self.scale1
            pred_u = (alpha - beta * u)*self.scale2
            mean_pred = torch.cat([pred_s, pred_u], dim=1)
            return mean_pred
        else:
            mean_pred = (beta * u - gamma * s) * self.scale1
            return mean_pred

    def generative(self, inference_output, input_data,velo,mask_u):
        unspliced_penalty = 0
        if self.pred_unspliced:
            unspliced_penalty += ((velo[:,self.n_genes:self.n_genes*2]*mask_u)** 2).sum(-1).mean()

        t= inference_output["t"]
        index = inference_output["index"]
        index2 = inference_output["index2"]
        
        y = input_data[index][index2]
        velo = velo[index][index2]

        y_pred = Velo_Euler_func(velo, y, t)
        
        sort_ridx = inference_output["sort_ridx"]
        t = t[sort_ridx]
        self.pseudotime = t

        reconst_loss = F.mse_loss(input_data, y_pred[sort_ridx], reduction='none').sum(-1).mean() + unspliced_penalty
        output={"y_pred": y_pred,
                "reconst_loss": reconst_loss,
                "t": t}
        return(output)

    def loss(
        self,
        inference_outputs,
        generative_outputs,
    ):
        #part1
        loss1 = inference_outputs["loss1"] + self.kl_weight*inference_outputs["kl_z"]
       
        #part2
        loss2=generative_outputs["reconst_loss"]
        
        if self.loss_select == "loss1_loss2" :
            with torch.no_grad():
                balance_ratio = loss1/ loss2
            loss = self.loss1_scale * loss1 + self.loss2_scale * loss2 * balance_ratio
        else:
            loss = self.loss1_scale * loss1

        return loss

    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError
        
    def get_kinetic_rates(self):
        return self.current_kinetic_rates