import numpy as np
import torch

from InterVelo.basetrainer import BaseTrainer
from InterVelo._utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, optimizer, config)
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker("loss", writer=self.writer)
        self.valid_metrics = MetricTracker("loss", writer=self.writer)
        self.saved_candidate_ids = {}

    def _compute_core(self, batch_data):
        data_dict = batch_data
        x_u, x_s, inputdata, mask_u = data_dict["Ux_sz"], data_dict["Sx_sz"], data_dict["inputdata"], data_dict["mask_u"]
        x_u, x_s, inputdata, mask_u = (
            x_u.to(self.device),
            x_s.to(self.device),
            inputdata.to(self.device),
            mask_u.to(self.device)
        )
            
        loss, pseudotime, pred = self.model(x_u, x_s, inputdata, mask_u)
        return loss, pseudotime, pred

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if (not self.data_loader.shuffle) and self.data_loader.is_large_batch:
            loader = self.data_loader.dataset.large_batch(self.device)
        else:
            loader = self.data_loader
        
        for batch_idx, batch_data in enumerate(loader):
            loss, pseudotime, mean_pred = self._compute_core(batch_data)
           
            self.optimizer.zero_grad()
            
            loss.backward()
            if self.config["trainer"].get("grad_clip", True):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                loss, pseudotime, mean_pred = self._compute_core(batch_data)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _get_latent(self,
                    batch_data,
                    alpha_z: float = .5,
                    alpha_predz: float = .5,):
        data_dict = batch_data
        x_u, x_s, inputdata = data_dict["Ux_sz"], data_dict["Sx_sz"], data_dict["inputdata"]
        x_u, x_s, inputdata = (
            x_u.to(self.device),
            x_s.to(self.device),
            inputdata.to(self.device),
        )
        
        inference_output = self.model.inference(inputdata)
        sort_ridx = inference_output["sort_ridx"]
        z=inference_output["z"][sort_ridx] 
        pred_z=inference_output["pred_z"][sort_ridx]

        mix_z = alpha_z * z + alpha_predz * pred_z
        return z, pred_z, mix_z

    def _get_vector_field(
        self,
        model,
        T,
        Z,
        time_reverse: bool,
    ):
        """
        Derive the vector field for cells.

        Parameters
        ----------
        model
            The trained scTour model.
        T
            The pseudotime for each cell.
        Z
            The latent representation for each cell.
        time_reverse
            Whether to reverse the vector field.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The estimated vector field.
        """

        model.eval()
        if time_reverse is None:
            raise RuntimeError(
                    'It seems you did not run `get_time()` function first after model training.'
                    )
        direction = 1
        if time_reverse:
            direction = -1
        return direction * model.z_encoder.lode_func(T, Z)

    def _get_alpha(
        self,
        model,
        T,
        Z,
    ):
        """
        Derive the vector field for cells.

        Parameters
        ----------
        model
            The trained scTour model.
        T
            The pseudotime for each cell.
        Z
            The latent representation for each cell.
        time_reverse
            Whether to reverse the vector field.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The estimated vector field.
        """

        model.eval()
        alpha = model.decoder(Z,T)
        return alpha


    def eval(self, eval_loader, return_kinetic_rates=False):
        """
        Evaluate the model on a given dataset, provided by eval_loader

        :param model: model to evaluate
        :param eval_loader: dataset loader containing dataset to evaluate on
        """
        self.model.eval()
        n_genes = self.config["arch"]["args"]["n_genes"]
        velo_mat = []
        velo_mat_u = []
        pseudotime = []
        mix_z = []
        kinetic_rates = {}
        alpha_rates = []
        if (not eval_loader.shuffle) and eval_loader.is_large_batch:
            loader = eval_loader.dataset.large_batch(self.device)
        else:
            loader = eval_loader
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                loss, pseudotime_, mean_pred= self._compute_core(batch_data)
                z,pred_z,mix_z_ = self._get_latent(batch_data,
                    alpha_z = self.config["arch"]["args"]["alpha_recon_lec"],
                    alpha_predz = self.config["arch"]["args"]["alpha_recon_lode"])
                mix_z.append(mix_z_.cpu().data)
                if return_kinetic_rates:
                    cur_alpha_rates = self._get_alpha(self.model, pseudotime, pred_z)
                    alpha_rates.append(cur_alpha_rates.cpu().data)

                pred_s = mean_pred[:, 0:n_genes]
                velo_mat.append(pred_s.cpu().data)
                pseudotime.append(pseudotime_.cpu().data)
                if self.config["arch"]["args"]["pred_unspliced"]:
                    pred_u = mean_pred[:, n_genes:n_genes*2]
                    velo_mat_u.append(pred_u.cpu().data)

            velo_mat = np.concatenate(velo_mat, axis=0)
            pseudotime = np.concatenate(pseudotime, axis=0)
            mix_z = np.concatenate(mix_z, axis=0)
            alpha_rates = np.concatenate(alpha_rates, axis=0)
            
            if self.config["arch"]["args"]["pred_unspliced"]:
                velo_mat_u = np.concatenate(velo_mat_u, axis=0)
        
        from InterVelo.loss import direction_loss
        loss_pearson, reverse = direction_loss(
                        velocity=torch.tensor(velo_mat).to(self.device),
                        spliced_counts=eval_loader.dataset.Sx_sz.to(self.device),
                        unspliced_counts=eval_loader.dataset.Ux_sz.to(self.device),
                        coeff_u=self.config["loss_pearson"]["coeff_u"],
                        coeff_s=self.config["loss_pearson"]["coeff_s"],
                        reduce=False,
                        )
        vector_field = self._get_vector_field(self.model, torch.tensor(pseudotime).to(self.device), torch.tensor(mix_z).to(self.device), reverse)
        vector_field = np.array(vector_field.cpu().data)
        if reverse:
            velo_mat = - velo_mat
            pseudotime = 1 - pseudotime
            print("Reverse pseudotime and velocity for the negative direction loss.")
            if self.config["arch"]["args"]["pred_unspliced"]:
                velo_mat_u = -velo_mat_u
        # record kinetic rates
        if return_kinetic_rates:
            cur_kinetic_rates = self.model.get_kinetic_rates()
            for k, v in cur_kinetic_rates.items():
                if k not in kinetic_rates:
                    kinetic_rates[k] = []
                kinetic_rates[k].append(v.cpu().data)
        if return_kinetic_rates:
            for k, v in kinetic_rates.items():
                kinetic_rates[k] = np.concatenate(v, axis=0)

        return velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field