#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Holt'

import copy

from utils.utils import *
import torch.nn as nn
import time
from tqdm import tqdm


__doc__ = """
This file contains a serial of useful functions,
like load CS matirx, all dataset, compute GSURE loss, and so on.
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = None
def set_device(dev):
    """Store the torch device used by the DEQ/DNN trainer.
    
    :param dev: Torch device used by tensors created in this module.
    :return: None.
    """
    global device
    device = dev

class Trainer(nn.Module):
    """Trainer for DEQ/DNN supervised NMSE and unsupervised GSURE objectives."""

    def __init__(self, net, train_loader, validation_loader, A=None, Sigma=None) -> None:
        """Create a training runner for one sensing model.
        
        :param net: DEQ or DNN estimator.
        :param train_loader: DataLoader yielding (measurements, channels, sigma_squared).
        :param validation_loader: DataLoader yielding validation batches.
        :param A: Real sensing matrix with shape (2M, 2N).
        :param Sigma: Optional real covariance basis with shape (2M, 2M).
        :return: None.
        """
        super().__init__()
        self.net = net.to(device)
        if A is None:
            raise ValueError("Trainer requires A to build projection losses.")
        self.A = A.to(device)
        self.pinvA = torch.linalg.pinv(self.A).to(device)
        self.P = torch.matmul(self.pinvA, self.A).to(device)
        self.Sigma = Sigma if Sigma is not None else getattr(self.net, "Sigma", None)
        if self.Sigma is not None:
            self.Sigma = self.Sigma.to(device)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.lr = 1e-3
        self.NMSE_loss = NMSE_loss()
        self.gsure_loss = None
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)

    def _ensure_gsure_ready(self):
        """Initialize GSURE helpers from the explicit sensing model."""
        if self.Sigma is None:
            raise ValueError("GSURE training requires Sigma for the sufficient-statistic covariance model.")
        if self.gsure_loss is None:
            self.gsure_loss = GSURE(self.net, self.A, self.Sigma).to(device)

    def compute_GSURE_and_MSE(self, channel_hat, channels):
        # only require the value, do not need grad
        """Compute projected oracle correction, MSE and channel power.
        
        :param channel_hat: Batch-first channel estimate with shape (batch, 2N).
        :param channels: Batch-first reference channels with shape (batch, 2N).
        :return: Tuple (IPMSE, MSE, channel_power).
        """
        with torch.no_grad():
            # expect GSURE + (I - P)*(h_hat - h) = mse
            E_Pmse = torch.mean(torch.sum(torch.square(torch.abs(torch.matmul(channel_hat - channels, (self.gsure_loss.E - self.gsure_loss.P).T))), 1))
            # Oracle = reduce + E_Pmse
            IPMSE = E_Pmse
            # compute mse
            MSE = torch.mean(torch.sum(torch.square(torch.abs(channel_hat - channels)), 1))
            channel_powers = torch.mean(torch.sum(torch.square(torch.abs(channels)), 1))

        return IPMSE, MSE, channel_powers

    def validation(self,):
        """Evaluate GSURE, oracle MSE, MSE and NMSE on the validation loader.

        :return: Tuple (validation_GSURE, validation_Oracle, validation_MSE, validation_NMSE).
        """
        validation_GSURE, validation_Oracle, validation_MSE, validation_NMSE, channels_power_ave = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for measurements, channels, sigma_squared in self.validation_loader:  # each batch has 2000 samples, i.e. all samples
                channels = channels.to(self.net.device())
                measurements = measurements.to(self.net.device())
                sigma_squared = sigma_squared.to(self.net.device())
                batch_size = measurements.shape[0]

                # measurements.requires_grad_(True)
                channel_hat = self.net(measurements)

                GSURE_batch = self.gsure_loss(channel_hat, measurements, sigma_squared)


                IPMSE, MSE_batch, channels_power_batch = self.compute_GSURE_and_MSE(channel_hat, channels)
                reduce_batch = GSURE_batch
                Oracle_batch = reduce_batch + IPMSE

                validation_GSURE += GSURE_batch * batch_size
                validation_Oracle += Oracle_batch * batch_size
                validation_MSE += MSE_batch * batch_size
                channels_power_ave += channels_power_batch * batch_size

        validation_GSURE /= len(self.validation_loader.dataset)
        validation_Oracle /= len(self.validation_loader.dataset)
        validation_MSE /= len(self.validation_loader.dataset)
        channels_power_ave /= len(self.validation_loader.dataset)

        validation_NMSE = validation_MSE / channels_power_ave

        return validation_GSURE, validation_Oracle, validation_MSE, validation_NMSE

    def net_device(self):
        # more robust than self.net.device() in case device() isn't defined
        """Return the device of the first network parameter."""
        return next(self.net.parameters()).device

    def _maybe_reset_peak_mem(self):
        """Reset CUDA peak memory statistics when CUDA is available.
        
        :return: None.
        """
        if torch.cuda.is_available():
            dev = self.net_device()
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)

    def _print_peak_mem(self, prefix: str = ""):
        """Print CUDA peak memory statistics when CUDA is available.
        
        :param prefix: String prepended to the printed memory line.
        :return: None.
        """
        if not torch.cuda.is_available():
            print(f"{prefix}[Peak GPU memory] CUDA not available.")
            return
        dev = self.net_device()
        torch.cuda.synchronize(dev)
        allocated = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)
        print(f"{prefix}[Peak GPU memory] max_memory_allocated = {allocated:.1f} MiB, max_memory_reserved = {reserved:.1f} MiB")

    def train_by_gsure(self, epochs):

        """Train the estimator with unsupervised GSURE.
        
        :param epochs: Number of training epochs.
        :return: Tuple (net, state, bestModel).
        """
        self._ensure_gsure_ready()
        fmt = '[{:3d}/{:3d}]: train NMSE - ({:.6f} dB), validation NMSE - ({:.6f} dB), '
        fmt += 'GSURE - ({:.6f}), Oracle - ({:.6f}), MSE - ({:.6f})'
        fmt += ' | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

        best_validation_GSURE = 99999.0  # GSURE always > 0
        bestModel = None

        total_time = 0.0
        time_hist = []
        validation_GSURE_hist = []
        validation_Oracle_hist = []
        validation_MSE_hist = []
        validation_NMSE_hist = []

        train_GSURE_hist = []
        train_Oracle_hist = []
        train_MSE_hist = []
        train_NMSE_hist = []

        print(self.net)
        print(model_parameters(self.net))

        self._maybe_reset_peak_mem()


        # epoch loop
        for epoch in range(epochs):
            time.sleep(0.5)  # slows progress bar so it won't print on multiple lines
            GSURE_loss_ave, Oracle_ave, MSE_ave, channels_power_ave, NMSE_ave = 0.0, 0.0, 0.0, 0.0, 0.0

            epoch_start_time = time.time()
            tot = len(self.train_loader)
            with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:
                tepoch.set_description("[{:3d}/{:3d}]".format(epoch + 1, epochs))

                for itr, (measurements, channels, sigma_squared) in enumerate(self.train_loader):

                    channels = channels.to(device)
                    measurements = measurements.to(device)  # this measurement is u = A.T * inv(Sigma) * y, rather than y.
                    sigma_squared = sigma_squared.to(device)

                    batch_size = measurements.shape[0]

                    # train mode
                    self.net.train()

                    # fixed-point find, any deploy Jacobin-free backpropagation
                    self.optimizer.zero_grad()
                    channel_hat = self.net(measurements)

                    # gsure loss function
                    gsure_loss = self.gsure_loss(channel_hat, measurements, sigma_squared)  # do not need h_labels
                    gsure_loss.backward()

                    # update parameters
                    self.optimizer.step()

                    # Output training stats after each epoch
                    train_IPMSE, train_MSE, channels_power = self.compute_GSURE_and_MSE(channel_hat, channels)  # need to calculate grad to input
                    train_Oracle = gsure_loss + train_IPMSE

                    GSURE_loss_val = gsure_loss.detach() * batch_size
                    GSURE_loss_ave += GSURE_loss_val
                    Oracle_val = train_Oracle.detach() * batch_size
                    Oracle_ave += Oracle_val
                    MSE_val = train_MSE.detach() * batch_size
                    MSE_ave += MSE_val
                    channels_power_val = channels_power.detach() * batch_size
                    channels_power_ave += channels_power_val

                    tepoch.update(1)
                    # please note that the order is reversed
                    tepoch.set_postfix(train_GSURE="{:.6f}".format(GSURE_loss_val / batch_size),
                                       train_Oracle="{:.6f}".format(Oracle_val / batch_size),
                                       train_MSE="{:.6f}".format(MSE_val / batch_size),
                                       depth="{}".format(self.net.depth),
                                       )

            # average on the whole training set
            GSURE_loss_ave = GSURE_loss_ave / len(self.train_loader.dataset)
            Oracle_ave = Oracle_ave / len(self.train_loader.dataset)
            MSE_ave = MSE_ave / len(self.train_loader.dataset)

            channels_power_ave = channels_power_ave / len(self.train_loader.dataset)
            NMSE_ave = MSE_ave / channels_power_ave
            NMSE_ave = 10 * torch.log10(NMSE_ave)  # NMSE on the 80000 training  dataset

            self.scheduler.step()

            # use the evaluation mode for validation
            self.net.eval()

            validation_GSURE, validation_Oracle, validation_MSE, validation_NMSE = self.validation()

            validation_NMSE = 10 * torch.log10(validation_NMSE)

            # Record training histories.
            validation_GSURE_hist.append(validation_GSURE.cpu())
            validation_Oracle_hist.append(validation_Oracle.cpu())
            validation_MSE_hist.append(validation_MSE.cpu())
            validation_NMSE_hist.append(validation_NMSE.cpu())

            train_GSURE_hist.append(GSURE_loss_ave.cpu())
            train_Oracle_hist.append(Oracle_ave.cpu())
            train_MSE_hist.append(MSE_ave.cpu())
            train_NMSE_hist.append(NMSE_ave.cpu())

            # stop timer
            epoch_end_time = time.time()
            time_epoch = epoch_end_time - epoch_start_time

            time_hist.append(time_epoch)
            total_time += time_epoch

            print(fmt.format(epoch + 1, epochs, NMSE_ave, validation_NMSE,
                             validation_GSURE, validation_Oracle, validation_MSE,
                             self.net.depth,
                             self.optimizer.param_groups[0]['lr'],
                             time_epoch))

            self.net.train()


            if validation_GSURE < best_validation_GSURE: # note we do not need NMSE, since which needs clean labels
                print(f"[BEST] Update bestModel @ epoch {epoch + 1}/{epochs}: "
                      f"validation_GSURE {best_validation_GSURE:.6f} -> {validation_GSURE:.6f}, "
                      f"validation_NMSE {validation_NMSE:.6f} dB")
                best_validation_GSURE = validation_GSURE
                bestModel = copy.deepcopy(self.net.state_dict())

        # save training history at the last epoch
        state = {
            'validation_GSURE_hist': validation_GSURE_hist,
            'validation_Oracle_hist': validation_Oracle_hist,
            'validation_MSE_hist': validation_MSE_hist,
            'validation_NMSE_hist': validation_NMSE_hist,
            'train_GSURE_hist': train_GSURE_hist,
            'train_Oracle_hist': train_Oracle_hist,
            'train_MSE_hist': train_MSE_hist,
            'train_NMSE_hist': train_NMSE_hist,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler
        }

        # ===== CUDA peak memory report (for DEQ vs DU comparison) =====
        self._print_peak_mem(prefix="")

        return self.net, state, bestModel

    def get_stats(self,):
        """Evaluate supervised validation loss and NMSE.

        :return: Tuple (validation_loss, validation_NMSE).
        """
        val_loss = 0.0

        with torch.no_grad():
            for y_val, h_val, _ in self.validation_loader:
                h_val = h_val.to(self.net.device())
                y_val = y_val.to(self.net.device())
                batch_size = y_val.shape[0]

                h_predict = self.net(y_val)
                batch_loss = self.NMSE_loss(h_predict, h_val)
                val_loss += batch_size * batch_loss

        val_loss /= len(self.validation_loader.dataset)
        val_nmse = compute_NMSE(h_predict, h_val)

        return val_loss, val_nmse

    def train_by_nmse(self, epochs):
        """Train the estimator with supervised NMSE.

        :param epochs: Number of training epochs.
        :return: Tuple (net, state, bestModel).
        """

        fmt = '[{:3d}/{:3d}]: train - ({:6.2f} dB, {:6.2e}), validation - ({:6.2f} dB, '
        fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

        best_validation_NMSE = 999999.0
        bestModel = None

        total_time = 0.0
        time_hist = []
        validation_loss_hist = []
        validation_NMSE_hist = []
        train_loss_hist = []
        train_NMSE_hist = []

        print(self.net)
        print(model_parameters(self.net))

        # epoch level loop
        for epoch in range(epochs):
            time.sleep(0.5)  # slows progress bar so it won't print on multiple lines
            loss_ave = 0.0
            train_NMSE = 0.0
            epoch_start_time = time.time()
            tot = len(self.train_loader)

            with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

                tepoch.set_description("[{:3d}/{:3d}]".format(epoch + 1, epochs))

                # (mini-)batch level loop
                # enumerate(train_loader) will take out a batch of data for training in each epoch
                for itr, (measurements, channels, _) in enumerate(self.train_loader):
                    channels = channels.to(self.net.device())
                    measurements = measurements.to(self.net.device())  # this measurement is u = A.T * inv(Sigma) * y, rather than y.
                    batch_size = measurements.shape[0]

                    # specify the train mode
                    self.net.train()

                    # Apply network to get fixed point, and then backprop directly at the fixed point via fixed point theorem
                    self.optimizer.zero_grad()
                    channels_pred = self.net(measurements)
                    output = None

                    # loss function
                    output = self.NMSE_loss(channels_pred, channels)
                    loss_val = output.detach().cpu().numpy() * batch_size
                    loss_ave += loss_val
                    output.backward()
                    self.optimizer.step()

                    # Output training stats after each epoch
                    train_NMSE = compute_NMSE(channels_pred, channels)
                    tepoch.update(1)
                    tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val / batch_size),
                                       train_NMSE="{:f}".format(train_NMSE),
                                       depth="{:5.1f}".format(self.net.depth))

            loss_ave = loss_ave / len(self.train_loader.dataset)

            self.scheduler.step()

            # use the evaluation mode for validation
            self.net.eval()

            validation_loss, validation_NMSE = self.get_stats()

            validation_loss_hist.append(validation_loss)
            validation_NMSE_hist.append(validation_NMSE)
            train_loss_hist.append(loss_ave)
            train_NMSE_hist.append(train_NMSE)

            epoch_end_time = time.time()
            time_epoch = epoch_end_time - epoch_start_time

            time_hist.append(time_epoch)
            total_time += time_epoch

            print(fmt.format(epoch + 1, epochs, train_NMSE, loss_ave,
                             validation_NMSE, validation_loss, self.net.depth,
                             self.optimizer.param_groups[0]['lr'],
                             time_epoch))

            # return to the train mode and continue training
            self.net.train()

            if validation_NMSE < best_validation_NMSE:  
                best_validation_NMSE = validation_NMSE
                bestModel = self.net.state_dict()

        # save training history at the last epoch
        state = {
            'validation_loss_hist': validation_loss_hist,
            'validation_NMSE_hist': validation_NMSE_hist,
            'train_loss_hist': train_loss_hist,
            'train_NMSE_hist': train_NMSE_hist,
            'lr_scheduler': self.scheduler,
            'time_hist': time_hist,
        }
        return self.net, state, bestModel



