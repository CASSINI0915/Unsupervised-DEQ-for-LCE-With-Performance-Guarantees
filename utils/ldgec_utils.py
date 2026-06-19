#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Holt'

import numpy as np
import torch
from prettytable import PrettyTable
import torch.nn as nn
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

__doc__ = """
This file contains a serial of useful functions,
like load CS matirx, all dataset, compute Gsure loss, and so on.
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = None

def set_device(dev):
    """Store the torch device used by legacy LDGEC utilities.
    
    :param dev: Torch device used by tensors created in this module.
    :return: None.
    """
    global device
    device = dev


class NMSE(nn.Module):
    """Normalized MSE loss used by the legacy LDGEC trainer."""

    def __init__(self):
        """Create the NMSE loss module."""
        super(NMSE, self).__init__()

    def forward(self, h_predict, h_label):
        """Compute NMSE = E[||h_predict - h_label||_2^2] / E[||h_label||_2^2].
        
        :param h_predict: Predicted channels with shape (batch, dim).
        :param h_label: Reference channels with shape (batch, dim).
        :return: Scalar NMSE loss.
        """
        numerator = torch.mean(torch.sum(torch.square(torch.abs(h_predict - h_label)), 0))
        denominator = torch.mean(torch.sum(torch.square(torch.abs(h_label)), 0))
        NMSE = numerator / denominator
        loss = NMSE
        return loss


class SURE(nn.Module):
    """Stein unbiased risk estimate for the LDGEC denoiser module.

    SURE estimates MSE from noisy denoiser input r = x + n:
        sure = ||xhat - r||_2^2 + 2 * sigma * div(xhat) - N * sigma
    where sigma is the noise variance and div(xhat) is estimated by
    Monte Carlo perturbations of the current denoiser input.
    """

    def __init__(self, net) -> None:
        """Create the SURE loss for the current LDGEC network.
        
        :param net: LDGEC network whose last active layer supplies Module_B.
        :return: None.
        """
        super(SURE, self).__init__()
        self.net = net
        self.sigma = None

    def monte_Carlo_div(self, xhat, xnoise, sigma):
        """Estimate denoiser divergence by Monte Carlo perturbation.

        :param xhat: Denoised channel estimate with shape (batch, dim).
        :param xnoise: Noisy denoiser input r with shape (batch, dim).
        :param sigma: Denoiser noise variance with shape (batch, 1).
        :return: Divergence estimate with shape (batch,).
        """
        # generate perturbation
        epsilon = torch.maximum(.001 * torch.max(torch.abs(xnoise)), torch.tensor(.00001))
        epsilon = epsilon.to(device)
        eta = torch.randn(xnoise.shape, dtype=xnoise.dtype)
        eta = eta.to(device)

        # calculate divergence
        if torch.is_complex(eta):
            u_perturbed_r = xnoise + torch.real(torch.mul(eta, epsilon))
            x_hat_perturbed_r, _, _ = self.net.Layers[self.net.depth - 1].Module_B(u_perturbed_r, sigma)
            del_x_hat_perturbed_r = torch.real(x_hat_perturbed_r - xhat)
            eta_dx_r = torch.sum(torch.mul(torch.real(eta), del_x_hat_perturbed_r), 1)
            MC_div_r = torch.div(eta_dx_r, epsilon)

            u_perturbed_i = xnoise + 1j * torch.imag(torch.mul(eta, epsilon))
            x_hat_perturbed_i, _, _ = self.net.Layers[self.net.depth - 1].Module_B(u_perturbed_i, sigma)
            del_x_hat_perturbed_i = torch.imag(x_hat_perturbed_i - xhat)
            eta_dx_i = torch.sum(torch.mul(torch.imag(eta), del_x_hat_perturbed_i), 1)
            MC_div_i = torch.div(eta_dx_i, epsilon)

            MC_div = (MC_div_r + MC_div_i)
        elif torch.is_floating_point(eta):
            u_perturbed = xnoise + torch.mul(eta, epsilon)
            x_hat_perturbed, _, _ = self.net.Layers[self.net.depth - 1].Module_B(u_perturbed, sigma)
            del_x_hat_perturbed = x_hat_perturbed - xhat
            eta_dx = torch.sum(torch.mul(eta, del_x_hat_perturbed), 1)
            MC_div = torch.div(eta_dx, epsilon)
        else:
            raise TypeError("Neither real or complex.")

        return MC_div

    def forward(self, xhat, xnoise, sigma):
        """Compute batch-mean SURE for denoiser MSE.
        
        :param xhat: Denoised channel estimate with shape (batch, dim).
        :param xnoise: Noisy denoiser input r with shape (batch, dim).
        :param sigma: Denoiser noise variance with shape (batch, 1).
        :return: Scalar SURE loss.
        """
        N = xhat.shape[1]

        # first term
        x2 = torch.sum(torch.square(torch.abs(xhat - xnoise)), 1)

        div = self.monte_Carlo_div(xhat, xnoise, sigma)

        # combine
        loss = torch.mean(x2 + 2 * sigma * torch.real(div) - N * sigma)

        return loss


class Trainer(nn.Module):
    """Legacy trainer for supervised and layer-wise SURE LDGEC training."""

    def __init__(self, net, train_loader, validation_loader, learning_rate, weight_decay) -> None:
        """Create an LDGEC training runner.
        
        :param net: LDGEC network to train.
        :param train_loader: DataLoader yielding (measurements, channels, sigma).
        :param validation_loader: DataLoader yielding validation batches.
        :param learning_rate: Adam learning rate.
        :param weight_decay: Adam weight decay.
        :return: None.
        """
        super(Trainer, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = net.to(self.device)
        self.device = next(net.parameters()).device
        self.net = net
        self.train_loader = train_loader
        self.val_loader = validation_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.stepsize = 6 # 12 for 20 epochs under nmse, 3 for 5 epochs under sure
        self.gamma = 0.5
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # self.lr_scheduler = StepLR(self.optimizer, step_size=300, gamma=0.8)
        self.save_dir = './results/'
        self.nmse_loss = NMSE()
        self.sure_loss = SURE(self.net)
        self.using_fine_tune = False
        self.fine_tune = False
        self.supervision = True

    def train_unsupervision(self, epochs):
        """Train the current active LDGEC layer with SURE.

        :param epochs: Number of epochs for this layer.
        :return: Trained LDGEC network.
        """

        fmt = '[{:3d}/{:3d}]: train NMSE - ({:.6f} dB), validation NMSE - ({:.6f} dB), '
        fmt += 'SURE - ({:.6f}), MSE - ({:.6f})'
        fmt += ' | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

        best_validation_sure = 99999.0

        total_time = 0.0
        time_hist = []
        validation_sure_hist = []
        validation_MSE_hist = []
        validation_NMSE_hist = []
        train_sure_hist = []
        train_MSE_hist = []
        train_NMSE_hist = []

        print(self.model_parameters())

        for epoch in range(epochs):
            time.sleep(0.05)
            sure_loss_ave, MSE_ave, denominator_ave = 0.0, 0.0, 0.0
            epoch_start_time = time.time()
            tot = len(self.train_loader)

            with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:
                tepoch.set_description("[{:3d}/{:3d}]".format(epoch + 1, epochs))

                for itr, (measurements, channels, sigma) in enumerate(self.train_loader):
                    channels = channels.to(self.net.device())
                    measurements = measurements.to(self.net.device())
                    sigma = sigma.to(self.net.device())
                    batch_size = measurements.shape[0]

                    self.net.train()
                    self.optimizer.zero_grad()
                    depth_ave = self.net.depth

                    self.net.return_state = True
                    self.net.depth = self.net.depth - 1
                    _, r1z, v1z, r2h, v2h = self.net(measurements, sigma)
                    self.net.depth = self.net.depth + 1

                    current_layer = self.net.Layers[self.net.depth - 1]
                    r2z, v2z = current_layer.Module_A(r1z, v1z, measurements, sigma)
                    xnoise, v1h = current_layer.Module_C1(r2z, v2z, r2h, v2h)
                    denoiser_sigma = torch.mean(v1h, 1).unsqueeze(-1)
                    channel_hat, _, _ = current_layer.Module_B(xnoise, denoiser_sigma)
                    self.net.return_state = False

                    sure_loss = self.sure_loss(channel_hat, xnoise, denoiser_sigma)
                    sure_loss.backward()

                    for name, param in self.net.named_parameters():
                        if param.requires_grad and torch.isnan(param.grad).any():
                            print('{} grad has Nan !!!'.format(name))
                            self.optimizer.zero_grad()
                            break
                    else:
                        self.optimizer.step()

                    train_MSE, train_denominator = self.compute_denoiser_mse(channel_hat, channels)
                    sure_loss_val = sure_loss * batch_size
                    sure_loss_ave += sure_loss_val
                    MSE_val = train_MSE * batch_size
                    MSE_ave += MSE_val
                    denominator_val = train_denominator * batch_size
                    denominator_ave += denominator_val

                    tepoch.update(1)
                    tepoch.set_postfix(train_SURE="{:.6f}".format(sure_loss_val / batch_size),
                                       train_MSE="{:.6f}".format(MSE_val / batch_size),
                                       depth="{}".format(self.net.depth),
                                       )

            sure_loss_ave = sure_loss_ave / len(self.train_loader.dataset)
            MSE_ave = MSE_ave / len(self.train_loader.dataset)
            denominator_ave = denominator_ave / len(self.train_loader.dataset)
            NMSE_ave = MSE_ave / denominator_ave
            NMSE_ave = 10 * torch.log10(NMSE_ave)

            self.net.eval()
            validation_sure, validation_MSE, validation_NMSE = self.validation()
            validation_NMSE = 10 * torch.log10(validation_NMSE)

            validation_sure_hist.append(validation_sure.cpu())
            validation_MSE_hist.append(validation_MSE.cpu())
            validation_NMSE_hist.append(validation_NMSE.cpu())

            train_sure_hist.append(sure_loss_ave.cpu())
            train_MSE_hist.append(MSE_ave.cpu())
            train_NMSE_hist.append(NMSE_ave.cpu())

            epoch_end_time = time.time()
            time_epoch = epoch_end_time - epoch_start_time
            time_hist.append(time_epoch)
            total_time += time_epoch

            print(fmt.format(epoch + 1, epochs, NMSE_ave, validation_NMSE,
                             validation_sure, validation_MSE,
                             depth_ave, self.optimizer.param_groups[0]['lr'],
                             time_epoch))

            self.net.train()

            if validation_sure < best_validation_sure:
                best_validation_sure = validation_sure
                if self.net.depth == self.net.max_depth:
                    state = {
                        'validation_sure_hist': validation_sure_hist,
                        'validation_MSE_hist': validation_MSE_hist,
                        'validation_NMSE_hist': validation_NMSE_hist,
                        'net_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler
                    }
                    file_name = self.save_dir + self.net.name() + '_weights.pth'
                    torch.save(state, file_name)
                    print('Model parameters saved to ' + file_name)

            if epoch + 1 == epochs:
                if self.net.depth == self.net.max_depth:
                    state = {
                        'validation_sure_hist': validation_sure_hist,
                        'validation_MSE_hist': validation_MSE_hist,
                        'validation_NMSE_hist': validation_NMSE_hist,
                        'train_sure_hist': train_sure_hist,
                        'train_MSE_hist': train_MSE_hist,
                        'train_NMSE_hist': train_NMSE_hist,
                        'net_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler
                    }
                    file_name = self.save_dir + self.net.name() + '_history.pth'
                    torch.save(state, file_name)
                    print('Training history saved to ' + file_name)

            self.lr_scheduler.step()
            epoch_start_time = time.time()
        return self.net

    def model_parameters(self,):
        # display the parameters and size, i.e. number of parameters
        """Build a table of trainable LDGEC parameter counts.
        
        :return: PrettyTable containing each trainable parameter name and size.
        """
        table = PrettyTable(["Network Parameters name", "# Parameters size"])
        num_params = 0
        for name, parameter in self.net.named_parameters():
            if not parameter.requires_grad:
                continue
            table.add_row([name, parameter.numel()])
            num_params += parameter.numel()
        table.add_row(['TOTAL', num_params])
        return table

    def compute_NMSE(self, h_predict, h_label):
        """Compute global NMSE in dB.
        
        :param h_predict: Predicted channels with shape (batch, dim).
        :param h_label: Reference channels with shape (batch, dim).
        :return: Scalar tensor containing 10 * log10(||h_predict - h_label||_2^2 / ||h_label||_2^2).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h_predict = h_predict.to(device)
        h_label = h_label.to(device)

        with torch.no_grad():
            numerator = torch.sum(torch.square(torch.abs(h_predict - h_label)))
            denominator = torch.sum(torch.square(torch.abs(h_label)))
            NMSE = numerator / denominator
            NMSE = 10 * torch.log10(NMSE)
        return NMSE

    def compute_denoiser_mse(self, channel_hat, channels):
        """Compute batch-mean denoiser MSE and channel power.
        
        :param channel_hat: Denoised channel estimate with shape (batch, dim).
        :param channels: Reference channels with shape (batch, dim).
        :return: Tuple (MSE, channel_power).
        """
        with torch.no_grad():
            MSE = torch.mean(torch.sum(torch.square(torch.abs(channel_hat - channels)), 1))
            denominator = torch.mean(torch.sum(torch.square(torch.abs(channels)), 1))

        return MSE, denominator

    def get_stats(self,):
        """Evaluate supervised validation loss and NMSE.

        :return: Tuple (validation_loss, validation_NMSE).
        """
        test_loss = 0.0

        with torch.no_grad():
            for y_test, h_test, sigma_squared_test in self.val_loader:
                h_test = h_test.to(self.net.device())
                y_test = y_test.to(self.net.device())
                sigma_squared_test = sigma_squared_test.to(self.net.device())
                batch_size = y_test.shape[0]

                h_predict = self.net(y_test, sigma_squared_test)
                batch_loss = self.nmse_loss(h_predict, h_test)
                test_loss += batch_size * batch_loss

        test_loss /= len(self.val_loader.dataset)
        test_NMSE = self.compute_NMSE(h_predict, h_test)

        return test_loss, test_NMSE

    def validation(self,):
        """Evaluate SURE, denoiser MSE and NMSE on the validation loader.

        :return: Tuple (validation_sure, validation_MSE, validation_NMSE).
        """
        validation_sure, validation_MSE, validation_NMSE = 0.0, 0.0, 0.0
        with torch.no_grad():
            for measurements, channels, sigma in self.val_loader:  # each batch has 2000 samples, i.e. all samples
                channels = channels.to(self.net.device())
                measurements = measurements.to(self.net.device())
                sigma = sigma.to(self.net.device())
                batch_size = measurements.shape[0]

                self.net.return_state = True
                self.net.depth = self.net.depth - 1
                _, r1z, v1z, r2h, v2h = self.net(measurements, sigma)
                self.net.depth = self.net.depth + 1

                current_layer = self.net.Layers[self.net.depth - 1]
                r2z, v2z = current_layer.Module_A(r1z, v1z, measurements, sigma)
                xnoise, v1h = current_layer.Module_C1(r2z, v2z, r2h, v2h)
                denoiser_sigma = torch.mean(v1h, 1).unsqueeze(-1)
                channel_hat, _, _ = current_layer.Module_B(xnoise, denoiser_sigma)
                self.net.return_state = False

                sure_batch = self.sure_loss(channel_hat, xnoise, denoiser_sigma)
                MSE_batch, NMSE_batch = self.compute_denoiser_mse(channel_hat, channels)

                validation_sure += sure_batch * batch_size
                validation_MSE += MSE_batch * batch_size
                validation_NMSE += NMSE_batch * batch_size

            validation_sure /= len(self.val_loader.dataset)
            validation_MSE /= len(self.val_loader.dataset)
            validation_NMSE /= len(self.val_loader.dataset)

        validation_NMSE = validation_MSE / validation_NMSE

        # validation_NMSE = 10 * torch.log10(validation_NMSE)

        return validation_sure, validation_MSE, validation_NMSE

    def train_supervision(self, epochs):
        """Train LDGEC with supervised NMSE for the current depth.

        :param epochs: Number of epochs for this stage.
        :return: Trained LDGEC network.
        """

        # print('_____________________________')
        # print('Train {}-th single layer, training : '.format(depth))
        # print('_____________________________')

        fmt = '[{:3d}/{:3d}]: train - ({:6.2f} dB, {:6.2e}), validation - ({:6.2f} dB, '
        fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

        depth_ave = 0.0
        best_validation_NMSE = 0.0

        total_time = 0.0
        time_hist = []
        validation_loss_hist = []
        validation_NMSE_hist = []
        train_loss_hist = []
        train_NMSE_hist = []

        # print(net.Layers[:depth])
        print(self.model_parameters())
        # summary(self.net, input_size=(512, 128))

        # epoch level loop
        for epoch in range(epochs):
            time.sleep(0.05)  # slows progress bar so it won't print on multiple lines
            loss_ave = 0.0
            train_NMSE = 0.0
            epoch_start_time = time.time()
            tot = len(self.train_loader)

            with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

                tepoch.set_description("[{:3d}/{:3d}]".format(epoch + 1, epochs))

                # (mini-)batch level loop
                # enumerate(train_loader) will take out a batch of data for training in each epoch
                for itr, (measurements, channels, sigma) in enumerate(self.train_loader):
                    channels = channels.to(self.net.device())
                    measurements = measurements.to(self.net.device())
                    sigma = sigma.to(self.net.device())
                    batch_size = measurements.shape[0]

                    # specify the train mode
                    self.net.train()

                    # Apply network to get fixed point, and then backprop directly at the fixed point via fixed point theorem
                    self.optimizer.zero_grad()
                    channels_pred = self.net(measurements, sigma)

                    # depth_ave = 0.99 * depth_ave + 0.01 * depth
                    depth_ave = self.net.depth
                    output = None

                    # loss function
                    output = self.nmse_loss(channels_pred, channels)
                    loss_val = output.detach().cpu().numpy() * batch_size
                    loss_ave += loss_val
                    output.backward()
                    #-----check Nan-------
                    for name, param in self.net.named_parameters():
                        if param.requires_grad and torch.isnan(param.grad).any():
                            print('{} grad has Nan !!!'.format(name))
                            self.optimizer.zero_grad()
                            # channels_pred = self.net(measurements, sigma)
                            # output = self.nmse_loss(channels_pred, channels)
                            # output.backward()
                            break
                    # -----check Nan-------
                    else:
                        self.optimizer.step()

                    # Output training stats after each epoch
                    train_NMSE = self.compute_NMSE(channels_pred, channels)
                    tepoch.update(1)
                    tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val / batch_size),
                                       train_NMSE="{:f}".format(train_NMSE),
                                       depth="{:5.1f}".format(self.net.depth))

            loss_ave = loss_ave / len(self.train_loader.dataset)

            # # --------validation---------
            # with torch.no_grad():
            #     for y_test, h_test, sigma_squared_test in self.val_loader:
            #         batch_size = y_test.shape[1]
            #
            #         h_predict = self.net(y_test, sigma_squared_test)
            #         validation_loss = self.nmse_loss(h_predict, h_test)
            #
            # validation_NMSE = self.compute_NMSE(h_predict, h_test)
            # # --------validation---------

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
                             validation_NMSE, validation_loss, depth_ave,
                             self.optimizer.param_groups[0]['lr'],
                             time_epoch))

            # return to the train mode and continue training
            self.net.train()

            # save weights
            if validation_NMSE < best_validation_NMSE:
                best_validation_NMSE = validation_NMSE
                if self.net.depth == self.net.max_depth and (self.fine_tune or not self.using_fine_tune):
                    state = {
                        'test_loss_hist': validation_loss_hist,
                        'test_NMSE_hist': validation_NMSE_hist,
                        'net_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler
                    }
                    file_name = self.save_dir + str(self.net.depth) + '-layers_net' + '_weights.pth'
                    torch.save(state, file_name)
                    print('Model weights saved to ' + file_name)

            # save training history at the last epoch
            if epoch + 1 == epochs:
                if self.net.depth == self.net.max_depth and (self.fine_tune or not self.using_fine_tune):
                    state = {
                        'test_loss_hist': validation_loss_hist,
                        'test_NMSE_hist': validation_NMSE_hist,
                        'train_loss_hist': train_loss_hist,
                        'train_NMSE_hist': train_NMSE_hist,
                        'lr_scheduler': self.lr_scheduler,
                        'time_hist': time_hist,
                    }
                    file_name = self.save_dir + str(self.net.depth) + '-layers_net' + '_history.pth'
                    torch.save(state, file_name)
                    print('Training history saved to ' + file_name)

            self.lr_scheduler.step()
            epoch_start_time = time.time()
        return self.net

    def optimizer_layer(self, weight_decay):  # set requre_grad of parameters without loading into optimizer as false
        """Build an optimizer for the current active LDGEC layer only.
        
        :param weight_decay: Adam weight decay.
        :return: Adam optimizer over parameters of the current layer.
        """
        for t in range(len(self.net.Layers)):
            if t + 1 == self.net.depth:
                for param in self.net.Layers[t].parameters():
                    param.requires_grad = True
            else:
                for param in self.net.Layers[t].parameters():
                    param.requires_grad = False

        # initialize next layer as the last layer
        if self.net.depth >= 2:
            self.param_copy()

        optimizer = torch.optim.Adam(self.net.Layers[self.net.depth - 1].parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        return optimizer

    def optimizer_tune(self, weight_decay):
        """Build an optimizer for fine-tuning all active LDGEC layers.
        
        :param weight_decay: Adam weight decay.
        :return: Adam optimizer over all active trainable parameters.
        """
        for t in range(len(self.net.Layers)):
            if t + 1 <= self.net.depth:
                for param in self.net.Layers[t].parameters():
                    param.requires_grad = True
            else:
                for param in self.net.Layers[t].parameters():
                    param.requires_grad = False

        optimizer = torch.optim.Adam(self.net.parameters(), lr=.1 * self.learning_rate, weight_decay=weight_decay)
        return optimizer

    def param_copy(self,):
        """Copy parameters from the previous layer into the current layer.
        
        :return: None.
        """
        source_layer_index = self.net.depth - 2
        target_layer_index = self.net.depth - 1

        source_params = list(self.net.Layers[source_layer_index].parameters())
        target_params = list(self.net.Layers[target_layer_index].parameters())
        for source_param, target_param in zip(source_params, target_params):
            target_param.data.copy_(source_param.data)

        return None

    def layer_By_layer(self, epochs):
        """Train LDGEC one unfolded layer at a time.
        
        :param epochs: Number of epochs per layer or fine-tuning stage.
        :return: Trained LDGEC network.
        """
        max_depth = len(self.net.Layers)
        for t in range(max_depth):
            # single layer
            self.fine_tune = False
            self.net.depth = t + 1
            self.print_single()
            self.optimizer = self.optimizer_layer(self.weight_decay)
            self.lr_scheduler = StepLR(self.optimizer, step_size=self.stepsize, gamma=self.gamma)
            if self.supervision:
                self.net.return_state = False
                self.net = self.train_supervision(epochs)
            else:
                self.net.return_state = False
                self.net = self.train_unsupervision(epochs)

            # Fine-tune is supervised-only. SURE training remains layer-by-layer.
            if self.supervision and self.using_fine_tune:
                self.fine_tune = True
                self.print_all()
                self.optimizer = self.optimizer_tune(self.weight_decay)
                self.lr_scheduler = StepLR(self.optimizer, step_size=self.stepsize, gamma=self.gamma)
                self.net.return_state = False
                self.net = self.train_supervision(epochs)

        return self.net

    def end2end(self, epochs):
        """Fine-tune the full LDGEC network end to end with supervised NMSE.
        
        :param epochs: Number of fine-tuning epochs.
        :return: Trained LDGEC network.
        """
        max_depth = len(self.net.Layers)
        self.net.depth = max_depth

        if not self.supervision:
            raise ValueError("LDGEC unsupervised SURE training only supports layer-by-layer training.")

        self.fine_tune = True
        self.print_all()
        self.optimizer = self.optimizer_tune(self.weight_decay)
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.stepsize, gamma=self.gamma)
        self.net.return_state = False
        self.net = self.train_supervision(epochs)

        return self.net


    def print_single(self,):
        """Print the header for training one active LDGEC layer.
        
        :return: None.
        """
        print('_____________________________')
        print('Train {}-th single layer... '.format(self.net.depth))
        print('_____________________________')

    def print_all(self,):
        """Print the header for fine-tuning all active LDGEC layers.
        
        :return: None.
        """
        print('_____________________________')
        print('Fine tune {}-layers net... '.format(self.net.depth))
        print('_____________________________')


