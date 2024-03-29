# Authors:
#   Helion du Mas des Bourboux <helion.dumasdesbourboux'at'thalesgroup.com>
#
# MIT License
#
# Copyright (c) 2022 THALES
#   All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 2022 october 21

import torch
import numpy as np

from dataset.modifier import Modifier


class Losses:
    @staticmethod
    def get_loss_function_for(loss_name):
        loss = Losses.get_softmax_cross_entropy_loss
        if loss_name == 'hinge':
            loss = Losses.get_hinge_loss
        elif loss_name == 'squared hinge':
            loss = Losses.get_squared_hinge_loss
        elif loss_name == 'cubed hinge':
            loss = Losses.get_cubed_hinge_loss
        elif loss_name == 'cauchy-schwarz':
            loss = Losses.get_cauchy_schwarz_divergence
        return loss

    @staticmethod
    def get_loss(loss_function: callable, loss_name, outputs, targets, reduction="sum"):
        if "cross" in loss_name:
            return loss_function(outputs, targets, reduction=reduction).detach().cpu().numpy()
        return loss_function(outputs, targets).item()

    @staticmethod
    def get_cauchy_schwarz_divergence(outputs, targets):
        """Arguments:
                outputs: the output of the last level neurons
                targets: the desired output
        """
        outputs_probability = torch.nn.functional.softmax(outputs, dim=1)
        numerator = torch.sum(outputs_probability * targets)
        outputs_norm = torch.sqrt(torch.sum(outputs_probability ** 2))
        targets_norm = torch.sqrt(torch.sum(targets ** 2))
        denominator = outputs_norm * targets_norm
        divergence = -torch.log(numerator / denominator)
        return divergence

    @staticmethod
    def get_hinge_loss(outputs, targets, exponent=1):
        """Arguments:
                outputs: the output of the last level neurons
                targets: the desired output
                exponent: which the exponent for the maximum in the loss
                (hinge,squared hinge,cubed hinge)
        """
        targets = Modifier.convert_from_one_hot_to_minus_one_plus_one_encoding(targets)
        right_part = torch.mul(outputs, targets)
        loss = torch.clamp(0.5 - right_part, min=0.0).pow(exponent).sum()
        return loss

    @staticmethod
    def get_squared_hinge_loss(outputs, targets):
        """Arguments:
                outputs: the output of the last level neurons
                targets: the desired output
        """
        return Losses.get_hinge_loss(outputs, targets, exponent=2)

    @staticmethod
    def get_cubed_hinge_loss(outputs, targets):
        """Arguments:
                outputs: the output of the last level neurons
                targets: the desired output
        """
        return Losses.get_hinge_loss(outputs, targets, exponent=3)

    @staticmethod
    def get_softmax_cross_entropy_loss(outputs, targets, reduction="mean"):
        """softmax_cross_entropy_loss:
            The confidence loss for the target model
            corresponds to L_{01} in the article

        Arguments:
            outputs: the output of the last level neurons
            targets: the desired output
            reduction: how we sum the cross entropy
        """

        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        loss = -targets * log_probs

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.sum() / outputs.shape[0]

        # normalize cross-entropy loss
        loss /= np.log(outputs.shape[1])

        return loss

    @staticmethod
    def get_gan_confidence_fooling_loss(inputs, reduction="mean"):
        """
        GAN loss:
            Get the maximum of the logits
            corresponds to L_{10} in the article

        Arguments:
            inputs (torch.FloatTensor): Outputs of the classifier for the different GAN produced images
            reduction (str): Either 'mean' or 'sum'
        """

        loss = -inputs.max(dim=1)[0] + torch.log(torch.exp(inputs).sum(dim=1))

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()

        loss /= np.log(inputs.shape[1])

        return loss

    @staticmethod
    def get_gan_generated_diversity_loss(
            rand_inputs, gan_outputs, indexes=np.arange(10), reduction="mean", norm=2
    ):
        """
        GAN generated images diversity loss:
            compare the diversity of images with indexes generated by the gan
            corresponds to L_{11} in the article

        Arguments:
            rand_inputs (torch.FloatTensor): random inputs at the entry of the GAN
            gan_outputs (torch.FloatTensor): outputs of the GAN, i.e. generated images
            indexes (list of tuple): indices to compare
            norm (int) : the norm and the power used in the loss (in the formula it's m)
        """

        indexes = indexes[indexes < rand_inputs.shape[0]]

        loss = 0.0
        random_nr_diff = 0.0
        for i in indexes:
            for j in indexes[:i]:
                tmp_rands = (torch.absolute(gan_outputs[i] - gan_outputs[j]) ** norm).mean()
                tmp_outputs = (torch.absolute(rand_inputs[i] - rand_inputs[j]) ** norm).mean()
                loss += tmp_outputs / (1.0 + tmp_rands)
                random_nr_diff += tmp_outputs
        loss /= random_nr_diff

        loss /= 1.0 + 2.0 ** norm

        return loss

    @staticmethod
    def get_gan_model_decision_diversity_loss(gan_inputs, net_outputs, indexes=None, norm=2):
        """
        GAN diversity loss:
            compare the outputs of the classifier
            to two different GAN generated images
            corresponds to L_{12} in the article

        Arguments:
            gan_inputs (torch.FloatTensor): random inputs at the entry of the GAN
            net_outputs (torch.FloatTensor): Outputs of the classifier for the different GAN produced images
            indexes (list of tuple): indices to compare
            norm (int) : the norm and the power used in the loss (in the formula it's m)
        """

        probs = torch.nn.functional.softmax(net_outputs, dim=1)
        log_probs = torch.nn.functional.log_softmax(net_outputs, dim=1)
        # this is not consistent! why did you use 10 above, but here everything?
        if indexes is None:
            indexes = np.arange(net_outputs.shape[1])
        indexes = indexes[indexes < gan_inputs.shape[0]]

        loss = 0.0
        random_numbers_diff = 0.0
        for i in indexes:
            for j in indexes[:i]:
                tmp_target = (-probs[i] * log_probs[j]).sum()
                tmp_rands = (torch.absolute(gan_inputs[i] - gan_inputs[j]) ** norm).mean()
                loss += tmp_rands / (1.0 + tmp_target)
                random_numbers_diff += tmp_rands

        loss /= random_numbers_diff
        loss /= 1.0 + np.log(net_outputs.shape[1])  # ???

        return loss

    @staticmethod
    def get_combined_gan_loss(rand_inputs, gan_outputs, net_outputs, reduction="mean"):
        """ Combined gan losses:
                represents the mean of all the losses for the gan
            Arguments:
                rand_inputs (torch.FloatTensor): random inputs at the entry of the GAN
                gan_outputs (torch.FloatTensor): the outputs of the gan
                net_outputs (torch.FloatTensor): Outputs of the classifier for the different GAN produced images
                reduction (str) : the reduction that we will use for the confidence fooling loss
        """

        loss10 = Losses.get_gan_confidence_fooling_loss(net_outputs, reduction=reduction)
        loss11 = Losses.get_gan_model_decision_diversity_loss(rand_inputs, net_outputs)
        loss12 = Losses.get_gan_generated_diversity_loss(rand_inputs, gan_outputs, reduction=reduction)
        loss = loss10 + loss11 + loss12
        loss /= 3.0

        return loss
