#!/usr/bin/env python
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


import os
import click

from py.training.components.paths import Paths
from py.training.components.state import State
from py.training.components.hyperparameters import Hyperparameters
from py.training.training_pipeline import TrainingPipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@click.command(
    context_settings={"show_default": True, "help_option_names": ["_h", "--help"]}
)
@click.option(
    "--path_to_root_folder",
    help="Path where to save results",
    required=True
)
@click.option(
    "--path_to_dataset",
    help="Path to the dataset_handler to run training on",
    required=True
)
@click.option(
    "--nr_classes",
    help="Number of classes in the dataset_handler",
    required=True,
    type=int
)
@click.option(
    "--total_epochs",
    default=100,
    help="Number of epochs"
)
@click.option(
    "--nr_steps_target_model_on_gan",
    default=1,
    help="Number of steps to run the Model against the GAN per Model step",
)
@click.option(
    "--nr_steps_gan",
    default=1,
    help="Number of steps to run the GAN per Model step"
)
@click.option(
    "--nr_steps_target_model_alone",
    default=0,
    help="Number of steps for the model to be alone, without GAN attacks",
)
@click.option(
    "--proportion_target_model_alone",
    default=0.0,
    help="Proportion of the epochs where only the model is learning",
)
@click.option(
    "--path_to_load_target_model",
    default=None,
    help="Path to load a model from"
)
@click.option(
    "--path_to_load_gan",
    default=None,
    help="Path to load a GAN from"
)
@click.option(
    "--batch_size",
    default=512,
    help="Batch size"
)
@click.option(
    "--device",
    default="cuda:0",
    help="Device to run training on"
)
@click.option(
    "--verbose",
    default=True
)
@click.option(
    "--path_to_load_full_given_target_model",
    default=None,
    type=str,
    help="Path where to load a trained network for the given task",
)
@click.option(
    "--target_model_network_type",
    default="Net",
    type=str,
    help="Network name",
)
@click.option(
    "--validation_interval",
    default=25,
    type=int,
    help="At which interval to run k-fold validation",
)
@click.option(
    "--k_fold",
    default=5,
    type=int,
    help="The k for performing k-fold validation",
)
@click.option(
    "--path_to_performances",
    default=None,
    type=str,
    help="The path for loading the performances",
)
@click.option(
    '--target_model_loss',
    type=click.Choice(['cross-entropy', 'hinge', 'squared hinge', 'cubed hinge', 'cauchy-schwarz'],
                      case_sensitive=False),
    default='cross-entropy',
    help="The type of loss used for training the targets model",
)
@click.option(
    '--target_model_on_gan_loss',
    type=click.Choice(['cross-entropy'],
                      case_sensitive=False),
    default='cross-entropy',
    help="The type of loss used for training the targets model on gan",
)
@click.option(
    '--target_model_residual_units_number',
    type=int,
    default=7,
    help="The number of residual units for the targets model",
)
@click.option(
    '--gan_residual_units_number',
    type=int,
    default=5,
    help="The number of residual units for gan",
)
def main(
        path_to_root_folder,
        path_to_dataset,
        nr_classes,
        total_epochs,
        nr_steps_target_model_on_gan,
        nr_steps_gan,
        nr_steps_target_model_alone,
        proportion_target_model_alone,
        path_to_load_target_model,
        path_to_load_gan,
        batch_size,
        device,
        verbose,
        path_to_load_full_given_target_model,
        target_model_network_type,
        validation_interval,
        k_fold,
        path_to_performances,
        target_model_loss,
        target_model_on_gan_loss,
        target_model_residual_units_number,
        gan_residual_units_number
):
    hyperparameters = Hyperparameters(
        batch_size=batch_size,
        total_epochs=total_epochs,
        nr_steps_target_model_on_gan=nr_steps_target_model_on_gan,
        nr_steps_gan=nr_steps_gan,
        nr_step_target_model_alone=nr_steps_target_model_alone,
        proportion_target_model_alone=proportion_target_model_alone,
        target_model_network_type=target_model_network_type,
        k_fold=k_fold,
        validation_interval=validation_interval,
        target_model_loss=target_model_loss,
        target_model_on_gan_loss=target_model_on_gan_loss,
        target_model_residual_units_number=target_model_residual_units_number,
        gan_residual_units_number=gan_residual_units_number
    )
    paths = Paths(dataset=path_to_dataset, root_folder=path_to_root_folder, load_target_model=path_to_load_target_model,
                  load_gan=path_to_load_gan, performances=path_to_performances)
    state = State(
        nr_classes=nr_classes,
        given_target_model=path_to_load_full_given_target_model,
        verbose=verbose,
        device_name=device)

    training_pipeline = TrainingPipeline(
        hyperparameters=hyperparameters,
        paths=paths,
        state=state
    )
    training_pipeline.train()


if __name__ == "__main__":
    main()
