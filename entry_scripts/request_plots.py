#!/usr/entry_scripts/env python
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
from py.utils.graphs_plotter import GraphsPlotter
from py.utils.images_plotter import ImagesPlotter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@click.command(
    context_settings={"show_default": True, "help_option_names": ["_h", "--help"]}
)
@click.option("--path_to_root_folder",
              help="Path where to save results",
              required=True)
@click.option("--total_epochs",
              default=100,
              help="Number of epochs")
@click.option(
    "--validation_interval",
    default=25,
    type=int,
    help="At which interval to run k-fold validation",
)
@click.option(
    "--path_to_performances",
    default=None,
    type=str,
    help="The path for loading the performances",
)
def main(
        path_to_root_folder,
        total_epochs,
        validation_interval,
        path_to_performances,
):
    produce_plots(path_to_root_folder, total_epochs, validation_interval, path_to_performances)


def produce_plots(root_folder, total_epochs, validation_interval, path_to_performances):
    graphs_plotter = GraphsPlotter(root_folder=root_folder, total_epochs=total_epochs,
                                   validation_interval=validation_interval,
                                   path_to_performances=path_to_performances)
    graphs_plotter.plot_performances()
    graphs_plotter.plot_execution_time()
    ImagesPlotter.create_gif(root_folder=root_folder, pattern="example_image_is_best_step_*.png")
    ImagesPlotter.create_gif(root_folder=root_folder, pattern="example_image_not_is_best_step_*.png")
    ImagesPlotter.create_gif(root_folder=root_folder, pattern="example_true_image_min_step_*.png")
    ImagesPlotter.create_gif(root_folder=root_folder, pattern="example_true_image_max_step_*.png")


if __name__ == "__main__":
    main()
