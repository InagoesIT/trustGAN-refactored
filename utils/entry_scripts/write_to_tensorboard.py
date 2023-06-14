import click

from utils.tensorboard_writer import TensorboardWriter


@click.command(
    context_settings={"show_default": True, "help_option_names": ["-h", "--help"]}
)
@click.option(
    "--total_epochs",
    required=True,
    type=int,
    help="Total epochs compiled.",
)
@click.option(
    "--path_to_root_folder",
    required=True,
    type=str,
    help="Path where to save results.",
)
@click.option(
    "--validation_interval",
    required=True,
    type=int,
    help="At which intervals was validation done.",
)
@click.option(
    "--path_to_performances",
    default=None,
    type=str,
    help="The path for loading the performances",
)
@click.option(
    "--path_to_execution_data",
    default=None,
    type=str,
    help="The path for loading the execution data",
)
@click.option(
    "--plot_together",
    type=bool,
    is_flag=True,
    help="The path for loading the performances",
)
@click.option(
    "--plot_only_average_performances",
    type=bool,
    is_flag=True,
    help="""Plot only the average performances of this folder, 
    preparing to merge it with another performances.""",
)
@click.option(
    "--plot_execution_data",
    type=bool,
    is_flag=True,
    help="Plot only the execution data of the model.",
)
def main(
        path_to_root_folder,
        total_epochs,
        validation_interval,
        path_to_performances,
        path_to_execution_data,
        plot_together,
        plot_only_average_performances,
        plot_execution_data
):
    tensorboard_writer = TensorboardWriter(path_to_root_folder,
                                           total_epochs,
                                           validation_interval,
                                           path_to_performances, 
                                           path_to_execution_data)
    if plot_only_average_performances:
        label = path_to_performances.split("_")[-1].split(".")[0] + "_model"
        tensorboard_writer.plot_model_performances(model_label=label)
    elif plot_together:
        tensorboard_writer.plot_models_together()
    elif plot_execution_data:        
        label = path_to_execution_data.split("_")[-1].split(".")[0] + "_model"
        tensorboard_writer.plot_execution_time(model_label=label)
    else:
        tensorboard_writer.plot_models_separately()


if __name__ == "__main__":
    main()
