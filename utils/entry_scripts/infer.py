import click
import torch

from dataset.modifier import Modifier
from inference.inference import Inference
from training.components.data_loaders import DataLoaders
from training.components.hyperparameters import Hyperparameters
from training.components.networks_data import NetworksData
from training.components.state import State


@click.command(
    context_settings={"show_default": True, "help_option_names": ["-h", "--help"]}
)
@click.option(
    "--path_to_dataset",
    required=True,
    help="Path to the dataset",
)
@click.option(
    '--dataset_type',
    type=click.Choice(['test', 'train', 'validation'],
                      case_sensitive=False),
    default='test',
    help="The type of the dataset [train|validation|test]",
)
@click.option(
    "--path_to_root_folder",
    required=True,
    type=str,
    help="Path where to save results.",
)
@click.option(
    "--nr_classes",
    default=10,
    type=int,
    help="The number of classes in the dataset.",
)
@click.option(
    "--path_to_load_target_model",
    required=True,
    type=str,
    help="Path to load a model from"
)
@click.option(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to run inference on"
)
@click.option(
    "--target_model_network_type",
    default="Net",
    type=str,
    help="Network type name",
)
@click.option(
    '--target_model_residual_units_number',
    type=int,
    default=7,
    help="The number of residual units for the target model",
)
def infer(
        path_to_dataset,
        path_to_root_folder,
        nr_classes,
        path_to_load_target_model,
        device,
        target_model_network_type,
        target_model_residual_units_number,
        dataset_type
):
    data_loaders = DataLoaders(path_to_dataset=path_to_dataset, nr_classes=nr_classes,
                               batch_size=None, k_fold=1)
    data_loaders_by_name = {"train": data_loaders.train[0], "validation": data_loaders.validation[0],
                            "test": data_loaders.test}

    modifier = Modifier(data_loader=data_loaders.validation[0])
    modified_data = modifier(next(iter(data_loaders.validation[0])))
    nr_dimensions = modified_data[0].ndim - 2

    hyperparameters = Hyperparameters(
        batch_size=None,
        target_model_network_type=target_model_network_type,
        k_fold=1,
        target_model_loss='',
        target_model_on_gan_loss='',
        target_model_residual_units_number=target_model_residual_units_number
    )

    state = State(nr_classes=nr_classes, nr_channels=modifier.nr_channels, nr_dimensions=nr_dimensions)
    networks_data = NetworksData(state=state, training_hyperparameters=hyperparameters, create_target_model_only=True)
    networks_data.load_models_if_present(path_to_load_target_model=path_to_load_target_model, path_to_load_gan=None)

    inference = Inference(networks_data.target_model, modifier, device, nr_classes)
    truths, predictions, score = inference.get_inference_results(loader=data_loaders_by_name[dataset_type])
    predictions_correctness = torch.eq(truths, predictions)
    accuracy = torch.sum(predictions_correctness) / len(predictions_correctness)
    print(f"accuracy is: {accuracy.item()}")


if __name__ == "__main__":
    infer()
