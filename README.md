# TrustGAN

TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks

This package provides the code developped for the paper:\
"TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks"\
presented at the CAID-2022 (Conference on Artificial Intelligence for Defence) <https://arxiv.org/abs/2211.13991>

## Install

With python, pip and setuptools installed, you simply need to:

```bash
python -m pip install .
```

## Get the data and run a model

We will work within `./execution_data/`, but you can choose whatever over name you desire:

### Get the datasets

You can download in-distribution (ID) sample datasets within `data/`:

```bash
python utils/entry_scripts/download_data.py --path_to_root_folder "execution_data/data/" --dataset "MNIST"
```

You can download out-of-distribution (OOD) sample datasets within `data/`:

```bash
python utils/entry_scripts/download_data.py --path_to_root_folder "execution_data/data/" --dataset "FashionMNIST"
python utils/entry_scripts/download_data.py --path_to_root_folder "execution_data/data/" --dataset "CIFAR10"
```

### Train a model

We will now run two models, one without TrustGAN and another with it,
with a selected device `<device>`:

```bash
python utils/entry_scripts/train_models.py --path_to_root_folder ".."  --path_to_dataset "execution_data/data/MNIST" --nr_classes 10 --total_epochs 1 --batch_size 512 --proportion_target_model_alone 1 --k-fold 1 --device "cuda:0"
```

```bash
python utils/entry_scripts/train_models.py --path_to_root_folder "execution_data" --path_to_dataset "execution_data/data/MNIST" --nr_classes 10 --total_epochs 1 --validation_interval 1 --nr_steps_target_model_alone 1 --device "cuda:0" --k_fold 1 --gan_residual_units_number 1 --target_model_residual_units_number 1 --batch_size 512 --path_to_load_target_model "execution_data/target_model_0.pth" --path_to_load_gan "execution_data/gan_0.pth" 
```

## Test

You can get summary plots and gifs with:

```bash
python utils/entry_scripts/request_plots.py --path_to_root_folder ".." --total_epochs 100 --validation_interval 25 --path_to_performances "average_performances_gan3.npy"
```

You can get convert from npy to tensorboard with:
If you want to plot average performances and then compare it with another model:

```bash
python utils/entry_scripts/write_to_tensorboard.py --path_to_root_folder "results/200-nets/combined" --plot_only_average_performances --total_epochs 100 --validation_interval 25 --path_to_performances "average_performances_hingecubed.npy"
```

If you want to plot average performances and performances for other models from k-fold and compare then:

```bash
python utils/entry_scripts/write_to_tensorboard.py --path_to_root_folder "results/200-nets/hingecubed" --plot_together --total_epochs 100 --validation_interval 25 --path_to_performances "average_performances.npy"
```

If you want to plot average performances and performances for other models from k-fold separately:

```bash
python utils/entry_scripts/write_to_tensorboard.py --path_to_root_folder "results/100-nets/gan3" --total_epochs 100 --validation_interval 25 --path_to_performances "average_performances.npy"
--plot_execution_data
```

If you want to plot the execution data from a folder:

```bash
python utils/entry_scripts/write_to_tensorboard.py --path_to_root_folder "results/200-nets/execution_data" --total_epochs 200 --validation_interval 25 --plot_execution_data --path_to_execution_data "execution_data_big.npy"
```

## Inference

You can get inference results with:

```bash
python utils/entry_scripts/infer.py --path_to_dataset "execution_data/data/MNIST" --path_to_root_folder "infer" --path_to_load_target_model "execution_data/target_model_0.pth" --target_model_residual_units_number 1 --dataset_type "train"
```

## Contributing

If you are interested in contributing to the project, start by reading the [Contributing guide](/CONTRIBUTING.md).

## License

This repository is licensed under the terms of the MIT License (see the file [LICENSE](/LICENSE)).

## Citing

Please cite the following paper if you are using TrustGAN

```bibtex
@ARTICLE{2022arXiv221113991D,
       author = {{du Mas des Bourboux}, H{\'e}lion},
        title = "{TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = nov,
          eid = {arXiv:2211.13991},
        pages = {arXiv:2211.13991},
archivePrefix = {arXiv},
       eprint = {2211.13991},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221113991D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
