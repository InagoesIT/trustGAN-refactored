#!/bin/bash
clear
rm -R build
rm -R trustgan.egg-info

python -m pip install .
# shellcheck disable=SC2164
#cd xps
#python3 ../bin/trustgan-model-gan-combined-training.py     --path2save "mnist-wo-gan/"     --path2dataset "data/MNIST"     --nr-classes 10     --proportion_net_alone 1     --nr-epochs 10     --batch-size 512     --device "cuda:0"^C
