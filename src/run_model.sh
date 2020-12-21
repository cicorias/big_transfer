#!/usr/bin/env bash
export PYTHONPATH="$(dirname $(pwd))"
echo "pypath=$PYTHONPATH"
python -m bit_pytorch.train --name cifar10_$(date +%F_%H%M%S) \
    --model BiT-M-R50x3 --datadir ../data --logdir ../logs --dataset cifar10 \
    --bit_pretrained_dir ../models --batch_split 4 --eval_every 4 \
    --examples_per_class_seed 42 --examples_per_class 1 --save 