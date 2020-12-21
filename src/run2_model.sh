#!/usr/bin/env bash
export PYTHONPATH="$(dirname $(pwd))"
echo "pypath=$PYTHONPATH"
BSPLIT=4
EVALEVERY=100

EX_PER_CLASS=5
for S in 55 2 12 85 22
do
  python -m bit_pytorch.train --name cifar10_5_class_$(date +%F_%H%M%S) \
    --model BiT-M-R50x3 --datadir ../data --logdir ../logs --dataset cifar10 \
    --bit_pretrained_dir ../models --batch_split $BSPLIT --eval_every $EVALEVERY \
    --examples_per_class_seed $S --examples_per_class $EX_PER_CLASS --save 
done
