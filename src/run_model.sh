#!/usr/bin/env bash
export PYTHONPATH="$(dirname $(pwd))"
echo "pypath=$PYTHONPATH"
BSPLIT=4
EVALEVERY=100

EX_PER_CLASS=5
for S in 42 7 21 58 99
do
  python -m bit_pytorch.train --name cifar10_$(date +%F_%H%M%S) \
    --model BiT-M-R50x3 --datadir ../data --logdir ../logs --dataset cifar10 \
    --bit_pretrained_dir ../models --batch_split $BSPLIT --eval_every $EVALEVERY \
    --examples_per_class_seed $S --examples_per_class 1 --save 
done


EX_PER_CLASS=5
for S in 42 7 21 58 99
do
  python -m bit_pytorch.train --name cifar10_$(date +%F_%H%M%S) \
    --model BiT-M-R50x3 --datadir ../data --logdir ../logs --dataset cifar10 \
    --bit_pretrained_dir ../models --batch_split $BSPLIT --eval_every $EVALEVERY \
    --examples_per_class_seed $S --examples_per_class $EX_PER_CLASS --save 
done