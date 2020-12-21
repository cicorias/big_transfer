


pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy<=1.19.3
# get the model:
curl https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz
copy  .\BiT-M-R50x1-ILSVRC2012.npz BiT-M-R50x1.npz
# run the model -- adding --batch_split 8 per the docs/readmd helps with out of memory issues in GPU cards.
python -m bit_pytorch.train --name cifar10_fff --model BiT-M-R50x1 --logdir ./bit_logs --dataset cifar10 --datadir .\data  --batch_split 8
# you STOP this with a ctrl+c - otherwise it will run forever ---




python -m bit_pytorch.train --name cifar10_fff --model BiT-M-R50x1 --logdir ./bit_logs2 --dataset cifar10 --datadir ./data  --batch_split 3 --save --eval_every 10


