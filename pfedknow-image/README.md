

## Getting Started

- Python 3.6
- Pytorch  `conda install torch`(To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).)
- Numpy   `conda install numpy`
- Pandas   `conda install pandas`

## Instruction



## Some Examples

We provide some examples here.

#### Cifar10

```powershell
#iid
python pfedknow.py  --data_dir ../data/cifar  --backbone Vgg_backbone --heat 1 --prunetimes 2  --num_users 100 --heat-epochs 30  --dataset cifar10  --batch_size 10 --num_epochs 200 --label_rate 0.1 --iid iid --local_finetune 3 --distill_round 5 --prunerate 0.1 --device cuda:0  --log_fn  pfedknow-svhn-iid
```

```powershell
#noniid
python pfedknow.py  --data_dir ../data/cifar  --backbone Vgg_backbone --heat 1 --prunetimes 2  --num_users 100 --heat-epochs 30  --dataset cifar10  --batch_size 10 --num_epochs 200 --label_rate 0.1 --iid noniid --local_finetune 3 --distill_round 5 --prunerate 0.1 --device cuda:0  --log_fn  pfedknow-svhn-noniid
```

#### Svhn

```powershell
#iid
python pfedknow.py  --data_dir ../data/svhn  --backbone Vgg_backbone --heat 1 --prunetimes 2  --num_users 100 --heat-epochs 30  --dataset svhn  --batch_size 10 --num_epochs 200 --label_rate 0.1 --iid iid --local_finetune 3 --distill_round 5 --prunerate 0.1 --device cuda:0  --log_fn  pfedknow-svhn-iid
```

```powershell
#noniid
python pfedknow.py  --data_dir ../data/svhn  --backbone Vgg_backbone --heat 1 --prunetimes 2  --num_users 100 --heat-epochs 30  --dataset svhn  --batch_size 10 --num_epochs 200 --label_rate 0.1 --iid noniid --local_finetune 3 --distill_round 5 --prunerate 0.1 --device cuda:0  --log_fn  pfedknow-svhn-noniid
```

