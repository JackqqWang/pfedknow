## Getting Started

- Python 3.6
- Pytorch  `conda install torch`(To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).)
- Numpy   `conda install numpy`
- Pandas   `conda install pandas`
- transformers "3.1.0"

## Some Examples

We provide some examples here.

#### AG

```powershell
#iid
python  federated_main.py  --log_fn pfedknow-ag-iid.log  --iid 1 --gpu 1 --gpuid cuda:0 --epochs 200  --dataset ag --num_classes 4 --local_bs 32 --bert_model bert-base-uncased --data_dir /data/ag --task_name a --output_dir logs --label_rate 0.01 --num_users 50 --model TINYBERT --model_G BERT 
```

```powershell
#noniid
python  federated_main.py  --log_fn pfedknow-ag-noniid.log  --iid 0 --gpu 1 --gpuid cuda:0 --epochs 200  --dataset ag --num_classes 4 --local_bs 32 --bert_model bert-base-uncased --data_dir /data/ag --task_name a --output_dir logs --label_rate 0.01 --num_users 50 --model TINYBERT --model_G BERT
```
