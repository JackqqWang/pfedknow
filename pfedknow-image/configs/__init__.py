import argparse
import os
import torch

import numpy as np
import torch
import random



def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    else:
        print("Non-deterministic")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    # random initial or not
    parser.add_argument('--ran_initial',  action='store_true', help='--ran_initial will initial a random, default = Fault')
    # recovery or not
    parser.add_argument('--recovery',  action='store_true', help='--recovery will start recovery, default = False')
    # training specific args
    parser.add_argument('--agg_strat', type=str, default='strat1', help='choose from strat1, strat2, strat3')

    parser.add_argument('--dataset', type=str, default='mnist', help='choose from random, stl10, mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prunerate', type = float, default = 0.2)
    # parser.add_argument('--frequency', type = int, default = 80)
    parser.add_argument('--frequency', nargs='*', type=int, default=30)
    parser.add_argument('--finetunetimes', type = int, default = 5)
    parser.add_argument('--prunetimes', type=int, default=5)
    # training with sparsity
    parser.add_argument('--pr',  action='store_true', help='pruning global model')
    parser.add_argument('--fine_tune',  action='store_true', help='fine tune model')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--percent', type=float, default=0.7,
                        help='scale sparse rate (default: 0.5)')
    #Multiteacher
    parser.add_argument("--lamda", type=int, default=1, help="factor of loss2")
    parser.add_argument("--T", type=int, default=2, help="temperature")
    # parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--data_dir', type=str, default='../data/mnist')
    parser.add_argument('--log_fn', type=str, default='output-pfedknow')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument("--log_dir",help="dir for log file;",type=str,default="logs")
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--use_default_hyperparameters', action='store_true')
    # model related params
    parser.add_argument('--model', type=str, default='byol')
    parser.add_argument('--backbone', type=str, default='Vgg_backbone')#'resnet50') Vgg_backbone
    parser.add_argument('--num_epochs', type=int, default=200, help='This will affect learning rate decay')
    parser.add_argument('--heat-epochs', type=int, default=60, help='Number of preheat epoches')
    parser.add_argument('--stop_at_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--proj_layers', type=int, default=None, help="number of projector layers. In cifar experiment, this is set to 2")
    # optimization params
    parser.add_argument('--optimizer', type=str, default='lars_simclr', help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--warmup_lr', type=float, default=0, help='Initial warmup learning rate')
    parser.add_argument('--base_lr', type=float, default=0.3)
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--initial-lr', default=0.0, type=float,metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',help='length of learning rate cosine rampdown (>= length of training)')
 
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1.5e-6)

    parser.add_argument('--eval_after_train', type=str, default=None)
    parser.add_argument('--head_tail_accuracy', action='store_true', help='the acc in first epoch will indicate whether collapse or not, the last epoch shows the final accuracy')
    
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: E")
    parser.add_argument('--local_finetune', type=int, default=5, help="number of local finetune epochs ")
    parser.add_argument('--threhold', type=float, default=0.7, help="number of local finetune threhold")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--distill_round', type=int, default=3, help="Round of multiteacher distillation")
    parser.add_argument('--finetune_round_global', type=int, default=1, help="Round of finetune after ditillation")

    parser.add_argument('--label_rate', type=float, default=0.01, help="the fraction of labeled data")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--threshold_pl', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--phi_g', type=int, default=10, help="tipping point 1")
    parser.add_argument('--psi_g', type=int, default=40, help="tipping point 2")
    parser.add_argument('--comu_rate',type=float, default=0.5,help="the comu_rate of ema model")
    parser.add_argument('--ramp',type=str,default='linear', help="ramp of comu")
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--iid', type=str, default='iid', help='iid')
    parser.add_argument("--seed", type=int, default=1, help="Value of random seed")

    #For PFedMe
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")

    parser.add_argument("--local_epochs", type=int, default=20)
    # parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe", choices=["pFedMe", "PerAvg", "FedAvg"])
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('--heat', type=int, default=0)
    args = parser.parse_args()
    
    if args.debug:
        args.batch_size = 2
        args.stop_at_epoch = 2
        args.num_epochs = 3 # train only one epoch
        args.num_workers = 0
        args.frequency = 1

    assert not None in [args.output_dir, args.data_dir]
    os.makedirs(args.output_dir, exist_ok=True)
    # assert args.stop_at_epoch <= args.num_epochs
    if args.stop_at_epoch is not None:
        if args.stop_at_epoch > args.num_epochs:
            raise Exception
    else:
        args.stop_at_epoch = args.num_epochs

    if args.use_default_hyperparameters:
        raise NotImplementedError
    return args
