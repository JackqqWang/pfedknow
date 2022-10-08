
import os
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
import numpy as np

import copy
import gc
import matplotlib.pyplot as plt

import random
random.seed(0)

from torch.autograd import Variable
import itertools
import logging
import os.path
from PIL import Image
from torch.utils.data.sampler import Sampler
from models.byol import BYOL,BYOLP
import re
import argparse
import shutil
import time
import math
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from torch import nn, autograd

from configs import get_args
from augmentations import get_aug
from models import get_model
from models.backbones import *
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from tools.test import test_img,test_img2
from tools.loss import softmax_mse_loss, softmax_kl_loss, symmetric_mse_loss,softmax_kl_loss2
from tools.ramps import get_current_consistency_weight
from datasets.sampling import iid, noniid, DatasetSplit,iid2,noniid2
from tools.fed import FedAvg

from tools.prune_utils import *

from tools.prune_utils import updateBN, prune_globel_simple_weight,prune_network_sliming, recover_network, zero_out_gradient,prune_network
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary
def updateBN(model,args):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))



def cosVector(x, y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1 = 0.0;
    result2 = 0.0;
    result3 = 0.0;
    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i] ** 2  # sum(X*X)
        result3 += y[i] ** 2  # sum(Y*Y)
    # print(result1)
    # print(result2)
    # print(result3)
    print("result is " + str(result1 / ((result2 * result3) ** 0.5)))
    return result1 / ((result2 * result3) ** 0.5)
def main(device, args):
    log_dir=args.log_dir
    log_fn = args.log_fn
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")
    stderr=sys.stderr
    # sys.stderr=log_fp
    # define loss function
    loss1_func = nn.CrossEntropyLoss()
    loss2_func = softmax_kl_loss
    loss3_func=softmax_kl_loss2
    #define dataset
    dataset_kwargs = {
        'dataset':args.dataset,
        'data_dir': args.data_dir,
        'download':True,
        'debug_subset_size':args.batch_size if args.debug else None
    }
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataloader_unlabeled_kwargs = {
        'batch_size': args.batch_size,#*5,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataloader_unlabeled_kwargs2 = {
        'batch_size': 1,#*5,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataset_train =get_dataset(
        transform=get_aug(args.dataset, True),
        train=True,
        **dataset_kwargs
    )


    if args.iid == 'iid':
        dict_users_labeled, dict_users_unlabeled,dict_users_unlabeled_train,dict_users_unlabeled_test = iid2(dataset_train, args.num_users, args.label_rate,args.seed)
    else:
        dict_users_labeled, dict_users_unlabeled,dict_users_unlabeled_train,dict_users_unlabeled_test = noniid2(dataset_train, args.num_users, args.label_rate,args.seed)


    #initialize global model
    # model_glob = get_model('global', args.backbone, dataset=args.dataset).to(device)
    if (args.heat == 0):  ##Load pre_trained model
        if (args.backbone == "Vgg_backbone"):
            if (args.dataset == "cifar10"):
                model_glob = torch.load('Model_Cifar_vgg19.pkl', map_location='cpu')
                model_glob2 = get_model('global', 'cifar', dataset='cifar').to(device)
                model_glob.to(device)
            if(args.dataset=="svhn"):
                model_glob=torch.load('Model_Svhn_vgg19.pkl', map_location='cpu')
                model_glob2 = get_model('global', 'cifar', dataset='cifar').to(device)
                model_glob.to(device)
            model_glob_sum = sum(p.numel() for p in model_glob.parameters())
            model_glob2_sum = sum(p.numel() for p in model_glob2.parameters())
            #test VGG model
            test_loader = torch.utils.data.DataLoader(
                dataset=get_dataset(
                    transform=get_aug(args.dataset, False, train_classifier=False),
                    train=False,
                    **dataset_kwargs),
                shuffle=False,
                **dataloader_kwargs
            )
            model_glob.eval()
            acc, loss_train_test_labeled = test_img(model_glob, test_loader, args)
            print("Big model performance:",acc)
            print("parameters number of vgg:",model_glob_sum)
            print("parameters number of cnn:", model_glob2_sum)
        else:
                model_glob_heat = torch.load('Model_Mnist_vgg19.pkl', map_location='cpu')
        # model_glob_sum = sum(p.numel() for p in model_glob_heat.parameters())
        # model_glob_heat.to(device)
    else:
        if (args.backbone == "Vgg_backbone"):
            model_glob = get_model('global', args.backbone, dataset=args.dataset).to(device)
            model_glob2=get_model('global', "Mnist", dataset='cifar').to(device)
        if (args.backbone == "vgg"):
            model_glob = vgg().to(device)
            model_glob2 = get_model('global', 'cifar', dataset='cifar').to(device)
        if (args.backbone == "Mnist"):
            model_glob = get_model('global', args.backbone, dataset=args.dataset).to(device)


        model_glob_sum = sum(p.numel() for p in model_glob.parameters())
        model_glob2_sum = sum(p.numel() for p in model_glob2.parameters())

        print("parameters number of vgg:",model_glob_sum)
        print("parameters number of cnn:", model_glob2_sum)



        #Initialize indexs/accuracy/model list......
        model_local_idx = set()  # local model index
        model_local_dict_backbone = {}
        model_local_dict_fc = {}
        accuracy = []
        best_test_acc = float('-inf')
        best_train_acc = float('-inf')
        lr_scheduler = {}
        accuracy_log = []
        #To obtain a well-trained VGG Using labeled data
        print("-------------------Get Global Big Model---------------------")
        for iter in range(200):
            model_glob.train()
            optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01)
            # Load labeled data :batch size is setted here
            train_loader_labeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_labeled),  # load labeled data from dataset_train
                shuffle=True,
                **dataloader_kwargs
            )
            for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
                if torch.cuda.is_available():
                    labels = labels.to(device)
                    images1=images1.to(device)
                z1= model_glob(images1)
                loss = loss1_func(z1, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # test model
            if iter % 1 == 0:
                if iter>=2:
                    dataset_kwargs['download'] = False
                # Load test set
                test_loader = torch.utils.data.DataLoader(
                    dataset=get_dataset(
                        transform=get_aug(args.dataset, False, train_classifier=False),
                        train=False,
                        **dataset_kwargs),
                    shuffle=False,
                    **dataloader_kwargs
                )
                model_glob.eval()
                acc, loss_train_test_labeled = test_img(model_glob, test_loader, args)

                if acc>best_test_acc:
                    best_test_acc = acc
                accuracy.append(str(acc))
                if iter%1==0:
                    print('Round {:3d}, Best Test Acc {:.2f}%'.format(iter, acc))
                del test_loader
                gc.collect()
                torch.cuda.empty_cache()
            print("Warm up stage:accuracy:",best_test_acc,file=log_fp, flush=True)
        if (args.dataset == "cifar10"):
            print("save model as Model_Cifar_vgg19.pkl")
            torch.save(model_glob, 'Model_Cifar_vgg19.pkl')
        if (args.dataset == "svhn"):
            print("save model as Model_Svhn_vgg19.pkl")
            torch.save(model_glob,'Model_Svhn_vgg19.pkl')


        # preheat
    accuracy=[]
    best_test_acc=0
    print("model global 2:",model_glob2)
    print("--------------------warm up-------------------------",args.heat_epochs)
    for iter in range(args.heat_epochs):
        model_glob2.train()
        optimizer = torch.optim.SGD(model_glob2.parameters(), lr=0.01)
        # Load labeled data :batch size is setted here
        train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_labeled),  # load labeled data from dataset_train
            shuffle=True,
            **dataloader_kwargs
        )
        for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
            if torch.cuda.is_available():
                labels = labels.to(device)
                images1=images1.to(device)
            with torch.no_grad():
                z1 = model_glob(images1)
                z2 = softmax(z1, args.T)
            model_glob2.train()
            pred= model_glob2(images1)
            loss2 = loss2_func(softmax(pred, args.T), z2)
            # print("Distill Loss:",loss2)
            loss1 = loss1_func(pred, labels)
            # print("Classification Loss:",loss1)
            loss=loss1+loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test model
        if iter % 1 == 0:
            if iter>=2:
                dataset_kwargs['download'] = False
            # Load test set
            test_loader = torch.utils.data.DataLoader(
                dataset=get_dataset(
                    transform=get_aug(args.dataset, False, train_classifier=False),
                    train=False,
                    **dataset_kwargs),
                shuffle=False,
                **dataloader_kwargs
            )
            model_glob2.eval()
            acc, loss_train_test_labeled = test_img(model_glob2, test_loader, args)
            train_acc, loss_train = test_img2(model_glob2, train_loader_labeled, args)
            if acc>best_test_acc:
                best_test_acc = acc
            accuracy.append(str(acc))
            if iter%1==0:
                print('Round {:3d}, Best Test Acc {:.2f}%'.format(iter, acc))
                print('Round {:3d}, Best Train Acc {:.2f}%'.format(iter, train_acc))

            del test_loader
            gc.collect()
            torch.cuda.empty_cache()
        print("Warm up stage:accuracy:",best_test_acc,file=log_fp, flush=True)
        #Model-prunning at server-side
        # if (args.backbone == "Vgg_backbone"):
        #     print("begin server prunning")
        #     model_glob.backbone, cfg_mask = prune_network_sliming(model_glob.backbone, args.prunerate, args.backbone,
        #                                                            args.dataset, args.device)
        #     model_glob.teacher = nn.Sequential(model_glob.backbone, model_glob.fc)
        #     # fine-tune
        #     for iter in range(3):
        #         model_glob.train()
        #         optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01)
        #         # Load labeled data :batch size is setted here
        #         train_loader_labeled = torch.utils.data.DataLoader(
        #             dataset=DatasetSplit(dataset_train, dict_users_labeled),  # load labeled data from dataset_train
        #             shuffle=True,
        #             **dataloader_kwargs
        #         )
        #         for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
        #             if torch.cuda.is_available():
        #                 labels = labels.to(device)
        #                 images1 = images1.to(device)
        #             z1 = model_glob(images1)
        #             loss = loss1_func(z1, labels)
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()
        #     # evaluate the pre-trained model
        #     test_loader = torch.utils.data.DataLoader(
        #         dataset=get_dataset(
        #             transform=get_aug(args.dataset, False, train_classifier=False),
        #             train=False,
        #             **dataset_kwargs),
        #         shuffle=False,
        #         **dataloader_kwargs
        #     )
        #     model_glob.eval()
        #     acc, loss_train_test_labeled = test_img(model_glob, test_loader, args)
        # Save the model
    if(args.dataset=="svhn"):
        print("save model as Model_Svhn_vgg19.pkl")
        torch.save(model_glob, 'Model_Svhn_vgg19.pkl')
        torch.save(model_glob2, 'Model_Svhn_CNN.pkl')
    if(args.dataset=="cifar10"):
        print("save model as Model_Cifar_vgg19.pkl")
        print("save model as Model_Cifar_CNN.pkl")
        torch.save(model_glob, 'Model_Cifar_vgg19.pkl')
        torch.save(model_glob2, 'Model_Cifar_CNN.pkl')
    if(args.dataset=='mnist'):
        print("save model as Model_Mnist_vgg19.pkl")
        torch.save(model_glob,'Model_Mnist_vgg19.pkl')

    print("New model accuracy:",acc)
    model_glob_heat = model_glob2
    model_glob_sum = sum(p.numel() for p in model_glob_heat.parameters())
    print("model global sum:",model_glob_sum)






    model_glob_heat.to(device)
    model_list=[[] for idx in range(args.num_users)]
    accuracy_list=[[] for idx in range(args.num_users)]
    averge_accuracy_list=[]
    model_pruned_statisic=np.zeros(args.num_users, dtype=int)
    print("lengh of model list:",model_list)
    #test each clients' model
    for idx in range(args.num_users):
        test_loader_unlabeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_unlabeled_test[idx]),  # load unlabeled data for user i
            shuffle=True,
            **dataloader_unlabeled_kwargs
        )
        model_glob_heat.eval()
        with torch.no_grad():
            acc, loss_train_test_labeled = test_img2(model_glob_heat, test_loader_unlabeled, args)
        accuracy_list[idx] = acc.numpy()
    print("accuracy_list", accuracy_list)
    total_num_list = []
    total_num_list.append(model_glob_sum*10)
    #Training
    print("==============begin training======================")
    for iter in range(args.num_epochs):
        id_accuracy_list1 = []  # idxs before training
        id_accuracy_list2 = []  # ...  before distillationprint("training iter:",iter,"of",args.num_epochs)
        id_accuracy_list3 = []  # ... after distillationprint("training iter:", iter, "of", args.num_epochs,file=log_fp, flush=True)

        #preparation
        w_locals, loss_locals, loss0_locals, loss2_locals ,id_list= [], [], [], [],[]
        cfg_mask_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), 10, replace=False)
        #Begin training
        total_num=0
        # total_num_list=[]
        for idx in idxs_users:

            print("===Load model===")
            if model_list[idx]:

                model_local=model_list[idx]# Download personalized model
            else:

                model_local=copy.deepcopy(model_glob_heat)


            train_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled_train[idx]),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            test_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled_test[idx]),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            model_local.eval()
            with torch.no_grad():
                acc, loss_train_test_labeled = test_img2(model_local, test_loader_unlabeled, args)
            print("-------before training  accuracy of:",idx)
            print("Accuracy:",acc)
            id_accuracy_list1.append(acc)

            dict_unlabeled_highscore = highscoresampling(dataset_train, dict_users_unlabeled_train[idx], args.threhold,
                                                         model_local, device)
            print("unlabeled dataset", len(dict_users_unlabeled_train[idx]))
            print("fine_tuned dataset:", len(dict_unlabeled_highscore))
            if len(dict_unlabeled_highscore)==0:
                dict_unlabeled_highscore=dict_users_unlabeled_train[idx]
            model_label=copy.deepcopy(model_local)
            optimizer = torch.optim.SGD(model_local.parameters(), lr=0.01)
            model_local.train()  # Begin to train local model
            #Train local data
            print("{:3d} begin trainning".format(idx))
            ###save  local model
            for j in range(args.local_ep):
                for i, ((images1, images2), labels) in enumerate(train_loader_unlabeled):

                    images1=images1.to(device)
                    z1= model_local(images1.to(device, non_blocking=True))
                    with torch.no_grad():
                        label1=model_label(images1)#generate psudo label
                        label1_hard=label1.argmax(dim=1)

                    loss=loss1_func(z1, label1_hard)

                    optimizer.zero_grad()
                    loss.backward()
                    if args.sr:
                        updateBN(model_local,args)
                    optimizer.step()


            print("begin prune")
            model_local_sum = sum(p.numel() for p in model_local.parameters())
            print("local parameters before training:",model_local_sum)
            ###prune
            if model_pruned_statisic[idx]<args.prunetimes:
                if (args.backbone == "vgg"):
                    model_classifier = copy.deepcopy(model_local.classifier)
                    model_local, cfg_mask = prune_network_sliming(model_local, args.prunerate, args.backbone,
                                                                  args.dataset, args.device)
                    model_local.classifier = model_classifier
                else:
                    # backbone=args.backbone
                    backbone='CNN_Cifar_pruned'
                    model_local.backbone, cfg_mask = prune_network_sliming(model_local.backbone,args.prunerate,backbone,
                                                                           args.dataset,args.device)
                    model_local.teacher = nn.Sequential(model_local.backbone, model_local.fc)
                    model_local_sum=sum(p.numel() for p in model_local.teacher.parameters())
                    model_pruned_statisic[idx] += 1
                cfg_mask_locals.append(cfg_mask)
                print("End prune~~~times of been pruned",model_pruned_statisic[idx])#咋传的参数这么多
                print("local parameters after prunning:",model_local_sum)
            else:
                cfg_local_mask2 = []
                for k, m in enumerate(model_local.backbone.modules()):
                    if isinstance(m, nn.BatchNorm2d):
                        weight = m.weight.data.clone()
                        mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
                        cfg_local_mask2.append(mask.clone())
                cfg_mask_locals.append(cfg_local_mask2)
                model_pruned_statisic[idx] += 1
                print("Times of been trained",model_pruned_statisic[idx])
                print("Don not prune")

            print("begin finetune")
            optimizer = torch.optim.SGD(model_local.parameters(), lr=0.01)
            train_loader_unlabeled_finetune = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_unlabeled_highscore),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            model_local.train()
            for j in range(args.local_finetune):
                for i, ((images1, images2), labels) in enumerate(train_loader_unlabeled_finetune):
                    labels=labels.to(device)
                    images1 = images1.to(device)
                    z1 = model_local(images1)
                    with torch.no_grad():
                        label1 = model_label(images1)
                        label1_hard = label1.argmax(dim=1)
                    optimizer.zero_grad()
                    loss = loss1_func(z1, label1_hard)
                    loss.backward()
                    optimizer.step()
            ###Finetune
            ##test finetuned model
            dataset_kwargs['download'] = False


            model_local.eval()
            with torch.no_grad():
                acc, loss_train_test_labeled = test_img2(model_local, test_loader_unlabeled, args)
            id_accuracy_list2.append(acc)
            print("Accuracy after finetune",acc)

            total_num = total_num + sum(p.numel() for p in model_local.parameters())

            id_list.append(idx)
            model_list[idx]=copy.deepcopy(model_local)

            del model_local
            gc.collect()
            del model_label
            gc.collect()
            del train_loader_unlabeled
            gc.collect()
            torch.cuda.empty_cache()


        train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_labeled),  # load labeled data from dataset_train
            shuffle=True,
            **dataloader_kwargs
        )
        total_num_list.append(total_num)
        print("total_num_list",total_num_list,file=log_fp, flush=True)
        print("total_num_list", total_num_list)

        num_locals=10
        #generate model masks
        i=0
        cfg_mask_locals2=[]
        cfg_mask_locals2_sum=[]
        for idx in idxs_users:
            model=copy.deepcopy(model_list[idx])
            cfg_local_mask=cfg_mask_locals[i]
            i=i+1
            # backbone=args.backbone
            backbone = 'CNN_Cifar_pruned'
            newmodel=recover_network(model.backbone, cfg_local_mask, backbone, dataset=args.dataset, args=args)
            cfg_local_mask2 = []
            for k, m in enumerate(newmodel.modules()):
                if isinstance(m, nn.BatchNorm2d):
                    weight = m.weight.data.clone()
                    mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
                    mask_2=mask.cpu().numpy().tolist()
                    cfg_local_mask2.extend(mask_2)
            #         print("mask in this layer:",mask_2)
            # print("cfg in this model:",cfg_local_mask2)
            cfg_mask_locals2.append(cfg_local_mask2)
            cfg_mask_locals2_sum.append(sum(cfg_local_mask2))
            del model
            torch.cuda.empty_cache()
        print("====sum:=====")
        print(cfg_mask_locals2_sum)
        #generate weight matrix
        i=0
        list_2=[]
        for mask in cfg_mask_locals2:
            list_1=[]
            for j in range(len(cfg_mask_locals2)):
                mask_compare=cfg_mask_locals2[j]
                similarity=cosVector(mask,mask_compare)
                list_1.append(similarity)
                # print(list_1)
            list_2.append(list_1)
        print(list_2)

        #Multiteacher distillation
        print("Multiteacher distillation")
        for k in range(args.distill_round):
            i=0
            for idx in idxs_users:
                model=model_list[idx]
                for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
                    labels = labels.to(device)
                    images1 = images1.to(device)
                    sum_z1 = 0
                    j=0
                    teacher_logits=0

                    for idx2 in idxs_users:
                        if (idx2 != idx):
                            model_temp = model_list[idx2]
                            model_temp.eval()
                            with torch.no_grad():
                                z1 = model_temp(images1)
                                z2=softmax(z1, args.T)
                                weight_list=list_2[i]
                                weight=weight_list[j]/(sum(weight_list)-weight_list[i])
                                # print("i:",i,"j",j)
                                # print("weight:",weight)
                                teacher_logits+=weight*z2
                                z3=model_glob(images1)
                                z3=softmax(z3, args.T)
                        j = j + 1
                    model.train()
                    model.zero_grad()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    pred = model(images1)
                    loss1 = loss1_func(pred, labels)
                    # print("loss1",loss1)
                    loss2 = loss2_func(softmax(pred, args.T), teacher_logits)
                    loss3=loss2_func(softmax(pred, args.T), z3)
                    # print("loss_multi",loss2)
                    # print("loss1:",loss1)
                    # print("loss2",loss2)
                    # print("loss3",loss3)
                    loss = loss1 + args.lamda*loss2+args.lamda*loss3
                    loss.backward()
                    optimizer.step()
                    # temp += 1
                    del  pred
                    torch.cuda.empty_cache()
                i=i+1
                print("loss1:", loss1)
                print("loss2", loss2)
                print("loss3", loss3)
                del images1, images2, labels
                torch.cuda.empty_cache()
        print("end Multiteacher distillation")










        # print("Multiteacher distillation")
        # for k in range(args.distill_round):
        #     for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
        #         labels = labels.to(device)
        #         images1 = images1.to(device)
        #
        #         z1_list = []
        #         z2_list = []
        #         sum_z1 = 0
        #         sum_z2 = 0
        #         for id in id_list:
        #             model=model_list[id]
        #             model.eval()
        #             with torch.no_grad():
        #                 z1 = model(images1)
        #                 z1_list.append(softmax(z1, 2))
        #                 sum_z1 += softmax(z1)
        #
        #         temp = 0
        #         for id in id_list:
        #             model=model_list[id]
        #             with torch.no_grad():
        #                 teacher_logits_1 = (sum_z1 - z1_list[temp]) / (num_locals - 1)
        #
        #             model.train()
        #             optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        #             pred = model(images1)
        #             loss1 = loss1_func(pred, labels)  # 0.34
        #             # print("loss1",loss1)
        #             loss2 = loss2_func(softmax(pred, 2), teacher_logits_1)
        #             # print("loss_multi",loss2)
        #             loss = loss1 + loss2
        #
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()
        #             temp += 1
        #             del loss,loss1,loss2,pred
        #             torch.cuda.empty_cache()
        #     del images1,images2,labels
        #     torch.cuda.empty_cache()
        #
        #
        # print("length of model list:", len(model_list))
        #
        # # Multiteacher(w_locals,id_list,model_list,data_loader=train_loader_labeled,args=args,device=device)
        # print("end Multiteacher distillation")
        for idx in idxs_users:
            test_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled_test[idx]),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            model_local=model_list[idx]
            with torch.no_grad():
                acc, loss_train_test_labeled = test_img2(model_local, test_loader_unlabeled, args)
            accuracy_list[idx] = acc.numpy()
            id_accuracy_list3.append(acc)

        print("===============================================",file=log_fp,flush=True)
        print("round:",iter,file=log_fp,flush=True)
        print("round:",iter)
        print("Participants:",idxs_users,file=log_fp,flush=True)
        print("Accuracy before training:",file=log_fp,flush=True)
        print(id_accuracy_list1,file=log_fp,flush=True)
        print(id_accuracy_list1)
        print("Accuracy After finetune:", file=log_fp, flush=True)
        print(id_accuracy_list2, file=log_fp, flush=True)
        print(id_accuracy_list2)
        print("Accuracy After distillation:", file=log_fp, flush=True)
        print(id_accuracy_list3, file=log_fp, flush=True)
        print(id_accuracy_list3)
        print("=============================================",file=log_fp,flush=True)

        print("accuracy", accuracy_list)
        averge_accuracy = sum(accuracy_list) / len(accuracy_list)
        print("accuracy", accuracy_list)
        print("accuracy",accuracy_list,file=log_fp,flush=True)
        print("average accuracy", averge_accuracy)
        print("average accuracy", averge_accuracy, file=log_fp, flush=True)
        averge_accuracy_list.append(averge_accuracy)
        print("average accuracy list", averge_accuracy_list, file=log_fp, flush=True)
        print("-------------------------------------------",file=log_fp,flush=True)




        del train_loader_labeled
        gc.collect()
        torch.cuda.empty_cache()

    print("accuracy trace:",averge_accuracy_list)
    print("total_num_list", total_num_list)
    plt.xlabel("Training Round")
    plt.ylabel("Average Accuracy")
    plt.plot(averge_accuracy_list)
    plt.savefig('cifar10(non-iid).svg')
    plt.show()
    plt.xlabel("Training Round")
    plt.ylabel("numbers of parameters")
    plt.plot(total_num_list)
    plt.savefig('Traffic-Cifar10(noniid).svg')
    plt.show()
    log_fp.close()
    sys.stderr=stderr
def highscoresampling(dataset_train,dict_users_unlabeled,threhold,model,device):

    k = list(dict_users_unlabeled)
    dict_unlabeled_highscore = set()
    num_classes=10
    for image_id in k:
        ((images1, images2), labels) = dataset_train[image_id]

        images1 = images1.unsqueeze(0)
        images1 = images1.to(device)
        with torch.no_grad():

            psudo_label = model(images1)

            psudo_label_hard=psudo_label.argmax(dim=1)
            psudo_label_soft=softmax(psudo_label)
            confidence_score=psudo_label_soft.max()
            confidence_score_np=confidence_score.cpu().numpy()
            evidence = F.relu(psudo_label)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)


            uncertainty_np=uncertainty.cpu().numpy()

        if uncertainty_np<0.7:

            dict_unlabeled_highscore.add(image_id)

    return dict_unlabeled_highscore




def Multiteacher(w_locals,id_list,model_list,data_loader,args,device):

    loss1_func = nn.CrossEntropyLoss()
    loss3_func = softmax_kl_loss2
    loss2_func = softmax_kl_loss
    num_locals=len(w_locals)
    for k in range(args.distill_round):
        for batch_idx, ((images1, images2), labels) in enumerate(data_loader):
            labels = labels.to(device)
            images1 = images1.to(device)
            #generate logits
            z1_list = []
            z2_list = []
            sum_z1=0
            sum_z2=0
            for model in w_locals:
                model.eval()
                with torch.no_grad():
                    z1=model(images1)
                    z1_list.append(softmax(z1,2))
                    sum_z1+=softmax(z1)

            temp=0
            for model in w_locals:
                with torch.no_grad():
                    teacher_logits_1=(sum_z1-z1_list[temp])/(num_locals-1)

                model.train()

                optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
                pred = model(images1)
                loss1=loss1_func(pred, labels) #0.34

                loss2=loss2_func(softmax(pred,2), teacher_logits_1)##不知道匹不匹配 0.0 10.33

                loss=loss1+loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp+=1
    ##fine-tune

    temp=0
    for model in w_locals:

        id=id_list[temp]
        c=model_list[id]
        del c
        gc.collect()

        model_list[id]=model
        temp+=1
    torch.cuda.empty_cache()
    print("length of model list:",len(model_list))

def softmax(X,T=1):
    X=X/T
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)