import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import copy
import gc
# from tqdm import tqdm
import random
# from sklearn import metrics
from torch.autograd import Variable
import itertools
import logging
import os.path
from PIL import Image
from torch.utils.data.sampler import Sampler
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

def iid2(dataset, num_users, label_rate,seed):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled, dict_users_unlabeled, dict_users_unlabeled_test, dict_users_unlabeled_train = set(), {}, {}, {}
    np.random.seed(seed+1)
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))

    for i in range(num_users):
        np.random.seed(seed+2)
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items), replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i])
        # dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
        unlabeled_semi = dict_users_unlabeled[i] - dict_users_labeled
        dict_users_unlabeled[i]=dict_users_unlabeled[i] - dict_users_labeled
        list_temp = list(unlabeled_semi)  # Local unlabeled data without server data
        frac = 0.2
        random.seed(seed+3)
        ran_li = random.sample(list_temp, int(len(list_temp) * 0.2))
        dict_users_unlabeled_test[i] = set(ran_li)  # test set(without server data)

        dict_users_unlabeled_train[i] = dict_users_unlabeled[i] - dict_users_unlabeled_test[i]  # local train data(with server data)
    return dict_users_labeled, dict_users_unlabeled, dict_users_unlabeled_train, dict_users_unlabeled_test

def iid(dataset, num_users, label_rate):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled, dict_users_unlabeled = set(), {}
    dict_users_unlabeled_train={}
    dict_users_unlabeled_test={}
    
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
        
    for i in range(num_users):
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items) , replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


    return dict_users_labeled, dict_users_unlabeled


def noniid(dataset, num_users, label_rate):

    num_shards, num_imgs = 2 * num_users, int(len(dataset)/num_users/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]  #label
        
    num_items = int(len(dataset)/num_users)
    dict_users_labeled = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]#索引值
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled



    return dict_users_labeled, dict_users_unlabeled

def noniid2(dataset, num_users, label_rate,seed):
    num_shards, num_imgs = 2 * num_users, int(len(dataset) / num_users / 2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_unlabeled_test, dict_users_unlabeled_train = {}, {}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]  # label

    num_items = int(len(dataset) / num_users)
    dict_users_labeled = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 索引值
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        np.random.seed(seed+1)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate(
                (dict_users_unlabeled[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    np.random.seed(seed+1)
    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))

    for i in range(num_users):
        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        # dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
        unlabeled_semi = dict_users_unlabeled[i] - dict_users_labeled
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
        list_temp = list(unlabeled_semi)
        frac = 0.2
        np.random.seed(seed+1)
        ran_li = np.random.choice(list_temp, int(len(list_temp) * 0.2))

        dict_users_unlabeled_test[i] = set(ran_li)
        dict_users_unlabeled_train[i] = dict_users_unlabeled[i] - dict_users_unlabeled_test[i]
    return dict_users_labeled, dict_users_unlabeled, dict_users_unlabeled_train, dict_users_unlabeled_test

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (images1, images2), labels = self.dataset[self.idxs[item]]
        return (images1, images2), labels

