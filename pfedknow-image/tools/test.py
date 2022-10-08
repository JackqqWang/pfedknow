#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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

def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    
    for idx, (data, target) in enumerate(data_loader):
        # print("")
        if torch.cuda.is_available():
            data, target = data.to(args.device), target.to(args.device)
        log_probs= net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss


def test_img2(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0

    for idx, ((images1, images2),target) in enumerate(data_loader):
        # print("")
        images1=images1.to(args.device)
        target=target.to(args.device)
        log_probs = net_g(images1)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum', ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    return accuracy, test_loss