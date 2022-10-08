import copy
import torch

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_nonzero(w):
    w_avgs = []
    for index in range(len(w)):
        # print(w_local)
        w_avg = copy.deepcopy(w[index])
        remain_w =  [x for i,x in enumerate(w) if i!=index]
        for k in w_avg.keys():
            for i in range(len(remain_w)):
                w_avg[k] += remain_w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        w_avgs.append(w_avg)
    return w_avgs