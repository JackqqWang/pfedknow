import torch.nn as nn
import torch.nn.utils.prune as prune
import torch
import numpy as np
from models import get_model
from models import get_pruned_backbone
from models.backbones import vgg
from models.backbones import *

# def print_grad(model):
#     for m in model.backbone.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             print(m)
#             # print(m.weight)
#             print(m.weight.grad)
def updateBN(model,s):
    for m in model.backbone.modules(): ###TODO check this logic
        # print(m)
        if isinstance(m, nn.BatchNorm2d):
            # print(m)
            # print(m.weight.is_leaf)
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1


def zero_out_gradient(model, cfg_mask, backbone):


    layer_id_in_cfg = 0
    if 'mnist' in backbone.lower():
        start_mask = torch.ones(1)
    else:
        start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    layer_index = -1

    for m0 in model.modules():
        layer_index += 1
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # remaining channel index
            # idx_is_0 = np.squeeze(np.where(np.asarray(end_mask.cpu().numpy())==0)) # pruned channel index

            back_up_grad = m0.weight.grad[idx1]
            m0.weight.grad.fill_(0)
            # m.bias.grad.fill_(0)
            # m1.running_var.fill_(0)
            # m1.running_mean.fill_(0)

            m0.weight.grad[idx1] = back_up_grad

            # m0.bias.grad[idx_is_0] = 0
            # m0.running_mean[idx_is_0] = 0
            # m0.running_var[idx_is_0] = 0

            # ****************************************************************************************************************#
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        # *****************************conv没看懂应该怎么改*********************************************#
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.argwhere(np.asarray(start_mask.cpu().numpy())).flatten()
            idx0_is_0 =[i for i in range(len(start_mask)) if i not in idx0]
            idx1 = np.argwhere(np.asarray(end_mask.cpu().numpy())).flatten()
            idx1_is_0 = np.squeeze(np.where(np.asarray(end_mask.cpu().numpy())==0)) # pruned channel index

            # try:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            # except:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0, idx1.shape[0]))

            # w = m1.weight.data[:, idx0, :, :].clone()
            # if len(w.shape)==3:
            #     w = w.unsqueeze(1)
            # w = w[idx1, :, :, :].clone()
            # m1.weight.data = w.clone()
            ## set zero
            back_up_grad = m0.weight.grad[idx1[:, None], idx0[None, :], :, :]
            m0.weight.grad.fill_(0)
            # m0.weight.data.fill_(0)
            ## set value
            m0.weight.grad[idx1[:, None], idx0[None, :], :, :] = back_up_grad
            # m1.weight.data[idx1,:,:,:][:,idx0,:,:] = m0.weight.data.clone() this does not work, see previous line. i don't why though.
            # m1.bias.data = m0.bias.data[idx1].clone()
        elif isinstance(m0, nn.Linear):
            pass
            # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # m1.weight.data.fill_(0)
            # m1.weight.data[:,idx0] = m0.weight.data.clone()
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

    return model


def pseudo_prune_network_sliming(model, percent, backbone, dataset):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []  # [4, 'M', 4, 'M', 25] something like that, 数字是mask中1的和，也就是多少个保留下来了
    cfg_mask = []  # 里面是sublist, 每个sublist是1011这种
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()  # weightcopy is a tensor
            # print("weight_copy is:")

            if torch.cuda.is_available():
                mask = weight_copy.abs().gt(thre).float().cuda()
            else:
                mask = weight_copy.abs().gt(thre).float()
            # mask is a tensor, mask.shape[0] is original total channel number; torch.sum(mask) 是留下的channel个数
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            #       format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned / total

    print('Pre-processing Successful!')

    # Make real prune
    print(cfg)

    # newmodel =  vgg(cfg=cfg) #get_pruned_backbone('Vgg_backbone',cfg)
    # elif name == 'global':
    if cfg != None:
        newmodel = get_pruned_backbone(backbone, cfg, dataset=dataset)  # 函数加载个backbone， potential bug here... no dataset
    else:
        raise NotImplementedError
        # model = get_backbone(backbone)
    # newmodel = vgg(cfg=cfg)
    if torch.cuda.is_available():
        newmodel.cuda()

    # cfg = [cfg[0], 64, cfg[2], 128, cfg[4]] ##
    # cfg_mask = [cfg_mask[0],torch.ones(64),cfg_mask[1],torch.ones(128),cfg_mask[2]]
    layer_id_in_cfg = 0
    if 'mnist' in backbone.lower():
        start_mask = torch.ones(1)  #####TODO 3 or 1
    else:
        start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    layer_index = -1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        layer_index += 1
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone() ## m0.running_mean.mul_(end_mask)
            m1.running_var = m0.running_var[idx1].clone()  ## m0.running_var.mul_(end_mask)
            m0.weight.data.mul_(end_mask) ## TODO whether this change the value of m0.weight
            m0.bias.data.mul_(end_mask)
            m0.running_mean.data.mul_(end_mask)
            m0.running_var.data.mul_(end_mask)
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # try:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            # except:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0, idx1.shape[0]))
            w = m0.weight.data[:, idx0, :, :].clone()
            m0.weight.data = torch.mul(m0.weight.permute(0,2,3,1),start_mask.cuda()).permute(0,3,1,2)
            if len(w.shape) == 3:
                w = w.unsqueeze(1)
            w = w[idx1, :, :, :].clone()
            m0.weight.data = torch.mul(m0.weight.permute(1,2,3,0),end_mask).permute(3,0,1,2)
            m1.weight.data = w.clone()
            # m1.bias.data = m0.bias.data[idx1].clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m0.weight.data.mul_(start_mask.cuda())
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

    # return newmodel
    return model, cfg_mask

def recover_network(model, cfg_mask, backbone,dataset,args):
    if 'mnist' in backbone.lower():
        cfg = [32, 'M', 48, 'M', 64]
    elif backbone=='CNN_Cifar_pruned':
        cfg=[32, 'M', 128, 'M', 256]
    else:
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]  ## complete model cfg
    if cfg!=None:
        newmodel = get_pruned_backbone(backbone,cfg,dataset=dataset) # 函数加载个backbone 原始完整model
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        newmodel.to(args.device)
        model.to(args.device)

    layer_id_in_cfg = 0
    if 'mnist' in backbone.lower():
        start_mask = torch.ones(1)
    else:
        start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    layer_index = -1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        layer_index+=1
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # remaining channel index
            # idx_is_0 = np.squeeze(np.where(np.asarray(end_mask.cpu().numpy())==0)) # pruned channel index

            if args.ran_initial:
                m1.weight.data.fill_(np.random.rand())
                m1.bias.data.fill_(np.random.rand())
                m1.running_var.fill_(np.random.rand())
                m1.running_mean.fill_(np.random.rand())                    
            else:
                m1.weight.data.fill_(0)
                m1.bias.data.fill_(0)
                m1.running_var.fill_(0)
                m1.running_mean.fill_(0)

            m1.weight.data[idx1] = m0.weight.data.clone()
            m1.bias.data[idx1] = m0.bias.data.clone()
            m1.running_mean[idx1] = m0.running_mean.clone()
            m1.running_var[idx1] = m0.running_var.clone()

            #****************************************************************************************************************#
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        #*****************************conv没看懂应该怎么改*********************************************#
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.argwhere(np.asarray(start_mask.cpu().numpy())).flatten()
            # idx0_is_0 =[i for i in range(len(start_mask)) if i not in idx0]
            idx1 = np.argwhere(np.asarray(end_mask.cpu().numpy())).flatten()
            # idx1_is_0 = np.squeeze(np.where(np.asarray(end_mask.cpu().numpy())==0)) # pruned channel index
            
            # try:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            # except:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0, idx1.shape[0]))

            # w = m1.weight.data[:, idx0, :, :].clone()
            # if len(w.shape)==3:
            #     w = w.unsqueeze(1)
            # w = w[idx1, :, :, :].clone()
            # m1.weight.data = w.clone()
            ## set zero
            m1.weight.data.fill_(0)
            ## set value
            m1.weight.data[idx1[:,None],idx0[None,:],:,:] = m0.weight.data.clone()
            # m1.weight.data[idx1,:,:,:][:,idx0,:,:] = m0.weight.data.clone() this does not work, see previous line. i don't why though.
            # m1.bias.data = m0.bias.data[idx1].clone()
        elif isinstance(m0, nn.Linear):
            pass
            # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # m1.weight.data.fill_(0)
            # m1.weight.data[:,idx0] = m0.weight.data.clone()
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

    return newmodel


def prune_network_sliming(model,percent,backbone,dataset,device):
    total = 0
    total_num =sum(p.numel() for p in model.parameters())
    print("Before pruning:",total_num)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    # print("total:",total)
    print("thre_index",thre_index)
    thre = y[thre_index]
    print("percent:",percent)
    pruned = 0
    cfg = [] #[4, 'M', 4, 'M', 25] something like that, 数字是mask中1的和，也就是多少个保留下来了
    cfg_mask = [] # 里面是sublist, 每个sublist是1011这种
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone() #weightcopy is a tensor
            # print("weight_copy is:")
            
            if torch.cuda.is_available():####
                mask = weight_copy.abs().gt(thre).float().to(device)

            else:
                mask = weight_copy.abs().gt(thre).float()
            #mask is a tensor, mask.shape[0] is original total channel number; torch.sum(mask) 是留下的channel个数
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            #       format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned / total
    print("prunerd_channel_ratio:",pruned_ratio)

    # print('Pre-processing Successful!')

    # Make real prune
    print(cfg)

    # newmodel =  vgg(cfg=cfg) #get_pruned_backbone('Vgg_backbone',cfg)
    # elif name == 'global':
    if cfg!=None:
    #*********************这部分newmodel应该初始化成完整的model，这样205-208行才不会报错********************#
        if(backbone=="vgg"):
            newmodel=vgg(cfg)
        else:
            newmodel = get_pruned_backbone(backbone, cfg=cfg, dataset=dataset)  # 函数加载个backbone
        # print("newmodel",newmodel)
        # print("oldmodel",model)
        # print(ac)
        # if(backbone=="mnist"):
        #     newmodel=CNN_BN_Mnist(cfg)
        # newmodel = vgg(cfg)
        # newmodel = model.cuda()   #这里不使用剪枝之后的
    else:
        raise NotImplementedError
        # model = get_backbone(backbone)
    # newmodel = vgg(cfg=cfg)
    if torch.cuda.is_available():
        # newmodel.cuda()
        newmodel.to(device)


    layer_id_in_cfg = 0
    if 'mnist' in backbone.lower():
        start_mask = torch.ones(1)
    else:
        start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    layer_index = -1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        layer_index+=1
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # remaining channel index 
            # idx_is_0 = np.squeeze(np.where(np.asarray(end_mask.cpu().numpy())==0)) # pruned channel index

            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            #
            # #******************这部分会报错，感觉是elif没有改的原因, 初始化就是从原来model读取的？************************#
            # m1.weight.data[idx_is_0] = m0.weight.data[idx_is_0].clone()
            # m1.bias.data[idx_is_0] = m0.bias.data[idx_is_0].clone()
            # m1.running_mean[idx_is_0] = m0.running_mean[idx_is_0].clone()
            # m1.running_var[idx_is_0] = m0.running_var[idx_is_0].clone()

            #****************************************************************************************************************#
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        #*****************************conv没看懂应该怎么改*********************************************#
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # try:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            # except:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0, idx1.shape[0]))
            w = m0.weight.data[:, idx0, :, :].clone()
            if len(w.shape)==3:
                w = w.unsqueeze(1)
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
            # m1.bias.data = m0.bias.data[idx1].clone()
        elif isinstance(m0, nn.Linear):
            pass
            # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

    total_num1 = sum(p.numel() for p in newmodel.parameters())
    pruned_parameters_rate=(total_num-total_num1)/total_num
    print("After pruning parameters of backbone:", total_num1)
    print("Reduced rate of backbone",pruned_parameters_rate)
    # print("cfg_mask:",cfg_mask)
    return newmodel, cfg_mask

def prune_globel_simple_weight(model):
    ##TODO prune global model
    ##Input is a model, output is a pruned model
    parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')

    return model

def prune_network(model_weights, cfg_mask, backbone, dataset,args):
    if 'mnist' in backbone.lower():
        cfg = [32, 'M', 48, 'M', 64]
    else:
        # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512,
        #        512]  ## complete model cfg
        cfg = [32, 'M', 128, 'M', 256]

    if cfg != None:
        model = get_pruned_backbone(backbone, cfg, dataset=dataset)  # 函数加载个backbone 原始完整model
    else:
        raise NotImplementedError
    model.load_state_dict(model_weights)
    model.to(args.device)
    index = 0
    cfg = []  # [4, 'M', 4, 'M', 25] something like that, 数字是mask中1的和，也就是多少个保留下来了

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            mask = cfg_mask[index]
            # mask.cpu()
            cfg.append(int(torch.sum(mask)))
            index += 1
            weight_copy = m.weight.data.clone()  # weightcopy is a tensor
            # pruned = pruned + mask.shape[0] - torch.sum(mask)
            # print(m.weight.data)
            # print(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
        # Make real prune
    print(cfg)

    # newmodel =  vgg(cfg=cfg) #get_pruned_backbone('Vgg_backbone',cfg)
    # elif name == 'global':
    if cfg != None:
        # *********************这部分newmodel应该初始化成完整的model，这样205-208行才不会报错********************#
        newmodel = get_pruned_backbone(backbone, cfg, dataset=dataset)  # 函数加载个backbone
        # newmodel = model.cuda()   #这里不使用剪枝之后的
    else:
        raise NotImplementedError
        # model = get_backbone(backbone)
    # newmodel = vgg(cfg=cfg)
    if torch.cuda.is_available():
        newmodel.to(args.device)

    layer_id_in_cfg = 0
    if 'mnist' in backbone.lower():
        start_mask = torch.ones(1)
    else:
        start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    layer_index = -1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        layer_index += 1
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # remaining channel index
            # idx_is_0 = np.squeeze(np.where(np.asarray(end_mask.cpu().numpy())==0)) # pruned channel index

            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            #
            # #******************这部分会报错，感觉是elif没有改的原因, 初始化就是从原来model读取的？************************#
            # m1.weight.data[idx_is_0] = m0.weight.data[idx_is_0].clone()
            # m1.bias.data[idx_is_0] = m0.bias.data[idx_is_0].clone()
            # m1.running_mean[idx_is_0] = m0.running_mean[idx_is_0].clone()
            # m1.running_var[idx_is_0] = m0.running_var[idx_is_0].clone()

            # ****************************************************************************************************************#
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        # *****************************conv没看懂应该怎么改*********************************************#
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # try:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            # except:
            #     print('In shape: {:d} Out shape:{:d}'.format(idx0, idx1.shape[0]))
            w = m0.weight.data[:, idx0, :, :].clone()
            if len(w.shape) == 3:
                w = w.unsqueeze(1)
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
            # m1.bias.data = m0.bias.data[idx1].clone()
        elif isinstance(m0, nn.Linear):
            pass
            # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

    return newmodel