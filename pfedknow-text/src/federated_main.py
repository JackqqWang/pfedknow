import os
import copy
import pickle
import time
import gc
import numpy as np
from tqdm import tqdm
from transformers import AdamW, BertConfig
from transformers import AutoModel,BertForSequenceClassification,AutoModelForSequenceClassification

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import (
#     BertForSequenceClassification,
#     BertConfig,
# )
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformer.modeling_pruning import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
import torch
from torch.utils.data import DataLoader, Dataset
import options
from loss import *
from tensorboardX import SummaryWriter


from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNCifar
from utils import get_dataset, average_weights, exp_details

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        batch = self.dataset[self.idxs[item]]
        return batch

def softmax(X,T=1):
    X=X/T
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition
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
    # print("result is " + str(result1 / ((result2 * result3) ** 0.5)))
    return result1 / ((result2 * result3) ** 0.5)
def highscoresampling(dataset_train,dict_users_unlabeled,threhold,model,device):

    k = list(dict_users_unlabeled)
    dict_unlabeled_highscore = set()
    num_classes=10
    for sequence_id in k:
        (b_input_ids, b_input_mask,b_labels) = dataset_train[sequence_id]
        b_labels = b_labels - 1
        b_input_ids=b_input_ids.to(device)
        b_input_mask=b_input_mask.to(device)
        b_labels=b_labels.to(device)


        b_input_ids=b_input_ids.unsqueeze(0)
        b_input_mask=b_input_mask.unsqueeze(0)


        with torch.no_grad():

            output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
            loss, logits = output.loss, output.logits


            psudo_label_soft=softmax(logits)
            confidence_score=psudo_label_soft.max()


        if confidence_score>0.7:

            dict_unlabeled_highscore.add(sequence_id)

    return dict_unlabeled_highscore
class Prune():
    def __init__(
            self,
            model,
            pretrain_step=0,
            sparse_step=0,
            frequency=100,
            prune_dict={},
            restore_sparsity=False,
            fix_sparsity=False,
            balance='none',
            prune_device='default'):
        self._model = model
        self._t = 0
        self._initial_sparsity = {}  # 0
        self._pretrain_step = pretrain_step  # 1 epoch; but it is step
        self._sparse_step = sparse_step  # 4 vs 16 epochs
        self._frequency = frequency  # 20, 100 for sparse_step=4; when sparse_step=16, use the good frequency value (MRPC: 3.5k samples)
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._balance = balance
        self._prune_device = prune_device
        self._mask = {}

        self._prepare()

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():



                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
                        self._initial_sparsity[name] = 1 - mask.sum().numpy().tolist() / weight.view(-1).shape[0]
                        self._mask[name] = mask
                    else:
                        self._initial_sparsity[name] = 0
                        self._mask[name] = torch.ones_like(weight)


    def _update_mask(self, name, weight, keep_k):
        if keep_k >= 1:
            thrs = torch.topk(weight.abs().view(-1), keep_k)[0][-1]
            mask = torch.where(weight.abs() >= thrs, torch.ones_like(weight), torch.zeros_like(weight))
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = self._pretrain_step < self._t < self._pretrain_step + self._sparse_step
        condition3 = (self._t - self._pretrain_step) % self._frequency == 0
        return condition1 and condition2 and condition3

    def _get_weight(self, parameter):
        if self._prune_device == 'default':
            weight = parameter.data
        elif self._prune_device == 'cpu':
            weight = parameter.data.to(device=torch.device('cpu'))
        return weight

    def prune(self,prune_times):
        with torch.no_grad():
            self._t = self._t + 1
            for name, parameter in self._model.named_parameters():
                # print("name:",name)
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    p=self._update_mask_conditions()
                    # print("p",p)
                    if self._update_mask_conditions() or 1:

                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (self._t - self._pretrain_step) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = target_sparsity + (self._initial_sparsity[name] - target_sparsity) * (
                                    1.0 - current_sparse_step / total_srarse_step) ** 3
                        # keep_k = int(weight.view(-1).shape[0] * (1.0 - current_sparsity))
                        if(prune_times==1):
                            keep_k = int(weight.view(-1).shape[0] * 0.8)
                        if(prune_times==2):
                            keep_k=int(weight.view(-1).shape[0] * 0.6)
                        if(prune_times==3):
                            keep_k = int(weight.view(-1).shape[0] * 0.4)
                        if (prune_times>3):
                            keep_k=int(weight.view(-1).shape[0] * 0.2)

                        if self._balance == 'none':

                            self._update_mask(name, weight, keep_k)

                    parameter.mul_(self._mask[name])
    def sparsity(self):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}
        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().numpy()
                total_param = total_param + temp.size

                total_nonezero = total_nonezero + np.flatnonzero(temp).size

                layer_sparse_rate[name] = 1 - np.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate



def init_weights(m: torch.nn.Module, filter_list = []) -> None:
    if isinstance(m, tuple(filter_list)):
        return
    if hasattr(m, 'reset_parameters'):
        return m.reset_parameters()
    else:
        for layer in m.children():
            init_weights(layer, filter_list=filter_list)

if __name__ == '__main__':
    start_time = time.time()
    #define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    parser = options.args_parser()
    options.training_args(parser)
    options.fp16_args(parser)
    options.pruning_args(parser)
    options.eval_args(parser)
    options.analysis_args(parser)
    args = parser.parse_args()

    exp_details(args)
    log_dir = args.log_dir
    log_fn = args.log_fn
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")

    if torch.cuda.is_available():  
        dev = args.gpuid
    else:  
        dev = "cpu"  
    device = torch.device(dev)  


    train_dataset, test_dataset, dict_users_labeled, dict_users_unlabeled = get_dataset(args) #TODO get a new dataset
    if args.model == 'BERT':
        if args.dataset == 'yahoo':
            class_num = 10
        elif args.dataset == 'ag':
            class_num = 4
        global_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = class_num, # The number of output labelsfor classification.   
    # output_attentions = False, # Whether the model returns attentions weights.
    # output_hidden_states = False, # Whether the model returns all hidden-states.
                                                                   )
    elif args.model == 'TINYBERT':
        if args.dataset == 'yahoo':
            class_num = 10
        elif args.dataset == 'ag':
            class_num = 4
        global_model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",num_labels=class_num)
        # global_model = TinyBertForSequenceClassification.from_scratch(num_labels=class_num)
        init_weights(global_model)

    else:
        exit('Error: unrecognized model')


    print("MODEL INITIALIZED")
    # set model to train and send it to device
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    # training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    #obtain a "Good" teacher model
    if args.model_G == 'BERT':
        if args.dataset == 'yahoo':
            class_num = 10
        elif args.dataset == 'ag':
            class_num = 4
        global_model2 = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = class_num, # The number of output labelsfor classification.
    # output_attentions = False, # Whether the model returns attentions weights.
    # output_hidden_states = False, # Whether the model returns all hidden-states.
                                                                   )
    global_model2.to(device)
    for iter in range(5):  # -->add argument afterwards
        global_server = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users_labeled, logger=logger)
        global_model2.train()
        w, loss = global_server.update_weights(model=global_model2, global_round=iter, epoch=1)
        with torch.no_grad():
            acc, loss = global_server.inference(model=global_model2)
        print("iter:", iter, "test accuracy", acc)
    print("End Big model stage,accuracy", acc)

    # warm-up stage(no distillation)
    # for iter in range(2):#-->add argument afterwards
    #     global_server=LocalUpdate(args = args, dataset = train_dataset, idxs = dict_users_labeled, logger = logger)
    #     global_model.train()
    #     w,loss=global_server.update_weights(model=global_model,global_round = iter,epoch=1)
    #     with torch.no_grad():
    #         acc, loss = global_server.inference(model=global_model)
    #     print("iter:",iter,"test accuracy",acc)
    # print("End warm-up stage,accuracy",acc)
    # warm-up stage(have distillation)
    optimizer = AdamW(global_model.parameters(),
                      lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    for iter in range(100):  # -->add argument afterwards
        global_server = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users_labeled, logger=logger)
        trainloader = DataLoader(DatasetSplit(train_dataset, dict_users_labeled), batch_size=32, shuffle=True)
        for step, batch in enumerate(trainloader):
            j = 0
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_labels = b_labels.cpu()
            b_labels = b_labels - 1
            b_labels = b_labels.to(device)
            with torch.no_grad():
                output_temp2 = global_model2(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             labels=b_labels)
                _, teacher_logits_2 = output_temp2.loss, output_temp2.logits
            global_model.train()
            global_model.zero_grad()
            output = global_model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss1, logits = output.loss, output.logits
            loss2=softmax_kl_loss(softmax(logits), teacher_logits_2)
            loss=loss1+loss2
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc, loss = global_server.inference(model=global_model)
        print("iter:", iter, "test accuracy", acc)
    print("End warm-up stage,accuracy", acc)

    model_list = [[] for idx in range(args.num_users)]
    accuracy_list = [[] for idx in range(args.num_users)]
    averge_accuracy_list = []
    traffic_history=[]
    num_1=sum(p.numel() for p in global_model.parameters())
    traffic_history.append(num_1*5)
    model_pruned_statisic = np.zeros(args.num_users, dtype=int)

    for idx in range(args.num_users):#--->add argument
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users_unlabeled[idx], logger=logger)
        with torch.no_grad():
            acc,loss=local_model.inference(model=global_model)
        accuracy_list[idx]=acc
        model_list[idx]=copy.deepcopy(global_model).cpu()


    print("accuracy_list", accuracy_list)
    averge_accuracy = sum(accuracy_list) / len(accuracy_list)
    averge_accuracy_list.append(averge_accuracy)
    print("Average initial accuracy:",averge_accuracy)


    for epoch in tqdm(range(args.epochs)):
        id_accuracy_list1 = []  # idxs before training
        id_accuracy_list2 = []  # ...  before distillationprint("training iter:",iter,"of",args.num_epochs)
        id_accuracy_list3 = []
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} | \n')
        print(f'\n | Global Training Round : {epoch + 1} | \n',file=log_fp,flush=True)
        # global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m , replace = False) #True 可以反复取
        total_num=0
        for idx in idxs_users:
            print("User_{}:".format(idx))
            print("===Load model===")
            if model_list[idx]:
                model_lc = model_list[idx].to(device)  # Download personalized model
            else:
                model_lc = copy.deepcopy(global_model)


            model_lb=copy.deepcopy(model_lc)
            #high-score sampling
            dict_unlabeled_highscore = highscoresampling(train_dataset, dict_users_unlabeled[idx], 0.7,
                                                         model_lb, device)
            print("all training set:",len(dict_users_unlabeled[idx]))
            print("High score set:",len(dict_unlabeled_highscore))
            # print(dict_unlabeled_highscore)



            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs = dict_users_unlabeled[idx], logger = logger)
            with torch.no_grad():
                acc, loss = local_model.inference(model = model_lc)
            print("accuracy before training： ", acc)
            id_accuracy_list1.append(acc)
            w, loss = local_model.update_weights2(model = model_lc, global_round = epoch,label_model=model_lb)
            #accuracy before prunning
            with torch.no_grad():
                acc, loss = local_model.inference(model = model_lc)
            print("accuracy before prunning： ",acc)


            num_steps_per_epoch =30
            prune_dict = {}
            for k, v in model_lc.named_parameters():

                if ('intermediate.dense.weight' in k or 'output.dense.weight' in k) and (
                        'attention.output.dense.weight' not in k):
                    prune_dict[k] = 0.95  # 0.95
                # Att nn
                if 'attention.self.query.weight' in k or 'attention.self.key.weight' in k or 'attention.self.value.weight' in k or 'attention.output.dense.weight' in k:
                    prune_dict[k] = 0.95  # 0.95

            num_steps_per_epoch = int(
                len(dict_users_unlabeled[idx]) / 32)
            prune = Prune(model_lc, 0, num_steps_per_epoch * len(dict_users_unlabeled[idx]),
                          round(num_steps_per_epoch * 400 / 100), prune_dict, False, False,'none')  # "round(num_steps_per_epoch * args.num_train_epochs / 100)" indicates we execute 100 model prunings, which works well in general but "100" can be tuned
            model_pruned_statisic[idx]+=1
            prune.prune(prune_times=model_pruned_statisic[idx])
            cfg_mask_locals2=[]
            cfg_local_mask2=[]

            # with torch.no_grad():
            #     for name, parameter in model_lc.named_parameters():
            #         if any(name == one for one in prune_dict):
            #             weight = parameter.data.clone()
            #             mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
            #             mask2=mask.view(-1)
            #             print("mask:",mask2)
            #             mask_2 = mask2.cpu().numpy().tolist()
            #             cfg_local_mask2.extend(mask_2)
            #     cfg_sum=sum(cfg_local_mask2)
            #     print("cfg_sum:",cfg_sum)
            #
            # # print("====cfg_local_mask2===:",cfg_local_mask2)
            # cfg_mask_locals2_sum.append(sum(cfg_local_mask2))
            with torch.no_grad():
                acc, loss = local_model.inference(model=model_lc)
                print("accuracy after prune：", acc)

            layer_sparse_rate, total_sparse_rate = prune.sparsity()
            print("total_sparse_rate:", total_sparse_rate)
            # print("w sprase:",model_lc.state_dict())
            #fine_tune using high-quality label
            w, loss = local_model.update_weights2(model=model_lc, global_round=epoch,label_model=model_lb)
            # prune again with same sparse rate
            prune.prune(prune_times=model_pruned_statisic[idx])
            with torch.no_grad():
                acc, loss = local_model.inference(model=model_lc)
                print("accuracy after finetune：", acc)
            id_accuracy_list2.append(acc)
            layer_sparse_rate, total_sparse_rate = prune.sparsity()
            print("total_sparse_rate2:", total_sparse_rate)

            #parameters counting
            para_num=sum(p.numel() for p in model_lc.parameters())*(1-total_sparse_rate)
            total_num = total_num + para_num
            #save model to model_list
            model_list[idx]=copy.deepcopy(model_lc)
            del model_lc
            gc.collect()
            del model_lb
            gc.collect()
            # torch.cuda.empty_cache()

        print("total  num:", total_num)
        traffic_history.append(total_num)

        #generate mask
        # prune_dict = {}
        # for k, v in model.named_parameters():
        #
        #     if ('intermediate.dense.weight' in k or 'output.dense.weight' in k) and (
        #             'attention.output.dense.weight' not in k):
        #         prune_dict[k] = 0.95  # 0.95
        #     # Att nn
        #     if 'attention.self.query.weight' in k or 'attention.self.key.weight' in k or 'attention.self.value.weight' in k or 'attention.output.dense.weight' in k:
        #         prune_dict[k] = 0.95  # 0.95
        cfg_mask_locals2 = []
        cfg_mask_locals2_sum = []
        for idx in idxs_users:
            model=copy.deepcopy(model_list[idx])
            cfg_local_mask2 = []
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if any(name == one for one in prune_dict):
                        weight = parameter.data.clone()
                        mask = torch.where(weight == 0, torch.zeros_like(weight), torch.ones_like(weight))
                        mask2 = mask.view(-1)
                        mask_2 = mask2.cpu().numpy().tolist()

                        cfg_local_mask2.extend(mask_2)
            cfg_mask_locals2.append(cfg_local_mask2)
            cfg_mask_locals2_sum.append(sum(cfg_local_mask2))

        del model
        # torch.cuda.empty_cache()
        print("====sum:=====")
        print(cfg_mask_locals2_sum)
        # generate weight matrix
        i = 0
        list_2 = []
        for mask in cfg_mask_locals2:
            list_1 = []
            for j in range(len(cfg_mask_locals2)):
                mask_compare = cfg_mask_locals2[j]
                similarity = cosVector(mask, mask_compare)
                list_1.append(similarity)
                # print(list_1)
            list_2.append(list_1)
        print(list_2)



        # torch.cuda.empty_cache()
        trainloader = DataLoader(DatasetSplit(train_dataset, dict_users_labeled), batch_size=32, shuffle=True)
        for k in range(args.distill_round):
            i = 0
            for idx in idxs_users:
                model=model_list[idx]
                optimizer = AdamW(model.parameters(),
                                  lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                  )
                for step, batch in enumerate(trainloader):
                    j = 0
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    b_labels = b_labels.cpu()
                    b_labels = b_labels - 1
                    b_labels = b_labels.to(device)


                    sum_z1=0
                    for idx2 in idxs_users:
                        if(idx2!=idx):
                            model_temp=model_list[idx2]
                            model_temp.eval()
                            with torch.no_grad():
                                output_temp = model_temp(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
                                output_temp2=global_model2(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
                            loss, logits = output_temp.loss, output_temp.logits
                            loss2, teacher_logits_2 = output_temp2.loss, output_temp2.logits
                            weight_list = list_2[i]
                            weight = weight_list[j] / (sum(weight_list) - weight_list[i])
                            sum_z1+=softmax(logits)*weight
                        j = j + 1
                    teacher_logits_1=sum_z1

                    model.train()
                    model.zero_grad()
                    output = model(b_input_ids,
                                             token_type_ids=None,
                                             attention_mask=b_input_mask,
                                             labels=b_labels)
                    loss,logits=output.loss,output.logits
                    loss2 = softmax_kl_loss(softmax(logits), teacher_logits_1)
                    loss3 = softmax_kl_loss(softmax(logits), teacher_logits_2)
                    loss4=loss+1*loss2+loss3
                    loss4.backward()
                    optimizer.step()
                i = i + 1

        #evaluate
        for idx in idxs_users:
            model=model_list[idx]
            prune = Prune(model, 0, num_steps_per_epoch * len(dict_users_unlabeled[idx]),
                          round(num_steps_per_epoch * 400 / 100), prune_dict, False, False, 'none')
            prune.prune(prune_times=model_pruned_statisic[idx])
            layer_sparse_rate, total_sparse_rate = prune.sparsity()
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users_unlabeled[idx], logger=logger)


            print("model id:",idx)
            print("sparsity:",total_sparse_rate)
            with torch.no_grad():
                acc, loss = local_model.inference(model=model)
                print("accuracy after distillation：", acc)
            accuracy_list[idx]=acc
            id_accuracy_list3.append(acc)
            model_list[idx].cpu()
        averge_accuracy = sum(accuracy_list) / len(accuracy_list)
        averge_accuracy_list.append(averge_accuracy)

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
        print("accuracy",accuracy_list,file=log_fp,flush=True)
        print("average accuracy", averge_accuracy)
        print("average accuracy", averge_accuracy, file=log_fp, flush=True)
        # averge_accuracy_list.append(averge_accuracy)
        print("average accuracy list", averge_accuracy_list, file=log_fp, flush=True)
        print("average accuracy list", averge_accuracy_list)
        print("traffic history:",traffic_history)
        print("traffic history:", traffic_history,file=log_fp, flush=True)
        print("-------------------------------------------",file=log_fp,flush=True)






        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))        






