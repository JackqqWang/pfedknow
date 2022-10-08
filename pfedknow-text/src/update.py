import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, AdamW, BertConfig
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import (
#     BertForSequenceClassification,
#     BertConfig,
# )
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from logger import logger
from classifier_eval import (
    evaluate,
    calculate_head_importance,
    analyze_nli,
    predict,
)
from torch.optim import Optimizer

# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return torch.tensor(image), torch.tensor(label) 
import time
import datetime
import copy
import pruning
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        batch = self.dataset[self.idxs[item]]
        return batch




    # calculate sparsity rate
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

class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)
        return loss

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:  # param_groups = list(params)
            for p, localweight in zip(group['params'], weight_update):
                p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss #更新的是local model

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']

def penalty(model,copy_):

    w_epoch = list(model.parameters())
    copy_list=list(copy_.parameters())
    # print("layer0 :", w_epoch[0])
    # print("layer2:",w_epoch[2])
    deltaw_list = []
    k=[]
    # sum_w = 0
    for p in range(len(w_epoch)):
        deltaw_list.append(w_epoch[p]-copy_list[p])
        # print("i:",p,"layer:",deltaw_list[p].flatten())
        c=torch.dot(deltaw_list[p].flatten(), deltaw_list[p].flatten()).reshape(-1)[0]
        # print("c",c)
        if p==0:
            k.append(c)
        else:
            k.append(c+k[p-1])
            # print("sum_list", k)
    sum_w=k[p]
    return sum_w

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# from transformers import get_linear_schedule_with_warmup

# # Number of training epochs. The BERT authors recommend between 2 and 4. 
# # We chose to run for 4, but we'll see later that this may be over-fitting the
# # training data.
# epochs = 4

# # Total number of training steps is [number of batches] x [number of epochs]. 
# # (Note that this is not the same as the number of training samples).
# total_steps = len(train_dataloader) * epochs

# # Create the learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps)

class LocalUpdate(object):
    def __init__(self, args, dataset,idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.device = torch.device(args.gpuid) if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):

        """
        return train, val, test dataloaders for a given dataset and user indexes
    
        """
        idxs_train = idxs[: int(0.8 * len(idxs))]
        # idxs_val = idxs[int(0.8 * len(idxs)): int(0.89 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size =self.args.local_bs, shuffle = True)
        self.traindata=DatasetSplit(dataset, idxs_train)
        self.testdata=DatasetSplit(dataset, idxs_test)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size = 32, shuffle = False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size = int(len(idxs_test)/10), shuffle = False)
        
        # train_dataloader = DataLoader(
        #     train_dataset,  # The training samples.
        #     sampler = RandomSampler(train_dataset), # Select batches randomly
        #     batch_size = 32 # Trains with this batch size.
        # )
        #
        # validation_dataloader = DataLoader(
        #     val_dataset, # The validation samples.
        #     sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
        #     batch_size = 32 # Evaluate with this batch size.
        # )

        return trainloader, testloader


    def update_weights(self, model, global_round,epoch=1):

        model.train()
        # epoch_loss = []
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr, momentum = 0.5)
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr, weight_decay = 1e-4)

        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                     )

        t0 = time.time()
        total_train_loss = 0
        model_local_static = copy.deepcopy(model).to(self.device)
        for iter in range(epoch):
            print("")
            print('======== Epoch {:} / {:} ========'.format(iter + 1, self.args.local_ep))
            print('Training...')
            # for batch_idx, (images, labels) in enumerate(self.trainloader):
            for step, batch in enumerate(self.trainloader):
                if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
            
            # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.trainloader), elapsed))
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                b_labels = b_labels.cpu()
                b_labels = b_labels - 1
                b_labels = b_labels.to(self.device)
                model.zero_grad()
                output = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

                loss, logits = output.loss,output.logits
                # loss, logits = output
                total_train_loss += loss.item()
                penalt = penalty(model,model_local_static)
                loss1=loss+penalt
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # scheduler.step()
                avg_train_loss = total_train_loss / len(self.trainloader)            
    
    # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        # del self.trainloader


            #     if self.args.verbose and (batch_idx % 10 == 0):
            #         print('| Global Round: {} | Local Epoch: {} | [{}/{} ({:.0f}%)]\t Loss:{:.6f}'.format(
            #             global_round, iter, batch_idx * len(images), len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()
            #         ))
            #     self.logger.add_scalar('loss', loss.item())
            #     batch_loss.append(loss.item())
            # epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # return model.state_dict(), sum(epoch_loss)/len(epoch_loss)
        return model.state_dict(), avg_train_loss



    def update_weights2(self, model, global_round,label_model):

        model.train()
        # epoch_loss = []
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr, momentum = 0.5)
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr, weight_decay = 1e-4)

        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        optimizer = AdamW(model.parameters(),
                          lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )
        t0 = time.time()
        total_train_loss = 0
        model_local_static = copy.deepcopy(model).to(self.device)
        for iter in range(self.args.local_ep):
            print("")
            print('======== Epoch {:} / {:} ========'.format(iter + 1, self.args.local_ep))
            print('Training...')
            # for batch_idx, (images, labels) in enumerate(self.trainloader):
            for step, batch in enumerate(self.trainloader):
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.trainloader), elapsed))
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                b_labels = b_labels.cpu()
                b_labels = b_labels - 1
                b_labels = b_labels.to(self.device)
                with torch.no_grad():
                    output2 = label_model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
                    loss2,logits2=output2.loss, output2.logits
                    label1_hard = logits2.argmax(dim=1)
                model.zero_grad()
                output = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=label1_hard)

                loss, logits = output.loss, output.logits
                # loss, logits = output
                total_train_loss += loss.item()
                # penalt = penalty(model, model_local_static)
                loss1 = loss
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # scheduler.step()
                avg_train_loss = total_train_loss / len(self.trainloader)

                # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        #     if self.args.verbose and (batch_idx % 10 == 0):
        #         print('| Global Round: {} | Local Epoch: {} | [{}/{} ({:.0f}%)]\t Loss:{:.6f}'.format(
        #             global_round, iter, batch_idx * len(images), len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()
        #         ))
        #     self.logger.add_scalar('loss', loss.item())
        #     batch_loss.append(loss.item())
        # epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # return model.state_dict(), sum(epoch_loss)/len(epoch_loss)
        return model.state_dict(), avg_train_loss

    def update_weights_pFedMe(self, model, global_round,args):

        # param.data = new_param.data.clone()
        model.train()

        # local_model = copy.deepcopy(list(model.parameters()))
        local_model=copy.deepcopy(model)
        optimizer = AdamW(model.parameters(),
                          lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )
        # optimizer=pFedMeOptimizer(model.parameters(), lr=5e-5,lamda=0)
        t0 = time.time()
        total_train_loss = 0
        # model_local_static = copy.deepcopy(model).to(self.device)
        # for iter in range(self.args.local_ep):
        #     print("")
        #     print('======== Epoch {:} / {:} ========'.format(iter + 1, self.args.local_ep))
        #     print('Training...')
            # for batch_idx, (images, labels) in enumerate(self.trainloader):

        for step, batch in enumerate(self.trainloader):
            if(step<1000000):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                b_labels = b_labels.cpu()
                b_labels = b_labels - 1
                b_labels = b_labels.to(self.device)
                for i in range(args.K):
                    model.zero_grad()
                    output = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

                    loss, logits = output.loss, output.logits
                    # total_train_loss += loss.item()
                    loss1 = loss
                    loss2= penalty(model,local_model)
                    loss3=loss1+args.lamda*loss2*0.5
                    loss3.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # print("loss:",loss)
                    optimizer.step()
                    # persionalized_model_bar, _ = optimizer.step(local_model)
                    torch.cuda.empty_cache()
                total_train_loss += loss.item()
                for new_param, localweight in zip(model.parameters(), list(local_model.parameters())):#updata local model
                    localweight.data = localweight.data - args.lamda * args.learning_rate * (localweight.data - new_param.data)
        persionalized_model_bar=copy.deepcopy(list(model.parameters()))
        for param, new_param in zip(model.parameters(), list(local_model.parameters())):#update model
            param.data = new_param.data.clone()


        # scheduler.step()
        avg_train_loss = total_train_loss / len(self.trainloader)
        del local_model
        torch.cuda.empty_cache()

                # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        # print("")
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        # print("  Training epcoh took: {:}".format(training_time))

        #     if self.args.verbose and (batch_idx % 10 == 0):
        #         print('| Global Round: {} | Local Epoch: {} | [{}/{} ({:.0f}%)]\t Loss:{:.6f}'.format(
        #             global_round, iter, batch_idx * len(images), len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()
        #         ))
        #     self.logger.add_scalar('loss', loss.item())
        #     batch_loss.append(loss.item())
        # epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # return model.state_dict(), sum(epoch_loss)/len(epoch_loss)
        return model.state_dict(), avg_train_loss,persionalized_model_bar
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            batch = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            batch = next(self.iter_trainloader)
        return batch
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            batch = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            batch = next(self.iter_testloader)
        return batch
    def update_weights_PerAvg(self, model, global_round,args):

        # param.data = new_param.data.clone()
        model.train()

        # local_model = copy.deepcopy(list(model.parameters()))
        local_model=copy.deepcopy(list(model.parameters()))
        optimizer = AdamW(model.parameters(),
                          lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )
        # optimizer=pFedMeOptimizer(model.parameters(), lr=5e-5,lamda=0)
        # optimizer = MySGD(model.parameters(), lr=5e-5)
        t0 = time.time()
        total_train_loss = 0
        # model_local_static = copy.deepcopy(model).to(self.device)
        # for iter in range(self.args.local_ep):
        #     print("")
        #     print('======== Epoch {:} / {:} ========'.format(iter + 1, self.args.local_ep))
        #     print('Training...')
            # for batch_idx, (images, labels) in enumerate(self.trainloader):
        for iter in range(args.local_epochs):
            # print("epoch:", iter+1)
            model.train()
            temp_model = copy.deepcopy(list(model.parameters()))

            # step 1
            batch = self.get_next_train_batch()
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_labels = b_labels.cpu()
            b_labels = b_labels - 1
            b_labels = b_labels.to(self.device)
            optimizer.zero_grad()
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss, logits = output.loss, output.logits
            loss.backward()
            optimizer.step()

            #step 2
            batch = self.get_next_train_batch()
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_labels = b_labels.cpu()
            b_labels = b_labels - 1
            b_labels = b_labels.to(self.device)
            optimizer.zero_grad()
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss, logits = output.loss, output.logits
            loss.backward()

            for old_p, new_p in zip(model.parameters(), temp_model):
                old_p.data = new_p.data.clone()
            optimizer.step()




            for param, clone_param in zip(model.parameters(), local_model):
                clone_param.data = param.data.clone()
            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(self.trainloader)
        model_agg=copy.deepcopy(model.state_dict())
        #train_one_step
        # step 1
        batch = self.get_next_test_batch()
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        b_labels = b_labels.cpu()
        b_labels = b_labels - 1
        b_labels = b_labels.to(self.device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss, logits = output.loss, output.logits
        loss.backward()
        optimizer.step()

        # step 2
        batch = self.get_next_test_batch()
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        b_labels = b_labels.cpu()
        b_labels = b_labels - 1
        b_labels = b_labels.to(self.device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss, logits = output.loss, output.logits
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()




        return model_agg,model.state_dict(), avg_train_loss
    def update_weights_PerAvg_onestep(self, model, global_round,args):
        model.train()
        # optimizer = MySGD(model.parameters(), lr=5e-5)
        optimizer = AdamW(model.parameters(),
                          lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )
        # train_one_step
        # step 1

        batch = self.get_next_test_batch()
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        b_labels = b_labels.cpu()
        b_labels = b_labels - 1
        b_labels = b_labels.to(self.device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss, logits = output.loss, output.logits
        loss.backward()
        optimizer.step()

        # step 2
        batch = self.get_next_test_batch()
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        b_labels = b_labels.cpu()
        b_labels = b_labels - 1
        b_labels = b_labels.to(self.device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss, logits = output.loss, output.logits
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        return  model.state_dict()


    def model_prune(self, model, args):
        prune_sequence = pruning.determine_pruning_sequence(
            args.prune_number,
            args.prune_percent,
            model.bert.config.num_hidden_layers,
            model.bert.config.num_attention_heads,
            args.at_least_x_heads_per_layer,
        )
        to_prune = {}
        for step, n_to_prune in enumerate(prune_sequence):
            if step == 0 or args.exact_pruning:
                # Calculate importance scores for each layer
                head_importance = calculate_head_importance(
                    model,
                    self.traindata,
                    batch_size=32,
                    device=self.device,
                    normalize_scores_by_layer=args.normalize_pruning_by_layer,
                    subset_size=args.compute_head_importance_on_subset,
                    verbose=True,
                    disable_progress_bar=args.no_progress_bars,
                )
                logger.info("Head importance scores")
                for layer in range(len(head_importance)):
                    layer_scores = head_importance[layer].cpu().data
                    logger.info("\t".join(f"{x:.5f}" for x in layer_scores))
            # Determine which heads to prune
            to_prune = pruning.what_to_prune(
                head_importance,
                n_to_prune,
                to_prune={} if args.retrain_pruned_heads else to_prune,
                at_least_x_heads_per_layer=args.at_least_x_heads_per_layer
            )
            # Actually mask the heads
            # if args.actually_prune:
            model.bert.prune_heads(to_prune)
            print("model parameters:", sum(p.numel() for p in model.parameters()))
            print(sum(p.numel() for p in model.bert.parameters()))
            # else:
            #     model.bert.mask_heads(to_prune)

        return model
    def inference(self, model):
        """
            return inference and test acc & loss
        """
        model.eval()
        loss, total, correct = 0, 0, 0
        t0 = time.time()
        for step, batch in enumerate(self.testloader):
            if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            b_labels = b_labels - 1
            model.zero_grad()
            output = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            loss, logits = output.loss,output.logits
            # loss, logits = output
        # for _, (images, labels) in enumerate(self.testloader):
            # images, labels = images.to(self.device), labels.to(self.device)
            # # inference
            # outputs = model(images)
            # batch_loss = self.criterion(outputs, labels)
            # loss += batch_loss.item()
            # # prediction
            _, pred_labels = torch.max(logits, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, b_labels)).item()
            total += len(b_labels)
        accuracy = correct/total
        return accuracy, loss
                
def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device(args.gpuid) if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(
    test_dataset,  # The training samples.
    sampler = RandomSampler(test_dataset), # Select batches randomly
    batch_size = 32 # Trains with this batch size.
    )
    # testloader = DataLoader(test_dataset, batch_size=32,
    #                         shuffle=False)
    
    t0 = time.time()
    for step, batch in enumerate(testloader):
        if step % 40 == 0 and not step == 0:
        # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_labels = b_labels - 1
        model.zero_grad()
        output = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        loss, logits = output.loss,output.logits
        _, pred_labels = torch.max(logits, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, b_labels)).item()
        total += len(b_labels)
    accuracy = correct/total
    return accuracy, loss



    # for batch_idx, (images, labels) in enumerate(testloader):
    #     images, labels = images.to(device), labels.to(device)

    #     # Inference
    #     outputs = model(images)
    #     batch_loss = criterion(outputs, labels)
    #     loss += batch_loss.item()

    #     # Prediction
    #     _, pred_labels = torch.max(outputs, 1)
    #     pred_labels = pred_labels.view(-1)
    #     correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #     total += len(labels)

    # accuracy = correct/total
    # return accuracy, loss
        
