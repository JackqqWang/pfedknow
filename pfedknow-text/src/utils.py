from torchvision import datasets, transforms
from torchvision.datasets import cifar
from sampling import *
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
import copy
import torch
import torchtext
import os
from torchtext.datasets import YahooAnswers,AG_NEWS
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def generate_tok_dataloader(input_dataset, tokenizer):

#     tokens = []
#     max_len = 0
    
#     for label, line in input_dataset:
#         tokens += tokenizer.tokenize(line)
#         input_ids = tokenizer.encode(line, add_special_tokens=True, truncation=True)
# # Update the maximum sentence length.
#         max_len = max(max_len, len(input_ids))
#     print('Max sentence length: ', max_len)

    input_ids = []
    attention_masks = []
    # For every sentence...
    labels = []
    count = 0
    for label, line in input_dataset:
        labels.append(label)
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            line,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # padding = True,
                            max_length = 200,          # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        # 减少数据量做测试
        # count += 1
        # if count >= 500:
        #     break
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset




def get_dataset(args):    
    #return train and test datasets, and a user group where key is the user index and value is the data
    if args.dataset == 'yahoo':
        # if not os.path.isdir('../data/yahoo/'):
        #     os.mkdir('../data/yahoo')
        home_root = '/home/jmw7289/ys/own_vanilla_bert_fedavg_shenglai'
        train_raw_dataset = YahooAnswers(root = home_root +'/data/yahoo', split = 'train')
        test_raw_dataset = YahooAnswers(root = home_root +'/data/yahoo', split = 'test')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_dataset = generate_tok_dataloader(train_raw_dataset, tokenizer)
        test_dataset = generate_tok_dataloader(test_raw_dataset, tokenizer)
        if args.iid:
            dict_users_labeled, dict_users_unlabeled = iid(train_dataset, args.num_users,args.label_rate,args.seed)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                dict_users_labeled, dict_users_unlabeled = noniid(train_dataset, args.num_users,args.label_rate,args.seed)
    if args.dataset=='ag':
        home_root='/home/jmw7289/ys/own_vanilla_bert_fedavg_shenglai'
        train_raw_dataset=AG_NEWS(root=home_root +'/data/ag', split='train')
        test_raw_dataset=AG_NEWS(root = home_root +'/data/ag', split = 'test')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_dataset = generate_tok_dataloader(train_raw_dataset, tokenizer)
        test_dataset = generate_tok_dataloader(test_raw_dataset, tokenizer)
        if args.iid:
            dict_users_labeled, dict_users_unlabeled = iid(train_dataset, args.num_users,args.label_rate,args.seed)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                dict_users_labeled, dict_users_unlabeled = noniid(train_dataset, args.num_users,args.label_rate,args.seed)

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train = True, download = True, transform = apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train = False, download = True, transform = apply_transform)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset =='fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, dict_users_labeled, dict_users_unlabeled


def average_weights(w):

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    GPUID     : {args.gpuid}')
    print(f'    Model     : {args.model}')
    # print(f'    Optimizer : {args.optimizer}')
    # print(f'    LR  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Training Epochs: {args.local_ep}\n')
    return