from os import replace
import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def iid(dataset, num_users, label_rate,seed):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled, dict_users_unlabeled, dict_users_unlabeled_test, dict_users_unlabeled_train = set(), {}, {}, {}
    # np.random.seed(seed+1)
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))

    for i in range(num_users):
        # np.random.seed(seed+2)
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items), replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i])
        # dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
        unlabeled_semi = dict_users_unlabeled[i] - dict_users_labeled
        # dict_users_unlabeled[i]=dict_users_unlabeled[i] - dict_users_labeled
        list_temp = list(unlabeled_semi)  # Local unlabeled data without server data
        frac = 0.2
        np.random.seed(seed+3)
        ran_li = np.random.choice(list_temp, int(len(list_temp) * 0.2),replace=False)
        dict_users_unlabeled_test[i] = set(ran_li)  # test set(without server data)

        dict_users_unlabeled_train[i] = dict_users_unlabeled[i] - dict_users_unlabeled_test[i]  # local train data(with server data)
    return dict_users_labeled, dict_users_unlabeled

def noniid(dataset, num_users, label_rate,seed):
    num_shards, num_imgs = 2 * num_users, int(len(dataset) / num_users / 2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_unlabeled_test, dict_users_unlabeled_train = {}, {}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))

    for i in range(len(dataset)):
        labels[i] = dataset[i][2]  # label

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
        # dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
        list_temp = list(unlabeled_semi)
        frac = 0.2
        np.random.seed(seed + 3)
        ran_li = np.random.choice(list_temp, int(len(list_temp) * 0.2), replace=False)

        dict_users_unlabeled_test[i] = set(ran_li)
        dict_users_unlabeled_train[i] = dict_users_unlabeled[i] - dict_users_unlabeled_test[i]
    return dict_users_labeled, dict_users_unlabeled


def cifar_iid(dataset, num_users):
    label_rate=0.1
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users,label_rate,seed):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    num_shards, num_imgs = 2 * num_users, int(len(dataset) / num_users / 2)
    # num_shards, num_imgs  = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_imgs)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.train_labels)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace = False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand+1) * num_imgs]), axis = 0)
    return dict_users,dict_users_labeled
