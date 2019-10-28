# coding=utf-8

import torch
import numpy as np


def sample_train(train_dataset, test_dataset, batch_size, k, n_classes, seed, shuffle_train=True, return_idx=True):
    '''Randomly form unlabel data in training dataset

    :param train_dataset:
    :param test_dataset:
    :param batch_size:
    :param k: keep k labeled data in whole training set, other witout label
    :param n_classes:
    :param seed: random seed for shuffle
    :param shuffle_train:
    :param return_idx: whether to return the indexes of labeled data
    :return:
    '''
    n = len(train_dataset)
    rrng = np.random.RandomState(seed)
    indices = torch.zeros(k) # indices of keep labeled data
    others = torch.zeros(n - k) # indices of unlabeled data
    card = k // n_classes
    cpt = 0

    for i in range(n_classes):
        class_items = (train_dataset == i).nonzero() # indices of samples with label i
        n_class = len(class_items) # number of samples with label i
        rd = rrng.permutation(np.arange(n_class)) # shuffle them
        indices[i * card : (i+1) * card] = class_items[rd[:card]]
        others[cpt:cpt+n_class-card] = class_items[rd[card:]]
        cpt += (n_class-card)

    train_dataset.train_labels[others] = -1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               shuffle=shuffle_train)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=False)

    if return_idx:
        return train_loader, test_loader, indices
    return train_loader, test_loader

def exp_lr_scheduler(optimizer, global_step, init_lr, decay_steps, decay_rate, lr_clip, staircase=True):
    if staircase:
        lr = init_lr * decay_rate**(global_step // decay_steps)
    else:
        lr = init_lr * decay_rate**(global_step / decay_steps)

    lr = max(lr, lr_clip)

    if global_step % decay_steps == 0:
        print('LR is decayed to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr