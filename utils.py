# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape, std):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size, ) + input_shape
        self.std = std
        self.noise = torch.zeros(self.shape).cuda()

    def forward(self, x):
        self.noise.normal_(mean=0, std=self.std)
        # print(self.noise.shape)

        return x + self.noise


def temporal_losses(out1, out2, w, labels):
    '''Calculate total loss

    :param out1: current output
    :param out2: temporal output
    :param w: weight for MSE loss
    :param labels:
    :return: the temporal loss
    '''

    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    total_loss = sup_loss + w * unsup_loss

    return total_loss, sup_loss, unsup_loss, nbsup


def mse_loss(out1, out2):
    quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)

    return quad_diff / out1.data.nelement()


def masked_crossentropy(out, labels):
    cond = (labels >= 0)
    nnz = torch.nonzero(cond)  # array of labeled sample index
    nbsup = len(nnz)  # number of supervised samples
    # check if labeled samples in batch, return 0 if none
    if nbsup > 0:
        # select lines in out with label
        masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
        masked_labels = labels[cond]
        loss = F.cross_entropy(masked_outputs, masked_labels)
        return loss, nbsup
    loss = torch.tensor([0.], requires_grad=False).cuda()
    return loss, 0


def sample_train(train_dataset, test_dataset, batch_size, k, n_classes, seed, shuffle_train=False, return_idx=True):
    '''Randomly form unlabeled data in training dataset

    :param train_dataset:
    :param test_dataset:
    :param batch_size:
    :param k: keep k labeled data in whole training set, other witout label
    :param n_classes:
    :param seed: random seed for shuffle
    :param shuffle_train: default false cause every epoch all samples only change once, they need to be consistent
    :param return_idx: whether to return the indexes of labeled data
    :return:
    '''
    n = len(train_dataset)  # 60000 for mnist
    rrng = np.random.RandomState(seed)
    indices = torch.zeros(k)  # indices of keep labeled data
    others = torch.zeros(n - k)  # indices of unlabeled data
    card = k // n_classes
    cpt = 0

    for i in range(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero()  # indices of samples with label i
        n_class = len(class_items)  # number of samples with label i
        rd = rrng.permutation(np.arange(n_class))  # shuffle them
        indices[i * card: (i+1) * card] = torch.squeeze(class_items[rd[:card]])
        others[cpt: cpt+n_class-card] = torch.squeeze(class_items[rd[card:]])
        cpt += (n_class-card)

    # tensor as indices must be long, byte or bool
    others = others.long()
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


def weight_scheduler(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    '''Weight scheduler

    :param epoch: current epoch
    :param max_epochs: maximum epoch, after that weight keep same
    :param max_val: maximum weight val, usually is 1 (the two is equally contributed in loss)
    :param mult: controls how slow weight goes up, default 5
    :param n_labeled: number of labeled samples
    :param n_samples: number of total samples

    '''
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)


def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(-mult * (1. - float(epoch) / max_epochs) ** 2)


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


def calc_metrics(model, loader):
    correct = 0
    total = 0
    for i, (samples, labels) in enumerate(loader):
        samples = samples.cuda()
        labels = labels.requires_grad_(False).cuda()
        outputs = model(samples)
        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels.detach().view_as(predicted)).sum()
    acc = 100 * float(correct) / total
    return acc
