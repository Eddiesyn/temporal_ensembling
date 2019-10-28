# coding=utf-8

import torch
from model import HandyModel

from datasets import prepare_mnist
import utils

num_epochs = 5
init_lr = 0.1
batch_size = 32
keep_label = 100 # keep k labeled data in whole training set, other witout label
seed = 1
decay_epoch = 1


if __name__ == '__main__':

    for epoch in num_epochs:
        t_dataset, v_dataset = prepare_mnist()

        model = HandyModel(batch_size, training=True)
        model.cuda() # transfer model to device

        ntrain = len(t_dataset)

        t_loader, v_loader, indices = utils.sample_train(t_dataset, v_dataset, batch_size, keep_label, 10, seed)

        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
        # utils.exp_lr_scheduler(optimizer, epoch, init_lr, )

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.99))

        model.train()
        losses = []
        sup_losses = []