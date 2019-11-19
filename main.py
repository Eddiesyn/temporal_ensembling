# coding=utf-8

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from timeit import default_timer as timer
import torchvision.transforms as transforms

from models.Handymodel import HandyModel
from datasets.datasets import prepare_mnist
import utils
import config

cfg = vars(config)


def train(model, writer, seed, k=100, alpha=0.6, lr=0.002, num_epochs=150, batch_size=64, n_classes=10, max_epochs=80, max_val=1.):
    # prepare data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg['m'], cfg['s'])])
    train_dataset, val_dataset = prepare_mnist(root='~/datasets/MNIST', transform=transform)
    ntrain = len(train_dataset)

    # build model and feed to GPU
    model.cuda()

    # make data loaders
    train_loader, val_loader, indices = utils.sample_train(train_dataset, val_dataset, batch_size, k, n_classes, seed, shuffle_train=False)

    # setup param optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # model.train()

    Z = torch.zeros(ntrain, n_classes).float().cuda()  # intermediate values
    z = torch.zeros(ntrain, n_classes).float().cuda()  # temporal outputs
    outputs = torch.zeros(ntrain, n_classes).float().cuda()  # current outputs

    losses = []
    suplosses = []
    unsuplosses = []
    best_loss = 30.0
    for epoch in range(num_epochs):
        t = timer()
        print('\nEpoch: {}'.format(epoch+1))
        model.train()
        # evaluate unsupervised cost weight
        w = utils.weight_scheduler(epoch, max_epochs, max_val, 5, k, 60000)

        w = torch.tensor(w, requires_grad=False).cuda()
        print('---------------------')

        # targets change only once per epoch
        for i, (images, labels) in enumerate(train_loader):
            batch_size = images.size(0)  # retrieve batch size again cause drop last is false
            images = images.cuda()
            labels = labels.requires_grad_(False).cuda()

            optimizer.zero_grad()
            out = model(images)
            zcomp = z[i * batch_size: (i+1) * batch_size]
            # zcomp = torch.tensor(zcomp, requires_grad=False)
            zcomp.requires_grad_(False)
            loss, suploss, unsuploss, nbsup = utils.temporal_losses(out, zcomp, w, labels)

            # save outputs
            outputs[i * batch_size: (i+1) * batch_size] = out.clone().detach()
            losses.append(loss.item())
            suplosses.append(nbsup * suploss.item())
            unsuplosses.append(unsuploss.item())

            # backprop
            loss.backward()
            optimizer.step()

            # print loss every 100 steps
            # if (i + 1) % 100 == 0:
            #     print('Step [%d/%d], Loss: %.6f, Time: %.2f s' % (i+1, len(train_dataset) // batch_size,
            #                                                       float(np.mean(losses)), timer()-t))

        loss_mean = np.mean(losses)
        supl_mean = np.mean(suplosses)
        unsupl_mean = np.mean(unsuplosses)

        writer.add_scalar('total loss', loss_mean, (epoch + 1) * ntrain)
        print('Epoch [%d/%d], Loss: %.6f, Supervised Loss: %.6f, Unsupervised Loss: %.6f, Time: %.2f' %
              (epoch + 1, num_epochs, float(loss_mean), float(supl_mean), float(unsupl_mean), timer()-t))
        writer.add_scalar('supervised loss', supl_mean, (epoch + 1) * ntrain)
        writer.add_scalar('unsupervised loss', unsupl_mean, (epoch + 1) * ntrain)

        Z = alpha * Z + (1. - alpha) * outputs
        z = Z * (1. / (1. - alpha ** (epoch + 1)))

        if loss_mean < best_loss:
            best_loss = loss_mean
            torch.save({'state_dict': model.state_dict()}, 'model_best.pth')

        model.eval()
        acc = utils.calc_metrics(model, val_loader)
        writer.add_scalar('Acc', acc, (epoch + 1) * ntrain)
        print('Acc : %.2f' % acc)

    # test best model
    checkpoint = torch.load('model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    acc_best = utils.calc_metrics(model, val_loader)
    print('Acc (best model): %.2f' % acc_best)


if __name__ == '__main__':
    for i in range(cfg['n_exp']):
        model = HandyModel(cfg['batch_size'], cfg['std'], fm1=cfg['fm1'], fm2=cfg['fm2'])
        seed = cfg['seeds'][i]
        writer = SummaryWriter('results/exp_{}'.format(i+1))
        train(model, writer, seed)
