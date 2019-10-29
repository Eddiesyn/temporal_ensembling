# coding=utf-8

import torchvision.datasets as datasets
import torchvision.transforms as transforms

def prepare_mnist():
    # normalize data
    m = (0.1307,)
    st = (0.3081,)
    normalize = transforms.Normalize(m, st)

    # load train data
    train_dataset = datasets.MNIST(
        root='/usr/home/sut/datasets',
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True)

    # load test data
    test_dataset = datasets.MNIST(
        root='/usr/home/sut/datasets',
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    return train_dataset, test_dataset
