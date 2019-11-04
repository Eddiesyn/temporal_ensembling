# coding=utf-8

import torchvision.datasets as datasets
import torchvision.transforms as transforms

def prepare_mnist(root, transform):
    """Prepare mnist dataset train and val

    :param root: path of mnist data
    :param transform: transform method of data
    :return:
    """
    # load train data
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=True)

    # load test data
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform, download=True)

    return train_dataset, test_dataset
