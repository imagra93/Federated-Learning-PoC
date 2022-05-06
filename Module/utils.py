# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:53:24 2022

@author: TEYYC-TL6
"""


from torchvision import datasets, transforms
from Module.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal


def get_dataset(settings):
    """ 
    Returns train and test datasets and a user group which is a dict where
    the keys are the client index and the values are the corresponding data for
    each of those clients.
    client_groups[index] -> List of indexes.
    
    """

    data_dir = '../data/'+settings.dataset+'/'

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)

    client_groups = []

    if settings.iid is not None:

        # sample training data amongst users
        if settings.iid == 1:
            shards_per_client = 8 # low non-iid
        elif settings.iid == 2:
            shards_per_client = 4 # medium non-iid
        else:
            shards_per_client = 2 # large non-iid

        if settings.iid == 0:
            # Sample IID user data from Mnist
            client_groups = mnist_iid(train_dataset, settings.num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if settings.unequal:
                # Chose uneuqal splits for every user
                client_groups = mnist_noniid_unequal(train_dataset, settings.num_clients,shards_per_client)
            else:
                # Chose euqal splits for every user
                client_groups = mnist_noniid(train_dataset, settings.num_clients,shards_per_client)

    return train_dataset, test_dataset, client_groups


def get_dataset_3lvl(settings):

    """ 
    Returns train and test datasets and a user group which is a dict where
    the keys are the client index and the values are the corresponding data for
    each of those clients.
    client_groups[index] -> List of indexes.
    
    """

    data_dir = '../data/'+settings.dataset+'/'

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)

    client_groups = []

    if settings.iid is not None:
        # sample training data amongst users
        if settings.iid == 1:
            shards_per_client = 8 # low non-iid
        elif settings.iid == 2:
            shards_per_client = 4 # medium non-iid
        else:
            shards_per_client = 2 # large non-iid

        if settings.iid == 0:
            # Sample IID user data from Mnist
            client_groups = mnist_iid(train_dataset, settings.num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if settings.unequal:
                # Chose uneuqal splits for every user
                client_groups = mnist_noniid_unequal(train_dataset, settings.num_clients,shards_per_client)
            else:
                # Chose euqal splits for every user
                client_groups = mnist_noniid(train_dataset, settings.num_clients,shards_per_client)

    cluster_groups = []
    num_clients_per_cluster = int(settings.num_clients / settings.num_clusters)
    for i in range(num_clients_per_cluster):
        cluster_groups.append(list(range(i*int(settings.num_clusters),(i+1)*int(settings.num_clusters))))
    
    return train_dataset, test_dataset, cluster_groups, client_groups

