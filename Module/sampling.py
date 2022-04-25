# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:52:27 2022

@author: TEYYC-TL6
"""

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_clients):
    """
    Sample I.I.D. client data from MNIST dataset
    
    """
    num_items = int(len(dataset)/num_clients)
    dict_users = {}
    available_indexes = [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(available_indexes, num_items,
                                             replace=False))
        # remove used indexes.
        available_indexes = list(set(available_indexes) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_clients, shards_per_client=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    
    The lower the "shards_per_client" the more non-IID.
    
    """
    num_shards = shards_per_client * num_clients # two shards per client
    images_per_shard = int(len(dataset)/num_shards)
    #
    index_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    indexes = np.arange(images_per_shard*num_shards)
    labels = dataset.targets.numpy()
    #
    # sort labels
    indexes_labels = np.vstack((indexes, labels)) # Size 2 x training imags
    indexes_labels = indexes_labels[:, indexes_labels[1, :].argsort()] # Ordered.
    indexes = indexes_labels[0, :]
    #
    # divide and assign shards_per_client shards/client
    for i in range(num_clients):
        random_set = set(np.random.choice(index_shard, shards_per_client, replace=False))
        index_shard = list(set(index_shard) - random_set)
        for rand_item in random_set:
            index_of_shard = range(rand_item*images_per_shard,(rand_item+1)*images_per_shard) # "images_per_shard" elements most likely of the same label.
            dict_clients[i] = np.concatenate((dict_clients[i], indexes[index_of_shard]), axis=0)
    # Safety checks.
    # [len(dict_clients[x]) for x in dict_clients.keys()]  # All equal
    # [set([labels[int(y)] for y in list(dict_clients[x])]) for x in dict_clients.keys()] # Only some labels per client.
    return dict_clients


def mnist_noniid_unequal(dataset, num_clients, shards_per_client=2):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    
    """
    shards_per_client_i_possible = range(0, shards_per_client) # From 0 to "shards_per_client"
    num_shards = shards_per_client * num_clients # two shards per client
    images_per_shard = int(len(dataset)/num_shards)
    #
    index_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    indexes = np.arange(images_per_shard*num_shards)
    labels = dataset.targets.numpy()
    #
    # sort labels
    indexes_labels = np.vstack((indexes, labels)) # Size 2 x training imags
    indexes_labels = indexes_labels[:, indexes_labels[1, :].argsort()] # Ordered.
    indexes = indexes_labels[0, :]
    #
    # We start with 1 shard by client as lower-bound
    for i in range(num_clients):
        random_set = set(np.random.choice(index_shard, 1, replace=False)) 
        index_shard = list(set(index_shard) - random_set)
        for rand_item in random_set:
            index_of_shard = range(rand_item*images_per_shard,(rand_item+1)*images_per_shard) # "images_per_shard" elements most likely of the same label.
            dict_clients[i] = np.concatenate((dict_clients[i], indexes[index_of_shard]), axis=0)
    # Now we set the rest of the shards.
    while len(index_shard) > 0: 
        for i in range(num_clients):
            shards_per_client_i = np.random.choice(shards_per_client_i_possible, 1, replace=False) # Random ammount of shards.
            random_set = set(np.random.choice(index_shard, shards_per_client_i, replace=False)) if len(index_shard)>shards_per_client else set(index_shard)
            index_shard = list(set(index_shard) - random_set)
            for rand_item in random_set:
                index_of_shard = range(rand_item*images_per_shard,(rand_item+1)*images_per_shard) # "images_per_shard" elements most likely of the same label.
                dict_clients[i] = np.concatenate((dict_clients[i], indexes[index_of_shard]), axis=0)
    # [len(dict_clients[x]) for x in dict_clients.keys()]
    # [set([labels[int(y)] for y in list(dict_clients[x])]) for x in dict_clients.keys()]
    return dict_clients

