#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import time
import numpy as np
from tqdm import tqdm
import torch

from Module.Model import CNNmodel
from Module.Aggregation import federated_average
from Module.utils import get_dataset_3lvl
from Module.Update import LocalUpdate
from Module.test_utils import test_model
from Module.settingsParser import args_parser_fed_3lvl


def main(settings):

    print(settings)

    if settings.criterion == 'Nllloss':
        settings.criterion = torch.nn.NLLLoss()

    start_time = time.time()

    # BUILD MODEL
    global_model = CNNmodel()

    if torch.cuda.device_count():
        torch.cuda.set_device(0)
        settings.device = 'cuda'
    else:
        settings.device = 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, cluster_groups, client_groups = get_dataset_3lvl(settings)

    """
    for cluster_clients in cluster_groups:
        # cluster_clients - List of clients in cluster.
        for client in cluster_clients:
            client_groups[client] # Set of samples of client.
    """   

    # Set the model to train and send it to device.
    global_model.to(settings.device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_acc = [], []
    print_every_round = 2

    # Layer 1 -  Total rounds between Central node and each cluster.
    #tqdm
    for epoch in range(settings.global_rounds):

        cluster_weights_list = []

        # Total Rounds in Layer 2 - each Cluster.
        for cluster_id, cluster_clients in enumerate(cluster_groups):

            # Layer 2 - Start each cluster from the current global model.
            global_model.load_state_dict(global_weights)

            for cluster_epoch in range(settings.cluster_rounds):

                    local_weights, local_losses = [], []
                    print(f' Global Round : {epoch+1}/{settings.global_rounds} | Cluster id {cluster_id + 1}/{settings.num_clusters} | Cluster Epoch : {cluster_epoch+1}/{settings.cluster_rounds}')

                    global_model.train()

                    for id_client in cluster_clients:
                        local_model = LocalUpdate(settings=settings, dataset=train_dataset, indexes=client_groups[id_client])
                        w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, client=id_client)
                        local_weights.append(copy.deepcopy(w))
                        local_losses.append(copy.deepcopy(loss))

                    # update cluster weights dict
                    cluster_weights = federated_average(local_weights)

                    # update model inside cluster from info inside its clients.
                    global_model.load_state_dict(cluster_weights)
            
            # Copy the State Dict of the cluster.
            cluster_weights_list.append(copy.deepcopy(global_model.state_dict()))  

        # Layer 1 - Update global weights dict from cluster weights.
        global_weights = federated_average(cluster_weights_list)

        # Layer 1 - Update global model
        global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every Layer 1 round
        list_acc, list_loss = [], []
        global_model.eval()
        for client in range(settings.num_clients):
            local_model = LocalUpdate(settings=settings, dataset=train_dataset, indexes=client_groups[client])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_acc.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'print_every_round' rounds
        if (epoch + 1) % print_every_round == 0:
            print(f' \nAvg Training statistics after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print(f'Train Acc: {train_acc[-1]:.3f}% \n')

    # Test the global model.
    test_acc, test_loss = test_model(global_model, test_dataset, settings.criterion)

    print(f' \n Results after {settings.global_rounds} rounds of training:')
    print(f"|---- Avg Train Acc: {train_acc[-1]:.3f}%")
    print(f"|---- Test Acc: {test_acc:.3f}%")
    print(f'\n Total Time: {time.time()-start_time:0.2f}')

if __name__ == "__main__":
    settings = args_parser_fed_3lvl()
    main(settings)
