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
from Module.utils import get_dataset
from Module.Update import LocalUpdate
from Module.test_utils import test_model
from Module.settingsParser import args_parser_fed


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
    train_dataset, test_dataset, client_groups = get_dataset(settings)

    # Set the model to train and send it to device.
    global_model.to(settings.device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_acc = [], []
    print_every_round = 2

    for epoch in tqdm(range(settings.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        numClients = max(int(settings.frac * settings.num_clients), 1)
        indexes_clients = np.random.choice(range(settings.num_clients), numClients, replace=False)

        for id_client in indexes_clients:
            local_model = LocalUpdate(settings=settings, dataset=train_dataset, indexes=client_groups[id_client])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, client = id_client)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights dict
        global_weights = federated_average(local_weights)

        # update global model
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for client in range(settings.num_clients):
            local_model = LocalUpdate(settings=settings, dataset=train_dataset,indexes=client_groups[client])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_acc.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'print_every_round' rounds
        if (epoch+1) % print_every_round == 0:
            print(f' \nAvg Training statistics after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print(f'Train Acc: {train_acc[-1]:.3f}% \n')

    # Test the global model.
    test_acc, test_loss = test_model(global_model, test_dataset, settings.criterion)

    print(f' \n Results after {settings.rounds} rounds of training:')
    print(f"|---- Avg Train Acc: {train_acc[-1]:.3f}%")
    print(f"|---- Test Acc: {test_acc:.3f}%")
    print(f'\n Total Time: {time.time()-start_time:0.2f}')

    # PLOTTING
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    plt.figure()
    plt.title('Training Loss vs Rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Rounds')
    plt.show()
    nameFile = f'./save/fed_R[{settings.rounds}]_C[{settings.frac}]_iid[{settings.iid}]_' + \
                    f'E[{settings.local_epoch}]_B[{settings.local_batch}]_Lr[{settings.lr}]_Opt[{settings.optimizer}]_loss.png'
    plt.savefig(nameFile)

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Rounds')
    plt.plot(range(len(train_acc)), train_acc, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Rounds')
    plt.show()
    nameFile = f'./save/fed_R[{settings.rounds}]_C[{settings.frac}]_iid[{settings.iid}]_' + \
                    f'E[{settings.local_epoch}]_B[{settings.local_batch}]_Lr[{settings.lr}]_Opt[{settings.optimizer}]_acc.png'
    plt.savefig(nameFile)

if __name__ == "__main__":
    settings = args_parser_fed()
    main(settings)

