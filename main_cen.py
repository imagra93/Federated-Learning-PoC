# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:54:46 2022

@author: TEYYC-TL6
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import DataLoader

from Module.Model import CNNmodel
from Module.utils import get_dataset
from Module.test_utils import test_model
from Module.settingsParser import args_parser_cen

"""   
        SETTINGS
"""
if __name__ == "__main__":
    settings = args_parser_cen()
    print(settings)
else:
    # Dataset
    settings = args_parser_cen()
    settings.dataset = 'mnist'
    settings.num_classes = 10
    # Learning
    settings.epochs = 10
    settings.batch_size = 8
    # Optimizer
    settings.optimizer = 'sgd' # adam
    settings.lr = 0.01
    settings.momentum = 0.5
    settings.weight_decay = 1e-4
    # Loss Functions
    settings.criterion = 'Nllloss' #torch.nn.NLLLoss()

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
train_dataset, test_dataset, _ = get_dataset(settings)

# Set the model to train and send it to device.
global_model.to(settings.device)
global_model.train()
print(global_model)

# Training
# Set optimizer and criterion
if settings.optimizer == 'sgd':
    optimizer = torch.optim.SGD(global_model.parameters(), lr=settings.lr,
                                momentum=settings.momentum)
elif settings.optimizer == 'adam':
    optimizer = torch.optim.Adam(global_model.parameters(), lr=settings.lr,
                                 weight_decay=settings.weight_decay)

trainloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
criterion = settings.criterion.to(settings.device)
epoch_loss = []

for epoch in tqdm(range(settings.epochs)):
    batch_loss = []

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(settings.device), labels.to(settings.device)
        # reset optimizer (it is accumulative)
        optimizer.zero_grad()
        outputs = global_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % (len(trainloader)/40) == 0:
            print(f'Train Epoch: {epoch+1} | \t{100. * batch_idx / len(trainloader):.0f}% | \tLoss: {loss.item():.3f}')
        batch_loss.append(loss.item())

    loss_avg = sum(batch_loss)/len(batch_loss)
    print(f'\nTrain loss: {loss_avg}\n')
    epoch_loss.append(loss_avg)

# testing
test_acc, test_loss = test_model(global_model, test_dataset, settings.criterion)
print(f'Test on {len(test_dataset)} samples')
print(f"Test Accuracy: {100*test_acc:.2f}%")

# Plot loss
plt.figure()
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.title(f"Test acc: {test_acc:.3f}")
plt.show()
nameFile = f'./save/cen_E[{settings.epochs}]_B[{settings.batch_size}]_Lr[{settings.lr}]_Opt[{settings.optimizer}]_loss.png'
plt.savefig(nameFile)

