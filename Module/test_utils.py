# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:10:29 2022

@author: TEYYC-TL6
"""

import torch
from torch.utils.data import DataLoader

def test_model(model, test_dataset, criteria):
    """ 
    Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    if torch.cuda.device_count():
        torch.cuda.set_device(0)
        device = 'cuda'
    else:
        device = 'cpu'
    criterion = criteria.to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss