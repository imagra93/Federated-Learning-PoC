# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:57:39 2022

@author: TEYYC-TL6
"""

import torch 
import copy


def federated_average(weights):
    """
    Returns the average of the weights.
    """
    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    return weights_avg

