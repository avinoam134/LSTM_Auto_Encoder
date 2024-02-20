import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np

def get_optimizer(optimizer_name, model, learning_rate):

    optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name](model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')
    


def split_dataset(dataset, batch_size, train_size=0.6, test_size=0.2):
    dataset_size = len(dataset)
    train_size = int(train_size * dataset_size)
    test_size = int(test_size * dataset_size)
    val_size = dataset_size - train_size - test_size
    train_dataset, test_val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    test_dataset, val_dataset = torch.utils.data.random_split(test_val_dataset, [test_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--gradient_clipping', type=float, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    return parser.parse_args()