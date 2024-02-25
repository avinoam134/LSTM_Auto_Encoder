import torch
import os
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random

datas_path = os.path.join('..', 'data')

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


#synthetic data generator:
def generate_syntethic_data(num_sequences=10000, sequence_length=50):
    dataset = []
    for _ in range(num_sequences):
        sequence = np.random.rand(sequence_length)
        i = random.randint(20, 30)
        sequence[i-5:i+6] *= 0.1
        dataset.append(sequence)
    dataset = torch.tensor(np.array(dataset), dtype=torch.float32)
    return dataset

def load_syntethic_data(batch_size=128):  
    dataset = generate_syntethic_data()
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader


#MNIST data generator:
def load_MNIST_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_pth = os.path.join(datas_path, 'mnist_data')
    train_dataset = datasets.MNIST(root=mnist_pth, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=mnist_pth, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def generate_snp_data():
    snp_path = os.path.join(datas_path, 'snp500', 'snp500.csv')
    data = pd.read_csv(snp_path)
    data = data.dropna()
    return data

def load_snp_data(batch_size=128):
    pass