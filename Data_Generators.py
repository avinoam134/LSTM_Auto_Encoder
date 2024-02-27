import torch
import os
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random
from datetime import datetime



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
    mnist_pth = 'mnist_data'
    train_dataset = datasets.MNIST(root=mnist_pth, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=mnist_pth, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def generate_snp_data(company=None, sequence_length=50):
    snp_path = os.path.join('snp500', 'snp500.csv')
    data = pd.read_csv(snp_path)
    data['symbol'] = data['symbol'].astype(str)
    if company is not None:
        data = data[data['symbol'] == company]
        data = data['high'].dropna().to_numpy()
        data = company_to_sequences(data, sequence_length)
    else:
        symbols = set(data['symbol'])
        #iterate the symbols and create a dataset for each symbol:
        datasets = []
        for symbol in symbols:
            company_data = data[data['symbol'] == symbol]
            company_data = company_data['high'].dropna().to_numpy()
            company_sequentualised = company_to_sequences(company_data, sequence_length)
            for seq in company_sequentualised:
                datasets.append(seq)
        data = np.array(datasets)

def convert_dates_to_integers(data):
    converted = np.array([np.array([datetime.strptime(row[0], '%Y-%m-%d'), row[1]]) for row in data if row[0]!=np.nan and '0' in row[0]])
    return converted

def generate_snp_company_with_dates(company):
    snp_path = os.path.join('snp500', 'snp500.csv')
    data = pd.read_csv(snp_path)
    data['symbol'] = data['symbol'].astype(str)
    data['date'] = data['date'].astype(str)
    data = data[data['symbol'] == company]
    data = data[['date', 'high']].dropna()
    return convert_dates_to_integers(data.to_numpy())

def load_snp_data(company=None, sequence_length=50, batch_size=128):
    data = generate_snp_data(company)
    dataset = torch.tensor(data, dtype=torch.float32)
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader

def company_to_sequences (company_data, sequence_length = 50):
    sequences = []
    com_len = len(company_data)
    unified_length = com_len - (com_len % sequence_length)
    i=0
    while i < unified_length:
        sequence = company_data[i:i+sequence_length]
        i+=sequence_length
        sequences.append(sequence)
    return np.array(sequences)



def load_snp_data_for_cross_validation(sequence_length = 50, num_batches_for_validation = 18 ,mini_batch_size=128, train_size=0.6, test_size=0.2):
    dataset = generate_snp_data(company=None)
    #split dataset to a list of num_batches_for_validation batches:
    batch_size = len(dataset) // num_batches_for_validation
    batches = torch.split(dataset, batch_size)
    loaders = []
    for batch in batches:
        train_loader, test_loader, val_loader = split_dataset(batch, mini_batch_size, train_size, test_size)
        loaders.append((train_loader, test_loader, val_loader))
    return loaders

    