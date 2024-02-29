import torch
import os
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from Utils import save_script_out_to_json, load_script_out_from_json


def split_dataset(dataset, batch_size, train_size=0.6, test_size=0.2, shuffle = (True, False, False), save_test = False):
    dataset_size = len(dataset)
    train_size = int(train_size * dataset_size)
    test_size = int(test_size * dataset_size)
    val_size = dataset_size - train_size - test_size
    train_dataset, test_val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    test_dataset, val_dataset = torch.utils.data.random_split(test_val_dataset, [test_size, val_size])
    if save_test:
        torch.save(test_dataset, 'scripts_test_data.pth')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle[0])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle[1])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle[2])
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

def get_snp_stock_max_high(data, company):
    data = data[data['symbol'] == company]
    return data['high'].max()

def normalise_company(data, company):
    max_high = get_snp_stock_max_high(data, company)
    data.loc[data['symbol'] == company, 'high'] /= max_high
    return data

def normalise_snp_data(data):
    companies = set(data['symbol'])
    for company in companies:
        data = normalise_company(data, company)
    return data



def generate_snp_data(stocks_combined = True, company=None, sequence_length=30):
    snp_path = os.path.join('snp500', 'snp500.csv')
    data = pd.read_csv(snp_path)
    data['symbol'] = data['symbol'].astype(str)
    data = normalise_snp_data(data)
    if company is not None:
        data = data[data['symbol'] == company]
        data = data['high'].dropna().to_numpy()
        data = torch.tensor(company_to_sequences(data, sequence_length), dtype=torch.float32)
    elif stocks_combined:
        data = data['high'].dropna().to_numpy()
        data = torch.tensor(company_to_sequences(data, sequence_length), dtype=torch.float32)
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
        data = torch.tensor(np.array(datasets), dtype=torch.float32)

    return data

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

def load_snp_data(company=None, sequence_length=30, batch_size=128):
    data = generate_snp_data(company, sequence_length=sequence_length)
    dataset = torch.tensor(data, dtype=torch.float32)
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader

def company_to_sequences (company_data, sequence_length = 30):
    sequences = []
    com_len = len(company_data)
    unified_length = com_len - (com_len % sequence_length)
    i=0
    while i < unified_length:
        sequence = company_data[i:i+sequence_length]
        i+=sequence_length
        sequences.append(sequence)
    return np.array(sequences)



def load_snp_data_for_cross_validation(sequence_length = 30, num_batches_for_validation = 18 ,mini_batch_size=128, train_size=0.6, test_size=0.2):
    dataset = generate_snp_data(company=None, stocks_combined=True)
    #split dataset to a list of num_batches_for_validation batches:
    batch_size = len(dataset) // num_batches_for_validation
    batches = torch.split(dataset, batch_size)
    loaders = []
    for batch in batches:
        train_loader, test_loader, val_loader = split_dataset(batch, mini_batch_size, train_size, test_size, get_test_raw=True)
        loaders.append((train_loader, test_loader, val_loader))
    return loaders

def count_companies_appearences(data):
    #create a dict that matches each unique entry of data['symbol'] to its recurrences in data['symbol']:
    companies = {}
    for symbol in data['symbol']:
        if symbol in companies:
            companies[symbol] += 1
        else:
            companies[symbol] = 1
    return companies

def remove_non_dominant_companies (data, threshold = 1007):
    companies = count_companies_appearences(data)
    dominant_companies = [company for company in companies if companies[company] >= threshold]
    return data[data['symbol'].isin(dominant_companies)]

def remove_dates_without_all_the_dominant_companies(data):
    data_dominant = remove_non_dominant_companies(data)
    dominant_companies_set = set(data_dominant['symbol'])
    dates_list_of_companies = []
    dates = set(data_dominant['date'].astype(str))
    for date in dates:
        companies_on_data = data_dominant[data_dominant['date'] == date]
        companies_on_date_as_set = set(companies_on_data['symbol'])
        if companies_on_date_as_set != dominant_companies_set:
            data_dominant = data_dominant[data_dominant['date'] != date]
        else:
            dates_list_of_companies.append(companies_on_data['high'].to_numpy())
    #dates_list_of_companies is of shape(1007, 479)
    #doesnt leave an amount of data but still something to work with.
    #will prolly take alot of time to process such sequence length
    return data_dominant, np.array(dates_list_of_companies)



def generate_snp_data_with_sequences_as_dates():
    data = pd.read_csv(os.path.join('snp500', 'snp500.csv'))
    data, date_sequences = remove_dates_without_all_the_dominant_companies(data)
    return 

def load_snp_data_by_dates_as_sequences_for_cross_validation(num_batches_for_validation = 18 ,mini_batch_size=128, train_size=0.6, test_size=0.2):
    dataset = generate_snp_data_with_sequences_as_dates()
    #split dataset to a list of num_batches_for_validation batches:
    batch_size = len(dataset) // num_batches_for_validation
    batches = torch.split(dataset, batch_size)
    loaders = []
    for batch in batches:
        train_loader, test_loader, val_loader = split_dataset(batch, mini_batch_size, train_size, test_size)
        loaders.append((train_loader, test_loader, val_loader))
    return loaders



def company_to_sequences_and_labels (company_data, sequence_length = 30):
    sequences = []
    labels = []
    com_len = len(company_data)
    unified_length = com_len - (com_len % sequence_length)
    i=0
    while i < unified_length:
        sequence = company_data[i:i+sequence_length-1]
        label = company_data[i+sequence_length-1]
        i+=sequence_length
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def generate_snp_data_with_labels(sequence_length=30):
    snp_path = os.path.join('snp500', 'snp500.csv')
    data = pd.read_csv(snp_path)
    data['symbol'] = data['symbol'].astype(str)
    data = normalise_snp_data(data)
    symbols = set(data['symbol'])
    #iterate the symbols and create a dataset for each symbol:
    datasets = []
    labels = []
    for symbol in symbols:
        company_data = data[data['symbol'] == symbol]
        company_data = company_data['high'].dropna().to_numpy()
        company_sequentualised, company_labels = company_to_sequences_and_labels(company_data, sequence_length)
        for seq in company_sequentualised:
            datasets.append(seq)
        for label in company_labels:
            labels.append(label)
    data = torch.tensor(np.array(datasets), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    return data, labels
    
def load_snp_data_with_labels(sequence_length=30, batch_size=128):
    data, labels = generate_snp_data_with_labels(sequence_length)
    dataset = torch.utils.data.TensorDataset(data, labels)
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size, train_size=0.8, test_size=0.2, save_test=True)
    return train_loader, test_loader, val_loader


def load_snp_data_with_labels_for_cross_validation (num_batches_for_validation = 18 ,mini_batch_size=128, train_size=0.7, test_size=0.1):
    data, labels = generate_snp_data_with_labels()
    #split dataset to a list of num_batches_for_validation batches:
    dataset = torch.utils.data.TensorDataset(data, labels)
    batch_size = len(dataset) // num_batches_for_validation
    batches = torch.split(dataset, batch_size)
    loaders = []
    for batch in batches:
        train_loader, test_loader, val_loader = split_dataset(batch, mini_batch_size, train_size, test_size, save_test=True)
        loaders.append((train_loader, test_loader, val_loader))
    return loaders




def save_companies_max_values(data):
    companies = set(data['symbol'])
    max_values = {}
    for company in companies:
        max_values[company] = get_snp_stock_max_high(data, company)
    save_script_out_to_json(max_values, 'snp_companies_max_values.json')

def load_companies_max_values():
    return load_script_out_from_json('snp_companies_max_values.json')