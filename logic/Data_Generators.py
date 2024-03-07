import torch
import os.path as path
import torch.utils.data as datas
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
from logic.Utils import save_script_out_to_json    


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

def load_syntethic_data(batch_size=128, train_size = 0.9):  
    dataset = generate_syntethic_data()
    train_loader, test_loader = datas.random_split(dataset, [train_size, 1 - train_size])
    return train_loader, test_loader


#MNIST data generator:
def load_MNIST_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    return train_dataset, test_dataset


def normalise_company(data, company, save_normaliser=True):
    company_data = data[data['symbol'] == company]['high'].to_numpy()
    scaler = MinMaxScaler()
    company_data = company_data.reshape(-1, 1)
    company_data = scaler.fit_transform(company_data)
    data.loc[data['symbol'] == company, 'high'] = company_data
    if save_normaliser:
        joblib.dump(scaler, path.join('outputs', 'snp500', 'normalizers',  f'{company}_normaliser.pkl'))
    return data


def normalise_snp_data(data):
    companies = set(data['symbol'])
    for company in companies:
        data = normalise_company(data, company)
    return data


def convert_dates_to_integers(data):
    converted = np.array([np.array([datetime.strptime(row[0], '%Y-%m-%d'), row[1]]) for row in data if row[0]!=np.nan and '0' in row[0]])
    return converted

def generate_snp_company_with_dates(company):
    snp_path = path.join('snp500', 'snp500.csv')
    data = pd.read_csv(snp_path)
    data['symbol'] = data['symbol'].astype(str)
    data['date'] = data['date'].astype(str)
    data = data[data['symbol'] == company]
    data = data[['date', 'high']].dropna()
    return convert_dates_to_integers(data.to_numpy())

def cut_out_sample_from_each_snp_company_for_later(data, sequence_length=30):
    symbols = set(data['symbol'])
    symbols_samples = {}
    for symbol in symbols:
        company_data = data[data['symbol'] == symbol]
        company_data = company_data['high']
        i = random.randint(0 , len(company_data.index) - sequence_length)
        sample_to_cut = company_data[i:i+sequence_length+1]
        sample = sample_to_cut[:-1].to_list()
        test = sample_to_cut[1:].to_list()
        symbols_samples[symbol] = {"sample": sample, "test": test}
        data = data.drop(data.index[i:i+sequence_length+1])
        company_data = company_data.drop(company_data.index[i:i+sequence_length+1])
        data.loc[data['symbol'] == symbol, 'high'] = company_data
    save_script_out_to_json(symbols_samples, path.join("outputs", "snp500", 'snp500_visualization_samples.json'))
    return data


def count_companies_appearences(data):
    companies = {}
    for symbol in data['symbol']:
        if symbol in companies:
            companies[symbol] += 1
        else:
            companies[symbol] = 1
    return companies


def remove_non_dominant_companies (data, threshold = 300):
    companies = count_companies_appearences(data)
    dominant_companies = [company for company in companies if companies[company] >= threshold]
    return data[data['symbol'].isin(dominant_companies)]
    
def generate_snp_data_with_labels(sequence_length=30):
    print ("Loading data...")
    snp_path = path.join('data', 'snp500' ,'snp500.csv')
    data = pd.read_csv(snp_path)
    data['symbol'] = data['symbol'].astype(str)
    print("Cleaning...")
    data = remove_non_dominant_companies(data)
    print("Normalizing...")
    data = normalise_snp_data(data)
    print("Saving pieces for visualizations...")
    data = cut_out_sample_from_each_snp_company_for_later(data, sequence_length)
    symbols = set(data['symbol'])
    datasets = []
    labels = []
    for symbol in symbols:
        company_data = data[data['symbol'] == symbol]
        company_data = company_data['high'].dropna()
        company_sequentualised, company_labels = company_to_sequences_and_labels_v3(company_data, sequence_length)
        for seq in company_sequentualised:
            datasets.append(seq)
        for label in company_labels:
            labels.append(label)
    data = torch.tensor(np.array(datasets), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    return data, labels
    

def load_snp_data_for_kfolds (train_size=0.9, test_size=0.1):
    data, _ = generate_snp_data_with_labels()
    train_set, test_set = datas.random_split(data, [train_size, test_size])
    train_set = train_set.dataset
    test_set = test_set.dataset
    return train_set, test_set

def load_snp_data_with_labels_for_kfolds (train_size=0.9, test_size=0.1):
    data, labels = generate_snp_data_with_labels()
    dataset = datas.TensorDataset(data, labels)
    train_set, test_set = datas.random_split(dataset, [train_size, test_size])
    return train_set, test_set


def company_to_sequences_and_labels_v3(company_data, sequence_length = 30):
    sequences = []
    labels = []
    for i in range (len(company_data) - sequence_length):
        sequence = company_data[i : i+sequence_length]
        label = company_data[i+1 :i+sequence_length+1]
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)




