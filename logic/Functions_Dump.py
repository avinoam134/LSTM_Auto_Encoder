import torch
import os
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from logic.Utils import save_script_out_to_json, load_script_out_from_json
from sklearn.preprocessing import MinMaxScaler
from Data_Generators import *
from LSTMS import *
from Trainers import *
from Utils import *


##################################################################################################################################################
'''
This file is used in any question's final solution.
It is saved as a somewhat proof of attempts made to succeed at some questions, and thus shows
my learning process.
'''
#################################################################################################################################################################3


def get_snp_stock_max_high(data, company):
    data = data[data['symbol'] == company]
    return data['high'].max()


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

def load_snp_data_with_labels(sequence_length=30, batch_size=128):
    data, labels = generate_snp_data_with_labels(sequence_length)
    dataset = datas.TensorDataset(data, labels)
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


def load_snp_data(company=None, sequence_length=30, batch_size=128):
    data = generate_snp_data(company, sequence_length=sequence_length)
    dataset = torch.tensor(data, dtype=torch.float32)
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader


def save_companies_max_values(data):
    companies = set(data['symbol'])
    max_values = {}
    for company in companies:
        max_values[company] = get_snp_stock_max_high(data, company)
    save_script_out_to_json(max_values, 'snp_companies_max_values.json')


def company_to_sequences_and_labels_v2 (company_data, sequence_length = 30):
    sequences = []
    labels = []
    for i in range (len(company_data) - sequence_length):
        sequence = company_data[i:i+sequence_length-1]
        label = company_data[i+sequence_length-1]
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def get_accuracy_level_as_percentage (predictions, labels):
    return 100*abs(predictions-labels)/labels
