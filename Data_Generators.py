import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Utils import split_dataset, parse_args, get_optimizer
import numpy as np
from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER, LSTM_AE_CLASSIFIER_V2, LSTM_AE_CLASSIFIER_V3

#synthetic data generator:
def load_syntethic_data(batch_size=128):

    def generate_syntethic_data(num_sequences=10000, sequence_length=50):
        dataset = []
        for _ in range(num_sequences):
            sequence = np.random.rand(sequence_length)
            i = random.randint(20, 30)
            sequence[i-5:i+6] *= 0.1
            dataset.append(sequence)
        dataset = torch.tensor(np.array(dataset), dtype=torch.float32)
        return dataset
    
    dataset = generate_syntethic_data()
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader


#MNIST data generator:
def get_MNIST_train_test_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
