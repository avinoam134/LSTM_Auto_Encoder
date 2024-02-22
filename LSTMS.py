import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from Trainers import Basic_Trainer, Classifier_Trainer

'''basic LSTM AE as in the diagram. used in lstm_ae_toy.py and as a basis to other models'''
class LSTM_AE(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, layers=1):
        super(LSTM_AE, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layers = layers
        self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
        dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
        enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
        dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
        return dec

    

'''AE with classification based SOLELY on the encoded image (single linear layer for classification).
Used as a benchmark for the other models (~90% accuracy on MNIST).'''
class LSTM_AE_CLASSIFIER(nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER, self).__init__()
        self.lstm_ae = LSTM_AE(input_size, hidden_size, layers)
        self.classifier = nn.Linear(input_size**2, 10)

    def forward(self, x):
        dec = self.lstm_ae(x)
        classification = self.classifier(x.reshape(-1, self.lstm_ae.input_size**2))
        return dec, classification
    

'''AE with classification based SOLELY on the decoded / reconstructed image (single linear layer for classification).'''
class LSTM_AE_CLASSIFIER_V2 (nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER_V2, self).__init__()
        self.lstm_ae = LSTM_AE(input_size, hidden_size, layers)
        self.classifier = nn.Linear(input_size, 10)

    def forward(self, x):
        dec = self.lstm_ae(x)
        classification = self.classifier(dec[:, -1, :])
        return dec, classification
    






































def get_model_and_trainer(model_name, input_size, hidden_size):
    if model_name == 'LSTM_AE':
        return LSTM_AE(input_size, hidden_size, 1), Basic_Trainer
    elif model_name == 'LSTM_AE_CLASSIFIER':
        return LSTM_AE_CLASSIFIER(input_size, hidden_size, 1), Classifier_Trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V2':
        return LSTM_AE_CLASSIFIER_V2(input_size, hidden_size, 1), Classifier_Trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V3':
        raise ValueError('Invalid model name')
    else:
        raise ValueError('Invalid model name')