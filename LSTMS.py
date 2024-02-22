import torch
import torch.utils.data
import torch.nn as nn
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

    

'''AE with classification based SOLELY on the original image (single linear layer for classification).
Used as a benchmark for the other models (~90% accuracy on MNIST).'''
class LSTM_AE_CLASSIFIER_V1(nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER_V1, self).__init__()
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
    
class LSTM_AE_CLASSIFIER_V3_Experimental (nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER_V3_Experimental, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layers = layers
        self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, x):
        batch_size = x.size(0)
        ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
        dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
        enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
        dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
        classification = self.classifier(enc[:, -1, :])
        return dec, classification

class LSTM_AE_CLASSIFIER_V3 (nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER_V3, self).__init__()
        self.reconstructor_ae = LSTM_AE(input_size, hidden_size, layers)
        self.classifier_ae = LSTM_AE(input_size, hidden_size, layers)
        self.classifier = nn.Linear(input_size, 10)

    def forward(self, x):
        dec = self.reconstructor_ae(x)
        classification = self.classifier_ae(x)
        classification = self.classifier(classification[:, -1, :])
        return dec, classification
    
class LSTM_AE_CLASSIFIER_V4_Experimental (nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER_V4_Experimental, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layers = layers
        self.encoder_rec = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
        self.encoder_clas = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, x):
        batch_size = x.size(0)
        ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
        dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
        enc_enc, _ = self.encoder_rec(x, (ench0c0[0], ench0c0[1]))
        enc_clas, _ = self.encoder_clas(x, (ench0c0[0], ench0c0[1]))
        dec, _ = self.decoder(enc_enc, (dech0c0[0], dech0c0[1]))
        classification = self.classifier(enc_clas[:, -1, :])
        return dec, classification
    
class LSTM_AE_CLASSIFIER_V4 (nn.Module):
    def __init__(self, input_size=28, hidden_size=16, layers=1):
        super(LSTM_AE_CLASSIFIER_V4, self).__init__()
        self.reconstructor_ae = LSTM_AE(input_size, hidden_size, layers)
        self.classifier_ae = LSTM_AE(input_size, hidden_size, layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 10)
        )

    def forward(self, x):
        dec = self.reconstructor_ae(x)
        classification = self.classifier_ae(x)
        classification = self.classifier(classification[:, -1, :])
        return dec, classification
    


def get_model_and_trainer(model_name, input_size, hidden_size):
    clas_trainer = Classifier_Trainer(nn.MSELoss(), nn.CrossEntropyLoss())
    basic_trainer = Basic_Trainer(nn.MSELoss())
    if model_name == 'LSTM_AE':
        return LSTM_AE(input_size, hidden_size, 1), basic_trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V1':
        return LSTM_AE_CLASSIFIER_V1(input_size, hidden_size, 1), clas_trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V2':
        return LSTM_AE_CLASSIFIER_V2(input_size, hidden_size, 1), clas_trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V3':
        return LSTM_AE_CLASSIFIER_V3(input_size, hidden_size, 1), clas_trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V4':
        return LSTM_AE_CLASSIFIER_V4(input_size, hidden_size, 1), clas_trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V3_Experimental':
        return LSTM_AE_CLASSIFIER_V3_Experimental(input_size, hidden_size, 1), clas_trainer
    elif model_name == 'LSTM_AE_CLASSIFIER_V4_Experimental':
        return LSTM_AE_CLASSIFIER_V4_Experimental(input_size, hidden_size, 1), clas_trainer

    else:
        raise ValueError('Invalid model name')