import torch
import torch.utils.data
import torch.nn as nn
from copy import deepcopy

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
        x = x.reshape(x.shape[0], -1, self.reconstructor_ae.input_size)
        dec = self.reconstructor_ae(x)
        classification = self.classifier_ae(x)
        classification = self.classifier(classification[:, -1, :])
        return dec, classification
    

'''A predictor from a sequence of days to the next day'''    
class LSTM_AE_PREDICTOR (nn.Module):
    def __init__(self, input_size=1, hidden_size=16, layers=1):
        super(LSTM_AE_PREDICTOR, self).__init__()
        self.reconstructor_ae = LSTM_AE(input_size, hidden_size, layers)
        self.prediction_ae = LSTM_AE(input_size, hidden_size, layers)
        self.predictior = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1)
        )
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.reconstructor_ae.input_size)
        dec = self.reconstructor_ae(x)
        prediction = self.prediction_ae(x)
        prediction = self.predictor(prediction[:, -1, :])
        return dec, prediction
    
'''A predictor from a sequence of days TO ANOTHER SEQUENCE of the input days shifted by 1,
but prediction is initialised with the real value of the shared days with the input sequence'''
class LSTM_AE_PREDICTOR_V2 (nn.Module):
    def __init__(self, input_size=1, hidden_size=16, layers=1):
        super(LSTM_AE_PREDICTOR_V2, self).__init__()
        self.reconstructor_ae = LSTM_AE(input_size, hidden_size, layers)
        self.prediction_ae = LSTM_AE(input_size, hidden_size, layers)
        

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.reconstructor_ae.input_size)
        dec = self.reconstructor_ae(x)
        y = deepcopy(x)
        y[:, -1, :] = 0
        prediction = self.prediction_ae(y)
        return dec, prediction
    
'''A predictor from a sequence of days TO ANOTHER SEQUENCE of the input days shifted by 1, only '''
class LSTM_AE_PREDICTOR_V3 (nn.Module):
    def __init__(self, input_size=1, hidden_size=16, layers=1):
        super(LSTM_AE_PREDICTOR_V3, self).__init__()
        self.reconstructor_ae = LSTM_AE(input_size, hidden_size, layers)
        self.prediction_ae = LSTM_AE(input_size, hidden_size, layers)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.reconstructor_ae.input_size)
        dec = self.reconstructor_ae(x)
        prediction = self.prediction_ae(x)
        return dec, prediction


