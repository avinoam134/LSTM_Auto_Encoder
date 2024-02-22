# #A general MNIST classifier CNN based model to incorporate in some versions of the LSTM_AE_MNIST model:
# class CNN_MNIST(nn.Module):
#     def __init__(self):
#         super(CNN_MNIST, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 7 * 7, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# #AE with classification based SOLELY on the original image (single linear layer for classification)
# class LSTM_AE_MNIST(nn.Module):
#     def __init__(self, input_size=1, hidden_size=16, layers=1):
#         super(LSTM_AE_MNIST, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.layers = layers
#         self.classifier = nn.Linear(input_size**2, 10)
#         self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)
        

#     def forward(self, x):
#         batch_size = x.size(0)
#         ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
#         dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
#         enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
#         dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
#         classification = self.classifier(x.reshape(-1, self.input_size**2))
#         return dec, classification
    
    

# #AE with classification based SOLELY on the reconstructed/decoded image (single linear layer for classification)
# class LSTM_AE_MNIST_V2(nn.Module):
#     def __init__(self, input_size=1, hidden_size=16, layers=1):
#         super(LSTM_AE_MNIST_V2, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.layers = layers
#         self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)
#         self.classifier = nn.Linear(input_size, 10)

#     def forward(self, x):
#         batch_size = x.size(0)
#         ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
#         dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
#         enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
#         dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
#         classification = self.classifier(dec[:, -1, :])
#         return dec, classification
    

# #AE with classification based SOLELY on the encoded image (single linear layer for classification)
# class LSTM_AE_MNIST_V3(nn.Module):
#     def __init__(self, input_size=1, hidden_size=16, layers=1):
#         super(LSTM_AE_MNIST_V3, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.layers = layers
#         self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)
#         self.classifier = nn.Linear(hidden_size, 10)

#     def forward(self, x):
#         batch_size = x.size(0)
#         ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
#         dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
#         enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
#         dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
#         classification = self.classifier(enc[:, -1, :])
#         return dec, classification

    
# def train(model, train_loader, recon_criterion, classif_criterion, optimizer, epochs, gradient_clipping, recon_dominance=0.5):
#     model.train()
#     total_loss = 0.0
#     for epoch in range(epochs):
#         for images, labels in train_loader:
#             # Remove the extra dimension that represents the number of channels (1 in MNIST grayscale images)
#             images = images.squeeze(1) 
#             reconstructions, classifications = model(images)        
#             # Flatten reconstructions for loss calculation
#             reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
#             # Compute reconstruction loss (MSE)
#             recon_loss = recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
#             # Compute classification loss (Cross-Entropy)
#             classif_loss = classif_criterion(classifications, labels)       
#             # Total loss
#             loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
#             # Backpropagation and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
#             optimizer.step()
#             total_loss += loss.item()
#     return total_loss / len(train_loader)

# def test(model, test_loader, recon_criterion, classif_criterion, recon_dominance=0.5):
#     model.eval()
#     total_loss = 0.0
#     batches_accuracy = []
#     with torch.no_grad():
#         for images, labels in test_loader:
#             # Reshape images for LSTM input (sequence length, batch size, input size)
#             images = images.squeeze(1)# Swap dimensions to match LSTM input format
#             reconstructions, classifications = model(images)        
#             # Flatten reconstructions for loss calculation
#             reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
#             # Compute reconstruction loss (MSE)
#             recon_loss = recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
#             # Compute classification loss (Cross-Entropy)
#             classif_loss = classif_criterion(classifications, labels)       
#             # Total loss
#             loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
#             total_loss += loss.item()
#             # Predicted digits
#             predicted_batch = torch.argmax(classifications, 1)
#             batch_accuracy = (predicted_batch == labels).sum().item() / len(labels)
#             batches_accuracy.append(batch_accuracy)
#     #save the model:
#     torch.save(model, 'lstm_ae_mnist_model.pth')
#     total_loss/=len(test_loader)
#     accuracy = np.mean(np.array(batches_accuracy))
#     return total_loss, accuracy

import torch
from torch import nn

class LSTM_ETH(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 64, num_layers: int = 1, output_size: int = 10):
        super(LSTM_ETH, self).__init__()
        self.main = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hn, cn) = self.main(x, None)
        result = self.fc(output[:, -1, :])
        return result


if __name__ == '__main__':
    print(LSTM()(torch.randn(10, 3, 28)))