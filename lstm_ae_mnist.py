import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Utils import split_dataset, parse_args, get_optimizer
import numpy as np

class LSTM_AE_MNIST(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, layers=1):
        super(LSTM_AE_MNIST, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layers = layers
        self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)
        self.classifier = nn.Linear(input_size, 10)

    def forward(self, x):
        batch_size = x.size(0)
        ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
        dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
        enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
        dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
        classification = self.classifier(dec[:, -1, :])
        return dec, classification
    
def train(model, train_loader, recon_criterion, classif_criterion, optimizer, epochs, gradient_clipping):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        for images, labels in train_loader:
            # Reshape images for LSTM input (sequence length, batch size, input size)
            images = images.squeeze(1).permute(0, 2, 1)  # Swap dimensions to match LSTM input format
            reconstructions, classifications = model(images)        
            # Flatten reconstructions for loss calculation
            reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
            # Compute reconstruction loss (MSE)
            recon_loss = recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
            # Compute classification loss (Cross-Entropy)
            classif_loss = classif_criterion(classifications, labels)       
            # Total loss
            loss = recon_loss + classif_loss
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, recon_criterion, classif_criterion):
    model.eval()
    total_loss = 0.0
    batches_accuracy = []
    with torch.no_grad():
        for images, labels in test_loader:
            # Reshape images for LSTM input (sequence length, batch size, input size)
            images = images.squeeze(1).permute(0, 2, 1)# Swap dimensions to match LSTM input format
            reconstructions, classifications = model(images)        
            # Flatten reconstructions for loss calculation
            reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
            # Compute reconstruction loss (MSE)
            recon_loss = recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
            # Compute classification loss (Cross-Entropy)
            classif_loss = classif_criterion(classifications, labels)       
            # Total loss
            loss = recon_loss + classif_loss
            total_loss += loss.item()
            # Predicted digits
            _, predicted_batch = torch.max(classifications, 1)
            batch_accuracy = (predicted_batch == labels).sum().item() / len(labels)
            batches_accuracy.append(batch_accuracy)
    #save the model:
    torch.save(model, 'lstm_ae_mnist_model.pth')
    total_loss/=len(test_loader)
    accuracy = np.mean(np.array(batches_accuracy))
    return total_loss, accuracy


def get_train_test_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    args = parse_args()
    model = LSTM_AE_MNIST(args.input_size, args.hidden_size)
    train_loader, test_loader = get_train_test_loaders(args.batch_size)
    recon_criterion = nn.MSELoss()
    classif_criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    train(model, train_loader, recon_criterion, classif_criterion, optimizer, args.epochs, args.gradient_clipping)
    accuracy, _ = test(model, test_loader, recon_criterion, classif_criterion)
    print(accuracy)
    args = parse_args()

if __name__ == '__main__':
    main()





