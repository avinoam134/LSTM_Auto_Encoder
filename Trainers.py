import torch
import torch.utils.data
import torch.nn.utils as clip
import numpy as np


class Basic_Trainer:
    def __init__(self, criterion):
        self.criterion = criterion

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping, _):
        model.train()
        all_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for batch, _ in train_loader:
                batch = batch.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = self.criterion(outputs, batch)
                epoch_losses.append(loss.item())
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            all_losses.append(sum(epoch_losses)/len(epoch_losses))
        return all_losses

    def test(self, model, test_loader):
        model.eval() 
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.unsqueeze(-1)
                outputs = model(data)
                loss = self.criterion(outputs, data)
                test_loss += loss.item()
        average_test_loss = test_loss / len(test_loader)
        return average_test_loss


class Classifier_Trainer:
    def __init__(self, recon_criterion, classif_criterion):
        self.recon_criterion = recon_criterion
        self.classif_criterion = classif_criterion

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping, recon_dominance):
        all_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for images, labels in train_loader:
                model.train()
                optimizer.zero_grad()
                # Remove the extra dimension that represents the number of channels (1 in MNIST grayscale images)
                images = images.squeeze(1) 
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = self.classif_criterion(classifications, labels)       
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
                epoch_losses.append(loss.item())
                # Backpropagation and optimization
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                #print(f'epoch:{epoch}, itertaion: {}')
            all_losses.append(epoch_losses)
        return all_losses

    def test(self, model, test_loader, recon_dominance):
        model.eval()
        total_loss = 0.0
        batches_accuracy = []
        with torch.no_grad():
            for images, labels in test_loader:
                # Reshape images for LSTM input (sequence length, batch size, input size)
                images = images.squeeze(1)# Swap dimensions to match LSTM input format
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = self.classif_criterion(classifications, labels)       
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
                total_loss += loss.item()
                # Predicted digits
                predicted_batch = torch.argmax(classifications, 1)
                batch_accuracy = (predicted_batch == labels).sum().item() / len(labels)
                batches_accuracy.append(batch_accuracy)
        total_loss/=len(test_loader)
        accuracy = np.mean(np.array(batches_accuracy))
        return total_loss, accuracy
    
class Classifier_Trainer_With_Epochs_List:
    def __init__(self, recon_criterion, classif_criterion):
        self.recon_criterion = recon_criterion
        self.classif_criterion = classif_criterion

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping, recon_dominance):
        all_losses = []
        all_accuracies = []
        for epoch in range(epochs):
            epoch_accuracy = 0
            epoch_losses = []
            for images, labels in train_loader:
                model.train()
                optimizer.zero_grad()
                # Remove the extra dimension that represents the number of channels (1 in MNIST grayscale images)
                images = images.squeeze(1) 
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = self.classif_criterion(classifications, labels)       
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
                epoch_losses.append(loss.item())
                # Backpropagation and optimization
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                #print(f'epoch:{epoch}, itertaion: {}')
            all_losses.append(epoch_losses.mean())
        return all_losses

    def test(self, model, test_loader, recon_dominance):
        model.eval()
        total_loss = 0.0
        batches_accuracy = []
        with torch.no_grad():
            for images, labels in test_loader:
                # Reshape images for LSTM input (sequence length, batch size, input size)
                images = images.squeeze(1)# Swap dimensions to match LSTM input format
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = self.classif_criterion(classifications, labels)       
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
                total_loss += loss.item()
                # Predicted digits
                predicted_batch = torch.argmax(classifications, 1)
                batch_accuracy = (predicted_batch == labels).sum().item() / len(labels)
                batches_accuracy.append(batch_accuracy)
        total_loss/=len(test_loader)
        accuracy = np.mean(np.array(batches_accuracy))
        return total_loss, accuracy