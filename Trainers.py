import torch
import torch.utils.data
import torch.nn as nn
import numpy as np


class Basic_Trainer:
    def train(model, criterion, optimizer, train_loader, epochs, gradient_clipping):
        model.train()
        all_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_loader:
                batch = batch.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                epoch_losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            all_losses.append(sum(epoch_losses)/len(epoch_losses))
        return all_losses

    def test(model, test_loader, criterion):
        model.eval() 
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.unsqueeze(-1)
                outputs = model(data)
                loss = criterion(outputs, data)
                test_loss += loss.item()
        average_test_loss = test_loss / len(test_loader)
        return average_test_loss


class Classifier_Trainer:
    def train(model, train_loader, recon_criterion, classif_criterion, optimizer, epochs, gradient_clipping, recon_dominance):
        model.train()
        all_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for images, labels in train_loader:
                # Remove the extra dimension that represents the number of channels (1 in MNIST grayscale images)
                images = images.squeeze(1) 
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((-1, reconstructions.size(-1)))        
                # Compute reconstruction loss (MSE)
                recon_loss = recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = classif_criterion(classifications, labels)       
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
                epoch_losses.append(loss.item())
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            all_losses.append(epoch_losses)
        return all_losses

    def test(model, test_loader, recon_criterion, classif_criterion, recon_dominance):
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
                recon_loss = recon_criterion(reconstructions, images.reshape((-1, images.size(-1))))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = classif_criterion(classifications, labels)       
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