import torch
import sys
from torchvision import datasets, transforms
import torch.utils.data
import torch.nn.utils as clip
import numpy as np
import copy
#from sklearn.model_selection import KFold, TimeSeriesSplit


class Basic_Trainer:
    def __init__(self, criterion):
        self.criterion = criterion

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping):
        model.train()
        all_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_loader:
                batch = batch.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = self.criterion(outputs, batch)
                epoch_losses.append(loss.item())
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            all_losses.append(np.mean(epoch_losses))
        return np.array(all_losses), np.zeros(epochs)

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
    def __init__(self, recon_criterion, classif_criterion, input_size=28):
        self.recon_criterion = recon_criterion
        self.classif_criterion = classif_criterion
        self.input_size = input_size

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping, recon_dominance):
        all_losses = []
        all_accuracies = []
        for epoch in range(epochs):
            epoch_accuracies = []
            epoch_losses = []
            for images, labels in train_loader:
                batch_size = images.size(0)
                model.train()
                optimizer.zero_grad()
                # Remove the extra dimension that represents the number of channels (1 in MNIST grayscale images)
                images = images.squeeze(1)
                #print to stderr the shape of the images:
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((batch_size, -1, self.input_size))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, images.reshape((batch_size, -1, self.input_size)))        
                # Compute classification loss (Cross-Entropy)
                classif_loss = self.classif_criterion(classifications, labels)       
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*classif_loss
                #save losses and accuracies
                epoch_losses.append(loss.item())
                batch_accuracy = (torch.argmax(classifications, 1) == labels).sum().item() / len(labels) 
                epoch_accuracies.append(batch_accuracy) 
                # Backpropagation and optimization
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                #print(f'epoch:{epoch}, itertaion: {}')
            all_accuracies.append(np.mean(np.array(epoch_accuracies)))
            all_losses.append(np.mean(np.array(epoch_losses)))
        return all_losses, all_accuracies

    def test(self, model, test_loader, recon_dominance):
        model.eval()
        total_loss = 0.0
        batches_accuracy = []
        with torch.no_grad():
            for images, labels in test_loader:
                batch_size = images.size(0)
                # Reshape images for LSTM input (sequence length, batch size, input size)
                images = images.squeeze(1)# Swap dimensions to match LSTM input format
                reconstructions, classifications = model(images)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((batch_size,-1, self.input_size))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, images.reshape((batch_size,-1, self.input_size)))        
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
    

# class Predictor_Trainer:
#     def __init__(self, input_size=1, kfolds=10):
#         self.recon_criterion = torch.nn.MSELoss()
#         self.pred_criterion = torch.nn.MSELoss()
#         self.input_size = input_size
#         #self.kf = KFold(n_splits=kfolds, shuffle=True)

#     def train(self, model, train_loader, optimizer, epochs, gradient_clipping, recon_dominance):
#         recon_losses = []
#         pred_losses = []
#         all_losses = []
#         for epoch in range(epochs):
#             epoch_recon_losses = []
#             epoch_pred_losses = []
#             epoch_total_losses = []
#             for sequences, labels in train_loader:
#                 batch_size = sequences.size(0)
#                 model.train()
#                 optimizer.zero_grad()
#                 reconstructions, predictions = model(sequences)        
#                 # Flatten reconstructions for loss calculation
#                 reconstructions = reconstructions.reshape((batch_size, -1,self.input_size))        
#                 # Compute reconstruction loss (MSE)
#                 recon_loss = self.recon_criterion(reconstructions, sequences.reshape((batch_size, -1, self.input_size)))
#                 epoch_recon_losses.append(recon_loss.item())        
#                 # Compute classification loss (Cross-Entropy)
#                 pred_loss = self.pred_criterion(predictions, labels) 
#                 epoch_pred_losses.append(pred_loss.item())     
#                 # Total loss
#                 loss = (recon_dominance)*recon_loss + (1-recon_dominance)*pred_loss
#                 epoch_total_losses.append(loss.item())
#                 # Backpropagation and optimization
#                 loss.backward()
#                 clip.clip_grad_norm_(model.parameters(), gradient_clipping)
#                 optimizer.step()
#                 #print(f'epoch:{epoch}, itertaion: {}')
#             recon_losses.append(np.mean(np.array(epoch_recon_losses)))
#             pred_losses.append(np.mean(np.array(epoch_pred_losses)))
#             all_losses.append(np.mean(np.array(epoch_total_losses)))
#         return all_losses, recon_losses, pred_losses


#     def test(self, model, test_loader, recon_dominance):
#         model.eval()
#         total_loss = 0.0
#         pred_acc_precentile = []
#         with torch.no_grad():
#             for sequences, labels in test_loader:
#                 batch_size = sequences.size(0)
#                 reconstructions, predictions = model(sequences)        
#                 # Flatten reconstructions for loss calculation
#                 reconstructions = reconstructions.reshape((batch_size,-1, self.input_size))        
#                 # Compute reconstruction loss (MSE)
#                 recon_loss = self.recon_criterion(reconstructions, sequences.reshape((batch_size, -1, self.input_size)))        
#                 # Compute classification loss (Cross-Entropy)
#                 pred_loss = self.pred_criterion(predictions, labels)
#                 cur_acc_percentile = 100*abs(predictions-labels)/labels
#                 pred_acc_precentile.append(cur_acc_percentile)       
#                 # Total loss
#                 loss = (recon_dominance)*recon_loss + (1-recon_dominance)*pred_loss
#                 total_loss += loss.item()
#         total_loss/=len(test_loader)
#         total_percentile = np.mean(np.array(pred_acc_precentile))
#         return total_loss, total_percentile
    
#     def kfolds_train(self, model, kf ,train_data, optimizer, epochs, gradient_clipping, recon_dominance, batch_size=128):
#         best_model = None
#         best_test_loss = float('inf')
#         best_train_loss = float('inf')
#         best_test_losses = None
#         for train_idx, test_idx in kf.split(train_data):
#             train_set = torch.utils.data.Subset(train_data, train_idx)
#             test_set = torch.utils.data.Subset(train_data, test_idx)
#             train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#             test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
#             cur_model = copy.deepcopy(model)
#             cur_optimizer = copy.deepcopy(optimizer)
#             train_losses, recon_losses, pred_losses = self.train(cur_model, train_loader, cur_optimizer, epochs, gradient_clipping, recon_dominance)
#             test_loss, test_loss_precentile = self.test(model, test_loader, recon_dominance)
#             if test_loss < best_test_loss:
#                 best_test_loss = test_loss
#                 best_test_losses = (test_loss, test_loss_precentile)
#                 best_train_losses = (train_losses, recon_losses, pred_losses)
#                 best_model = cur_model
#             #todo: save all the train losseses
#             #make a percentile loss for test and save it aswell
#         return best_model, best_test_losses, best_train_losses 


class Predictor_Trainer:
    def __init__(self, input_size=1, kfolds=10):
        self.recon_criterion = torch.nn.MSELoss()
        self.pred_criterion = torch.nn.MSELoss()
        self.input_size = input_size
        #self.kf = KFold(n_splits=kfolds, shuffle=True)

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping, recon_dominance):
        recon_losses = []
        pred_losses = []
        all_losses = []
        for epoch in range(epochs):
            epoch_recon_losses = []
            epoch_pred_losses = []
            epoch_total_losses = []
            for sequences, labels in train_loader:
                batch_size = sequences.size(0)
                model.train()
                optimizer.zero_grad()
                reconstructions, predictions = model(sequences)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((batch_size, -1,self.input_size))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, sequences.reshape((batch_size, -1, self.input_size)))
                epoch_recon_losses.append(recon_loss.item())        
                # Compute classification loss (Cross-Entropy)
                pred_loss = self.pred_criterion(predictions.reshape((batch_size, -1,self.input_size)), labels.reshape((batch_size, -1,self.input_size))) 
                epoch_pred_losses.append(pred_loss.item())     
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*pred_loss
                epoch_total_losses.append(loss.item())
                # Backpropagation and optimization
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                #print(f'epoch:{epoch}, itertaion: {}')
            recon_losses.append(np.mean(np.array(epoch_recon_losses)))
            pred_losses.append(np.mean(np.array(epoch_pred_losses)))
            all_losses.append(np.mean(np.array(epoch_total_losses)))
            print ("epoch: ", epoch, "loss: ", all_losses[-1])
        return all_losses, recon_losses, pred_losses


    def test(self, model, test_loader, recon_dominance):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sequences, labels in test_loader:
                batch_size = sequences.size(0)
                reconstructions, predictions = model(sequences)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((batch_size,-1, self.input_size))        
                # Compute reconstruction loss (MSE)
                recon_loss = self.recon_criterion(reconstructions, sequences.reshape((batch_size, -1, self.input_size)))        
                # Compute classification loss (Cross-Entropy)
                pred_loss = self.pred_criterion(predictions.reshape((batch_size, -1, self.input_size)), labels.reshape((batch_size, -1, self.input_size)))
                # Total loss
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*pred_loss
                total_loss += loss.item()
        total_loss/=len(test_loader)
        return total_loss
    
    def kfolds_train(self, model, kf ,train_data, optimizer, epochs, gradient_clipping, recon_dominance, batch_size=128):
        best_model = None
        best_test_loss = float('inf')
        for fold, (train_idx, test_idx) in enumerate(kf.split(train_data)):
            print (f"Fold {fold} started.")
            train_set = torch.utils.data.Subset(train_data, train_idx)
            test_set = torch.utils.data.Subset(train_data, test_idx)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
            self.recon_criterion = torch.nn.MSELoss()
            self.pred_criterion = torch.nn.MSELoss()
            cur_model = copy.deepcopy(model)
            cur_optimizer = copy.deepcopy(optimizer)
            train_losses, recon_losses, pred_losses = self.train(cur_model, train_loader, cur_optimizer, epochs, gradient_clipping, recon_dominance)
            test_loss = self.test(cur_model, test_loader, recon_dominance)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_train_losses = (train_losses, recon_losses, pred_losses)
                best_model = cur_model
        return best_model, best_test_loss, best_train_losses 

