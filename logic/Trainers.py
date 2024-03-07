import torch
import torch.utils.data
import torch.nn.utils as clip
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
from logic.LSTMS import *


class Basic_Trainer:
    def __init__(self):
        self.criterion = torch.nn.MSELoss()

    def train(self, model, train_loader, optimizer, epochs, gradient_clipping, recon_dominance=None):
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
            print(f'epoch: {epoch}, loss: {all_losses[-1]}')
        return {'all_losses' : all_losses}

    def test(self, model, test_loader, recon_dominance=None):
        model.eval() 
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.unsqueeze(-1)
                outputs = model(data)
                loss = self.criterion(outputs, data)
                test_loss += loss.item()
        average_test_loss = test_loss / len(test_loader)
        return {'test_loss' : average_test_loss}


class Classifier_Trainer:
    def __init__(self, input_size=28):
        self.recon_criterion = torch.nn.MSELoss()
        self.classif_criterion = torch.nn.CrossEntropyLoss()
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
            all_accuracies.append(np.mean(epoch_accuracies[:-1]))
            all_losses.append(np.mean(epoch_losses))
            print ("epoch: ", epoch, "loss: ", all_losses[-1], "accuracy: ", all_accuracies[-1])
        return {'all_losses': all_losses, 
                'all_accuracies': all_accuracies}

    def test(self, model, test_loader, recon_dominance=0.5):
        model.eval()
        total_loss = 0.0
        batches_accuracy = []
        with torch.no_grad():
            for images, labels in test_loader:
                batch_size = images.size(0)
                # Remove the extra dimension that represents the number of channels (1 in MNIST grayscale images)
                images = images.squeeze(1)
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
                batch_accuracy = (predicted_batch == labels).sum().item()
                batches_accuracy.append(batch_accuracy)
        total_loss/=len(test_loader)
        accuracy = np.sum(batches_accuracy) / len(test_loader.dataset)
        return {'test_loss' : total_loss, 
                'accuracy' : accuracy}
    
    
class Predictor_Trainer:
    def __init__(self, input_size=1, kfolds=10):
        self.recon_criterion = torch.nn.MSELoss()
        self.pred_criterion = torch.nn.MSELoss()
        self.input_size = input_size

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
                # Compute losses
                recon_loss = self.recon_criterion(reconstructions, sequences.reshape((batch_size, -1, self.input_size)))
                epoch_recon_losses.append(recon_loss.item())        
                pred_loss = self.pred_criterion(predictions.reshape((batch_size, -1,self.input_size)), labels.reshape((batch_size, -1,self.input_size))) 
                epoch_pred_losses.append(pred_loss.item())     
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*pred_loss
                epoch_total_losses.append(loss.item())
                # Backpropagation and optimization
                loss.backward()
                clip.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            recon_losses.append(np.mean(epoch_recon_losses))
            pred_losses.append(np.mean(epoch_pred_losses))
            all_losses.append(np.mean(epoch_total_losses))
            print ("epoch: ", epoch, "loss: ", all_losses[-1])

        return {'all_losses' : all_losses, 
                'recon_losses': recon_losses, 
                'pred_losses' : pred_losses }


    def test(self, model, test_loader, recon_dominance=0.5):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sequences, labels in test_loader:
                batch_size = sequences.size(0)
                reconstructions, predictions = model(sequences)        
                # Flatten reconstructions for loss calculation
                reconstructions = reconstructions.reshape((batch_size,-1, self.input_size))        
                # Compute reconstruction loss
                recon_loss = self.recon_criterion(reconstructions, sequences.reshape((batch_size, -1, self.input_size)))        
                pred_loss = self.pred_criterion(predictions.reshape((batch_size, -1, self.input_size)), labels.reshape((batch_size, -1, self.input_size)))
                loss = (recon_dominance)*recon_loss + (1-recon_dominance)*pred_loss
                total_loss += loss.item()
        total_loss/=len(test_loader)
        return {'test_loss' : total_loss}
    
def should_update_best_model(best_test_loss_dict, test_results_dict, classification):
    if classification:
        cur_acc = test_results_dict['accuracy']
        best_acc = best_test_loss_dict['accuracy']
        return cur_acc > best_acc
    else:
        cur_loss = test_results_dict['test_loss']
        best_loss = best_test_loss_dict['test_loss']
        return cur_loss < best_loss



def kfolds_train(args, data ,folds = 2, tune_hyperparams=False):
    def to_list(x):
        return x if isinstance(x, list) else [x]
    args.hidden_size = to_list(args.hidden_size)
    args.learning_rate = to_list(args.learning_rate)
    args.gradient_clipping = to_list(args.gradient_clipping)
    args.reconstruction_dominance = to_list(args.reconstruction_dominance)
    print ("---Starting KFolds Train--")
    hyperparams = []
    if tune_hyperparams:
        hyperparams = [(i, j, k, l) 
                        for i in args.hidden_size
                        for j in args.learning_rate
                        for k in args.gradient_clipping
                        for l in args.reconstruction_dominance]
        folds = len(hyperparams)
    kf = KFold(n_splits=folds, shuffle=True)
    best_model = None
    best_train_dict = None
    best_test_loss_dict = {'test_loss' : float('inf'),
                           'accuracy' : 0.0}
    best_params_dict = {}
    

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        if tune_hyperparams:
            hidden_size, learning_rate, gradient_clipping, reconstruction_dominance = hyperparams[fold]
        else:
            hidden_size, learning_rate, gradient_clipping, reconstruction_dominance = args.hidden_size[0], args.learning_rate[0], args.gradient_clipping[0], args.reconstruction_dominance[0]
        train_set = torch.utils.data.Subset(data, train_idx)
        test_set = torch.utils.data.Subset(data, test_idx)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
        model, trainer = get_model_and_trainer(args.model, args.input_size, hidden_size)
        optimizer = get_optimizer(args.optimizer, model, learning_rate)
        train_results_dict = trainer.train(model, train_loader, optimizer, args.epochs, gradient_clipping, reconstruction_dominance)
        test_results_dict = trainer.test(model, test_loader, reconstruction_dominance)
        factor = "accuracy" if args.classification else "test_loss"
        if should_update_best_model(best_test_loss_dict, test_results_dict, args.classification):
            best_test_loss_dict = test_results_dict
            best_train_dict = train_results_dict
            best_model = model
            best_params_dict = {
                'hidden_size' : hidden_size, 
                'learning_rate' : learning_rate,
                'gradient_clipping' : gradient_clipping,
                'reconstruction_dominance' : reconstruction_dominance
            }
        print (f"Fold {fold+1} finished. Test {factor}: {best_test_loss_dict[factor]}")
            
    res = {}
    for dc in [best_train_dict, best_test_loss_dict, best_params_dict]:
        res.update(dc)
    return best_model, res



def get_optimizer(optimizer_name, model, learning_rate):
    optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
    }
    if optimizer_name in optimizers:
        return optimizers[optimizer_name](model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')   
    

def get_model_and_trainer(model_name, input_size, hidden_size):

    basic_trainer = Basic_Trainer()
    clas_trainer = Classifier_Trainer(input_size)
    pred_trainer =  Predictor_Trainer(input_size)

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
    elif model_name == 'LSTM_AE_PREDICTOR':
        return LSTM_AE_PREDICTOR(input_size, hidden_size, 1), pred_trainer
    elif model_name == 'LSTM_AE_PREDICTOR_V2':
        return LSTM_AE_PREDICTOR_V2(input_size, hidden_size, 1), pred_trainer
    elif model_name == 'LSTM_AE_PREDICTOR_V3':
        return LSTM_AE_PREDICTOR_V3(input_size, hidden_size, 1), pred_trainer
    else:
        raise ValueError('Invalid model name')
    

