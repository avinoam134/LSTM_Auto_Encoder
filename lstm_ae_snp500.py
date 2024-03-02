import torch
import numpy as np
from Utils import parse_args, get_optimizer, save_script_out_to_json, os
from Data_Generators import load_snp_data, load_snp_data_for_cross_validation, load_snp_data_with_labels_for_cross_validation, load_snp_data_with_labels_for_kfolds
from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER_V1, LSTM_AE_CLASSIFIER_V2, LSTM_AE_CLASSIFIER_V3, LSTM_AE_CLASSIFIER_V4, LSTM_AE_PREDICTOR, LSTM_AE_PREDICTOR_V2, LSTM_AE_PREDICTOR_V3 ,get_model_and_trainer
from Trainers import Basic_Trainer, Predictor_Trainer
from sklearn.model_selection import KFold, TimeSeriesSplit


def find_best_reconstruction_model():
    input_size = 1
    hidden_sizes = [8, 16, 32]
    layers = 1
    epochs = 10000
    learning_rates = [0.1, 0.01, 0.001]
    gradient_clipping = [1,5]
    trainer = Basic_Trainer(torch.nn.MSELoss())
    data_loaders = load_snp_data_for_cross_validation()
    best_loss = float('inf')
    best_params = None
    best_model = None
    test_data = None
    #create an iterator for data_loaders:
    data_loaders_iter = iter(data_loaders)
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for clip in gradient_clipping:
                model = LSTM_AE(input_size, hidden_size, layers)
                train_loader, test_loader, val_loader = next(data_loaders_iter)
                optimizer = get_optimizer('Adam', model, learning_rate)
                trainer.train(model, train_loader, optimizer, epochs, clip)
                cur_loss = trainer.test(model, val_loader)
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_params = (hidden_size, learning_rate, clip)
                    best_model = model
                    test_data = test_loader
    torch.save(best_model, 'lstm_ae_snp500_model.pth')
    save_script_out_to_json({'best_params': best_params,
                             'test_loader' : test_data}, 'scripts_out.json')
    
def find_best_prediction_model():
    input_size = 1
    hidden_sizes = [8, 16, 32]
    layers = 1
    epochs = 10000
    learning_rates = [0.1, 0.01, 0.001]
    gradient_clipping = [1,5]
    trainer = Predictor_Trainer(torch.nn.MSELoss(), torch.nn.MSELoss())
    data_loaders = load_snp_data_with_labels_for_cross_validation()
    best_loss = float('inf')
    best_params = None
    best_model = None
    test_data = None
    #create an iterator for data_loaders:
    data_loaders_iter = iter(data_loaders)
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for clip in gradient_clipping:
                model = LSTM_AE_PREDICTOR(input_size, hidden_size, layers)
                train_loader, test_loader, val_loader = next(data_loaders_iter)
                optimizer = get_optimizer('Adam', model, learning_rate)
                trainer.train(model, train_loader, optimizer, epochs, clip)
                cur_loss = trainer.test(model, val_loader)
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_params = (hidden_size, learning_rate, clip)
                    best_model = model
                    test_data = test_loader
    final_test_loss = trainer.test(best_model, test_data)
    torch.save(best_model, 'lstm_ae_snp500_model.pth')
    save_script_out_to_json({'best_params': best_params,
                             'test_loss' : final_test_loss}, 'scripts_out.json')

def k_folds_find_best_reconstruction_model():
    input_size = 1
    hidden_sizes = [8, 16, 32]
    layers = 3
    epochs = 1000
    learning_rates = [0.1, 0.01, 0.001]
    gradient_clipping = [1,5]
    hyperparams = [(hidden_sizes[i], learning_rates[j], gradient_clipping[k]) for i in range(3) for j in range(3) for k in range(2)]
    trainer = Predictor_Trainer()
    dataset, test_set = load_snp_data_with_labels_for_kfolds()
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
    test_data = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)
    best_loss = float('inf')
    best_params = None
    best_model = None
    best_train_loss = float('inf')
    kf = KFold(n_splits=10, shuffle=True)
    for hidden_size, learning_rate, clip in hyperparams:
        model = LSTM_AE_PREDICTOR_V2(input_size, hidden_size, layers)
        optimizer = get_optimizer('Adam', model, learning_rate)
        model, test_losses, train_losses = trainer.kfolds_train(model, kf, dataset, optimizer, epochs, clip, 0.5)
        if test_losses[0] < best_loss:
            best_loss = test_losses[0]
            best_loss_percentile = test_losses[1]
            best_params = (hidden_size, learning_rate, clip)
            best_model = model
            best_train_losses = train_losses
    final_test_loss = trainer.test(best_model, test_data)
    torch.save(best_model, 'lstm_ae_snp500_model.pth')
    save_script_out_to_json({'best_params': best_params,
                            'fold_test_loss' : best_loss,
                            'fold_test_loss_percentile' : best_loss_percentile,
                            'final_test_loss' : final_test_loss[0],
                            'final_test_loss_percentile' : final_test_loss[1],
                            'train_loss' : best_train_loss}, 'scripts_out.json')
    

def kfolds_train(trainer, model_args, kf ,train_data, optimizer_args, epochs, gradient_clipping, recon_dominance, batch_size=128):
    best_model = None
    best_test_loss = float('inf')
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_data)):
        print (f"Fold {fold} started.")
        train_set = torch.utils.data.Subset(train_data, train_idx)
        test_set = torch.utils.data.Subset(train_data, test_idx)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        trainer.recon_criterion = torch.nn.MSELoss()
        trainer.pred_criterion = torch.nn.MSELoss()
        # cur_model = copy.deepcopy(model)
        # cur_optimizer = copy.deepcopy(optimizer)
        cur_model = LSTM_AE_PREDICTOR_V3(*model_args)
        optim_name, learning_rate = optimizer_args
        cur_optimizer = get_optimizer(optim_name, cur_model, learning_rate)
        train_losses, recon_losses, pred_losses = trainer.train(cur_model, train_loader, cur_optimizer, epochs, gradient_clipping, recon_dominance)
        test_loss = trainer.test(cur_model, test_loader, recon_dominance)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_losses = (train_losses, recon_losses, pred_losses)
            best_model = cur_model
    return best_model, best_test_loss, best_train_losses 

def k_folds_train_predictor_model():
    hidden_size = 16
    input_size = 1
    layers = 1
    learning_rate = 0.001
    clip = 5
    epochs = 20
    trainer = Predictor_Trainer()
    dataset, test_set = load_snp_data_with_labels_for_kfolds()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100000, shuffle = False)
    kf = KFold(n_splits=2, shuffle=True)
    # model = LSTM_AE_PREDICTOR_V2(input_size, hidden_size, layers)
    # optimizer = get_optimizer('Adam', model, learning_rate)
    model_args = (input_size, hidden_size, layers)
    optimizer_args = ('Adam', learning_rate)
    # dl = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = True)
    # trainer.train(model, dl, optimizer, epochs, clip, 0.5)
    model, test_loss, train_losses = kfolds_train(trainer, model_args, kf, dataset, optimizer_args, epochs, clip, 0.5)
    print ("im here1")
    final_test_loss = trainer.test(model, test_loader, 0.5)
    print ("im here2")
    torch.save(model, 'lstm_ae_snp500_model.pth')
    print ("im here3")
    save_script_out_to_json({'final_test_loss' : final_test_loss,
                             'folds_test_loss' : test_loss,
                            'train_loss' : train_losses}, 'scripts_out.json')
    print ("im here4")
    torch.save(test_set, 'scripts_test_data.pt')
    print ("im here5")

    


def main():
    k_folds_train_predictor_model()
    # args = parse_args()
    # if args.function == 'find_best_reconstruction_model':
    #     find_best_reconstruction_model()
    # elif args.function == 'find_best_prediction_model':
    #     find_best_prediction_model()
    # elif args.function == 'k_folds_find_best_reconstruction_model':
    #     k_folds_find_best_reconstruction_model()
    # elif args.function == 'k_folds_train_predictor_model':
    #     k_folds_train_predictor_model()
    # else:
    #     raise ValueError('Invalid function')


if __name__ == '__main__':
    main()

