import torch
import numpy as np
from Utils import parse_args, get_optimizer, save_script_out_to_json, os
from Data_Generators import load_snp_data, load_snp_data_for_cross_validation
from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER_V1, LSTM_AE_CLASSIFIER_V2, LSTM_AE_CLASSIFIER_V3, LSTM_AE_CLASSIFIER_V4, get_model_and_trainer
from Trainers import Basic_Trainer


def find_best_reconstruction_model():
    input_size = 1
    hidden_sizes = [8, 16, 32]
    layers = 1
    epochs = 10
    learning_rates = [0.1, 0.01, 0.001]
    gradient_clipping = [1,5]
    trainer = Basic_Trainer(torch.nn.MSELoss())
    data_loaders = load_snp_data_for_cross_validation()
    best_loss = float('inf')
    best_params = None
    best_model = None
    #create an iterator for data_loaders:
    data_loaders_iter = iter(data_loaders)
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for clip in gradient_clipping:
                model, _ = get_model_and_trainer('LSTM_AE', input_size, hidden_size, layers)
                train_loader, test_loader, val_loader = next(data_loaders_iter)
                optimizer = get_optimizer('Adam', model, learning_rate)
                trainer.train(model, train_loader, optimizer, epochs, clip)
                cur_loss = trainer.test(model, val_loader)
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_params = (hidden_size, learning_rate, clip)
                    best_model = model
    return best_model, best_params

def main():
    bestmodel, best_params = find_best_reconstruction_model()
    torch.save(bestmodel, 'lstm_ae_snp500_model.pth')
    print(best_params)


if __name__ == '__main__':
    main()

