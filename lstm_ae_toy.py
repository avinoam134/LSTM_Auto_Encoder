import torch
import numpy as np
import matplotlib.pyplot as plt
from logic.Utils import parse_args
from logic.Data_Generators import load_syntethic_data, generate_syntethic_data
from logic.LSTMS import LSTM_AE
from logic.Trainers import Basic_Trainer, kfolds_train


def P1_Q1_plot_signal_vs_time():
    dataset = generate_syntethic_data()
    plt.plot(dataset[0])
    plt.show()


def P1Q2_find_best_hyperparams_and_reconstruct_syntethic_data(args):
    args.model = 'LSTM_AE'
    dataset, testset = load_syntethic_data()
    best_model, _ = kfolds_train(args, dataset, tune_hyperparams=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    test_samples = [next(iter(test_loader))[0].unsqueeze(-1), next(iter(test_loader))[0].unsqueeze(-1)]
    test_samples = torch.tensor(np.array(test_samples))
    test_samples_reconstruction = best_model(test_samples)
    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].set_title(f"Sample {i+1}")
        ax[i].plot(test_samples[i].detach().numpy(), linewidth=2.5)
        ax[i].plot(test_samples_reconstruction[i].detach().numpy(), linewidth=5, alpha=0.5)
        ax[i].legend(['Original', 'Reconstruction'], loc='upper right')
    plt.show()


def main():
    args = parse_args()
    if args.function == 'P1_Q1_plot_signal_vs_time':
        #called by:
        '''
        python3 lstm_ae_toy.py --function P1_Q1_plot_signal_vs_time
        '''
        P1_Q1_plot_signal_vs_time()
    elif args.function == 'P1Q2_find_best_hyperparams_and_reconstruct_syntethic_data':
        #called by:
        '''
        python3 lstm_ae_toy.py --function P1Q2_find_best_hyperparams_and_reconstruct_syntethic_data --model LSTM_AE --input_size 1 --hidden_size 8 16 32 --batch_size 128 --epochs 100 --learning_rate 0.1 0.01 0.001 --gradient_clipping 1 2 5
        '''
        P1Q2_find_best_hyperparams_and_reconstruct_syntethic_data(args)


if __name__ == '__main__':
    main()





    