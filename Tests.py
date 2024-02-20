from lstm_ae_toy import LSTM_AE_TOY, generate_syntethic_data, load_syntethic_data
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import torch
import os


def P1_Q1_plot_signal_vs_time():
    dataset = generate_syntethic_data()
    plt.plot(dataset[0])
    plt.show()


def find_best_perfoming_model(dry_run=False):
    if dry_run and os.path.exists('lstm_ae_toy_model.pth'):
        return torch.load('lstm_ae_toy_model.pth').eval()
    learning_rates = [0.001, 0.01, 0.1]
    hidden_state_sizes = [8, 16, 32]
    gradient_clipping = [1, 5]
    epochs = 100
    best_params = [0,0,0,0]
    best_loss = float('inf')
    best_model = None
    for i,learning_rate in enumerate(learning_rates):
        for j,hidden_state_size in enumerate(hidden_state_sizes):
            for k,gradient_clip in enumerate(gradient_clipping):
                result = subprocess.run(['python3', 'lstm_ae_toy.py',
                                         '--input_size', '1',
                                        '--optimizer', 'Adam',
                                        '--learning_rate', str(learning_rate),
                                        '--hidden_size', str(hidden_state_size),
                                        '--epochs', str(epochs),
                                        '--gradient_clipping', str(gradient_clip)],
                                        text=True, capture_output=True)
                curr_loss = float(result.stdout)
                print (f'iteration {i}.{j}.{k}:\n learning_rate: {learning_rate}, hidden_state_size: {hidden_state_size}, gradient_clip: {gradient_clip}, final_loss: {curr_loss}')
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_params = (learning_rate, hidden_state_size, gradient_clip)
                    best_model = torch.load('lstm_ae_toy_model.pth').eval()

    print (f'grid search done. best validation loss: {best_loss}. learning_rate: {best_params[0]}, hidden_state_size: {best_params[1]}, gradient_clip: {best_params[2]}')
    torch.save(best_model, 'lstm_ae_toy_model.pth')
    return best_model

def Q2_select_hyperparameters_and_train_model():
    model = find_best_perfoming_model(dry_run=True)
    _, test_loader, _ = load_syntethic_data(128)
    #make test_samples contain 2 single samples from the test_loader:
    test_samples = [next(iter(test_loader))[0].unsqueeze(-1), next(iter(test_loader))[0].unsqueeze(-1)]
    #make test samples as a tensor:
    test_samples = torch.tensor(np.array(test_samples))
    #make test_samples_reconstruction contain the reconstruction of the samples in test_samples:
    test_samples_reconstruction = model(test_samples)
    #plot the original and reconstructed samples on the same plot (side by side):
    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].set_title(f"Sample {i+1}")
        ax[i].plot(test_samples[i].detach().numpy(), linewidth=2.5)
        ax[i].plot(test_samples_reconstruction[i].detach().numpy(), linewidth=5, alpha=0.5)
        ax[i].legend(['Original', 'Reconstruction'], loc='upper right')
    plt.show()


def main():
    Q2_select_hyperparameters_and_train_model()

if __name__ == '__main__':
    main()









