import lstm_ae_toy as p1
import matplotlib.pyplot as plt
import subprocess


def P1_Q1_plot_signal_vs_time():
    dataset = p1.generate_syntethic_data()
    plt.plot(dataset[0])
    plt.show()


def find_best_perfoming_hyperparameters():
    learning_rates = [0.001, 0.01, 0.1]
    hidden_state_sizes = [16, 32, 64, 128]
    gradient_clipping = [1, 5, 10]
    epochs = 1000
    best_loss = float('inf')
    best_hyperparameters = None
    for learning_rate in learning_rates:
        for hidden_state_size in hidden_state_sizes:
            for gradient_clip in gradient_clipping:
                #run lst_ae_toy.py script with the hyperparameters as args using os.subprocess:
                result = subprocess.run(['python3', 'lstm_ae_toy.py',
                                        '--optimiser', 'Adam',
                                        '--learning_rate', str(learning_rate),
                                        '--hidden_state_size', str(hidden_state_size),
                                        '--epochs', str(epochs),
                                        '--gradient_clipping', str(gradient_clip)],
                                        text=True, capture_output=True)
                print (result.stderr)
                curr_loss = float(result.stdout)
                
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_hyperparameters = (learning_rate, hidden_state_size, gradient_clip)
    return best_hyperparameters

def Q2_select_hyperparameters_and_train_model():
    hyperparams = find_best_perfoming_hyperparameters()
    train_loader, test_loader, val_loader = p1.load_syntethic_data(128)
    model = p1.LSTM_AE(10, hyperparams[1])
    p1.train(model, p1.nn.MSELoss(), p1.get_optimizer('Adam', model, hyperparams[0]),
            train_loader, 1000, hyperparams[2])
    #make test_samples contain 2 single samples from the test_loader:
    test_samples = [next(iter(test_loader))[0], next(iter(test_loader))[0]]
    #make test_samples_reconstruction contain the reconstruction of the samples in test_samples:
    test_samples_reconstruction = model(test_samples)
    #plot the original and reconstructed samples:
    plt.plot(test_samples[0].detach().numpy())
    plt.plot(test_samples_reconstruction[0].detach().numpy())
    plt.show()

def main():
    Q2_select_hyperparameters_and_train_model()

if __name__ == '__main__':
    main()









