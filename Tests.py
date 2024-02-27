from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER_V1, LSTM_AE_CLASSIFIER_V2, LSTM_AE_CLASSIFIER_V3, LSTM_AE_CLASSIFIER_V4
from Data_Generators import generate_syntethic_data, load_syntethic_data, load_MNIST_data, generate_snp_company_with_dates, load_snp_data
from Utils import load_script_out_from_json
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import torch
import os


scripts_out_path = 'scripts_out.json'
toy_model_path = 'lstm_ae_toy_model.pth'
toy_script_path = 'lstm_ae_toy.py'
mnist_model_path =  'lstm_ae_mnist_model.pth'
mnist_script_path = 'lstm_ae_mnist.py'
snp_model_path = 'lstm_ae_snp500_model.pth'
snp_script_path = 'lstm_ae_snp500.py'


def P1_Q1_plot_signal_vs_time():
    dataset = generate_syntethic_data()
    plt.plot(dataset[0])
    plt.show()


def find_best_perfoming_toy_model(dry_run=False):

    if dry_run and os.path.exists(toy_model_path):
        return torch.load(toy_model_path).eval()
    learning_rates = [0.001, 0.01, 0.1]
    hidden_state_sizes = [8, 16, 32]
    gradient_clipping = [1, 5]
    epochs = 20
    best_params = [0,0,0,0]
    model = 'LSTM_AE'
    optimizer = 'Adam'
    best_loss = float('inf')
    best_model = None
    for i,learning_rate in enumerate(learning_rates):
        for j,hidden_state_size in enumerate(hidden_state_sizes):
            for k,gradient_clip in enumerate(gradient_clipping):
                result = subprocess.run(['python3', toy_script_path,
                                         '--input_size', str(1),
                                        '--optimizer', optimizer,
                                        '--learning_rate', str(learning_rate),
                                        '--hidden_size', str(hidden_state_size),
                                        '--epochs', str(epochs),
                                        '--gradient_clipping', str(gradient_clip),
                                        '--model', model],
                                        text=True, capture_output=True)
                print (result.stderr)
                curr_loss = float(result.stdout)
                print (f'iteration {i}.{j}.{k}:\n learning_rate: {learning_rate}, hidden_state_size: {hidden_state_size}, gradient_clip: {gradient_clip}, final_loss: {curr_loss}')
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_params = (learning_rate, hidden_state_size, gradient_clip)
                    best_model = torch.load(toy_model_path).eval()

    print (f'grid search done. best validation loss: {best_loss}. learning_rate: {best_params[0]}, hidden_state_size: {best_params[1]}, gradient_clip: {best_params[2]}')
    torch.save(best_model, toy_model_path)
    return best_model

def P1_Q2_select_hyperparameters_and_train_model():
    model = find_best_perfoming_toy_model(dry_run=False)
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



def find_best_mnist_hyperparams(model = 'LSTM_AE_CLASSIFIER_V3'):
    learning_rates = [0.001, 0.01, 0.1]
    hidden_state_sizes = [8, 16, 32]
    gradient_clipping = [1, 5]
    epochs = 10
    best_params = [0,0,0,0]
    optimizer = 'Adam'
    best_acc = 0
    best_model = None
    for i,learning_rate in enumerate(learning_rates):
        for j,hidden_state_size in enumerate(hidden_state_sizes):
            for k,gradient_clip in enumerate(gradient_clipping):
                result = subprocess.run(['python3', mnist_script_path,
                                         '--input_size', str(28),
                                        '--optimizer', optimizer,
                                        '--learning_rate', str(learning_rate),
                                        '--hidden_size', str(hidden_state_size),
                                        '--epochs', str(epochs),
                                        '--gradient_clipping', str(gradient_clip),
                                        '--model', model],
                                        text=True, capture_output=True)
                cur_acc = float(result.stdout)
                print (f'iteration {i}.{j}.{k}:\n learning_rate: {learning_rate}, hidden_state_size: {hidden_state_size}, gradient_clip: {gradient_clip}, final_accuracy: {cur_acc}')
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    best_params = (learning_rate, hidden_state_size, gradient_clip)
                    best_model = torch.load(mnist_model_path).eval()

    print (f'grid search done. best accuracy: {best_acc}. learning_rate: {best_params[0]}, hidden_state_size: {best_params[1]}, gradient_clip: {best_params[2]}')
    torch.save(best_model, mnist_model_path)
    return best_model


def find_best_reconstruction_to_classification_ratio():
    best_acc = 0
    best_ratio=0
    for ratio in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
        result = subprocess.run(['python3', mnist_script_path,
                    '--input_size', str(28),
                    '--hidden_size', str(8),
                    '--epochs', str(10),
                    '--reconstruction_dominance', str(ratio),
                    '--model', 'LSTM_AE_CLASSIFIER_V3'],
                    text=True, capture_output=True)
        cur_acc = float(result.stdout)
        if cur_acc > best_acc:
            best_ratio = ratio
            best_acc = cur_acc
            best_model = torch.load(mnist_model_path).eval()
    torch.save(best_model, mnist_model_path)
    print (f'grid search done. best ratio: {best_ratio}. accuracy: {best_acc}')
    return best_model, best_acc

def find_best_mnist_model():
    models = [f'LSTM_AE_CLASSIFIER_V{i}' for i in range(1,5)]
    best_acc = 0
    best_model_obj = None
    best_model = ''
    for model in models:
        model_obj, acc = get_best_mnist_model(dry_run=False, model = model)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_obj = model_obj
        print (f'model: {model}. accuracy: {acc}')
    print (f'grid search done. best model: {best_model}. accuracy: {best_acc}')
    return best_model_obj, best_acc


def get_best_mnist_model(input_size = 28, hidden_size = 8,  epochs = 10, learning_rate = 0.01, gradient_clipping = 5, optimizer = 'Adam', reconstruction_dominance = 0.5,
                         batch_size = 128, dry_run=False, model = 'LSTM_AE_CLASSIFIER_V4', classify = True):
    if dry_run and os.path.exists(mnist_model_path):
        return torch.load(mnist_model_path).eval(), 0.982
    if not classify:
        reconstruction_dominance=1
    result = subprocess.run(['python3', mnist_script_path,
                    '--input_size', str(input_size),
                    '--hidden_size', str(hidden_size),
                    '--batch_size', str(batch_size),
                    '--epochs', str(epochs),
                    '--learning_rate', str(learning_rate),
                    '--gradient_clipping', str(gradient_clipping),
                    '--optimizer', optimizer,
                    '--reconstruction_dominance', str(reconstruction_dominance),
                    '--model', model],
                    text=True, capture_output=True)
    print (result.stderr)
    print (result.stdout)
    return torch.load(mnist_model_path).eval(), float(result.stdout)
    

def P2_Q1_reconstruct_mnist_images(model = None):
    if model is None:
        model, _ = get_best_mnist_model(dry_run=True, model = 'LSTM_AE_CLASSIFIER_V1')
    _, test_loader = load_MNIST_data(128)
    #make test_samples contain 2 single samples from the test_loader:
    test_samples, _ = next(iter(test_loader))
    test_samples_squeezed = test_samples.squeeze(1)
    #make test_samples_reconstruction contain the reconstruction of the samples in test_samples:
    test_samples_reconstruction, _ = model(test_samples_squeezed)
    recon = test_samples_reconstruction[:3].detach().numpy().reshape(3,28,28)
    #plot the original and reconstructed samples on the same plot (side by side):
    fig, ax = plt.subplots(2, 3)
    for i in range(3):
        ax[0,i].set_title(f"Sample {i+1}")
        ax[1,i].set_title(f"Reconstruction {i+1}")
        ax[0,i].imshow(test_samples[i].squeeze(0), cmap='gray')
        ax[1,i].imshow(recon[i], cmap='gray')
    plt.show()


def P2Q2_train_and_plot_mnist_classifier_and_reconstructor():
    _, _ = get_best_mnist_model(classify=False)
    reconstructor_dict = load_script_out_from_json(scripts_out_path)
    _, _ = get_best_mnist_model(classify=True)
    classifier_dict = load_script_out_from_json(scripts_out_path)
    fig, ax = plt.subplots(1,2)
    #add a title for the entire plot:
    fig.suptitle('Accuracy: ' + str(format(classifier_dict['accuracies'][-1], ".2f"))  + '         Model: LSTM_AE_CLASSIFIER_V4' , color='red')
    ax[0].set_title('Reconstruction Architechture Loss')
    ax[1].set_title('Classification Architecture Accuracy & Loss')
    ax[0].plot(reconstructor_dict['losses'])
    ax[1].plot(classifier_dict['losses'])
    ax[1].plot(classifier_dict['accuracies'])
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    # create a legend for each axis
    ax[0].legend(['Reconstruction Loss'], loc='upper right')
    ax[1].legend(['Classification Loss', 'Classification Accuracy'], loc='center right')
    plt.show()

def P2Q3_reconstruct_and_classify_over_1_input_size():
    model, acc = get_best_mnist_model(input_size=1)
    _, test_loader = load_MNIST_data(128)
    #make test_samples contain 2 single samples from the test_loader:
    test_samples, _ = next(iter(test_loader))
    test_samples_squeezed = test_samples.squeeze(1)
    #make test_samples_reconstruction contain the reconstruction of the samples in test_samples:
    test_samples_reconstruction, _ = model(test_samples_squeezed)
    recon = test_samples_reconstruction[:2].detach().numpy().reshape(2,28,28)
    #plot the original and reconstructed samples on the same plot (side by side):
    fig, ax = plt.subplots(2, 2)
    #add a title for the entire plot:
    fig.suptitle('Accuracy: ' + str(acc))
    for i in range(2):
        ax[0,i].set_title(f"Sample {i+1}")
        ax[1,i].set_title(f"Reconstruction {i+1}")
        ax[0,i].imshow(test_samples[i].squeeze(0), cmap='gray')
        ax[1,i].imshow(recon[i], cmap='gray')
    plt.show()


def P3Q1_show_snp500_data():
    amazon = generate_snp_company_with_dates('AMZN')
    googe = generate_snp_company_with_dates('GOOGL')
    data_dict = {"AMZN": amazon, "GOOGL": googe}
    #show 2 figures side by side of each comany's stock price over time:
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Stock Price vs. Date')
    for i,name in enumerate(["AMZN", "GOOGL"]):
        data = data_dict[name]
        ax[i].plot(data[:,0], data[:,1], label=name)
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel('Price')
        ax[i].set_title(f'Price vs. Date - {name}')
        ax[i].legend()
        ax[i].tick_params(rotation=45) 
    plt.tight_layout()
    plt.show()


def P3Q2_reconstruct_snp500_data():
    result = subprocess.run(['python3', 'lstm_ae_snp500.py'], text=True, capture_output=True)
    print (result.stderr)
    model = torch.load('lstm_ae_snp500_model.pth').eval()
    print (result.stdout)
    data = load_snp_data('AMZN')
    test_samples = data[:2]
    test_samples_reconstruction = model(test_samples)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Stock Price vs. Date')
    for i in range(2):
        ax[i].plot(test_samples[i], label='Original')
        ax[i].plot(test_samples_reconstruction[i].detach().numpy(), label='Reconstruction')
        ax[i].legend()
    plt.show()







def main():
    P3Q2_reconstruct_snp500_data()

if __name__ == '__main__':
    main()








































