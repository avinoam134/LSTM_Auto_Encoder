import torch
import matplotlib.pyplot as plt
import os.path as path
from logic.Trainers import Basic_Trainer, Classifier_Trainer, kfolds_train
from logic.Utils import parse_args, save_script_out_to_json, load_script_out_from_json
from logic.Data_Generators import load_MNIST_data
from logic.LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER_V1, LSTM_AE_CLASSIFIER_V2, LSTM_AE_CLASSIFIER_V3, LSTM_AE_CLASSIFIER_V4

MNIST_PATH = path.join('outputs', 'mnist')

MNIST_RECON_MODEL = path.join(MNIST_PATH, 'mnist_recon_model.pth')
MNIST_RECON_DATA = path.join(MNIST_PATH, 'mnist_recon_data.json')
MNIST_RECON_TEST = path.join(MNIST_PATH, 'mnist_recon_test.pt')


MNIST_CLASSIF_MODEL = path.join(MNIST_PATH, 'mnist_pred_model.pth')
MNIST_CLASSIF_DATA = path.join(MNIST_PATH, 'mnist_pred_data.json')
MNIST_CLASSIF_TEST = path.join(MNIST_PATH, 'mnist_pred_test.pt')



def find_best_hyperparams(args):
    dataset, testset = load_MNIST_data()
    best_model, res_dict = kfolds_train(args, dataset, tune_hyperparams=True)
    if args.classification:
        torch.save(best_model, MNIST_CLASSIF_MODEL)
        save_script_out_to_json(res_dict, MNIST_CLASSIF_DATA)
        torch.save(testset, MNIST_CLASSIF_TEST)
    else:
        torch.save(best_model, MNIST_RECON_MODEL)
        save_script_out_to_json(res_dict, MNIST_RECON_DATA)
        torch.save(testset, MNIST_RECON_TEST)



def find_best_hyperparams_and_reconstruct(args):
    if not args.dry_run:
        args.model = 'LSTM_AE_CLASSIFIER_V1'
        args.classification = False
        args.reconstruction_dominance = [1]
        find_best_hyperparams(args)
    best_model = torch.load(MNIST_RECON_MODEL)
    res_dict = load_script_out_from_json(MNIST_RECON_DATA)
    testset = torch.load(MNIST_RECON_TEST)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    hidden_size, learning_rate, gradient_clip, best_ratio = res_dict['hidden_size'], res_dict['learning_rate'], res_dict['gradient_clipping'], res_dict['reconstruction_dominance']
    test_samples, _ = next(iter(test_loader))
    test_samples_squeezed = test_samples.squeeze(1)
    test_samples_reconstruction, _ = best_model(test_samples_squeezed)
    recon = test_samples_reconstruction[:3].detach().numpy().reshape(3,28,28)
    fig, ax = plt.subplots(2, 3)
    fig.suptitle(f'Hidden Size: {hidden_size}, Learning Rate: {learning_rate}, Gradient Clip: {gradient_clip}, Reconstruction Dominance: {best_ratio}', color='red')
    for i in range(3):
        ax[0,i].set_title(f"Sample {i+1}")
        ax[1,i].set_title(f"Reconstruction {i+1}")
        ax[0,i].imshow(test_samples[i].squeeze(0), cmap='gray')
        ax[1,i].imshow(recon[i], cmap='gray')
    plt.show()


def find_best_classification_model(args):
    if not args.dry_run:
        args.model = 'LSTM_AE_CLASSIFIER_V4'
        args.classification = True
        find_best_hyperparams(args)
    best_model = torch.load(MNIST_CLASSIF_MODEL)
    res_dict = load_script_out_from_json(MNIST_CLASSIF_DATA)
    testset = torch.load(MNIST_CLASSIF_TEST)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    trainer = Classifier_Trainer()
    test_results = trainer.test(best_model, test_loader, res_dict['reconstruction_dominance'])
    test_loss = test_results['test_loss']
    test_accuracy = test_results['accuracy']
    learning_rate, hidden_size, gradient_clip, best_ratio = res_dict['learning_rate'], res_dict['hidden_size'] ,res_dict['gradient_clipping'], res_dict['reconstruction_dominance']
    plt.title(f'Accuracy: {format(test_accuracy, ".3f")}       Model: LSTM_AE_CLASSIFIER_V4\nLearning Rate: {learning_rate}, Hidden Size: {hidden_size}, Gradient Clip: {gradient_clip}, Reconstruction Dominance: {best_ratio}', color='red')
    plt.plot(res_dict['all_losses'], label='Training Loss')
    plt.plot(res_dict['all_accuracies'], label='Training Accuracy')
    plt.legend()
    plt.show()



def reconstruct_and_classify_over_1_input_size(args):
    if not args.dry_run:
        args.input_size = 1
        args.model = 'LSTM_AE_CLASSIFIER_V4'
        args.classification = True
        find_best_hyperparams(args)
    best_model = torch.load(MNIST_CLASSIF_MODEL)
    res_dict = load_script_out_from_json(MNIST_CLASSIF_DATA)
    testset = torch.load(MNIST_CLASSIF_TEST)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    trainer = Classifier_Trainer()
    test_results = trainer.test(best_model, test_loader, res_dict['reconstruction_dominance'])
    test_accuracy = test_results['accuracy']
    test_loader_small = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False)
    test_samples, _ = next(iter(test_loader_small))
    test_samples_squeezed = test_samples.squeeze(1)
    test_samples_reconstruction, _ = best_model(test_samples_squeezed)
    recon = test_samples_reconstruction[:2].detach().numpy().reshape(2,28,28)
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f'Reconstruction Samples with Input Size of 1:\n        Total Accuracy: ' + format(test_accuracy, ".3f"))
    for i in range(2):
        ax[0,i].set_title(f"Sample {i+1}")
        ax[1,i].set_title(f"Reconstruction {i+1}")
        ax[0,i].imshow(test_samples[i].squeeze(0), cmap='gray')
        ax[1,i].imshow(recon[i], cmap='gray')
    plt.show()





def main():
    args = parse_args()
    if args.function == 'find_best_hyperparams_and_reconstruct':
        #called by:
        '''
        python3 lstm_ae_mnist.py --function find_best_hyperparams_and_reconstruct --model LSTM_AE_CLASSIFIER_V1 --input_size 28 --hidden_size 8 16 --epochs 10 --learning_rate 0.01 --gradient_clipping 5 --batch_size 128 
        '''
        find_best_hyperparams_and_reconstruct(args)
    elif args.function == 'find_best_classification_model':
        #called by:
        '''
        python3 lstm_ae_mnist.py --function find_best_classification_model --model LSTM_AE_CLASSIFIER_V4 --input_size 28 --hidden_size 8 16 --epochs 10 --learning_rate 0.1 0.01 --gradient_clipping 2 5 --batch_size 128 --reconstruction_dominance 0.5
        '''
        find_best_classification_model(args)
    elif args.function == 'reconstruct_and_classify_over_1_input_size':
        #callled by:
        '''
        python3 lstm_ae_mnist.py --function reconstruct_and_classify_over_1_input_size --model LSTM_AE_CLASSIFIER_V --input_size 1 --hidden_size 8 16 --epochs 10 --learning_rate 0.1 0.01 0.001 --gradient_clipping 2 5 --reconstruction_dominance 0.5
        '''
        reconstruct_and_classify_over_1_input_size(args)
    else:
        raise ValueError('Invalid function')

    

if __name__ == '__main__':
    main()





